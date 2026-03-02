# EC Transfer Fault Tolerance Implementation

This document describes the fault tolerance implementation for EC (Encoder Cache) transfer, modeled after the existing KV transfer fault tolerance mechanism.

## Overview

The EC transfer fault tolerance mechanism provides robust error handling for encoder cache transfers between disaggregated encoder and prefill/decode instances. When EC transfers fail, the system can either:
1. **Recompute (default)**: Fallback to local encoder computation
2. **Fail**: Immediately fail the request with an error

## Key Components

### 1. Configuration (`vllm/config/ec_transfer.py`)

Added `ec_load_failure_policy` configuration option:

```python
ec_load_failure_policy: Literal["recompute", "fail"] = "recompute"
```

- **recompute**: Fallback to local encoder computation when EC transfer fails (default)
- **fail**: Immediately fail the request with an error finish reason

### 2. Output Tracking (`vllm/v1/outputs.py`)

Enhanced `ECConnectorOutput` to track failed transfers:

```python
@dataclass
class ECConnectorOutput:
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    invalid_mm_hashes: set[str] = field(default_factory=set)  # NEW
```

The `invalid_mm_hashes` field contains mm_hashes (multimodal identifiers) that failed to transfer.

### 3. Worker-Side Failure Detection (`vllm/distributed/ec_transfer/ec_connector/mooncake_connector.py`)

#### Added Failure Tracking Structure

```python
@dataclass
class FailedReceiveMMHashSet:
    """Track mm_hashes that failed to receive."""
    set: set[MMHash]
    lock: asyncio.Lock
```

#### Enhanced `receive_ec()` Method

The receiver now catches and tracks multiple failure scenarios:
- Transfer timeout (60s)
- Transfer errors (TRANS_ERROR response)
- ZMQ context termination
- Tensor loading failures

Failed mm_hashes are added to `failed_recving_mm_hashes` set.

#### Updated `get_finished()` Method

Now returns three values instead of two:

```python
def get_finished(self, finished_req_ids: set[str]) -> tuple[
    set[str] | None,  # finished_sending
    set[str] | None,  # finished_recving
    set[str] | None,  # failed_recving (NEW)
]:
```

### 4. Scheduler-Side Failure Handling (`vllm/v1/core/sched/scheduler.py`)

#### Added Configuration and State Tracking

```python
# In __init__:
self.recompute_ec_load_failures = True
if self.vllm_config.ec_transfer_config is not None:
    ec_load_failure_policy = (
        self.vllm_config.ec_transfer_config.ec_load_failure_policy
    )
    self.recompute_ec_load_failures = ec_load_failure_policy == "recompute"

# Track failed EC transfers for retry
self.failed_recving_ec_mm_hashes: set[str] = set()
```

#### Added `_handle_invalid_ec_items()` Method

Similar to `_handle_invalid_blocks()` for KV transfers, this method:

1. **Identifies affected requests**: Finds all requests that reference failed mm_hashes
2. **Applies failure policy**:
   - **fail**: Marks requests as `FINISHED_ERROR`
   - **recompute**: Clears `do_remote_encode` flag to trigger local encoding
3. **Tracks failures**: Adds failed mm_hashes to `failed_recving_ec_mm_hashes` for monitoring
4. **Returns affected request IDs**: To skip in the main update loop

#### Integration in `update_from_output()`

```python
# Handle EC transfer failures
ec_connector_output = model_runner_output.ec_connector_output
failed_ec_load_req_ids = None
if ec_connector_output and ec_connector_output.invalid_mm_hashes:
    failed_ec_load_req_ids = self._handle_invalid_ec_items(
        ec_connector_output.invalid_mm_hashes
    )

# Skip failed EC requests in main loop
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
    if failed_ec_load_req_ids and req_id in failed_ec_load_req_ids:
        continue
    # ... process request
```

### 5. Base Connector Interface (`vllm/distributed/ec_transfer/ec_connector/base.py`)

Updated `get_finished()` signature to include failed transfers:

```python
def get_finished(
    self, finished_req_ids: set[str]
) -> tuple[set[str] | None, set[str] | None, set[str] | None]:
    """
    Returns:
        Tuple of (finished_sending, finished_recving, failed_recving).
    """
    return None, None, None
```

### 6. Model Runner Integration (`vllm/v1/worker/ec_connector_model_runner_mixin.py`)

Updated to handle the new return value:

```python
(
    output.finished_sending,
    output.finished_recving,
    output.invalid_mm_hashes,  # NEW
) = ec_connector.get_finished(scheduler_output.finished_req_ids)
```

## Failure Scenarios Handled

### 1. Transfer Timeout
- **Detection**: ZMQ socket timeout (60 seconds)
- **Action**: Mark mm_hash as failed, log error

### 2. Transfer Error Response
- **Detection**: Receiver gets `TRANS_ERROR` instead of `TRANS_DONE`
- **Action**: Mark mm_hash as failed, log error

### 3. Tensor Loading Failure
- **Detection**: Exception during `transfer_buffer.load_tensor()`
- **Action**: Mark mm_hash as failed, log exception

### 4. ZMQ Context Termination
- **Detection**: `zmq.ContextTerminated` exception
- **Action**: Mark mm_hash as failed, exit gracefully

## Comparison with KV Transfer Fault Tolerance

| Feature | KV Transfer | EC Transfer |
|---------|-------------|-------------|
| **Configuration** | `kv_load_failure_policy` | `ec_load_failure_policy` |
| **Failure Tracking** | `invalid_block_ids` | `invalid_mm_hashes` |
| **Recompute Strategy** | Recompute KV blocks | Fallback to local encoding |
| **Fail Strategy** | Mark as `FINISHED_ERROR` | Mark as `FINISHED_ERROR` |
| **State Tracking** | `failed_recving_kv_req_ids` | `failed_recving_ec_mm_hashes` |
| **Handler Method** | `_handle_invalid_blocks()` | `_handle_invalid_ec_items()` |

## Usage

### Configuration Example

```python
# Encoder instance (producer)
--ec-transfer-config '{
    "ec_connector": "MooncakeECConnector",
    "ec_role": "ec_producer",
    "ec_load_failure_policy": "recompute"
}'

# Prefill/Decode instance (consumer)
--ec-transfer-config '{
    "ec_connector": "MooncakeECConnector",
    "ec_role": "ec_consumer",
    "ec_load_failure_policy": "recompute"
}'
```

### Policy Options

1. **recompute (default)**: Best for production, provides graceful degradation
   ```python
   "ec_load_failure_policy": "recompute"
   ```

2. **fail**: Best for debugging, fails fast on errors
   ```python
   "ec_load_failure_policy": "fail"
   ```

## Logging

The implementation provides detailed logging at different levels:

- **ERROR**: Transfer failures, timeouts, policy=fail
- **WARNING**: Recovered failures with recompute policy
- **DEBUG**: Individual mm_hash operations, flag clearing

Example logs:

```
[EC_WORKER_RECEIVER] Transfer TIMEOUT after 60s for mm_hashes=['abc123...']
[EC_WORKER_RECEIVER] Marked 1 mm_hashes as failed: ['abc123...']
WARNING: Recovered from EC load failure: 1 request(s) will fallback to local encoding (1 mm_hashes affected).
DEBUG: Cleared do_remote_encode for req_id=xyz789, mm_hash=abc123
```

## Testing

To test the fault tolerance mechanism:

1. **Simulate transfer failure**: Kill encoder instance during transfer
2. **Verify recompute policy**: Check that requests complete with local encoding
3. **Verify fail policy**: Check that requests fail with `FINISHED_ERROR`
4. **Check logs**: Verify appropriate error messages and recovery actions

## Future Enhancements

Potential improvements to consider:

1. **Retry with backoff**: Retry failed transfers before falling back to local encoding
2. **Configurable max retries**: Allow users to set maximum retry attempts
3. **Metrics collection**: Track failure rates, recovery times, etc.
4. **Partial transfer recovery**: Handle partial mm_hash failures in batch transfers
5. **Health checks**: Proactive health checking of encoder instances

## Files Modified

1. `vllm/config/ec_transfer.py` - Added `ec_load_failure_policy` config
2. `vllm/v1/outputs.py` - Added `invalid_mm_hashes` to `ECConnectorOutput`
3. `vllm/distributed/ec_transfer/ec_connector/base.py` - Updated `get_finished()` signature
4. `vllm/distributed/ec_transfer/ec_connector/mooncake_connector.py` - Added failure detection and tracking
5. `vllm/v1/core/sched/scheduler.py` - Added `_handle_invalid_ec_items()` and integration
6. `vllm/v1/worker/ec_connector_model_runner_mixin.py` - Updated to handle failed transfers

## Summary

This implementation brings EC transfer fault tolerance to parity with KV transfer, providing:
- ✅ Configurable failure policies (recompute/fail)
- ✅ Comprehensive failure detection (timeout, errors, exceptions)
- ✅ Graceful degradation (fallback to local encoding)
- ✅ Detailed logging and monitoring
- ✅ Consistent API with KV transfer

The system is now robust against encoder instance failures, network issues, and other transfer errors.
