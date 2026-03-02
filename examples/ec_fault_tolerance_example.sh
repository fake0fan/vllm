#!/bin/bash
# Example: EC Transfer with Fault Tolerance
# This script demonstrates how to configure EC transfer with fault tolerance

# Configuration options:
# 1. ec_load_failure_policy: "recompute" (default) or "fail"
#    - recompute: Fallback to local encoder computation on failure
#    - fail: Immediately fail the request on transfer error

# Shared storage path for EC cache
EC_SHARED_STORAGE_PATH="/tmp/ec_cache"
mkdir -p $EC_SHARED_STORAGE_PATH

# Model to use
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

echo "=========================================="
echo "EC Transfer Fault Tolerance Example"
echo "=========================================="
echo ""
echo "This example shows two configurations:"
echo "1. Encoder instance (producer)"
echo "2. Prefill/Decode instance (consumer) with fault tolerance"
echo ""

# ==========================================
# 1. Encoder Instance (Producer)
# ==========================================
echo "Starting Encoder instance..."
echo ""
echo "Configuration:"
echo "  - Role: ec_producer"
echo "  - Policy: recompute (default)"
echo ""

# Example command (not executed):
cat << 'EOF'
vllm serve $MODEL \
  --port 8001 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 999999 \
  --mm-encoder-only \
  --ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_producer",
    "ec_load_failure_policy": "recompute",
    "ec_connector_extra_config": {
      "shared_storage_path": "/tmp/ec_cache"
    }
  }'
EOF

echo ""
echo "=========================================="
echo ""

# ==========================================
# 2. Prefill/Decode Instance (Consumer)
# ==========================================
echo "Starting Prefill/Decode instance..."
echo ""
echo "Configuration:"
echo "  - Role: ec_consumer"
echo "  - Policy: recompute (fallback to local encoding on failure)"
echo ""

# Example command (not executed):
cat << 'EOF'
vllm serve $MODEL \
  --port 8000 \
  --ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_consumer",
    "ec_load_failure_policy": "recompute",
    "ec_connector_extra_config": {
      "shared_storage_path": "/tmp/ec_cache"
    }
  }'
EOF

echo ""
echo "=========================================="
echo ""

# ==========================================
# Alternative: Fail-Fast Configuration
# ==========================================
echo "Alternative: Fail-Fast Configuration"
echo ""
echo "For debugging or strict error handling, use 'fail' policy:"
echo ""

cat << 'EOF'
vllm serve $MODEL \
  --port 8000 \
  --ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_consumer",
    "ec_load_failure_policy": "fail",
    "ec_connector_extra_config": {
      "shared_storage_path": "/tmp/ec_cache"
    }
  }'
EOF

echo ""
echo "=========================================="
echo ""

# ==========================================
# Testing Fault Tolerance
# ==========================================
echo "Testing Fault Tolerance:"
echo ""
echo "1. Start both encoder and PD instances"
echo "2. Send a request with multimodal input"
echo "3. Kill the encoder instance during transfer"
echo "4. Observe the behavior:"
echo ""
echo "   With 'recompute' policy:"
echo "   - Request continues with local encoding"
echo "   - Logs show: 'Recovered from EC load failure'"
echo "   - Request completes successfully"
echo ""
echo "   With 'fail' policy:"
echo "   - Request fails immediately"
echo "   - Logs show: 'Failing request(s) due to EC load failure'"
echo "   - Client receives error response"
echo ""

# ==========================================
# Expected Logs
# ==========================================
echo "=========================================="
echo "Expected Logs:"
echo "=========================================="
echo ""

echo "On Transfer Failure (recompute policy):"
cat << 'EOF'
[EC_WORKER_RECEIVER] Transfer TIMEOUT after 60s for mm_hashes=['abc123...']
[EC_WORKER_RECEIVER] Marked 1 mm_hashes as failed: ['abc123...']
WARNING: Recovered from EC load failure: 1 request(s) will fallback to local encoding (1 mm_hashes affected).
DEBUG: Cleared do_remote_encode for req_id=xyz789, mm_hash=abc123
EOF

echo ""
echo ""

echo "On Transfer Failure (fail policy):"
cat << 'EOF'
[EC_WORKER_RECEIVER] Transfer TIMEOUT after 60s for mm_hashes=['abc123...']
[EC_WORKER_RECEIVER] Marked 1 mm_hashes as failed: ['abc123...']
ERROR: Failing 1 request(s) due to EC load failure (failure_policy=fail, 1 mm_hashes affected). Request IDs: {'xyz789'}
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
