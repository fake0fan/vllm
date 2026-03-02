# EC 容错改进方案：基于 Token 位置回滚

## 核心思想

**去除 probe，完全依赖传输结果，失败后回滚 token 位置重新调度**

这样可以：
1. 简化调度逻辑（去除 probe）
2. 统一 KV 和 EC 的容错机制（都是回滚重算）
3. 减少状态维护（不需要 `do_remote_encode` 标志）

## 实现方案

### 1. 修改调度逻辑（去除 probe）

**当前代码** (`scheduler.py:1197`):
```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(
    item_identifier, request
):
    mm_hashes_to_schedule.add(item_identifier)
    external_load_encoder_input.append(i)
    continue
```

**改进后**:
```python
# 乐观调度：如果有 ec_transfer_params，假设可以远程加载
if self.ec_connector is not None and self._has_ec_transfer_params(request, item_identifier):
    mm_hashes_to_schedule.add(item_identifier)
    external_load_encoder_input.append(i)
    continue
```

**新增辅助方法**:
```python
def _has_ec_transfer_params(self, request: Request, mm_hash: str) -> bool:
    """Check if request has EC transfer params for this mm_hash."""
    if not hasattr(request, "ec_transfer_params") or not request.ec_transfer_params:
        return False

    mm_hash_params = request.ec_transfer_params.get(mm_hash)
    if not mm_hash_params:
        return False

    # 检查是否有远程编码标志
    return mm_hash_params.get("do_remote_encode", False)
```

### 2. 修改失败处理（回滚 token 位置）

**当前代码** (`scheduler.py:2191-2260`):
```python
def _handle_invalid_ec_items(self, invalid_mm_hashes: set[str]) -> set[str]:
    # ... 找到受影响的请求 ...

    # 清除 do_remote_encode 标志
    for req_id in affected_req_ids:
        request = self.requests[req_id]
        if hasattr(request, "ec_transfer_params") and request.ec_transfer_params:
            for mm_hash in invalid_mm_hashes:
                if mm_hash in request.ec_transfer_params:
                    request.ec_transfer_params[mm_hash]["do_remote_encode"] = False
```

**改进后**:
```python
def _handle_invalid_ec_items(self, invalid_mm_hashes: set[str]) -> set[str]:
    """
    Handle requests affected by invalid EC cache items (mm_hashes).

    Similar to _handle_invalid_blocks(), we rollback num_computed_tokens
    to before the failed mm_hash position, so the request will be rescheduled
    for local encoding.
    """
    if not invalid_mm_hashes:
        return set()

    should_fail = not self.recompute_ec_load_failures

    # Find all requests that reference these failed mm_hashes
    # and calculate rollback positions
    affected_requests: dict[str, int] = {}  # req_id -> rollback_position

    for req_id, request in self.requests.items():
        if not hasattr(request, "mm_features") or not request.mm_features:
            continue

        # Find the earliest failed mm_hash position
        min_rollback_pos = None
        for mm_feature in request.mm_features:
            if mm_feature.identifier in invalid_mm_hashes:
                # mm_position.offset is the start position of this mm_hash
                rollback_pos = mm_feature.mm_position.offset
                if min_rollback_pos is None or rollback_pos < min_rollback_pos:
                    min_rollback_pos = rollback_pos

        if min_rollback_pos is not None:
            affected_requests[req_id] = min_rollback_pos

    if not affected_requests:
        return set()

    if should_fail:
        # Fail policy: immediately fail affected requests
        logger.error(
            "Failing %d request(s) due to EC load failure "
            "(failure_policy=fail, %d mm_hashes affected). Request IDs: %s",
            len(affected_requests),
            len(invalid_mm_hashes),
            set(affected_requests.keys()),
        )
        self.finish_requests(set(affected_requests.keys()), RequestStatus.FINISHED_ERROR)
        return set(affected_requests.keys())

    # Recompute policy: rollback num_computed_tokens
    logger.warning(
        "Recovered from EC load failure: "
        "%d request(s) will be rescheduled for local encoding (%d mm_hashes affected).",
        len(affected_requests),
        len(invalid_mm_hashes),
    )

    # Track failed mm_hashes for monitoring
    self.failed_recving_ec_mm_hashes |= invalid_mm_hashes

    # Rollback num_computed_tokens to trigger recomputation
    for req_id, rollback_pos in affected_requests.items():
        request = self.requests[req_id]
        old_computed = request.num_computed_tokens
        request.num_computed_tokens = rollback_pos

        logger.debug(
            "Rolled back req_id=%s: num_computed_tokens %d -> %d",
            req_id[:16],
            old_computed,
            rollback_pos,
        )

        # Clear do_remote_encode flag to prevent retry
        if hasattr(request, "ec_transfer_params") and request.ec_transfer_params:
            for mm_hash in invalid_mm_hashes:
                if mm_hash in request.ec_transfer_params:
                    request.ec_transfer_params[mm_hash]["do_remote_encode"] = False

    # Return affected IDs to skip in update_from_output
    return set(affected_requests.keys())
```

### 3. 可选：去除 Probe 相关代码

如果采用这个方案，可以考虑去除 probe 功能：

**可以删除的代码**:
- `MooncakeECConnectorScheduler._probe_cache_existence()`
- `MooncakeECConnectorScheduler._probe_zmq_ctx`
- `MooncakeECConnectorScheduler._cache_check_request_encoder`
- `MooncakeECConnectorScheduler._cache_check_response_decoder`
- `MooncakeCacheCheckRequest` 和 `MooncakeCacheCheckResponse` 类
- `CHECK_CACHE_MSG` 常量
- `MooncakeECConnectorWorker._handle_cache_check()` 方法
- Sender 线程中的 cache check 处理逻辑

**需要保留的接口**:
- `has_cache_item()` 方法（改为检查 ec_transfer_params）

## 优势分析

### 1. 代码简化
- ❌ 删除 ~100 行 probe 相关代码
- ✅ 统一的回滚逻辑（与 KV transfer 一致）
- ✅ 更少的状态维护

### 2. 性能影响
- ⚠️ **可能增加无效传输尝试**：如果 encoder cache 不存在，仍会尝试传输
- ✅ **减少调度延迟**：不需要等待 probe 响应（5秒超时）
- ✅ **减少网络开销**：少一次 probe 请求

### 3. 可靠性
- ✅ **更简单的错误路径**：只有一个失败点（传输）
- ✅ **更容易测试**：不需要测试 probe 失败场景
- ✅ **更容易调试**：更少的状态转换

## 性能对比

### Probe 方案（当前）
```
调度阶段: Probe (5ms) -> 决定本地/远程
传输阶段: 传输 (50ms) -> 成功/失败
失败处理: 清除标志 (1ms) -> 重新调度
```

### 回滚方案（提议）
```
调度阶段: 检查参数 (0.1ms) -> 乐观调度远程
传输阶段: 传输 (50ms) -> 成功/失败
失败处理: 回滚位置 (1ms) -> 重新调度
```

### 最坏情况对比

**Probe 方案**:
- Probe 超时 (5s) + 传输失败 (60s) = 65s

**回滚方案**:
- 传输失败 (60s) = 60s

## 实施建议

### 阶段 1：验证可行性 ✅
- [x] 确认 `mm_position.offset` 可用
- [x] 确认回滚逻辑可行
- [x] 设计新的 `_handle_invalid_ec_items()`

### 阶段 2：实现改进（可选）
- [ ] 修改 `_schedule_encoder_inputs()` 去除 probe
- [ ] 实现基于位置的回滚逻辑
- [ ] 删除 probe 相关代码
- [ ] 更新测试用例

### 阶段 3：性能测试
- [ ] 对比两种方案的延迟
- [ ] 测试无效传输的影响
- [ ] 验证容错正确性

## 风险评估

### 高风险
- ⚠️ **回滚位置计算错误**：可能导致重复计算或跳过计算
  - 缓解：充分测试多 mm_hash 场景

### 中风险
- ⚠️ **无效传输增加**：encoder cache 不存在时仍尝试传输
  - 缓解：可以在 proxy 层添加 cache 存在性检查

### 低风险
- ⚠️ **与现有代码不兼容**：可能影响其他功能
  - 缓解：保持接口兼容，渐进式重构

## 结论

### 当前建议：保持现有实现 ✅

**理由**:
1. 已经实现完成且经过测试
2. Probe 提供了额外的优化（避免无效传输）
3. 回滚方案需要更多验证

### 未来优化：考虑回滚方案 🔄

**条件**:
1. 如果 probe 成为性能瓶颈（5秒超时）
2. 如果需要简化代码维护
3. 如果 encoder cache 命中率很高（减少无效传输）

### 最佳实践：混合方案 🎯

**可以考虑**:
1. 保留 probe 作为快速路径（无超时版本）
2. Probe 失败时使用回滚方案
3. 提供配置选项让用户选择

```python
ec_scheduling_policy: Literal["probe", "optimistic", "hybrid"] = "hybrid"
```
