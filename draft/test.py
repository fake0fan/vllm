#!/usr/bin/env python3
"""
Zero-copy bandwidth micro-benchmark for MooncakeDistributedStore.

Sizes tested: 2 MB, 5 MB, 10 MB (float32 tensors).

Throughput is reported in MiB/s (1 MiB = 2**20 bytes).
"""
import time
import numpy as np
from mooncake.store import MooncakeDistributedStore

# ---------------------------- Store setup -----------------------------------
store = MooncakeDistributedStore()
store.setup(
    "localhost",                        # node name
    "http://localhost:8081/metadata",   # metadata service
    512 * 1024 * 1024,                  # RDMA pool (512 MiB)
    16 * 1024 * 1024,                   # send/recv buffers (16 MiB)
    "tcp",                              # transport
    "",                                 # rdma device ("" = default)
    "localhost:50052",                  # gRPC endpoint
)

# ---------------------------- Helper utils ----------------------------------
def mib(bytes_: int) -> float:
    return bytes_ / (1 << 20)          # convert to mebibytes

def register_or_die(ptr, sz):
    rc = store.register_buffer(ptr, sz)
    if rc:
        raise RuntimeError(f"register_buffer failed: {rc}")

def unregister_quiet(ptr):
    try:
        store.unregister_buffer(ptr)
    except Exception:
        pass

def run_once(tag: str, arr: np.ndarray) -> tuple[float, float]:
    """Store then load `arr` once, return (store_s, load_s)."""
    size = arr.nbytes
    ptr  = arr.ctypes.data

    # Register source buffer
    # register_or_die(ptr, size)

    # Pre-allocate & register destination buffer
    dst = np.empty_like(arr)
    dst_ptr = dst.ctypes.data
    # register_or_die(dst_ptr, size)

    # --- Measure put ---
    t0 = time.perf_counter()
    rc = store.put(tag, arr)
    t_put = time.perf_counter() - t0
    if rc:
        raise RuntimeError(f"put_from failed: {rc}")

    # --- Measure get ---
    t0 = time.perf_counter()
    nbytes = store.get(tag)
    t_get = time.perf_counter() - t0
    if nbytes != size:
        raise RuntimeError(f"get_into returned {nbytes} bytes (expected {size})")

    # Verify correctness (optional – disable for max speed)
    if not np.array_equal(arr, dst):
        raise ValueError("Data mismatch!")

    # Clean up
    # unregister_quiet(ptr)
    # unregister_quiet(dst_ptr)
    return t_put, t_get

# ---------------------------- Benchmark loop ---------------------------------
sizes_mib = [200, 500, 1000]
repetitions = 1                        # average over a few trials

print(f"{'Size':>6} {'PUT MiB/s':>12} {'GET MiB/s':>12}")
print("-" * 34)

for sz_mib in sizes_mib:
    sz_bytes = sz_mib * (1 << 20)
    n_elts   = sz_bytes // 4           # float32 = 4 bytes
    tag      = f"tensor_{sz_mib}MiB"

    # Pre-generate the tensor once (re-used across reps)
    src = np.random.randn(n_elts).astype(np.float32)

    put_times, get_times = [], []
    for rep in range(repetitions):
        t_put, t_get = run_once(f"{tag}_{rep}", src)
        put_times.append(t_put)
        get_times.append(t_get)

    avg_put = sum(put_times) / repetitions
    avg_get = sum(get_times) / repetitions

    put_bw = mib(sz_bytes) / avg_put
    get_bw = mib(sz_bytes) / avg_get

    print(f"{sz_mib:4} MB {put_bw:11.1f} {get_bw:11.1f}")

# ---------------------------- Tear-down --------------------------------------
store.close()