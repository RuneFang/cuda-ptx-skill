# Chapter 5: SIMT Kernels, Memory Spaces & Shared Memory

## Core Idea
Writing efficient SIMT kernels means understanding the **device memory spaces** (global, shared, local, constant, registers) and using **shared memory + `__syncthreads()`** for intra-block cooperation. Best performance keeps synchronization inside a block.

## Frameworks Introduced
- **GPU device memory spaces** (decreasing scope, increasing speed):
  - **Global memory**: device-wide, large, high-latency DRAM; the main data store.
  - **Shared memory**: on-chip, per-block, low latency (≈L1); for cooperation within a block.
  - **Registers / local memory**: per-thread.
  - **Constant memory**: read-only, cached.
- **Block synchronization**: `__syncthreads()` — barrier; all threads in the block must arrive before any proceed. Lightweight; **intra-block only**.
- **Static shared memory**: `__shared__ float buf[N];` (size known at compile time).
- **Dynamic shared memory**: `extern __shared__ float buf[];` with byte count passed as the 3rd triple-chevron launch parameter `k<<<g, b, sharedBytes>>>()`.
- **Cross-block coordination** (when truly needed): thread block **clusters** (intra-cluster sync) or **Cooperative Groups** (Ch 8).

## Key Concepts
- **global / shared / local / constant memory; registers**: the memory hierarchy.
- **`__syncthreads()`**: intra-block barrier.
- **`__shared__`**: shared-memory variable specifier.
- **memory coalescing**: warp threads accessing contiguous global addresses → combined transactions (fast). Understanding warps explains this.
- **shared memory bank conflicts**: simultaneous accesses to the same bank serialize.
- **atomic functions**: `atomicAdd`, etc. — let blocks contribute to a common result without barriers.

## Code Examples
Dynamic shared-memory kernel skeleton:
```cpp
__global__ void reduce(const float* in, float* out, int n) {
    extern __shared__ float s[];          // size from launch param
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();                       // all loads done
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t+stride];
        __syncthreads();                   // each step visible to all
    }
    if (t == 0) out[blockIdx.x] = s[0];
}
// launch with sharedBytes = blockDim.x * sizeof(float)
reduce<<<blocks, threads, threads*sizeof(float)>>>(in, out, n);
```
- **What it demonstrates**: dynamic shared memory, `__syncthreads()` between read/modify phases, partial-block reduction.

## Reference Tables
| Space | Scope | Latency | Declared |
|---|---|---|---|
| Registers | thread | fastest | (compiler) |
| Shared | block | low (~L1) | `__shared__` / `extern __shared__` |
| Global | device | high | `cudaMalloc` / pointers |
| Constant | device (RO) | cached | `__constant__` |
| Local | thread | high (spills) | (compiler) |

## Worked Example
**Coalescing in the reduction.** The initial load `s[t] = in[blockIdx.x*blockDim.x + t]` makes consecutive threads (`t = 0,1,2,...`) read consecutive global addresses — a fully coalesced transaction. If instead threads read strided addresses (e.g. `in[t*blockDim.x]`), each warp would issue many separate memory transactions, multiplying latency. The lesson: map `threadIdx.x` to the *innermost*, contiguous data dimension.

## Anti-patterns
- **`__syncthreads()` inside divergent control flow**: if not all threads reach the same barrier, behavior is undefined.
- **Relying on cross-block sync without clusters/Cooperative Groups**: blocks aren't guaranteed co-resident.
- **Strided global access**: breaks coalescing; prefer contiguous per-warp access.

## Key Takeaways
1. Know the memory hierarchy: registers → shared → global, fast → slow.
2. `__syncthreads()` coordinates a block's shared-memory phases; it is intra-block only.
3. Static (`__shared__`) vs dynamic (`extern __shared__` + launch byte param) shared memory.
4. Coalesce global accesses by mapping `threadIdx.x` to contiguous addresses.
5. Use atomics for cross-block contributions; keep heavy sync within a block.

## Connects To
- **Ch 1**: warps & SIMT explain coalescing and bank conflicts.
- **Ch 8 (Cooperative Groups)**: structured multi-level synchronization.
- **Ch 11 (Performance/Compute Capability)**: occupancy & memory-bandwidth tuning.
