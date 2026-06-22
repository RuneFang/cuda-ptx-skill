# Chapter 2: Writing Kernels in CUDA C++

## Core Idea
A CUDA C++ kernel is a `__global__` function launched with **triple-chevron** `<<<blocks, threads>>>` syntax. Each thread computes its own work index from built-in variables, optionally bounds-checks, and operates on memory the GPU can reach.

## Frameworks Introduced
- **`__global__` kernel + triple chevron launch**:
  - When to use: any data-parallel computation.
  - How: declare `__global__ void k(...)`, launch with `k<<<gridDim, blockDim, sharedBytes, stream>>>(args)`. Last two launch params are optional.
- **Global thread-index idiom**: `int i = threadIdx.x + blockDim.x * blockIdx.x;` — the canonical 1-D mapping; replicate per dimension for 2-D/3-D.
- **Grid-size ("ceil div") idiom**: `blocks = (n + threads - 1) / threads;` or use `cuda::ceil_div(n, threads)` from `<cuda/cmath>` (CCCL).
- **Bounds checking**: guard with `if (i < n) { ... }` so you may over-launch threads safely. Launching extra *threads* is cheap; launching blocks where *no* thread works is wasteful — avoid that.

## Key Concepts
- **`threadIdx` / `blockIdx` / `blockDim` / `gridDim`**: built-in 3-component (.x/.y/.z) index & dimension variables.
- **`__global__`**: callable from host, runs on device (a kernel).
- **`__device__` / `__host__`**: device-only / host-only function qualifiers (combine `__host__ __device__`).
- **`__managed__`**: variable specifier placing data in unified memory.
- **NVCC**: the CUDA compiler; `nvcc file.cu -o exe`.
- **CCCL**: CUDA Core Compute Library (`cuda::ceil_div`, `<cuda/cmath>`, etc.).

## Code Examples
The canonical vector-add kernel with bounds checking:
```cpp
__global__ void vecAdd(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
// launch
int threads = 256;
int blocks  = cuda::ceil_div(n, threads);  // <cuda/cmath>
vecAdd<<<blocks, threads>>>(A, B, C, n);
```
- **What it demonstrates**: index computation, bounds guard, ceil-div launch sizing. 256 threads/block is "quite often a good value to start with."

## Reference Tables
| Function qualifier | Callable from | Runs on |
|---|---|---|
| `__global__` | host (and device, via dynamic parallelism) | device (kernel) |
| `__device__` | device | device |
| `__host__` | host | host |
| `__host__ __device__` | host & device | host & device |

## Worked Example
**Why over-launching is fine but partial blocks aren't.** For `n = 1000`, `threads = 256` → `blocks = ceil_div(1000,256) = 4` → 1024 threads launched. Threads 1000–1023 hit `if (i < n)` → false → exit immediately. Cost: 24 idle lanes in the last block, negligible. The anti-pattern would be computing `blocks = 5` (launching a 5th block where *every* thread is idle) — that block does real scheduling work for zero output.

## Key Takeaways
1. Kernel = `__global__`; launch = `<<<blocks, threads>>>`.
2. Memorize `i = threadIdx.x + blockDim.x * blockIdx.x`.
3. Always bounds-check so block size need not divide the problem size.
4. Use `cuda::ceil_div` for launch sizing; start at 256 threads/block.
5. Kernel launches are **asynchronous** w.r.t. the host (see Ch 3 / Ch 6).

## Connects To
- **Ch 3**: getting the data the kernel needs into GPU-accessible memory.
- **Ch 6 (Streams)**: kernel launches are async; synchronize before using results.
- **Ch 4 (PTX)**: drop to `asm()` when you need an instruction C++ doesn't expose.
