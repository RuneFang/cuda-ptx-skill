# Chapter 3: Memory Management — Unified vs Explicit

## Core Idea
Before a kernel runs, its data must live in GPU-accessible memory. Two strategies: **Unified Memory** (driver migrates data automatically — simple) and **Explicit Memory Management** (`cudaMalloc` + `cudaMemcpy` — verbose but gives control over *when* and *where* data moves, enabling overlap).

## Frameworks Introduced
- **Unified Memory**: allocate with `cudaMallocManaged(&p, bytes)` or declare `__managed__`. Driver makes memory accessible to whichever processor touches it. Free with `cudaFree`. On some Linux systems (ATS / HMM) *all* system memory is already unified — no special allocation needed.
  - When to use: prototyping, irregular access, simplicity.
- **Explicit Memory Management**: `cudaMalloc` device buffers, `cudaMemcpy` to/from host, `cudaFree`. Use `cudaMallocHost` for **page-locked (pinned)** host buffers used in transfers.
  - When to use: when you need to control/overlap transfers for performance.
- **Page-locked (pinned) host memory**: `cudaMallocHost` / `cudaFreeHost`. Faster copies and *required* for async transfers. Best practice: pin only buffers used for GPU transfers (over-pinning degrades the whole system).

## Key Concepts
- **`cudaMallocManaged` / `__managed__`**: unified memory allocation.
- **`cudaMalloc` / `cudaFree`**: device allocate / free (also frees managed memory).
- **`cudaMemcpy(dst, src, bytes, kind)`**: **synchronous** copy; blocks until done.
- **`cudaMemcpyKind`**: `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, `cudaMemcpyDefault` (infer from pointers).
- **`cudaMallocHost` / `cudaFreeHost`**: pinned host buffer.
- **`cudaMemset`**: set device memory to a byte value.

## Code Examples
Unified memory — no explicit copies:
```cpp
float *A, *B, *C;
cudaMallocManaged(&A, n*sizeof(float));
cudaMallocManaged(&B, n*sizeof(float));
cudaMallocManaged(&C, n*sizeof(float));
initArray(A, n); initArray(B, n);          // host writes directly
vecAdd<<<blocks, threads>>>(A, B, C, n);   // driver migrates as needed
cudaDeviceSynchronize();                   // results in C, host-readable
cudaFree(A); cudaFree(B); cudaFree(C);
```
Explicit — control every transfer:
```cpp
float *devA, *devB, *devC;
cudaMalloc(&devA, n*sizeof(float));        // device buffers
cudaMalloc(&devB, n*sizeof(float));
cudaMalloc(&devC, n*sizeof(float));
cudaMemcpy(devA, A, n*sizeof(float), cudaMemcpyDefault);  // host->device
cudaMemcpy(devB, B, n*sizeof(float), cudaMemcpyDefault);
vecAdd<<<blocks, threads>>>(devA, devB, devC, n);
cudaDeviceSynchronize();
cudaMemcpy(C, devC, n*sizeof(float), cudaMemcpyDefault);  // device->host
cudaFree(devA); cudaFree(devB); cudaFree(devC);
```
- **What it demonstrates**: the verbosity/control tradeoff. `cudaMemcpyDefault` lets CUDA infer direction from the pointers.

## Reference Tables
| Concern | Unified Memory | Explicit |
|---|---|---|
| API | `cudaMallocManaged` / `__managed__` | `cudaMalloc` + `cudaMemcpy` |
| Code volume | Low | High |
| Control of transfer timing | Driver-managed (hintable) | Full |
| Overlap copies w/ compute | Via prefetch/advise hints | Via async copies + streams |
| Host buffer for transfers | n/a | `cudaMallocHost` (pinned) |

## Worked Example
**Tuning unified memory toward explicit performance without rewriting.** Start with `cudaMallocManaged` for simplicity. If profiling shows migration stalls, keep the unified allocation but add **Memory Advise / Prefetch** hints (`cudaMemAdvise`, `cudaMemPrefetchAsync`) to tell the driver residency intent — recovering much of the benefit of explicit management while keeping the simpler allocation/access code.

## Key Takeaways
1. Unified = simple (`cudaMallocManaged`); Explicit = control (`cudaMalloc`/`cudaMemcpy`).
2. `cudaMemcpy` is **synchronous** — it blocks the host.
3. Use `cudaMallocHost` (pinned) host buffers for transfers; required for async copies.
4. `cudaMemcpyDefault` infers copy direction from pointers.
5. Over-pinning host memory hurts the whole system — pin only transfer buffers.

## Connects To
- **Ch 2**: kernels need data here first.
- **Ch 6 (Async/Streams)**: async copies + pinned memory enable transfer/compute overlap.
- **Ch 7 (Unified Memory advanced)**: prefetch & advise, oversubscription, system memory.
