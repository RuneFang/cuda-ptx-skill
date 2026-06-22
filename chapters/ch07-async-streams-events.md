# Chapter 7: Asynchronous Execution — Streams & Events

## Core Idea
CUDA expresses concurrency through **streams** (ordered work queues) and **events** (timestamps/sync points). Overlapping host compute, device compute, and memory transfers — instead of running them serially — is the main lever for hiding latency.

## Frameworks Introduced
- **CUDA Stream**: an abstraction that orders a sequence of operations (copies, launches). Within a stream, ops run **in issue order**; across streams they may overlap. Kernel launches & async APIs are asynchronous w.r.t. the host.
  - Create/destroy: `cudaStreamCreate(&s)` / `cudaStreamDestroy(s)`.
  - **Default stream**: ops without an explicit stream go here (special blocking semantics).
  - Streams can carry a **priority** hint (not a guarantee).
- **CUDA Event**: marker recorded into a stream; used for timing and cross-stream synchronization.
- **Three synchronization styles**: **blocking** (wait until done, e.g. `cudaStreamSynchronize`, `cudaDeviceSynchronize`), **polling/non-blocking** (query status, e.g. `cudaStreamQuery`), **callback** (host function runs on completion).
- **Async copies**: `cudaMemcpyAsync` (requires pinned host memory) overlaps transfer with compute when issued on a separate stream.

## Key Concepts
- **`cudaStream_t`**: stream handle.
- **`cudaStreamCreate` / `cudaStreamDestroy` / `cudaStreamSynchronize` / `cudaStreamQuery`**: lifecycle & sync.
- **`cudaEvent_t`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`**: events & timing.
- **`cudaMemcpyAsync`**: non-blocking copy (needs pinned host buffer).
- **default stream**: implicit stream with blocking semantics relative to others (configurable per-thread).

## Code Examples
Overlap copy and compute across two streams:
```cpp
cudaStream_t s0, s1;
cudaStreamCreate(&s0); cudaStreamCreate(&s1);
// pinned host buffers h0,h1 via cudaMallocHost
cudaMemcpyAsync(d0, h0, bytes, cudaMemcpyHostToDevice, s0);
kernel<<<g, b, 0, s0>>>(d0);
cudaMemcpyAsync(d1, h1, bytes, cudaMemcpyHostToDevice, s1); // overlaps with s0 work
kernel<<<g, b, 0, s1>>>(d1);
cudaStreamSynchronize(s0);   // wait only for s0
cudaStreamSynchronize(s1);
cudaStreamDestroy(s0); cudaStreamDestroy(s1);
```
- **What it demonstrates**: stream as 4th launch param, async copy + pinned memory, per-stream synchronization instead of a global device sync.

## Reference Tables
| Need | Use |
|---|---|
| Order a sequence of ops | one stream |
| Overlap independent work | multiple streams |
| Wait for everything | `cudaDeviceSynchronize()` |
| Wait for one stream | `cudaStreamSynchronize(s)` |
| Check without blocking | `cudaStreamQuery(s)` |
| Time GPU work | events + `cudaEventElapsedTime` |
| Run host code on completion | stream callback |

## Worked Example
**Why `cudaDeviceSynchronize` is the wrong tool in a multi-stream app.** With two independent pipelines on `s0` and `s1`, calling `cudaDeviceSynchronize()` after `s0`'s work forces the host to wait for *both* streams, destroying the overlap you built. `cudaStreamSynchronize(s0)` waits only for `s0`, letting `s1` keep running — preserving concurrency. Prefer the narrowest synchronization that gives correctness.

## Anti-patterns
- **Async copy from pageable (non-pinned) host memory**: silently becomes synchronous / incorrect for overlap.
- **Over-using the default stream**: its blocking semantics serialize work you intended to overlap.
- **Global `cudaDeviceSynchronize` in hot loops**: kills concurrency; use stream/event sync.

## Key Takeaways
1. Streams order work; separate streams enable overlap.
2. Kernel launches & async copies return immediately — synchronize before using results.
3. Async copies need pinned host memory.
4. Synchronize at the narrowest scope (stream/event), not the whole device.
5. Events give precise GPU timing and cross-stream ordering.

## Connects To
- **Ch 3**: pinned (`cudaMallocHost`) memory enables async copies.
- **Ch 8 (CUDA Graphs)**: capture a stream workflow once, replay with low overhead.
- **Ch 4 (Error checking)**: async errors surface at stream sync points.
