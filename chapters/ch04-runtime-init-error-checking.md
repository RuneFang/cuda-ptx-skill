# Chapter 4: Runtime Initialization & Error Checking

## Core Idea
The CUDA runtime lazily creates a **primary context** per device. Every CUDA API returns a `cudaError_t` — in production you must check **every** call, including catching **asynchronous** errors that surface later.

## Frameworks Introduced
- **Primary context lifecycle**: created on first API needing a context (device code JIT-compiled + loaded transparently); shared across host threads. As of CUDA 12.0, `cudaInitDevice` and `cudaSetDevice` explicitly initialize the runtime + primary context; before 12.0 `cudaSetDevice` did not. `cudaDeviceReset` destroys the current device's primary context.
- **Synchronous error checking**: wrap each call in a macro that checks the returned `cudaError_t == cudaSuccess`.
- **Asynchronous error checking**: kernel launches & async ops report errors *later*. Two error states exist; query with `cudaGetLastError()` (clears) / `cudaPeekAtLastError()` (doesn't). A sticky error persists until reset.
- **`CUDA_LOG_FILE`**: environment variable for richer error descriptions during development (esp. when the same error code covers multiple causes).

## Key Concepts
- **`cudaError_t`**: enumerated return type; success is `cudaSuccess`.
- **`cudaGetLastError()` / `cudaPeekAtLastError()`**: retrieve last error (clearing / non-clearing).
- **`cudaGetErrorString(err)`**: human-readable message.
- **synchronous vs asynchronous error**: API-call-time error vs error detected during later device execution.
- **`cudaInitDevice` / `cudaSetDevice` / `cudaDeviceReset`**: init / select / tear-down context.

## Code Examples
The standard error-checking macro pattern:
```cpp
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err_ = (call);                                  \
    if (err_ != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",         \
                cudaGetErrorName(err_), __FILE__, __LINE__,     \
                cudaGetErrorString(err_));                      \
        abort();                                                \
    }                                                           \
} while (0)

CUDA_CHECK(cudaMalloc(&p, bytes));
kernel<<<g, b>>>(p);
CUDA_CHECK(cudaGetLastError());      // catch launch-config errors
CUDA_CHECK(cudaDeviceSynchronize()); // catch async execution errors
```
- **What it demonstrates**: check synchronous return values *and* probe for async errors after launches.

## Worked Example
**Why you check twice after a launch.** A bad launch configuration (e.g. too many threads/block) is reported synchronously and caught by `cudaGetLastError()` immediately after `<<<>>>`. But an out-of-bounds device memory access executes *during* the kernel and is only observable after the work runs — `cudaDeviceSynchronize()` (or the next blocking call) returns it. Checking only one of the two silently drops a whole class of bugs.

## Key Takeaways
1. Every CUDA API returns `cudaError_t` — check it in production.
2. After a kernel launch, check **both** `cudaGetLastError()` (sync/config) and a sync point (async/execution).
3. As of CUDA 12.0, check `cudaSetDevice`'s return — it now initializes the runtime.
4. Don't use CUDA interfaces before `main` / after `main` returns — undefined behavior.
5. Set `CUDA_LOG_FILE` during development for clearer diagnostics.

## Connects To
- **Ch 2**: launches are async, hence async errors.
- **Ch 6 (Streams)**: per-stream synchronization points are where async errors surface.
- **Ch 4 PTX Error Checking** (Inline PTX guide): error checking specific to PTX/driver level.
