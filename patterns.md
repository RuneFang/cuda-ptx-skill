# Patterns & Techniques — CUDA / PTX

## Global Thread-Index Mapping
**When to use**: every data-parallel SIMT kernel mapping threads to data.
**How**: `int i = threadIdx.x + blockDim.x * blockIdx.x;` Replicate per dimension for 2-D/3-D.
**Trade-offs**: trivial and universal; map `threadIdx.x` to the contiguous data dimension to keep global access coalesced.

## Ceil-Div Launch Sizing + Bounds Check
**When to use**: problem size not a multiple of block size (almost always).
**How**: `blocks = cuda::ceil_div(n, threads)` (or `(n+threads-1)/threads`), and guard the kernel body with `if (i < n)`.
**Trade-offs**: lets you over-launch threads cheaply; never over-launch whole idle *blocks*. Start at 256 threads/block.

## Unified vs Explicit Memory
**When to use**: Unified (`cudaMallocManaged`) for simplicity/prototyping; Explicit (`cudaMalloc`+`cudaMemcpy`) when you need transfer control/overlap.
**How**: Unified = allocate, access from either side, `cudaDeviceSynchronize`, `cudaFree`. Explicit = `cudaMalloc` device buffers, `cudaMemcpy` H↔D with pinned (`cudaMallocHost`) host buffers.
**Trade-offs**: Unified is less code but driver-controlled migration; tune it with `cudaMemAdvise`/`cudaMemPrefetchAsync` before rewriting to explicit.

## Shared-Memory Block Reduction
**When to use**: combine per-thread values within a block (sum/max/etc.).
**How**: load into `extern __shared__`, `__syncthreads()`, halve-stride loop with `__syncthreads()` each step, thread 0 writes the block result; then reduce across blocks (second pass or atomics).
**Trade-offs**: fast on-chip cooperation; watch bank conflicts and never call `__syncthreads()` in divergent control flow.

## Stream Overlap (Copy ∥ Compute)
**When to use**: hide transfer latency behind computation.
**How**: separate streams; `cudaMemcpyAsync` (pinned host memory) + `kernel<<<g,b,0,stream>>>`; synchronize per-stream (`cudaStreamSynchronize`), not the whole device.
**Trade-offs**: requires pinned memory; default-stream usage and global sync destroy the overlap.

## CUDA Graph Replay
**When to use**: a workflow launched many times where CPU launch overhead dominates (short kernels).
**How**: stream-capture (`cudaStreamBeginCapture`...`EndCapture`) or explicit Graph API → `cudaGraphInstantiate` once → `cudaGraphLaunch` in the loop.
**Trade-offs**: big win for launch-bound loops; instantiation cost amortized only if replayed many times; graph must be re-instantiated if topology changes.

## Robust Error Checking
**When to use**: all production code.
**How**: `CUDA_CHECK(...)` macro on every API; after a launch, check **both** `cudaGetLastError()` (config/sync) and a sync point (`cudaDeviceSynchronize`/stream sync) for async errors. Set `CUDA_LOG_FILE` while developing.
**Trade-offs**: negligible overhead; skipping the async check hides whole bug classes.

## Inline PTX with Scoped Temp Registers
**When to use**: you need a PTX instruction C++ doesn't expose, in an inlinable `__device__` function.
**How**: wrap the asm body in `{ ... }` and declare `.reg` inside, so each inlining gets a fresh scope; use `=`/`+` modifiers correctly; add `volatile` + `"memory"` clobber for side-effecting reads.
**Trade-offs**: maximal control; template string isn't validated until `ptxas`; easy to get memory-space/optimization assumptions wrong.

## Compile-Time PTX Customization via "C" Constraint
**When to use**: select PTX instruction modifiers (e.g. rounding mode `.rn`/`.rz`) from compile-time values.
**How**: store the modifier in a `static constexpr const char[]` and pass it with the `"C"` constraint; the compiler splices its contents into the template.
**Trade-offs**: enables templated/specialized PTX without runtime branching; device-only, no constraint modifiers, the array must be constant-initialized.

## Tiled GEMM (matmul/mma)
**When to use**: matrix multiply in the tile programming model.
**How**: build `partition_view`s over A/B/C; loop `ceil(K/tk)` K-tiles calling `mma(a, b, acc)` with an **FP32 accumulator**; zero-pad partial K-tiles (`load_masked`/`PaddingMode.ZERO`) and discard OOB M/N edges on store (`store_masked`/masked `store`); cast acc to output dtype on store.
**Trade-offs**: maps to tensor cores automatically; FP32 accumulate preserves accuracy across precisions; tile dims must be power-of-two and compile-time known.

## Host↔Device Symbol Transfer
**When to use**: read/write a named `__device__`/`__constant__` variable from the host.
**How**: `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol` (and `cudaGetSymbolAddress`/`Size`). `__constant__` is host-write/device-read-only.
**Trade-offs**: clean for small config/coefficient data; not a substitute for bulk buffers.

## Forward-Portable Shipping (PTX + Fatbin)
**When to use**: one binary must run on current and future GPUs.
**How**: compile multiple `compute_XX`/`sm_XX` targets into a fatbin and include PTX so the driver JITs for newer hardware.
**Trade-offs**: larger binary + JIT cost on first run for new archs; arch/family-specific features still bind to exact CC/family.
