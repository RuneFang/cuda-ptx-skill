# CUDA / PTX Cheatsheet — Decisions & Defaults

## Block / launch sizing
- **Default block size**: start at **256 threads**; always a **multiple of 32** (else the last warp wastes lanes).
- **Grid size**: `blocks = cuda::ceil_div(n, threads)`; over-launch threads + `if (i<n)` guard. Never launch fully-idle blocks.
- **Launch form**: `k<<<blocks, threads, sharedBytes, stream>>>(args)` — 3rd/4th params optional.

## Memory: which API?
| Situation | Choose |
|---|---|
| Prototyping / irregular access / simplicity | `cudaMallocManaged` (unified) |
| Need transfer-timing control / overlap | `cudaMalloc` + `cudaMemcpy` (explicit) |
| Unified too slow | keep managed + `cudaMemPrefetchAsync` / `cudaMemAdvise` |
| Host buffer used in copies | `cudaMallocHost` (pinned) — **required** for async copies |
| Small named config/coeffs | `__constant__` + `cudaMemcpyToSymbol` |

## Synchronization: pick the narrowest
- One block's threads → `__syncthreads()` (never in divergent branches).
- One stream → `cudaStreamSynchronize(s)`.
- Whole device → `cudaDeviceSynchronize()` (**avoid** in multi-stream hot paths — kills overlap).
- Cross-block → clusters (CC≥9.0) or Cooperative Groups; not raw `__syncthreads()`.
- GPU timing → events + `cudaEventElapsedTime`.

## Error checking (always two checks after a launch)
1. `cudaGetLastError()` → launch-config / synchronous errors.
2. sync point (`cudaDeviceSynchronize` / stream sync) → asynchronous execution errors.
- Dev aid: set `CUDA_LOG_FILE`.

## SIMT vs Tile — when to use which
| If you… | Use |
|---|---|
| Need per-thread control, manual memory mapping | **SIMT** (`__global__`) |
| Want block-level data ops, no divergence, auto tensor cores | **Tile** (`__tile_global__` / `@ct.kernel`) |
| Do matmul/GEMM | Tile `mma`, **FP32 accumulator**, cast on store |

## Inline PTX `asm()` decision rules
- Output modifier: write-only → `=`; conditionally-updated / read-modify-write → `+`.
- Side-effecting or hidden memory read → add `volatile` **and** `"memory"` clobber.
- Inlined function with a `.reg` → wrap body in `{}` for per-inlining scope.
- Constraint by width: `h`=u16, `r`=u32, `l`=u64, `q`=u128, `f`=f32, `d`=f64, `n`=imm-int, `C`=compile-time `const char[]`.
- Remember: template string is **not** validated until `ptxas`.

## Coalescing rule
- Map `threadIdx.x` to the **innermost contiguous** data dimension → coalesced global access. Strided per-thread access multiplies transactions.

## Compute capability / portability
| Feature tier | Portable to later GPUs? | Target needed |
|---|---|---|
| Baseline | ✅ all subsequent | default |
| Architecture-specific (CC≥9.0) | ❌ only exact CC | arch-specific |
| Family-specific (CC≥10.0) | ✅ within family only | family-specific |
- Detect CC: `nvidia-smi --query-gpu=name,compute_cap` or `cudaDeviceGetAttribute(...ComputeCapabilityMajor/Minor)`.
- Ship **PTX + fatbin** for forward portability (driver JITs for new archs).

## Compiler pipeline (nvcc)
`device code → PTX (compute_XX) → ptxas → cubin (sm_XX) → fatbin`. Offline = `nvcc`; JIT = `nvrtc`. Inspect with `-v`; keep intermediates with `-keep`. `.cu`/`.cuh` for device code.

## CUDA Graph: worth it?
- ✅ Repeated workflow + short kernels + launch overhead ≳ GPU time → capture once, replay.
- ❌ One-shot work, or topology changes every iteration (re-instantiation cost).

## Deprecation handling
- Scoped silence: `__NV_SILENCE_HOST_DEPRECATION_BEGIN` … `_END` (temporary, not a fix).
- Removed `cudaDeviceAttr` (no replacement) → redesign feature detection.
- Pin CUDA version in CI; read **Known Issues** before adopting a new feature.

## Tells & smells
- Multi-stream app calling `cudaDeviceSynchronize` in the loop → you've serialized your overlap.
- `cudaMemcpyAsync` from pageable memory → silently not overlapping (needs pinned).
- Block size like 100/200 → wasted last-warp lanes (use 96/128/256).
- asm works at `-O0`, breaks optimized → missing `volatile`/`"memory"` clobber.
- Binary runs on dev GPU, fails on another → arch-specific target without PTX/fatbin fallback.
