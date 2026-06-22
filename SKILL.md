---
name: cudaskill
description: "Knowledge base from NVIDIA CUDA documentation (CUDA Programming Guide, Quick Start, Features Archive, Inline PTX Assembly, PTX Writer's Guide to Interoperability), CUDA 13.3. Use when writing or reviewing CUDA C++/Python kernels, applying the CUDA programming model (threads/warps/blocks/clusters/grids, SIMT, tiles), managing GPU memory, using streams/events/graphs/cooperative groups, writing inline PTX, targeting compute capabilities, or referencing CUDA APIs and concepts."
---

<!-- argument-hint: [topic, API name, or chapter number, e.g. "shared memory", "cudaMemcpyAsync", "ch07"] -->

# NVIDIA CUDA Programming & PTX
**Source**: NVIDIA CUDA documentation (5 docs) | **Release**: 13.3 | **Words**: ~263K | **Chapters**: 16 | **Generated**: 2026-06-22

## How to Use This Skill

- **Without arguments** — load core frameworks for reference.
- **With a topic** — ask about `streams`, `unified memory`, `inline PTX`, `tile programming`, `compute capability`, etc.; I find and read the relevant chapter.
- **With a chapter** — ask for `ch09`; I load that file.
- **Browse** — ask "what chapters do you have?" for the full index.

When you ask about a topic not in Core Frameworks below, I read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Heterogeneous model.** Code always *starts on the host (CPU)*; the GPU is a throughput coprocessor. Host code copies data, launches kernels, and synchronizes. CPU+GPU run concurrently — maximize both.

**Thread hierarchy (the central abstraction).**
`thread → warp (32, SIMT lock-step) → block (one SM, shared mem + __syncthreads) → cluster (CC≥9.0, one GPC, distributed shared mem) → grid`.
**Hard rule:** thread blocks must run in **any order** — this is what lets one binary scale from 1 SM to thousands. Never make one block depend on another in the same grid (except clusters / Cooperative Groups).

**SIMT ≠ SIMD.** All 32 warp threads run the same instruction, but each may branch; divergent branches **mask off** lanes (warp divergence = lost throughput). Size blocks as a **multiple of 32**.

**Kernel + launch (CUDA C++).** `__global__ void k(...)`, launch `k<<<blocks, threads, shBytes, stream>>>(args)`. Canonical index: `int i = threadIdx.x + blockDim.x*blockIdx.x;` then `if (i<n)`. Sizing: `blocks = cuda::ceil_div(n, threads)`; **start at 256 threads/block**. Launches are **asynchronous**.

**Memory choices.** Unified (`cudaMallocManaged`/`__managed__`) = simple, driver-migrated; tune with `cudaMemPrefetchAsync`/`cudaMemAdvise`. Explicit (`cudaMalloc`+`cudaMemcpy`) = control/overlap; use **pinned** host buffers (`cudaMallocHost`) for transfers (required for async). `cudaMemcpy` is **synchronous**.

**Memory hierarchy.** registers (thread) → shared `__shared__` (block, on-SM, ~L1) → global (device DRAM) → constant (RO, cached). **Coalesce** global access by mapping `threadIdx.x` to the contiguous dimension. `__syncthreads()` is an **intra-block** barrier (never in divergent control flow).

**Asynchrony.** Streams = ordered work queues; separate streams overlap. Events = timing/cross-stream sync. Synchronize at the **narrowest** scope (`cudaStreamSynchronize`, not `cudaDeviceSynchronize`). **CUDA Graphs**: capture a repeated workflow once, replay cheaply (huge win when launch overhead dominates short kernels). **Cooperative Groups** for structured cross-block sync.

**Tile programming (alternative to SIMT).** Write block-level code on immutable, power-of-two, compile-time-shaped **tiles**; the compiler maps ops to threads (and tensor cores). One control flow per block → **no warp divergence**. GEMM idiom: loop K-tiles with `mma(a,b,acc)`, **FP32 accumulate**, cast on store; mask partial edges.

**Inline PTX.** `asm("tmpl" : outs : ins : clobbers)`; outputs first (`%0,%1,...`). `=` write-only, `+` read-write. Add `volatile` + `"memory"` clobber for side-effecting/hidden-memory asm. Constraint letters by width: `h`/`r`/`l`/`q`=u16/u32/u64/u128, `f`/`d`=f32/f64, `n`=imm, `C`=compile-time string. Template strings aren't validated until `ptxas`.

**Error checking.** Every API returns `cudaError_t`. After a launch, check **both** `cudaGetLastError()` (config/sync) **and** a sync point (async execution).

**Compatibility.** cubin = arch binary (`sm_XX`); PTX = forward-portable IR (JIT at load); fatbin = bundle. **Baseline** features are forward-available; **architecture-specific** (CC≥9.0) and **family-specific** (CC≥10.0) features are not portable across all later GPUs and need matching compiler targets.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-programming-model.md) | The CUDA Programming Model | heterogeneous model, thread hierarchy, SIMT, clusters |
| [ch02](chapters/ch02-cuda-cpp-kernels.md) | Writing Kernels in CUDA C++ | `__global__`, triple chevron, index idiom, ceil-div, bounds check |
| [ch03](chapters/ch03-memory-management.md) | Memory Management | unified vs explicit, `cudaMalloc`/`cudaMemcpy`, pinned memory |
| [ch04](chapters/ch04-runtime-init-error-checking.md) | Runtime Init & Error Checking | primary context, sync vs async errors, CUDA_CHECK macro |
| [ch05](chapters/ch05-simt-shared-memory.md) | SIMT, Memory Spaces & Shared Memory | memory hierarchy, `__syncthreads()`, coalescing, reduction |
| [ch06](chapters/ch06-tile-programming.md) | Tile Programming (cuTile) | tiles vs arrays, tile space, matmul/mma, masked load/store |
| [ch07](chapters/ch07-async-streams-events.md) | Async Execution — Streams & Events | streams, events, async copies, overlap |
| [ch08](chapters/ch08-advanced-features.md) | Advanced CUDA Features | CUDA Graphs, Cooperative Groups, async copies, VMM, dynamic parallelism |
| [ch09](chapters/ch09-inline-ptx-assembly.md) | Inline PTX Assembly | `asm()`, constraints, modifiers, volatile/memory, pitfalls |
| [ch10](chapters/ch10-ptx-abi-interoperability.md) | PTX Writer's Guide (ABI) | type sizes/alignments, aggregates, bit fields, calling sequence, opaque objects |
| [ch11](chapters/ch11-cpp-language-extensions.md) | C/C++ Language Extensions | execution/memory specifiers, `__CUDA_ARCH__`, symbol APIs |
| [ch12](chapters/ch12-compute-capability-targets.md) | Compute Capability & Compiler Targets | CC detection, baseline/arch/family feature sets |
| [ch13](chapters/ch13-cuda-python.md) | CUDA in Python | `@ct.kernel`, `ct.bid`, `tiled_view`, one-call load/store |
| [ch14](chapters/ch14-nvcc-compilation.md) | NVCC — The CUDA Compiler | nvcc vs nvrtc, PTX→ptxas→cubin→fatbin, file types |
| [ch15](chapters/ch15-install-toolkit.md) | Installation & Toolkit | installer types, driver vs toolkit, verify, metapackages |
| [ch16](chapters/ch16-release-features-deprecations.md) | Release Features & Deprecations | feature taxonomy, deprecation suppression, removed APIs |

## Topic Index

- **async copy / overlap** → ch07, ch08, ch03
- **atomics** → ch05
- **bit fields / ABI / type layout** → ch10
- **bounds checking** → ch02
- **clusters / distributed shared memory** → ch01, ch08
- **coalescing** → ch05
- **compute capability** → ch12, ch01
- **constant memory** → ch05, ch11
- **Cooperative Groups** → ch08, ch05
- **cubin / fatbin / PTX / JIT** → ch12, ch14, ch01
- **CUDA Graphs / stream capture** → ch08, ch07
- **CUDA Python / tile Python** → ch13, ch06
- **deprecations / known issues** → ch16
- **dynamic parallelism** → ch08
- **error checking** → ch04, ch07
- **events / timing** → ch07
- **execution / memory specifiers** → ch11, ch02
- **inline PTX `asm()` / constraints** → ch09
- **installation / driver / toolkit** → ch15
- **kernels / launch / triple chevron** → ch02, ch11
- **matmul / mma / tensor cores** → ch06, ch12
- **nvcc / nvrtc / ptxas** → ch14, ch12
- **opaque objects (texture/surface)** → ch10
- **pinned (page-locked) memory** → ch03, ch07
- **programming model / heterogeneous** → ch01
- **runtime init / primary context** → ch04
- **shared memory / `__syncthreads()`** → ch05, ch02
- **SIMT / warps / divergence** → ch01, ch05
- **streams** → ch07, ch08
- **symbol transfer (`cudaMemcpyToSymbol`)** → ch11
- **thread/block/grid hierarchy** → ch01
- **tile programming / tiles** → ch06, ch13, ch01
- **unified memory** → ch03, ch01
- **virtual memory management** → ch08

## Supporting Files

- [glossary.md](glossary.md) — all key terms with definitions and chapter refs
- [patterns.md](patterns.md) — concrete techniques (index mapping, reduction, stream overlap, graph replay, tiled GEMM, inline-PTX scoping)
- [cheatsheet.md](cheatsheet.md) — decision rules, defaults, tells & smells

---

## Scope & Limits

This skill covers the five NVIDIA CUDA documents above (CUDA 13.3). It is conceptual/reference knowledge — combine with the official API reference for exhaustive signatures, and with `nvcc`/profilers for hands-on builds and tuning. For features beyond these docs, check NVIDIA's current documentation or ask directly.
