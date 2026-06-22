# Chapter 1: The CUDA Programming Model (Language-Independent)

## Core Idea
CUDA assumes a **heterogeneous system** (CPU host + GPU device) and a thread hierarchy (thread → warp → block → cluster → grid) that lets *one* program scale unchanged from a 1-SM GPU to a thousands-SM GPU. The hard rule that makes this work: **thread blocks must be able to run in any order, in parallel or in series.**

## Frameworks Introduced
- **Heterogeneous execution model**: Code always *starts on the host (CPU)*. Host code uses CUDA APIs to copy data, launch *kernels* (device functions invoked on the GPU), and wait for completion. CPU and GPU run concurrently; best performance maximizes utilization of both.
- **GPU hardware model**: GPU = collection of **Streaming Multiprocessors (SMs)** grouped into **Graphics Processing Clusters (GPCs)**. Each SM has a register file, a unified data cache (split at runtime into L1 + shared memory), and functional units.
- **Thread hierarchy**:
  - **Thread** → unique work item.
  - **Warp** = 32 threads executing in lock-step (SIMT).
  - **Thread block** = group of threads; *all run on a single SM*, can share on-chip shared memory and synchronize via `__syncthreads()`.
  - **Cluster** (compute capability ≥ 9.0, optional) = group of blocks scheduled together in one GPC; enables **distributed shared memory** and cross-block sync via Cooperative Groups.
  - **Grid** = all blocks of a kernel launch; 1/2/3-dimensional.
- **SIMT (Single-Instruction Multiple-Threads)**: All 32 warp threads execute the same instruction; divergent branches mask off lanes. Contrast with SIMD: SIMT threads each have their *own* control flow path (no fixed data width).

## Key Concepts
- **host / device**: CPU+its memory / GPU+its memory.
- **kernel**: function executed on the GPU; "launching" starts many threads running it.
- **execution configuration**: grid + block dims (+ optional cluster size, stream, SM config) specified at launch.
- **warp divergence**: threads in a warp taking different branches → serialized → lower utilization.
- **distributed shared memory**: shared memory of all blocks in a cluster, mutually accessible.
- **compute capability**: versioned feature/hardware level of an SM (e.g. 9.0).

## Mental Models
- **Think of a block as a team pinned to one SM** — they can whisper (shared memory) and line up together (`__syncthreads()`); blocks in different SMs cannot coordinate cheaply.
- **Design for order-independence**: if block A needs a result from block B in the same grid, your design is wrong (except clusters / cooperative launch).
- **Multiples of 32**: size blocks as a multiple of 32 threads, or the last warp wastes lanes.

## Anti-patterns
- **Inter-block data dependency in one grid**: blocks may never be co-resident → deadlock / wrong results.
- **Assuming hardware warp scheduling details**: "exploiting knowledge of how warp execution maps to real hardware is discouraged" — violating the SIMT model is undefined behavior that differs across GPUs.
- **Block size not a multiple of 32**: legal but wastes functional units in the final warp.

## Worked Example
A `(M, N)` array with tile shape `(tm, tn)` is conceptually partitioned into ⌈M/tm⌉ × ⌈N/tn⌉ **tile space**. A load at tile index `(i, j)` returns a `(tm, tn)` tile; edge tiles that overrun the array specify out-of-bounds handling (e.g. zero-fill). This "partition the array into a grid of tiles, index the tile you own" pattern is the conceptual basis for both classic blocked algorithms and the newer tile programming model (Ch 5).

## Key Takeaways
1. Execution always begins on the host; the GPU is a throughput coprocessor.
2. The thread hierarchy (warp 32 → block → cluster → grid) is the central abstraction.
3. Blocks must be order-independent — this is what makes one binary scale across GPU sizes.
4. SIMT ≠ SIMD: each thread may branch, but divergence costs performance.
5. Shared memory + `__syncthreads()` are intra-block only; cross-block coordination needs clusters or Cooperative Groups.

## Connects To
- **Ch 2**: how the model is expressed in CUDA C++ (kernels, launch, index intrinsics).
- **Ch 5 (Tile Programming)**: an alternative to per-thread SIMT coding.
- **Compute Capability** (glossary): determines which features (clusters, etc.) exist.
