# Chapter 8: Advanced CUDA Features

## Core Idea
Beyond basic kernels, CUDA offers a large feature surface for performance and scale: **CUDA Graphs** (record-once/replay work), **Cooperative Groups** (structured synchronization), **stream-ordered allocation**, **async barriers/pipelines/copies**, **virtual memory management**, **multi-GPU**, and **dynamic parallelism**.

## Frameworks Introduced
- **CUDA Graphs**: define a DAG of operations (nodes) connected by dependencies (edges) once, then launch repeatedly with minimal CPU overhead. Three stages: **definition → instantiation (executable graph) → execution**. Build via the **explicit Graph API** (`cudaGraphCreate`, `cudaGraphAddNode`) or **stream capture** (`cudaStreamBeginCapture` ... `cudaStreamEndCapture`).
  - When to use: a workflow launched many times where per-launch CPU overhead dominates (especially short kernels).
- **Cooperative Groups**: API to define and synchronize thread groups at multiple granularities (warp, block, cluster, grid, multi-grid). Enables structured cross-block sync that raw `__syncthreads()` can't.
- **Stream-Ordered Memory Allocator**: `cudaMallocAsync` / `cudaFreeAsync` — allocation/free ordered within a stream, enabling memory reuse and pooling without global sync.
- **Asynchronous Barriers / Pipelines / Async Data Copies**: `cuda::barrier`, pipeline primitives, and `memcpy_async` to overlap global→shared copies with compute (key for high-throughput kernels).
- **Virtual Memory Management**: reserve address ranges and map physical memory explicitly (`cuMemCreate`/`cuMemMap`) for advanced allocators.
- **CUDA Dynamic Parallelism**: kernels launch child kernels directly from the device.
- **Multi-GPU programming**: enumerate devices, peer access, and distribute work across GPUs.

## Key Concepts
- **graph node types**: kernel, CPU function, memcpy, memset, empty, event wait/record, external semaphore signal/wait, conditional, memory, child graph.
- **stream capture**: record stream operations into a graph instead of executing them.
- **executable graph**: instantiated, validated, launch-ready graph instance.
- **cooperative launch**: `cudaLaunchCooperativeKernel` for grid-wide sync.
- **programmatic dependent launch**: overlap tail of one kernel with start of the next (`cudaGraphDependencyTypeProgrammatic`).
- **green contexts**: partition SM resources for concurrent contexts.

## Code Examples
Stream capture is the easiest way to build a graph from existing stream code:
```cpp
cudaGraph_t graph; cudaGraphExec_t exec;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<g,b,0,stream>>>(...);          // recorded, not executed
cudaMemcpyAsync(..., stream);
kernelB<<<g,b,0,stream>>>(...);
cudaStreamEndCapture(stream, &graph);    // graph now describes the workflow
cudaGraphInstantiate(&exec, graph, 0);   // one-time setup
for (int i = 0; i < iters; ++i)
    cudaGraphLaunch(exec, stream);       // replay cheaply
```
- **What it demonstrates**: capture a repeated workflow once; replay it with near-zero per-launch CPU overhead.

## Reference Tables
| Feature | Primary API | Use when |
|---|---|---|
| CUDA Graphs | `cudaGraph*`, stream capture | repeated workflow, launch overhead matters |
| Cooperative Groups | `cooperative_groups::` | structured / cross-block sync |
| Stream-ordered alloc | `cudaMallocAsync`/`cudaFreeAsync` | frequent alloc/free, pooling |
| Async copies | `memcpy_async`, `cuda::pipeline` | overlap global→shared with compute |
| Virtual Memory Mgmt | `cuMemCreate`/`cuMemMap` | custom allocators, growable buffers |
| Dynamic Parallelism | device-side `<<<>>>` | data-dependent nested parallelism |

## Worked Example
**Graph payoff for short kernels.** Suppose a workflow of 20 small kernels each takes 5 µs on the GPU but ~8 µs of CPU launch overhead in stream mode. Per iteration: ~20 × 8 µs = 160 µs of CPU launch cost dominates the ~100 µs of GPU work. Capturing the 20 kernels into a graph pays the setup once at instantiation; each `cudaGraphLaunch` then submits the whole DAG with a single, much smaller overhead — often turning a launch-bound loop into a compute-bound one.

## Key Takeaways
1. CUDA Graphs: record once, replay cheaply — best for launch-overhead-bound, repeated workflows.
2. Build graphs via explicit API or (easier) stream capture; always instantiate before launching.
3. Cooperative Groups give structured, multi-granularity synchronization beyond `__syncthreads()`.
4. `cudaMallocAsync`/`cudaFreeAsync` make allocation stream-ordered and poolable.
5. Async copies + pipelines overlap data movement with compute — central to high-throughput kernels.

## Connects To
- **Ch 7 (Streams)**: graphs are built from / capture stream workflows.
- **Ch 5 (Shared memory)**: async copies feed shared memory while compute proceeds.
- **Ch 9 (Driver API)**: virtual memory & low-level context control live there.
