# Chapter 6: Tile Programming (CUDA Tile / cuTile)

## Core Idea
Tile programming is an **alternative to SIMT**: you write code at the **block level** operating on multidimensional **tiles** of data, and the compiler maps tile operations onto the block's threads (including tensor cores). One control flow per block → **no warp divergence**.

## Frameworks Introduced
- **Tile programming model**: programmer specifies only grid dimensions; the compiler picks threads-per-block from the tile operations. Block executes a single control flow; scalar ops run on one thread, tile ops run collectively in parallel.
- **Arrays vs Tiles**:
  - **Array** (global array): mutable, multidimensional container in device memory; has shape + dtype.
  - **Tile**: immutable, block-local collection of values; every op produces a *new* tile. May live in registers/shared memory (compiler's choice). Each tile dim must be a **power of two** and **compile-time known**. Tiles can't be kernel parameters.
- **Tile space + load/store**: conceptually partition an array of shape `(M,N)` into `⌈M/tm⌉ × ⌈N/tn⌉` tiles; a load at index `(i,j)` returns a `(tm,tn)` tile. Edge tiles specify OOB handling (zero-fill, masked).
- **Broadcasting & arithmetic**: element-wise ops produce a new tile of the broadcast shape; scalars broadcast across all elements. Mixed types → the higher-precision/range type wins (`int + float → float`). C++ rejects narrowing scalar-tile ops; Python promotes.
- **Tile primitives**: factory (`iota`, `full`), load/store, element-wise arithmetic, and **matmul/mma** (mapped to tensor cores).

## Key Concepts
- **`__tile_global__`** (C++) / `@ct.kernel` (Python): tile kernel entry.
- **`cuda::tiles` / `ct`** (C++), **`cutile`/`ct`** (Python): the tile namespaces.
- **matmul (`a @ b`) vs mma (`a @ b + acc`)**: pure multiply vs multiply-accumulate (accumulator carries partial products across K-tiles).
- **`partition_view` / `tensor_span` / `extents` / `shape`**: views that map arrays into tile space.
- **`load_masked` / `store_masked` (C++), `PaddingMode.ZERO` (Python)**: handle partial edge tiles.

## Code Examples
Tiled GEMM (the canonical pattern — FP32 accumulate, cast on store):
```cpp
__tile_global__ void gemm(const __half* A, const __half* B, float* C,
                          std::size_t M, std::size_t K, std::size_t N) {
    namespace ct = cuda::tiles; using namespace ct::literals;
    using f32_acc = ct::tile<float, ct::shape<32,32>>;
    constexpr auto tm=32_ic, tn=32_ic, tk=16_ic;
    auto aView = ct::partition_view{ct::tensor_span{A, ct::extents{M,K}}, ct::shape{tm,tk}};
    auto bView = ct::partition_view{ct::tensor_span{B, ct::extents{K,N}}, ct::shape{tk,tn}};
    auto cView = ct::partition_view{ct::tensor_span{C, ct::extents{M,N}}, ct::shape{tm,tn}};
    auto [bx,by,bz] = ct::bid();
    auto acc = ct::full<f32_acc>(0.0f);                 // FP32 accumulator
    std::size_t num_k = (K + tk - 1) / tk;
    for (auto k : ct::irange(std::size_t{0}, num_k))
        acc = ct::mma(aView.load_masked(bx,k),          // zero-pad partial K-tile
                      bView.load_masked(k,by), acc);     // acc += a @ b
    cView.store_masked(acc, bx, by);                     // drop OOB edge lanes
}
```
- **What it demonstrates**: tile-space views, K-loop with mma, masked load/store for edges, FP32-accumulate idiom.

## Reference Tables
| | SIMT model | Tile model |
|---|---|---|
| Programmer writes | per-thread code | per-block code on tiles |
| Threads/block | you choose | compiler decides |
| Divergence | possible (warps) | none (single block control flow) |
| Data unit | scalars / arrays | tiles (immutable) |
| Tensor cores | manual/intrinsics | automatic via matmul/mma |

## Key Takeaways
1. Tiles are immutable, block-local, power-of-two, compile-time-shaped.
2. You give grid dims; the compiler chooses threads/block.
3. No warp divergence — one control flow per block.
4. matmul/mma map to tensor cores; accumulate in FP32, cast on store.
5. Use masked load/store (or `PaddingMode.ZERO`) for partial edge tiles.

## Connects To
- **Ch 1**: tile space generalizes the "partition into a grid of tiles" idea.
- **Ch 5 (SIMT)**: the model you'd otherwise use; tiles trade control for productivity.
- **Ch 11**: tensor-core throughput by compute capability.
