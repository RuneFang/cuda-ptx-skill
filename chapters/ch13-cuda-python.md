# Chapter 13: CUDA in Python

## Core Idea
CUDA Python lets you write SIMT *and* tile kernels in Python with the same programming model as C++. Kernels are decorated functions; index intrinsics, memory transfers, and tile loads/stores mirror the C++ APIs.

## Frameworks Introduced
- **CUDA Python ecosystem**: use NVIDIA GPU libraries from Python and/or write kernels directly. Setup and launch covered in the guide's "Getting Setup" / "Running CUDA Python Applications".
- **SIMT kernels in Python**: define a kernel, specify it, launch it across multi-dimensional grids/blocks, use thread/grid index intrinsics — the SIMT model identical to C++.
- **Memory in Python GPU computing**: instantiate arrays on the GPU, copy between host and GPU memory, work with the **`ndarray`** object type; synchronize CPU↔GPU.
- **Tile kernels in Python** (`@ct.kernel`): two equivalent access styles —
  - **`tiled_view`**: bind the tile shape once to a view, then `.load(index)` / `.store(index, tile)` (preferred when reused).
  - **One-call `ct.load` / `ct.store`**: pass `index` and `shape` inline (concise for one-offs).
- **Error checking in CUDA Python**: Pythonic error handling for CUDA calls.

## Key Concepts
- **`@ct.kernel`**: decorator marking a tile kernel.
- **`ct.bid(axis)`**: this block's tile-space index along an axis (Python analog of `ct::bid()`).
- **`ct.Constant[int]`**: compile-time constant kernel parameter (e.g. `TILE`).
- **`tiled_view(shape)` / `.load` / `.store`**: view-based tile access.
- **`ct.load(array, index, shape)` / `ct.store(array, index, tile)`**: one-call tile access.
- **`PaddingMode.ZERO`**: zero-fill partial edge tiles on load.
- **scalar–tile typing**: Python *promotes* on narrowing scalar-tile ops (C++ rejects them).

## Code Examples
Tile vector-add — view-based (preferred for reuse):
```python
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    a_view = a.tiled_view((TILE,))
    b_view = b.tiled_view((TILE,))
    c_view = c.tiled_view((TILE,))
    bid = ct.bid(0)
    a_tile = a_view.load((bid,))
    b_tile = b_view.load((bid,))
    c_view.store((bid,), a_tile + b_tile)
```
One-call form (concise for a single load/store):
```python
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))
    ct.store(c, index=(bid,), tile=a_tile + b_tile)
```
- **What it demonstrates**: the two equivalent tile-access styles and where the tile shape lives (bound to a view vs supplied inline).

## Reference Tables
| Concept | C++ | Python |
|---|---|---|
| Tile kernel | `__tile_global__` | `@ct.kernel` |
| Block tile index | `ct::bid()` | `ct.bid(axis)` |
| Compile-time const | `8_ic` literal | `ct.Constant[int]` |
| View access | `partition_view` `.load/.store` | `tiled_view` `.load/.store` |
| One-call access | — | `ct.load` / `ct.store` |
| Edge handling | `load_masked` / `store_masked` | `PaddingMode.ZERO` |

## Key Takeaways
1. The CUDA programming model is identical in Python and C++.
2. Tile kernels use `@ct.kernel`; index with `ct.bid(axis)`.
3. Prefer `tiled_view` when reusing a partitioning; use one-call `ct.load/ct.store` for one-offs.
4. Mark per-kernel compile-time constants with `ct.Constant[int]`.
5. Python promotes narrowing scalar-tile ops; C++ rejects them — keep literals in the tile's element type.

## Connects To
- **Ch 6 (Tile programming)**: the model and concepts these Python APIs implement.
- **Ch 1 / Ch 2**: SIMT model shared with C++.
- **Ch 3 (Memory)**: host↔device transfers underlie Python `ndarray` usage.
