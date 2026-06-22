# Chapter 11: C/C++ Language Extensions

## Core Idea
CUDA C++ adds **execution-space** and **memory-space** specifiers, built-in variables, intrinsics, and `__CUDA_ARCH__`-based conditional compilation on top of standard C++. These annotations tell NVCC *where* a function runs and *where* a variable lives.

## Frameworks Introduced
- **Execution space specifiers**: `__host__`, `__device__`, `__global__`, `__tile__`, `__tile_global__` — declare host / SIMT / tile context. Combine (e.g. `__host__ __device__`) to compile for multiple contexts.
- **Memory space specifiers**: `__device__`, `__constant__`, `__managed__`, `__shared__`, `__tile__` — declare a variable's storage location and lifetime.
- **`__CUDA_ARCH__` conditional compilation**: differentiate device vs host code paths in a `__host__ __device__` function; also encodes the target compute capability in device code.
- **Host access to device symbols**: `cudaGetSymbolAddress`, `cudaGetSymbolSize`, `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol` for `__device__`/`__constant__`/`__tile__` variables.

## Key Concepts
- **`__global__` / `__tile_global__` constraints**: must return `void`; can't be a class/struct/union member; require an execution configuration; no recursion; calls are **asynchronous**.
- **`__constant__`**: device-readable, read-only; modifiable only from host via the runtime API.
- **`__managed__`**: unified memory variable (host + device, automatic migration).
- **`__shared__`**: per-block on-SM storage, block lifetime.
- **built-in variables**: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize`.

## Code Examples
Dual-context function with arch-specific paths:
```cpp
__host__ __device__ void func() {
#if defined(__CUDA_ARCH__)
    // device code path (and __CUDA_ARCH__ == target compute capability * 10)
#else
    // host code path
#endif
}
```
Host ↔ device symbol transfer:
```cpp
__device__   float device_var   = 4.0f;
__constant__ float constant_var = 4.0f;     // read-only on device

float *device_ptr;
cudaGetSymbolAddress((void**)&device_ptr, device_var);
size_t sz; cudaGetSymbolSize(&sz, device_var);     // 4 bytes
float host_var;
cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // D->H
host_var = 3.0f;
cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // H->D
```
- **What it demonstrates**: `__CUDA_ARCH__` branching; the four symbol APIs for moving data to/from named device/constant variables.

## Reference Tables
Execution space — executed in / callable from:
| Specifier | Executed in | Callable from |
|---|---|---|
| `__host__` (or none) | Host | Host |
| `__device__` | SIMT | SIMT |
| `__global__` | SIMT (kernel) | Host & SIMT |
| `__tile__` | Tile | Tile |
| `__tile_global__` | Tile (kernel) | Host |
| `__host__ __device__ __tile__` | all | all |

Memory space:
| Specifier | Location | Accessible by | Lifetime |
|---|---|---|---|
| `__device__` | device global | grid threads / runtime API | program |
| `__constant__` | constant (RO) | grid threads / runtime API | program |
| `__managed__` | host+device (auto) | host+device threads | program |
| `__shared__` | on-SM | block threads | block |
| (none) | registers | single thread | thread |

## Key Takeaways
1. Execution specifiers say *where code runs*; memory specifiers say *where data lives*.
2. `__global__`/`__tile_global__`: void return, no recursion, async, need a launch config.
3. Use `__CUDA_ARCH__` to split host/device paths in shared functions.
4. `__constant__` is host-writable / device-read-only.
5. Move data to named device symbols with `cudaMemcpyTo/FromSymbol`.

## Connects To
- **Ch 2**: `__global__` kernels and built-in index variables.
- **Ch 3 / Ch 5**: `__managed__` (unified) and `__shared__` storage.
- **Ch 12**: `__CUDA_ARCH__` ties into compute-capability compiler targets.
