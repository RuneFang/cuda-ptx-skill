# Chapter 14: NVCC — The CUDA Compiler

## Core Idea
`nvcc` is the offline CUDA toolchain. It **separates device code from host code**, compiles device code to PTX then to cubin (per target SM), compiles host code with the system compiler, and bundles everything — possibly multiple PTX/cubin targets — into a **fatbin**.

## Frameworks Introduced
- **NVCC compilation workflow**:
  1. Split device vs host code.
  2. GPU compiler → **PTX** (per virtual ISA, e.g. `compute_90`).
  3. **`ptxas`** → **cubin** (per hardware ISA / SM version, e.g. `sm_90`).
  4. Host compiler builds host code (a compatible host compiler is required).
  5. Bundle PTX + cubin targets into a **fatbin** so one binary supports multiple virtual & hardware ISAs.
- **Offline (nvcc) vs JIT (nvrtc)**: `nvcc` is ahead-of-time; **`nvrtc`** is the runtime compiler for online/JIT compilation.
- **File-type conventions**: `.cu` (device or mixed), `.cuh` (CUDA headers), `.c/.cpp/.cc/.cxx` (host-only), `.h/.hpp/...` (headers that may contain device code).

## Key Concepts
- **`nvcc`**: top-level driver coordinating compiler, linker, PTX/cubin assemblers.
- **`ptxas`**: assembles PTX into cubin for a target SM.
- **`nvrtc`**: runtime (JIT) compiler.
- **virtual ISA (`compute_XX`) vs real ISA (`sm_XX`)**: PTX target vs hardware target.
- **fatbin**: container bundling multiple PTX/cubin targets.
- **`-v` / `-keep` / `--keep-dir`**: show full workflow / keep intermediates / set intermediate dir.

## Code Examples
A minimal CUDA program and its build:
```cpp
// example.cu
#include <stdio.h>
__global__ void kernel() { printf("Hello from kernel\n"); }
void kernel_launcher() { kernel<<<1,1>>>(); cudaDeviceSynchronize(); }
int main() { kernel_launcher(); return 0; }
```
```bash
nvcc example.cu -o example          # basic build
nvcc -v example.cu -o example       # show full tool invocation workflow
nvcc -keep example.cu -o example    # keep intermediate PTX/cubin files
```
- **What it demonstrates**: the standard build command; `-v` reveals the device→PTX→ptxas→cubin→fatbin pipeline; `-keep` saves intermediates for inspection.

## Reference Tables
| Extension | Content |
|---|---|
| `.c` | host-only C |
| `.cpp`/`.cc`/`.cxx` | host-only C++ |
| `.cu` | device or mixed host/device |
| `.cuh` | CUDA header (device/host/mixed) |
| `.h`/`.hpp`/... | header (may contain device code) |

| Stage | Tool | Output |
|---|---|---|
| Device → IR | GPU compiler | PTX (`compute_XX`) |
| IR → binary | `ptxas` | cubin (`sm_XX`) |
| Host code | host compiler | host objects |
| Bundle | nvcc | fatbin |
| Runtime JIT | nvrtc | SASS at load |

## Key Takeaways
1. `nvcc` is offline; `nvrtc` is the JIT/runtime compiler.
2. Pipeline: device code → PTX (`compute_XX`) → `ptxas` → cubin (`sm_XX`) → fatbin.
3. Put device/mixed code in `.cu`/`.cuh`; host-only code can use the system compiler and link with nvcc objects.
4. A compatible host compiler is required (see toolkit support policy).
5. Use `-v` to inspect the workflow and `-keep` to retain intermediates.

## Connects To
- **Ch 12**: virtual vs real ISA targets and feature-set compiler targets.
- **Ch 1**: cubin/fatbin/PTX compatibility and JIT.
- **Ch 15 (Install)**: nvcc ships in the CUDA Toolkit.
