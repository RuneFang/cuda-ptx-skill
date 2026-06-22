# Chapter 12: Compute Capability, Feature Sets & Compiler Targets

## Core Idea
A device's **compute capability** (major.minor, e.g. 9.0) defines its features and limits. Features split into **baseline** (forward-available), **architecture-specific**, and **family-specific** sets — each requiring the matching compiler target, with different portability guarantees.

## Frameworks Introduced
- **Compute capability**: the versioned feature/spec level of an SM. All NVIDIA GPUs are little-endian. Query it three ways:
  - `nvidia-smi --query-gpu=name,compute_cap`
  - Runtime: `cudaDeviceGetAttribute(..., cudaDevAttrComputeCapabilityMajor/Minor, dev)`
  - Driver: `cuDeviceGetAttribute(..., CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR, dev)`; NVML: `nvmlDeviceGetCudaComputeCapability` (link `-lnvidia-ml`).
- **Three feature sets / compiler targets**:
  - **Baseline**: features introduced to be available on *all* subsequent architectures (the default forward-compatible set).
  - **Architecture-specific** (CC ≥ 9.0): specialized features (e.g. some Tensor Core ops) *not* guaranteed on later architectures. Requires an arch-specific target; **runs only on the exact compute capability compiled for**.
  - **Family-specific** (CC ≥ 10.0): arch-specific features shared by a *family* of compute capabilities; guaranteed across the family. Requires a family-specific target; runs only on family members.
- **Binary vs PTX compatibility** (from Ch 1): **cubins** are arch-specific binaries; **PTX** is forward-portable IR; **fatbins** bundle multiple; **JIT** compiles PTX at load for newer GPUs.

## Key Concepts
- **compute capability (major.minor)**: feature/spec level (a.k.a. SM version).
- **baseline / architecture-specific / family-specific features**: portability tiers.
- **cubin / PTX / fatbin**: arch binary / portable IR / bundle.
- **JIT compilation**: driver compiles PTX to SASS at load time.
- **`__CUDA_ARCH__`**: in device code, equals the target compute capability × 10.

## Code Examples
Query compute capability at runtime:
```cpp
#include <cuda_runtime_api.h>
int major, minor;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
// e.g. major=9, minor=0  -> compute capability 9.0
```
- **What it demonstrates**: programmatic capability detection to gate optional features at runtime.

## Reference Tables
| Feature set | Introduced | Portability | Compiler target |
|---|---|---|---|
| Baseline | any | available on all later archs | default |
| Architecture-specific | CC ≥ 9.0 | only the exact CC compiled for | arch-specific |
| Family-specific | CC ≥ 10.0 | all members of the family | family-specific |

| Code form | Portable? | Notes |
|---|---|---|
| cubin | no | exact-arch binary |
| PTX | yes (forward) | JIT-compiled at load on newer GPUs |
| fatbin | depends | bundles cubins + PTX |

## Key Takeaways
1. Compute capability = feature/limit level; detect via `nvidia-smi`, runtime, driver, or NVML.
2. Baseline features are forward-available; arch-specific (CC≥9.0) and family-specific (CC≥10.0) are not portable across all later GPUs.
3. Match the compiler target to the feature set you use, or the binary won't run.
4. Ship PTX (or fatbins with PTX) for forward portability via JIT.
5. `__CUDA_ARCH__` = target CC × 10 inside device code.

## Connects To
- **Ch 1**: cubins/fatbins/PTX & JIT compatibility.
- **Ch 11**: `__CUDA_ARCH__` conditional compilation.
- **Ch 14 (NVCC)**: how to specify targets at compile time.
