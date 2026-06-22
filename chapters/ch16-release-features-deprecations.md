# Chapter 16: Release Features, Deprecations & Known Issues

## Core Idea
The CUDA Features Archive tracks, per release (13.x back through 11.x), what's **new**, **resolved**, **known-broken**, and **deprecated/dropped** across General CUDA, the CUDA Compiler, and Developer Tools. Use it to gate feature usage and plan migrations.

## Frameworks Introduced
- **Per-release feature taxonomy** (each version section, e.g. "CUDA 13.1", "CUDA 13.0"):
  - **Features** — new capabilities (General CUDA / CUDA Compiler / CUDA Developer Tools).
  - **Resolved Issues** — bugs fixed (often CUDA Compiler).
  - **Known Issues** — current limitations to design around.
  - **Deprecated or Dropped Features** — APIs/architectures/OSes/toolchains being removed.
- **Deprecation categories**: General CUDA APIs/fields, **Deprecated Architectures**, **Deprecated/Dropped Operating Systems**, **Deprecated/Dropped CUDA Toolchains**, and CUDA Tools.
- **Host-deprecation suppression**: the macro pair `__NV_SILENCE_HOST_DEPRECATION_BEGIN` / `__NV_SILENCE_HOST_DEPRECATION_END` locally silences deprecation warnings.
- **Removed-fields migration**: tables of "Removed Fields and Their Replacements" and "Removed `cudaDeviceAttr` Types (No Replacement Available)".

## Key Concepts
- **deprecated vs dropped**: still present but discouraged vs removed entirely.
- **`cudaDeviceAttr`**: device attribute enum; some entries removed without replacement.
- **`__NV_SILENCE_HOST_DEPRECATION_BEGIN/END`**: scoped warning suppression.
- **release cadence**: 13.3 → 13.1 → 13.0 → 12.9 → 12.8 → 12.6 → 12.5 → 12.4 → 12.3 → 12.2 → 12.1 → 12.0 → 11.8 → 11.7 → 11.6 (sections in the archive).

## Code Examples
Suppress a host deprecation warning around a call you must keep temporarily:
```cpp
__NV_SILENCE_HOST_DEPRECATION_BEGIN
legacyDeprecatedApi(...);   // known-deprecated; migration tracked separately
__NV_SILENCE_HOST_DEPRECATION_END
```
- **What it demonstrates**: the sanctioned way to quiet a specific deprecation without globally disabling warnings.

## Reference Tables
| Section type | Tells you | Action |
|---|---|---|
| Features | what's newly available | gate on CUDA version / compute capability |
| Resolved Issues | what got fixed | drop local workarounds |
| Known Issues | current limitations | design around / pin versions |
| Deprecated/Dropped | what's going away | migrate before upgrade |
| Removed Fields → Replacements | renamed/replaced APIs | mechanical migration |
| Removed `cudaDeviceAttr` (no replacement) | gone for good | redesign feature detection |

## Key Takeaways
1. Treat the Features Archive as a changelog: Features / Resolved / Known / Deprecated per release.
2. "Deprecated" ≠ "dropped" — deprecated still works; dropped is removed.
3. Use `__NV_SILENCE_HOST_DEPRECATION_BEGIN/END` for scoped, temporary suppression — not as a fix.
4. Some `cudaDeviceAttr` types are removed with no replacement; re-architect detection if you used them.
5. Check Known Issues before relying on a brand-new feature; pin CUDA versions in CI.

## Connects To
- **Ch 12**: deprecated **architectures** map to compute capabilities.
- **Ch 15 (Install)**: dropped OSes/toolchains affect which installer/version you can use.
- **Ch 4 (Error checking)**: removed/renamed APIs surface as compile/link or runtime errors.
