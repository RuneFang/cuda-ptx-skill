# Chapter 15: Installation & Toolkit Quick Start

## Core Idea
Installing CUDA means installing a compatible **NVIDIA driver** plus the **CUDA Toolkit** (nvcc, libraries, tools), via a platform-appropriate installer. The Quick Start Guide enumerates per-OS paths; pick the one matching your distro and packaging preference.

## Frameworks Introduced
- **Installer types**:
  - **Network installer** / **Local installer** (Windows & Linux): package-manager repo vs self-contained download.
  - **RPM installer** vs **Runfile installer** (RHEL/CentOS, Fedora, SUSE, OpenSUSE, Amazon Linux): distro package vs standalone runfile.
  - **Debian installer** vs **Runfile** (Ubuntu, Debian).
  - **Pip wheels** and **Conda** (Windows & Linux): Python-ecosystem installs via **metapackages**.
  - **WSL**: special path for CUDA on Windows Subsystem for Linux.
- **Metapackages**: convenience packages that pull in the right set of CUDA components for pip/conda installs.

## Key Concepts
- **NVIDIA driver vs CUDA Toolkit**: driver runs the GPU; toolkit provides the SDK (compiler, libs, headers). Both must be compatible (see Ch 12 for driver/toolkit versioning).
- **`nvidia-smi`**: driver tool; verifies the GPU is visible and reports driver/compute info.
- **`libcuda.so` symbolic link**: on some Linux setups you must add it manually.
- **GPG key / repo metadata**: package-manager installs require importing NVIDIA's repo key and metadata first.

## Code Examples
Representative Linux package-manager flow (Ubuntu/Debian style) — verify after install:
```bash
# (after adding NVIDIA repo metadata + GPG key per the guide)
sudo apt-get update
sudo apt-get install cuda            # installs toolkit + driver metapackage
# verify
nvidia-smi                            # driver + GPU visible
nvcc --version                        # toolkit/compiler version
```
Conda / pip (Python ecosystem):
```bash
conda install -c nvidia cuda          # conda metapackage
# or
pip install <cuda metapackage wheels> # per Quick Start "Pip Wheels" section
```
- **What it demonstrates**: the install-then-verify pattern; `nvidia-smi` checks the driver, `nvcc --version` checks the toolkit.

## Reference Tables
| Platform | Common install methods |
|---|---|
| Windows | Network/Local installer, Pip wheels, Conda |
| RHEL/CentOS/Fedora/SUSE/OpenSUSE/Amazon Linux | RPM installer, Runfile installer |
| Ubuntu/Debian | Debian installer, Runfile installer |
| WSL | dedicated WSL repo path |
| Python any | Pip wheels, Conda (metapackages) |

## Key Takeaways
1. You need a compatible **driver** + **CUDA Toolkit**; both versions must align.
2. Choose installer by OS and packaging preference (repo vs runfile vs pip/conda).
3. Package-manager installs require importing NVIDIA's GPG key and repo metadata first.
4. Verify with `nvidia-smi` (driver) and `nvcc --version` (toolkit).
5. On some Linux systems add the `libcuda.so` symlink and reboot after install.

## Connects To
- **Ch 12**: driver/toolkit & compute-capability compatibility.
- **Ch 14 (NVCC)**: the compiler you just installed.
- **Ch 16 (Release features)**: which CUDA version introduced/removed what.
