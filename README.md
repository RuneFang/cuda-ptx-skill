<div align="center">

# 🚀 cudaskill

**An agent skill distilled from the official NVIDIA CUDA documentation (CUDA 13.3)**

Turn 263K words of CUDA & PTX docs into a structured, on-demand knowledge base your AI agent can actually use.

[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/cuda/)
[![Skill](https://img.shields.io/badge/type-agent--skill-blue)]()
[![Chapters](https://img.shields.io/badge/chapters-16-orange)]()
[![Size](https://img.shields.io/badge/total-~18K%20tokens-purple)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

</div>

---

## ✨ What is this?

`cudaskill` is a **book-to-skill** knowledge pack built from five official NVIDIA CUDA documents:

- CUDA Programming Guide
- CUDA Quick Start Guide
- CUDA Features Archive
- Inline PTX Assembly
- PTX Writer's Guide to Interoperability

It extracts **structure, not summaries** — named frameworks, exact APIs, code examples, anti-patterns, and decision rules — so an AI coding agent (Claude Code, GitHub Copilot CLI, Amp, …) can apply CUDA knowledge while you work, instead of re-reading PDFs.

> **Not a book report.** Each chapter is a toolkit: "use X when Y", real kernels (vecAdd, tiled GEMM with `mma`, scoped-register inline PTX), reference tables, and pitfalls.

---

## 📦 What's inside

```
cudaskill/
├── SKILL.md          # core frameworks + chapter & topic index  (~2.3K tok)
├── glossary.md       # ~120 terms with chapter refs              (~1.8K tok)
├── patterns.md       # 12 concrete techniques                    (~1.3K tok)
├── cheatsheet.md     # decision rules, defaults, "tells & smells" (~1.0K tok)
└── chapters/         # 16 on-demand topic chapters (~800–1,200 tok each)
    ├── ch01-programming-model.md
    ├── ch02-cuda-cpp-kernels.md
    ├── ch03-memory-management.md
    ├── ch04-runtime-init-error-checking.md
    ├── ch05-simt-shared-memory.md
    ├── ch06-tile-programming.md
    ├── ch07-async-streams-events.md
    ├── ch08-advanced-features.md
    ├── ch09-inline-ptx-assembly.md
    ├── ch10-ptx-abi-interoperability.md
    ├── ch11-cpp-language-extensions.md
    ├── ch12-compute-capability-targets.md
    ├── ch13-cuda-python.md
    ├── ch14-nvcc-compilation.md
    ├── ch15-install-toolkit.md
    └── ch16-release-features-deprecations.md
```

Chapter files are loaded **on demand** — they don't cost tokens until the agent actually opens them.

---

## 🧭 Coverage map

| Area | Chapters |
|------|----------|
| **Programming model** — threads / warps / blocks / clusters / grids, SIMT | ch01 |
| **Kernels (C++ & Python)** — `__global__`, triple chevron, tile kernels | ch02, ch13 |
| **Memory** — unified vs explicit, shared memory, coalescing, pinned | ch03, ch05 |
| **Tile programming** — tiles vs arrays, matmul / `mma`, tensor cores | ch06 |
| **Async & scaling** — streams, events, CUDA Graphs, Cooperative Groups | ch07, ch08 |
| **PTX** — inline `asm()`, constraints, ABI & interoperability | ch09, ch10 |
| **Toolchain** — language extensions, compute capability, NVCC, install | ch11, ch12, ch14, ch15 |
| **Lifecycle** — runtime init, error checking, release/deprecations | ch04, ch16 |

---



---

## 🛠 Installation

Drop the `cudaskill/` folder into your agent's skills directory, then reload:

| Agent | Personal skills path | Reload |
|-------|----------------------|--------|
| **Claude Code** | `~/.claude/skills/` | restart session |
| **GitHub Copilot CLI** | `~/.copilot/skills/` | `/skills reload` |
| **Amp** | `~/.agents/skills/` | restart session |

```bash
# example: Claude Code
cp -r cudaskill ~/.claude/skills/
```

## 💬 Usage

Once installed, just talk to your agent:

## 📄 License & attribution

- Skill structure & synthesis: **MIT**.
- Underlying technical content is derived from **NVIDIA CUDA documentation** (© NVIDIA Corporation). This is an unofficial, educational distillation — consult the [official CUDA docs](https://docs.nvidia.com/cuda/) for authoritative, exhaustive references.

---

<div align="center">
<sub>Built with the book-to-skill converter · CUDA 13.3 · 2026-06-22</sub>
</div>
