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

## 📈 关注数曲线 / Star History

> Stars over time since the first public release. *(Sample data — replace with live values once published. The animated dot marks the latest point.)*

<p align="center">

<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" role="img" aria-label="Star history curve">
  <!-- card background -->
  <rect x="0" y="0" width="680" height="340" rx="14" fill="#ffffff" stroke="#e4e8ec"/>
  <!-- title -->
  <text x="32" y="40" font-size="17" font-weight="700" fill="#1f2328">⭐ Star History — cudaskill</text>
  <text x="32" y="60" font-size="12" fill="#6b7280">GitHub stars over the first 12 months</text>

  <!-- plot frame -->
  <!-- y gridlines + labels (0,200,400,600,800,1000) plot area: x 70..640, y 90..280 -->
  <g stroke="#eef1f4" stroke-width="1">
    <line x1="70" y1="280" x2="640" y2="280"/>
    <line x1="70" y1="242" x2="640" y2="242"/>
    <line x1="70" y1="204" x2="640" y2="204"/>
    <line x1="70" y1="166" x2="640" y2="166"/>
    <line x1="70" y1="128" x2="640" y2="128"/>
    <line x1="70" y1="90"  x2="640" y2="90"/>
  </g>
  <g font-size="11" fill="#9aa4ae" text-anchor="end">
    <text x="60" y="284">0</text>
    <text x="60" y="246">200</text>
    <text x="60" y="208">400</text>
    <text x="60" y="170">600</text>
    <text x="60" y="132">800</text>
    <text x="60" y="94">1000</text>
  </g>

  <!-- axes -->
  <line x1="70" y1="90" x2="70" y2="280" stroke="#cbd2d9" stroke-width="1.5"/>
  <line x1="70" y1="280" x2="640" y2="280" stroke="#cbd2d9" stroke-width="1.5"/>

  <!-- x labels (months) -->
  <g font-size="11" fill="#9aa4ae" text-anchor="middle">
    <text x="70"  y="300">M1</text>
    <text x="122" y="300">M2</text>
    <text x="174" y="300">M3</text>
    <text x="226" y="300">M4</text>
    <text x="278" y="300">M5</text>
    <text x="330" y="300">M6</text>
    <text x="382" y="300">M7</text>
    <text x="434" y="300">M8</text>
    <text x="486" y="300">M9</text>
    <text x="538" y="300">M10</text>
    <text x="590" y="300">M11</text>
    <text x="640" y="300">M12</text>
  </g>

  <!-- area fill under curve -->
  <!-- data points (x, y) where y = 280 - stars*0.19 ; stars: 8,25,55,110,180,280,400,520,640,760,880,980 -->
  <defs>
    <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#76B900" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#76B900" stop-opacity="0.02"/>
    </linearGradient>
  </defs>
  <path d="M70,278 L122,275 L174,270 L226,259 L278,246 L330,227 L382,204 L434,181 L486,158 L538,136 L590,113 L640,94 L640,280 L70,280 Z"
        fill="url(#grad)"/>

  <!-- the curve -->
  <path d="M70,278 L122,275 L174,270 L226,259 L278,246 L330,227 L382,204 L434,181 L486,158 L538,136 L590,113 L640,94"
        fill="none" stroke="#76B900" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>

  <!-- data dots -->
  <g fill="#5a8f00">
    <circle cx="70"  cy="278" r="3"/>
    <circle cx="122" cy="275" r="3"/>
    <circle cx="174" cy="270" r="3"/>
    <circle cx="226" cy="259" r="3"/>
    <circle cx="278" cy="246" r="3"/>
    <circle cx="330" cy="227" r="3"/>
    <circle cx="382" cy="204" r="3"/>
    <circle cx="434" cy="181" r="3"/>
    <circle cx="486" cy="158" r="3"/>
    <circle cx="538" cy="136" r="3"/>
    <circle cx="590" cy="113" r="3"/>
  </g>

  <!-- latest point: pulsing dot + label -->
  <circle cx="640" cy="94" r="9" fill="#76B900" opacity="0.25">
    <animate attributeName="r" values="6;13;6" dur="1.8s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.35;0;0.35" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <circle cx="640" cy="94" r="5" fill="#76B900" stroke="#ffffff" stroke-width="2"/>
  <g>
    <rect x="556" y="64" width="78" height="22" rx="11" fill="#1f2328"/>
    <text x="595" y="79" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">980 ★</text>
  </g>
</svg>

</p>

To wire up a **live** star curve once the repo is public, replace the SVG above with:

```markdown
[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USER/cudaskill&type=Date)](https://star-history.com/#YOUR_USER/cudaskill&Date)
```

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

---

## 💬 Usage

Once installed, just talk to your agent:

## 📄 License & attribution

- Skill structure & synthesis: **MIT**.
- Underlying technical content is derived from **NVIDIA CUDA documentation** (© NVIDIA Corporation). This is an unofficial, educational distillation — consult the [official CUDA docs](https://docs.nvidia.com/cuda/) for authoritative, exhaustive references.

---

<div align="center">
<sub>Built with the book-to-skill converter · CUDA 13.3 · 2026-06-22</sub>
</div>
