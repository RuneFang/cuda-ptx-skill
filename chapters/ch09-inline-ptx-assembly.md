# Chapter 9: Inline PTX Assembly (`asm()`)

## Core Idea
`asm()` statements inject arbitrary PTX into CUDA C++ when you need an instruction the language doesn't expose. The syntax mirrors GNU inline asm: `asm("template" : outputs : inputs : clobbers);` with `%0,%1,...` referencing operands and constraint letters naming register types.

## Frameworks Introduced
- **Basic asm**: `asm("membar.gl;");` inserts that PTX at this point.
- **Operand syntax**: `asm("template-string" : "constraint"(output) : "constraint"(input));`
  - `%n` indexes operands in text order; **outputs are listed first** → get smallest indices. References may be reordered or repeated.
  - Drop the final colon if no inputs; adjacent colons if no outputs.
  - Escape a literal `%` in PTX as `%%` (e.g. `%%clock`).
- **Modifiers**: `=` = write-only output, `+` = read-write output (use `+` when the output is *conditionally* updated — there's an implicit read).
- **Multi-instruction asm**: separate PTX instructions with `;`; split across lines via C string concatenation; terminate each (except last) with `\n\t` for readable PTX.
- **`volatile`**: prevents the compiler from deleting/moving the asm (it assumes asm has no side effects beyond outputs).
- **`memory` clobber** (3rd colon): stops memory optimizations around the asm for hidden reads/writes.

## Key Concepts
- **constraint letters** (register types): `"h"`=.u16, `"r"`=.u32, `"l"`=.u64, `"q"`=.u128 (only where `__int128` supported), `"f"`=.f32, `"d"`=.f64.
- **`"n"`**: immediate integer operand with known value.
- **`"C"`**: array-of-`const char` whose contents are known at compile time — used to splice compile-time strings into PTX (e.g. rounding-mode suffixes). Device code only; no modifiers allowed.
- **clobbers**: `"memory"` tells the compiler memory may change unexpectedly.

## Code Examples
Compute `x*x*x` with a scoped temp register (avoids duplicate-definition errors when inlined):
```cpp
__device__ int cube(int x) {
    int y;
    asm("{\n\t"                       // braces give local scope per inlining
        " .reg .u32 t1;\n\t"          // temp reg
        " mul.lo.u32 t1, %1, %1;\n\t" // t1 = x*x
        " mul.lo.u32 %0, t1, %1;\n\t" // y  = t1*x
        "}"
        : "=r"(y)                     // output (write-only)
        : "r"(x));                    // input
    return y;
}
```
Conditional update needs `+` (implicit read of the output):
```cpp
__device__ int cond(int x) {
    int y = 0;
    asm("{\n\t .reg .pred %p;\n\t"
        " setp.eq.s32 %p, %1, 34;\n\t"   // x == 34 ?
        " @%p mov.s32 %0, 1;\n\t"        // y = 1 if true
        "}"
        : "+r"(y) : "r"(x));             // y is read-modify-write
    return y;
}
```
Reading a special register safely:
```cpp
asm volatile("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
```
- **What it demonstrates**: local-scope braces, `=` vs `+`, predicate use, `volatile` + `memory` clobber for a side-effecting read.

## Reference Tables
| Constraint | PTX register / meaning |
|---|---|
| `"h"` | .u16 reg |
| `"r"` | .u32 reg |
| `"l"` | .u64 reg |
| `"q"` | .u128 reg (only with `__int128`) |
| `"f"` | .f32 reg |
| `"d"` | .f64 reg |
| `"n"` | immediate int (known value) |
| `"C"` | compile-time `const char[]` spliced into PTX |

## Anti-patterns (from "Pitfalls")
- **Namespace conflicts**: a `.reg` declared in an inlined `asm` collides across inlinings. Fix: scope it in `{}` or don't inline.
- **Memory-space confusion**: asm can't know a register's memory space; for sm_20+, pointer args are passed as **generic** addresses — use the right PTX instruction yourself.
- **Incorrect optimization**: without `volatile`, the compiler may move/delete asm; without a `memory` clobber it may reorder memory ops around hidden reads/writes.
- **Incorrect PTX**: the front end does **not** validate the template string — errors only surface at `ptxas`. Operand modifiers like `%n1` aren't supported and pass through to `ptxas`, causing undefined behavior.

## Key Takeaways
1. `asm("tmpl" : outs : ins : clobbers)`; outputs first, `%0,%1,...` in text order.
2. `=` write-only, `+` read-write (use for conditional outputs).
3. Pick the constraint letter matching the PTX register width/type.
4. Use `volatile` and a `memory` clobber for side-effecting / hidden-memory asm.
5. Template strings aren't parsed by the front end — bugs appear at `ptxas`.

## Connects To
- **Ch 10 (PTX ABI)**: how PTX-level code interoperates with CUDA C++ (calling convention, ABI).
- **Ch 2**: asm lives inside `__device__`/`__global__` C++ functions.
- **Ch 12 (C++ extensions)**: prefer intrinsics where available before dropping to asm.
