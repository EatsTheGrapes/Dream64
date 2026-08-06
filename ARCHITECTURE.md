# Architecture

## Compatibility contract

The engine targets source and observable behavior compatibility, not BYOND's
private `.dmb` format or undocumented process internals. Existing games are
recompiled from their `.dme`, `.dm`, `.dmm`, `.dmf`, and resource files.

Compatibility is tracked in independently testable tiers:

1. Preprocessor and lexer parity.
2. Parser, object tree, static checks, and diagnostic parity.
3. Proc execution, values, lists, inheritance, scheduling, and runtime-error
	behavior.
4. Map, atom, appearance, savefile, database, and external-call behavior.
5. Headless Monkestation boot and round execution.
6. Client rendering, input, audio, browser UI, and networking.

Passing a lower tier never implies a higher tier.

## Compiler pipeline

```text
DM source -> preprocessor -> loss-aware lexer -> syntax tree -> object tree
          -> typed semantic IR -> portable bytecode -> interpreter
                                             `-> optimized IR -> JIT/AOT
```

The interpreter is the semantic reference and debugging tier. Optimized tiers
must pass the same fixture corpus and may deoptimize to the interpreter at
observable boundaries.

Project compilation preserves textual `#include` splice order while retaining
each declaration's original file identity and byte span. The compiler database
owns parsed syntax snapshots, diagnostics, the canonical type/member tree, and
semantic procedure registries. Tree-local numeric IDs are never persisted as
cross-build identities; durable tooling keys use canonical paths and source
locations.

The object tree starts from the DM standard roots (`/datum`, `/atom`,
`/atom/movable`, `/area`, `/turf`, `/obj`, and `/mob`), applies source-ordered
`parent_type` assignments, rejects invalid inheritance cycles, and only then
derives child and inherited-member links.

## Runtime model

Ordinary DM procs retain deterministic, cooperative semantics. The coordinator
owns mutable world state and advances fibers in a reproducible order. Native
systems receive immutable snapshots or isolated partitions and return command
buffers. Buffers are validated and committed in a stable order.

This gives the engine useful multicore work without making existing DM code
racy:

- visibility and spatial queries;
- appearance derivation and resource packing;
- networking encode/decode and compression;
- database and filesystem completion;
- pathfinding and explicitly pure jobs;
- garbage collection and profiling work where safe.

Parallel jobs cannot mutate live datums directly. An opt-in future API may
accept pure data and return values, with determinism verified in tests.

## Procedure frames

Each running procedure has an explicit frame containing its stable procedure
identity, instruction pointer, local slots, operand-stack base, complete passed
argument vector, `src`, `usr`, return value, and caller link. Keeping these
fields explicit supports deterministic scheduling, debugger inspection,
replay, and useful runtime stack traces without relying on the native machine
stack.

Compatibility binding preserves the distinction between declared parameters
and passed arguments. Missing parameters receive their declared default or
`null`; extra positional arguments remain available through `args`. Calls to
the current proc and parent proc may reuse the current argument vector. Named
arguments, `arglist()`, override chains, and `..()` are resolved before the
optimized tiers so the interpreter remains the reference behavior.

A configurable frame limit produces a deterministic runtime error instead of
overflowing the host stack. Sleeping or yielding suspends a frame inside its
fiber; it never blocks an operating-system worker thread.

A separate shared instruction budget is charged before every interpreted
bytecode operation across the complete call stack. Exhaustion is source-mapped
and deterministic, which prevents infinite loops from hanging tooling or a
server tick while preserving a configurable production ceiling.

## Values and memory

The host process is 64-bit. Object handles and internal indices are not limited
to BYOND's 32-bit process address space. DM `num` remains IEEE-754 binary32
unless a conformance fixture proves a different observable rule. This preserves
existing rounding, equality, serialization, and bitwise expectations while the
engine can expose opt-in typed numeric buffers for high-throughput systems.

Managed handles, rather than raw pointers, cross the VM boundary. This supports
moving collection, snapshot/replay, safe extension calls, and stale-reference
detection.

## Extension boundary

The preferred extension interface is versioned and data-oriented. WebAssembly
is the portable sandboxed tier; a native ABI is permitted for trusted server
extensions. BYONDAPI source compatibility can be provided by an adapter where
its public contract is sufficient, but binary compatibility with proprietary
32-bit DLL assumptions is not a goal.

## User interface layers

The engine has three separate UI products sharing one protocol and inspection
model:

- the game client used by players;
- a Dream Maker-style development application;
- an optional server administration dashboard over the headless runtime.

The player client preserves BYOND skin behavior without inheriting its old
rendering implementation. A DMF parser builds a typed control tree for windows,
panes, menus, macros, map surfaces, browser surfaces, input, output, tabs,
grids, and buttons. `winset()`, `winget()`, `winclone()`, `winshow()`,
`winexists()`, `output()`, and client commands operate on that tree through a
versioned UI protocol.

Map and appearance rendering use the native GPU renderer. Browser controls use
a modern Chromium-compatible webview with a compatibility bridge injected as
`window.Byond`. Existing TGUI code must be able to call `Byond.winset()`,
`Byond.winget()`, `Byond.command()`, and topic/output operations without a
game-specific port.

The development application is not coupled to the runtime process. It consumes
the compiler's syntax/object trees, diagnostics, source maps, debugger events,
and profiler stream. This supports both a first-party Dream Maker replacement
and standard editor integrations through LSP and DAP.
