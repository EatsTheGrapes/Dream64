# Dream64 engine

This directory is the clean-room implementation workspace for a 64-bit,
multicore-capable engine that accepts Dream Maker source code. The immediate
target is Monkestation's pinned BYOND 516.1663 behavior.

The project does not decode BYOND bytecode, copy proprietary implementation
details, or promise compatibility from syntax acceptance alone. Compatibility
is measured with public documentation and black-box fixtures that run through
both Dream Maker/Dream Daemon and this engine.

## Current milestone

- A 64-bit-only core with an explicit 32-bit DM number representation.
- A loss-aware lexer for DM strings, raw text, line continuations,
	indentation, and exact source spans.
- A deterministic project loader that follows active includes, shared defines,
	nested conditionals, and the target compiler-version macros.
- A declaration parser that indexes paths, types, vars, procs, verbs,
	parameters, overrides, and opaque procedure bodies across the full active
	Monkestation source graph.
- A reusable compiler database that retains syntax by file identity, builds a
	project-wide object tree, and maps frontend diagnostics back to source spans.
- Portable stack bytecode and a deterministic reference interpreter supporting
	explicit call frames, forward and recursive procedure calls, permissive DM
	argument arity, parameters, locals, assignment, arithmetic, comparisons, and
	nested indentation-based `if`/`else` control flow.
- A bounded reference-compiler runner for differential fixtures.
- A written compatibility boundary and runtime architecture.

Run the checks from this directory once Rust is installed:

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Probe a fixture with a known-good Dream Maker executable:

```powershell
cargo run -p dm-conformance -- probe `
	--compiler "C:\path\to\byond\bin\dm.exe" `
	--project fixtures\compiler\basic\basic.dme
```

The temporary probe output is placed outside the fixture and removed after the
compiler exits. A timeout is treated as a distinct result rather than a failed
compatibility assertion.

Inspect the declaration tree produced for a DM source file:

```powershell
cargo run -p dm-conformance -- syntax path\to\source.dm
```

Inspect the deterministic file graph for a complete Dream Maker project:

```powershell
cargo run -p dm-conformance -- project path\to\world.dme
```

Load that graph and run declaration parsing across every DM source file:

```powershell
cargo run -p dm-conformance -- check path\to\world.dme
```

Build the combined frontend snapshot and object tree:

```powershell
cargo run -p dm-conformance -- frontend path\to\world.dme
```

Compile and execute a supported procedure through the reference interpreter:

```powershell
cargo run -p dm-conformance -- execute `
	fixtures\runtime\arithmetic.dm /proc/arithmetic_probe 4
```

Measure current bytecode-lowering coverage across a complete project:

```powershell
cargo run -p dm-conformance -- compile-check path\to\world.dme
```
