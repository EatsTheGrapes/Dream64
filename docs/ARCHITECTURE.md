# Dream64 Architecture

Dream64 is organized as a Cargo workspace. The architectural goal is **local reasoning**: a contributor should be able to change one subsystem without understanding the entire engine.

## Dependency direction

The intended direction is from low-level representation toward orchestration:

```text
lexer / syntax / core / value
            ↓
        semantics
            ↓
        compiler / lowering
            ↓
       runtime / VM
            ↓
 lifecycle / world / project orchestration
            ↓
       binaries / clients
```

A higher-level crate may depend on lower-level crates, but lower-level crates should not reach upward into lifecycle orchestration.

## Module responsibilities

- `dm-core`: shared identifiers, spans, and fundamental compiler types.
- `dm-lexer` / `dm-syntax`: source lexing and syntax representation.
- `dm-semantics`: semantic/type/procedure information.
- `dm-compiler` / `dm-lowering`: compilation and executable representation.
- `dm-value`: runtime values and value serialization.
- `dm-runtime`: runtime object/state representation.
- `dm-vm`: execution engine and scheduler primitives.
- `dm-map`: DMM parsing and map representation.
- `dm-world`: world planning/materialization.
- `dm-lifecycle`: orchestration that connects compilation, runtime, VM, and world initialization.
- `dm-client`: presentation/client-side behavior.
- `dm-conformance`: compatibility/conformance tooling.

## Lifecycle crate rule

`dm-lifecycle` is an orchestration crate, not a dumping ground. Its public API should remain small and its implementation should be divided by responsibility:

- lifecycle resolution/indexing
- initialization planning
- execution/scheduling
- map parsing/cache products
- artifact serialization
- readiness/diagnostics
- IPC

`artifact.rs` and `ipc.rs` are already separate modules. New work should follow the same boundary instead of adding unrelated functionality to `src/lib.rs`.

## Tests

Tests that exercise a public crate API should prefer `crates/<crate>/tests/` integration tests when they do not need private implementation details. Unit tests that specifically validate private helpers may remain beside the implementation.

The target is not to eliminate every unit test. The target is to make the test suite communicate the architecture:

```text
crates/dm-lifecycle/tests/
    portable_catalogs.rs
    lifecycle_index.rs
    initialization.rs
    scheduler.rs
    maps.rs
```

Each file should have a narrow reason to exist.

## Refactoring rules

1. Preserve behavior unless a change explicitly says otherwise.
2. Establish regression coverage before moving behavior across module boundaries.
3. Prefer small, reviewable extraction commits.
4. Keep public APIs stable while moving private implementation details.
5. Do not introduce a module solely to shorten `lib.rs`; every module needs a coherent responsibility.
6. Avoid circular dependencies and cross-subsystem reach-through.
7. Keep compatibility quirks documented at the boundary where they are required.

## Long-term target

No production source file should require whole-engine knowledge for routine maintenance. Large modules should be decomposed when they contain multiple independently testable responsibilities, with `lib.rs` acting primarily as a public API/module facade.
