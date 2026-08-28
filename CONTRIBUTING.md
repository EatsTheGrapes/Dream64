# Contributing to Dream64

Thanks for helping build Dream64.

## Where should I work?

Start with `docs/ARCHITECTURE.md` and identify the narrowest crate/module that owns the behavior you want to change. Avoid making unrelated changes in `dm-lifecycle/src/lib.rs` or another orchestration module when a lower-level subsystem owns the behavior.

## Tests

Add or update tests with the change.

- Use `crates/<crate>/tests/` for public-API and subsystem integration tests.
- Keep unit tests beside implementation when they need private functions or internal state.
- Give each integration test file a focused subject rather than a catch-all suite.
- Prefer regression tests for compatibility behavior before refactoring it.

For example, lifecycle artifact/catalog behavior belongs in `crates/dm-lifecycle/tests/`, while VM execution behavior belongs with `dm-vm`.

## Refactoring

Dream64 is actively moving from rapid implementation toward long-term maintainability. Refactors should preserve behavior and be split into reviewable steps.

A good refactor:

1. Adds/updates regression coverage.
2. Establishes a clear module boundary.
3. Moves implementation without changing semantics.
4. Builds and tests the affected workspace crates.
5. Keeps the resulting public API understandable.

Avoid a mechanical "split the file every N lines" approach. A module should represent a coherent responsibility.

## Pull requests

Keep PRs focused. Explain:

- what subsystem changed;
- why the boundary is correct;
- what tests cover the behavior;
- whether public API or compatibility behavior changed.

If a change is part of a larger architectural refactor, call that out so reviewers can evaluate it as an intentional migration rather than unrelated churn.
