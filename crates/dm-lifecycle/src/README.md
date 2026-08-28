# Lifecycle implementation boundaries

This directory is intentionally organized by responsibility. Keep orchestration in `lib.rs` small; implementation that can be reasoned about independently belongs in a dedicated module.

## Extraction order

1. Artifact/catalog encoding and decoding
2. DMM discovery/parsing/cache construction
3. Lifecycle indexing and planning
4. Scheduler/execution
5. Readiness/diagnostics
6. IPC integration

Each extraction should preserve the public API until the migration is complete and should add or retain regression coverage before moving behavior.