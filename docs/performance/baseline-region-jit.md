# Dream64 baseline region JIT — design

The cold-boot research note (`boot-architecture-research.md`) concluded that a
five-minute Monkestation boot needs ~2.5x faster DM execution and that only a
real baseline JIT has that reach. Instruction-level quickening is now built
(PC-local sidecar, #70/#74/#75) and measured out: after field-read, nothing
incremental is worth more than ~3 s. This note designs the tier that follows —
compiling **regions** of DM bytecode to native code with Cranelift, calling a
stable Rust slow path for everything the region cannot do inline.

## Non-goals

- Not a tracing JIT. Regions are compiled from static bytecode shape, cached
  per `(module, procedure, entry PC)`, and never respecialize on values.
- Not a replacement for the reference interpreter. Every region can side-exit
  to it at any bytecode boundary, and the interpreter stays the semantic
  authority for parity tests.
- Not `unsafe` in `dm-vm`. All raw-pointer / `transmute` / FFI work lives in
  `dm-jit` (workspace lint `unsafe_code = "forbid"` applies to `dm-vm`).

## What a "region" is

A maximal run of bytecode starting at a hot entry PC that the region compiler
can lower end to end under the current supported-op set. A region:

- has one entry (the hot PC — a procedure start, or a loop header) and may have
  internal control flow (forward/backward branches within the region);
- ends at the first unsupported instruction, at `Return`, or at a branch that
  leaves the compiled range — each of which becomes a **side-exit** that
  materializes VM state and hands the exact resume PC back to the interpreter;
- charges one logical VM step per bytecode it retires and accounts the
  4096-step scheduler checkpoint exactly as the interpreter loop does;
- treats every loop backedge and every slow-path call as a **safepoint**: it
  checks the remaining step budget and can suspend with an exact resume PC.

The whole procedure is a region when every instruction is supported.

## Operand model

The interpreter frame is `locals: Vec<Value>` + `stack: Vec<Value>` + `result`.
Native code cannot hold `Value` (an enum with heap handles the GC must trace).
Two representations, chosen per operand at compile time from a simple forward
type/shape pass:

1. **Unboxed number** — a Cranelift `f64` SSA value in a register. Used while an
   operand is provably a number (constant, arithmetic result, a field guarded
   as numeric). The overwhelmingly common case in hot init math.
2. **Rooted slot** — a `u32` index into a VM-owned `SmallVec<Value>` scratch
   array passed to the region (the same mechanism as
   `dm_jit::CompiledRootedBlock`: the slot array *is* a GC root set, so a
   `Value` parked in a slot across a slow-path call stays traced). Used for
   any non-number operand and for numbers about to cross the slow-path ABI.

Locals mirror this: a `locals_kind: [Unboxed|Slot; local_count]` plan. A local
that is only ever a number lives in an `f64` stack slot; anything else is a
rooted slot. On side-exit the region writes every live operand and local back
into the interpreter frame as a real `Value` (numbers re-box trivially).

## The slow-path ABI

One `#[repr(C)]` context pointer (`*mut RegionVm`, opaque to native code) plus
the rooted slot array and its length. Every callback is
`unsafe extern "C" fn(ctx, ...) -> u64` returning a packed
`{ status, resume_pc, steps }` so native code can propagate a side-exit,
budget-exhaustion, or runtime error without unwinding across the FFI boundary.
`RegionVm` on the Rust side owns `&mut ExecutionState`, `&mut Vec<CallFrame>`,
the frame index, and the module — exactly the arguments `dispatch_instruction`
already threads.

Initial ABI surface (each maps to an existing `value_ops` / interpreter helper):

| callback | Rust target |
|---|---|
| `load_field(recv_slot, field_id) -> slot` | `datum_field_or_shared` + the #70 PC inline cache |
| `store_field(recv_slot, field_id, val_slot)` | `assign_datum_or_shared_field` |
| `load_global(global_id) -> slot` | `ExecutionState::global` |
| `list_get / list_set / list_len / list_add` | `read_list_value` / `write_list_value` / … |
| `call_static(proc_id, argc) -> side_exit` | push a `CallFrame`, side-exit so the interpreter runs the callee |
| `alloc_datum(type_id) -> slot` | `allocate_initialized_datum` — side-exits (runs `New`/`Initialize`) |
| `truthy(slot) -> i32`, `to_number(slot) -> f64`, `box_number(f64) -> slot` | canonicalisation |

`field_id` / `global_id` / `type_id` / `proc_id` are dense indices resolved
once at compile time (see "Dense IDs" below). Calls and allocations always
side-exit in v1 — the region stops, the interpreter runs the callee/constructor
and re-enters the region at the return PC if it is still hot. v2 can inline
leaf calls.

## Safepoints, budget, deopt

- The region carries a `steps_remaining: u32` in a register, decremented per
  retired bytecode. At each backedge and before each slow-path call it checks
  `steps_remaining == 0` → suspend with `{ SUSPEND, current_pc, steps_done }`.
  The interpreter re-enters the region on the next slice.
- Any guard failure (a field that was numeric is now a datum; a receiver whose
  type left the inline cache) → `{ DEOPT, current_pc, steps_done }`, materialize,
  interpreter takes over. Deopt is always safe, never a correctness event.
- A slow-path callback that returns `RUNTIME_ERROR` → the region materializes
  and returns the error to `run_frames`, which raises it exactly as an
  interpreted op would (same `execution_error` path, same source span).

## Dense IDs

Regions need `FieldId`, `GlobalId`, `ProcId`, `TypeId` as `u32` array indices,
not `FieldName` hashes. `GlobalStore` is already slot-dense internally;
`Module::procedures` is `Vec`-indexed; type intervals are dense. The missing
piece is a per-module `FieldName -> FieldId` table and a `TypePath -> TypeId`
table, both built at module load, immutable thereafter. A datum shape maps
`FieldId -> value slot`; the region's `load_field` fast path is one shape/version
compare then a slot read (this is the #70 cache, re-expressed as an integer
guard). Dynamic `vars[...]`, additions, deletions stay on the generic path and
bump the shape version.

## `PcCache` integration

New variant:

```rust
enum PcCache {
    Cold,
    FieldRead(Box<FieldSlotCache>),
    Region(Box<CompiledRegion>),   // installed at the region's entry PC
    RegionRejected,                // compiled once, unsupported — never retry
}
```

`run_frames_inner` already holds `&mut ProcedureSidecar` for the active
procedure (#75). At `instruction_index == 0` and at loop headers, if the slot is
`Cold` and the procedure has been entered enough times, attempt compilation on
a worker (JIT compile is immutable-input work — see the research note's
"parallelize only immutable preparation"); install `Region` or `RegionRejected`.
When the slot is `Region`, call it instead of dispatching.

## Milestones and gates

Every step gates on: the 650 `dm-vm` + 71 `dm-value` lib tests, then
`DREAM64_BOOT_MAX_SLICES=1` boot-to-pregame parity (same `field_quickening` and
`total_instructions` within boot noise, `rc=0`), then a matched A/B for the perf
number. Reject anything that regresses parity or the short gates.

1. **ABI + `RegionVm` + side-exit plumbing, zero ops supported.** Every region
   compiles to "side-exit at PC 0". Proves the enter/exit/materialize/step
   accounting round-trips with no behaviour change. Pure infrastructure PR.
2. **Numeric core.** Constants, locals (unboxed), arithmetic, comparisons,
   `Not`/`And`/`Or`, `Jump`/`JumpIfFalse`, `Return` of a number. This is
   `compile_numeric_field_trace` generalised to the region entry/exit model.
   First measurable target: whole numeric leaf procedures (the 2036 the
   Cranelift guarded JIT already catches — move them onto this path and delete
   the old one).
3. **Guarded field r/w.** `load_field`/`store_field` via the integer shape
   guard + slot read, numeric fields stay unboxed. Target: `update_lumcount`
   (~1.8 % of boot, 48 instrs, its bespoke JIT never matches Monke) and the
   `light_source/update_corners` numeric prologue.
4. **Globals, list index/length, type predicates** through the ABI.
5. **Calls and allocations as side-exits**, so a region spanning
   `atom/Initialize`'s straight-line body compiles and only stops at each
   `Initialize()` sub-call. This is where the InitAtom throughput (6.5 %) and
   `update_corners` (10 %) start to move.
6. **Leaf-call inlining** for already-compiled numeric callees.

Expected by milestone 5: a first double-digit-percent boot improvement.
Milestone 6 and beyond is where the 2–4x in VM-heavy regions is plausible.

## Risks

- Cranelift compile latency on a cold boot. Mitigate by compiling on workers
  and only after an entry-count threshold, and by caching compiled regions in
  the ready-world image keyed by the engine-semantics fingerprint.
- GC safety of parked `Value`s. The rooted-slot array is the whole answer and
  it already exists (`CompiledRootedBlock`); the discipline is "every `Value`
  that outlives a slow-path call lives in a slot, never a register".
- Parity drift. The interpreter stays authoritative; regions are gated on
  byte-exact `field_quickening` / instruction-count parity every step.
