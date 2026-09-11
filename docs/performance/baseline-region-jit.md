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
   array passed to the region. **Revised after reading the real code ahead of
   M5:** `dm_jit::CompiledRootedBlock` is not the ready-made version of this —
   its own scratch array (`try_run_rooted_list_jit`'s local `SmallVec<Value>`,
   `fastpath_jit.rs`) is never added to `heap_gc.rs`'s root scan at all; its
   safety comes from atomicity instead (nothing that can trigger a collection
   is reachable from inside `CompiledRootedBlock::run_with`, so the window
   just never opens). That guarantee doesn't extend to a region that side-exits
   to run an arbitrary DM proc call, which very much can allocate/collect. A
   real rooted-slot array for this milestone means a new side-array actually
   wired into `heap_gc.rs`'s scan (owned by `ExecutionState` or the
   `CallFrame`, analogous to how `locals`/`stack` are already scanned) — net
   new work, not reuse. Used for any non-number operand and for numbers about
   to cross the slow-path ABI, once built.

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
side-exit in v1 — the region stops, the interpreter runs the
callee/constructor. **Revised after reading the real code ahead of M5:**
automatic re-entry "at the return PC if it is still hot" is NOT v1 — it needs
two things that don't exist yet and aren't free: (a) region lookup indexed by
an arbitrary PC, not just PC 0 (the sidecar's per-instruction `PcCache` array
already supports this structurally — `field_read_cache_or_install` proves
it — so this part is a small, mechanical change when it's actually needed);
(b) a way to cold-start `NumericExecutionState` from a live interpreter frame
mid-procedure, which today only exists for PC 0 (fresh locals, empty stack).
(b) is the real gap: a resume point right after a call has the call's return
value sitting on `frame.stack`, not just fresh locals, and there's no
reconstruction logic for that. v1 (M5, this session) is simpler and needs
neither: a region compiles its procedure's straight-line **prefix**, up to
the *first* call/alloc/dynamic-dispatch instruction, and permanently
side-exits there — same mechanism every field/global decline already uses
(rematerialize what the interpreter needs onto `frame.stack`, resume at that
exact instruction, never return to native code for the rest of that call).
Arguments and receivers are accepted only when `validate()`'s existing
`Number|Src` kind-tracking (built for M3b) proves them plain numbers — a
`Src`-kind argument (passing `src` itself into a call) is out of scope here
for the same reason general non-`src` field receivers were: it needs the
rooted-slot work above. A later milestone can add true resume-after-call
(needs (a)+(b)) and leaf-call inlining (M6, doesn't need either — a callee
that's *also* a compiled numeric region can be invoked through a callback
exactly like `RegionCallbacks` today, no side-exit required, as long as its
arguments and return are numeric).

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

## Dense IDs — revised after reading the real field-storage code (M3)

The paragraph below this heading was the pre-M3 plan; it assumed a per-`Datum`
shape/version stamp that **does not exist** and was never built for #70.
Corrected understanding, from reading `Datum`/`FieldSlotCache` directly
(`dm-value/src/lib.rs`, `dm-vm/src/execution/state.rs`):

- A datum's fields are a per-instance `Vec<(FieldName, Value)>`
  (`DatumFields::Owned`/`Shared`); a "slot" is a position in *that specific
  datum's* vector, not a compiler-assignable struct offset. Two datums of the
  same `TypePath` are not guaranteed the same slot layout — `delete_field`
  physically shifts later slots, and `DatumLayoutCache` tracks up to 8
  distinct layouts per type. There is no shape/version counter anywhere.
- #70's actual guard is therefore **not** an integer compare: it's `(receiver
  TypePath → cached slot)` in a small per-callsite MRU list, then on *every*
  hit `Datum::field_at_validated_slot` re-reads the `FieldName` stored at that
  position and compares it (by `Arc` string) to the name baked into the
  bytecode. A slot shift is caught here, not by a version bump.
- `datum_field_or_shared`/`assign_datum_or_shared_field` (the two helpers this
  doc already named as the read/write targets) take a `FieldName`, full stop
  — no integer-indexed overload exists, and building a per-module dense
  `FieldName -> FieldId` table would only let native code skip *looking up*
  which `FieldName` to ask for; it does nothing to avoid the per-hit name
  recheck above, which is where the real safety lives.

Given that, `load_field`/`store_field` are not going to be an inlined
"shape-guard then struct-offset read" the way a compiled struct access would
be in a language with fixed layouts — they were already specified as
**slow-path callbacks** in the ABI table above (`unsafe extern "C" fn(ctx,
...)`), and M3 keeps them exactly that: a call out to Rust that runs the
existing #70-shaped lookup (MRU by type, validate by name at the slot) and
hands back a rooted slot index (or a side-exit if the field isn't a number).
No new dense-id table, no new `Datum` version field — M3 is ABI plumbing
(`RegionVm` context, the rooted-slot array as a GC root, the packed
`{status, resume_pc, steps}` outcome, one `unsafe extern "C"` trampoline per
callback, mirroring `CompiledRootedBlock`'s existing safe-closure pattern
exactly) plus wiring the Cranelift side to *call* `load_field`/`store_field`
rather than to inline a guard. `TypeId`/`ProcId`/`GlobalId` dense tables
remain future work for M4/M5, where the underlying stores (`GlobalStore`,
`Module::procedures`, type intervals) really are already dense and this
concern doesn't apply.

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
   **Done** (`023a754`) — proven behaviour-inert via a full boot-to-pregame A/B.
2. **Numeric core. Done.** Constants, locals (unboxed), arithmetic, comparisons,
   `Not`/`And`/`Or`, `Jump`/`JumpIfFalse`, `Return` of a number — `dm-vm`'s
   `PcCache::Region` now installs a `dm_jit::CompiledNumericTrace` directly (no
   separate region wrapper type: for an all-numeric procedure the existing
   trace compiler *is* the region entry/exit model, so nothing new was needed
   in `dm-jit` beyond the `Not`/`And`/`Or` opcodes themselves — `compile_numeric_trace`,
   `NumericExecutionState`, `run_budgeted` are unchanged). Replaces the old
   whole-procedure guarded JIT's generic-numeric path outright (thread-local
   cache → sidecar-hosted warm-up/install at PC 0; `try_run_numeric_jit`/
   `numeric_jit_prefix_candidate` deleted); lumcount's bespoke field trace is
   untouched. `Not`/`And`/`Or` are new coverage the old JIT never had — DM's
   `&&`/`||` always short-circuit to `Jump`/`JumpIfFalse` (so eager `And`/`Or`
   only ever come from a `switch` statement's `to`-range or multi-value
   alternatives), `!` compiles straight to `Not`. `numeric_trace_instructions`
   also grew a reachability pass so a procedure whose every real path already
   returns doesn't get rejected over its compiler-appended, unreachable
   trailing `LoadResult; Return` — a pattern common enough (any exhaustive
   `if`/`else` or `switch`) to be worth not paying full interpreter cost for.
   Resumption across a budget boundary reuses the pre-existing per-`CallFrame`
   `numeric_jit_state` slot exactly as the old JIT did (same safe, already
   continuation-tested mechanism — deliberately not reinvented).
3. **Guarded field read/write for the `src` receiver. Done.**
   `LoadFieldDynamic` calls a slow-path
   callback (`dm-jit`'s first mid-region call, not just a whole-body
   dispatcher) that runs the existing #70-shaped lookup and hands back a
   plain `f32` — no rooted slot needed, since scope stayed numeric-fields-only
   (see the "Dense IDs" revision above for why there's no shape/version guard,
   just the callback re-deriving the value live every time). Receiver is
   always the region's implicit `src` (proven by the translator only ever
   accepting a
   `LoadField` immediately preceded by `LoadSrc`, never a general expression)
   — general (non-`src`) receivers need the mixed-kind operand stack the
   original sketch above assumed and are deferred, likely alongside real
   rooted-slot support once calls/allocations (milestone 5) need it anyway.
   `numeric_trace_instructions` also grew a per-region `FieldName` table.
   A side-exit here (field declined) is a **distinct outcome from budget
   exhaustion** (`NumericRunOutcome::SideExit`, not `BudgetExhausted`) —
   retrying a declined field natively would decline forever, so the VM must
   know to hand off to the interpreter rather than resume native execution;
   getting this outcome distinction right (and rematerializing `src` onto
   `frame.stack` before the interpreter resumes — the ONE bug this milestone's
   testing caught, never shipped) is the real substance of this milestone.
   Target: `update_lumcount` (~1.8 % of boot, 48 instrs, its bespoke JIT never
   matches Monke) and the `light_source/update_corners` numeric prologue —
   still not directly hit (both have shapes outside this milestone's narrow
   scope), but `jit_guarded` telemetry shows real broader effect already:
   steps served natively +19.9% in the read-support parity boot, from other
   `src.field`-reading procedures the translator previously rejected
   outright (a smaller, still-positive bump from write support landing
   after). **Writes** (`StoreFieldDynamic`) needed more than the read side's
   adjacency check: a store's receiver sits under an arbitrary-length value
   expression, not immediately below the store, so `dm-jit`'s `validate`
   grew real operand-*kind* tracking (`Number | Src`, mirroring its existing
   per-PC depth tracking) as the soundness proof instead — which then let the
   VM-side translator drop its adjacency heuristic for reads too and lower
   `LoadSrc`/`LoadField`/`StoreField` unconditionally, `validate` alone
   carrying the whole proof for both. General (non-`src`) receivers still
   need the fuller mixed-kind rooted-slot operand model the original sketch
   above assumed, and remain deferred, likely alongside real rooted-slot
   support once calls/allocations (milestone 5) need it anyway.
4. **Guarded globals. Done** (globals only — list index/length and type
   predicates narrowed out, see below). `LoadGlobalDynamic`/`StoreGlobalDynamic`
   mirror the field-access callbacks but are structurally simpler: a global has
   no receiver (`LoadGlobal`/`StoreGlobal` are pop0/pop1, never popping a
   `src`), so there's no receiver-kind proof to build — just a dense
   per-region `FieldName` table for globals alongside the existing one for
   fields. The callback ABI was refactored ahead of this milestone: two more
   closures would have pushed `run_budgeted` to six parameters on top of the
   `RefCell` wrapping already needed for two closures to share
   `&mut ExecutionState`. Replaced both with a single `RegionCallbacks` trait
   (`load_field`/`store_field`/`load_global`/`store_global` as `&mut self`
   methods on one object crossing the FFI boundary as one fat-pointer
   context) — one implementor can answer every callback from the same
   `&mut ExecutionState` because native code only ever calls one method at a
   time, and the borrow checker accepts sequential `&mut self` calls without
   needing `RefCell` to convince it. All pre-existing field-access tests
   passed unchanged under the refactor before globals were added on top,
   confirming it was behavior-preserving on its own.
   List index/length and type predicates were dropped from this milestone's
   scope: unlike a global (no receiver at all) or `src` (an implicit,
   never-materialized receiver already known to the callback context), a list
   receiver is an arbitrary `Value` that has to actually flow through the
   region — as a local, a call result, or a nested expression — which needs
   the mixed-kind rooted-slot operand model the field-write milestone already
   flagged as out of scope for the same reason (see the M3 writeup above).
   **Correction, written alongside milestone 5's scoping below:** that
   infrastructure turned out NOT to be milestone 5's either — milestone 5's
   real (narrower) scope needs no rooted `Value` support at all, so list ops
   and type predicates are deferred further, to whichever later milestone
   actually builds true resume-after-call (see milestone 5's writeup).
   Parity: `exec_steps` delta
   +0.0022% against the M3b baseline (998,258,700 → 998,280,937), both
   `rc=0`, `field_quickening hit_pct=87.1` identical in both runs — the
   tightest parity result of any milestone so far. `jit_guarded` telemetry
   showed the expected signal: `numeric_compiled` up 4504→4640 (+3.0%, more
   procedures touching globals now qualify for the region tier).
5. **Calls and allocations as a prefix-ending side-exit.** Narrowed after
   reading the real call/alloc/GC code (see the "Operand model" and slow-path
   ABI corrections above) — the original "stops at *each* sub-call and
   re-enters after" vision needs true mid-procedure region re-entry and a
   real rooted-`Value` side-array wired into `heap_gc.rs`, neither of which
   exists yet, and the mechanism it was modeled on
   (`dm_jit::CompiledRootedBlock`) turns out not to provide the second one
   either. **v1 scope:** a region compiles a procedure's straight-line
   *prefix* — constants/locals/arithmetic/guarded field+global access, same
   as milestones 2-4 — up to the *first* `Call`/`CallCurrent`/`CallParent`/
   `AllocateDatum`/`AllocateCurrentDatum` instruction, and permanently
   side-exits there, reusing the exact mechanism every field/global decline
   already uses: rematerialize what the interpreter needs onto `frame.stack`
   (here, the call's own popped arguments — `validate()`'s existing
   `Number|Src` kind tracking from M3b proves each argument slot's kind, so a
   `Src`-kind argument, e.g. passing `src` itself into a call, is rejected
   for now the same way a general non-`src` field receiver was), resume the
   interpreter AT that instruction, and never return to native code for the
   rest of that call. `CallDynamic` (runtime-selected callee) is out of scope
   for v1 — only the statically-resolvable call shapes. No new GC work, no
   PC-indexed region re-entry, no rooted slots: this is a straight
   generalization of the side-exit machinery M3-M4 already proved out, with
   compile-time-certain unconditional exits (no callback/FFI call needed at
   the call site itself, since whether to exit isn't a runtime decision).
   This buys the straight-line prologue of hot procedures like
   `atom/Initialize()` — real but smaller than the full vision, since
   anything from the first call onward (often most of the procedure) still
   runs interpreted.
6. **Leaf-call inlining** for already-compiled numeric callees whose
   arguments and return are provably numeric — invoked through a callback
   exactly like `RegionCallbacks` today (a real Cranelift `call` to a Rust
   trampoline that runs the callee's own `CompiledNumericTrace` and hands
   back a plain `f32`), no side-exit, no rooted slots needed either, since
   scope stays numeric-in-numeric-out by construction.

True resume-after-call and rooted-`Value` support (the InitAtom-throughput
and `update_corners`-class wins the original milestone 5 targeted) are now a
later milestone, once (a) PC-indexed region re-entry and (b) a real
GC-scanned rooted side-array both exist — (a) is structurally cheap (the
sidecar's per-instruction cache array already supports arbitrary-PC entries),
(b) is the real work. Milestone 6 and beyond is where the 2–4x in VM-heavy
regions is plausible, once that lands.

## Risks

- Cranelift compile latency on a cold boot. Mitigate by compiling on workers
  and only after an entry-count threshold, and by caching compiled regions in
  the ready-world image keyed by the engine-semantics fingerprint.
- GC safety of parked `Value`s. **Revised ahead of milestone 5** (see the
  "Operand model" correction above): no ready-made root-scanned rooted-slot
  array exists yet — `CompiledRootedBlock`'s scratch array is real but is
  never added to `heap_gc.rs`'s scan; its safety today comes from atomicity
  (nothing GC-triggering runs while it's live), a guarantee side-exiting into
  an arbitrary DM proc call doesn't have. Building one is real, net-new work:
  a side-array actually wired into the root scan, owned by `ExecutionState`
  or `CallFrame`. Milestones 2-5 sidestep this entirely by staying
  numeric-only and side-exiting (never rooting) anything that isn't a plain
  number; the discipline once the rooted array exists is still "every `Value`
  that outlives a slow-path call lives in a slot, never a register."
- Parity drift. The interpreter stays authoritative; regions are gated on
  byte-exact `field_quickening` / instruction-count parity every step.
