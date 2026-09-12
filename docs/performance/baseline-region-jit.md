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
5. **Calls and allocations as a prefix-ending side-exit. Done.** Narrowed
   after reading the real call/alloc/GC code (see the "Operand model" and
   slow-path ABI corrections above) — the original "stops at *each* sub-call
   and re-enters after" vision needs true mid-procedure region re-entry and a
   real rooted-`Value` side-array wired into `heap_gc.rs`, neither of which
   exists yet, and the mechanism it was modeled on
   (`dm_jit::CompiledRootedBlock`) turns out not to provide the second one
   either. **Shipped scope:** a region compiles a procedure's straight-line
   *prefix* — constants/locals/arithmetic/guarded field+global access, same
   as milestones 2-4 — up to the *first* `Call`/`CallCurrent`/`CallParent`/
   `AllocateCurrentDatum` instruction, and permanently side-exits there,
   reusing the exact mechanism every field/global decline already uses:
   rematerialize what the interpreter needs onto `frame.stack` (here, the
   call's own popped arguments — `validate()`'s existing `Number|Src` kind
   tracking from M3b proves each argument slot's kind, so a `Src`-kind
   argument, e.g. passing `src` itself into a call, substitutes the region's
   implicit receiver on rematerialization, the same way a field receiver
   already does), resume the interpreter AT that instruction, and never
   return to native code for the rest of that call. `CallDynamic`
   (runtime-selected callee) and `AllocateDatum` (its popped type-path
   operand isn't representable in the numeric-only operand model — no
   `TypePath`-kind tracking exists) are out of scope. The call must also be
   reached by a **pure straight line from procedure entry** — no branch
   anywhere before it — because bytecode *array* order isn't execution
   order once a branch exists; accepting a call reachable only through one
   path while other array positions belong to paths that never reach it
   at all risks silently truncating a still-reachable branch. No new GC
   work, no PC-indexed region re-entry, no rooted slots: this is a straight
   generalization of the side-exit machinery M3-M4 already proved out, with
   compile-time-certain unconditional exits (no callback/FFI call needed at
   the call site itself, since whether to exit isn't a runtime decision).
   This buys the straight-line prologue of hot procedures like
   `atom/Initialize()` — real but smaller than the full vision, since
   anything from the first call onward (often most of the procedure) still
   runs interpreted.
   - **A real, general bug found via this milestone's own boot-testing,
     predating it.** Every side-exiting instruction's rematerialization —
     including the 4 already-merged from M3/M3b/M4 — only ever reconstructed
     what the *specific declining instruction* needed (a receiver, a stashed
     value); nothing accounted for a value pushed *earlier in the same
     expression* and still pending underneath. `return 2 * value` (constant
     first, then a field read) was already a latently-reachable shape on
     `main`, unrelated to this milestone — it simply had never been
     exercised by a boot before. This milestone made the shape concretely
     common (constant, then a declining dynamic op, then a call), and a real
     Monkestation procedure hit it on the first boot-parity attempt:
     `/proc/random_color()` (`return random_string(6, hex_characters)` — a
     constant, a non-numeric global read, then a call) crashed with
     `bytecode stack underflow at instruction 2`, because the declining
     global read's rematerialization lost the already-computed `6`. Fixed
     by requiring, in `validate()`, that every side-exiting instruction's
     own operands are the *only* thing on the operand stack — nothing
     pending beneath them — for all five side-exiting instructions
     (`LoadFieldDynamic`/`StoreFieldDynamic`/`LoadGlobalDynamic`/
     `StoreGlobalDynamic`/`CallSideExit`), not just the new one. Confirmed
     this doesn't regress any of M3/M3b/M4's existing coverage: every
     pre-existing test already (accidentally) satisfied this requirement,
     since each one happened to put its declining instruction first in its
     own expression.
6. **Leaf-call inlining. Done — as a compile-time splice, not the
   originally-sketched runtime callback.** The doc's original plan called
   for "a callback exactly like `RegionCallbacks` today... that runs the
   callee's own `CompiledNumericTrace`" — reading the real code before
   implementing found this architecturally blocked: `run_frames` hollows
   `state.program_sidecars` out into a disjoint local for the whole run,
   and neither `try_run_region_numeric_jit` nor its `RegionDispatch`
   callback context ever receives the sidecars — there is no path from
   inside a running region's callback back to another procedure's compiled
   trace at runtime. Built instead as a genuine compiler inline: when the
   translator (`numeric_trace_instructions`) reaches a `Call` whose callee
   provably qualifies, it splices the callee's own translated instructions
   directly into the caller's sequence at compile time (argument binding via
   ordinary `StoreLocal`s, the callee's locals renumbered into fresh slots
   past the caller's own, its final `Return` dropped since the value it
   would have popped already sits on the shared native operand stack).
   Qualification is narrow and safety-first, matching every milestone's own
   pattern: both caller *and* callee must be entirely branch-free (a jump
   target is an absolute instruction index; splicing shifts everything
   after it out of alignment, so requiring branch-freedom on both sides
   sidesteps rewriting those targets entirely rather than risking that math
   being wrong); the callee must be a true leaf (no field/global access, no
   calls or allocations of its own, which bounds this to exactly one level
   of inlining by construction); the callee's declared parameter count must
   exactly match the call site's argument count; and any of the callee's
   *other* locals (DM reserves extra compiler-internal slots — the implicit
   `.` variable among them — regardless of declared parameter count) must
   be provably safe to default to zero, reusing the same
   `local_is_definitely_initialized_before_load` check
   `try_run_region_numeric_jit` already applies to a region's own top-level
   entry locals. A real design bug — comparing the call site's argument
   count against the callee's *total* local count instead of its *declared*
   parameter count, which would have rejected nearly every real callee —
   was caught by a translator-level test asserting the exact spliced
   instruction sequence, before ever reaching a boot.

**All six originally-planned milestones are now complete and merged.**
True resume-after-call and rooted-`Value` support (the InitAtom-throughput
and `update_corners`-class wins this project originally targeted) remain
future work, needing (a) PC-indexed region re-entry and (b) a real
GC-scanned rooted side-array — (a) is structurally cheap (the sidecar's
per-instruction cache array already supports arbitrary-PC entries), (b) is
the real work. That's now a distinct, larger follow-on effort rather than a
seventh numbered milestone in this plan — the 2–4x in VM-heavy regions this
doc opens with is plausible once it lands.

## Milestone 7 (follow-on, beyond the six-milestone plan) — PC-indexed
call-resume regions. Done, narrower in scope than "true resume-after-call"
above promised, and its boot-time payoff at current scale is within
measurement noise.

Delivers piece (a) from the follow-on paragraph above — PC-indexed region
re-entry — but deliberately not piece (b): this still stays numeric-only,
exactly like milestones 2-6, so it does not unlock list/datum-shaped
resume tails. What "true resume-after-call" needs is a real rooted-`Value`
GC-scanned side-array; this milestone doesn't build one and doesn't need
one, because it only ever resumes into more constants/locals/arithmetic/
guarded field+global access — the same numeric surface every earlier
milestone already proved safe.

**Mechanism.** `dm-jit`'s `validate`/BFS and Cranelift codegen turned out to
already be entry-PC-generic (this is how budget-exhaustion resume already
worked) — only the BFS seed needed a parameter
(`compile_numeric_field_trace_at`/`CompiledNumericTrace::initial_state_at`).
The real new work is on the `dm-vm` side: `numeric_trace_instructions_at`
translates starting at an arbitrary `entry_pc`, filling positions before it
with an inert `Return` placeholder (never actually reached, so position
alignment with real bytecode — which every side-exit resume-PC and jump
target depends on — never has to be renumbered); the sidecar's
`PcCache` array (already proven arbitrary-PC-capable by the pre-existing
field-read cache) grows `is_region_site`/`register_region_candidate` so a
second, independent warm-up counter can live at a non-zero PC alongside PC
0's own. The candidate PC itself comes from a narrow, local check
(`safe_call_resume_pc`) rather than a general whole-procedure operand-stack
analysis (which would need an accurate pop/push table for every DM
`Instruction` variant, including a genuinely runtime-dependent one,
`ExpandArgumentLists` — real risk for comparatively little gain): a
call-family side-exit's resume candidate is accepted only when the
instruction immediately following the call is `Pop`/`StoreResult`/
`StoreLocal` (all provably pop-1-push-0, so the position after *that* has
depth 0 unconditionally) and the resulting position is never a jump target
anywhere in the procedure — proven once, by scanning every jump-shaped
instruction, rather than trusting an accumulated depth count.

**A real interaction found via this milestone's own end-to-end test, before
ever reaching a boot.** The pre-existing tier-1 `numeric_dispatch_candidate`
quick-block (`numeric_core.rs`, predates the region tier entirely — a
cheaper, still-interpreted batched-dispatch fast path, not a JIT) already
covers `PushNumber`/`LoadLocal`/`StoreLocal`/arithmetic/comparisons/branches.
Since a call's own consumer (`Pop`/`StoreResult`/`StoreLocal`) is itself one
of its candidates, it gets first crack at any resume site starting from one
instruction *before* this milestone's own entry point, and (correctly, just
inconveniently) runs the whole numeric span in one shot — meaning the
interpreter's main loop can go on to never revisit this milestone's own
entry PC as its own dispatch, so the sidecar never sees it and a resume
region that would otherwise be perfectly valid never gets the chance to
warm up or compile at all. Not a correctness bug — the quick-block computes
the same answer — but a real, silent way for this milestone's own mechanism
to never fire for an all-arithmetic resume tail. Confirmed empirically: **a
pure-arithmetic resume tail is already "handled well enough" by the older
mechanism**, so this milestone's own distinct, additive contribution is
specifically for resume tails that touch a field or global (the tier-1
quick-block has no `LoadField`/`LoadGlobal` support at all) — a real and
probably common shape (e.g. `var/result = SomeProc(x); src.value = result`),
but narrower than "every call site with a numeric tail" might suggest.

**Boot-parity result — mechanism correct, payoff not measurable yet.** Full
lib suites green (30 dm-jit + 71 dm-value + 673 dm-vm), zero new clippy
warnings, boot-to-pregame parity clean (`rc=0`, `field_quickening
hit_pct=87.0`, identical to the M6 baseline). `jit_guarded` telemetry
confirms the mechanism is real and exercised, not just present:
`numeric_compiled` 6,027 → 6,284 (+257 more regions installed, at call-resume
sites this milestone specifically enables), native region `runs` 1,879,644
→ 2,172,351 (+15.6%), steps served natively 3,275,155 → 3,475,534 (+6.1%).
But the wall-clock/`avg_ns_per_instr` delta this produced (+5.5%, +33s;
GC time +16.3%) turned out to be **within the same boot's own measurement
noise floor**: a same-binary control (the unmodified M6 baseline, run again
on the same machine) produced a comparable swing on every metric —
`avg_ns_per_instr` +4.15%, GC time +15.2%, `exec_steps` +0.51% (larger than
this milestone's own +0.12% exec_steps delta) — from nothing but ordinary
map-generation/run-to-run variance. Cross-checking `update_corners` (a
procedure structurally incapable of using any region-tier optimization, in
either run) showed its own step count scaling proportionally with each
run's overall workload, confirming the swing is workload noise, not a
regression this milestone introduced. **Net assessment: this milestone adds
a real, correctness-proven, measurably-exercised capability, not yet a
proven boot-time win** — extracting one, per the pre-existing "Risks"
section below, most likely needs the compile side of this (many more
independent call-resume sites than PC-0 entries alone, each a synchronous
Cranelift compile on the boot thread) to stop paying for itself inline and
move to a worker, rather than more of the numeric-surface coverage this
milestone itself adds.

## Background region compilation (follow-on to Milestone 7) — moves
Cranelift compilation off the boot-critical thread. Done; a real, modest
win, honestly measured.

Directly answers Milestone 7's own closing finding above and the "Cranelift
compile latency on a cold boot" risk this doc has flagged, unmitigated,
since the tier's very first milestone. `compile_region_trace_at` only ever
reads `Module`/`Program` — immutable, `Send + Sync` data with zero
dependency on `ExecutionState` or the executing frame — so it needed no
`unsafe` (`dm-vm` forbids it) and no new external dependency (`std::thread`
+ `std::sync::mpsc`, the same primitives `compile.rs`'s
`parallel_collect_ordered` and `tgm_planner.rs`'s `with_workers` already use
elsewhere in this crate, just aimed at a persistent worker instead of a
one-shot joined batch).

**Mechanism** (`crates/dm-vm/src/execution/region_compile_worker.rs`): one
background thread, spawned lazily and shared process-wide via a
`OnceLock<mpsc::Sender<_>>` (a long-running server process has no shutdown
story to build — the same reason `worker_lane`'s joined batches need none).
`ProcedureSidecar` gains a `RegionCompiling` `PcCache` state: crossing the
warm-up threshold now either compiles synchronously and installs
immediately (`async_compile: None` — every case before this) or submits a
request and marks the slot `RegionCompiling` (`Some`), with the finished
`CompiledRegion` installed later via `ProgramSidecars::drain_async_region_results`.
Draining happens at a procedure switch — an already-paid-for point in
`run_frames_inner` (a `sidecars.resolve` call already happens there), not a
new per-instruction cost — rather than blocking dispatch to poll a channel
every step.

**Deliberately opt-in, not the tier's new default.** Every existing
region-JIT test (15+ of them, back to Milestone 1) asserts a region
installs *synchronously*, the instant a warm-up counter crosses threshold —
enabling this unconditionally would make that timing depend on real
thread-scheduling variance and flake them. `ExecutionState::async_region_compile`
defaults to `None`; only `ExecutionState::enable_async_region_compile`,
called explicitly by the one real production entry point
(`dm_runtime::RuntimeImage::decode_linked_artifact`, the `.d64`
linked-artifact boot path every real `dream64-server boot` invocation goes
through), turns it on. Every one of those existing tests passed unchanged
under this change — confirmed via a full suite run before and after, not
assumed. One new end-to-end test
(`region_jit_async_compile_installs_a_correct_region_off_the_calling_thread`)
proves the opt-in path itself: a region must NOT be installed the instant
threshold crosses (it must still be `RegionCompiling`), and must eventually
install (via a bounded retry loop, not a fixed sleep) with results
identical to the synchronous path.

**Boot-parity result, measured against the same rigor Milestone 7's own
false alarm demanded.** Full suites green
(30 dm-jit + 71 dm-value + 674 dm-vm + 65 dm-runtime), zero new clippy
warnings, `rc=0`, `field_quickening` unchanged. Comparing against the M7
baseline directly: `avg_ns_per_instr` 693.4 → ~679.8 (mean of two runs,
**~2.0% faster**), `measured_s` 628.2s → ~618.0s (**~10s faster**), while
`numeric_compiled` (6,284 → ~6,120) and native region `runs` (2,172,351 →
~2,174,300) stayed within the same noise band throughout — i.e., this
change does not alter *how much* region-tier work happens, only *where its
compile cost lands*, exactly as designed. Unlike Milestone 7's own single-A/B
false alarm, this result held up under the SAME same-binary-control
discipline that caught that one: two independent runs of the async-compile
binary landed at 680.9 and 678.7 ns/instr, only 0.3% apart — a noise floor
far tighter than the ~4% swing a same-binary M6 control pair showed earlier
in this same investigation — and BOTH runs sit clearly below the single M7
sample being compared against. Honestly: this rests on two async samples
against one M7 sample, not a full paired A/B in both directions, so treat
the ~2% figure as directionally real but not tightly bounded — the
mechanistic explanation (compile cost relocated, not eliminated or
duplicated, confirmed by `numeric_compiled`/`runs` parity) matters more here
than the exact percentage. Independent of the measured magnitude, this
change is justified on its own architectural merits: it closes a
structural risk this doc has carried since Milestone 1, one that only grows
more expensive as later milestones (this one included) keep adding more
independent compile sites than a procedure's own entry alone ever created.

## Risks

- Cranelift compile latency on a cold boot. **Compiling on workers: done**
  (see "Background region compilation" above) — opt-in, real boots only,
  every region-JIT test still exercises the synchronous path. Compiling
  only after an entry-count threshold was already in place from Milestone 1.
  Caching compiled regions in the ready-world image keyed by the
  engine-semantics fingerprint remains undone — every boot still recompiles
  every region from scratch, just off the critical path now instead of on it.
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
