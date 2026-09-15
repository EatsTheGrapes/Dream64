# BYOND release-note archaeology

A clean-room reconstruction of BYOND's runtime architecture from its published
release-note archive, read as evidence of which expensive DM semantics the
engine progressively stopped executing generically.

This document contains **no proprietary implementation detail**. Every entry is
either (a) a public statement from BYOND's own release notes, (b) a public
statement from BYOND's own reference documentation, (c) a fact established by
the local read-only binary audit already recorded in this repository, or
(d) an explicitly labelled architectural inference drawn from those. No
disassembly, private structure layout, algorithm, or source text is reproduced.

## 0. Provenance and method

### 0.1 What the archive is

BYOND's official release-note archive is published as one HTML page per major
version at `https://www.byond.com/docs/notes/<version>.html`. The archive
confirmed to exist spans at least:

- **2.x/3.x era:** 219p3, 224, 225, 243, 252, 266, 267, 272, 273, 276, 282,
  305/306, 307b (beta), 307–314, 318
- **4.0 era:** 400.950, 401.963, 402.964, 405.970, 413.978, 415.980, 430.1005,
  432.1007, 448.1030, 459.1051, 463.1065, 483.1092, 490.1113, 494.1135, 499
- **5.0 era:** 500, 501, 504, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516

A per-build binary index also exists at `https://www.byond.com/download/build/`,
with build folders reported back to build 354.

### 0.2 Retrieval constraint — read this before trusting a build number

**This session cannot reach byond.com.** Both `www.byond.com` and
`secure.byond.com` are denied by the environment's egress policy (HTTP 403 at
the CONNECT layer, and `EGRESS_BLOCKED` through the fetch tool). Mirrors of the
notes (`byond-builds.dm-lang.org`, `spacestation13.github.io`) and the Internet
Archive are blocked as well. Only GitHub and package registries are reachable.
I did not route around the block.

The archive content below was therefore recovered **through search-engine
excerpts of the official pages**, restricted by domain to `byond.com`. This has
one material consequence:

> **Build numbers are not recoverable for the 5.0 series.** The 5.0 notes are
> published per major version (`512.html`, `515.html`, …) and each individual
> entry's build number lives in the page body, which the excerpts do not carry.
> Where a build number appears in this document it comes from the page *title*
> (the 4.0-era pages are titled e.g. "BYOND 4.0 Version 413.978 Release
> Notes"), and is marked as such.

Every entry whose build is unknown says `not captured` rather than a guess.
Nothing in the `build` column is inferred, reconstructed, or estimated. If you
later obtain the archive directly, the version attributions below are the index
to re-key against; only the `build` column needs filling.

One attribution was actively contested and resolved: an early excerpt placed the
O(1) `istype()`/`typesof()` work in 515, but two independent 514-targeted
retrievals both state it as 514, and the 515 page's own reference to "backported
speedups" concerns the *compiler*, not type lookup. It is recorded as **514**.

### 0.3 Confidence taxonomy

| Tag | Meaning |
| --- | --- |
| **D** | Documented fact. A statement made by BYOND's official release notes or reference docs, retrieved as an excerpt of the official page. |
| **D-bin** | Documented by local evidence. Established by this repo's read-only static audit of the locally installed 516.1680 binaries (`byond-516.1680-dll-audit.md`). |
| **I** | Architectural inference. My reading of what a documented change implies about internal representation. Not asserted as BYOND's design. |

`I` entries are the load-bearing ones for Dream64's design, and they are the
ones to distrust first. They are stated as inferences on purpose.

### 0.4 Optimization phase

`compile` = work moved into DM compilation; `startup` = work done once at world
load; `runtime` = work in the steady-state execution loop. Some entries are
`startup+runtime` where a representation choice pays at both.

---

## 1. Type / prototype representation

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 514 | not captured | `istype()` lookups and generation of `typesof()` lists optimized to **O(1) instead of O(depth)**; the sole exception is that `typesof()` for mob types works the old way | Per-call ancestor walk proportional to inheritance depth. DM idiom is saturated with `istype()`; a deep tree made every type test scale with tree depth | runtime | `dm-vm::execution::type_metadata` — `subtype_interval()` already returns an `(u32,u32)` interval per path | **D** |
| 514 | — | *(implication)* An O(1) subtype test **and** O(1) `typesof()` generation together require a precomputed flattened index over the type tree — contiguous subtree numbering (interval containment) or a per-type descendant bitset — built once from the tree rather than walked per query. The mob exception implies mob `typesof()` carries an extra dynamic component that resists the flat index | — | startup (build index) / runtime (query) | Dream64's interval representation is the same family of answer | **I** |
| 515 | not captured | The `parent_type` var did not return the correct low-level type in many cases, producing situations where `atom.parent_type` was not exactly equal to `/datum` | Divergence between the DM-visible path tree and an internal type identity | runtime | `dm-object-tree` canonical path/parent links | **D** |
| 515 | — | *(implication)* BYOND maintains an internal **low-level type identity distinct from the DM path tree**, consistent with native base classes implemented as engine structs with the DM tree layered over them. A pure path-tree model would make this class of bug impossible | — | — | Dream64's split of `dm-object-tree` (paths) from native root seeding in `type_metadata` mirrors this, and inherits the same hazard | **I** |
| 512 | not captured | "Generating the object tree was slower than it needed to be. Optimizations have improved the speed significantly." | Object-tree construction cost at compile/startup on large projects | compile / startup | `dm-object-tree` construction | **D** |
| 307–314 | not captured | Fixed initialization of built-in variables when user-defined variables with the same name were defined at higher inheritance levels | Name collision between engine-owned var slots and user-declared vars in the same instance layout | startup | `dm-value` field-name layout interning | **D** |
| 307–314 | — | *(implication)* Built-in and user vars share one per-type slot layout, resolved by name at layout-construction time — not two disjoint namespaces | — | — | Dream64 interns "distinct immutable field-name layouts" and "distinct physical field-index allocations" (`dm-value`), the same merge | **I** |

**Reading.** Type identity is the first thing BYOND flattened. The 514 change is
the archive's cleanest example of the whole pattern this document is looking
for: a semantic operation that is *naturally* a tree walk was replaced by a
precomputed index, paying startup cost once to make the hot query constant-time.
Dream64 already holds the same answer; the useful detail is the **mob exception**,
which says the flat index was not universal and that at least one type family
needed a dynamic path.

---

## 2. Var / default representation

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | not captured | "An optimization has been made to **reduce memory and improve object init speed by reorganizing how default vars are looked up**." | Per-instance materialization of inherited default values: both the bytes to store them and the time to write them at construction | startup + runtime | `dm-vm::execution::state` — "sparse inherited scalar defaults"; `dm-value` layout interning | **D** |
| 512 | — | *(implication)* **The most load-bearing sentence in the archive for Dream64's boot cost.** A single change that reduces memory *and* speeds object init, by altering *default var lookup*, means instances stopped carrying eagerly-copied inherited defaults and began resolving them through a shared per-type table, materializing a slot only on write. Copy-on-write instance vars | — | startup + runtime | Dream64's sparse-default + override-on-write model is the same design, arrived at independently | **I** |
| 512 | not captured | `world.vars` and `global.vars` did not behave like proper lists | The `vars` surface is a synthesized view over a slot layout, not a stored list | runtime | `dm-vm` `vars` builtin surface | **D** |
| ≤512 | not captured | `initial()` did not work correctly with `bound_x/y/width/height` or the combined `bounds` var, "and this also resulted in bounds appearing in savefiles when they hadn't actually changed"; `initial()` for any bound var returned the value of `initial(bounds)` | Aliased/compound engine vars breaking the initial-value table | runtime | `dm-vm::value_ops::engine_fields` | **D** |
| — | — | *(implication)* **Savefile serialization diffs each var against its type default** and stores only what differs. A wrong `initial()` therefore silently corrupts the delta test and bloats savefiles. Persistence correctness is coupled to the default-value table, so the default table must be exact, not merely fast | — | runtime | Any Dream64 savefile writer must key on the same `initial()` table that the sparse-default path serves | **I** |
| 314 | not captured | Fixed initialization of built-in vars shadowed by user vars at higher inheritance levels (see §1) | — | startup | — | **D** |

**Reading.** BYOND spent 512 doing precisely the thing Dream64's boot profile
demands: taking per-instance default state out of the instance. The archive
independently confirms that this is simultaneously a memory win and an
initialization-latency win, which is the argument for prioritizing it.

---

## 3. Proc representation and dispatch

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 515 | not captured | Byondapi's `Byond_CallProc()` no longer needs the **verbified name** of a proc to call it correctly. "Although this is how things work internally with regular proc calls, `call()` actually uses a better method, and this has been changed to work like `call()` instead" | Name-keyed proc resolution on every call | runtime | `dm-semantics` procedure registries; `dm-vm` call lowering | **D** |
| 515 | — | *(implication)* The most revealing dispatch statement in the archive. Ordinary DM proc calls historically resolved through a **name-derived (verbified) key**, i.e. a string-keyed lookup, while `call()` used a more direct mechanism that BYOND itself characterises as "better". This is a name-hash dispatch, not a vtable slot index | — | runtime | Dream64 resolves to stable procedure identities at lowering time — structurally ahead of the documented BYOND path | **I** |
| 513 | not captured | Compiler can now infer the type when the left hand of a chained `.` or `?.` is an assignment, or one of several built-in procs whose return type is known, "which **avoids the costly downward lookup used by the colon operator**" | Runtime downward search of the type tree for a member reachable only dynamically | compile | `dm-semantics` typed IR; `dm-vm::compile_expr` | **D** |
| 513 | — | *(implication)* The `:` operator's runtime cost is a **downward** search (scan descendants for a matching member), not an upward inheritance walk. Every statically-recoverable type turns that search into a direct reference, which is why type inference is framed as a compile-time *speedup* rather than a correctness feature | — | compile | Dream64's typed semantic IR is the same lever | **I** |
| 512 | not captured | "Some small optimizations have been made to proc code, including constant loads, some var reads, and **proc data initialization**" | Per-call setup cost of a proc frame | runtime | `dm-vm::execution::frame`; frame/state pooling | **D** |
| 516 | not captured | Byondapi: `Byond_CRASH()` added (does not trip stack unwinding or error traps in the calling library) | — | runtime | extension boundary | **D** |

**Reading.** Dispatch is where the archive most clearly shows BYOND *not*
having flattened something. As late as 515 the documented internal path for an
ordinary proc call is name-keyed, and the fix was to adopt what `call()` already
did. For Dream64 this is the clearest case where the reference engine's
behaviour is not the design to copy.

---

## 4. Init / New / `..()` execution

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | not captured | "During **analysis of init procs to consolidate them**, objects could be created that were sometimes not let go of properly or worse, called the soft `Del()` proc. Now init procs will no longer create atoms, images, or hard datums during analysis." | Per-type variable initializers executed as separate work per instance | compile / startup | `dm-lifecycle::initialization_plan`, `precompile` | **D** |
| 512 | — | *(implication)* **BYOND runs a consolidation pass over per-type init procs**, and that pass *evaluates* initializer expressions at analysis time — which is exactly why it could accidentally construct live atoms and trip `Del()`. The engine folds what it can of a type's initializers into one consolidated routine ahead of instantiation, and the 512 fix is a containment boundary: analysis may fold, but may no longer allocate live objects | — | compile / startup | `dm-lifecycle`'s non-executing plan model already draws this boundary explicitly ("deterministic, non-executing lifecycle resolution") | **I** |
| ≤512 | not captured | Calling `image()` did not call the overridden `New()` for `/image`; deleting images, directly or by GC, did not call `Del()` | `/image` bypassed the normal datum lifecycle | runtime | `dm-lifecycle` | **D** |
| — | — | *(implication)* `/image` is engine-allocated off the general datum construction path — a specialized allocator for a high-volume type, retrofitted into the DM lifecycle only later | — | runtime | relevant if Dream64 specializes high-volume types | **I** |
| 516 | not captured | `for(thing as() in list)` restored to old behaviour for empty expressions, "codified in the language" | — | compile | `dm-vm::compile_stmt` | **D** |

**Reading.** The phrase *"analysis of init procs to consolidate them"* is the
single strongest evidence in the archive that BYOND treats per-type
initialization as a **compile/startup-time program to be folded**, not as code to
re-interpret per instance. Dream64's `initialization_plan` is the same idea; the
archive's contribution is the warning that the folding pass must be side-effect
free.

---

## 5. Bytecode / interpreter optimization

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | not captured | Small optimizations to proc code: **constant loads, some var reads, and proc data initialization** | Generic operand decode for the most common instruction shapes | compile → runtime | `dm-vm::compact_wordcode` (one 32-bit word per instruction: 8-bit selector + 24-bit operand) | **D** |
| 276 | not captured | The **code generation optimizer** was mis-compiling certain obscure `switch()` situations | — | compile | `dm-vm::compile_stmt` | **D** |
| 276 | not captured | Subtraction of constants was not being evaluated at compile time | Constant folding gap | compile | `dm-globals::constant` | **D** |
| 513 | not captured | Adding constant numbers and strings at compile time did not compile correctly, producing wrong runtime results | Constant folding across mixed types | compile | `dm-globals::constant` | **D** |
| 516.1680 | 1680 | Local audit: the dispatcher is a **word-oriented threaded interpreter** — 32-bit bytecode words, a 16-bit word-offset program counter, an absolute-address jump table of 397 legal slots with 396 distinct handlers, and execution-context cache fields written by dedicated opcodes | — | runtime | `dm-vm::execution::interpreter` | **D-bin** |
| — | — | *(implication)* **BYOND has no JIT.** Across the entire archive the answer to interpreter cost is always *specialize the opcode or fold the operand*, never *emit machine code*. A ~400-entry opcode table with dedicated cache-field opcodes is a mature superinstruction/quickening design, not a tiering design | — | — | Dream64's `dm-jit` + region tiers are **outside** BYOND's documented design space | **I** |

**Reading.** That a codegen optimizer and constant folding existed by build 276
— the 2.x/3.x era — while no JIT exists at 516 is the sharpest architectural
statement in this whole document. BYOND's entire performance strategy is
compile-time folding plus opcode specialization over a hand-tuned native
interpreter.

---

## 6. List representation and operations

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 515 | not captured | `for(thing in list)` optimized to better handle the cases where the list is `range`, `orange`, `viewers`, `oviewers`, `hearers`, or `ohearers` — "whereas previously **`view()` and `block()` were the only lists that were optimized in this way**" | Materializing a temporary list for a spatial query that is immediately consumed by a loop | compile + runtime | `dm-vm::compile_stmt` loop lowering — **no equivalent specialization found** | **D** |
| 515 | — | *(implication)* BYOND special-cases the *syntactic pattern* `for(x in <generator>)` and fuses the generator into the loop, skipping list allocation entirely. `view()`/`block()` got it first; 515 extended it to the whole spatial family. This is deforestation applied to the hottest idiom in DM | — | compile + runtime | **Concrete Dream64 gap — see §23** | **I** |
| 506 | not captured | "Further performance tweaks have been added, mainly impacting **associative lists and strings**"; the **string tree** was susceptible to more imbalances than necessary when items were deleted from it | Assoc-list and string lookup cost; tree degradation under deletion | runtime | `dm-value` ordered list storage | **D** |
| 516 | not captured | **`alist()`** added — an associative-only list type "for improved performance in some cases where you want key-value pairs without other list baggage" | The general `/list` carries positional *and* associative machinery even when only one is used | runtime | `dm-value` list representation | **D** |
| 516 | — | *(implication)* BYOND's `/list` is a **hybrid** structure paying for both an ordered vector and an assoc index on every list. `alist()` is the admission that the hybrid is too expensive for key-value-only workloads — a representation split rather than an algorithmic fix | — | runtime | Dream64 can specialize representation internally without a new DM type | **I** |
| 516 | not captured | `for(item,value in list)` can loop items and grab the associated value at the same time | Second lookup per iteration to fetch `list[item]` | compile + runtime | loop lowering | **D** |
| 516 | not captured | New convenience procs for fast operations on associative lists holding numbers as values (e.g. `list("apple"=3)`), "mainly to drive performance in certain games" | Interpreting a DM loop to do bulk numeric aggregation over an assoc list | runtime | `dm-vm::builtins::lists` | **D** |
| 516 | not captured | Breaking: `~=` and `~!` now take associated values into account when comparing associative lists | — | runtime | `dm-value` equality | **D** |
| ≤512 | not captured | Assigning a list wholesale to built-in lists (`overlays`, `underlays`, `filters`, `verbs`, …) used to clear them and add an item at a time — "often inefficient and resulted in **appearance churn**" — now optimized internally | Each incremental add re-derived a new appearance; N adds produced N intermediate appearances | runtime | `dm-vm::builtins::qdel_appearance`, appearance derivation | **D** |
| ≤512 | — | *(implication)* Writes to appearance-bearing lists are **transactional**: the engine batches the whole assignment and derives one final appearance rather than one per element. Appearance derivation is expensive enough to justify a special path | — | runtime | directly relevant to Dream64 appearance caching | **I** |
| 516 | not captured | Byondapi `Byond_ReadListAssoc()` reads assoc lists as key/value pairs, avoiding per-index `Byond_ReadListIndex()` calls | Per-element crossing of the extension boundary | runtime | extension boundary | **D** |

**Reading.** Lists are where BYOND's recent effort concentrates, and the moves
are all representational: fuse the generator into the loop (515), split the
representation when the hybrid is wasteful (516 `alist`), batch the boundary
crossing, and make wholesale writes transactional.

---

## 7. Object / datum allocation

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| ≤513 | not captured | "The limit for datums, previously **65,535**, has been expanded to over **16 million**" | A 16-bit datum identity space exhausted by large worlds | startup + runtime | `dm-value` generational heap identities (64-bit host, not 32-bit-address bound) | **D** |
| — | — | *(implication)* Datums are addressed by an **index into a global datum table**, not by pointer. The original index was 16-bit; the expansion to "over 16 million" is consistent with widening to 24 bits of index within a tagged word rather than moving to a full 32-bit handle | — | — | Dream64 skips this era entirely by construction | **I** |
| 512 | not captured | Default var lookup reorganization improves **object init speed** (see §2) | Per-instance default writes at construction | startup + runtime | sparse defaults | **D** |
| 516.1680 | 1680 | Local audit: `Byond_New` and `Byond_NewArglist` both funnel into **one private construction routine**; eleven direct call sites total; off-owner-thread calls are marshalled through the thread manager and wait for the authoritative thread; **no public bulk atom constructor and no worker fan-out on the inspected path** | — | runtime | `dm-lifecycle::execute` | **D-bin** |
| — | — | *(implication)* Datum allocation is a **scalar, single-threaded, heavily optimized** path. BYOND's atom-creation speed comes from constant-factor tuning of one routine, not from batching or parallelism. Any Dream64 plan that hopes to beat it by parallel construction is not copying BYOND | — | runtime | reinforces "accelerate the scalar path" in the boot research | **I** |

---

## 8. Garbage collection / reference management

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| ref docs | — | BYOND's collector is **reference counting**: once an item is no longer referenced by any variable it is deleted. **Circular references are never collected**; null the cycle or `del` each object | — | runtime | `dm-vm::execution::heap_gc` | **D** |
| 307–314 | not captured | A **sweeper optimization** was turned on that "assumes the reference count is accurate and **aborts a sweep after finding as many dangling references as were expected**", making deletion faster when there are few dangling refs; "many sanity checks" remain | Full heap scan on every deletion | runtime | `heap_gc` reachability collector | **D** |
| 307–314 | not captured | Further: `del X` was still invoking the sweeper even when the only remaining reference was `X` itself; it now **explicitly clears the value of `X` first** because that is much faster. Works for a variable or a list item (`List[i]`) | The common case — deleting through the last reference — paid for a full sweep | runtime | `heap_gc` | **D** |
| 413.978 | **978** | The **5-minute garbage collection cycle was removed**; refcounting "should still work normally, so this cycle was probably redundant and eating processor time for no good reason" | A periodic collection pass running regardless of need | runtime | `heap_gc` growth policy | **D** *(build from page title)* |
| 508 | not captured | A garbage collection change was **reverted**, having been found to cause issues after some time in various games | — | runtime | — | **D** |
| 508 | not captured | Garbage collection did not scan **images** properly | Root-set omission | runtime | GC roots | **D** |
| 511 | not captured | GC "didn't trigger often enough in some cases", and **appearance IDs being reused before client-side deletion** caused a memory leak | Collection frequency; client/server lifetime skew on interned appearance IDs | runtime | appearance interning + client sync | **D** |
| ≤516 | not captured | `Del()` procs "wreaked havoc on natural garbage collection (removal of existing refs) **when a large number of sleeping/spawned procs were in the scheduler**", due to longstanding mistakes preventing early bailout when refs were exhausted and **not scanning the current proc first** | Deletion cost scaling with the number of suspended procs | runtime | `heap_gc` + `scheduler` root set | **D** |
| ≤516 | not captured | Follow-up: the fix was ineffective when `Del()` had local or internal vars set to the reference being deleted; now "**the currently running proc's vars are always checked during a deletion before scanning scheduled procs**" | — | runtime | — | **D** |
| — | — | *(implication)* **The most important GC finding.** BYOND is not pure refcounting: deletion runs a **sweeper that scans for dangling references across the heap *and every suspended proc frame in the scheduler***. The scheduler is part of the GC root set, so `del` cost scales with the number of sleeping/spawned procs — precisely the SS13 workload shape. The optimizations are all about *bounding* that sweep: abort once the expected count is found, scan the likeliest holder (the current frame) first, and avoid the sweep entirely when the refcount says you are the last holder | — | runtime | Dream64's `heap_gc` rooted-operand slot and adaptive collector face the same coupling | **I** |

**Reading.** This category rewards close reading more than any other. A naive
model of BYOND — "it refcounts" — misses that `del` is a heap-and-scheduler
scan, and that fifteen years of release notes are spent making that scan bail
out earlier. For an engine targeting SS13-scale worlds with thousands of
suspended procs, that is the failure mode to design against from the start.

---

## 9. Scheduler / spawn / sleep

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 252 | not captured | `world.tick_lag` fixed to allow resolution **below 1.0**, enabling fast arcade-style games (e.g. `tick_lag` 0.1) | Tick granularity floor of 1/10 s | runtime | `dm-vm::execution::scheduler` | **D** |
| 318 | not captured | Fixed `sleep(X)` where `X < world.tick_lag` rounding `X` down to 0, "which caused uncontrolled speed under **the new timing system**"; the previous patch "had numerous problems with **timer backlogs**" | Sub-tick sleeps degenerating to busy execution; timer queue backlog | runtime | scheduler yield accounting | **D** |
| ~510 | not captured | New `world.tick_usage` var reports the percentage of the current server tick used; "except when this comes from a player command or instant event, **this happens before any maps are sent to the players**" | No in-language visibility into tick budget consumption | runtime | `dm-vm::profiling`, scheduler budgets | **D** |
| — | — | *(implication)* The server tick has a documented internal order: **DM execution runs first, map sending happens after**. `tick_usage` measures the DM portion against the whole tick, which is why overrunning it directly delays map delivery | — | runtime | Dream64's coordinator/commit ordering | **I** |
| guide | — | `sleep` and `spawn` may be used with no argument, giving the shortest possible delay (one tick) | — | runtime | scheduler | **D** |
| 516.1680 | 1680 | Local audit: the host loop pumps timer, socket, callback/state, rendering and bookkeeping work before scheduling the next timer; **a long DM dispatch that fails to return to this loop delays ping and socket service** even when background workers exist | Head-of-line blocking of network service by DM execution | runtime | scheduler step budgets + host services | **D-bin** |

---

## 10. Spatial / map representation

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 508 | not captured | "The number of **unique map cells** (distinct combination of **turf type, turf appearance, and area ID**) has been increased from **65,535 to a four-byte value**" | A 16-bit identity space for distinct map cells, exhausted by large or varied maps | startup + runtime | `dm-world` `PlannedCell` / template model; `dm-vm::execution::world_geometry` | **D** |
| 508 | — | *(implication)* **The single most valuable structural finding in the archive.** The map is stored as a grid of **interned cell IDs**, where a cell is the tuple *(turf type, turf appearance, area)*. Turfs are not independently allocated objects in the base representation — they are materialized from a shared, deduplicated cell template. This is the only model under which a 65,535 limit on *unique cells* (as opposed to turfs) is a sane engineering choice, and it explains why cells need their own garbage collection (below). It is a flattening + interning + lazy-materialization design for the largest object population in the world | — | startup + runtime | **See §23 — the highest-value comparison for Dream64's 289 s Mapping / 432 s SSatoms phases** | **I** |
| 511 | not captured | "Although the limit for unique map cells was increased in a previous version, a bug in the way that was handled caused errors to occur after crossing the **64K boundary**"; and map cells "did not go through garbage collection often enough when the list grew rapidly, causing large and avoidable memory increases" | Widening bug; unbounded growth of the interned cell table | runtime | cell table lifetime | **D** |
| 515 | not captured | `for(thing in list)` specialized for `range`, `orange`, `viewers`, `oviewers`, `hearers`, `ohearers`; previously only `view()` and `block()` (see §6) | Temporary list allocation per spatial query | compile + runtime | loop lowering — **gap** | **D** |
| ≤515 | not captured | The `view()` and `range()` families had a memory leak when the reference atom **straddled multiple tiles**; memory grew over time handling objects covering multiple tiles | Multi-tile (pixel-movement) atoms in spatial queries | runtime | `dm-vm::builtins::spatial` | **D** |
| 511 | not captured | With a very large number of objects per turf, map updates while moving caused client stuttering; optimizations to client **and** server plus "additional info in the message format" improved it | Per-turf content volume driving update cost and wire size | runtime | `dm-client` / map delta transport | **D** |

---

## 11. Map loading and startup

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 490.1113 | **1113** | "Worlds with **large maps** took a much longer time to startup than they should have" — fixed | Superlinear behaviour in world startup as map size grows | startup | `dm-world::allocation`, `dm-lifecycle` | **D** *(build from page title)* |
| 512 | not captured | "**Compiler speed has been greatly improved for large projects**" | Whole-project compile latency | compile | `dm-compiler`, `dm-project` | **D** |
| 512 | not captured | Object tree generation speed improved significantly (see §1) | startup/compile tree construction | compile / startup | `dm-object-tree` | **D** |
| 512 | not captured | Default var lookup reorganization improves object init speed (see §2) | Per-instance init cost at world load | startup | sparse defaults | **D** |
| 514 | not captured | Some compiler speedups intended for 515 were **backported to 514** while fixing bugs, "so large projects can reap some of those benefits early. Even more speedups appear in 515" | Compile latency on large projects | compile | `dm-compiler` | **D** |

**Reading.** Every documented BYOND startup win is in one of three places:
object-tree construction, per-instance default initialization, or large-map
handling. Dream64's measured profile (≈6.5 s to compile-complete, ≈5 s
materializing type/default metadata, then ~830 s dominated by DM-level subsystem
init) says the first two are already solved locally and the remaining cost is
*DM execution*, which BYOND addresses through §5 (opcode specialization) and §4
(init consolidation) rather than through load-time tricks.

---

## 12. Caching / precomputation

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 513 | not captured | "The server now **caches function lookups from external .dll/.so calls** and should process them faster" | Per-call symbol resolution across the extension boundary | runtime | extension boundary | **D** |
| 506 / 513 | not captured | The **string tree**: rebalancing weakness on deletion (506); strings containing ANSI or re-encoded malformed UTF-8 not added to the tree properly, and occasional tree corruption on Linux (513) | Interned-string storage and lookup | runtime | `dm-value` text storage | **D** |
| 282 | not captured | Object **appearance data shared between the client and server** "was getting **re-used each time the client needed to look at an object with exactly the same appearance**", but was not garbage collected, so unbounded unique appearances would exhaust server space | Per-object appearance storage and transmission | runtime | appearance interning; `dm-client` | **D** |
| 508 | not captured | Unique **map cell** interning widened to four bytes (see §10) | — | startup + runtime | — | **D** |
| 4.0 | 950 | As of BYOND 4.0, dynamic resources are stored in a **separate cache** (`[world]_dyn.rsc`) so the resource cache built with the world "remains constant"; dynamic resources persist between sessions per `world.cache_lifespan` | Runtime-generated resources polluting the immutable build-time resource file | startup | resource/cache boundary | **D** |
| ≤516 | not captured | Icons created at runtime are **purged from the `.dyn.rsc` cache** following some reboots and when the world starts up, preventing unbounded growth | Dynamic cache growth across sessions | startup | cache lifetime | **D** |
| 516.1680 | 1680 | Local audit: CRC-addressed cache lookup, resource-list availability negotiation, cache-read CRC validation, length/CRC metadata, gzip paths, and file-mapping imports | — | startup + runtime | `dm-mmap`, resource cache | **D-bin** |
| — | — | *(implication)* BYOND's caching philosophy is **content-addressed interning of derived aggregates** — appearances, map cells, strings, resources — each with its own identity space, its own widening history, and its own lifetime bug class. The recurring failure mode is never the cache hit; it is the **lifetime** of interned entries (§8, §10) | — | — | Dream64's appearance/DMI caches and world-plan cache inherit the same failure mode | **I** |

---

## 13. Client / server separation

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 282 | not captured | Shared client/server appearance data reused for identical appearances (see §12) — appearance interning documented as early as the 2.x/3.x era | Redundant appearance transmission and storage | runtime | `dm-client` appearance transport | **D** |
| 463.1065 | **1065** | **Nagle's algorithm disabled**, with internal queuing "to prevent a bunch of tiny packets from swamping the server's bandwidth"; "significantly improves the performance of every action-oriented game, and games that use pixel movement extensively" | Latency from TCP coalescing vs. bandwidth from tiny packets | runtime | IPC transport | **D** *(build from page title)* |
| 508 | not captured | "**DSification round 2**" of the webclient "takes a great deal of work off the server so it behaves more like a Dream Daemon to Dream Seeker connection. Server performance has been vastly improved" | Server doing presentation work on behalf of thin clients | runtime | `dm-client` protocol | **D** |
| 511 | not captured | A **structure alignment error** caused the server to believe objs and mobs known to the client had changed when they had not, "resulting in increased network traffic in some games" | False-positive change detection in the client-known-state diff | runtime | map delta / change detection | **D** |
| 511 | — | *(implication)* The server maintains a **per-client model of known object state** and diffs against it to decide what to send. The diff is sensitive enough to a byte-level comparison that struct padding corrupted it — consistent with comparing packed appearance/state records rather than field-by-field | — | runtime | Dream64's versioned, view-filtered map deltas | **I** |
| ≤514 | not captured | Access to resources from a locally hosted game is faster because "the client and server now **share the same memory objects** used to access the world rsc file" | Duplicate resource I/O and memory for co-hosted client/server | startup + runtime | `dm-mmap` | **D** |
| 516.1680 | 1680 | Local audit: `MapIconList` pipeline (`FillTargets`, `GroupPlanes`, `SetParents`, `SetBounds`, `Sort`, `SubSort`), a dedicated client-side hit pipeline, and render metadata naming `plane`, `subplane`, `visual_bounds`, `mouse_opacity`, `filters`, `render_target` | — | runtime | `dm-client` renderer | **D-bin** |

---

## 14. Internal indexing / data structures

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 506, 513 | not captured | The **string tree** — a balanced tree of interned strings, with documented rebalancing and corruption bugs | String identity and lookup | runtime | `dm-value` text storage | **D** |
| ≤513 | not captured | Datum limit **65,535 → over 16 million** | 16-bit datum index space | runtime | 64-bit handles | **D** |
| 508 | not captured | Unique map cells **65,535 → four-byte value** | 16-bit map-cell index space | startup + runtime | world plan cells | **D** |
| ≤511 | not captured | `client.bound` vars were limited to a **−32K…32K** range, "which was inappropriate for large maps and icons" | 16-bit signed pixel coordinates | runtime | `dm-client` | **D** |
| 515 | not captured | "Creation of a `ref()` string (**not including string tree lookup**) is an **order of magnitude faster** than before" | Formatting cost of reference encoding | runtime | `dm-value` handle formatting | **D** |
| 515 | — | *(implication)* `ref()` is a **formatted encoding of a tagged index** (type tag + table index), and the parenthetical concedes that the string-tree insertion of the resulting text remained the dominant residual cost. The order-of-magnitude win was in the formatting, not the identity lookup | — | runtime | Dream64's logical handles make this a formatting concern only | **I** |
| — | — | *(implication)* **The unifying thread of the archive.** BYOND's core identity spaces — datums, map cells, appearances, client bounds — were all originally **16-bit interning tables**, and a decade of releases is largely the story of widening them, each widening accompanied by a boundary bug at 64K (§10). Dream64's 64-bit handles skip this entire evolutionary era by construction | — | — | `dm-value` generational handles; "not limited to BYOND's 32-bit process address space" (ARCHITECTURE.md) | **I** |

---

## 15. Multithreading / parallelism

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 504 | not captured | An **experimental multithreading** feature was added for Dream Seeker, "which **decouples the server and client threads**, so that user interface interaction shouldn't slow down the server (or vice-versa) on most systems" | UI work and server work contending on one thread in the co-hosted client | runtime | `dm-client` / runtime split | **D** |
| ≤516 | not captured | The linux/bsd Dream Daemon accepts `-threads [on/off]` and `-map-threads [on/off]` to enable/disable all threading and map threading, overriding the config | — | runtime | deployment config | **D** |
| ≤516 | not captured | Threading was enabled for the first time in DreamDaemon for Linux (`threads off` / `map-threads off` in `daemon.txt` to disable); thread settings were later changed so that **threads are off by default for the stable build**; Linux threading **crashed some games on startup** | — | runtime | — | **D** |
| 516.1680 | 1680 | Local audit: `DungServer::SetSendMapsThreadCount` stores 0 for requests ≤ 1, else `min(requested, 64)`; `DungThreadManager` appends tasks under a lock and **invokes callbacks one at a time on the calling (owner) thread**; `DungThreadPool::WaitAll` uses bounded 1000 ms waits | — | runtime | `dm-vm::worker_lane`, `region_compile_worker` | **D-bin** |
| — | — | *(implication)* BYOND's parallelism is confined to **map sending** and **client/server decoupling**, behind a serialized owner-thread commit point, and it has been **off by default in stable builds** after destabilizing games. There is no documented concurrent DM datum execution and no parallel `Initialize`. The documented design is exactly Dream64's stated rule: workers prepare immutable data, the owner thread commits | — | runtime | ARCHITECTURE.md "Runtime model"; `worker_lane` | **I** |

---

## 16. World reboot / reload behavior

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| ≤515 | not captured | `Reboot` was resetting `world.visibility` and `world.channel` to defaults; it now **preserves the previous settings** | Reboot discarding live world configuration | startup | `dm-world` reboot path | **D** |
| ≤514 | not captured | Partially transmitted resource files in `byond.rsc` were not handled correctly until the client rebooted — reachable by logging out mid-transfer or by a transfer in flight during `world.Reboot()` | Torn resource state across a reboot boundary | startup | resource transfer | **D** |
| 4.0 | 950 | Dynamic resources persist between sessions per `world.cache_lifespan`; as of 4.0 they live in a separate `[world]_dyn.rsc` so the build-time cache stays constant | Reboot/reload invalidating the immutable resource set | startup | cache boundary | **D** |
| ≤516 | not captured | Runtime-created icons are purged from `.dyn.rsc` "following some reboots and when the world starts up" | Unbounded dynamic cache growth across reboots | startup | cache lifetime | **D** |
| ≤514 | not captured | A reboot or reconnect sometimes **scrambled icons**, "due to the client not properly unloading some internal icon info" | Client-side interned icon state surviving a reboot it should not have | startup | `dm-client` reset | **D** |
| — | — | *(implication)* Reboot is a **partial** reset: interned/derived state (resource cache, client icon tables, appearance IDs) deliberately survives it for speed, and most documented reboot bugs are cases where something survived that should not have, or was discarded that should not have been. Reboot correctness is a *cache-invalidation* problem, not a teardown problem | — | startup | relevant to any Dream64 reboot/hot-reload work | **I** |

---

## 17. Runtime profiling / diagnostics

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 282 | not captured | **Profiling introduced**: "Profile World" menu option when self-hosting, manual refresh for snapshots of world proc usage, and remote invocation via the `.debug profile` command with admin privileges | No visibility into which procs consume CPU | runtime | `dm-vm::profiling` | **D** |
| 513 | not captured | **`world.Profile()`** proc added, "makes it possible to programmatically interact with the server's profiler" | Profiling only drivable by hand from a UI or console | runtime | `dm-vm::profiling` | **D** |
| ~510 | not captured | `world.tick_usage` (see §9) | No in-language tick-budget visibility | runtime | scheduler telemetry | **D** |
| — | — | *(implication)* BYOND's profiler is a **per-proc CPU accounting** model sampled/aggregated by the server, exposed first as a UI snapshot (282) and only 30-odd builds later as a programmable API (513). It is proc-granular, not instruction-granular | — | runtime | Dream64's instruction-level boot/atom profiles are finer-grained than the documented BYOND facility | **I** |

---

## 18. Memory optimization

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | not captured | Default var lookup reorganization **reduces memory** and improves init speed (see §2) | Per-instance storage of inherited defaults | startup + runtime | sparse defaults | **D** |
| 511 | not captured | Map cells not garbage collected often enough when the list grew rapidly, "causing large and avoidable memory increases"; appearance IDs reused before client-side deletion leaked | Lifetime of interned aggregates | runtime | interning lifetimes | **D** |
| ≤514 | not captured | BYOND compiled **large address aware**, "allowing it to theoretically use closer to 4 GB rather than being limited to just under 2 GB" | 2 GB user-address ceiling on 32-bit Windows | — | Dream64 is 64-bit by construction | **D** |
| ≤515 | not captured | `view()`/`range()` leak with multi-tile reference atoms (see §10) | — | runtime | `builtins::spatial` | **D** |
| ≤513 | not captured | Memory leak in assignments of `atom.icon` to dynamically created icons | Dynamic icon lifetime | runtime | `dm-icon` | **D** |
| ≤509 | not captured | Object-based icon manipulation is "directly modifiable in memory, whereas the old style operators often resulted in extra loads from disk or at least copying of cached data in memory" | Disk re-reads and copies per icon operation | runtime | `dm-icon` | **D** |

---

## 19. Large-world scalability

| Ver | Build | Documented change | Bottleneck addressed | Phase | Dream64 subsystem | Conf |
| --- | --- | --- | --- | --- | --- | --- |
| 508 | not captured | Unique map cells 65,535 → four-byte | Map variety ceiling | startup + runtime | world plan | **D** |
| ≤513 | not captured | Datums 65,535 → 16M+ | Object population ceiling | runtime | 64-bit handles | **D** |
| 490.1113 | **1113** | Large-map world startup time | Startup scaling | startup | `dm-world` | **D** |
| ≤511 | not captured | `client.bound` ±32K inappropriate for large maps and icons | Coordinate range ceiling | runtime | `dm-client` | **D** |
| ≤514 | not captured | Large address aware (~4 GB) | Address-space ceiling | — | — | **D** |
| 511 | not captured | Very large object counts per turf slowing client updates | Per-turf content scaling | runtime | map delta transport | **D** |
| — | — | *(implication)* Every documented scalability limit BYOND hit was a **fixed-width identity or coordinate field**, not an algorithmic wall. The engine's algorithms scaled; its **encodings** did not. This is the strongest argument for Dream64's decision to make handles and indices wide from the start rather than optimizing them later | — | — | ARCHITECTURE.md "Values and memory" | **I** |

---

## 20. Documented startup optimization (consolidated)

Every startup-relevant entry in the archive, in one place:

| Ver | Build | Documented change | Phase | Conf |
| --- | --- | --- | --- | --- |
| 490.1113 | **1113** | Worlds with large maps took much longer to start up than they should have — fixed | startup | **D** |
| 512 | not captured | Compiler speed greatly improved for large projects | compile | **D** |
| 512 | not captured | Object tree generation was slower than it needed to be; significantly optimized | compile / startup | **D** |
| 512 | not captured | Default var lookup reorganized — reduces memory **and improves object init speed** | startup + runtime | **D** |
| 512 | not captured | Init procs consolidated by an analysis pass (which may no longer allocate live objects) | compile / startup | **D** |
| 514 | not captured | Compiler speedups backported from 515; more in 515 | compile | **D** |
| 514 | not captured | `istype()`/`typesof()` O(1) — requires a type index built at startup | startup (build) / runtime (query) | **D** |
| ≤516 | not captured | Runtime icons purged from `.dyn.rsc` at world start | startup | **D** |
| ≤516 | not captured | Linux threading crashed some games **on startup**; threads off by default in stable | startup | **D** |

**There is no documented BYOND optimization that parallelizes world
instantiation, batches atom construction, or defers `New()`/`Initialize()`.**
Every documented startup win is either compile-time (tree, compiler, init
consolidation) or a representation change that makes each instantiation cheaper
(default vars, type index, map cells). This is a negative result, and it is the
most decision-relevant finding in the document: it independently corroborates
the conclusion already reached in `docs/performance/boot-architecture-research.md`
from the binary side.

---

## 21. Chronological architecture timeline

Read this as the order in which BYOND learned to stop doing things generically.

### 2.x — foundations and the first indices

- **219p3–225** — earliest archived notes; global variable and map-size material.
- **224 / 243 / 252** — `world.tick_lag` gains sub-1.0 resolution (**252**), making
  the scheduler's tick granularity a tunable rather than a constant.
- **266 / 267 / 272 / 273** — spatial and resource-era fixes.
- **276** — a **code-generation optimizer** already exists (it has `switch()`
  bugs), and **constant folding** is already expected (constant subtraction was
  not being folded). *DM has had a compile-time optimizer since the 2.x era.*
- **282** — two structural firsts: **appearance interning** between client and
  server ("re-used each time the client needed … exactly the same appearance"),
  and the **profiler** (`Profile World`, `.debug profile`).

### 3.x — bounding the collector

- **305/306, 307b, 307–314** — the **sweeper** era. The collector is made to
  *assume the refcount is accurate and abort early*; then `del X` is taught to
  clear the variable first so the common case skips the sweep entirely.
  Built-in/user var slot-layout collisions are fixed. Icon loading, drawing and
  resource access are made more efficient.
- **318** — the "new timing system" is bedded in: sub-`tick_lag` sleeps no longer
  round to zero, after an earlier patch caused timer backlogs.

### 4.0 — the client rewrite, and dropping periodic GC

- **400.950** — OpenGL map drawing, arbitrary map scaling, skins, `.dmp`→`.dmm`,
  Dream Daemon overhaul, and the **split of dynamic resources into
  `[world]_dyn.rsc`** so the build-time resource cache stays immutable.
- **413.978** — the **5-minute garbage collection cycle is removed** as redundant
  against refcounting. GC becomes purely event-driven.
- **463.1065** — Nagle disabled with internal queuing; framed as a significant
  win for every action-oriented and pixel-movement game.
- **490.1113** — **large-map world startup** fixed.

### 5.0 / 50x — widening the identity spaces

- **501 / 504** — **experimental threading**: server/client thread decoupling in
  Dream Seeker; map-send threads; later off by default in stable after crashes.
- **506** — associative-list and **string-tree** performance; tree rebalancing on
  deletion.
- **507** — appearance/overlay and blend-mode correctness.
- **508** — **unique map cells widen from 65,535 to four bytes**; webclient
  "DSification round 2" moves work off the server; a GC change is reverted.
- **509 / 510** — `world.tick_usage` exposes the tick budget to DM.
- **511** — the 64K map-cell boundary bug; **map-cell GC frequency**; appearance
  ID reuse leak; per-turf object-count scaling; a struct-alignment bug inflating
  client update traffic.

### 51x — flattening the hot semantics

- **512** — the **startup release**: compiler speed for large projects; object
  tree generation; **default var lookup reorganized (memory + init speed)**;
  **init-proc consolidation analysis** constrained to be side-effect free; proc
  code optimizations (constant loads, var reads, proc data init).
- **513** — external `.dll`/`.so` **function-lookup caching**; compile-time type
  inference for chained `.`/`?.` to avoid the colon operator's *downward*
  lookup; `world.Profile()`; string-tree encoding and Linux corruption fixes.
- **514** — **`istype()`/`typesof()` become O(1)** (mob `typesof()` excepted);
  compiler speedups backported from 515.
- **515** — `for(x in …)` **generator fusion extended** from `view()`/`block()` to
  `range`, `orange`, `viewers`, `oviewers`, `hearers`, `ohearers`; `ref()`
  creation an order of magnitude faster; `Byond_CallProc` moved off verbified-name
  resolution onto `call()`'s better method; stricter runtime errors for invalid
  list var access in 515-compiled worlds.
- **516** — **`alist()`** splits the assoc-only case out of the hybrid list;
  `for(item,value in list)` fuses the value fetch; fast numeric assoc-list procs;
  `~=`/`~!` become assoc-aware; `Byond_ReadListAssoc` batches the extension
  boundary; `byondStorage`/`BYOND.winset`/`winget` JS APIs.

---

## 22. Synthesis — what BYOND actually learned to stop interpreting

Eight patterns account for essentially every architecture-relevant entry above.
This is the answer to the question the brief asked.

1. **Intern every derived aggregate, then fight its lifetime forever.**
   Appearances (282), strings (506/513), map cells (508), resources (4.0). The
   cache hit was never the problem; **every** one of these later produced a
   lifetime bug — appearance IDs reused too early (511), map cells not collected
   often enough (511), string-tree imbalance on delete (506). *If Dream64 interns
   an aggregate, the collection policy is part of the feature, not a follow-up.*

2. **Flatten the map out of the object model.** The map is a grid of interned
   *(turf type, turf appearance, area)* cell IDs — not a grid of allocated turfs.
   This is the largest single representational departure from a naive DM
   implementation, and it is the reason a 65,535 *unique cell* ceiling was ever
   tolerable. **(Inference, from the 508/511 wording.)**

3. **Replace tree walks with precomputed indices.** `istype()`/`typesof()` went
   from O(depth) to O(1) in 514 by building the index once. The mob exception
   shows where the flat index didn't reach.

4. **Move defaults out of instances.** 512's default-var reorganization bought
   memory *and* init speed at once — the signature of moving inherited defaults
   into a shared per-type table with copy-on-write slots.

5. **Fold initialization at compile time.** 512's "analysis of init procs to
   consolidate them" is an explicit compile/startup pass that merges per-type
   initializers — with the hard-won constraint that the folding pass must not
   allocate live objects.

6. **Fuse generators into loops.** `for(x in view())`, then in 515 the whole
   `range`/`orange`/`viewers`/`oviewers`/`hearers`/`ohearers` family, and in 516
   `for(item,value in list)`. BYOND attacks the hottest DM idiom by never
   materializing the intermediate list at all.

7. **Specialize opcodes; never JIT.** A codegen optimizer since 276, constant
   folding, 512's constant-load/var-read/frame-init work, and a ~400-entry
   handler table with dedicated context-cache opcodes at 516. Across 20+ years
   the answer is always *make the opcode do more*, never *emit machine code*.

8. **Keep parallelism off the DM heap.** Threading exists only for map sending
   and client/server decoupling, behind a serialized owner-thread commit, and
   was turned **off by default** in stable builds after destabilizing games.
   There is no documented parallel atom initialization.

**The negative results matter as much as the positive ones.** Across the entire
archive there is *no* documented: JIT or native code generation; parallel world
instantiation or bulk atom construction; deferred or batched `New()`/`Initialize()`;
or tracing/generational garbage collection. BYOND's speed is constant-factor
excellence in a scalar native interpreter over aggressively flattened data — not
a cleverer execution model.

---

## 23. Dream64 correspondence and gaps

Correspondences were checked by targeted inspection of this repository, not by a
full audit. "Present" means the mechanism was located; it does not assert
equivalent coverage or performance.

### Already held

| BYOND finding | Dream64 counterpart | Evidence |
| --- | --- | --- |
| 514 O(1) `istype()`/`typesof()` | `subtype_interval(&self, path) -> Option<(u32,u32)>` — interval containment over the type tree | `crates/dm-vm/src/execution/type_metadata.rs:115` |
| 512 default-var lookup reorganization | "sparse inherited scalar defaults"; overrides allocated on write | `crates/dm-vm/src/execution/state.rs:428` |
| 314 built-in/user var slot merging | Interned field-name layouts and physical field-index allocations | `crates/dm-value/src/lib.rs` (`intern`, `intern_unindexed_names`) |
| 512 proc-code specialization | `compact_wordcode` — one 32-bit word per instruction, 8-bit selector + 24-bit operand, with an escape back to the reference path | `crates/dm-vm/src/compact_wordcode.rs` |
| 512 init-proc consolidation | `dm-lifecycle` non-executing `initialization_plan` / `precompile` — the side-effect-free folding boundary BYOND had to retrofit in 512 | `crates/dm-lifecycle/src/` |
| 4.0 immutable vs dynamic resource caches | Separately persisted cache boundaries; CRC/digest-addressed lookup | `dm-mmap`, world-plan cache |
| Threading confined to non-heap work | `worker_lane` (immutable snapshots, owner-thread commit), `region_compile_worker` | `crates/dm-vm/src/worker_lane.rs` |
| Identity-space widening (65,535 → …) | 64-bit host, logical generational handles, no 32-bit address dependence | `ARCHITECTURE.md`, `dm-value` |
| 282/513 profiling | Instruction-level boot/atom/TGM profiles — finer-grained than the documented BYOND profiler | `crates/dm-vm/src/profiling.rs` |
| (beyond BYOND) | `dm-jit` Cranelift tiers + region JIT — **no BYOND analogue is documented at any version** | `crates/dm-jit/` |

### Gaps worth acting on

1. **Spatial generator fusion — `for(x in range/orange/viewers/oviewers/hearers/ohearers)`.**
   BYOND fused `view()`/`block()` early and extended it to the full spatial family
   in 515. A grep of `crates/dm-vm/src/compile_stmt.rs` found loop lowering only
   for numeric `for(i in a to b)` ranges; no generator fusion for the spatial
   families was located. This is the most concrete, bounded, directly-transferable
   optimization in the whole archive, and it targets exactly the idiom SS13-family
   code runs constantly. **Verify, then implement.**

2. **Runtime map-cell interning.** `dm-world` already splits a plan into
   `templates` + `PlannedCell`s with `allocate_template` / `link_cell_locations`,
   which is plan-level deduplication. What is *not* established is whether the
   **runtime** map retains interned cell identity the way BYOND's four-byte cell
   IDs imply, or materializes independent per-turf state. Given that Mapping
   (~289 s) and SSatoms (~432 s) dominate the measured 830 s boot, this is the
   highest-value thing in this document to check next.

3. **GC/scheduler coupling.** BYOND's hard-won lessons — bail out of the sweep
   once the expected number of dangling refs is found; scan the *currently
   running* frame before scheduled procs; skip the sweep entirely when the
   refcount says you hold the last reference — are all about keeping deletion
   cost from scaling with the number of suspended procs. Worth auditing
   `heap_gc` against all three, since SS13 boots with very large numbers of
   sleeping/spawned procs.

4. **Assoc-only list representation.** 516's `alist()` concedes the hybrid
   list is too expensive when only key-value behaviour is used. Dream64 can make
   the same split **internally**, keyed on observed usage, without adding a
   DM-visible type.

5. **Transactional appearance-list assignment.** Wholesale assignment to
   `overlays`/`underlays`/`filters`/`verbs` must derive one final appearance, not
   one per element. BYOND's fix names "appearance churn" explicitly.

6. **`initial()` ↔ savefile delta coupling.** BYOND's `bounds`/`bound_x` bug
   shows savefile serialization diffs against the type default. Any Dream64
   savefile writer must key on the same table the sparse-default path serves, or
   it will silently write unchanged vars.

### Open questions this archive cannot settle

- Whether BYOND's type index is interval-based, bitset-based, or something else,
  and what makes mob `typesof()` resist it.
- Whether map cells are materialized lazily per access or eagerly at load.
- Whether the "consolidated" init proc is a single fused routine per type or a
  folded constant-initializer table.
- The actual internal proc-identity representation behind the "verbified name"
  wording in 515.

All four need differential black-box oracles, not more reading.

---

## 24. Sources

Official BYOND release-note archive (canonical location; **not reachable from
this environment** — see §0.2, retrieved as search excerpts of these pages):

- [Release notes index — `byond.com/docs/notes/`](https://www.byond.com/docs/notes/516.html) and per-version pages:
  [225](https://www.byond.com/docs/notes/225.html) ·
  [252](https://secure.byond.com/docs/notes/252.html) ·
  [276](https://secure.byond.com/docs/notes/276.html) ·
  [282](https://secure.byond.com/docs/notes/282.html) ·
  [306](https://www.byond.com/docs/notes/306.html) ·
  [307b](https://secure.byond.com/docs/notes/307b.html) ·
  [307–314](http://www.byond.com/docs/notes/314.html) ·
  [318](https://www.byond.com/docs/notes/318.html) ·
  [400.950](https://www.byond.com/docs/notes/400.html) ·
  [413.978](https://secure.byond.com/docs/notes/413.html) ·
  [463.1065](https://secure.byond.com/docs/notes/463.html) ·
  [490.1113](https://secure.byond.com/docs/notes/490.html) ·
  [504](https://secure.byond.com/docs/notes/504.html) ·
  [506](http://www.byond.com/docs/notes/506.html) ·
  [507](http://www.byond.com/docs/notes/507.html) ·
  [508](https://secure.byond.com/docs/notes/508.html) ·
  [509](https://secure.byond.com/docs/notes/509.html) ·
  [510](http://www.byond.com/docs/notes/510.html) ·
  [511](http://www.byond.com/docs/notes/511.html) ·
  [512](https://secure.byond.com/docs/notes/512.html) ·
  [513](http://www.byond.com/docs/notes/513.html) ·
  [514](https://www.byond.com/docs/notes/514.html) ·
  [515](https://www.byond.com/docs/notes/515.html) ·
  [516](https://secure.byond.com/docs/notes/516.html)
- [BYOND build archive index](https://www.byond.com/download/build/)
- [DM Reference](https://secure.byond.com/docs/ref/info.html) (garbage-collection
  and `ref` semantics)
- [DM Guide](https://www.byond.com/docs/guide/chap13.html) (`sleep`/`spawn`/`tick_lag`)

Local primary evidence:

- [`docs/reverse-engineering/byond-516.1680-dll-audit.md`](./byond-516.1680-dll-audit.md) — read-only static audit of locally installed 516.1680 binaries
- [`docs/performance/boot-architecture-research.md`](../performance/boot-architecture-research.md) — measured Dream64 boot baseline

Community mirror consulted (documentation only, no release notes):

- [`F0lak/dm_open_ref`](https://github.com/F0lak/dm_open_ref) — community DM Reference mirror, tracking 516.1681

---

## 25. How to complete this document

If the archive becomes reachable, the work is mechanical and bounded:

1. Fetch `https://www.byond.com/docs/notes/<v>.html` for every version in §0.1.
2. For each entry already recorded here, fill the `build` column from the page body.
3. Sweep for architecture-relevant entries this excerpt-based pass missed —
   coverage of 500–511 and the 4.0 series is thinner than 512–516, and the
   2.x/3.x pages were sampled rather than read end to end.
4. Re-check the two attributions flagged as contested or approximate: the 514/515
   `istype()` placement (§0.2) and the `world.tick_usage` introduction version (§9).

The version-level attributions and the analysis in §§21–23 should not need to
change; they rest on the documented wording, not on the build numbers.
