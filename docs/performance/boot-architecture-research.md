# Dream64 cold-boot architecture research

This note records the evidence used to choose the next boot-performance work.
It separates techniques that can change Dream64's measured critical path from
optimizations that are already below the noise floor.

## Measured Dream64 baseline

The current complete cached-artifact boot reaches headless-ready in 830.833
seconds (13:50.833). Peak working set is about 3.71 GB and peak private memory
is about 3.83 GB. A fixed-300 scheduler run takes 121.935 seconds and a
fixed-1000 run takes 315.983 seconds.

Release-layout experiments on the same fixed-300 gate produced:

- default Cargo release: 121.935 seconds;
- one-codegen-unit ThinLTO: 118.596 seconds (2.74% faster);
- ThinLTO plus an LLVM instrumentation profile trained on the fixed-300
  MonkeStation boot: 115.739 seconds (5.08% faster than the original).

The fixed-300-trained PGO binary also completed the untrained fixed-1000 gate
in 311.733 seconds versus the 315.983-second baseline (1.35% faster). The
smaller later-stage win shows that profile coverage matters; release training
should eventually include later subsystem startup rather than only the first
300 scheduler rounds.

The dense-global-slot experiment took 122.642 seconds and was rejected. This
confirms that a sampled hot global lookup is not automatically a whole-boot
lever. PGO is retained as a release multiplier, but a five-minute boot still
requires broad DM JIT/quickening coverage: static optimization alone is nowhere
near the required 2.77x improvement.

The cached preparation path is no longer the dominant problem:

- project validation, artifact checksum, parallel section decode, and registry
  construction reach compile-complete in about 6.5 seconds;
- a cached world-plan lookup is about 63 ms;
- runtime variable/type/default/initializer metadata materialization is about
  5 seconds;
- the production world then executes 145,139 lifecycle events before subsystem
  startup;
- observed long subsystem phases include SSatoms (~432 s), Mapping (~289 s),
  shuttle (~76.6 s), lighting (~69.6 s), icon smoothing (~66.2 s), and air
  (~45.5 s). These timings can contain nested work and must not simply be
  summed.

The atom profile recorded approximately 2.61 million list reads, 1.02 million
list writes, 6.16 million field reads, 1.07 million field writes, 4.25 million
calls, 11.57 million branches, and 50.26 million other bytecodes. The five
minute target therefore requires about a 2.77x whole-boot improvement. Small
host helpers cannot produce that result; the broad DM execution tier must get
materially faster.

## What the reference engines actually do

### BYOND 516.1680

The local read-only DLL audit found a scalar `Byond_New`/`Byond_NewArglist`
gateway. Off-owner-thread calls are marshalled through the thread manager and
wait for the authoritative thread. No public bulk atom constructor or worker
fan-out was found. This supports parallel immutable preparation, not concurrent
mutation of the live DM heap. The audit also found separately persisted DB,
map, and resource/cache boundaries, CRC-addressed cache lookup, and file-mapping
imports. See [the local BYOND audit](../reverse-engineering/byond-516.1680-dll-audit.md).

The important inference is that BYOND's speed is consistent with a compact,
highly optimized native scalar VM/object model. The DLL evidence does not
support a hidden multicore `/atom/Initialize` shortcut.

### OpenDream

OpenDream also loads mapped areas/turfs first, then serially calls `New()` on
areas and turfs and creates mapped objects/mobs in observable order. It does
not contain a parallel live-atom initialization pass. Useful transferable
techniques in its runtime include:

- skipping proc/thread setup when both initialization and `New` are null;
- pooling DM proc states, initialization states, stacks, and lists;
- retaining type defaults centrally and allocating an instance override
  dictionary only after a write;
- caching repeated turf/area appearance combinations.

Dream64 already has equivalents for several of these: sparse inherited scalar
defaults, shared datum layouts, frame/list optimizations, and appearance/DMI
caches. OpenDream is valuable as a semantic cross-check, but it does not expose
a missing order-of-magnitude startup mechanism.

Primary source:

- https://github.com/OpenDreamProject/OpenDream/blob/master/OpenDreamRuntime/Map/DreamMapManager.cs
- https://github.com/OpenDreamProject/OpenDream/blob/master/OpenDreamRuntime/Objects/DreamObject.cs
- https://github.com/OpenDreamProject/OpenDream/blob/master/OpenDreamRuntime/Procs/DMProc.cs

### RobustToolbox

RobustToolbox is a native C# entity-component engine, not a DM interpreter.
Its prototype loader parses YAML files with PLINQ and instantiates prototype
groups with `Parallel.For`. Its map/entity deserializer deliberately separates
read, allocate, component load, hierarchy construction, start, and map-init
phases. Its serialization generator emits direct constructors and copy/read
code for data definitions.

That staged shape is useful, but the speed is not directly comparable to
MonkeStation under DM. Robust does not run the same arbitrary DM constructor
and `Initialize` code for every entity. Dream64 can copy parallel parsing,
generated/static copy paths, and allocate/load/start staging only where the
preparation is immutable and the final commit preserves DM order.

Primary source:

- https://github.com/space-wizards/RobustToolbox/blob/master/Robust.Shared/Prototypes/PrototypeManager.YamlLoad.cs
- https://github.com/space-wizards/RobustToolbox/blob/master/Robust.Shared/Prototypes/PrototypeManager.cs
- https://github.com/space-wizards/RobustToolbox/blob/master/Robust.Shared/EntitySerialization/EntityDeserializer.cs
- https://github.com/space-wizards/RobustToolbox/blob/master/Robust.Serialization.Generator/Generator.cs

## Ranked changes

### 1. Build a real adaptive Tier 1 and broad baseline JIT

This is the only change class with enough reach to plausibly close most of the
five-minute gap.

Dream64's current Cranelift path recognizes a few numeric or exact rooted-list
traces. Production profiles show that most startup time remains in general
field, list, call, branch, and stack bytecodes. Exact Rust helpers and global
selector caches have repeatedly failed benchmarks because their guard and
handoff overhead is paid inside the hot loop while too little dispatch work is
removed.

The replacement should be tiered:

1. Give each program a compact mutable execution form and a PC-local cache
   sidecar. Do not add a process-global hash lookup to every opcode.
2. Start generic instructions with small saturating counters. Rewrite hot
   instructions to guarded specialized forms, and de-specialize cheaply on a
   shape/type/version miss.
3. First families: declared field load/store, global load/store, resolved
   calls, list index/length/iteration, type predicates, and common local/branch
   superinstructions.
4. At hot calls or loop backedges, compile the whole supported procedure or
   basic-block region with Cranelift. Native code keeps locals/stack state in
   registers, performs cheap shape guards, and calls a stable Rust slow-path ABI
   for complex DM operations.
5. Every loop backedge, call, yield, and slow-path exit remains a scheduler
   safepoint. Native execution must charge the exact logical instruction count
   and return an exact continuation PC.
6. Measure executed-bytecode coverage, compile latency, IC hit/miss/deopt rate,
   and time in native versus slow paths. Procedure-count coverage is not a
   useful success metric.

This mirrors established dynamic-language designs. CPython's adaptive
interpreter rewrites individual bytecodes and stores caches inline; its own
guidance emphasizes minimal branches and pointer chasing. SpiderMonkey uses a
baseline interpreter with IC stubs and then a fast baseline compiler that
removes dispatch while calling C++ for complex operations. Deegen demonstrates
that a two-tier interpreter/baseline-JIT design can have negligible startup
delay when specialization, ICs, register pinning, and hot/cold splitting are
designed together.

Primary source:

- https://peps.python.org/pep-0659/
- https://github.com/python/cpython/blob/main/InternalDocs/interpreter.md
- https://firefox-source-docs.mozilla.org/js/how-we-optimize.html
- https://arxiv.org/abs/2411.11469
- https://arxiv.org/abs/2201.09268

Expected impact: high, plausibly 2-4x in VM-heavy regions, but not guaranteed.
The fixed-300 and fixed-1000 gates must prove it before a full boot.

### 2. Replace names in hot operations with dense IDs and guarded slots

Dream64 already shares field-name layouts across matching datums, but compiled
declared-field operations still carry `FieldName` values and ultimately perform
linear or hash lookup. Globals are also name-keyed in runtime maps.

At link time, assign dense `FieldId`, `GlobalId`, and layout/shape identities.
A datum shape should map a declared field ID to a value slot. A quickened field
instruction then checks one integer shape/version and directly reads or writes
the cached slot. Dynamic `vars`, string-based field access, additions, and
deletions remain on the generic path and transition/invalidate the shape.
Globals should be a dense value array indexed by `GlobalId`, with name maps
retained only for reflection/dynamic access.

V8 uses the same basic split: objects share a hidden class describing property
locations, fast properties are array slots, and dictionary properties are the
slow dynamic fallback.

Primary source:

- https://v8.dev/blog/fast-properties
- https://v8.dev/docs/hidden-classes

Expected impact: medium-to-high for the measured 7.2 million atom-phase field
operations, and it also makes baseline JIT guards cheap. It is unlikely to hit
five minutes alone.

### 3. Split compiler state from the runtime artifact and lazily decode bodies

The current `.d64` is about 650 MB. A measured cold artifact write contained
about 552 MB of frontend/compiler data and only about 94 MB of executable data;
roughly 85% of the file is compiler-side state. The server maps the combined
artifact and decodes the frontend even on a cache hit.

Create two independently versioned artifacts:

- a compiler database used for incremental source rebuilds;
- a runtime image containing structural metadata, compact executable bytecode,
  map-plan identity, and resources needed by the server.

Keep procedure bodies independently addressable and decode only startup-
reachable bodies. Precompiled/native pages can be memory-mapped and faulted in
on demand. Wasmtime uses the same principles: precompilation removes compilation
from startup and lazy `mmap` lowers resident memory; copy-on-write images and
pre-resolved imports reduce instantiation work.

Primary source:

- https://docs.wasmtime.dev/examples-pre-compiling-wasm.html
- https://docs.wasmtime.dev/examples-fast-instantiation.html

Expected impact: high RAM reduction (hundreds of MB of backing and potentially
more decoded heap), low direct time reduction because current parallel decode
is only several seconds. This should run in parallel with Tier-1 work, not be
mistaken for the five-minute solution.

### 4. Add a content-addressed startup snapshot at a defined ready boundary

A snapshot can make repeated development boots dramatically faster by
restoring an already initialized heap rather than replaying deterministic work.
V8 reports this pattern reducing context initialization from 40 ms to under 2
ms on desktop in its example.

Dream64 cannot safely snapshot arbitrary live startup today. Time, RNG, config,
database handles, sockets, filesystem state, worker queues, and client/resource
state must be explicitly excluded or rebound. The snapshot key must include the
engine semantics version, executable digest, map digest, configuration digest,
and any content affecting initialization. The initial implementation should
snapshot only a documented deterministic boundary and verify restored state
against a normal boot oracle.

Primary source:

- https://v8.dev/blog/custom-startup-snapshots

Expected impact: very high for unchanged development/restart boots, no benefit
for the first boot of new content, and high correctness complexity.

Implementation began after the 830.833-second full-ready run proved that the
remaining target could not be reached by compile/map-plan work. The value heap
now has a pointer-free, versioned binary section with bounded decoding. Capture
streams directly from live arenas instead of cloning the full graph, and restore
preserves datum/list slot indices, generations, aliases, mixed-list order, and
free-list allocation order while rebuilding derived indexes. The remaining
sections are globals and runtime indexes, procedure statics, scheduler
continuations, and explicit rebinding of ephemeral host state.

### 5. Parallelize only immutable preparation with enough work per job

RobustToolbox proves parallel parsing/prototype instantiation can help native
content preparation. BYOND and OpenDream both reinforce an authoritative live
DM execution thread. Dream64 should use workers for source/resource decode,
hashing, decompression, icon analysis, static initializer planning, and JIT
compilation. Workers return immutable results; the owner thread commits them in
stable order.

Do not parallelize coordinate arithmetic merely because it is available: the
existing Dream64 planning benchmark measured about 113.8 ms sequential versus
189 ms parallel, so task overhead made that specific work slower. Do not run
live atom constructors, global/list mutation, signals, RNG, or lifecycle procs
concurrently unless the DM language model itself gains deterministic isolation.

Expected impact: selective. It can hide preparation and JIT compile latency,
but it cannot replace faster scalar DM execution.

## Rejected directions

- **LLVM versus Cranelift as a slogan:** JIT is an execution strategy, while
  Cranelift and LLVM are compiler backends. Cranelift is appropriate for a
  low-latency baseline tier; changing backend without broadening bytecode
  coverage does not improve startup.
- **More one-procedure fusions:** repeated fixed-300/fixed-1000 experiments
  were neutral or slower. A fusion is acceptable only when generated from a
  general specialization framework and its guard/handoff cost is measured.
- **Parallel live atom initialization:** contradicts observed owner-thread
  construction boundaries and risks datum IDs, globals, lists, RNG, signals,
  and constructor order.
- **Codebase-size-only cache invalidation:** two different source trees can have
  the same byte count. Keep metadata as a cheap prefilter and a strong digest as
  the artifact identity.
- **Optimizing compile/map-plan setup:** together these are seconds, not the
  missing 531 seconds.

## Implementation order and gates

1. Keep one-codegen-unit ThinLTO for release builds and train release PGO on a
   representative startup corpus. The fixed-300 profile is proven; fixed-1000
   and full-ready gates must check that it does not overfit early startup.
2. Retain the procedure-dump diagnostic and add per-PC hotness/IC telemetry with
   effectively zero cost when disabled.
3. Introduce a PC-local quickening sidecar without changing semantics;
   specialize declared field and resolved-call families first. Do not revive
   the rejected second global-index design without new evidence.
4. Build a broad Cranelift baseline region ABI with exact continuation and
   budget accounting; start with the production `RegisterSignal`, `_tgm_load`,
   and `build_coordinate` shapes but generate it from general op families.
5. Split the compiler database from the runtime artifact and make executable
   bodies independently addressable.
6. Gate every step: 562 VM tests, fixed-300, fixed-1000, then one full cold boot.
   Reject any candidate that regresses the shorter gates beyond normal noise.
7. Build the ready-world image in independently versioned sections. The heap
   section is implemented; add globals/runtime indexes and scheduler
   continuations next. Never persist clients, wall-clock `Instant` values,
   external jobs, profiling state, or process-local JIT/dispatch caches.
8. Key the completed image by engine semantics, executable, map, and startup
   configuration digests. Compare a restored-state oracle against a normal
   boot before making snapshot restore the default warm path.

The five-minute goal remains ambitious but technically plausible only if Tier 1
removes dispatch/name-resolution overhead across most executed startup
bytecodes. The research does not support another hidden BYOND/OpenDream switch
that makes serial DM initialization disappear.
