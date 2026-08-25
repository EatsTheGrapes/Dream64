# BYOND 516.1680 DLL compatibility audit

This document records a read-only static-analysis cross-reference for the
locally installed BYOND 516.1680 binaries. Dream64 does not load, redistribute,
patch, or depend on these files at runtime. Addresses are build-specific RVAs,
not stable APIs.

## Binary identity

| File | Size | SHA-256 | Architecture |
| --- | ---: | --- | --- |
| `byondwin.dll` | 2,720,768 | `2DB6E55993AEEC02107642C4FABB20ED65B4FDA2B7A5D57A04AB3E2C9E929972` | PE32 x86 |
| `byondcore.dll` | 4,477,952 | `E96396D32BF0DB945F27C8692F90F2C104B4573B83992983AA5762B6122A025B` | PE32 x86 |
| `byondext.dll` | 1,976,320 | `1EE8DD4DF96F8EAAC4C021E03181DC821911F6B3B33818296788DC9516AEE7D5` | PE32 x86 |

`byondwin.dll` exposes 4,254 named exports and imports 1,715 symbols.
`byondcore.dll` exposes 2,826 symbols. The unusually rich decorated C++ export
tables retain enough names to establish subsystem boundaries without inferring
private packet layouts.

## Architectural conclusions

### DMF model and native controls

The abstract DMF model lives in `byondcore.dll`. Evidence includes `IWindow`,
`IWindowElem`, `ParseParams`, `GetParams`, `SetParams`, `ListParams`,
`FormatXML`, `string2type`, and `type2elem`. Skin lifecycle exports include
`GetSkinFile`, `SetLocalSkinPath`, `SkinInfo_CIO`, and `SkinLoaded_CIO`.

`byondwin.dll` owns native adapters over those typed elements:

- `CICtrl` stores and updates a `UIElem`.
- `CIWindowCtrl` adds position, visibility, focus, context-menu, event, and
  synchronization behavior.
- `CIBrowserCtrl` consumes `UIBrowserElem` and owns a WebView2 surface.
- `CIOutputCtrl` consumes `UIOutputElem` and derives from rich-edit and
  `DMTextPrinter` machinery.
- `CIMapCtrl` consumes `UIMapElem` and owns map-specific viewport behavior.

This supports Dream64's independent `ControlTree` plus typed client-adapter
design. Authoritative property updates should be applied centrally, then
dispatched to a control-specific surface while suppressing feedback events.

### Browser

Relevant `byondwin.dll` RVAs include:

- `CIBrowserCtrl` constructor: `0x000D6780`
- `CIBrowserCtrl::PutFile`: `0x000D6CA0`
- `CIBrowserCtrl::PutText`: `0x000D6D00`
- `CIBrowserCtrl::Update`: `0x000D7360`
- `CVHTMLCtrl2::HandleByondUrls`: `0x000AD840`
- `CVHTMLCtrl2::InvokeScript`: `0x000ADE10`
- `CVHTMLCtrl2::Navigate`: `0x000ADFD0`
- `CVHTMLCtrl2::OnWebResourceRequested`: `0x000B04A0`
- `CVHTMLCtrl2::OnWebMessageReceived`: `0x000AFFF0`

The DLL directly imports
`WebView2Loader.dll!CreateCoreWebView2EnvironmentWithOptions` and calls core
resource helpers such as `GetRscFile`, `LookupUrlResource`, and `ParseUrl`.
Browser navigation is asynchronous. Browse content must remain separate from
OUTPUT text, and resources must be registered before dependent HTML is loaded.

Dream64 currently preserves ordered `BrowseResource` then `Browse` delivery.
The next fidelity step is a scoped WebView resource-request handler for nested
CSS, fonts, scripts, and dynamic URLs instead of permanent HTML/data-URI
rewriting.

### Output

Relevant `byondwin.dll` RVAs include:

- `CIOutputCtrl` constructor: `0x000F6650`
- `CIOutputCtrl::PutObject`: `0x000F6DD0`
- `CIOutputCtrl::PutText`: `0x000F6F00`
- `CIOutputCtrl::Update`: `0x000F7570`
- `CIOutputCtrl::UpdatePrinter`: `0x000F7880`
- `CIOutputCtrl::GetLines`: `0x000F69A0`
- `CIOutputCtrl::SetLines`: `0x000F7410`

RTTI/vtables explicitly show `CRichEditCtrl` and `DMTextPrinter` bases.
Dream64 is correct to keep OUTPUT separate from WebView. Remaining work includes
rich text, links, embedded images/objects, and style properties.

`SetLines` stores the supplied integer in the output printer state and tail-calls
the rich-edit line-limit routine; `GetLines` reads the effective value back from
that same printer. Dream64 now applies the effective per-control DMF `lines`
property, including runtime `winset` overrides, when trimming retained OUTPUT
history. A bounded 512-line fallback remains for skins that omit the property.

### Map

Relevant exports include:

- `CIMapCtrl` constructor: `0x000EEE40`
- `CIMapCtrl::Update`: `0x000F03D0`
- tile size: `0x000F01E0`
- icon size: `0x000EFFF0`
- letterbox: `0x000F0020`
- style: `0x000F0100`
- text mode: `0x000F01C0`
- zoom/mode: `0x000F0270` / `0x000F02A0`
- `DungClient::GetMapIcons` in core: `0x00088980`

The core `MapIconList` pipeline exposes `FillTargets`, `GroupPlanes`,
`SetParents`, `Sort`, and `SubSort`. Strings identify staged pane/view/visibility,
map-tile, screen-tile, populate, group/bounds, sort, particle, and wrap-up work.
Dream64 should transport world and screen appearances with parent, plane, layer,
stable-order, and hit-target data, then sort/render client-side. Screen objects,
particles, filters, and complete hit testing remain compatibility gaps.

#### Renderer and pointer-picking deep dive

A second read-only pass over the same 516.1680 binaries found a more explicit
division of rendering work. These decorated exports are direct evidence of the
operations and data boundaries; they do not establish private structure layouts
or exact comparison formulas:

- `MapIconList::FillTargets`, `GroupPlanes`, `SetParents`, `SetBounds`,
  `SetIconBounds`, `PlaneSourceBounds`, `Sort`, `SubSort`, and
  `GroupChildrenAt` show that bounds/parent/plane construction precedes final
  ordering. The trace strings independently order the stages as map tiles,
  screen tiles, population, grouping and bounds, sorting, particle waiting, and
  rendering.
- `PixelBounds::TransformDisplay`, `Contains`, `ContainsInclusive`,
  `Intersects`, `Offset`, and `Hash`, plus
  `MapIconContext::PrecalcBoundsRotation` and `RotateOffset`, show that picking
  and clipping use transformed appearance bounds rather than only the owning
  turf's icon-size square.
- `GraphicsWnd::HitCid`, `HitCidList`, `SubHitCid`, `SubHitCidList`, and
  `ScreenEdge`, together with `IconList::Hit`, form a dedicated client-side hit
  pipeline. `DungClient::GetCidMouseOpacity`, `GenClickCommand`, and the
  `GenMouse*Command` family then construct DM mouse commands from the selected
  target.
- `MapTextLink::HitTest`, `MapTextLink::Add(PixelBounds const&)`, and
  `MapIconList::AddMapTextLinks` establish a separate link-hit layer for
  maptext; alpha/icon picking alone is insufficient.
- `CIMapCtrl::SetTileSize`, `SetZoom`, and `SetZoomMode`, and the core DMF
  parameter string `zoom;letterbox;zoom-mode`, confirm that map display
  transforms are control state. Drawing and hit-testing must consume the same
  effective transform.
- Core render metadata strings name `plane`, `subplane`, `is_plane`,
  `is_group`, `screen_object`, `screen_edge`, `screen_x_percent`,
  `screen_y_percent`, `relative_width`, `relative_height`, `visual_bounds`,
  `content_visual_bounds`, `physical_bounds`, `mouse_opacity`, `filters`, and
  `render_target`. This is direct evidence that a transport limited to numeric
  plane/layer and `screen_loc` cannot reproduce the complete scene.
- Core recognizes `PLANE_MASTER`, `KEEP_TOGETHER`, `KEEP_APART`, `TILE_BOUND`,
  `PIXEL_SCALE`, and `PASS_MOUSE`. `MapIconList::AddFilter`,
  `FilterChain::Overflow(PixelBounds&)`, `FilterBase::SetTransform`, and
  `FilterBase::Interpolate` demonstrate that filters can alter visual bounds
  and transforms; they cannot safely be treated as a post-composite color-only
  effect.
- `byondwin.dll` contains separate D3D sampler configurations using POINT and
  LINEAR minification/magnification/mipmap filtering. This confirms multiple
  scaling modes exist, but static strings alone do not map a particular DMF
  zoom mode to one sampler.

The current Dream64 renderer already uses a shared `MapTransform` for tile
placement and coordinate lookup, draws native-size world sprites across turf
boundaries, sorts the visible world scene by plane/layer/stable position, and
supports a useful subset of `screen_loc`. The remaining concrete gaps are:

1. `LocalClientAppearance` does not transport transform matrices, blend mode,
   appearance flags, mouse opacity, visual bounds, filters, render source/target,
   or plane-master/subplane/parent relationships.
2. World drawing scales the turf grid for `tile-size`/zoom but blits native
   sprites and pixel offsets without the corresponding display transform.
3. `map_hit_at` searches only the appearances owned by the resolved turf and
   supplies turf-local coordinates. It therefore misses a large or offset
   appearance extending from a neighboring turf and can disagree with drawing.
4. Screen picking reverses client-screen insertion order and tests composed
   alpha, but does not apply plane/layer grouping, `mouse_opacity`, `PASS_MOUSE`,
   transformed/expanded bounds, or maptext-link targets.
5. Plane masters/render relays are currently skipped to avoid visible helper
   quads. Correct compatibility requires retaining them as off-screen grouping,
   filter, blend, and render-target nodes instead of either drawing or dropping
   them.

The defensible implementation sequence is to introduce one retained display
list carrying stable IDs, parent/plane/subplane data, native and transformed
bounds, mouse policy, and render-target/filter metadata; derive both drawing
and reverse-order picking from that same list; then add plane-master render
passes and filter-bound expansion. Differential oracles are still required for
negative-plane ordering, equal-layer tie breaking, fractional zoom sampling,
`screen_loc` ranges/percent forms, and `PASS_MOUSE` precedence.

### Guest and client lifecycle

Relevant `byondcore.dll` exports include:

- `DelayLogin`: `0x00087890`
- `GuestLogin`: `0x00088E50`
- `ImportResources`: `0x00088FD0`
- `InitClient`: `0x000890E0`
- `Login`: `0x00089310`
- `SkinLoaded`: `0x00089C50`
- `SkinInfo_CIO`: `0x0001C290`
- `SkinLoaded_CIO`: `0x0001D7F0`
- `ResourcesReceived_CIO`: `0x00007B90`

Both client-server and server classes expose guest-login policy. Behavioral
oracles independently confirm that key and mob are assigned before
`/client/New()`, and that skin/control availability precedes ordered lobby UI
actions. Dream64's scheduler-owned guest connection lifecycle matches that
shape. Explicit skin-ready and resource-ready acknowledgements remain to be
implemented.

### Typed UI actions and resources

Core strings identify distinct actions including `browse_file`, `browse_rsc`,
`browse_text`, `output_file`, and `preload_rsc`. This validates Dream64's typed,
monotonically ordered `Winset`, `Output`, `BrowseResource`, and `Browse` stream.

Core cache strings show resource-list availability negotiation, CRC conflicts,
cache-read CRC validation, length/CRC metadata, gzip, and cache-control paths.
`byondext.dll` is a synchronous codec/archive helper, exporting zlib/minizip and
image-format routines but no socket/thread ownership. Representative RVAs:

- `crc32`: `0x00001410`
- `deflate`: `0x00002360`
- `inflate`: `0x000062C0`
- `uncompress`: `0x0000A1E0`
- `unzOpen`: `0x0000A770`
- `unzReadCurrentFile`: `0x0000AB80`
- `zipOpen`: `0x0000BAB0`
- `zipWriteInFileInZip`: `0x0000C0B0`

Dream64 should use a collision-resistant content digest as its primary cache
identity and CRC32 only for compatibility/integrity, negotiate availability,
chunk transfers, and retain sequenced batches until acknowledged.

### Networking, threading, and scheduling

`byondcore.dll` directly imports Winsock accept/bind/listen/select/send/recv,
async socket notification, WinHTTP async callbacks, threads, events, critical
sections, and file mapping. `byondext.dll` does not.

Core exports include:

- `DungServer::SetSendMapsThreadCount`: `0x0011F020`
- `DungClient::TickClient`: `0x00089CE0`
- `DungThreadManager::ProcessCallbacks`: `0x00297700`
- `DungThreadPool::WaitAll`: `0x002989F0`
- `Byond_ThreadSync`: `0x00239DE0`

Selected x86 disassembly makes the ownership boundary more concrete:

- `DungServer::SetSendMapsThreadCount` (`0x0011F020`) stores zero for requested
  counts `<= 1`; otherwise it stores `min(requested, 64)` in a process-global.
  This is a specialized map-send concurrency setting, not evidence that DM
  datum execution is concurrent.
- `DungThreadManager::AddTask` (`0x00293DC0`) takes the manager lock, appends to
  a linked task queue, updates the tail pointer, and releases the lock.
- `DungThreadManager::ProcessCallbacks` (`0x00297700`) takes the callback queue
  under its lock, detaches the head, releases the lock, then invokes and
  destroys callbacks one by one on the calling thread. It repeats if producers
  queued more work while callbacks ran. Worker completion therefore crosses a
  serialized owner-thread commit point.
- `DungThreadPool::WaitAll` (`0x002989F0`) walks its worker slots and waits in
  bounded 1000-millisecond calls; `WaitAny` delegates to the pool event. This is
  explicit join/wakeup machinery rather than unconstrained shared execution.

`dreamdaemon.exe` imports the global `timelib_thread`, `timelib`, and `socklib`
objects plus `DungThreadManager::ProcessCallbacks`. Its timer dispatch wrapper
at executable RVA `0x001C710` calls `TimeLib::Event_io` and then immediately
calls `timelib_thread.ProcessCallbacks`; its socket wrapper at RVA `0x001C5F0`
calls `SocketLib::Event_io`. Thus network/timer event pumping and serialized
worker-result callbacks are explicit host-loop responsibilities. A long DM
dispatch that fails to return to this loop can delay ping/socket service even
when background workers exist.

`DungClient::TickClient` is a thunk to the core tick routine at `0x0007E000`.
That routine performs a sequence of timer, socket, callback/state, rendering,
and bookkeeping calls before scheduling the next timer. The evidence supports
short, recurring host-loop quanta; it does not expose a license to mutate the
live DM heap from those auxiliary operations.

This supports Dream64's rule that workers operate on immutable snapshots and
the DM thread commits live state. It does not justify concurrent heap mutation.
Current IPC gaps are the single synchronous accepted connection, destructive
poll drain without ACK/replay, unbounded responses, scheduler-thread resource
file I/O, and full-map snapshots without deltas/backpressure.

For Dream64, the closest safe analogue is: (1) cap every VM dispatch by both an
instruction budget and a wall-clock deadline, preserving the exact
continuation; (2) service accept/read/write, transport-only ping, timers, and
completed-worker queues between every dispatch; (3) build immutable map/resource
jobs on the owner thread, run encoding/compression/file reads in a bounded pool,
then commit results in stable sequence order; and (4) never wait for the whole
pool from the latency-sensitive host loop. Instrument p50/p95/p99 host-loop gap,
VM-slice time, callback-queue age, socket-service age, and outbound queue bytes;
the compatibility gate should keep loop-gap p99 below one BYOND tick under
startup load.

### Public value/proc boundary

Core exposes explicit tagged/refcounted value operations (`ByondValue_Clear`,
`IncRef`, `DecRef`, `Equals`, `Equiv`, type predicates, numeric/ref accessors,
and setters), list create/read/write operations, proc calls by name or string
ID, variable reads/writes, `Byond_Block`, `Byond_Return`, and `Byond_LastError`.

This supports Dream64's owned values, stable logical IDs, scheduler frames, and
string-ID fast paths. It does not establish BYOND's private struct layout or
authorize copying its ABI into generated code.

### Datum construction and atom-initialization boundary

A read-only x86 pass over `byondcore.dll` establishes a central scalar
construction gateway:

- `Byond_New` is exported at RVA `0x00237D00` and `Byond_NewArglist` at RVA
  `0x00238040`.
- Both owner-thread branches call the same private routine at RVA
  `0x001678F0`. A raw relative-call scan finds eleven direct calls to that
  routine in executable sections: nine private engine sites plus the two public
  wrappers.
- `Byond_New` allocates `argument_count * 8` bytes, validates and copies each
  tagged argument into that contiguous buffer, and passes the buffer and count
  to the private routine. `Byond_NewArglist` passes its already-materialized
  argument-list value to the same routine.
- Both wrappers first call the same owner-thread predicate at RVA `0x00296CE0`.
  When invoked off-owner-thread, they allocate a small task record containing
  the arguments and wrapper entry point, enqueue it through
  `DungThreadManager::AddTask` at RVA `0x00293DC0`, and wait for the serialized
  result. They do not construct live datums concurrently on the caller thread.
- The private routine branches on the runtime value/type tag and dispatches
  through internal type/procedure helpers. The inspected entry path contains no
  worker-pool fan-out or public bulk-atom constructor.

This is evidence for a highly optimized native scalar constructor and one
authoritative mutation thread, not a hidden parallel `/atom/Initialize` pass.
It also fits the callback-queue ownership evidence above: workers may prepare
immutable data, but live datum construction and DM lifecycle dispatch commit on
the owner thread.

The absence of a direct call does not prove that no private or inlined bulk
helper exists elsewhere. It does rule out treating BYOND's public construction
API as such a helper. For Dream64, the defensible performance strategy is to
keep map decoding and immutable resource work parallel, while accelerating the
scalar VM/value/list/type-dispatch path exercised by every `New` and
`Initialize`. Production frontier benchmarks must preserve scheduler
instruction accounting; advancing farther only by under-counting fused
bytecodes is not a valid speedup.

## Prioritized compatibility work

1. Apply effective `UiState` layout/visibility changes to native controls after
   `winset`, with event-feedback suppression.
2. Add skin-ready and resource-ready acknowledgement barriers.
3. Add retained sequence/ACK/replay for UI and resource batches.
4. Add digest-based resource manifests, availability negotiation, chunking,
   bounded caches, and asynchronous file/compression work.
5. Add multi-client connection workers feeding one bounded scheduler queue.
6. Add versioned, view-filtered map deltas with snapshot fallback.
7. Implement MAP zoom/tile-size/letterbox/text-mode, screen objects, maptext,
   filters/particles, and hit targets.
8. Implement native OUTPUT rich-text/link/image/max-line semantics.
9. Add typed browser topic/link/new-window/status/title callbacks with strict
   origin, capability, and control-address enforcement.

## Analysis limits

COMDAT folding causes unrelated virtual/default methods to share RVAs, so a
shared address is not proof of shared high-level behavior. Export names,
imports, strings, and selected direct-call candidates do not reveal the private
server packet format, exact private value layout, or every thread-affinity rule.
Those behaviors require independent black-box oracles and differential tests.

## DM instruction dispatcher

A follow-up pass correlated the installed DLL with the public `auxtools` and
`dmasm` implementations. The result identifies the 516.1680 interpreter
dispatcher directly rather than inferring it from public procedure gateways:

- The current auxtools Windows signature for build 1616 and later matches the
  DLL exactly once, at RVA `0x0013E244`.
- The execution context holds a pointer to 32-bit bytecode words at offset
  `+0x10` and a 16-bit word offset/program counter at `+0x14`.
- The dispatcher loads the current word, rejects values above `0x18C`, and
  jumps through the absolute-address table at RVA `0x00153AC8`.
- That table contains 397 legal slots (`0x000` through `0x18C`) and 396 distinct
  native handler addresses. Opcodes `0x11B` (`Bounds`) and `0x11C` (`OBounds`)
  are the only two slots sharing an entry address in this build.
- Simple handlers confirm word-oriented operands. For example, handler
  `0x084` at RVA `0x0013E269` advances the program counter, reads the next
  32-bit word, stores it in the execution context cache field at `+0x08`, and
  advances again. Handler `0x085` does the same for the cache-key field at
  `+0x0C`.

The opcode names and operand schemas are already substantially described by
`dmasm`; native analysis should therefore concentrate on its unknown operands,
unassigned slots, and semantic TODOs instead of rediscovering the whole table.
The read-only helper `scripts/audit-byond-vm.py` reproduces bounded x86
disassembly by export name or RVA and records the input DLL hash.

The generated evidence is stored separately from the proprietary binary:

- `generated/byondcore-516.1680-manifest.json` contains the five hashed PE
  sections, 2,826 named exports, and 407 imports.
- `generated/byondcore-516.1680-opcodes.json` contains all 397 table entries,
  396 distinct handler RVAs, evidence-graded classifications for every entry,
  and bounded
  native disassembly for every handler.

Auxtools `debug_server` 2.3.7 also builds successfully for
`i686-pc-windows-msvc` in the local environment. Its source explicitly selects
the post-1668 execution-context representation used by build 516.1680. The
remaining unnamed dmasm slots are therefore candidates for controlled live
instruction capture rather than speculative naming from machine code alone.

### Live bytecode oracle

The `fixtures/oracle/auxtools_vm` world loads the locally built probe DLL and
receives `SUCCESS` from `auxtools_init` under DreamDaemon 516.1680. The
`tools/auxtools-vm-dump` hook can then return raw bytecode words and dmasm output
for a requested procedure without requiring an editor/debug-adapter session.

One controlled procedure establishes opcode `0x17B`:

```dm
var/loaded = load_ext(library, function_name)
return call_ext(loaded)(arglist(arguments))
```

Its raw tail is `GETVAR local(loaded)`, `GETVAR arg(arguments)`, `0x17B`,
`RET`, `END`. This identifies `0x17B` as `CallExtLoadedArgList`; the generated
opcode map records the name with provenance `dream64-live-oracle`. Static
inspection of its handler at RVA `0x00148F27` is consistent with consuming a
list-backed argument vector and dispatching a previously loaded external-call
handle. The complete formerly-unnamed batch is documented in
`byond-516.1680-unknown-opcodes.md`; all slots now have behavioral compatibility
names, operand shapes, and handler classifications.

## Compiled database, map, and cache deep dive

This follow-up examined the installed binaries without loading them into
Dream64. The identities used were `byondcore.dll`
`E96396D32BF0DB945F27C8692F90F2C104B4573B83992983AA5762B6122A025B`,
`byondext.dll`
`1EE8DD4DF96F8EAAC4C021E03181DC821911F6B3B33818296788DC9516AEE7D5`,
and `dreammaker.exe`
`15688BAF5DF17A07C1BE57B5B76EFBE841D67DE5EF7AB1682602E482760DBCA7`.
All are PE32 x86 binaries from 516.1680.

### Direct evidence

`byondcore.dll` exports six distinct builder entry points:

| Export | RVA |
| --- | ---: |
| `DungBuilder::LoadDB(char const *)` | `0x0010B500` |
| `DungBuilder::LoadDMP(char const *)` | `0x0010B520` |
| `DungBuilder::LookupCache(uint32, uint32 *)` | `0x0010B560` |
| `DungBuilder::LookupCacheByCRC(uint32)` | `0x0010B5D0` |
| `DungBuilder::SaveDB(char const *)` | `0x0010B860` |
| `DungBuilder::SaveDMP(char const *)` | `0x0010B880` |

The exported functions are narrow wrappers over different internal routines:
`LoadDB` calls RVA `0x000F9DF0`, `LoadDMP` calls `0x000FA620`, `SaveDB` calls
`0x000FAFE0`, and `SaveDMP` calls `0x000FB0C0`. This is direct evidence that
compiled database and map persistence are separate operations.

The `LoadDB` body recognizes `.dm`, `.dme`, `.dmm`, and `.dmp`, reports
`loading %s`, and has a separate `Failed to load map file %s` path. The
`SaveDB` path derives `.dmb` and `.sym` outputs and calls an internal writer.
That writer opens the output with `_wfopen`, emits `world bin v%u` and minimum
compatibility records, calls multiple independent serialization routines, and
uses `ftell`/`fseek` before closing. The observed seek-back behavior is
consistent with lengths or offsets being backpatched into a sectioned compiled
database. It does not reveal the private table schema.

Both cache lookup exports read the global builder/cache owner and delegate to
different methods. `LookupCache` supplies an identifier and an output pointer;
`LookupCacheByCRC` supplies a CRC and zero as its second argument. Both return
null through a common error-reporting path when lookup fails. Thus ID lookup
and CRC lookup are genuinely distinct entry points.

Cache strings establish persistent and temporary cache files, entry length and
CRC32 metadata, CRC verification on reads, corruption removal, CRC collision
reporting between resource paths, resource availability negotiation, gzip,
cache-control handling, cache compaction, and bounded/default cache sizes.

`dreammaker.exe` imports all six builder functions from `byondcore.dll`. It
also imports `crc32`, `deflate`, and `inflate` from `byondext.dll`, plus
`CreateFileMappingA`, `MapViewOfFile`, and `UnmapViewOfFile`. `byondcore.dll`
independently imports wide-character file mapping, zlib, `CreateThread`, waits,
and critical sections. `byondext.dll` exports synchronous zlib/minizip codecs
but does not import thread creation or file mapping. `dm.exe` imports the DB/map
save/load boundary and `DungThread`; `dreamdaemon.exe` imports thread-manager
callback processing.

### Supported inference and limits

The evidence supports independently persisted compiled program, map, and
resource/cache data, plus CRC-checked lazy cache lookup and compression outside
the live DM value model. It is reasonable to infer that immutable mapping and
workers help some loading, editor, resource, or delivery paths.

The imports alone do **not** prove that `.dmb` or `.dmp` is always mapped, that
every compiled section is independently compressed, or that atom initialization
mutates the world from worker threads. No direct call from the narrow DB/map
wrapper bodies to file mapping was established. CRC32 is an integrity and
compatibility key here, not a collision-resistant artifact identity.

### Dream64 cross-reference

Dream64 now matches the defensible parts of this shape: `.d64` has a bounded
section table; large immutable artifacts are read-only mapped with buffered
fallback; `RuntimeStructuralSeed` avoids rebuilding type ancestry; and the
independent `WorldPlan` cache is fingerprint-keyed, CRC-protected, bounded, and
atomically replaced.

The next cold-boot step is to keep procedure bodies, map plans, and resource
blobs independently addressable, eagerly decode only metadata and
startup-reachable bodies, and run resource decompression/integrity checks as
immutable worker jobs. Dream64 should retain a strong digest as cache identity
and CRC only as a cheap corruption check. Live atoms, globals, lifecycle calls,
and scheduler state remain under the authoritative VM owner.

## Client readiness and resource-transfer follow-up

This pass focused on the boundary between connection, skin creation, resource
delivery, and the first usable UI. It used the same read-only 516.1680 binaries
and did not recover or depend on BYOND's private packet layout.

### Readiness is multi-phase, not a single attach

`byondcore.dll` has separate exported transitions for
`DungClient::SkinLoaded` (`0x00089C50`), `SkinLoaded_CIO` (`0x0001D7F0`), and
`ResourcesReceived_CIO` (`0x00007B90`). It also exports
`PreloadedRscUrl_Callback` (`0x0001BA20`) and `ResetCache` (`0x00089700`). Core
strings include `Resource list received: %d of %d are already available.` and
`Required resources received.` These are direct evidence that transport attach,
native-skin readiness, resource availability negotiation, and completion of
required transfers are distinct states.

Dream64 currently treats a successful `attach` response as client readiness.
The client immediately requests snapshots and destructively polls UI events;
there is no client-to-server `skin_ready` or `resources_ready` transition. The
release-compatible state machine should instead be:

1. attach and assign the authoritative `/client` and `/mob`;
2. publish the skin/control manifest and wait for `skin_ready`;
3. publish required resource metadata, let the client report available items,
   and transfer only misses;
4. wait for `resources_ready` before releasing resource-dependent ordered UI;
5. enable interactive map/input only after the initial UI/snapshot generation
   is acknowledged.

This barrier is especially important for MonkeStation: delivering `browse()` or
screen appearances before their DMI/HTML dependencies are installed can produce
the observed partial lobby without indicating a rendering failure.

### Resource protocol evidence and Dream64 gaps

Core strings distinguish an initial availability list from file transmission
and diagnose `Unexpected file transmission (packet_num = %u (expected =
%u),max=%lu,pid=%d)`. Together with cache-entry length/CRC metadata, this is
evidence for ordered, bounded resource transfer with an expected packet number;
it is not evidence for the exact packet encoding.

Dream64 protocol 2 currently returns a complete requested file as hex in one
response. Snapshot construction performs one synchronous request per visible
DMI, and `browse_resource` embeds the complete bytes in the UI event. This has
four actionable problems: 2x hex expansion, unbounded response frames,
scheduler-side filesystem reads, and no availability/ACK/retry state. Introduce
a digest-keyed manifest, client availability set, bounded binary chunks with
`(resource_id, offset_or_sequence, total_length, integrity)`, and explicit
ACK/retry. Keep CRC32 only as the compatibility/integrity field and use the
existing strong project/resource digest as the cache identity.

`ui_events` also calls `take_local_client_outbound_events`, so polling removes
events before the client acknowledges application. A lost connection between
response creation and presentation loses the lobby stream permanently. Retain
batches by monotonically increasing sequence until `ui_ack`, replay after
reconnect, and impose a byte/event window for backpressure.

### Browser and output semantics

The browser adapter has asynchronous `PutFile`, `PutText`, and a
`PutTextCallback`; its WebView implementation exposes completion, error, link,
popup, status, and resource-request callbacks. The injected bridge fires a
`byond-ready` DOM event and implements asynchronous `winset`, callback-based
`winget`, and command dispatch. Therefore `BrowseResource` ordering alone is
necessary but insufficient: Dream64 needs a per-control navigation generation
and must acknowledge completion only after WebView resource interception can
resolve every dependency for that generation. Late callbacks from an older
navigation must not complete a newer one.

`CIOutputCtrl` separately exports `PutClear`, `PutFile`, `PutObject`, `PutText`,
`SetImage`, `SetLines`, style/font/link-color properties, and image callbacks.
This confirms OUTPUT is not a browser alias. Dream64's text-only output path is
safe as a compatibility floor, but image/object/file and bounded-line behavior
remain independently testable requirements.

### Concrete release gates

Before another expensive visible boot, add transport-level tests that disconnect
after receiving but before acknowledging a UI/resource batch, reconnect, and
prove byte-identical replay without duplicate application. Add a cold-cache
test in which HTML references nested CSS, font, image, and DMI resources and
assert that `resources_ready` and WebView navigation completion precede input
enablement. Finally, bound every frame/chunk and verify a malformed advertised
length cannot allocate attacker-sized memory.
