# turf_overlays behavioral oracle

Black-box probe for the observable semantics of `atom.overlays` / `atom.underlays`
that any heap-identity interning of turf overlay lists must preserve.

Captured from BYOND 516.1680:

```
dm.exe turf_overlays.dme
DreamDaemon.exe turf_overlays.dmb -trusted -close
type turf_overlays.out            # compare against expected-byond-516.1680.txt
```

Load-bearing facts (`expected-byond-516.1680.txt`):

- `islist(t.overlays)` is `1`; reading the field yields a list.
- `t.overlays == t.overlays` is `1` — repeated reads of one atom's field observe
  a stable identity (Dream64 already matches this).
- `t1.overlays == t2.overlays` is `0` — **distinct atoms must expose distinct
  list identities.** This rules out pointing several turf fields at one shared
  heap `List` identity; an interning scheme must keep per-atom identity as
  observed by `==`.
- `t.overlays == GLOB_SHARED_SOURCE` is `0` — the field identity is never the
  appended source list's identity.
- `t1.overlays += X` never changes `t2.overlays` or the shared source list
  (`+=`, `.Cut()`, whole reassignment are all isolated per atom).
- `+=` of a list appends its elements; re-appending the same source stacks
  duplicates (`len` 3 -> 6).
- Text/appearance overlay elements stringify to `""`; only element *count* and
  isolation are stable cross-engine projections, so the probe asserts on `len`
  and mutation isolation rather than element text.
