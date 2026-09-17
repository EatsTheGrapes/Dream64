# Booting Monkestation on Linux

The boot-performance numbers in this directory were historically produced on
Windows through `scripts/boot-visible.ps1`. This note records what a Linux
reproduction additionally needs, because none of it is obvious from a failure
and each one costs a full compile to rediscover.

## Steps

```sh
git clone --depth 1 https://github.com/Monkestation/monkestation2.0
cd monkestation2.0
touch libauxlua.so libdreamluau.so librust_g.so

(cd tgui && bun install --frozen-lockfile && bun run tgui:build)   # ~25s

dream64-compiler -DCBT tgstation.dme          # ~105-155s, ~955MB artifact
HOME="$PWD" dream64-server boot tgstation.d64  # artifact carries its own map
```

Add `DREAM64_PROFILE_DATUM_ALLOC=1` for allocation/initializer telemetry, and
`DREAM64_BOOT_MAX_SLICES=N` to stop at a slice boundary.

## Why each piece is needed

**`-DCBT` is not optional.** `MAP_SWITCH` in `code/__DEFINES/mapping.dm` picks
its *left* operand when `CBT` is defined and its *right* one otherwise, and
`SETUP_MAP_ICONS` uses it to choose between a real icon state and a map-editor
placeholder that is spelled as a type path:

```dm
SETUP_MAP_ICONS("jumpsuit", "/obj/item/clothing/under/color/grey")
```

Without `CBT`, `icon_state` becomes the literal string
`"/obj/item/clothing/under/color/grey"`, and SSearly_assets dies on
`Invalid target bundle icon_state`. That is correct preprocessing of what the
source says, not an engine bug -- the Monkestation build tool defines `CBT`,
so a normal build never sees the other branch. Dream64's compiler takes
`-D NAME[=VALUE]`.

**`HOME` must point inside the project.** `__detect_rust_g` and
`__detect_auxtools` probe `[world.GetConfig("env", "HOME")]/.byond/bin/...`,
an absolute path outside the project root, which the engine's file guard
refuses. Both probes sit behind `world.system_type == UNIX`, so a Windows dev
box never reaches them. Pointing `HOME` at the project makes the probe legal;
it then simply finds nothing and falls through.

**TGUI must be built.** `/datum/asset/simple/tgui` registers
`file("tgui/public/tgui.bundle.js")` and its siblings, which are gitignored
build artifacts -- a fresh clone has only four hand-written files in
`tgui/public/`. Without them `SSassets` reaches `md5asfile()`, whose
`fcopy(file, "tmp/...")` silently copies nothing, and the following
`rustg_hash_file` raises on the file that was never written. That raise
unwinds the Master Controller thread, so the boot dies in SSassets rather
than reporting which asset was missing. `tgui/package.json` pins
`bun@1.3.6`; `bun install --frozen-lockfile && bun run tgui:build` produces
the real bundles in about 25 seconds.

**The three `.so` stubs** stop `__detect_auxtools` from `CRASH`ing when a
native library is absent. They are empty files: nothing is loaded from them.
Only DM code paths that actually call into auxlua/dreamluau would notice, and
boot-to-pregame does not.

## Known stopping point

A boot currently reaches SSatoms and initializes atoms. Anything past that has
not been validated here; failures beyond this point are unexplored rather than
known-good.
