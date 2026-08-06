# BYOND 516 compatibility inventory

This inventory defines investigation areas, not completed support. The baseline
is BYOND 516.1663 because that is Monkestation's declared minimum/default build.

| Area | First observable milestone | Evidence source |
| --- | --- | --- |
| Preprocessor | Includes, defines, conditionals, built-in macros, source locations | Public DM guide and fixture output |
| Syntax | Paths, indentation, vars, procs, operators, control flow, proc arguments | Public reference and compiler diagnostics |
| Object tree | Inheritance, overrides, static vars, proc lookup, `..()` | Compiler object/code-tree output and fixtures |
| Values | Null, binary32 numbers, strings, paths, refs, truth and comparison rules | Public BYONDAPI header and runtime fixtures |
| Lists | Ordered and associative entries, mutations, iteration, operators | Public reference and runtime fixtures |
| Execution | Call frames, `src`/`usr`/`.`, sleep, spawn, runtime recovery | Public reference and runtime traces |
| World model | Datum/atom/area/turf/movable/mob/client behavior | Public reference and small worlds |
| Map | DMM parsing, coordinates, initialization order, movement | Map fixtures and generated state traces |
| Appearance | Icons, states, overlays, filters, planes, animation | Client screenshot and state fixtures |
| I/O | Files, savefiles, JSON, HTTP, database, shell policy | Sandboxed fixtures |
| UI/client | DMF, winset/winget, macros, input, browser controls, audio | Instrumented reference client sessions |
| Extensions | Public BYONDAPI value/list/proc/map/thread contracts | Installed public `byondapi.h` and adapter tests |

## Compatibility gates

- `compile`: both compilers accept or reject a fixture consistently, with
	diagnostics normalized separately from semantic outcomes.
- `run`: both runtimes emit the same typed event stream.
- `state`: both runtimes serialize the same selected world state at checkpoints.
- `render`: reference images and input traces agree within an explicitly stated
	tolerance.
- `skin`: the DMF control tree, default-control selection, focus, saved
	properties, runtime mutations, macro dispatch, and JavaScript bridge agree.
- `tgui`: Monkestation's unmodified production TGUI bundle can open, act,
	resize, move, suspend, restore, and close through the compatibility bridge.
- `monkestation`: a pinned checkout compiles, boots headlessly, initializes all
	subsystems, and completes deterministic scenario tests.

Known version differences are recorded as named expectations. They are never
silently folded into the default behavior.
