# BYOND behavioral oracles

These projects are small black-box probes for behavior that optimized Dream64
execution must preserve. Compile a project with the target BYOND 516 Dream Maker
and run the resulting DMB with DreamDaemon. `jit_core` writes `jit_core.out` and
then shuts down.

The initial JIT-safe subset is arithmetic, comparisons, truth conversion,
branches, locals, and proven direct calls. `spawn`, `sleep`, dynamic calls, and
runtime errors are deliberate interpreter/deoptimization boundaries. The
`jit_errors` procedures are independent entry points so one runtime error does
not prevent observing another.

Example from PowerShell:

```powershell
Push-Location fixtures\oracle\jit_core
& 'C:\Program Files (x86)\BYOND\bin\dm.exe' jit_core.dme
& 'C:\Program Files (x86)\BYOND\bin\DreamDaemon.exe' jit_core.dmb -trusted -close
Get-Content jit_core.out | Where-Object { $_ -ne '' }
Pop-Location
```

The `jit_errors` entry points compile under BYOND without constant-folding the
failure away. They can be invoked individually by a differential runner. For a
manual DreamDaemon probe, temporarily call the desired entry point from its
`world/New`; the runtime diagnostic must retain both the failing proc and, for
`jit_error_nested`, its caller.
