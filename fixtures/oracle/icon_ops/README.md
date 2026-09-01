# /icon pixel pipeline oracle

BYOND 516.1680 probe for the raster operations `dm-icon` reproduces. It loads
the committed 32x32 greyscale `template.dmi` (states `box` and `stripe`; `box`
is an opaque mid-grey `#808080` square filling the bottom-left quadrant) and
runs `Blend` (ICON_MULTIPLY / ICON_ADD / ICON_OVERLAY), `Scale`, `Crop`,
`Flip`, `Turn`, and `SwapColor`, emitting `GetPixel` colour strings and
dimensions.

`expected-byond-516.1680.txt` is the captured DM output (`key=value` lines, in
`icon_ops.dm` order). `dm-icon`'s `pipeline::matches_byond_icon_ops_oracle`
test asserts the same values.

Regenerate from PowerShell:

```powershell
Push-Location fixtures\oracle\icon_ops
& 'C:\Program Files (x86)\BYOND\bin\dm.exe' icon_ops.dme
& 'C:\Program Files (x86)\BYOND\bin\DreamDaemon.exe' icon_ops.dmb -trusted -close
Get-Content icon_ops.out | Where-Object { $_ -ne '' }
Pop-Location
```

`template.dmi` was produced by `scratchpad/mkdmi.py` (PIL); the DMI carries a
BYOND `Description` `zTXt` chunk. `.dmb`/`.out`/`.rsc` are gitignored.
