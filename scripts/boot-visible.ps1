param(
    [Parameter(Mandatory = $true)]
    [int] $BootNumber,
    [string] $Executable,
    [string] $Dme = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\tgstation.dme",
    [string] $Map = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\_maps\map_files\generic\CentCom.dmm",
    [switch] $FullTrace,
    [switch] $AuditRuntime,
    [switch] $ProfileAtoms,
    [switch] $ProfileStartup
)

$ErrorActionPreference = "Stop"
$launcherErrorLog = Join-Path $PSScriptRoot "..\target\boot-visible-$BootNumber.launcher-error.log"
Remove-Item -LiteralPath $launcherErrorLog -Force -ErrorAction SilentlyContinue
trap {
    $rendered = $_ | Out-String
    [IO.File]::WriteAllText(
        $launcherErrorLog,
        $rendered,
        [Text.UTF8Encoding]::new($false)
    )
    Write-Host $rendered -ForegroundColor Red
    exit 1
}
$workspace = (Resolve-Path "$PSScriptRoot\..").Path
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $workspace "target\release\dm-lifecycle.exe"
}
$Executable = (Resolve-Path $Executable).Path
$Dme = (Resolve-Path -LiteralPath $Dme).Path
$monkRoot = Split-Path -Parent $Dme
$tguiBuild = Join-Path $monkRoot "tools\build\build.bat"
if (-not (Test-Path -LiteralPath $tguiBuild -PathType Leaf)) {
    throw "Monkestation TGUI build entrypoint not found: $tguiBuild"
}
$log = Join-Path $workspace "monkestation-headless-boot-$BootNumber.console.log"
# Keep only the active boot trace. These files are intentionally ignored by
# Git, and pruning them here prevents long debugging sessions from filling the
# workspace with obsolete multi-megabyte logs.
Get-ChildItem -LiteralPath $workspace -File -Filter "monkestation-headless-boot-*.console.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -ne $log } |
    Remove-Item -Force
if ($FullTrace) {
    $env:DREAM64_BOOT_TRACE = "1"
} else {
    Remove-Item Env:DREAM64_BOOT_TRACE -ErrorAction SilentlyContinue
}
$env:DREAM64_BOOT_DASHBOARD = "1"
if ($AuditRuntime) {
    $env:DREAM64_BOOT_AUDIT_RUNTIME = "1"
} else {
    Remove-Item Env:DREAM64_BOOT_AUDIT_RUNTIME -ErrorAction SilentlyContinue
}
if ($ProfileAtoms) {
    $env:DREAM64_PROFILE_ATOMS = "1"
} else {
    Remove-Item Env:DREAM64_PROFILE_ATOMS -ErrorAction SilentlyContinue
}
if ($ProfileStartup) {
    $env:DREAM64_PROFILE_STARTUP = "1"
} else {
    Remove-Item Env:DREAM64_PROFILE_STARTUP -ErrorAction SilentlyContinue
}

# Force non-fatal startup behavior for deterministic boot attempts unless a user
# explicitly adds one of these strict-mode env vars outside this launcher.
$startupStrictEnv = @(
    "DREAM64_STRICT_STARTUP_ERRORS",
    "DREAM64_FAIL_FAST_STARTUP_ERRORS",
    "DREAM64_STARTUP_FATAL",
    "DREAM64_STARTUP_NONFATAL"
)
foreach ($var in $startupStrictEnv) {
    Remove-Item "Env:$var" -ErrorAction SilentlyContinue
}

try {
    $Host.UI.RawUI.WindowTitle = "Dream64 - Monkestation Headless Boot $BootNumber"
} catch {
    # Some Windows Terminal/ConPTY hosts do not expose a writable RawUI title.
    # The launcher must remain functional when decoration is unavailable.
    Write-Warning "Could not set the console title: $($_.Exception.Message)"
}

try {
    Clear-Host
} catch {
    # Redirected and headless hosts can expose RawUI without a valid console
    # handle. Clearing the dashboard is cosmetic and must not block boot.
}
Write-Host ""
Write-Host "  DREAM64" -ForegroundColor Cyan -NoNewline
Write-Host "  /  MONKESTATION HEADLESS" -ForegroundColor White
Write-Host "  Boot $BootNumber" -ForegroundColor DarkGray
Write-Host "  ============================================================" -ForegroundColor DarkCyan
Write-Host "  Optimized 64-bit release | Runtime audit: $AuditRuntime | Full trace: $log" -ForegroundColor DarkGray
Write-Host ""

# Use Monkestation's exact incremental TGUI target. Juke owns its Bun/package
# cache and input/output freshness checks; Dream64 remains the only DM compiler
# and server process in this launcher.
Write-Host "  Preparing Monkestation TGUI assets..." -ForegroundColor Cyan
& $tguiBuild tgui
$tguiExitCode = $LASTEXITCODE
if ($tguiExitCode -ne 0) {
    Write-Host ""
    Write-Host "  TGUI build failed (exit $tguiExitCode); Dream64 was not started." -ForegroundColor Red
    exit $tguiExitCode
}
Write-Host "  TGUI assets ready." -ForegroundColor Green
Write-Host ""

$writer = [IO.StreamWriter]::new(
    $log,
    $false,
    [Text.UTF8Encoding]::new($false)
)
$writer.AutoFlush = $true
$started = Get-Date
$ready = $false
$failed = $false
$lastPhase = "starting"
$initializerCount = 0
$startupChecklistShown = $false
$startupChecklistCompleted = 0
$startupChecklistActive = @{}
$peakWorkingSetBytes = [int64]0
$peakPrivateBytes = [int64]0

function Get-BootMemorySnapshot {
    $processName = [IO.Path]::GetFileNameWithoutExtension($Executable)
    try {
        $process = Get-Process -Name $processName -ErrorAction SilentlyContinue |
            Where-Object { $_.StartTime -ge $started.AddSeconds(-5) } |
            Sort-Object StartTime -Descending |
            Select-Object -First 1
    } catch {
        return $null
    }
    if (-not $process) {
        return $null
    }

    $script:peakWorkingSetBytes = [math]::Max(
        $script:peakWorkingSetBytes,
        [int64]$process.PeakWorkingSet64
    )
    $script:peakPrivateBytes = [math]::Max(
        $script:peakPrivateBytes,
        [int64]$process.PrivateMemorySize64
    )
    return [pscustomobject]@{
        WorkingSetBytes = [int64]$process.WorkingSet64
        PrivateBytes = [int64]$process.PrivateMemorySize64
        PeakWorkingSetBytes = $script:peakWorkingSetBytes
        PeakPrivateBytes = $script:peakPrivateBytes
    }
}

function Write-BootLine {
    param([string] $Line)

    $script:writer.WriteLine($Line)
    if ($Line -match "HEADLESS READY|HeadlessReady|boot-vm: heap-gc") {
        $memory = Get-BootMemorySnapshot
    } else {
        $memory = $null
    }
    if ($memory -and $Line -match "boot-vm: heap-gc") {
        $script:writer.WriteLine(
            "boot-dashboard-memory working_set_bytes=$($memory.WorkingSetBytes) private_bytes=$($memory.PrivateBytes) peak_working_set_bytes=$($memory.PeakWorkingSetBytes) peak_private_bytes=$($memory.PeakPrivateBytes)"
        )
    }
    $elapsed = (Get-Date) - $started
    $stamp = "{0:mm\:ss}" -f $elapsed

    if ($Line -match "lifecycle event ([0-9]+)/([0-9]+)") {
        $current = [int64] $Matches[1]
        $total = [int64] $Matches[2]
        $percent = if ($total -gt 0) { [math]::Min(100, 100 * $current / $total) } else { 0 }
        Write-Progress -Activity "Dream64 map lifecycle" -Status "$current / $total events" -PercentComplete $percent
        Write-Host "  [$stamp] MAP LIFECYCLE  " -ForegroundColor Cyan -NoNewline
        Write-Host ("{0,7:N0} / {1:N0}  ({2:N1}%)" -f $current, $total, $percent) -ForegroundColor White
        return
    }
    if ($Line -match "HEADLESS READY|HeadlessReady") {
        $script:ready = $true
        Write-Progress -Activity "Dream64 map lifecycle" -Completed
        $readySeconds = ((Get-Date) - $started).TotalSeconds
        if ($memory) {
            $memoryLine = "boot-dashboard-memory ready_working_set_bytes=$($memory.WorkingSetBytes) ready_private_bytes=$($memory.PrivateBytes) peak_working_set_bytes=$($memory.PeakWorkingSetBytes) peak_private_bytes=$($memory.PeakPrivateBytes)"
            $script:writer.WriteLine($memoryLine)
        }
        Write-Host ""
        Write-Host "  ============================================================" -ForegroundColor Green
        Write-Host ("                 GAME READY! ({0:N1}s)" -f $readySeconds) -ForegroundColor Yellow
        Write-Host "              HEADLESS READY - SERVER LIVE" -ForegroundColor Green
        if ($memory) {
            Write-Host (
                "       RAM READY {0:N2} GiB private / {1:N2} GiB working | peak {2:N2} GiB" -f
                    ($memory.PrivateBytes / 1GB),
                    ($memory.WorkingSetBytes / 1GB),
                    ($memory.PeakWorkingSetBytes / 1GB)
            ) -ForegroundColor Cyan
        }
        Write-Host "  ============================================================" -ForegroundColor Green
        return
    }
    if ($Line.StartsWith("boot-vm: init-display|")) {
        $fields = @{}
        foreach ($part in $Line.Split('|') | Select-Object -Skip 1) {
            $separator = $part.IndexOf('=')
            if ($separator -ge 0) {
                $fields[$part.Substring(0, $separator)] = $part.Substring($separator + 1)
            }
        }
        $eventName = [string]$fields['event']
        $category = [string]$fields['category']
        if ($eventName -eq 'add') {
            $rawLabel = ([string]$fields['name'] -replace '<[^>]+>', '').Trim()
            $isChild = $rawLabel.StartsWith('>')
            $label = ($rawLabel -replace '^[>\-]\s*', '').Trim()
            $marker = if ($isChild) { '  >' } else { '-' }
            $stage = ([string]$fields['stage'] -replace '<[^>]+>', '').Trim()
            $seconds = 0.0
            [void][double]::TryParse(
                [string]$fields['seconds'],
                [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture,
                [ref]$seconds
            )
            if (-not $startupChecklistShown) {
                $script:startupChecklistShown = $true
                Write-Host ""
                Write-Host "  STARTUP CHECKLIST" -ForegroundColor Cyan
                Write-Host "  ------------------------------------------------------------" -ForegroundColor DarkCyan
            }
            if ($stage -match 'INITIALIZING|CREATING|LOADING') {
                $script:startupChecklistActive[$category] = @{
                    Label = $label
                    Marker = $marker
                    Started = [Diagnostics.Stopwatch]::GetTimestamp()
                }
                Write-Host ("  {0} " -f $marker) -ForegroundColor Yellow -NoNewline
                Write-Host ("{0}  {1}" -f $label, $stage) -ForegroundColor White
            } elseif ($stage -match 'DONE') {
                $script:startupChecklistCompleted++
                $script:startupChecklistActive.Remove($category)
                $duration = if ($seconds -gt 0) { " ({0:N1}s)" -f $seconds } else { "" }
                Write-Host ("  {0} " -f $marker) -ForegroundColor Green -NoNewline
                Write-Host ("{0}  DONE{1}" -f $label, $duration) -ForegroundColor Green
            } elseif ($stage -match 'FAILED|ERROR') {
                $script:startupChecklistActive.Remove($category)
                Write-Host ("  {0} " -f $marker) -ForegroundColor Red -NoNewline
                Write-Host ("{0}  {1}" -f $label, $stage) -ForegroundColor Red
            }
        } elseif ($eventName -eq 'remove' -and $startupChecklistActive.ContainsKey($category)) {
            $active = $startupChecklistActive[$category]
            $script:startupChecklistActive.Remove($category)
            $script:startupChecklistCompleted++
            $elapsed = ([Diagnostics.Stopwatch]::GetTimestamp() - [int64]$active.Started) / [Diagnostics.Stopwatch]::Frequency
            Write-Host ("  {0} " -f $active.Marker) -ForegroundColor Green -NoNewline
            Write-Host ("{0}  DONE ({1:N1}s)" -f $active.Label, $elapsed) -ForegroundColor Green
        }
        return
    }
    if ($Line -match "boot-vm: heartbeat steps=([0-9]+) depth=([0-9]+) procedure=(.+) instruction=([0-9]+)") {
        $steps = [int64] $Matches[1]
        $depth = [int] $Matches[2]
        $procedure = $Matches[3]
        $instruction = [int] $Matches[4]
        Write-Host "  [$stamp] DM STARTUP     " -ForegroundColor Cyan -NoNewline
        Write-Host ("{0:N0} ops | depth {1} | {2} @ {3}" -f $steps, $depth, $procedure, $instruction) -ForegroundColor White
        return
    }
    if ($Line -match "boot-vm: initializer-end path=(.+?) elapsed_ms=([0-9]+) steps=([0-9]+).*") {
        $script:initializerCount++
        $milliseconds = [int64] $Matches[2]
        if (($script:initializerCount % 25) -eq 0 -or $milliseconds -ge 500) {
            $procedure = $Matches[1]
            Write-Host "  [$stamp] GLOBAL INIT    " -ForegroundColor DarkCyan -NoNewline
            Write-Host ("#{0:N0} | {1} | {2:N0} ms" -f $script:initializerCount, $procedure, $milliseconds) -ForegroundColor White
        }
        return
    }
    if ($Line -match "boot-vm: subsystem-constructor type=([^ ]+) procedure=(.+)") {
        # Constructor inventory is useful in the ignored full trace, but it is
        # not subsystem initialization progress. The authoritative SStitle
        # events below replace this noisy preflight list in the dashboard.
        return
    }
    if ($Line -match "boot-audit-runtime-error: group=([0-9]+).*") {
        Write-Host "  [$stamp] AUDIT FAILURE  " -ForegroundColor Magenta -NoNewline
        Write-Host $Line -ForegroundColor White
        return
    }
    if ($Line -match "initialization: .* failed|CRASH:|panicked|fatal|error:") {
        $script:failed = $true
        Write-Progress -Activity "Dream64 map lifecycle" -Completed
        Write-Host "  [$stamp] $Line" -ForegroundColor Red
        return
    }
    if ($Line -match "runtime-phase-complete phase=([^ ]+) elapsed_ms=([0-9]+)") {
        $script:lastPhase = $Matches[1]
        $seconds = [math]::Round(([double] $Matches[2]) / 1000, 2)
        Write-Host "  [$stamp] COMPLETE       " -ForegroundColor Green -NoNewline
        Write-Host "$($Matches[1]) (${seconds}s)" -ForegroundColor White
        return
    }
    if ($Line -match "lifecycle-precompile-complete elapsed_ms=([0-9]+)") {
        $seconds = [math]::Round(([double] $Matches[1]) / 1000, 2)
        Write-Host "  [$stamp] COMPLETE       " -ForegroundColor Green -NoNewline
        Write-Host "lifecycle linker (${seconds}s)" -ForegroundColor White
        return
    }
    if ($Line -match "boot-progress: (compiling project|loading map|parsing map|precompiling lifecycle|materializing globals|indexing procedures|preflighting|allocating map world|executing lifecycle|completed lifecycle|scheduler termination)") {
        $message = $Matches[1]
        Write-Host "  [$stamp] PHASE          " -ForegroundColor Yellow -NoNewline
        Write-Host $message -ForegroundColor White
        return
    }
    if ($Line -match "call stack:|^  /" -or $script:failed) {
        Write-Host "  $Line" -ForegroundColor DarkRed
        return
    }
    if ($Line -match "boot-vm: (deferred|initializer-|global-read|slow-instruction|dcs-|list-gc)") {
        return
    }
    if ($Line.Trim().Length -gt 0) {
        Write-Host "  [$stamp] $Line" -ForegroundColor DarkGray
    }
}

$command = "`"$Executable`" boot `"$Dme`" `"$Map`" 2>&1"
try {
    & $env:ComSpec /d /s /c $command | ForEach-Object { Write-BootLine ([string] $_) }
    $exitCode = $LASTEXITCODE
} catch {
    $failed = $true
    $exitCode = 1
    Write-Host "  Dashboard error: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    $writer.Dispose()
}

Write-Progress -Activity "Dream64 map lifecycle" -Completed
if (-not $ready) {
    Write-Host ""
    Write-Host "  Boot $BootNumber stopped in phase '$lastPhase' (exit $exitCode)." -ForegroundColor Red
    Write-Host "  Full diagnostics: $log" -ForegroundColor DarkGray
    Start-Sleep -Seconds 3
}
exit $exitCode
