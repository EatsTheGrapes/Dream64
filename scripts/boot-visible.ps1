param(
    [Parameter(Mandatory = $true)]
    [int] $BootNumber,
    [string] $Executable = "$PSScriptRoot\..\target\release\dm-lifecycle.exe",
    [string] $Dme = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\tgstation.dme",
    [string] $Map = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\_maps\map_files\generic\CentCom.dmm",
    [switch] $FullTrace,
    [switch] $AuditRuntime
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path "$PSScriptRoot\..").Path
$Executable = (Resolve-Path $Executable).Path
$log = Join-Path $workspace "monkestation-headless-boot-$BootNumber.console.log"
if ($FullTrace) {
    $env:DREAM64_BOOT_TRACE = "1"
} else {
    Remove-Item Env:DREAM64_BOOT_TRACE -ErrorAction SilentlyContinue
}
if ($AuditRuntime) {
    $env:DREAM64_BOOT_AUDIT_RUNTIME = "1"
} else {
    Remove-Item Env:DREAM64_BOOT_AUDIT_RUNTIME -ErrorAction SilentlyContinue
}
$Host.UI.RawUI.WindowTitle = "Dream64 - Monkestation Headless Boot $BootNumber"

Clear-Host
Write-Host ""
Write-Host "  DREAM64" -ForegroundColor Cyan -NoNewline
Write-Host "  /  MONKESTATION HEADLESS" -ForegroundColor White
Write-Host "  Boot $BootNumber" -ForegroundColor DarkGray
Write-Host "  ============================================================" -ForegroundColor DarkCyan
Write-Host "  Optimized 64-bit release | Runtime audit: $AuditRuntime | Full trace: $log" -ForegroundColor DarkGray
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

function Write-BootLine {
    param([string] $Line)

    $script:writer.WriteLine($Line)
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
        Write-Host ""
        Write-Host "  ============================================================" -ForegroundColor Green
        Write-Host "                 HEADLESS READY - SERVER LIVE" -ForegroundColor Green
        Write-Host "  ============================================================" -ForegroundColor Green
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
        Write-Host "  [$stamp] SUBSYSTEM      " -ForegroundColor Magenta -NoNewline
        Write-Host "$($Matches[1]) -> $($Matches[2])" -ForegroundColor White
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
    if ($Line -match "boot-vm: (deferred|initializer-|global-read|slow-instruction)") {
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
