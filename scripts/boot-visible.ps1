param(
    [int] $BootNumber = 0,
    [string] $ServerExecutable,
    [string] $CompilerExecutable,
    [string] $ClientExecutable,
    [string] $Dme = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\tgstation.dme",
    [string] $Map = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\_maps\map_files\generic\CentCom.dmm",
    [string] $Skin = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\interface\skin.dmf",
    [switch] $ReuseInitializedWorld,
    [switch] $SkipCompile
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($BootNumber -le 0) {
    $BootNumber = [int](Get-Date -Format "HHmmss")
}
if ([string]::IsNullOrWhiteSpace($ServerExecutable)) {
    $ServerExecutable = Join-Path $workspace "target\release\dream64-server.exe"
}
if ([string]::IsNullOrWhiteSpace($CompilerExecutable)) {
    $CompilerExecutable = Join-Path $workspace "target\release\dream64-compiler.exe"
}
if ([string]::IsNullOrWhiteSpace($ClientExecutable)) {
    $ClientExecutable = Join-Path $workspace "target\release\dm-client.exe"
}

$ServerExecutable = (Resolve-Path -LiteralPath $ServerExecutable).Path
$CompilerExecutable = (Resolve-Path -LiteralPath $CompilerExecutable).Path
$ClientExecutable = (Resolve-Path -LiteralPath $ClientExecutable).Path
$Dme = (Resolve-Path -LiteralPath $Dme).Path
$Map = (Resolve-Path -LiteralPath $Map).Path
$Skin = (Resolve-Path -LiteralPath $Skin).Path

$logDirectory = Join-Path $workspace "logs"
[IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$serverLog = Join-Path $logDirectory "visible-boot-$BootNumber-server.log"
$clientLog = Join-Path $logDirectory "visible-boot-$BootNumber-client.log"
$compilerLog = Join-Path $logDirectory "visible-boot-$BootNumber-compiler.log"

# Production-visible boot: no instruction tracing, profilers, audit sweep,
# dashboard memory sampling, or atom inventory. The client receives lightweight
# startup phases directly over IPC and replaces them with the real lobby after
# its client/New -> mob/New -> Initialize -> Login chain attaches.
@(
    "DREAM64_BOOT_TRACE",
    "DREAM64_BOOT_DASHBOARD",
    "DREAM64_BOOT_AUDIT_RUNTIME",
    "DREAM64_PROFILE_ATOMS",
    "DREAM64_PROFILE_STARTUP",
    "DREAM64_PROFILE_MEMORY",
    "DREAM64_ENABLE_ROOTED_JIT",
    "DREAM64_INSPECT_PROCEDURE",
    "DREAM64_STRICT_STARTUP_ERRORS",
    "DREAM64_FAIL_FAST_STARTUP_ERRORS",
    "DREAM64_STARTUP_FATAL",
    "DREAM64_STARTUP_NONFATAL",
    "DREAM64_ENABLE_READY_WORLD_CACHE",
    "DREAM64_DISABLE_READY_CACHE"
) | ForEach-Object { Remove-Item "Env:$_" -ErrorAction SilentlyContinue }

if ($ReuseInitializedWorld) {
    $env:DREAM64_ENABLE_READY_WORLD_CACHE = "1"
} else {
    $env:DREAM64_DISABLE_READY_CACHE = "1"
}

$logicalProcessors = [Environment]::ProcessorCount
$env:RAYON_NUM_THREADS = [string]$logicalProcessors
$env:DREAM64_IPC_ADDR = "127.0.0.1:51664"

Write-Host ""
Write-Host "  DREAM64 / MONKESTATION" -ForegroundColor Cyan
Write-Host "  Visible boot $BootNumber" -ForegroundColor White
Write-Host "  $logicalProcessors logical processors available; affinity pinning disabled" -ForegroundColor Green
Write-Host "  Heavy trace/profile/audit modes disabled" -ForegroundColor DarkGray
if ($ReuseInitializedWorld) {
    Write-Host "  Initialized-world reuse enabled (development; map-load rolls are preserved)." -ForegroundColor Yellow
} else {
    Write-Host "  Fresh-random production boot (mapping, ruins, rooms, engine and round rolls run now)." -ForegroundColor Green
}
if ($env:DREAM64_RANDOM_SEED) {
    Write-Host "  Replaying random seed $($env:DREAM64_RANDOM_SEED)." -ForegroundColor Yellow
}
Write-Host ""

if (-not $SkipCompile) {
    Write-Host "  Checking compiler artifacts in a separate compiler process..." -ForegroundColor Cyan
    $quotedDme = '"' + $Dme + '"'
    $compiler = Start-Process -FilePath $CompilerExecutable `
        -ArgumentList @($quotedDme) `
        -WorkingDirectory $workspace `
        -RedirectStandardError $compilerLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($compiler.ExitCode -ne 0) {
        Write-Host "  Compiler failed with code $($compiler.ExitCode). Log: $compilerLog" -ForegroundColor Red
        exit $compiler.ExitCode
    }
    Write-Host "  Compiler artifacts ready; launching the runtime-only server." -ForegroundColor Green
}

$Artifact = [IO.Path]::ChangeExtension($Dme, ".d64")
$Artifact = (Resolve-Path -LiteralPath $Artifact).Path
$quotedArtifact = '"' + $Artifact + '"'
$quotedMap = '"' + $Map + '"'
$quotedSkin = '"' + $Skin + '"'
$clientArguments = @("--connect", "127.0.0.1:51664", "--skin", $quotedSkin)
$startupReplay = Join-Path $workspace "logs\monk-lobby-topic-fixed.d64r"
if (Test-Path -LiteralPath $startupReplay -PathType Leaf) {
    $clientArguments += @("--startup-replay", ('"' + $startupReplay + '"'))
    Write-Host "  Cached lobby splash enabled during live subsystem boot." -ForegroundColor Cyan
}
$server = $null
$client = $null

try {
    $server = Start-Process -FilePath $ServerExecutable `
        -ArgumentList @("boot", $quotedArtifact, $quotedMap) `
        -WorkingDirectory $workspace `
        -RedirectStandardError $serverLog `
        -WindowStyle Hidden `
        -PassThru
    $server.PriorityClass = [Diagnostics.ProcessPriorityClass]::AboveNormal

    # The listener is bound before validation, so the visible client can start
    # immediately and display preflight instead of an empty black window.
    $client = Start-Process -FilePath $ClientExecutable `
        -ArgumentList $clientArguments `
        -WorkingDirectory $workspace `
        -RedirectStandardError $clientLog `
        -PassThru
    $client.PriorityClass = [Diagnostics.ProcessPriorityClass]::AboveNormal

    Write-Host "  Client PID $($client.Id) | Server PID $($server.Id)" -ForegroundColor Yellow
    Write-Host "  Both are AboveNormal priority with all cores available." -ForegroundColor Green

    $lastProgress = ""
    while (-not $server.HasExited -and -not $client.HasExited) {
        Start-Sleep -Seconds 1
        $server.Refresh()
        $client.Refresh()
        if (Test-Path -LiteralPath $serverLog -PathType Leaf) {
            $progress = Get-Content -LiteralPath $serverLog -Tail 40 -ErrorAction SilentlyContinue |
                Where-Object { $_ -match '^(boot-progress|server-progress):' } |
                Select-Object -Last 1
            if ($progress -and $progress -ne $lastProgress) {
                $color = if ($progress -match 'cache=hit|headless ready|startup=accepting') {
                    'Green'
                } elseif ($progress -match 'cache=miss|store-failed|runtime-error|failed') {
                    'Yellow'
                } else {
                    'Cyan'
                }
                Write-Host "  $progress" -ForegroundColor $color
                $lastProgress = $progress
            }
        }
    }

    if ($server.HasExited) {
        Write-Host "  Server exited with code $($server.ExitCode)." -ForegroundColor Red
        if (-not $client.HasExited) {
            Stop-Process -Id $client.Id -ErrorAction SilentlyContinue
            Write-Host "  Closed the orphaned client to prevent a polling CPU loop." -ForegroundColor DarkYellow
        }
        Write-Host "  Server log: $serverLog" -ForegroundColor DarkGray
        exit $server.ExitCode
    }

    Write-Host "  Client closed; stopping the local server." -ForegroundColor DarkYellow
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
    }
    exit 0
} catch {
    Write-Host "  Visible boot failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($client -and -not $client.HasExited) {
        Stop-Process -Id $client.Id -ErrorAction SilentlyContinue
    }
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
    }
    exit 1
}
