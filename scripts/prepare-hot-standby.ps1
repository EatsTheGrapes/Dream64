param(
    [string] $ServerExecutable,
    [string] $CompilerExecutable,
    [string] $Dme = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\tgstation.dme",
    [string] $Map = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\_maps\map_files\generic\CentCom.dmm",
    [string] $DeploymentId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [UInt64] $RandomSeed = 0,
    [string] $StandbyAddress = "127.0.0.1:51665",
    [string] $RuntimeAddress = "127.0.0.1:51664"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ServerExecutable)) {
    $ServerExecutable = Join-Path $workspace "target\release\dream64-server.exe"
}
if ([string]::IsNullOrWhiteSpace($CompilerExecutable)) {
    $CompilerExecutable = Join-Path $workspace "target\release\dream64-compiler.exe"
}
$ServerExecutable = (Resolve-Path -LiteralPath $ServerExecutable).Path
$CompilerExecutable = (Resolve-Path -LiteralPath $CompilerExecutable).Path
$Dme = (Resolve-Path -LiteralPath $Dme).Path
$Map = (Resolve-Path -LiteralPath $Map).Path
if ($RandomSeed -eq 0) {
    $seedBytes = [byte[]]::new(8)
    [Security.Cryptography.RandomNumberGenerator]::Fill($seedBytes)
    $RandomSeed = [BitConverter]::ToUInt64($seedBytes, 0)
    if ($RandomSeed -eq 0) { $RandomSeed = 1 }
}

$logDirectory = Join-Path $workspace "logs"
[IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$compilerLog = Join-Path $logDirectory "prewarm-$DeploymentId-compiler.log"
$prewarmLog = Join-Path $logDirectory "prewarm-$DeploymentId-runtime.log"
$manifest = Join-Path $logDirectory "prewarm-$DeploymentId.json"
$quotedDme = '"' + $Dme + '"'
$quotedMap = '"' + $Map + '"'

Write-Host "  Dream64 background preparation $DeploymentId" -ForegroundColor Cyan
Write-Host "  Compiler and runtime are separate processes; both run below the live server priority." -ForegroundColor DarkGray
$compiler = Start-Process -FilePath $CompilerExecutable `
    -ArgumentList @($quotedDme) `
    -WorkingDirectory $workspace `
    -RedirectStandardError $compilerLog `
    -WindowStyle Hidden `
    -PassThru
$compiler.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal
$compiler.WaitForExit()
if ($compiler.ExitCode -ne 0) {
    throw "Compiler failed with code $($compiler.ExitCode). See $compilerLog"
}
$Artifact = [IO.Path]::ChangeExtension($Dme, ".d64")
$Artifact = (Resolve-Path -LiteralPath $Artifact).Path
$quotedArtifact = '"' + $Artifact + '"'

$env:DREAM64_PREWARM_READY_WORLD = "1"
Remove-Item Env:DREAM64_ACTIVATE_READY_WORLD -ErrorAction SilentlyContinue
Remove-Item Env:DREAM64_ENABLE_READY_WORLD_CACHE -ErrorAction SilentlyContinue
Remove-Item Env:DREAM64_DISABLE_READY_CACHE -ErrorAction SilentlyContinue
$env:DREAM64_RANDOM_SEED = [string]$RandomSeed
$env:DREAM64_DEPLOYMENT_ID = $DeploymentId
$env:DREAM64_PREWARM_STANDBY_ADDR = $StandbyAddress
$env:DREAM64_IPC_ADDR = $RuntimeAddress
$env:RAYON_NUM_THREADS = [string][Math]::Max(1, [Math]::Floor([Environment]::ProcessorCount / 4))

$standby = Start-Process -FilePath $ServerExecutable `
    -ArgumentList @("boot", $quotedArtifact, $quotedMap) `
    -WorkingDirectory $workspace `
    -RedirectStandardError $prewarmLog `
    -WindowStyle Hidden `
    -PassThru
$standby.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal

[ordered]@{
    deployment_id = $DeploymentId
    random_seed = $RandomSeed
    standby_pid = $standby.Id
    standby_address = $StandbyAddress
    runtime_address = $RuntimeAddress
    compiler_log = $compilerLog
    runtime_log = $prewarmLog
} | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding UTF8

Write-Host "  Standby PID $($standby.Id), seed $RandomSeed" -ForegroundColor Yellow
Write-Host "  It is initializing at BelowNormal priority on $($env:RAYON_NUM_THREADS) worker thread(s)." -ForegroundColor Green
Write-Host "  Manifest: $manifest" -ForegroundColor DarkGray
Write-Host "  Runtime log: $prewarmLog" -ForegroundColor DarkGray
