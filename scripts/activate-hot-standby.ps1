param(
    [Parameter(Mandatory = $true)]
    [string] $DeploymentId,
    [Parameter(Mandatory = $true)]
    [int] $OldServerPid,
    [string] $StandbyAddress = "127.0.0.1:51665"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$generationRoot = Join-Path $workspace "target\dream64-generations"
$generationPath = Join-Path $generationRoot $DeploymentId
$currentManifest = Join-Path $generationRoot "current.json"
if (-not (Test-Path -LiteralPath $currentManifest -PathType Leaf)) {
    throw "No selected Dream64 generation manifest exists"
}
$selected = Get-Content -LiteralPath $currentManifest -Raw | ConvertFrom-Json
if ($selected.deployment_id -ne $DeploymentId) {
    throw "Selected generation is $($selected.deployment_id), not $DeploymentId"
}
if (-not $PSBoundParameters.ContainsKey("StandbyAddress")) {
    $StandbyAddress = [string]$selected.standby_address
}
if (-not (Test-Path -LiteralPath $generationPath -PathType Container)) {
    throw "Generation directory does not exist: $generationPath"
}

function Write-AtomicJson([string] $Path, $Value) {
    $temporary = "$Path.tmp.$PID"
    $Value | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding UTF8
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        [IO.File]::Replace($temporary, $Path, $null)
    } else {
        [IO.File]::Move($temporary, $Path)
    }
}

$pendingLease = Join-Path $generationPath "activation.pending.json"
$activeLease = Join-Path $generationPath "active.json"
$lease = [ordered]@{
    deployment_id = $DeploymentId
    server_pid = [int]$selected.standby_pid
    artifact = [string]$selected.artifact
    requested_at = (Get-Date).ToString("o")
}
Write-AtomicJson $pendingLease $lease

$hostName, $portText = $StandbyAddress.Split(':', 2)
$client = [Net.Sockets.TcpClient]::new()
try {
    $client.Connect($hostName, [int]$portText)
    $writer = [IO.StreamWriter]::new($client.GetStream(), [Text.Encoding]::ASCII, 1024, $true)
    $writer.NewLine = "`n"
    $writer.WriteLine("ACTIVATE $DeploymentId")
    $writer.Flush()
    $lease["activated_at"] = (Get-Date).ToString("o")
    Write-AtomicJson $activeLease $lease
    Remove-Item -LiteralPath $pendingLease -Force -ErrorAction SilentlyContinue
} catch {
    Remove-Item -LiteralPath $pendingLease -Force -ErrorAction SilentlyContinue
    throw
} finally {
    $client.Dispose()
}

# The prepared process is already retrying the runtime port. Releasing the old
# owner after activation minimizes the interval in which neither process owns it.
Stop-Process -Id $OldServerPid -ErrorAction Stop
Write-Host "  Handoff requested for $DeploymentId; old server PID $OldServerPid released." -ForegroundColor Green
