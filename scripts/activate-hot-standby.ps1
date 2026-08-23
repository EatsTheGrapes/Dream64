param(
    [Parameter(Mandatory = $true)]
    [string] $DeploymentId,
    [Parameter(Mandatory = $true)]
    [int] $OldServerPid,
    [string] $StandbyAddress = "127.0.0.1:51665"
)

$ErrorActionPreference = "Stop"
$hostName, $portText = $StandbyAddress.Split(':', 2)
$client = [Net.Sockets.TcpClient]::new()
try {
    $client.Connect($hostName, [int]$portText)
    $writer = [IO.StreamWriter]::new($client.GetStream(), [Text.Encoding]::ASCII, 1024, $true)
    $writer.NewLine = "`n"
    $writer.WriteLine("ACTIVATE $DeploymentId")
    $writer.Flush()
} finally {
    $client.Dispose()
}

# The prepared process is already retrying the runtime port. Releasing the old
# owner after activation minimizes the interval in which neither process owns it.
Stop-Process -Id $OldServerPid -ErrorAction Stop
Write-Host "  Handoff requested for $DeploymentId; old server PID $OldServerPid released." -ForegroundColor Green
