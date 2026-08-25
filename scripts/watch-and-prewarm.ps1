param(
    [string] $Dme = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\tgstation.dme",
    [string] $Map = "C:\Users\Administrator\Desktop\RBMK Project\Monkestation2.0\_maps\map_files\generic\CentCom.dmm",
    [string] $ServerExecutable,
    [string] $CompilerExecutable,
    [int] $DebounceMilliseconds = 1500,
    [int] $ReadyTimeoutSeconds = 7200,
    [double] $MinimumFreeGigabytes = 2.0,
    [switch] $BuildImmediately,
    [switch] $ValidateOnly
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$prepareScript = (Resolve-Path (Join-Path $PSScriptRoot "prepare-hot-standby.ps1")).Path
$Dme = (Resolve-Path -LiteralPath $Dme).Path
$Map = (Resolve-Path -LiteralPath $Map).Path
$sourceRoot = Split-Path -Parent $Dme
if ([string]::IsNullOrWhiteSpace($ServerExecutable)) {
    $ServerExecutable = Join-Path $workspace "target\release\dream64-server.exe"
}
if ([string]::IsNullOrWhiteSpace($CompilerExecutable)) {
    $CompilerExecutable = Join-Path $workspace "target\release\dream64-compiler.exe"
}
$ServerExecutable = (Resolve-Path -LiteralPath $ServerExecutable).Path
$CompilerExecutable = (Resolve-Path -LiteralPath $CompilerExecutable).Path

if ($DebounceMilliseconds -lt 100) {
    throw "DebounceMilliseconds must be at least 100"
}
if ($ReadyTimeoutSeconds -lt 1) {
    throw "ReadyTimeoutSeconds must be positive"
}
if ($MinimumFreeGigabytes -lt 0.5) {
    throw "MinimumFreeGigabytes must be at least 0.5"
}

$generationDirectory = Join-Path $workspace "target\dream64-generations"
$logDirectory = Join-Path $workspace "logs"
[IO.Directory]::CreateDirectory($generationDirectory) | Out-Null
[IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$currentManifest = Join-Path $generationDirectory "current.json"
$watchdogLog = Join-Path $logDirectory "deployment-watchdog.log"

function Write-WatchdogLog([string] $Message, [ConsoleColor] $Color = [ConsoleColor]::Gray) {
    $line = "{0:o} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $watchdogLog -Value $line -Encoding UTF8
    Write-Host "  $Message" -ForegroundColor $Color
}

function Get-RelativeSourcePath([string] $Path) {
    $root = [IO.Path]::GetFullPath($sourceRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $candidate = [IO.Path]::GetFullPath($Path)
    if (-not $candidate.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { return $null }
    return $candidate.Substring($root.Length)
}

function Test-RelevantChange([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    $relative = Get-RelativeSourcePath $Path
    if ($null -eq $relative) { return $false }
    $segments = $relative -split '[\\/]'
    $ignoredDirectories = @(
        ".git", ".github", ".idea", ".vscode", "node_modules", "target",
        "data", "logs", "tmp", "temp", ".cache"
    )
    foreach ($segment in $segments) {
        if ($ignoredDirectories -contains $segment.ToLowerInvariant()) { return $false }
    }
    $name = [IO.Path]::GetFileName($Path)
    if ($name.StartsWith(".") -or $name.EndsWith("~") -or $name.EndsWith(".tmp")) {
        return $false
    }
    $extension = [IO.Path]::GetExtension($Path).ToLowerInvariant()
    return $extension -in @(
        ".dm", ".dme", ".dmm", ".dmf", ".dmi",
        ".json", ".toml", ".txt", ".yml", ".yaml",
        ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".ogg", ".wav", ".mp3", ".mid", ".midi",
        ".html", ".htm", ".css", ".js", ".tsx", ".ts"
    )
}

function New-RandomSeed {
    $bytes = [byte[]]::new(8)
    [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $seed = [BitConverter]::ToUInt64($bytes, 0)
    if ($seed -eq 0) { return [UInt64]1 }
    return $seed
}

function Get-FreeLoopbackAddress {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        $port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
        return "127.0.0.1:$port"
    } finally {
        $listener.Stop()
    }
}

function Publish-Generation($Manifest) {
    $temporary = "$currentManifest.tmp.$PID"
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding UTF8
    if (Test-Path -LiteralPath $currentManifest -PathType Leaf) {
        [IO.File]::Replace($temporary, $currentManifest, $null)
    } else {
        [IO.File]::Move($temporary, $currentManifest)
    }
}

function Remove-GenerationDirectory([IO.DirectoryInfo] $Directory) {
    $root = [IO.Path]::GetFullPath($generationDirectory).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $target = [IO.Path]::GetFullPath($Directory.FullName)
    if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove generation outside $generationDirectory"
    }
    Remove-Item -LiteralPath $target -Recurse -Force
    foreach ($suffix in @(".json", "-compiler.log", "-runtime.log")) {
        $logPath = Join-Path $logDirectory ("prewarm-{0}{1}" -f $Directory.Name, $suffix)
        Remove-Item -LiteralPath $logPath -Force -ErrorAction SilentlyContinue
    }
    Write-WatchdogLog "Removed retired generation $($Directory.Name)." DarkGray
}

function Send-StandbyCancellation($Manifest) {
    try {
        $hostName, $portText = ([string]$Manifest.standby_address).Split(':', 2)
        $client = [Net.Sockets.TcpClient]::new()
        $connect = $client.BeginConnect($hostName, [int]$portText, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne(500)) {
            $client.Dispose()
            return
        }
        $client.EndConnect($connect)
        try {
            $writer = [IO.StreamWriter]::new($client.GetStream(), [Text.Encoding]::ASCII, 1024, $true)
            $writer.NewLine = "`n"
            $writer.WriteLine("CANCEL $($Manifest.deployment_id)")
            $writer.Flush()
        } finally {
            $client.Dispose()
        }
        Write-WatchdogLog "Cancelled superseded standby $($Manifest.deployment_id)." DarkGray
    } catch {
        # A closed control socket normally means the generation already activated.
    }
}

function Reconcile-Generations {
    $selectedArtifact = $null
    if (Test-Path -LiteralPath $currentManifest -PathType Leaf) {
        try {
            $selected = Get-Content -LiteralPath $currentManifest -Raw | ConvertFrom-Json
            $selectedArtifact = [IO.Path]::GetFullPath([string]$selected.artifact)
        } catch {}
    }
    foreach ($directory in @(Get-ChildItem -LiteralPath $generationDirectory -Directory -ErrorAction SilentlyContinue)) {
        $artifact = Join-Path $directory.FullName "tgstation.d64"
        if ($null -ne $selectedArtifact -and
            [string]::Equals([IO.Path]::GetFullPath($artifact), $selectedArtifact, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if (Test-Path -LiteralPath (Join-Path $directory.FullName "activation.pending.json") -PathType Leaf) {
            continue
        }
        $activeLeasePath = Join-Path $directory.FullName "active.json"
        if (Test-Path -LiteralPath $activeLeasePath -PathType Leaf) {
            try {
                $activeLease = Get-Content -LiteralPath $activeLeasePath -Raw | ConvertFrom-Json
                $activeProcess = Get-Process -Id ([int]$activeLease.server_pid) -ErrorAction SilentlyContinue
                if ($null -ne $activeProcess -and $activeProcess.ProcessName -eq "dream64-server") {
                    continue
                }
            } catch {}
            Remove-GenerationDirectory $directory
            continue
        }
        $prewarmManifestPath = Join-Path $logDirectory ("prewarm-$($directory.Name).json")
        if (-not (Test-Path -LiteralPath $prewarmManifestPath -PathType Leaf)) {
            if ($directory.CreationTime -lt (Get-Date).AddHours(-3)) {
                Remove-GenerationDirectory $directory
            }
            continue
        }
        try {
            $prewarmManifest = Get-Content -LiteralPath $prewarmManifestPath -Raw | ConvertFrom-Json
            $process = Get-Process -Id ([int]$prewarmManifest.standby_pid) -ErrorAction SilentlyContinue
            if ($null -eq $process) {
                Remove-GenerationDirectory $directory
                continue
            }
            $activated = (Test-Path -LiteralPath $prewarmManifest.runtime_log -PathType Leaf) -and
                (Select-String -LiteralPath $prewarmManifest.runtime_log -Pattern 'prewarmed standby activation accepted' -Quiet -ErrorAction SilentlyContinue)
            if ($activated) { continue }
            Send-StandbyCancellation $prewarmManifest
        } catch {
            Write-WatchdogLog "Could not reconcile generation $($directory.Name): $($_.Exception.Message)" DarkYellow
        }
    }
}

function Start-Generation([string[]] $ChangedPaths) {
    $deploymentId = "watch-{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss"), ([Guid]::NewGuid().ToString("N").Substring(0, 8))
    $seed = New-RandomSeed
    $standbyAddress = Get-FreeLoopbackAddress
    $runtimeAddress = Get-FreeLoopbackAddress
    $generationPath = Join-Path $generationDirectory $deploymentId
    [IO.Directory]::CreateDirectory($generationPath) | Out-Null
    $artifactPath = Join-Path $generationPath "tgstation.d64"
    $reuseArtifact = $null
    if (Test-Path -LiteralPath $currentManifest -PathType Leaf) {
        try {
            $current = Get-Content -LiteralPath $currentManifest -Raw | ConvertFrom-Json
            if (Test-Path -LiteralPath $current.artifact -PathType Leaf) {
                $reuseArtifact = [string]$current.artifact
            }
        } catch {}
    }
    $generationDrive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($generationDirectory))
    $priorArtifactBytes = if ($null -ne $reuseArtifact) {
        (Get-Item -LiteralPath $reuseArtifact).Length
    } else {
        0
    }
    $requiredFreeBytes = [int64](($MinimumFreeGigabytes * 1GB) + $priorArtifactBytes + 256MB)
    if ($generationDrive.AvailableFreeSpace -lt $requiredFreeBytes) {
        $freeGb = [Math]::Round($generationDrive.AvailableFreeSpace / 1GB, 2)
        $requiredGb = [Math]::Round($requiredFreeBytes / 1GB, 2)
        Write-WatchdogLog "Generation $deploymentId refused: ${freeGb} GB free, ${requiredGb} GB required; current generation retained." Red
        return $false
    }
    $manifestPath = Join-Path $logDirectory "prewarm-$deploymentId.json"
    $quotedPrepare = '"' + $prepareScript + '"'
    $quotedDme = '"' + $Dme + '"'
    $quotedMap = '"' + $Map + '"'
    $quotedServer = '"' + $ServerExecutable + '"'
    $quotedCompiler = '"' + $CompilerExecutable + '"'
    $arguments = @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $quotedPrepare,
        "-Dme", $quotedDme, "-Map", $quotedMap,
        "-ServerExecutable", $quotedServer, "-CompilerExecutable", $quotedCompiler,
        "-Artifact", ('"' + $artifactPath + '"'),
        "-DeploymentId", $deploymentId, "-RandomSeed", [string]$seed,
        "-StandbyAddress", $standbyAddress, "-RuntimeAddress", $runtimeAddress
    )
    if ($null -ne $reuseArtifact) {
        $arguments += @("-ReuseArtifact", ('"' + $reuseArtifact + '"'))
    }

    $changeSummary = if ($ChangedPaths.Count -eq 0) {
        "initial generation"
    } else {
        "$($ChangedPaths.Count) changed path(s)"
    }
    Write-WatchdogLog "Building $deploymentId after $changeSummary." Cyan
    $prepare = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments `
        -WorkingDirectory $workspace -WindowStyle Hidden -Wait -PassThru
    if ($prepare.ExitCode -ne 0) {
        Write-WatchdogLog "Generation $deploymentId failed during compile/prewarm launch (exit $($prepare.ExitCode)); current generation retained." Red
        return $false
    }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        Write-WatchdogLog "Generation $deploymentId produced no manifest; current generation retained." Red
        return $false
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        $standby = Get-Process -Id ([int]$manifest.standby_pid) -ErrorAction SilentlyContinue
        if ($null -eq $standby) {
            Write-WatchdogLog "Generation $deploymentId exited before becoming ready; current generation retained." Red
            return $false
        }
        if (Test-Path -LiteralPath $manifest.runtime_log -PathType Leaf) {
            $ready = Select-String -LiteralPath $manifest.runtime_log `
                -Pattern 'prewarmed standby ready' -Quiet -ErrorAction SilentlyContinue
            if ($ready) { break }
        }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) {
        Stop-Process -Id ([int]$manifest.standby_pid) -ErrorAction SilentlyContinue
        Write-WatchdogLog "Generation $deploymentId exceeded the readiness timeout; current generation retained." Red
        return $false
    }

    $previous = $null
    if (Test-Path -LiteralPath $currentManifest -PathType Leaf) {
        try { $previous = Get-Content -LiteralPath $currentManifest -Raw | ConvertFrom-Json } catch {}
    }
    $published = [ordered]@{
        deployment_id = $manifest.deployment_id
        random_seed = $manifest.random_seed
        standby_pid = $manifest.standby_pid
        standby_address = $manifest.standby_address
        runtime_address = $manifest.runtime_address
        compiler_log = $manifest.compiler_log
        runtime_log = $manifest.runtime_log
        artifact = $manifest.artifact
        source_root = $sourceRoot
        changed_paths = $ChangedPaths
        ready_at = (Get-Date).ToString("o")
    }
    Publish-Generation $published
    Write-WatchdogLog "Promoted ready generation $deploymentId; active round remains untouched." Green
    Reconcile-Generations
    return $true
}

if ($ValidateOnly) {
    if (-not (Test-RelevantChange $Dme)) { throw "DME filter validation failed" }
    if (-not (Test-RelevantChange $Map)) { throw "DMM filter validation failed" }
    if (Test-RelevantChange (Join-Path $sourceRoot "tgstation.d64")) {
        throw "Generated .d64 files must not retrigger the watchdog"
    }
    Write-WatchdogLog "Validation passed for $sourceRoot." Green
    exit 0
}

$watcher = [IO.FileSystemWatcher]::new($sourceRoot)
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [IO.NotifyFilters]'FileName, DirectoryName, LastWrite, Size, CreationTime'
$watcher.InternalBufferSize = 65536
$watcher.EnableRaisingEvents = $true
$eventPrefix = "Dream64Watchdog.$PID"
$subscriptions = @()
$subscriptions += Register-ObjectEvent $watcher Changed -SourceIdentifier "$eventPrefix.Changed"
$subscriptions += Register-ObjectEvent $watcher Created -SourceIdentifier "$eventPrefix.Created"
$subscriptions += Register-ObjectEvent $watcher Deleted -SourceIdentifier "$eventPrefix.Deleted"
$subscriptions += Register-ObjectEvent $watcher Renamed -SourceIdentifier "$eventPrefix.Renamed"

Write-Host ""
Write-Host "  DREAM64 DEPLOYMENT WATCHDOG" -ForegroundColor Magenta
Write-WatchdogLog "Watching $sourceRoot (debounce ${DebounceMilliseconds}ms)." Green
Write-WatchdogLog "Successful builds prewarm in isolation; failed generations never replace current.json." DarkGray

try {
    $nextReconcile = Get-Date
    if ($BuildImmediately) {
        [void](Start-Generation @())
    }
    while ($true) {
        if ((Get-Date) -ge $nextReconcile) {
            Reconcile-Generations
            $nextReconcile = (Get-Date).AddSeconds(15)
        }
        $first = Wait-Event -Timeout 1
        if ($null -eq $first) { continue }
        $changed = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        $lastRelevant = Get-Date
        $events = @($first)
        while ($true) {
            foreach ($eventRecord in $events) {
                $path = [string]$eventRecord.SourceEventArgs.FullPath
                if (Test-RelevantChange $path) {
                    [void]$changed.Add($path)
                    $lastRelevant = Get-Date
                }
                Remove-Event -EventIdentifier $eventRecord.EventIdentifier -ErrorAction SilentlyContinue
            }
            if (((Get-Date) - $lastRelevant).TotalMilliseconds -ge $DebounceMilliseconds) { break }
            Start-Sleep -Milliseconds 100
            $events = @(Get-Event | Where-Object SourceIdentifier -Like "$eventPrefix.*")
        }
        if ($changed.Count -eq 0) { continue }
        $relativeChanges = @($changed | ForEach-Object { Get-RelativeSourcePath $_ } | Sort-Object)
        Write-WatchdogLog "Detected stable source change set: $($relativeChanges -join ', ')" Yellow
        [void](Start-Generation $relativeChanges)
    }
} finally {
    foreach ($subscription in $subscriptions) {
        Unregister-Event -SubscriptionId $subscription.Id -ErrorAction SilentlyContinue
    }
    $watcher.Dispose()
    Get-Event | Where-Object SourceIdentifier -Like "$eventPrefix.*" | Remove-Event -ErrorAction SilentlyContinue
}
