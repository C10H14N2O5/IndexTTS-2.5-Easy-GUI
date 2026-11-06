# Requires Windows PowerShell 5.1. No Git, package installation or elevation.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [ValidateSet('Update', 'Check', 'Rollback')][string]$Mode = 'Update',
    [switch]$Runner,
    [switch]$PauseWhenDone
)

$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop
Import-Module Microsoft.PowerShell.Management -ErrorAction Stop
$script:LogPath = $null
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)
$script:ScriptFolder = $PSScriptRoot

function Write-UpdateMessage([string]$Message) {
    Write-Host $Message
    if ($script:LogPath) {
        [System.IO.File]::AppendAllText($script:LogPath, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $Message + [Environment]::NewLine, $script:Utf8)
    }
}

function Assert-NoReparsePoint([string]$Path) {
    $cursor = [System.IO.Path]::GetFullPath($Path)
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Linked or redirected paths are not supported: $cursor"
            }
        }
        $parent = [System.IO.Path]::GetDirectoryName($cursor.TrimEnd('\', '/'))
        if (-not $parent -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
}

function Get-RootMutexName([string]$Kind, [string]$Folder) {
    $normalized = [System.IO.Path]::GetFullPath($Folder).Replace('/', '\').TrimEnd('\').ToLowerInvariant()
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $digest = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($normalized)) }
    finally { $sha.Dispose() }
    $hex = [System.BitConverter]::ToString($digest).Replace('-', '').ToLowerInvariant()
    return 'Local\IndexTTSEasyGUI-' + $Kind + '-' + $hex
}

function Enter-UpdateMutex([string]$Name) {
    $mutex = New-Object System.Threading.Mutex($false, $Name)
    try {
        try { $entered = $mutex.WaitOne(0) }
        catch [System.Threading.AbandonedMutexException] { $entered = $true }
        if (-not $entered) { throw 'Another GUI, installer or updater is using this deployment. Close it and retry.' }
        return $mutex
    }
    catch { $mutex.Dispose(); throw }
}

function Remove-OwnedDirectory([string]$Path, [string]$Parent) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $expected = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\')
    if ([System.IO.Path]::GetDirectoryName($full) -ne $expected -or [System.IO.Path]::GetFileName($full) -notmatch '^[0-9a-f]{32}$') {
        throw 'Refusing to clean an unexpected updater work directory.'
    }
    Assert-NoReparsePoint $full
    if (Test-Path -LiteralPath $full) {
        foreach ($entry in Get-ChildItem -LiteralPath $full -Recurse -Force) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Refusing to clean a linked updater work directory.'
            }
        }
        Remove-Item -LiteralPath $full -Recurse -Force
    }
}

function Invoke-UpdateHttp([hashtable]$Parameters, [string]$Destination = '') {
    if ($Destination) {
        Microsoft.PowerShell.Utility\Invoke-WebRequest @Parameters -UseBasicParsing -OutFile $Destination | Out-Null
        return
    }
    return Microsoft.PowerShell.Utility\Invoke-RestMethod @Parameters
}

function Invoke-GitHubRequest([string]$Uri, [string]$Destination = '') {
    # Fixed public GitHub URLs; no credentials or arbitrary repository URLs.
    $parameters = @{
        Uri = $Uri
        TimeoutSec = 60
        Headers = @{ Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28' }
        UserAgent = 'IndexTTS-EasyGUI-ZipUpdater'
        ErrorAction = 'Stop'
    }
    $proxy = $env:HTTPS_PROXY
    if (-not $proxy) { $proxy = $env:HTTP_PROXY }
    if ($proxy) { $parameters['Proxy'] = $proxy }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            if ($Destination) {
                Invoke-UpdateHttp $parameters $Destination
                if (-not (Test-Path -LiteralPath $Destination) -or (Get-Item -LiteralPath $Destination).Length -eq 0) {
                    throw 'Empty download.'
                }
                return
            }
            return Invoke-UpdateHttp $parameters
        }
        catch {
            # Do not print the exception: proxy URLs can include credentials.
            if ($Destination -and (Test-Path -LiteralPath $Destination)) { Remove-Item -LiteralPath $Destination -Force }
            Write-UpdateMessage "GitHub request failed (attempt $attempt of 3). Check connectivity, proxy settings or API rate limits."
            if ($attempt -eq 3) { throw 'GitHub could not be reached after three attempts. Program files were not changed.' }
            Start-Sleep -Seconds ([Math]::Min(2 * $attempt, 4))
        }
    }
}

function Invoke-Core([string[]]$Arguments) {
    $command = @('-X', 'utf8', '-B', (Join-Path $script:ScriptFolder 'update_core.py'), '--root', $script:RootPath) + $Arguments
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 turns native stderr into ErrorRecord objects.
        # A benign Python warning must not abort an otherwise successful check.
        $ErrorActionPreference = 'Continue'
        & $script:Python @command 2>&1 | ForEach-Object { Write-UpdateMessage ([string]$_) }
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    if ($code -ne 0) { throw "Updater verification or transaction failed (exit $code). See the log and any recovery instructions above." }
}

function Save-RecoverySnapshot([string]$StateDir) {
    if (Test-Path -LiteralPath (Join-Path $StateDir 'transaction.json')) {
        throw 'Cannot replace recovery files while an update transaction is unfinished.'
    }
    $recovery = Join-Path $StateDir 'recovery'
    Assert-NoReparsePoint $recovery
    New-Item -ItemType Directory -Path $recovery -Force | Out-Null
    foreach ($file in @('update.ps1', 'update_core.py', 'runtime_lock.py')) {
        $source = Join-Path $script:ScriptFolder $file
        $destination = Join-Path $recovery $file
        Assert-NoReparsePoint $source
        Assert-NoReparsePoint $destination
        $pending = $destination + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
        $previous = $destination + '.' + [guid]::NewGuid().ToString('N') + '.old'
        try {
            $inputStream = [System.IO.File]::OpenRead($source)
            try {
                $outputStream = [System.IO.File]::Open($pending, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
                try { $inputStream.CopyTo($outputStream); $outputStream.Flush($true) }
                finally { $outputStream.Dispose() }
            }
            finally { $inputStream.Dispose() }
            # PowerShell 5.1 coerces the null backup parameter of File.Replace
            # into an invalid empty filename. A unique backup avoids that binder
            # issue while retaining atomic publication of the known-good bytes.
            if (Test-Path -LiteralPath $destination) { [System.IO.File]::Replace($pending, $destination, $previous) }
            else { [System.IO.File]::Move($pending, $destination) }
            $durable = [System.IO.File]::Open($destination, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            try { $durable.Flush($true) }
            finally { $durable.Dispose() }
        }
        finally {
            if (Test-Path -LiteralPath $pending) { Remove-Item -LiteralPath $pending -Force }
            if (Test-Path -LiteralPath $previous) { Remove-Item -LiteralPath $previous -Force }
        }
    }
}

function Invoke-Update {
    $script:RootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    Assert-NoReparsePoint $script:RootPath
    if (-not (Test-Path -LiteralPath $script:RootPath -PathType Container)) { throw 'Deployment folder does not exist.' }
    if (Test-Path -LiteralPath (Join-Path $script:RootPath '.git')) { throw 'ZIP updates cannot run in a Git working tree. Use your normal Git workflow in the source repository.' }
    $script:Python = Join-Path $script:RootPath '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $script:Python -PathType Leaf)) { throw 'The existing project Python environment is required. Run 1- install.bat first.' }
    $stateDir = Join-Path $script:RootPath '.runtime\update'
    Assert-NoReparsePoint $stateDir
    $mutex = Enter-UpdateMutex (Get-RootMutexName 'update' $script:RootPath)
    try {
        # Probe before downloads. The core reacquires the same app mutex before
        # any baseline/rollback/update operation, closing the probe/apply race.
        $appMutex = Enter-UpdateMutex (Get-RootMutexName 'app' $script:RootPath)
        $appMutex.ReleaseMutex()
        $appMutex.Dispose()
        if (-not $Runner) {
            if (Test-Path -LiteralPath (Join-Path $stateDir 'transaction.json')) {
                $recoveryScript = Join-Path $stateDir 'recovery\update.ps1'
                Assert-NoReparsePoint $recoveryScript
                if (-not (Test-Path -LiteralPath $recoveryScript -PathType Leaf)) { throw 'Known-good recovery files are missing. Keep the update backups intact.' }
                & $recoveryScript -Root $script:RootPath -Mode $Mode -Runner
                if ($LASTEXITCODE -ne 0) { throw 'Recovery did not complete. Keep the update backups and transaction record intact.' }
                return
            }
            $runnerParent = Join-Path $stateDir 'runner'
            Assert-NoReparsePoint $runnerParent
            New-Item -ItemType Directory -Path $runnerParent -Force | Out-Null
            $runnerDir = Join-Path $runnerParent ([guid]::NewGuid().ToString('N'))
            New-Item -ItemType Directory -Path $runnerDir | Out-Null
            try {
                foreach ($file in @('update.ps1', 'update_core.py', 'runtime_lock.py')) {
                    $source = Join-Path $script:ScriptFolder $file
                    Assert-NoReparsePoint $source
                    Copy-Item -LiteralPath $source -Destination (Join-Path $runnerDir $file)
                }
                # Same PowerShell thread: the named mutex is reentrant. Both
                # the copied script and this already-parsed wrapper stay stable.
                & (Join-Path $runnerDir 'update.ps1') -Root $script:RootPath -Mode $Mode -Runner
                if ($LASTEXITCODE -ne 0) { throw 'The update did not complete. Review the messages above.' }
            }
            finally { Remove-OwnedDirectory $runnerDir $runnerParent }
            return
        }

        $expectedRunnerParent = Join-Path $stateDir 'runner'
        $recoveryDir = Join-Path $stateDir 'recovery'
        $isRecoveryRunner = $script:ScriptFolder -eq $recoveryDir
        $isPrivateRunner = [System.IO.Path]::GetDirectoryName($script:ScriptFolder) -eq $expectedRunnerParent -and
            [System.IO.Path]::GetFileName($script:ScriptFolder) -match '^[0-9a-f]{32}$'
        if (-not $isRecoveryRunner -and -not $isPrivateRunner) {
            throw 'The update must run from its private working copy.'
        }
        $logDir = Join-Path $stateDir 'logs'
        Assert-NoReparsePoint $logDir
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        $script:LogPath = Join-Path $logDir ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N') + '.log')
        Write-UpdateMessage "Mode: $Mode. Deployment: $script:RootPath"
        Write-UpdateMessage "Log: $script:LogPath"
        if ($Mode -eq 'Check') {
            if (Test-Path -LiteralPath (Join-Path $stateDir 'transaction.json')) {
                throw 'An unfinished update needs recovery. Run 3- update.bat without --check first.'
            }
        }
        else { Invoke-Core @('--recover') }
        if ($isRecoveryRunner) {
            Write-UpdateMessage 'Recovery completed. Run 3- update.bat again to check or install updates.'
            return
        }
        if ($Mode -eq 'Rollback') {
            Save-RecoverySnapshot $stateDir
            Invoke-Core @('--rollback')
            return
        }
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12
        $repository = 'C10H14N2O5/IndexTTS-2.5-Easy-GUI'
        $metadata = Invoke-GitHubRequest "https://api.github.com/repos/$repository/commits/main"
        $sha = [string]$metadata.sha
        if ($sha -notmatch '^[0-9a-fA-F]{40}$') { throw 'GitHub returned an invalid commit identifier.' }
        $sha = $sha.ToLowerInvariant()
        $current = 'uninitialized'
        $currentFile = Join-Path $stateDir 'current.json'
        if (Test-Path -LiteralPath $currentFile) {
            $record = Get-Content -LiteralPath $currentFile -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($record.commit) { $current = [string]$record.commit }
        }
        Write-UpdateMessage "Installed commit: $current"
        Write-UpdateMessage "GitHub main commit: $sha"
        if ($current -eq $sha) {
            Invoke-Core @('--verify')
            Write-UpdateMessage 'Already up to date. No program files changed.'
            return
        }
        $downloadParent = Join-Path $stateDir 'downloads'
        Assert-NoReparsePoint $downloadParent
        New-Item -ItemType Directory -Path $downloadParent -Force | Out-Null
        $downloadDir = Join-Path $downloadParent ([guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $downloadDir | Out-Null
        try {
            $archive = Join-Path $downloadDir 'update.zip'
            Write-UpdateMessage 'Downloading the fixed-commit source archive...'
            Invoke-GitHubRequest "https://api.github.com/repos/$repository/zipball/$sha" $archive
            $coreArguments = @('--archive', $archive, '--commit', $sha)
            if ($Mode -eq 'Check') { $coreArguments += '--check' }
            else { Save-RecoverySnapshot $stateDir }
            Invoke-Core $coreArguments
            if ($Mode -eq 'Update') { Write-UpdateMessage 'Update complete. Restart the GUI. Use 3- update.bat --rollback to restore the previous program version.' }
        }
        finally { Remove-OwnedDirectory $downloadDir $downloadParent }
    }
    finally {
        $mutex.ReleaseMutex()
        $mutex.Dispose()
    }
}

$exitCode = 0
try { Invoke-Update }
catch {
    $exitCode = 1
    # Known local/transaction errors contain useful paths, never HTTP credentials.
    Write-UpdateMessage ('ERROR: ' + $_.Exception.Message)
}
finally {
    if ($PauseWhenDone) { [void](Read-Host 'Press Enter to close') }
}
exit $exitCode
