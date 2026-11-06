param([Parameter(Mandatory = $true)][string]$Root)
$ErrorActionPreference = 'Stop'
$lease = $null
$acquired = $false
try {
    $Root = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $canonical = $Root.Replace('/', '\').ToLowerInvariant()
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $suffix = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
    $lease = New-Object Threading.Mutex($false, ('Local\IndexTTSEasyGUI-app-' + $suffix))
    try { $acquired = $lease.WaitOne(0) } catch [Threading.AbandonedMutexException] { $acquired = $true }
    if (-not $acquired) { throw 'The GUI, installer or updater is already running for this directory.' }
    if (Test-Path -LiteralPath (Join-Path $Root '.runtime\update\transaction.json')) {
        throw 'An interrupted update needs recovery. Run 3- update.bat first.'
    }
    $installer = Join-Path $Root 'tools\install_runtime.bat'
    if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) { throw 'Installation script is missing.' }
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $env:ComSpec
    $start.Arguments = '/d /s /c ""' + $installer + '""'
    $start.WorkingDirectory = $Root
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $process = [Diagnostics.Process]::Start($start)
    try { $process.WaitForExit(); $result = $process.ExitCode }
    finally { $process.Dispose() }
} catch {
    Write-Host ('[ERROR] ' + $_.Exception.Message) -ForegroundColor Red
    $result = 1
} finally {
    if ($acquired) { $lease.ReleaseMutex() }
    if ($null -ne $lease) { $lease.Dispose() }
}
exit $result
