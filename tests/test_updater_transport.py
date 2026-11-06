"""Exercise the real PowerShell 5.1 updater with offline transport fixtures."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == 'nt' and POWERSHELL.is_file(), 'Windows PowerShell 5.1 required')
class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='updater transport 中文 ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / 'scenario.ps1'

    def run_ps(self, body, setup=True):
        preamble = "$ErrorActionPreference='Stop'\nImport-Module Microsoft.PowerShell.Utility\nImport-Module Microsoft.PowerShell.Management\n"
        if setup:
            preamble += f"""
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile({ps_quote(ROOT/'tools/update.ps1')},[ref]$tokens,[ref]$errors)
if($errors.Count){{throw $errors[0]}}
foreach($f in $ast.FindAll({{param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]}},$false)){{Invoke-Expression $f.Extent.Text}}
$Root={ps_quote(self.root/'deployment')}
New-Item -ItemType Directory -Path (Join-Path $Root '.venv\\Scripts') -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $Root '.venv\\Scripts\\python.exe'),'stub')
$script:ScriptFolder=Join-Path $Root ('.runtime\\update\\runner\\'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $script:ScriptFolder -Force | Out-Null
foreach($file in @('update.ps1','update_core.py','runtime_lock.py')){{[IO.File]::WriteAllText((Join-Path $script:ScriptFolder $file),'snapshot fixture')}}
$Runner=$true
$script:Utf8=New-Object Text.UTF8Encoding($false)
$script:LogPath=$null
$script:Calls=@();$script:Requests=@();$script:Messages=@()
function Write-UpdateMessage([string]$Message){{$script:Messages+=,$Message}}
function Invoke-Core([string[]]$Arguments){{$script:Calls+=,@($Arguments)}}
function Invoke-GitHubRequest([string]$Uri,[string]$Destination=''){{
 $script:Requests+=,$Uri
 if($Destination){{[IO.File]::WriteAllText($Destination,'archive stub');return}}
 return @{{sha=('a'*40)}}
}}
"""
        self.script.write_text(preamble + body + "\nWrite-Output 'SCENARIO_OK'\n", encoding='utf-8-sig')
        result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(self.script)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode('utf-8', 'replace'))
        self.assertIn(b'SCENARIO_OK', result.stdout)

    def test_check_is_read_only_and_downloads_fixed_commit(self):
        self.run_ps("""
$Mode='Check';Invoke-Update
if($script:Calls.Count -ne 1 -or $script:Calls[0][-1] -ne '--check'){throw 'Check mutated or incorrect arguments'}
if($script:Requests.Count -ne 2 -or $script:Requests[1] -notmatch '/zipball/a{40}$'){throw 'Archive is not pinned'}
if(Test-Path (Join-Path $Root '.runtime/update/current.json')){throw 'Check initialized baseline'}
if(@(Get-ChildItem (Join-Path $Root '.runtime/update/downloads') -Force).Count){throw 'Download not cleaned'}
""")

    def test_update_recovers_before_apply_and_rollback_is_offline(self):
        self.run_ps("""
$Mode='Update';Invoke-Update
if($script:Calls.Count -ne 2 -or $script:Calls[0][0] -ne '--recover' -or $script:Calls[1][0] -ne '--archive'){throw 'Update sequence'}
$script:Calls=@();$script:Requests=@();$Mode='Rollback';Invoke-Update
if($script:Calls.Count -ne 2 -or $script:Calls[1][0] -ne '--rollback' -or $script:Requests.Count){throw 'Rollback network or sequence'}
""")

    def test_no_change_does_not_download_and_check_does_not_recover(self):
        self.run_ps("""
[IO.File]::WriteAllText((Join-Path $Root '.runtime/update/current.json'),'{"commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}')
$Mode='Check';Invoke-Update
if($script:Calls.Count -ne 1 -or $script:Calls[0][0] -ne '--verify' -or $script:Requests.Count -ne 1){throw 'No-op did not verify or downloaded archive'}
[IO.File]::WriteAllText((Join-Path $Root '.runtime/update/transaction.json'),'{}')
$script:Requests=@();$script:Calls=@();$failed=$false
try{Invoke-Update}catch{$failed=$true}
if(-not $failed -or $script:Requests.Count -or $script:Calls.Count){throw 'Check recovered unfinished transaction'}
""")

    def test_recovery_runner_only_recovers_and_does_not_replace_itself(self):
        self.run_ps("""
$script:ScriptFolder=Join-Path $Root '.runtime/update/recovery'
New-Item -ItemType Directory -Path $script:ScriptFolder -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $Root '.runtime/update/transaction.json'),'{}')
function Save-RecoverySnapshot([string]$StateDir){throw 'Recovery tried to replace itself'}
$Mode='Update';Invoke-Update
if($script:Calls.Count -ne 1 -or $script:Calls[0][0] -ne '--recover' -or $script:Requests.Count){throw 'Recovery runner did extra work'}
""")

    def test_invalid_commit_and_core_failure_leave_no_download(self):
        self.run_ps("""
function Invoke-GitHubRequest([string]$Uri,[string]$Destination=''){return @{sha='not-a-commit'}}
$Mode='Check';$failed=$false
try{Invoke-Update}catch{$failed=$true}
if(-not $failed -or $script:Calls.Count){throw 'Invalid SHA accepted'}
function Invoke-GitHubRequest([string]$Uri,[string]$Destination=''){
 if($Destination){[IO.File]::WriteAllText($Destination,'archive stub');return};return @{sha=('b'*40)}
}
function Invoke-Core([string[]]$Arguments){throw 'simulated failed verification'}
$failed=$false;try{Invoke-Update}catch{$failed=$true}
if(-not $failed){throw 'Core failure ignored'}
if(@(Get-ChildItem (Join-Path $Root '.runtime/update/downloads') -Force).Count){throw 'Failed staging not cleaned'}
""")

    def test_bounded_retry_redaction_and_partial_download_cleanup(self):
        self.run_ps("""
# Restore only the real retry function, leaving network I/O mocked.
$retry=$ast.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Invoke-GitHubRequest'},$false)[0]
Invoke-Expression $retry.Extent.Text
$script:Attempts=0
function Start-Sleep([int]$Seconds){}
function Invoke-UpdateHttp([hashtable]$Parameters,[string]$Destination=''){
 $script:Attempts++
 if($Parameters.TimeoutSec -ne 60){throw 'Missing request timeout'}
 if($Destination){[IO.File]::WriteAllText($Destination,'partial')}
 throw 'sensitive-user:sensitive-password@proxy'
}
$archive=Join-Path $Root 'partial.zip';$failed=$false
try{Invoke-GitHubRequest 'https://api.github.com/repos/test/test/zipball/ref' $archive}catch{
 $failed=$true;if($_.Exception.Message -match 'sensitive'){throw 'Credentials exposed'}
}
if(-not $failed -or $script:Attempts -ne 3 -or (Test-Path $archive)){throw 'Retry or cleanup failed'}
if(($script:Messages -join '') -match 'sensitive'){throw 'Log leaked credentials'}
""")

    def test_real_script_refuses_git_checkout_and_missing_python(self):
        for git in (True, False):
            with self.subTest(git=git):
                marker = self.root / '.git'
                if git:
                    marker.mkdir()
                else:
                    marker.rmdir()
                result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                    '-File', str(ROOT/'tools/update.ps1'), '-Root', str(self.root), '-Mode', 'Check'], capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 1)
                expected = b'Git working tree' if git else b'Python environment is required'
                self.assertIn(expected, result.stdout + result.stderr)
                self.assertFalse((self.root/'.runtime').exists())

    def test_batch_uses_parsed_handoff_after_self_replacement(self):
        batch = self.root / '3- update.bat'
        batch.write_bytes((ROOT/'3- update.bat').read_bytes())
        tools = self.root / 'tools'
        tools.mkdir()
        (tools/'update.ps1').write_text("""param([string]$Root,[string]$Mode,[switch]$PauseWhenDone)
[IO.File]::WriteAllText((Join-Path $Root '3- update.bat'), '@echo off' + [Environment]::NewLine + 'exit /b 37')
Write-Output ('MODE=' + $Mode)
exit 0
""", encoding='utf-8-sig')
        command = '"' + os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe') + '" /d /s /c ""' + str(batch) + '" --check"'
        result = subprocess.run(command, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, (result.stdout+result.stderr).decode('utf-8','replace'))
        self.assertIn(b'MODE=Check', result.stdout)
        self.assertIn('exit /b 37', batch.read_text())

    def test_batch_pending_transaction_uses_known_good_recovery_when_tools_missing(self):
        batch = self.root / '3- update.bat'
        batch.write_bytes((ROOT/'3- update.bat').read_bytes())
        recovery = self.root / '.runtime/update/recovery'
        recovery.mkdir(parents=True)
        (recovery.parent/'transaction.json').write_text('{}')
        (recovery/'update.ps1').write_text("""param([string]$Root,[string]$Mode,[switch]$Runner,[switch]$PauseWhenDone)
if(-not $Runner){exit 42}
Write-Output 'KNOWN_GOOD_RECOVERY'
exit 0
""", encoding='utf-8-sig')
        command = '"' + os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe') + '" /d /s /c ""' + str(batch) + '""'
        result = subprocess.run(command, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, (result.stdout+result.stderr).decode('utf-8','replace'))
        self.assertIn(b'KNOWN_GOOD_RECOVERY', result.stdout)
        self.assertFalse((self.root/'tools').exists())


if __name__ == '__main__':
    unittest.main()
