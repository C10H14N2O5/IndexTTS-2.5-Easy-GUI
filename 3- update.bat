@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if exist "%~dp0local-settings.bat" call "%~dp0local-settings.bat"
set "INDEXTTS_UPDATE_MODE=Update"
if /i "%~1"=="--check" set "INDEXTTS_UPDATE_MODE=Check"
if /i "%~1"=="--rollback" set "INDEXTTS_UPDATE_MODE=Rollback"
if not "%~1"=="" if /i not "%~1"=="--check" if /i not "%~1"=="--rollback" (
  echo Usage: "3- update.bat" [--check ^| --rollback]
  exit /b 2
)
if not "%~2"=="" (
  echo Only one update option is supported.
  exit /b 2
)
set "INDEXTTS_UPDATE_SCRIPT=%~dp0tools\update.ps1"
set "INDEXTTS_UPDATE_RUNNER="
if exist "%~dp0.runtime\update\transaction.json" (
  set "INDEXTTS_UPDATE_SCRIPT=%~dp0.runtime\update\recovery\update.ps1"
  set "INDEXTTS_UPDATE_RUNNER=-Runner"
  if not exist "%~dp0.runtime\update\recovery\update.ps1" (
    echo ERROR: Update recovery files are missing. Keep .runtime\update intact and restore the updater from your backup.
    exit /b 1
  )
)
rem CMD parses this entire final block before PowerShell starts. The exit does
rem not read batch bytes after the updater may have replaced this file.
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%INDEXTTS_UPDATE_SCRIPT%" -Root "%~dp0." -Mode "%INDEXTTS_UPDATE_MODE%" %INDEXTTS_UPDATE_RUNNER% -PauseWhenDone
  exit /b
)
