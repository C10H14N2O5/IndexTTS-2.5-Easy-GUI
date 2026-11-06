@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist "%~dp0local-settings.bat" call "%~dp0local-settings.bat"
rem Parse the complete handoff block before the child runs.
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install.ps1" -Root "%CD%"
  exit /b
)
