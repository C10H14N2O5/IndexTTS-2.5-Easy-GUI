@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist "%~dp0.runtime\update\transaction.json" (
  echo [ERROR] An interrupted update needs recovery. Run 3- update.bat first.
  pause
  exit /b 1
)

rem Optional machine-local settings; this file is excluded from Git.
if exist "%~dp0local-settings.bat" call "%~dp0local-settings.bat"
if not defined INDEXTTS_HOST set "INDEXTTS_HOST=127.0.0.1"
if not defined INDEXTTS_PORT set "INDEXTTS_PORT=7860"
title IndexTTS 2.5 Easy GUI

set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"
set "XDG_CACHE_HOME=%CD%\.runtime\cache"
set "HF_HOME=%CD%\.runtime\hf-cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "TMP=%CD%\.runtime\temp"
set "TEMP=%CD%\.runtime\temp"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "NO_PROXY=localhost,127.0.0.1,::1,%INDEXTTS_HOST%,%NO_PROXY%"
set "USE_MODELSCOPE=false"
set "NLTK_DATA=%CD%\.runtime\nltk_data"
set "PATH=%CD%\.runtime\bin;%PATH%"
set "TRITON_CACHE_DIR=%CD%\.runtime\triton-cache"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\.runtime\torchinductor-cache"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"

if not exist "%PY_EXE%" (
  echo [ERROR] Project environment not installed. Run install.bat first.
  pause
  exit /b 1
)
if not exist ".runtime\temp" mkdir ".runtime\temp"
if not exist "outputs" mkdir "outputs"
if not exist "voices" mkdir "voices"
if not exist "emotion_descriptions" mkdir "emotion_descriptions"

"%PY_EXE%" -B "%CD%\tools\run_gui.py"
if errorlevel 1 (
  echo.
  echo [ERROR] IndexTTS 2.5 Easy GUI exited with an error.
  pause
)
