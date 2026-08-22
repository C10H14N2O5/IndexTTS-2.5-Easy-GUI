@echo off
setlocal EnableExtensions
cd /d "%~dp0"
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

"%PY_EXE%" "%CD%\tools\verify_cuda.py"
if errorlevel 1 (
  echo [ERROR] The fixed CUDA runtime is unavailable or was modified. Rerun install.bat.
  pause
  exit /b 1
)
"%PY_EXE%" "%CD%\tools\verify_numpy.py"
if errorlevel 1 (
  echo [ERROR] The project NumPy installation is incomplete or shadowed.
  echo Run 1- install.bat with this project folder as the working directory.
  pause
  exit /b 1
)

"%PY_EXE%" "%CD%\easy_gui.py"
if errorlevel 1 (
  echo.
  echo [ERROR] IndexTTS 2.5 Easy GUI exited with an error.
  pause
)
