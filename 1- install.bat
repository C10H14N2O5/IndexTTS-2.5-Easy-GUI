@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title IndexTTS 2.5 Easy GUI - Fixed CUDA Installer

set "UV_VERSION=0.10.0"
set "PYTHON_VERSION=3.11.15"
set "UV_DIR=%CD%\.runtime\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "UV_PYTHON_INSTALL_DIR=%CD%\.runtime\python"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"
set "XDG_CACHE_HOME=%CD%\.runtime\cache"
set "HF_HOME=%CD%\.runtime\hf-cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "TMP=%CD%\.runtime\temp"
set "TEMP=%CD%\.runtime\temp"
set "UV_NO_CACHE=1"
set "UV_LINK_MODE=copy"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "TRITON_CACHE_DIR=%CD%\.runtime\triton-cache"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\.runtime\torchinductor-cache"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"

set "TRITON_VERSION=3.1.0.post17"
set "FA_WHEEL_NAME=flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl"
set "FA_WHEEL_URL=https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/%FA_WHEEL_NAME%"

for %%D in (".runtime" ".runtime\temp" "%UV_DIR%" "models" "outputs" "voices" "emotion_descriptions" "accelerator_wheels") do if not exist "%%~D" mkdir "%%~D"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [ERROR] NVIDIA driver / nvidia-smi was not detected.
  echo This release supports Windows x64 + NVIDIA CUDA only.
  goto :fail
)

echo [1/8] NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if not exist "%UV_EXE%" (
  echo [2/8] Downloading project-local uv %UV_VERSION%...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $zip='%TEMP%\uv-indextts.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip; Expand-Archive -Force $zip '%UV_DIR%'; Remove-Item -Force $zip"
  if errorlevel 1 goto :fail
) else (
  echo [2/8] Project-local uv already present.
)

if not exist "%UV_EXE%" (
  echo [ERROR] uv.exe was not found after bootstrap.
  goto :fail
)

echo [3/8] Installing project-local Python %PYTHON_VERSION%...
"%UV_EXE%" python install %PYTHON_VERSION% --no-cache
if errorlevel 1 goto :fail

echo [4/8] Synchronizing the frozen IndexTTS 2.5 CUDA environment...
"%UV_EXE%" sync --frozen --extra webui --no-cache --python %PYTHON_VERSION%
if errorlevel 1 goto :fail

if not exist "%PY_EXE%" (
  echo [ERROR] Project Python was not created at .venv\Scripts\python.exe.
  goto :fail
)

echo [5/8] Verifying the exact supported runtime...
"%PY_EXE%" "%CD%\tools\verify_cuda.py"
if errorlevel 1 goto :fail

echo [6/8] Installing fixed Triton Windows runtime...
"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache "triton-windows==%TRITON_VERSION%"
if errorlevel 1 (
  echo [ERROR] Triton installation failed.
  goto :fail
)

echo [7/8] Installing fixed FlashAttention Windows wheel...
call :install_flash
if errorlevel 1 (
  echo [WARNING] FlashAttention installation failed.
  echo [WARNING] Base CUDA inference and Triton remain installed and usable.
)

echo [8/8] Verifying accelerator status...
"%PY_EXE%" "%CD%\tools\verify_accelerators.py"

if exist "%UV_CACHE_DIR%" rmdir /s /q "%UV_CACHE_DIR%"

echo.
echo Installation complete.
echo Fixed runtime: Python %PYTHON_VERSION% / PyTorch 2.8.0+cu128 / CUDA runtime 12.8.
echo Triton Windows: %TRITON_VERSION%
echo FlashAttention: fixed prebuilt Windows wheel for CPython 3.11 / Torch 2.8 / cu128.
echo External CUDA Toolkit is not required.
echo Model weights are intentionally not downloaded during environment installation.
echo run.bat automatically downloads or repairs IndexTeam/IndexTTS-2.5 and auxiliary models before opening the GUI.
pause
exit /b 0

:install_flash
for %%F in ("accelerator_wheels\flash_attn*.whl") do if exist "%%~fF" (
  echo [FlashAttention] Using local wheel: %%~nxF
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps "%%~fF"
  exit /b %errorlevel%
)

"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache "ninja==1.13.0" "packaging==25.0"
if errorlevel 1 exit /b 1

"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps "%FA_WHEEL_URL%"
exit /b %errorlevel%

:fail
if exist "%UV_CACHE_DIR%" rmdir /s /q "%UV_CACHE_DIR%"
echo.
echo INSTALLATION FAILED. Review the error above.
pause
exit /b 1
