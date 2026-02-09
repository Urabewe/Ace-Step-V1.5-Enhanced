@echo off
setlocal enabledelayedexpansion

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE="
set "TORCH_VERSION=2.9.1"
set "TORCHVISION_VERSION=0.24.1"
set "TORCHAUDIO_VERSION=2.9.1"
set "FLASH_ATTN_VERSION=2.8.3"
set "FLASH_ATTN_RELEASE=v0.7.11"

echo.
echo ========================================
echo ACE-Step Venv Setup
echo ========================================
echo.

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [Setup] .venv already exists.
    set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
    goto :DetectCuda
)

echo [Setup] Creating .venv...
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3 -m venv "%VENV_DIR%"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL%==0 (
        python -m venv "%VENV_DIR%"
    ) else (
        echo [Error] Python not found. Please install Python 3.10+ and retry.
        exit /b 1
    )
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [Error] Failed to create .venv.
    exit /b 1
)

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

:DetectCuda
echo.
echo [Setup] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

set "CUDA_VERSION="
for /f "tokens=2 delims=:" %%A in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set "CUDA_VERSION=%%A"
for /f "tokens=1" %%B in ("%CUDA_VERSION%") do set "CUDA_VERSION=%%B"

set "TORCH_CUDA=cpu"

if not "%CUDA_VERSION%"=="" (
    for /f "tokens=1,2 delims=." %%A in ("%CUDA_VERSION%") do (
        set "CUDA_MAJOR=%%A"
        set "CUDA_MINOR=%%B"
    )
    if "!CUDA_MAJOR!"=="11" (
        set "TORCH_CUDA=cu118"
    ) else if "!CUDA_MAJOR!"=="12" (
        if "!CUDA_MINOR!"=="" (
            set "TORCH_CUDA=cu121"
        ) else if !CUDA_MINOR! LSS 4 (
            set "TORCH_CUDA=cu121"
        ) else if !CUDA_MINOR! LSS 6 (
            set "TORCH_CUDA=cu124"
        ) else if !CUDA_MINOR! LSS 8 (
            set "TORCH_CUDA=cu126"
        ) else (
            set "TORCH_CUDA=cu128"
        )
    ) else if "!CUDA_MAJOR!" GEQ "13" (
        echo [Warning] CUDA !CUDA_VERSION! detected; using cu128 wheels for torch 2.9.1.
        set "TORCH_CUDA=cu128"
    ) else (
        echo [Warning] CUDA !CUDA_VERSION! not supported; using CPU wheels.
        set "TORCH_CUDA=cpu"
    )
) else (
    echo [Info] nvidia-smi not found or no CUDA detected; using CPU wheels.
)

if "%TORCH_CUDA%"=="cpu" (
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
    set "TORCH_TAG=+cpu"
) else (
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/%TORCH_CUDA%"
    set "TORCH_TAG=+%TORCH_CUDA%"
)

echo.
echo [Setup] Installing torch %TORCH_VERSION%%TORCH_TAG% (%TORCH_CUDA%)...
"%PYTHON_EXE%" -m pip install ^
    torch==%TORCH_VERSION%%TORCH_TAG% ^
    torchvision==%TORCHVISION_VERSION%%TORCH_TAG% ^
    torchaudio==%TORCHAUDIO_VERSION%%TORCH_TAG% ^
    --index-url %TORCH_INDEX_URL%

if %ERRORLEVEL% NEQ 0 (
    echo [Error] Torch install failed.
    pause
    exit /b 1
)
echo [Setup] Torch install finished.

echo.
echo [Setup] Installing flash-attn (optional)...
if exist "%~dp0install_flashattn.py" (
    "%PYTHON_EXE%" "%~dp0install_flashattn.py"
    if %ERRORLEVEL% NEQ 0 (
        echo [Warning] flash-attn install failed; continuing without it.
    )
) else (
    echo [Warning] install_flashattn.py not found; skipping.
)

echo.
echo [Setup] Installing remaining dependencies...
echo [Setup] Requirements file: %~dp0requirements-base.txt
if not exist "%~dp0requirements-base.txt" (
    echo [Error] requirements-base.txt not found in %~dp0
    pause
    exit /b 1
)
echo [Setup] Requirements content:
type "%~dp0requirements-base.txt"
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements-base.txt"

if %ERRORLEVEL% NEQ 0 (
    echo [Error] Dependency install failed.
    pause
    exit /b 1
)

echo.
echo [Setup] Checking venv, please wait...
"%PYTHON_EXE%" -m pip check
if %ERRORLEVEL% NEQ 0 (
    echo [Setup] Missing dependencies found, installing now...
    for /f "usebackq delims=" %%L in ("%~dp0requirements-base.txt") do (
        set "REQ_LINE=%%L"
        if not "!REQ_LINE!"=="" if not "!REQ_LINE:~0,1!"=="#" (
            "%PYTHON_EXE%" -m pip install "!REQ_LINE!"
            if !ERRORLEVEL! NEQ 0 (
                echo [Error] Failed to install: !REQ_LINE!
                pause
                exit /b 1
            )
        )
    )
)

echo.
echo ========================================
echo Setup complete! You can now run:
echo   start_gradio_ui.bat
echo ========================================
echo.
pause
exit /b 0
