@echo off
title Machine Translator - Local Stack Manager
echo ====================================================================
echo   🌐 English-to-Arabic Machine Translator - Local Development
echo ====================================================================
echo.

:: Ensure working directory is project root
cd /d "%~dp0"

:: Set local environment defaults
set USE_SQLITE=True
set MONGO_FALLBACK=file

:: 1. Verify Node.js and Python
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH. Please install Node.js 18+.
    pause
    exit /b 1
)

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)

:: 2. Check/Activate Virtual Environment
if not exist "venv" (
    echo [INFO] Virtual environment 'venv' not found. Creating it...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

:: Ensure Python dependencies are installed if missing
if not exist "%~dp0venv\Scripts\django-admin.exe" (
    echo [INFO] Installing Python dependencies in virtual environment...
    pip install -r "%~dp0backend\requirements.txt"
    pip install -r "%~dp0ml_worker\requirements.txt"
    pip install sacremoses
) else (
    echo [INFO] Python dependencies already installed. Skipping pip install.
)

:: Ensure Frontend dependencies are installed
if not exist "%~dp0frontend\node_modules" (
    echo [INFO] Installing Frontend node_modules...
    cd /d "%~dp0frontend"
    call npm install
    cd /d "%~dp0"
)

:: 3. Check for GPU support
echo [INFO] Checking for GPU acceleration...
nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] NVIDIA GPU detected. GPU acceleration will be used.
) else (
    echo [INFO] No NVIDIA GPU detected. Running in CPU mode.
)

:: 4. Check if translation model is downloaded
if not exist "%~dp0models\opus-mt-en-ar\model.safetensors" (
    if not exist "%~dp0models\opus-mt-en-ar\pytorch_model.bin" (
        echo [INFO] Pre-trained translation model not found in models\opus-mt-en-ar.
        echo [INFO] Downloading model now...
        python "%~dp0download_model.py"
    )
)

:: 5. Run Django migrations
echo [INFO] Running Django database migrations (SQLite)...
cd /d "%~dp0backend"
python manage.py migrate
cd /d "%~dp0"

:: 6. Launch services in separate windows
echo.
echo [INFO] Starting all services in separate windows...
echo.
echo [1/3] Launching ML Worker (Flask) on port 8001...
start "ML Worker (Flask)" /D "%~dp0ml_worker" cmd /k "call "%~dp0venv\Scripts\activate.bat" && python worker.py"

echo [2/3] Launching Backend API (Django) on port 8000...
start "Backend API (Django)" /D "%~dp0backend" cmd /k "call "%~dp0venv\Scripts\activate.bat" && python manage.py runserver 0.0.0.0:8000"

echo [3/3] Launching Frontend (React) on port 3000...
start "Frontend (React)" /D "%~dp0frontend" cmd /k "npm start"

echo.
echo ====================================================================
echo   🚀 All services are starting up!
echo   - Frontend:  http://localhost:3000
echo   - Backend:   http://localhost:8000/api/
echo   - ML Worker: http://localhost:8001
echo ====================================================================
echo.
echo Keep this window open if you want to see overall status, or press any key to exit this wrapper.
pause
