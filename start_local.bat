@echo off
title Machine Translator - Local Stack Manager
echo ====================================================================
echo   🌐 English-to-Arabic Machine Translator - Local Development
echo ====================================================================
echo.

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
if not exist venv (
    echo [INFO] Virtual environment 'venv' not found. Creating it...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

:: Ensure Python dependencies are up to date
echo [INFO] Ensuring Python dependencies are up to date...
pip install -r backend\requirements.txt
pip install -r ml_worker\requirements.txt

:: 3. Check for GPU support
echo [INFO] Checking for GPU acceleration...
nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] NVIDIA GPU detected. GPU acceleration will be used.
) else (
    echo [INFO] No NVIDIA GPU detected. Running in CPU mode.
)

:: 4. Check if translation model is downloaded
if not exist "models\opus-mt-en-ar\model.safetensors" (
    if not exist "models\opus-mt-en-ar\pytorch_model.bin" (
        echo [INFO] Pre-trained translation model not found in models\opus-mt-en-ar.
        echo [INFO] Downloading model now...
        python download_model.py
    )
)

:: 5. Run Django migrations
echo [INFO] Running Django database migrations (SQLite)...
cd backend
python manage.py migrate
cd ..

:: 6. Launch services in separate windows
echo [INFO] Starting all services in separate windows...
echo.
echo [1/3] Launching ML Worker (Flask) on port 8001...
start "ML Worker (Flask)" cmd /k "call venv\Scripts\activate && cd ml_worker && python worker.py"

echo [2/3] Launching Backend API (Django) on port 8000...
start "Backend API (Django)" cmd /k "call venv\Scripts\activate && cd backend && python manage.py runserver 0.0.0.0:8000"

echo [3/3] Launching Frontend (React) on port 3000...
start "Frontend (React)" cmd /k "cd frontend && npm start"

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
