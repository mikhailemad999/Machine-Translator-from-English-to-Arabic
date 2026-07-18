@echo off
title Machine Translator - Docker Stack Manager
echo ====================================================================
echo   🌐 English-to-Arabic Machine Translator - Docker Mode
echo ====================================================================
echo.

:: Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH. Please install Docker Desktop.
    pause
    exit /b 1
)

:: Check if Docker daemon is running
docker info >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Desktop is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [INFO] Starting all services via Docker Compose...
docker-compose up --build

echo.
echo ====================================================================
echo   🚀 Services have stopped.
echo ====================================================================
pause
