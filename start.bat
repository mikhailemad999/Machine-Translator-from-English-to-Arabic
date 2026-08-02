@echo off
title Machine Translator - Docker Stack Manager
echo ====================================================================
echo   🌐 English-to-Arabic Machine Translator - Docker Mode
echo ====================================================================
echo.

:: Ensure working directory is project root
cd /d "%~dp0"

:: Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo [TIP] If you want to run locally without Docker, run 'start_local.bat' instead!
    echo.
    pause
    exit /b 1
)

:: Check if Docker daemon is running
docker info >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Desktop is not running. Please start Docker Desktop first.
    echo [TIP] If you want to run locally without Docker, run 'start_local.bat' instead!
    echo.
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
