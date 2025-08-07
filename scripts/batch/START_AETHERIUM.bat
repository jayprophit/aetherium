@echo off
title Aetherium AI Platform - Automated Startup

echo ========================================
echo    AETHERIUM AI PLATFORM STARTUP
echo ========================================
echo.
echo Starting comprehensive AI productivity platform...
echo.

cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.

:: Run the production deployment script
echo 🚀 Launching Aetherium Platform...
echo.

python scripts\production-deploy.py

if errorlevel 1 (
    echo.
    echo ❌ Deployment failed. Please check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Aetherium Platform is now running!
echo.
echo 🌐 Open your browser to: http://localhost:5173
echo 📊 API Documentation: http://localhost:8000/docs
echo.
echo Press any key to keep the platform running...
pause >nul

echo.
echo Platform will continue running in the background.
echo Close this window to stop the platform.
echo.