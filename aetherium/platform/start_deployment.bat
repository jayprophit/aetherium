@echo off
echo.
echo ========================================
echo   AETHERIUM AI PRODUCTIVITY SUITE
echo        PRODUCTION DEPLOYMENT
echo ========================================
echo.

cd /d "%~dp0"

echo [INFO] Starting deployment process...
python deploy_platform.py

pause