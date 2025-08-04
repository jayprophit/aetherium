@echo off
echo.
echo ========================================
echo   🚀 AETHERIUM AI PRODUCTIVITY SUITE
echo      AUTOMATED DEPLOYMENT SYSTEM
echo ========================================
echo.

cd /d "%~dp0"

echo [INFO] 🤖 Starting fully automated deployment...
echo [INFO] 📦 This will install dependencies and start servers
echo [INFO] 🌐 Browser will open automatically when ready
echo [INFO] 🛑 Press Ctrl+C to stop servers when finished
echo.

python auto_deploy_complete.py

echo.
echo [INFO] ✅ Deployment process completed
pause