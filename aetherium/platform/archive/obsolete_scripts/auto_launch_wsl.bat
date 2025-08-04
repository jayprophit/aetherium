@echo off
title Aetherium AI Platform - Automated Launch
color 0A

echo.
echo ==========================================
echo   🚀 AETHERIUM AI PRODUCTIVITY SUITE
echo        FULLY AUTOMATED LAUNCH
echo ==========================================
echo.
echo ✨ This will automatically:
echo   📦 Install all requirements
echo   🔧 Set up the platform
echo   🚀 Start all servers
echo   🌐 Open your browser
echo.
echo ⏳ Please wait while we prepare your platform...
echo    (This may take 2-3 minutes on first run)
echo.

REM Change to the correct directory
cd /d "%~dp0"

REM Run the automated WSL deployment
wsl bash -c "cd /mnt/c/Users/jpowe/CascadeProjects/github/aetherium/aetherium/platform && chmod +x auto_launch_aetherium.sh && ./auto_launch_aetherium.sh"

echo.
echo 🎉 Aetherium platform should now be opening in your browser!
echo 🌐 If browser doesn't open automatically, visit: http://localhost:8000
echo.
echo 🛑 To stop the platform: Close this window or press Ctrl+C
echo.
pause