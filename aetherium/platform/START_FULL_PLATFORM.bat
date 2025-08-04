@echo off
title Aetherium AI Platform - Full Interactive Version
color 0B

echo.
echo =====================================================
echo   🚀 AETHERIUM AI PLATFORM - FULL INTERACTIVE
echo      Complete with Chat Interface & AI Conversations
echo =====================================================
echo.
echo ✨ Features Include:
echo   💬 Full chat input boxes for your prompts
echo   🤖 Real-time AI thought processes (like ChatGPT/Claude)
echo   🔄 Interactive conversations with all AI tools
echo   📝 Message history and conversation tracking
echo.
echo ⏳ Starting full interactive platform...
echo    (Browser will open automatically)
echo.

cd /d "%~dp0"
python launch_full_interactive.py

echo.
echo 🎉 Platform stopped. Press any key to exit.
pause > nul