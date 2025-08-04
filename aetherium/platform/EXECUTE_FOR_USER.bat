@echo off
echo ========================================
echo EXECUTING AETHERIUM DEPLOYMENT FOR USER
echo ========================================
cd /d "C:\Users\jpowe\CascadeProjects\github\aetherium\aetherium\platform"
echo Current directory: %cd%
echo.
echo Starting deployment...
python SIMPLE_WORKING.py
echo.
echo Deployment completed!
pause