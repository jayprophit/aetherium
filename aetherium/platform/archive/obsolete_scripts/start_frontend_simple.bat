@echo off
echo.
echo 🎨 Starting Aetherium Frontend Server...
echo ========================================

cd /d "%~dp0\frontend"
echo 📁 Changed to frontend directory: %cd%

echo.
echo 📦 Checking if node_modules exists...
if not exist "node_modules" (
    echo ⚠️  node_modules not found. Installing dependencies...
    echo 🔄 Running npm install...
    npm install
    if errorlevel 1 (
        echo ❌ npm install failed. Please check your Node.js installation.
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully!
) else (
    echo ✅ Dependencies already installed
)

echo.
echo 🚀 Starting development server...
echo 🌐 Frontend will be available at: http://localhost:3000
echo 🛑 Press Ctrl+C to stop
echo.

npm start

pause