@echo off
title PDF Image Analyzer Launcher

echo ===============================================
echo   PDF Image Analyzer - Windows Launcher
echo ===============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo 💡 Please install Python 3.8+ and add it to PATH
    echo 🔗 Download from: https://python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python is available
echo.

:: Check if we're in the right directory
if not exist "app.py" (
    echo ❌ Error: app.py not found
    echo 💡 Please ensure you're running this from the PDF_Analyzer_App directory
    pause
    exit /b 1
)

echo ✅ Application files found
echo.

:: Try to install/update dependencies
echo 📦 Installing/updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️  Warning: Some dependencies may not have installed correctly
    echo 💡 You can continue, but the app might not work properly
    echo.
)

echo.
echo 🚀 Starting PDF Image Analyzer...
echo 📱 The web interface will open in your browser
echo 🔗 URL: http://localhost:8501
echo.
echo ⏹️  To stop the application, close this window or press Ctrl+C
echo.

:: Start the application
python run_app.py

echo.
echo 👋 Application closed
pause