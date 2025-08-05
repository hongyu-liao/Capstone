@echo off
echo === PDF Image Analyzer - Fix and Retry Script ===
echo.

echo 🧹 Cleaning Docker cache and failed builds...
docker system prune -a -f
docker builder prune -a -f

echo.
echo 📦 Removing any existing pdf-analyzer images...
docker image rm pdf-analyzer 2>nul

echo.
echo 🔧 Checking Docker system status...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

echo 🚀 Starting fresh build...
call deploy.bat

echo.
echo 🎉 Fix and retry complete!
pause