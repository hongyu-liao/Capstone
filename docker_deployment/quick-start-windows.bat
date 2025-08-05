@echo off
echo ================================================================
echo           PDF Image Analyzer - One-Click Setup
echo ================================================================
echo.
echo This script will automatically download and set up the 
echo PDF Image Analyzer Docker environment.
echo.
echo Requirements:
echo - Docker Desktop (with GPU support recommended)
echo - 8GB+ RAM available
echo - 15GB+ free disk space
echo - Stable internet connection
echo.
pause

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running!
    echo.
    echo Please start Docker Desktop and try again.
    echo You can download Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

REM Create working directory
set WORK_DIR=pdf-image-analyzer
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"
cd "%WORK_DIR%"

REM Download deployment package if not exists
if not exist "docker-deployment.zip" (
    echo 📥 Downloading deployment package...
    echo This may take a moment...
    curl -L -o docker-deployment.zip "https://github.com/your-username/pdf-image-analyzer/releases/latest/download/docker-deployment.zip"
    
    if errorlevel 1 (
        echo ❌ Download failed!
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
    echo ✅ Download complete
)

REM Extract deployment files
echo 📂 Extracting deployment files...
powershell -command "try { Expand-Archive -Path docker-deployment.zip -DestinationPath . -Force } catch { exit 1 }"

if errorlevel 1 (
    echo ❌ Extraction failed!
    echo Please ensure you have PowerShell and try again.
    pause
    exit /b 1
)

echo ✅ Files extracted
echo.

REM Enter deployment directory
cd docker_deployment

REM Create input/output directories
echo 📁 Setting up directories...
if not exist input mkdir input
if not exist output mkdir output

echo.
echo 🚀 Starting Docker deployment...
echo This will build the Docker image (may take 10-15 minutes on first run)
echo.

REM Run deployment
call deploy.bat

if errorlevel 1 (
    echo.
    echo ❌ Deployment failed!
    echo Check the error messages above for troubleshooting.
    echo.
    echo Common solutions:
    echo 1. Ensure you have enough disk space (15GB+)
    echo 2. Check your internet connection
    echo 3. Try running fix_and_retry.bat
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo                     🎉 Setup Complete!
echo ================================================================
echo.
echo Your PDF Image Analyzer is now ready to use!
echo.
echo Next steps:
echo 1. Copy your PDF files to: %cd%\input\
echo 2. Run: deploy.bat (to process the PDFs)
echo 3. Check results in: %cd%\output\
echo.
echo For detailed instructions, see: README.md
echo For troubleshooting, see: TROUBLESHOOTING.md
echo.
echo Example usage:
echo   copy "C:\path\to\your\document.pdf" input\
echo   deploy.bat
echo.
pause