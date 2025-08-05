@echo off
echo === PDF Image Analyzer - Docker Deployment Script ===
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check for NVIDIA GPU support
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  No NVIDIA GPU detected, using CPU only
    set GPU_SUPPORT=
) else (
    echo ✅ NVIDIA GPU detected
    set GPU_SUPPORT=--gpus all
)

REM Create input/output directories
echo 📁 Creating input/output directories...
if not exist input mkdir input
if not exist output mkdir output

REM Clean up any previous failed builds
echo 🧹 Cleaning up previous builds...
docker system prune -f

REM Build the Docker image
echo 🔨 Building Docker image (this may take 10-15 minutes)...
echo ⏰ Please be patient, downloading and installing dependencies...
docker build --no-cache -t pdf-analyzer .

if errorlevel 1 (
    echo.
    echo ❌ Docker build failed
    echo 🔧 Troubleshooting suggestions:
    echo 1. Check your internet connection
    echo 2. Ensure you have at least 10GB free disk space
    echo 3. Try running: docker system prune -f
    echo 4. Restart Docker Desktop and try again
    echo.
    pause
    exit /b 1
)

echo ✅ Docker image built successfully
echo.

REM Check for PDF files in input directory
dir input\*.pdf >nul 2>&1
if errorlevel 1 (
    echo 📄 No PDF files found in input directory
    echo.
    echo To use this system:
    echo 1. Copy your PDF files to: %cd%\input\
    echo 2. Run this script again: deploy.bat
    echo.
    echo Or run manually:
    echo docker run --rm %GPU_SUPPORT% -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" pdf-analyzer
) else (
    echo 📄 Found PDF files in input directory:
    dir input\*.pdf
    echo.
    
    echo 🚀 Starting processing...
    docker run --rm %GPU_SUPPORT% -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" pdf-analyzer
    
    echo.
    echo ✅ Processing complete! Check the output directory:
    dir output\
)

echo.
echo 🎉 Deployment complete!
pause