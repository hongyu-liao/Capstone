@echo off
echo ================================================================
echo        PDF Image Analyzer - GitHub Release Package Creator
echo ================================================================
echo.

REM Check if required tools are available
where /q powershell
if errorlevel 1 (
    echo ❌ PowerShell not found!
    pause
    exit /b 1
)

echo ✅ Creating GitHub Release package...
echo.

REM Create release directory
set RELEASE_DIR=release-assets
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%"

echo 📦 Preparing deployment package...

REM Create deployment ZIP (excluding unnecessary files)
powershell -command "Compress-Archive -Path @('main.py', 'Dockerfile', 'requirements.txt', 'process.sh', 'deploy.sh', 'deploy.bat', 'fix_and_retry.sh', 'fix_and_retry.bat', 'docker-compose.yml', 'README.md', 'start.md', 'TROUBLESHOOTING.md', 'example_usage.md', 'test_deployment.py', '.dockerignore') -DestinationPath '%RELEASE_DIR%\pdf-analyzer-docker-deployment-v1.0.0.zip' -Force"

echo ✅ Deployment package created
echo.

echo 📋 Copying quick-start scripts...

REM Copy quick-start scripts
copy "quick-start-windows.bat" "%RELEASE_DIR%\"
copy "quick-start-linux.sh" "%RELEASE_DIR%\"

echo ✅ Quick-start scripts copied
echo.

echo 📚 Copying documentation...

REM Copy important documentation
copy "RELEASE_NOTES.md" "%RELEASE_DIR%\"
copy "DOCKER_RELEASE.md" "%RELEASE_DIR%\"
copy "GITHUB_RELEASE_GUIDE.md" "%RELEASE_DIR%\"

echo ✅ Documentation copied
echo.

echo 🐳 Creating Docker image archive (optional)...
echo This may take several minutes...

docker images | findstr pdf-analyzer >nul
if errorlevel 1 (
    echo ⚠️  Docker image 'pdf-analyzer' not found
    echo Build the image first with: deploy.bat
    echo Skipping Docker image archive creation
) else (
    echo Creating Docker image archive...
    docker save pdf-analyzer:latest | gzip > "%RELEASE_DIR%\pdf-analyzer-docker-image-v1.0.0.tar.gz"
    echo ✅ Docker image archive created
)

echo.
echo ================================================================
echo                    📦 Release Package Ready!
echo ================================================================
echo.
echo Release assets created in: %RELEASE_DIR%\
echo.
dir "%RELEASE_DIR%"
echo.
echo 🚀 Next steps for GitHub Release:
echo.
echo 1. Create a new release on GitHub (tag: v1.0.0)
echo 2. Upload all files from %RELEASE_DIR%\ as release assets
echo 3. Use the content from RELEASE_NOTES.md as release description
echo 4. Follow GITHUB_RELEASE_GUIDE.md for detailed instructions
echo.
echo 📂 Asset descriptions for GitHub:
echo.
echo pdf-analyzer-docker-deployment-v1.0.0.zip:
echo   "Complete Docker deployment package with scripts and documentation"
echo.
echo quick-start-windows.bat:
echo   "One-click installer for Windows users"
echo.
echo quick-start-linux.sh:
echo   "One-click installer for Linux/macOS users"
echo.
echo pdf-analyzer-docker-image-v1.0.0.tar.gz:
echo   "Pre-built Docker image for offline installation (optional)"
echo.
pause