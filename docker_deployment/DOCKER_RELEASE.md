# Docker Release Guide

This guide explains how to package and release the PDF Image Analyzer as a Docker image for GitHub Release.

## 📦 Packaging for Release

### 1. Build Production Image

```bash
# Build with specific tag for release
docker build -t pdf-analyzer:v1.0.0 .
docker build -t pdf-analyzer:latest .
```

### 2. Create Multi-Platform Image (Optional)

```bash
# Create builder for multi-platform
docker buildx create --name multiplatform --use

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t pdf-analyzer:v1.0.0 --push .
```

### 3. Save Image as Archive

```bash
# Save image to file for GitHub Release
docker save pdf-analyzer:latest | gzip > pdf-analyzer-docker-v1.0.0.tar.gz
```

## 🚀 GitHub Release Process

### 1. Prepare Release Assets

Create the following files for the release:

```
release-assets/
├── pdf-analyzer-docker-v1.0.0.tar.gz     # Docker image archive
├── docker-deployment.zip                  # Complete deployment package
├── quick-start-windows.bat                # Windows one-click installer
├── quick-start-linux.sh                   # Linux one-click installer
└── README-RELEASE.md                      # Release-specific readme
```

### 2. Create Deployment Package

```bash
# Create deployment package
zip -r docker-deployment.zip . -x "*.git*" "*__pycache__*" "*.pyc" "input/*" "output/*" "*.log"
```

### 3. Create One-Click Installers

**Windows Installer (`quick-start-windows.bat`):**
```batch
@echo off
echo === PDF Image Analyzer - One-Click Setup ===
echo.

REM Download if not exists
if not exist "docker-deployment.zip" (
    echo Downloading deployment package...
    curl -L -o docker-deployment.zip https://github.com/your-username/pdf-image-analyzer/releases/latest/download/docker-deployment.zip
)

REM Extract
echo Extracting files...
powershell -command "Expand-Archive -Path docker-deployment.zip -DestinationPath . -Force"

REM Enter directory and deploy
cd docker_deployment
echo Starting deployment...
call deploy.bat

echo.
echo Setup complete! Check the output directory for results.
pause
```

**Linux Installer (`quick-start-linux.sh`):**
```bash
#!/bin/bash
echo "=== PDF Image Analyzer - One-Click Setup ==="
echo ""

# Download if not exists
if [ ! -f "docker-deployment.zip" ]; then
    echo "Downloading deployment package..."
    curl -L -o docker-deployment.zip https://github.com/your-username/pdf-image-analyzer/releases/latest/download/docker-deployment.zip
fi

# Extract
echo "Extracting files..."
unzip -o docker-deployment.zip

# Enter directory and deploy
cd docker_deployment
echo "Starting deployment..."
chmod +x deploy.sh
./deploy.sh

echo ""
echo "Setup complete! Check the output directory for results."
```

## 📝 Release Notes Template

```markdown
# PDF Image Analyzer v1.0.0 - Docker Edition

## 🚀 Features

- **One-Click Deployment**: Automated setup for Windows and Linux
- **Smart PDF Processing**: Using Docling SmolDocling VLM Pipeline
- **AI Image Analysis**: Intelligent image classification and description
- **Web-Enhanced Context**: Automatic contextual search for images
- **GPU Acceleration**: CUDA support for faster processing
- **Multiple Output Formats**: Original, enhanced, and NLP-ready versions

## 📦 Download Options

### Quick Start (Recommended)
- **Windows**: Download `quick-start-windows.bat` and run
- **Linux/macOS**: Download `quick-start-linux.sh` and run

### Manual Installation
- **Complete Package**: `docker-deployment.zip` contains all files
- **Docker Image**: `pdf-analyzer-docker-v1.0.0.tar.gz` for offline installation

## 🎯 Usage

1. Download the appropriate quick-start script for your platform
2. Place your PDF files in the `input/` directory
3. Run the script and wait for processing
4. Check results in the `output/` directory

## 📋 System Requirements

- Docker Desktop 4.0+
- 8GB+ RAM (16GB+ recommended)
- 15GB+ free disk space
- Internet connection (for first run)

## 🔧 GPU Support

For GPU acceleration:
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Docker runtime installed
- CUDA-compatible drivers

## 🆕 What's New

- Initial Docker release
- Automated deployment scripts
- Comprehensive documentation
- Multi-platform support
- Performance optimizations

## 🐛 Known Issues

- First run requires downloading ~10GB of AI models
- Large PDFs may require additional memory allocation
- GPU memory limitations with very large documents

## 📚 Documentation

- [README.md](README.md) - Complete documentation
- [start.md](start.md) - Quick start guide  
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

## 🙏 Acknowledgments

Built with Docling, Hugging Face Transformers, and PyTorch.

---

**Full Changelog**: https://github.com/your-username/pdf-image-analyzer/compare/...v1.0.0
```

## 🎯 Release Checklist

- [ ] Build and test Docker image locally
- [ ] Create deployment package
- [ ] Test one-click installers on clean systems
- [ ] Prepare release notes
- [ ] Tag repository with version
- [ ] Create GitHub Release
- [ ] Upload all assets
- [ ] Test download and installation
- [ ] Update documentation

## 📊 Asset Sizes (Approximate)

- `pdf-analyzer-docker-v1.0.0.tar.gz`: ~4-6GB (compressed Docker image)
- `docker-deployment.zip`: ~50-100KB (deployment scripts and configs)
- `quick-start-*.sh/.bat`: ~1-2KB each (installer scripts)

## 🔄 Auto-Update Strategy

Consider implementing:
- Version checking in deployment scripts
- Automatic image updates via Docker Hub
- Notification system for new releases

## 📈 Release Metrics

Track:
- Download counts per asset
- Success/failure rates
- User feedback and issues
- Performance benchmarks