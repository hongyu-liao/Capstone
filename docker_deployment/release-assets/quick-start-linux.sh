#!/bin/bash

echo "================================================================"
echo "           PDF Image Analyzer - One-Click Setup"
echo "================================================================"
echo ""
echo "This script will automatically download and set up the"
echo "PDF Image Analyzer Docker environment."
echo ""
echo "Requirements:"
echo "- Docker with GPU support (recommended)"
echo "- 8GB+ RAM available"
echo "- 15GB+ free disk space"
echo "- Stable internet connection"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo ""
    echo "Please start Docker and try again."
    echo "You can install Docker from:"
    echo "https://docs.docker.com/get-docker/"
    echo ""
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Create working directory
WORK_DIR="pdf-image-analyzer"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Download deployment package if not exists
if [ ! -f "docker-deployment.zip" ]; then
    echo "📥 Downloading deployment package..."
    echo "This may take a moment..."
    
    if command -v curl >/dev/null 2>&1; then
        curl -L -o docker-deployment.zip "https://github.com/your-username/pdf-image-analyzer/releases/latest/download/docker-deployment.zip"
    elif command -v wget >/dev/null 2>&1; then
        wget -O docker-deployment.zip "https://github.com/your-username/pdf-image-analyzer/releases/latest/download/docker-deployment.zip"
    else
        echo "❌ Neither curl nor wget found!"
        echo "Please install curl or wget and try again."
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Download failed!"
        echo "Please check your internet connection and try again."
        exit 1
    fi
    echo "✅ Download complete"
fi

# Extract deployment files
echo "📂 Extracting deployment files..."
if command -v unzip >/dev/null 2>&1; then
    unzip -o docker-deployment.zip
else
    echo "❌ unzip not found!"
    echo "Please install unzip and try again."
    echo "Ubuntu/Debian: sudo apt-get install unzip"
    echo "CentOS/RHEL: sudo yum install unzip"
    echo "macOS: unzip should be pre-installed"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ Extraction failed!"
    exit 1
fi

echo "✅ Files extracted"
echo ""

# Enter deployment directory
cd docker_deployment

# Make scripts executable
chmod +x *.sh

# Create input/output directories
echo "📁 Setting up directories..."
mkdir -p input output

echo ""
echo "🚀 Starting Docker deployment..."
echo "This will build the Docker image (may take 10-15 minutes on first run)"
echo ""

# Run deployment
./deploy.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Deployment failed!"
    echo "Check the error messages above for troubleshooting."
    echo ""
    echo "Common solutions:"
    echo "1. Ensure you have enough disk space (15GB+)"
    echo "2. Check your internet connection"
    echo "3. Try running: ./fix_and_retry.sh"
    echo ""
    exit 1
fi

echo ""
echo "================================================================"
echo "                     🎉 Setup Complete!"
echo "================================================================"
echo ""
echo "Your PDF Image Analyzer is now ready to use!"
echo ""
echo "Next steps:"
echo "1. Copy your PDF files to: $(pwd)/input/"
echo "2. Run: ./deploy.sh (to process the PDFs)"
echo "3. Check results in: $(pwd)/output/"
echo ""
echo "For detailed instructions, see: README.md"
echo "For troubleshooting, see: TROUBLESHOOTING.md"
echo ""
echo "Example usage:"
echo "  cp /path/to/your/document.pdf input/"
echo "  ./deploy.sh"
echo ""