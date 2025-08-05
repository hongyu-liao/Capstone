#!/bin/bash

echo "=== PDF Image Analyzer - Docker Deployment Script ==="
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check for NVIDIA GPU support
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA GPU detected"
    GPU_SUPPORT="--gpus all"
else
    echo "⚠️  No NVIDIA GPU detected, using CPU only"
    GPU_SUPPORT=""
fi

# Create input/output directories
echo "📁 Creating input/output directories..."
mkdir -p input output

# Clean up any previous failed builds
echo "🧹 Cleaning up previous builds..."
docker system prune -f

# Build the Docker image
echo "🔨 Building Docker image (this may take 10-15 minutes)..."
echo "⏰ Please be patient, downloading and installing dependencies..."
docker build --no-cache -t pdf-analyzer .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Docker build failed"
    echo "🔧 Troubleshooting suggestions:"
    echo "1. Check your internet connection"
    echo "2. Ensure you have at least 10GB free disk space"
    echo "3. Try running: docker system prune -f"
    echo "4. Restart Docker Desktop and try again"
    echo ""
    exit 1
fi

echo "✅ Docker image built successfully"
echo ""

# Check for PDF files in input directory
if ls input/*.pdf 1> /dev/null 2>&1; then
    echo "📄 Found PDF files in input directory:"
    ls -la input/*.pdf
    echo ""
    
    echo "🚀 Starting processing..."
    docker run --rm $GPU_SUPPORT \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        pdf-analyzer
    
    echo ""
    echo "✅ Processing complete! Check the output directory:"
    ls -la output/
    
else
    echo "📄 No PDF files found in input directory"
    echo ""
    echo "To use this system:"
    echo "1. Copy your PDF files to: $(pwd)/input/"
    echo "2. Run this script again: ./deploy.sh"
    echo ""
    echo "Or run manually:"
    echo "docker run --rm $GPU_SUPPORT -v \"\$(pwd)/input:/app/input\" -v \"\$(pwd)/output:/app/output\" pdf-analyzer"
fi

echo ""
echo "🎉 Deployment complete!"