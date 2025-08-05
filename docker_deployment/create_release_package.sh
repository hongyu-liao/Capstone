#!/bin/bash

echo "================================================================"
echo "        PDF Image Analyzer - GitHub Release Package Creator"
echo "================================================================"
echo ""

# Check required tools
if ! command -v zip >/dev/null 2>&1; then
    echo "❌ zip command not found!"
    echo "Please install zip and try again."
    exit 1
fi

echo "✅ Creating GitHub Release package..."
echo ""

# Create release directory
RELEASE_DIR="release-assets"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo "📦 Preparing deployment package..."

# Create deployment ZIP (excluding unnecessary files)
zip -r "$RELEASE_DIR/pdf-analyzer-docker-deployment-v1.0.0.zip" \
    main.py Dockerfile requirements.txt process.sh deploy.sh deploy.bat \
    fix_and_retry.sh fix_and_retry.bat docker-compose.yml README.md \
    start.md TROUBLESHOOTING.md example_usage.md test_deployment.py .dockerignore \
    -x "*__pycache__*" "*.pyc" "input/*" "output/*" "*.log" ".git*"

echo "✅ Deployment package created"
echo ""

echo "📋 Copying quick-start scripts..."

# Copy quick-start scripts
cp quick-start-windows.bat "$RELEASE_DIR/"
cp quick-start-linux.sh "$RELEASE_DIR/"

# Make Linux script executable
chmod +x "$RELEASE_DIR/quick-start-linux.sh"

echo "✅ Quick-start scripts copied"
echo ""

echo "📚 Copying documentation..."

# Copy important documentation
cp RELEASE_NOTES.md "$RELEASE_DIR/"
cp DOCKER_RELEASE.md "$RELEASE_DIR/"
cp GITHUB_RELEASE_GUIDE.md "$RELEASE_DIR/"

echo "✅ Documentation copied"
echo ""

echo "🐳 Creating Docker image archive (optional)..."
echo "This may take several minutes..."

if docker images | grep -q pdf-analyzer; then
    echo "Creating Docker image archive..."
    docker save pdf-analyzer:latest | gzip > "$RELEASE_DIR/pdf-analyzer-docker-image-v1.0.0.tar.gz"
    echo "✅ Docker image archive created"
else
    echo "⚠️  Docker image 'pdf-analyzer' not found"
    echo "Build the image first with: ./deploy.sh"
    echo "Skipping Docker image archive creation"
fi

echo ""
echo "================================================================"
echo "                    📦 Release Package Ready!"
echo "================================================================"
echo ""
echo "Release assets created in: $RELEASE_DIR/"
echo ""
ls -la "$RELEASE_DIR/"
echo ""
echo "🚀 Next steps for GitHub Release:"
echo ""
echo "1. Create a new release on GitHub (tag: v1.0.0)"
echo "2. Upload all files from $RELEASE_DIR/ as release assets"
echo "3. Use the content from RELEASE_NOTES.md as release description"
echo "4. Follow GITHUB_RELEASE_GUIDE.md for detailed instructions"
echo ""
echo "📂 Asset descriptions for GitHub:"
echo ""
echo "pdf-analyzer-docker-deployment-v1.0.0.zip:"
echo "  \"Complete Docker deployment package with scripts and documentation\""
echo ""
echo "quick-start-windows.bat:"
echo "  \"One-click installer for Windows users\""
echo ""
echo "quick-start-linux.sh:"
echo "  \"One-click installer for Linux/macOS users\""
echo ""
echo "pdf-analyzer-docker-image-v1.0.0.tar.gz:"
echo "  \"Pre-built Docker image for offline installation (optional)\""
echo ""