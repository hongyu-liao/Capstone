#!/bin/bash

echo "=== PDF Image Analyzer - Fix and Retry Script ==="
echo ""

echo "🧹 Cleaning Docker cache and failed builds..."
docker system prune -a -f
docker builder prune -a -f

echo ""
echo "📦 Removing any existing pdf-analyzer images..."
docker image rm pdf-analyzer 2>/dev/null

echo ""
echo "🔧 Checking Docker system status..."
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"
echo ""

echo "🚀 Starting fresh build..."
./deploy.sh

echo ""
echo "🎉 Fix and retry complete!"