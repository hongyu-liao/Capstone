# 🚀 PDF Image Analyzer - v1.1.0

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-orange.svg)](https://developer.nvidia.com/cuda-downloads)

AI-powered PDF analysis system that extracts and analyzes images, charts, and text from PDF documents.

## 🎯 Quick Start

### Option 1: Direct Python Execution (Faster)

**Requirements:** Python 3.11+, 16GB+ RAM

```bash
# Clone and install
git clone https://github.com/hongyu-liao/Capstone.git
cd Capstone/docker_deployment
pip install -r requirements.txt

# Run
python main.py your_document.pdf
```

### Option 2: Docker Deployment (No Python needed)

**Requirements:** Docker Desktop, stable internet

```bash
# Clone and build
git clone https://github.com/hongyu-liao/Capstone.git
cd Capstone/docker_deployment
docker build -t pdf-analyzer .

# Run
docker run --gpus all --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer
```

## 📋 Usage

### Interactive Mode
```bash
# Python
python main.py

# Docker
docker run --gpus all --rm -it \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer
```

**Important:** When prompted for PDF path, enter without quotes:
- ✅ `input/document.pdf`
- ❌ `"input/document.pdf"`

### Command Line Mode
```bash
# Python
python main.py document.pdf --model google/gemma-3-12b-it

# Docker
docker run --gpus all --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer input/document.pdf
```

## 📁 Output Files

- `document_enhanced.json` - Complete analysis with images
- `document_nlp_ready.json` - Text-only for NLP processing

## ⚠️ Common Issues

1. **File path errors:** Enter paths without quotes
2. **Docker build fails:** `docker system prune -a -f` then rebuild
3. **CUDA out of memory:** Remove `--gpus all` to use CPU only
4. **First run slow:** Downloads ~10GB of AI models

## 🔧 Options

```bash
# GPU/CPU selection
python main.py document.pdf --device gpu    # Use GPU
python main.py document.pdf --device cpu    # Use CPU only

# Disable features for faster processing
python main.py document.pdf --no-web-search
python main.py document.pdf --no-chartgemma
```