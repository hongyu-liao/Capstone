# 🚀 PDF Image Analyzer - Docker Deployment Quick Start Guide

## Important Notes
This Docker deployment version uses **SmolDocling** for PDF conversion and **Hugging Face transformers** for image analysis, running completely in Linux environment without requiring a graphical interface.

## Quick Start Steps

### 1. Ensure Environment is Ready
```bash
# Check if Docker is running
docker info

# Check GPU availability (optional)
nvidia-smi
```

## ⚠️ Important: If Previous Build Failed

If you previously ran `deploy.bat` and it failed, please run the fix script first:

```cmd
# Windows users (recommended)
fix_and_retry.bat

# Or manual cleanup
docker system prune -a -f
docker builder prune -a -f
docker image rm pdf-analyzer
```

### 2. Test Deployment (Recommended)
```bash
cd docker_deployment
python test_deployment.py
```

### 3. Prepare Documents
```bash
# Copy your PDF files to input directory
cp /path/to/your/documents/*.pdf ./input/
```

### 4. One-Click Processing

**Windows users:**
```cmd
deploy.bat
```

**Linux/Mac users:**
```bash
./deploy.sh
```

### 5. View Results
After processing completes, check the `output/` directory:
- `*_smoldocling.json` - Raw PDF conversion results
- `*_enhanced.json` - Enhanced version with AI image analysis
- `*_nlp_ready.json` - NLP-ready version with image data removed
- `processing.log` - Detailed processing logs

## Technical Architecture

### PDF Conversion
- Uses **Docling SmolDocling VLM Pipeline**
- Reference: https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/
- Default transformers framework

### Image Analysis
- Based on original prompt structure from your notebook
- Intelligent identification of informative vs decorative images
- Supports DATA_VISUALIZATION and CONCEPTUAL image classification

### Web Search
- DuckDuckGo search for conceptual images
- AI-generated summaries to enhance image descriptions
- Provides additional contextual information

## Command Line Options

```bash
# Basic usage
python main.py document.pdf

# Disable web search (faster)
python main.py document.pdf --no-web-search

# Keep image data
python main.py document.pdf --keep-images

# Use different model
python main.py document.pdf --model google/gemma-2-9b-it

# Custom output directory
python main.py document.pdf --output-dir /custom/output/path
```

## Performance Optimization

### GPU Acceleration
- Ensure NVIDIA Container Toolkit is installed
- Use `--gpus all` flag
- Recommended 8GB+ VRAM

### CPU Mode
- Remove GPU flag to use CPU
- Slower processing but still functional
- Suitable for servers without GPU

### Memory Optimization
```bash
# For large documents
docker run --memory=16g --gpus all ...

# For batch processing multiple documents
docker run --shm-size=2g --gpus all ...
```

## Troubleshooting

### Common Issues

1. **"Docker is not running"**
   - Start Docker Desktop
   - Wait for complete initialization

2. **"CUDA out of memory"**
   - Use smaller model: `--model google/gemma-2-2b-it`
   - Reduce concurrent document processing
   - Use CPU mode

3. **"No PDF files found"**
   - Ensure PDF files are in `input/` directory
   - Check file permissions

4. **Model download failed**
   - Ensure stable internet connection
   - First run requires downloading large models (10GB+)
   - Please wait patiently for download completion

### Debug Mode
```bash
# Interactive run for debugging
docker run -it --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer /bin/bash

# Manual run inside container
python main.py /app/input/document.pdf --output-dir /app/output
```

## Differences from Original Notebook

| Feature | Notebook Version | Docker Version |
|---------|------------------|----------------|
| PDF Conversion | LM Studio VLM | SmolDocling VLM |
| Image Analysis | LM Studio API | Simplified heuristic analysis* |
| Web Search | DuckDuckGo + LM Studio | DuckDuckGo + simplified summary |
| Deployment | Local GUI | Docker container |
| Dependencies | LM Studio server | Fully self-contained |

*Note: Image analysis uses heuristic methods based on image size and complexity. For full VLM analysis, actual vision-language models can be integrated.

## Next Steps

1. **Test deployment**: Run `python test_deployment.py`
2. **Prepare documents**: Place PDFs in `input/` directory
3. **Start processing**: Run `deploy.bat` or `./deploy.sh`
4. **Check results**: View result files in `output/` directory

🎉 **Ready to go! Your GUI-free PDF Image Analyzer is now available in Docker!**