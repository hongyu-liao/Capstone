# Example Usage - PDF Image Analyzer Docker

This document provides step-by-step examples for using the PDF Image Analyzer Docker deployment.

## Basic Workflow

### Step 1: Prepare Your Environment

1. **Navigate to the deployment directory:**
   ```bash
   cd docker_deployment
   ```

2. **Ensure Docker Desktop is running**
   - Start Docker Desktop application
   - Wait for it to fully initialize

### Step 2: Quick Start with Example

1. **Place a PDF file in the input directory:**
   ```bash
   # Create input directory if it doesn't exist
   mkdir -p input
   
   # Copy your PDF file
   cp /path/to/your/document.pdf ./input/
   ```

2. **Run the automated deployment:**
   
   **Windows:**
   ```cmd
   deploy.bat
   ```
   
   **Linux/Mac:**
   ```bash
   ./deploy.sh
   ```

### Step 3: Check Results

After processing completes, you'll find in the `output/` directory:

- `document_smoldocling.json` - Raw PDF conversion
- `document_enhanced.json` - With AI image analysis
- `document_nlp_ready.json` - Text-only version for NLP
- `processing.log` - Detailed processing logs

## Advanced Usage Examples

### Example 1: Process Multiple PDFs

```bash
# Place multiple PDFs in input directory
cp research_paper1.pdf research_paper2.pdf ./input/

# Run processing (will handle all PDFs automatically)
./deploy.sh
```

### Example 2: Custom Processing Options

```bash
# Build the image first
docker build -t pdf-analyzer .

# Process with custom options
docker run --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer \
  python main.py /app/input/document.pdf \
  --model google/gemma-2-9b-it \
  --no-web-search \
  --keep-images
```

### Example 3: CPU-Only Processing

```bash
# For systems without GPU
docker run \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer \
  python main.py /app/input/document.pdf
```

### Example 4: Interactive Debugging

```bash
# Start container in interactive mode
docker run -it --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer /bin/bash

# Inside container, run processing manually
python main.py /app/input/document.pdf --output-dir /app/output
```

## Expected Processing Time

- **Small PDF (1-5 pages, 2-3 images):** 2-5 minutes
- **Medium PDF (10-20 pages, 5-10 images):** 5-15 minutes  
- **Large PDF (50+ pages, 20+ images):** 15-45 minutes

*Times vary based on:*
- GPU availability and model size
- Internet speed (for web search)
- Image complexity and count
- System resources

## Output Structure Example

For a PDF named `research_paper.pdf`, you'll get:

```
output/
├── research_paper_smoldocling.json     # Raw extraction
├── research_paper_enhanced.json        # With AI analysis
├── research_paper_nlp_ready.json      # Text-only version
└── processing.log                      # Processing logs
```

## JSON Output Format

### Enhanced JSON Structure
```json
{
  "pictures": [
    {
      "ai_analysis": {
        "image_type": "DATA_VISUALIZATION",
        "description": "A bar chart showing...",
        "analysis_timestamp": "2024-01-15 14:30:22",
        "web_context": {
          "search_query": "statistical analysis",
          "sources": [...],
          "ai_summary": "..."
        }
      }
    }
  ],
  "enhancement_metadata": {
    "original_picture_count": 5,
    "enhanced_picture_count": 3,
    "processing_model": "google/gemma-3-12b-it"
  }
}
```

### NLP-Ready JSON Structure
```json
{
  "pictures": [
    {
      "ai_analysis": {
        "description": "Text description replacing the image...",
        "enriched_description": "Description with web context..."
      }
    }
  ],
  "nlp_ready_metadata": {
    "images_removed": true,
    "nlp_ready": true
  }
}
```

## Troubleshooting Common Issues

### Issue: "No PDF files found"
**Solution:** Ensure PDF files are in the `input/` directory before running

### Issue: "Docker build failed"
**Solution:** 
- Check internet connection
- Ensure sufficient disk space (5GB+)
- Restart Docker Desktop

### Issue: "CUDA out of memory"
**Solution:**
- Use a smaller model: `--model google/gemma-2-2b-it`
- Process fewer files at once
- Use CPU-only mode

### Issue: "Model download failed"
**Solution:**
- Check internet connection
- Clear Docker cache: `docker system prune`
- Try again (models are large, 10GB+)

## Performance Optimization

### For Better GPU Performance
```bash
# Use specific GPU
docker run --gpus device=0 ...

# Limit GPU memory
docker run --gpus all -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 ...
```

### For Faster Processing
```bash
# Disable web search for faster processing
python main.py document.pdf --no-web-search

# Use smaller, faster model
python main.py document.pdf --model google/gemma-2-2b-it
```