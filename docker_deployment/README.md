# 🚀 PDF Image Analyzer - Docker Edition

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-orange.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A containerized PDF analysis system that extracts, analyzes, and describes images from PDF documents using AI. Built with **Docling SmolDocling** for PDF processing and **Hugging Face transformers** for intelligent image analysis.

## 🌟 Features

- **📄 Smart PDF Processing**: Convert PDFs to structured JSON using Docling SmolDocling VLM Pipeline
- **🖼️ AI Image Analysis**: Intelligent classification and description of document images
- **🔍 Web-Enhanced Context**: Automatic web search for conceptual images with AI-generated summaries
- **🚫 No GUI Required**: Runs entirely in command-line interface
- **🐳 Containerized**: Ready-to-deploy Docker solution
- **⚡ GPU Accelerated**: CUDA support for faster processing (optional)
- **🎯 Device Selection**: Automatic detection and user selection of CPU/GPU devices
- **🌐 Multi-format Output**: Original, enhanced, and NLP-ready JSON formats

## 🎯 Quick Start

### Prerequisites

- Docker Desktop with GPU support (optional)
- 8GB+ RAM (16GB+ recommended)
- 15GB+ free disk space
- Stable internet connection

### One-Click Deployment

#### Windows
```cmd
git clone https://github.com/your-username/pdf-image-analyzer.git
cd pdf-image-analyzer/docker_deployment
deploy.bat
```

#### Linux/macOS
```bash
git clone https://github.com/your-username/pdf-image-analyzer.git
cd pdf-image-analyzer/docker_deployment
./deploy.sh
```

### Manual Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pdf-image-analyzer.git
   cd pdf-image-analyzer/docker_deployment
   ```

2. **Build the Docker image**
   ```bash
   docker build -t pdf-analyzer .
   ```

3. **Prepare your PDFs**
   ```bash
   mkdir input output
   cp /path/to/your/documents/*.pdf ./input/
   ```

4. **Run the analyzer**
   ```bash
   # With GPU support (recommended)
   docker run --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
   
   # CPU only
   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
   ```

## 🎯 Device Selection

The system automatically detects your Torch environment and available devices:

### Automatic Detection
- **PyTorch Version**: Displays current PyTorch version
- **CUDA Availability**: Checks if CUDA is available
- **GPU Devices**: Lists all available GPUs with names and memory
- **Device Selection**: Interactive menu to choose CPU or GPU

### Device Options
1. **CPU**: Slower but more compatible, works on all systems
2. **GPU**: Faster processing, requires CUDA support
   - Single GPU: Automatically selected
   - Multiple GPUs: User can choose specific GPU

### Command Line Options
```bash
# Automatic device detection and selection (default)
python main.py document.pdf --device auto

# Force CPU usage
python main.py document.pdf --device cpu

# Force GPU usage (falls back to CPU if CUDA unavailable)
python main.py document.pdf --device gpu
```

## 📊 Output Files

After processing, you'll find in the `output/` directory:

- **`*_smoldocling.json`** - Raw PDF conversion with embedded images
- **`*_enhanced.json`** - Enhanced with AI image analysis and web context
- **`*_nlp_ready.json`** - Text-only version optimized for NLP processing
- **`processing.log`** - Detailed processing logs

## 🔧 Advanced Usage

### Command Line Options

```bash
# Basic usage with device selection
python main.py document.pdf

# Disable web search (faster processing)
python main.py document.pdf --no-web-search

# Keep images in final output
python main.py document.pdf --keep-images

# Use different model
python main.py document.pdf --model google/gemma-2-9b-it

# Custom output directory
python main.py document.pdf --output-dir /custom/path

# Force specific device
python main.py document.pdf --device cpu
python main.py document.pdf --device gpu
```

### Performance Tuning

```bash
# Allocate more memory for large documents
docker run --memory=16g --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer

# Increase shared memory for batch processing
docker run --shm-size=2g --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer

# Use specific GPU device
docker run --gpus '"device=0"' -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
```

## 🏗️ Architecture

### PDF Processing Pipeline
1. **Device Detection**: Automatic Torch environment and GPU detection
2. **Device Selection**: User chooses between CPU and GPU
3. **Document Conversion**: Docling SmolDocling VLM extracts text and images
4. **Image Classification**: AI identifies informative vs decorative images
5. **Content Analysis**: Detailed description generation for informative images
6. **Web Enhancement**: Contextual search for conceptual images
7. **Output Generation**: Multiple format outputs for different use cases

### Technical Stack
- **Base Image**: PyTorch 2.6.0 with CUDA 12.4
- **PDF Processing**: [Docling](https://docling-project.github.io/docling/) SmolDocling VLM Pipeline
- **AI Models**: Hugging Face Transformers
- **Web Search**: DuckDuckGo Search API
- **Container**: Docker with NVIDIA runtime support
- **Device Management**: Automatic CUDA detection and device selection

## 🛠️ Troubleshooting

### Common Issues

1. **Build fails with dependency errors**
   ```bash
   # Clean and retry
   docker system prune -a -f
   ./fix_and_retry.sh  # or fix_and_retry.bat on Windows
   ```

2. **CUDA out of memory**
   ```bash
   # Use CPU mode
   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
   ```

3. **Model download fails**
   - Ensure stable internet connection
   - First run downloads ~10GB of models
   - Check available disk space

4. **Device selection issues**
   - Check CUDA installation: `nvidia-smi`
   - Verify Docker GPU support: `docker run --gpus all nvidia/cuda:12.4-base-ubuntu20.04 nvidia-smi`
   - Use CPU fallback: `--device cpu`

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 🔄 Development

### Testing Deployment
```bash
python test_deployment.py
```

### Interactive Development
```bash
docker run -it --gpus all -v $(pwd):/app pdf-analyzer /bin/bash
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Docling](https://docling-project.github.io/docling/) for PDF processing capabilities
- [Hugging Face](https://huggingface.co/) for transformer models
- [PyTorch](https://pytorch.org/) for deep learning framework

---

**⭐ If this project helps you, please consider giving it a star!**