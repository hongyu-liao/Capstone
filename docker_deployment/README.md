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
   # With GPU support
   docker run --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
   
   # CPU only
   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
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
# Basic usage
python main.py document.pdf

# Disable web search (faster processing)
python main.py document.pdf --no-web-search

# Keep images in final output
python main.py document.pdf --keep-images

# Use different model
python main.py document.pdf --model google/gemma-2-9b-it

# Custom output directory
python main.py document.pdf --output-dir /custom/path
```

### Performance Tuning

```bash
# Allocate more memory for large documents
docker run --memory=16g --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer

# Increase shared memory for batch processing
docker run --shm-size=2g --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-analyzer
```

## 🏗️ Architecture

### PDF Processing Pipeline
1. **Document Conversion**: Docling SmolDocling VLM extracts text and images
2. **Image Classification**: AI identifies informative vs decorative images
3. **Content Analysis**: Detailed description generation for informative images
4. **Web Enhancement**: Contextual search for conceptual images
5. **Output Generation**: Multiple format outputs for different use cases

### Technical Stack
- **Base Image**: PyTorch 2.6.0 with CUDA 12.4
- **PDF Processing**: [Docling](https://docling-project.github.io/docling/) SmolDocling VLM Pipeline
- **AI Models**: Hugging Face Transformers
- **Web Search**: DuckDuckGo Search API
- **Container**: Docker with NVIDIA runtime support

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

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📋 System Requirements

**Minimum:**
- 8GB RAM
- 5GB free disk space
- Docker Desktop 4.0+

**Recommended:**
- 16GB+ RAM
- 20GB+ free disk space
- NVIDIA GPU with 8GB+ VRAM
- Docker Desktop with NVIDIA Container Toolkit

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

## 📧 Support

- **Documentation**: Check [start.md](start.md) for quick start guide
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join GitHub Discussions for questions

## 🙏 Acknowledgments

- [Docling](https://docling-project.github.io/docling/) for PDF processing capabilities
- [Hugging Face](https://huggingface.co/) for transformer models
- [PyTorch](https://pytorch.org/) for deep learning framework

---

**⭐ If this project helps you, please consider giving it a star!**