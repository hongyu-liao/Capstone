# 📄 PDF Image Analyzer

> **A comprehensive AI-powered PDF analysis system that extracts, analyzes, and enhances PDF documents with intelligent image recognition and web search capabilities.**

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI Powered](https://img.shields.io/badge/AI-powered-brightgreen.svg)

## 🌟 Overview

PDF Image Analyzer is a cutting-edge solution that transforms static PDF documents into intelligent, searchable content. By combining advanced document processing with state-of-the-art AI vision models, it extracts images, analyzes their content, and enhances them with contextual information from web searches.

### 🔧 What's New

- **🎯 Advanced ChartGemma Integration**: State-of-the-art chart analysis with intelligent questioning
  - **🧠 AI-Generated Prompts**: Dynamic prompt generation using AI models for optimal chart analysis
  - **📊 Multi-Chart Detection**: Automatic detection and specialized handling of complex multi-panel figures
  - **🔧 Enhanced Chart Understanding**: Specialized analysis for line charts, bar charts, pie charts, scatter plots, etc.
  - **⚡ Full Platform Support**: Integrated across Streamlit APP, Docker deployment, and Jupyter notebooks
- **🤖 Intelligent Multi-Chart Workflow**: Smart handling of complex visualizations
  - Automatic detection of multi-panel figures and subplot arrangements
  - Dynamic workflow adjustment: DePlot skipped for multi-chart images
  - AI-generated intelligent prompts tailored to image complexity and content
- **📊 Three-Level Chart Analysis Pipeline**: Comprehensive chart understanding system
  - **Level 1**: Gemma AI visual analysis for general understanding and multi-chart detection
  - **Level 2**: DePlot data extraction for structured data tables (single charts only)
  - **Level 3**: ChartGemma specialized analysis with intelligent AI-generated questioning
- **🌐 Enhanced Web Search**: Migrated to `ddgs` package with improved error handling and fallback mechanisms
- **🎨 Dark Theme Compatibility**: Removed white backgrounds for better IDE integration
- **📝 Comprehensive Analysis Output**: DATA_VISUALIZATION images show 3 analysis levels, CONCEPTUAL images show 2 levels
- **🔧 Production-Ready Notebooks**: Updated `PDF_Parsing.ipynb` (formerly `Reproduce_Two_Stages.ipynb`) with complete multi-chart support and intelligent prompt generation
- New modular `Functions/` package with clean, testable functions:
  - PDF→JSON (`pdf_to_json.py`)
  - Image analysis, web search, and enrichment (`image_analysis.py`)
  - DePlot chart extraction with robust parser (`chart_extraction.py`)
  - Two-step processing pipeline (`pipeline_steps.py`)
  - Final JSON verification (no file output, returns structured results) (`verification.py`)
  - Simple logging setup (`utils_logging.py`)
- Reproducible two-stage notebook `PDF_Parsing.ipynb` to separately run:
  1) PDF→JSON; 2) JSON→Enhanced→NLP-ready + inline verification
- Headless DePlot debug tester `Functions/debug_deplot_test.py` (prints structure only)

### ✨ Key Features

- 🔍 **Smart PDF Processing** - Extract text and images using Docling with VLM pipeline
- 🤖 **Multi-AI Provider Support** - Compatible with OpenAI, Google Gemini, Anthropic Claude, and LM Studio
- 🎯 **Intelligent ChartGemma Integration** - Advanced chart analysis with AI-generated prompts
  - **🧠 Dynamic Prompt Generation**: AI models create optimal questions for each chart type and complexity
  - **📊 Multi-Chart Intelligence**: Specialized handling of complex multi-panel visualizations
  - **⚙️ Adaptive Analysis**: Context-aware questioning for line charts, bar charts, pie charts, scatter plots, etc.
- 📊 **Enhanced DePlot Chart Extraction** - Robust chart data extraction with AI verification and categorical X-axis support
- 🌐 **Advanced Web Search Integration** - Automatic contextual enhancement with native APIs and DuckDuckGo fallback
- 📊 **Interactive Reports** - Generate comprehensive HTML evaluation reports
- 🐳 **Production Ready** - Docker containerization for scalable deployment
- 🖥️ **User-Friendly GUI** - Streamlit web interface for easy interaction
- 📈 **Research Tools** - Specialized notebooks for chart data extraction and analysis

## 🚀 Quick Start

### Option 1: Direct Python Execution (If you have Python environment)

**For users with Python 3.11+ and PyTorch environment:**
```bash
# Clone the repository
git clone https://github.com/hongyu-liao/Capstone.git
cd Capstone/docker_deployment

# Install dependencies (from our torch260 environment)
pip install -r requirements.txt

# Run directly
python main.py your_document.pdf
```

### Option 2: Complete Docker Deployment (No Python needed)

**For users without Python/PyTorch environment - Docker provides everything:**
```bash
# Clone the repository
git clone https://github.com/hongyu-liao/Capstone.git
cd Capstone/docker_deployment

# Build the complete environment (includes Python, CUDA, all models)
docker build -t pdf-analyzer .

# Run with your PDF
docker run --gpus all --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-analyzer
```

### Option 3: Streamlit GUI Application

```bash
# Clone the repository
git clone https://github.com/hongyu-liao/Capstone.git
cd Capstone/PDF_Analyzer_App

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🔧 System Requirements & Compatibility

### 📋 Supported Versions

This project has been tested and confirmed compatible with:

- **PyTorch**: 2.6.0 (recommended)
- **CUDA**: 12.6 (officially supported), 12.9 (user-tested compatible)
- **Python**: 3.11+
- **Docling**: 2.40.0+

**Important Notes:**
- PyTorch 2.6.0 officially supports CUDA 12.6, but higher CUDA versions (like 12.9) may also be compatible
- For different CUDA/PyTorch versions, use the Jupyter notebooks with manual environment configuration

### 🔀 Environment Configuration Options

#### Option A: Docker/Streamlit APP (Recommended)
- Use the pre-configured Docker deployment or Streamlit APP for optimal compatibility
- Automatically handles dependencies and version conflicts

#### Option B: Manual Configuration (Custom Environments)
If your CUDA or PyTorch versions differ from the recommended versions:

**All Platforms (Recommended):**
- Use `PDF_Parsing.ipynb` with the `Functions/` folder
- Cross-platform compatibility using Ollama for AI model management
- HuggingFace integration for specialized chart analysis (ChartGemma, DePlot)
- Works on Windows, macOS, and Linux

**LM Studio Users (Windows-optimized):**
- Use `PDF_Parsing_LMStudio.ipynb` for LM Studio-specific workflow
- Requires LM Studio installation and model service mounting
- Optimized for Windows environments with local model management

### 📚 Reference Documentation

- **Docling Official Docs**: [https://docling-project.github.io/docling/](https://docling-project.github.io/docling/)
- **PyTorch Installation Guide**: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- **CUDA Compatibility Matrix**: [https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)

## 📦 Installation Options

### 🐳 Docker Deployment (Production)

Perfect for server environments, CI/CD pipelines, and production use cases.

**Prerequisites:**
- Docker Desktop installed
- NVIDIA GPU with 8GB+ VRAM (recommended)
- NVIDIA drivers (any recent version)
- NVIDIA Container Toolkit
- 16GB+ RAM
- 50GB+ free disk space

**Quick Setup:**
```bash
cd docker_deployment
./deploy.bat  # Windows
./deploy.sh   # Linux/macOS
```

**Features:**
- ✅ Headless operation (no GUI required)
- ✅ GPU acceleration support
- ✅ SmolDocling VLM pipeline
- ✅ Automated dependency management
- ✅ Volume mounting for input/output
- ✅ Production-optimized logging

### 🖥️ GUI Application (Development & Interactive Use)

Ideal for researchers, analysts, and interactive document processing.

**Prerequisites:**
- Python 3.11+
- AI provider API keys (OpenAI, Gemini, Claude) OR LM Studio

**Setup:**
```bash
# Navigate to GUI application
cd PDF_Analyzer_App

# Install dependencies
pip install -r requirements.txt

# Configure AI provider (choose one):
# - Set API keys in sidebar for cloud providers
# - Install and run LM Studio for local processing

# Launch application
streamlit run app.py
```

**Features:**
- ✅ Interactive web interface
- ✅ Real-time processing monitoring
- ✅ Multiple AI provider support
- ✅ Batch processing capabilities
- ✅ Interactive evaluation reports
- ✅ Advanced filtering and sorting

## 🛠️ AI Provider Configuration

### Cloud Providers (Recommended)

| Provider | Features | Setup |
|----------|----------|--------|
| **Google Gemini** | Native web search, latest models | Set `GOOGLE_API_KEY` |
| **OpenAI GPT** | GPT-4o, web browsing* | Set `OPENAI_API_KEY` |
| **Anthropic Claude** | Advanced reasoning | Set `ANTHROPIC_API_KEY` |

### Local Processing

| Provider | Features | Setup |
|----------|----------|--------|
| **LM Studio** | Privacy, offline | Install LM Studio + vision model |

*Web browsing availability varies by model and account type

## 📊 Processing Workflow

```mermaid
graph TD
    A[PDF Input] --> B[Docling Extraction]
    B --> C[Image Detection]
    C --> D[AI Analysis]
    D --> E{Image Type?}
    E -->|Data Visualization| F[Extract Data Points]
    E -->|Conceptual| G[Web Search]
    F --> H[Enhanced JSON]
    G --> H
    H --> I[NLP-Ready Output]
    H --> J[HTML Reports]
```

### Step-by-Step Process

1. **📄 Document Processing** - Extract text and images using Docling VLM pipeline
2. **🔍 Image Analysis** - Classify images as informative vs. non-informative
3. **🤖 AI Enhancement** - Generate detailed descriptions using vision models
4. **🌐 Web Context** - Add relevant background information via web search
5. **📊 Report Generation** - Create interactive HTML evaluation reports
6. **💾 Output Creation** - Generate multiple output formats for different use cases

## 📁 Project Structure

```
PDF-Image-Analyzer/
├── 🖥️ PDF_Analyzer_App/           # Streamlit GUI Application
│   ├── app.py                     # Main application entry point
│   ├── pdf_processor.py           # PDF to JSON conversion
│   ├── image_analyzer.py          # AI image analysis
│   ├── api_manager.py            # Multi-provider AI interface
│   ├── html_report_generator.py  # Report generation
│   └── output/                   # Processing results
│
├── 🧩 Functions/                 # Modular function package (new)
│   ├── pdf_to_json.py            # LM Studio + Docling PDF→JSON
│   ├── image_analysis.py         # Image analysis + web search + enrichment
│   ├── chart_extraction.py       # DePlot extraction + robust parser
│   ├── pipeline_steps.py         # Step1 (enhance) + Step2 (NLP-ready)
│   ├── verification.py           # Final JSON validator (returns dict)
│   ├── utils_logging.py          # Logging setup
│   └── debug_deplot_test.py      # Headless DePlot tester
│
├── 🐳 docker_deployment/          # Production Docker Setup
│   ├── main.py                   # Containerized processing engine
│   ├── Dockerfile                # Container configuration
│   ├── deploy.sh/.bat           # Automated deployment
│   ├── quick-start-*.sh/.bat    # One-click installers
│   └── docs/                    # Deployment documentation
│
├── 📓 Research Notebooks/          # Development & Analysis Tools
│   ├── PDF_extract_and_Picture_Describe.ipynb  # Original research
│   ├── data_extract.ipynb        # Chart data extraction
│   └── json2html.ipynb          # Format conversion utilities
│
├── 📂 Sample Data/
│   ├── Sample Papers/            # Test PDF documents
│   └── Sample Line Chart/        # Chart analysis examples
│
├── PDF_Parsing.ipynb              # Universal 2-stage pipeline notebook (Ollama + HuggingFace)
│                                  # Stage 1: PDF→JSON; Stage 2: JSON→Enhanced→NLP-ready + verification
│                                  # Cross-platform: Windows, macOS, Linux
├── PDF_Parsing_LMStudio.ipynb     # LM Studio optimized notebook (Windows-focused)
│                                  # Same functionality but uses LM Studio for local model management
│
└── 📚 Documentation/
    ├── PROJECT_FUNCTIONS_DOCUMENTATION.md  # Complete function reference
    └── README.md                 # This file
```

## 🔧 Advanced Usage

### Batch Processing

**GUI Application:**
```python
# Upload multiple PDFs through the web interface
# Processing happens automatically with progress tracking
# Results are organized in tabs for easy navigation
```

**Docker Command Line:**
```bash
# Place PDFs in input directory
docker run --gpus all -v ./input:/app/input -v ./output:/app/output pdf-analyzer:latest

# Results appear in output directory
# Enhanced JSON, NLP-ready versions, and HTML reports generated
```

### Custom AI Provider Integration

```python
# Add new AI provider in api_manager.py
def _call_custom_provider(self, image_uri: str, prompt: str, max_tokens: int):
    # Implement custom API integration
    pass
```

### Research and Development

```python
# Chart data extraction
from data_extract import get_color_mask, find_axes_automatically

# JSON to HTML conversion
from json2html import convert_json_to_html

# Original LM Studio workflow
from PDF_extract_and_Picture_Describe import convert_pdf_with_lmstudio
```

## 📈 Output Formats

### 📊 JSON Outputs

- **`filename.json`** - Original Docling extraction with embedded images
- **`filename_enhanced.json`** - Added AI analysis and web context
- **`filename_nlp_ready.json`** - Text-only version optimized for NLP processing

### 📑 HTML Reports

- **`filename_report.html`** - Interactive image analysis report
- **`filename_complete_report.html`** - Comprehensive evaluation dashboard

### 🔍 Report Features

- Interactive image viewing with click-to-enlarge
- Web search results with source attribution
- Processing statistics and quality metrics
- Evaluation checklists for systematic assessment
- Exportable results and downloadable assets

## 🎯 Use Cases

### 📚 Academic Research
- Extract and analyze figures from research papers
- Generate searchable databases of academic content
- Create enhanced digital libraries

### 📈 Business Intelligence
- Process financial reports and presentations
- Extract insights from charts and visualizations
- Generate summaries for executive briefings

### 📄 Document Management
- Digitize and enhance document archives
- Create searchable content databases
- Automate document classification

### 🔬 Data Science
- Preprocess documents for ML pipelines
- Extract structured data from unstructured sources
- Generate training datasets for vision models

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

# Install development dependencies
pip install -r PDF_Analyzer_App/requirements.txt
pip install -r docker_deployment/requirements.txt

# Install research dependencies
pip install jupyter opencv-python matplotlib

# Run tests
python -m pytest
```

## 📋 Requirements

### System Requirements

**Minimum:**
- 8GB RAM
- 20GB free disk space
- Python 3.11+
- Internet connection (for AI APIs and web search)

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ free disk space
- High-speed internet connection

### Software Dependencies

**Core Dependencies:**
- docling>=2.40.0 (PDF processing)
- streamlit>=1.47.1 (GUI interface)
- requests, pandas, numpy (data processing)
- Pillow (image handling)

**AI Provider SDKs:**
- openai (OpenAI GPT)
- google-genai (Google Gemini)
- anthropic (Claude)

**Optional Dependencies:**
- opencv-python (chart analysis)
- matplotlib (visualization)
- jupyter (research notebooks)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Docling Team** for the excellent PDF processing framework
- **Streamlit** for the intuitive web interface framework
- **AI Providers** (OpenAI, Google, Anthropic) for powerful vision models
- **LM Studio** for local AI processing capabilities
- **DuckDuckGo** for privacy-respecting web search

## 📞 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/yourusername/yourrepo/issues)
- 📖 **Documentation**: [Function Reference](PROJECT_FUNCTIONS_DOCUMENTATION.md)
- 🐳 **Docker Guide**: [Docker Deployment](docker_deployment/README.md)
- 🔧 **Troubleshooting**: [Docker Troubleshooting](docker_deployment/TROUBLESHOOTING.md)

## 🚀 Recent Updates

### v1.0.0 - Docker Production Release
- ✅ Complete Docker containerization
- ✅ One-click deployment scripts
- ✅ Multi-platform support (Windows/Linux/macOS)
- ✅ GitHub Release automation
- ✅ Comprehensive English documentation
- ✅ Production-ready error handling

### v1.1.0 - Functions Module and Reproducible Notebook
- 🧩 Introduced `Functions/` package (clean modularization)
- 🧪 Added `PDF_Parsing.ipynb` (formerly `Reproduce_Two_Stages.ipynb`) to independently run two-stage processing
- 🔍 Added headless DePlot tester and robust parser improvements
- ✅ Integrated inline final JSON verification (no file output)

### v1.2.0 - Intelligent ChartGemma Enhancement
- 🧠 **AI-Generated Prompt System**: Dynamic ChartGemma prompt generation using AI models
- 🎯 **Enhanced APP Integration**: Full intelligent prompt generation support in Streamlit APP
- 🐳 **Docker Intelligence Upgrade**: Advanced prompt generation with model-based and enhanced fallback methods
- ⚡ **Performance Optimization**: Smart content-based prompt selection for different chart types
- 🔧 **Complete Notebook Parity**: APP and Docker now match notebook's intelligent analysis capabilities

### v1.3.0 - Enhanced Docker Experience & Core Improvements
- 🤖 **Interactive Model Selection**: Docker now prompts for Hugging Face model addresses with intelligent defaults
- 📄 **Flexible PDF Conversion**: Choose between SmolDocling (recommended) or custom Hugging Face models
- 📊 **Progress Tracking**: Real-time progress display showing current image processing status
- 🎯 **Streamlined Output**: Always generates both Enhanced and NLP-ready JSON files with correct naming
- 🐳 **Enhanced Docker UX**: Better user interaction and configuration options for production deployments

### v1.4.0 - Universal Platform Support & Hybrid AI Architecture
- 🌐 **Universal Notebook**: `PDF_Parsing.ipynb` - True cross-platform compatibility (Windows, macOS, Linux)
- 🔄 **Hybrid AI Integration**: Combines Ollama (cross-platform) + HuggingFace (specialized) for optimal performance
- 🎯 **Smart Model Distribution**: Ollama for general analysis, ChartGemma/DePlot for specialized chart processing
- 🔧 **LM Studio Specialization**: `PDF_Parsing_LMStudio.ipynb` for Windows users preferring LM Studio workflow
- ⚡ **Optimized Performance**: Each AI model used for its strengths - Ollama for compatibility, HuggingFace for accuracy
- 📋 **Enhanced Documentation**: Clear platform support matrix and model usage guidelines

---

<div align="center">
  <strong>Transform your PDFs into intelligent, searchable content with AI-powered analysis!</strong>
  <br><br>
  ⭐ If this project helps you, please give it a star on GitHub! ⭐
</div>