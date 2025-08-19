# 📋 Project Files Log

*This file is for AI assistants to quickly understand the project structure and is not uploaded to GitHub.*

## 🏗️ Project Structure Overview

This is a comprehensive PDF image analysis system with multiple deployment options and advanced AI-powered chart analysis capabilities.

### 📁 Core Components

#### `/Functions/` - Core Function Library
- **`__init__.py`** - Package initialization
- **`chart_extraction.py`** - DePlot chart data extraction with robust parsing
- **`image_analysis.py`** - AI image analysis, web search, and ChartGemma integration
- **`pdf_to_json.py`** - PDF to JSON conversion using Docling
- **`pipeline_steps.py`** - Two-stage processing pipeline orchestration
- **`utils_logging.py`** - Logging configuration utilities
- **`verification.py`** - Final JSON verification and validation

#### `/PDF_Analyzer_App/` - Streamlit Web Application
- **`app.py`** - Main Streamlit application with ChartGemma integration
- **`api_manager.py`** - Multi-AI provider support (OpenAI, Gemini, Anthropic, LM Studio)
- **`batch_processor.py`** - Batch file processing with ChartGemma support
- **`config.py`** - Configuration management
- **`html_report_generator.py`** - HTML report generation
- **`image_analyzer.py`** - Image analysis with ChartGemma integration
- **`pdf_processor.py`** - PDF processing for Streamlit
- **`run_app.py`** - Application launcher
- **`start_app.bat`** - Windows launcher
- **`requirements.txt`** - Updated with ddgs and ChartGemma dependencies
- **`README.md`** - Updated with ChartGemma features

#### `/docker_deployment/` - Docker Deployment
- **`main.py`** - Docker main processing with ChartGemma support
- **`Dockerfile`** - Container configuration
- **`docker-compose.yml`** - Docker Compose setup
- **`requirements.txt`** - Updated with ddgs dependency
- **`README.md`** - Updated with ChartGemma options
- **Various deployment scripts** - Windows/Linux deployment automation

#### Main Notebooks
- **`Reproduce_Two_Stages.ipynb`** - **MAIN PIPELINE** with ChartGemma integration
  - Stage 1: PDF → Enhanced JSON (with AI descriptions, chart data, web context)
  - Stage 2: Enhanced JSON → NLP-ready JSON (image removal)
  - Stage 3: Visualization with multi-level analysis display
- **`PDF_extract_and_Picture_Describe.ipynb`** - Alternative PDF processing notebook

#### `/temp_testing_scripts/` - Archived Testing Files
- **`test_chartgemma.py`** - ChartGemma testing script (archived)
- **`run_chartgemma_simple.py`** - Simple ChartGemma testing (archived)
- **`Line_Chart_Extract.py`** - Legacy chart extraction (archived)
- **Various testing notebooks** - Data extraction and HTML conversion tests

### 🎯 Key Technologies Integrated

#### AI Models and APIs
- **ChartGemma** - `ahmed-masry/chartgemma` for specialized chart analysis
- **DePlot** - Google's `google/deplot` for chart data extraction
- **Multi-AI Support** - OpenAI GPT, Google Gemini, Anthropic Claude, LM Studio
- **Web Search** - `ddgs` (updated from `duckduckgo_search`)

#### Data Processing
- **Docling** - PDF processing with VLM pipeline
- **Image Classification** - INFORMATIVE/NON-INFORMATIVE, DATA_VISUALIZATION/CONCEPTUAL
- **Chart Data Extraction** - Three-level pipeline: AI analysis + DePlot + ChartGemma
- **Web Enhancement** - Contextual search for CONCEPTUAL images

#### Output Formats
- **Enhanced JSON** - Full analysis with images
- **NLP-ready JSON** - Text-only for downstream processing
- **HTML Reports** - Interactive visualization and analysis

### 🔄 Processing Workflow

1. **PDF Input** → Docling conversion → **Base JSON**
2. **Image Analysis** → AI classification → Enhanced descriptions
3. **Chart Processing** (for DATA_VISUALIZATION):
   - Gemma visual analysis
   - DePlot data extraction
   - ChartGemma specialized analysis
4. **Web Enhancement** (for CONCEPTUAL) → Search and summarization
5. **Output Generation** → Enhanced JSON + NLP-ready JSON + HTML reports

### 📊 Chart Analysis Pipeline

#### Three-Level Chart Analysis
1. **Gemma AI Visual Analysis** - General image understanding and classification
2. **DePlot Data Extraction** - Structured data table extraction with robust parsing
3. **ChartGemma Specialized Analysis** - Chart-type-specific questioning and insights

#### Chart Type Support
- **Line Charts** - Trend analysis, extrema, pattern identification
- **Bar Charts** - Category comparison, highest/lowest values
- **Pie Charts** - Segment percentages, largest/smallest portions
- **Scatter Plots** - Correlation analysis, outlier detection
- **Other Charts** - Generic detailed analysis

### 🌐 Deployment Options

1. **Streamlit Web App** - Interactive GUI with full ChartGemma integration
2. **Docker Container** - Headless processing with command-line options
3. **Jupyter Notebooks** - Research and development environment
4. **Function Library** - Modular components for custom integration

### 🔧 Recent Major Updates

- **ChartGemma Integration** - Added to all deployment methods
- **Enhanced Chart Pipeline** - Three-level analysis system
- **Web Search Updates** - Migrated to `ddgs` package
- **Dark Theme Support** - Removed white backgrounds
- **Improved Error Handling** - Robust fallback mechanisms
- **Image Data Removal** - Comprehensive NLP-ready processing

### 🚨 Important Notes for AI Assistants

1. **ChartGemma** is now fully integrated across all components
2. **Web search** uses `ddgs` package (not `duckduckgo_search`)
3. **Three analysis types** for DATA_VISUALIZATION: Gemma + DePlot + ChartGemma
4. **Two analysis types** for CONCEPTUAL: Gemma + Web Search
5. **Image removal** is comprehensive and recursive for NLP-ready outputs
6. **Dark theme compatibility** - no hardcoded white backgrounds
7. **Modular design** - Functions can be used independently

### 📝 File Naming Conventions

- **Base JSON** - `{filename}.json`
- **Enhanced JSON** - `{filename}_with_descriptions_and_chart_data.json`
- **ChartGemma Enhanced** - `{filename}_with_descriptions_and_chart_data_with_chartgemma.json`
- **NLP-ready JSON** - `{filename}_nlp_ready.json`
- **HTML Reports** - `{filename}_complete_report.html`

This project represents a comprehensive solution for intelligent PDF analysis with state-of-the-art chart understanding capabilities.
