# 📄 PDF Image Analyzer App

Simple Streamlit web app for analyzing PDF documents with AI-powered image recognition and description generation.

## 🚀 Quick Start

### Requirements
- Python 3.8+
- 8GB+ RAM

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📋 How to Use

### 1. 🤖 Select AI Provider
Choose from the sidebar:
- **🏠 LM Studio**: Local AI (free, requires LM Studio running)
- **🔮 Google Gemini**: Enter API key
- **🧠 OpenAI**: Enter API key  
- **🎭 Anthropic**: Enter API key

### 2. 📁 Upload File
- **PDF**: Complete analysis (PDF→JSON + image analysis)
- **JSON**: Quick analysis (skip PDF conversion)

### 3. 🎛️ Configure Options
- **🌐 Web Search**: Enhance images with context
- **📊 Chart Extraction**: Extract chart data
- **📝 NLP-Ready**: Generate text-only version

## 📁 Output Files

- `filename_enhanced.json` - Complete analysis with images
- `filename_nlp_ready.json` - Text-only for NLP processing
- `filename_report.html` - Visual analysis report

## ⚠️ Common Issues

1. **API errors**: Check your API keys
2. **Model not found**: Use exact model names
3. **LM Studio**: Ensure it's running with a vision model
4. **Memory issues**: Try smaller files or CPU mode

## 🔧 Advanced Options

```bash
# Install optional packages for enhanced features
pip install google-genai  # For Gemini web search
pip install torch transformers  # For chart analysis
```

For detailed troubleshooting, check the app's error messages and logs.