# 📄 PDF Image Analyzer Pro

A comprehensive tool for extracting, analyzing, and enhancing images from PDF documents using AI-powered analysis and web search capabilities. Now featuring **Batch Processing** for multiple files!

## 🌟 Features

### 🔄 Processing Modes
- **📁 Single File Processing**: Process individual PDF or JSON files with detailed analysis
- **📚 Batch Processing**: Upload and process multiple files simultaneously with organized output
- **⚡ JSON Analysis**: Skip PDF conversion by uploading pre-processed JSON files
- **🗂️ Organized Output**: Each file gets its own subdirectory with standardized naming

### 🤖 Multi-AI Provider Support
- **🏠 LM Studio (Local)**: Free local AI processing with any compatible model
- **🧠 OpenAI (ChatGPT)**: GPT-4o, GPT-4o-mini, and other ChatGPT models
- **🔮 Google (Gemini)**: Latest Gemini models including 2.5 Pro and Flash variants
- **🎭 Anthropic (Claude)**: Claude 3.5 Sonnet, Claude 4, and other Claude models

### 🌐 Advanced Web Search Integration
- **🌐 Native Web Search**: 
  - **Gemini**: Real-time Google Search integrated directly into AI analysis
  - **ChatGPT**: Web browsing capabilities for supported models
- **🦆 DuckDuckGo Fallback**: External search with AI summarization for local models and Claude
- **🔄 Automatic Selection**: Uses best available search method based on AI provider

### 🖼️ Intelligent Image Analysis
- **🔍 Smart Classification**: Automatically identifies informative vs. non-informative images
- **📊 Type Detection**: Distinguishes between data visualizations and conceptual diagrams
- **🎯 ChartGemma Integration**: Advanced chart analysis with specialized questioning for different chart types
- **📈 DePlot Chart Extraction**: Robust chart data extraction with AI verification
- **🎯 Logo Filtering**: Automatically skips publisher logos and watermarks
- **📝 Comprehensive Descriptions**: Generates detailed text descriptions for NLP processing

### 🎨 Enhanced User Interface
- **📱 Adaptive Display**: Images displayed at optimal resolution without forced stretching
- **🔍 Click-to-Enlarge**: Full-size image viewing with download options
- **📊 Three-Tab Design**: Organized workflow with Process, Results, and Analytics tabs
- **🎛️ Advanced Filtering**: Filter results by image type, search status, and quality

### 📋 Professional HTML Evaluation Reports
- **📊 Image Analysis Report**: Comprehensive HTML report showing all image recognition results
- **📋 Complete Evaluation Report**: Interactive three-tab report combining text extraction and image analysis
- **🎯 Evaluation-Focused**: Specifically designed for assessing PDF processing effectiveness
- **🖱️ Interactive Features**: Click-to-enlarge images, tabbed navigation, evaluation checklists

### 📊 Professional Analytics
- **📈 Processing History**: Track all analysis sessions with detailed statistics
- **📊 Comparative Analysis**: Compare results across multiple documents
- **📋 Comprehensive Reports**: Generate detailed analysis reports
- **📥 Multiple Export Formats**: JSON, CSV, Excel, ZIP, and HTML exports

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- For Gemini web search: `pip install google-genai`
- For local models: LM Studio running with a vision-capable model
- For external APIs: Valid API keys from OpenAI, Google, or Anthropic

### Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## 📖 How to Use

### 1. 🤖 Configure AI Provider

In the sidebar, select your preferred AI provider:

#### 🏠 LM Studio (Local)
- Start LM Studio with a vision-capable model
- Enter the LM Studio URL (default: `http://localhost:1234/v1/chat/completions`)
- **Web Search**: Uses DuckDuckGo with AI summarization

#### 🔮 Google (Gemini)
- Select from latest models: `gemini-2.5-pro`, `gemini-2.5-flash`, etc.
- Enter your Google AI API key
- **Web Search**: 🌐 **Native Google Search** - Real-time web results integrated directly

#### 🧠 OpenAI (ChatGPT)
- Choose models: `gpt-4o`, `gpt-4o-mini`, etc.
- Enter your OpenAI API key
- **Web Search**: Native browsing for supported models, DuckDuckGo fallback

#### 🎭 Anthropic (Claude)
- Select models: `claude-4-sonnet-latest`, `claude-3-5-sonnet`, etc.
- Enter your Anthropic API key
- **Web Search**: Uses DuckDuckGo with AI summarization

### 2. 📁 Upload Files

Choose your processing method:

#### 📁 Single File Processing
- **🆕 New PDF Processing**: Upload a PDF file for complete analysis (PDF→JSON conversion + image analysis)
- **⚡ Quick JSON Analysis**: Upload a previously converted JSON file (skips PDF conversion)

#### 📚 Batch Processing
- **Multiple PDF Files**: Upload multiple PDF files for batch processing
- **Multiple JSON Files**: Upload multiple pre-processed JSON files
- **Mixed Processing**: Handle different file types in one batch
- **Organized Output**: Each file gets its own subdirectory with standardized naming

### 3. 🎛️ Configure Options

- **🌐 Enable Web Search**: Enhance conceptual images with web context
- **📊 Enable Chart Data Extraction**: Extract chart data using DePlot + AI verification
- **🎯 Enable ChartGemma Analysis**: Use specialized ChartGemma model for advanced chart understanding
- **📝 Generate NLP-Ready JSON**: Create version without images for text processing
- **📁 Output Directory**: Set single file output or batch base directory

### 4. 🚀 Process and Analyze

#### Single File Processing

Click "🚀 Start Processing" to begin analysis. The system will:

1. **Convert/Load**: Process PDF or load JSON
2. **Load ChartGemma**: Initialize advanced chart analysis model (if enabled)
3. **Analyze Images**: AI classification and description with multi-level analysis:
   - Gemma AI visual analysis
   - DePlot chart data extraction (for DATA_VISUALIZATION)
   - ChartGemma specialized analysis (for DATA_VISUALIZATION)
4. **Web Enhancement**: Search for additional context (if enabled)
5. **Generate Outputs**: Create enhanced and NLP-ready versions

#### Batch Processing

Click "🚀 Start Batch Processing" to process multiple files:

1. **File Organization**: Create individual subdirectories for each file
2. **Sequential Processing**: Process each file with progress tracking
3. **Error Handling**: Continue processing even if individual files fail
4. **Result Aggregation**: Collect all results for batch analysis
5. **Batch Reports**: Generate HTML reports and summaries for all files

## 🌐 Web Search Features

### 🔮 Gemini Native Search
```python
# Gemini automatically integrates Google Search
response = gemini.analyze_with_web_search(
    image=image_data,
    prompt="Analyze this scientific diagram",
    tools=[GoogleSearch()]
)
# Web results are seamlessly integrated into the analysis
```

### 🦆 DuckDuckGo Fallback
```python
# For local models and Claude
keywords = ["climate change", "carbon cycle"]
search_results = duckduckgo.search(keywords)
ai_summary = ai.summarize(search_results)
```

## 📊 Output Files

### 📄 Enhanced JSON
- Original document structure + AI analysis
- Detailed image descriptions and classifications
- Web search context and sources
- Processing metadata and statistics

### 🔤 NLP-Ready JSON
- Text-only version without image data
- Preserves AI descriptions and web context
- Optimized for natural language processing
- Maintains document structure and relationships

### 📋 Analysis Reports
- Comprehensive processing statistics
- Image type distribution and quality metrics
- Web search effectiveness and source counts
- Processing time and provider information

### 🗜️ Image Exports
- Individual image files in original format
- Accompanying analysis text files
- Organized ZIP archives for easy distribution

## 🎯 Image Analysis Process

### 1. **🔍 Smart Filtering**
- Automatically detects and skips logos, watermarks, decorative elements
- Focuses on scientifically relevant content

### 2. **📊 Classification**
- **Data Visualization**: Charts, graphs, plots with actual data
- **Conceptual**: Flowcharts, diagrams, maps, methodological illustrations

### 3. **📝 Description Generation**
- Comprehensive text descriptions that can replace images
- Includes data patterns, relationships, labels, and context
- Suitable for accessibility and NLP processing

### 4. **🌐 Web Enhancement** (for Conceptual Images)
- **Gemini**: Real-time Google Search integrated in analysis
- **Others**: DuckDuckGo search with AI-generated summaries
- Provides background information and current context

## 🔧 Advanced Features

### 📊 Analytics Dashboard
- Processing history with detailed metrics
- Image type distribution charts
- AI provider usage statistics
- Performance comparison across sessions

### 🔍 Advanced Filtering
- Filter by image type, search status, quality
- Sort by relevance, processing order, or analysis quality
- Custom view options for different use cases

### 📥 Flexible Export Options
- **JSON**: Machine-readable structured data
- **CSV/Excel**: Spreadsheet-compatible formats
- **ZIP**: Complete image and analysis packages
- **Reports**: Human-readable summary documents

### 🎛️ Batch Processing
- Process multiple PDFs simultaneously
- Parallel image analysis for faster processing
- Comprehensive batch reports and comparisons
- Automatic cleanup of intermediate files

## 🛠️ Configuration Options

### 🤖 AI Provider Settings
- **Model Selection**: Choose from latest available models
- **Custom Models**: Enter any model identifier for supported providers
- **Connection Testing**: Verify API connectivity before processing
- **Token Limits**: Configurable maximum response lengths

### 🌐 Web Search Settings
- **Enable/Disable**: Toggle web search functionality
- **Method Selection**: Automatic based on provider capabilities
- **Source Limits**: Control number of search results
- **Summary Generation**: AI-powered result summarization

### 📊 Processing Options
- **Parallel Processing**: Enable multi-threading for faster analysis
- **Intermediate Files**: Option to save or cleanup temporary files
- **Progress Tracking**: Detailed status updates and logging
- **Error Recovery**: Graceful handling of failed analyses

## 🚨 Troubleshooting

### Common Issues

#### ❌ "Model 'xxx' not found"
- **Check model name**: Ensure exact spelling (e.g., `gemini-2.5-flash` not `models/gemini-2.5-flash`)
- **Verify API key**: Confirm valid credentials for the provider
- **Try custom model**: Use "🔧 Custom Model" option with exact identifier

#### 🦆 All images skipped with Gemini
- **Install google-genai**: `pip install google-genai`
- **Check API key**: Verify Google AI API key is valid
- **Model availability**: Confirm model exists and is accessible

#### 🌐 Web search not working
- **For Gemini**: Ensure `google-genai` package is installed
- **For DuckDuckGo**: Check `duckduckgo-search` package
- **Network**: Verify internet connectivity and firewall settings

#### 📱 Images appear blurry
- **Resolution**: Fixed! Images now display at optimal resolution
- **Click enlarge**: Use "🔍 Enlarge" button for full-size viewing
- **Download**: Save original images via download button

## 📚 Dependencies

### Core Requirements
```
streamlit>=1.28.0
docling>=2.0.0
requests>=2.31.0
pandas>=2.0.0
Pillow>=9.5.0
```

### AI Provider SDKs
```
google-genai>=0.2.0      # For Gemini native web search
```

### Optional Features
```
duckduckgo-search>=3.8.0  # For fallback web search
openpyxl>=3.1.0          # For Excel export
xlsxwriter>=3.0.0        # Enhanced Excel features
```

## 🔐 Security & Privacy

- **API Keys**: Stored only in session memory, automatically cleared on exit
- **Local Processing**: LM Studio option provides complete data privacy
- **No Data Retention**: No user data is stored permanently by the application
- **Secure Communication**: All API calls use HTTPS encryption

## 🎯 Use Cases

### 📚 Academic Research
- Extract and analyze scientific figures and diagrams
- Generate text descriptions for accessibility compliance
- Create searchable databases of research imagery
- Enhance literature reviews with visual content analysis

### 📊 Business Intelligence
- Process reports and presentations for key visual insights
- Extract data visualizations and convert to structured formats
- Analyze competitor documents and market research
- Create summaries of complex technical documentation

### 🔬 Scientific Publishing
- Prepare manuscripts with enhanced figure descriptions
- Verify figure quality and scientific accuracy
- Generate alt-text for accessibility requirements
- Create figure databases for institutional repositories

### 🏛️ Document Management
- Digitize and analyze large document collections
- Extract visual information for search and indexing
- Create metadata-rich document archives
- Support compliance and audit requirements

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is provided as-is for educational and research purposes.

---

**PDF Image Analyzer Pro** - Transforming document analysis with AI-powered image understanding and web-enhanced insights. 🚀

## 📋 HTML Evaluation Reports

### 🎯 Purpose
The HTML evaluation reports are specifically designed to help you assess the effectiveness of PDF text extraction and image analysis. These reports provide a comprehensive view that makes it easy to compare the AI processing results against the original PDF.

### 📊 Report Types

#### **1. Image Analysis Report** (`filename_step2_image_analysis_report.html`)
- **📸 Visual Overview**: All analyzed images displayed with their AI descriptions
- **📈 Statistics Dashboard**: Summary of image types, web search usage, and processing metrics
- **🔍 Interactive Images**: Click to enlarge any image for detailed inspection
- **🌐 Web Search Results**: Shows both native (Gemini) and DuckDuckGo search enhancements
- **📝 Evaluation Focus**: Perfect for reviewing AI image recognition accuracy

#### **2. Complete Evaluation Report** (`filename_complete_evaluation_report.html`)
- **📄 Text Extraction Tab**: Full text content extracted from the PDF
- **🖼️ Image Analysis Tab**: Detailed image recognition results
- **⚖️ Comparison Tab**: Side-by-side evaluation tools with checklists
- **✅ Interactive Checklists**: Track evaluation progress for both text and images
- **📝 Assessment Notes**: Built-in text area for recording evaluation findings

### 🚀 How to Generate Reports

1. **Process your PDF file** using any AI provider
2. **Navigate to Results tab** after processing completes
3. **Click "📊 Generate HTML Reports"** in the Generated Files section
4. **Wait for generation** (usually takes 5-10 seconds)
5. **Use the "🌐 Open" buttons** to view reports in your browser

### 💡 Evaluation Workflow

#### **For Text Extraction Quality:**
1. Open the **Complete Evaluation Report**
2. Switch to the **📄 Text Extraction** tab
3. Compare against your original PDF:
   - ✅ Are all paragraphs captured?
   - ✅ Is formatting preserved?
   - ✅ Are tables and lists accurate?
   - ✅ Are formulas and citations intact?

#### **For Image Analysis Quality:**
1. Review the **Image Analysis Report** for detailed results
2. In the **Complete Evaluation Report**, check the **🖼️ Image Analysis** tab:
   - ✅ Are images correctly classified?
   - ✅ Are descriptions accurate and detailed?
   - ✅ Are logos/decorative elements filtered?
   - ✅ Does web search enhance understanding?

#### **For Overall Assessment:**
1. Use the **⚖️ Comparison** tab in the Complete Evaluation Report
2. Fill out the interactive checklists
3. Record your findings in the assessment notes area
4. Identify areas for improvement

### 🎨 Report Features

#### **Interactive Elements:**
- **🔍 Click-to-enlarge**: All images can be clicked for full-size viewing
- **📑 Tabbed Navigation**: Easy switching between different report sections
- **✅ Evaluation Checklists**: Interactive checkboxes to track your assessment
- **📝 Notes Area**: Built-in text area for recording evaluation findings

#### **Professional Styling:**
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🎨 Clean Layout**: Professional appearance suitable for sharing
- **📊 Visual Statistics**: Charts and metrics for quick assessment
- **🔗 Clickable Links**: Direct links to web sources for verification

### 📁 File Naming Convention

All files now follow a consistent naming pattern based on your original PDF:

```
original_document.pdf
├── original_document_step1_docling.json          # Raw Docling extraction
├── original_document_step2_enhanced.json         # With AI image analysis
├── original_document_step3_nlp_ready.json        # Image data removed
├── original_document_step2_image_analysis_report.html
└── original_document_complete_evaluation_report.html
```

### 🔧 Technical Details

- **📱 Self-Contained**: HTML files include all images and styles (no external dependencies)
- **🖼️ Embedded Images**: Images are base64-encoded directly in the HTML
- **🌐 Browser Compatible**: Works in all modern browsers
- **📄 Printable**: Reports can be printed or saved as PDF from browser
- **🔗 Portable**: Can be shared via email or file sharing services

## 🆕 Batch Processing Features

### 📚 Multiple File Support
- **Upload Multiple Files**: Support for both PDF and JSON files in one batch
- **Progress Tracking**: Real-time progress monitoring for each file
- **Error Resilience**: Continue processing even if individual files fail
- **Organized Output**: Each file gets its own subdirectory

### 📊 Batch Results Management
- **Individual Tabs**: Each processed file gets its own results tab
- **Batch Statistics**: Summary metrics across all processed files
- **Batch Actions**: Generate reports, summaries, and archives for all files
- **Error Reporting**: Detailed error logs for failed files

### 🗂️ Batch Organization Structure
```
batch_output/
├── document1_name/
│   ├── document1_name_step1_docling.json
│   ├── document1_name_step2_enhanced.json
│   ├── document1_name_step3_nlp_ready.json
│   └── document1_name_reports...
├── document2_name/
│   └── ...
└── batch_summary_YYYYMMDD_HHMMSS.txt
```

### 🔧 Batch Actions
- **📊 Generate All HTML Reports**: Create evaluation reports for all processed files
- **📥 Download Batch Summary**: Get a text summary of all processing results
- **🗜️ Create Batch ZIP**: Download all results in a single ZIP archive
- **📁 Open Individual Folders**: Direct access to each file's output directory

---

**Made with** ❤️ **and Streamlit**