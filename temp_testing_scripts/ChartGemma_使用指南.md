# ChartGemma Academic Chart Analysis Guide

## 📖 About ChartGemma

[ChartGemma](https://huggingface.co/ahmed-masry/chartgemma) is a state-of-the-art visual language model developed by Ahmed Masry et al., specifically designed for chart understanding and reasoning. Built on the PaliGemma architecture, it excels at:

- 📊 Understanding and describing various types of charts
- 🤔 Answering questions about chart content with academic precision
- 📈 Analyzing trends and patterns in data visualizations
- 💬 Engaging in "conversational" interaction with charts
- 🎯 **NEW**: Automatic chart type detection and specialized academic analysis

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r chartgemma_requirements.txt
```

### 2. Running Options

#### Option 1: Simple Single Chart Test
```bash
python run_chartgemma_simple.py "Sample Line Chart/3.png"
```

Or with a custom question:
```bash
python run_chartgemma_simple.py "Sample Line Chart/3.png" "What are the maximum and minimum values for each line series?"
```

#### Option 2: Full Academic Batch Analysis (RECOMMENDED)
```bash
python test_chartgemma.py
```

This will analyze ALL images in the `Sample Line Chart` folder with:
- ✅ **Automatic chart type detection**
- ✅ **Specialized question sets** for each chart type
- ✅ **Academic-focused analysis**
- ✅ **Professional HTML reports**

## 📁 Output Files

After running, the following files will be generated in the `chartgemma_results` folder:

1. **JSON Results**: `chartgemma_analysis_results_YYYYMMDD_HHMMSS.json`
   - Structured data for all Q&A pairs
   - Processing time statistics
   - Chart type detection results
   - Raw response texts

2. **HTML Report**: `chartgemma_academic_report_YYYYMMDD_HHMMSS.html`
   - Professional visual analysis report
   - Chart type badges and specialized questions
   - Processing statistics
   - Academic-style formatting

## 🎯 Specialized Question Sets by Chart Type

The script automatically detects chart types and applies specialized question sets:

### 📈 **Line Charts**
- Identify each line/trend and describe patterns over time
- Maximum and minimum values for each series
- Overall trend direction analysis
- Notable intersections, peaks, or valleys
- Time period coverage
- Rate of change calculations

### 📊 **Bar Charts**
- List all categories and corresponding values
- Highest and lowest value identification
- Ratio calculations between values
- Arrangement pattern analysis
- Total sum calculations
- Significant differences between categories

### 🥧 **Pie Charts**
- Segment percentages and proportions
- Largest and smallest portion identification
- Approximately equal segments
- Color/pattern meaning
- Combined percentage of top 3 segments

### 📍 **Scatter Plots**
- Correlation pattern analysis
- Outlier identification
- Trend direction assessment
- Cluster identification
- Axis range analysis
- Linear/non-linear relationship evidence

### 📊 **Histograms**
- Distribution shape description
- Value range analysis
- Highest frequency bin identification
- Gap and pattern analysis
- Central tendency inference
- Total observation estimation

### 📦 **Box Plots**
- Median and quartile identification
- Interquartile range analysis
- Distribution comparison
- Outlier identification
- Skewness description
- Data variability insights

## 💻 System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ recommended
- **GPU**: Optional, CUDA-enabled GPU will significantly accelerate processing
- **Network**: First run requires downloading ~3GB model files
- **Storage**: ~5GB free space for model cache

## ⚠️ Important Notes

1. **First Run**: Requires downloading model files from Hugging Face (may take several minutes)
2. **GPU Recommended**: While CPU works, GPU dramatically improves processing speed
3. **Memory Requirements**: Large model, ensure sufficient system memory
4. **Network Connection**: Good internet connection needed for initial model download

## 🔧 Troubleshooting

### Model Download Failed
```bash
# Set Hugging Face mirror (if in China)
export HF_ENDPOINT=https://hf-mirror.com
```

### Out of Memory
```python
# In script, change torch_dtype to more memory-efficient option
torch_dtype=torch.float32  # instead of float16
```

### CUDA Errors
```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

## 🆕 New Features in Academic Version

### ✨ **Smart Chart Type Detection**
The script automatically identifies chart types using ChartGemma's visual understanding capabilities.

### 🎯 **Specialized Academic Question Sets**
Each chart type gets tailored questions relevant to academic analysis:
- **Line Charts**: Focus on temporal trends, rates of change, and comparative analysis
- **Bar Charts**: Emphasis on categorical comparisons and quantitative relationships  
- **Pie Charts**: Proportion analysis and percentage calculations
- **And more**: Scatter plots, histograms, box plots all get specialized treatment

### 📊 **Enhanced Reporting**
- Professional HTML reports with chart type badges
- Processing statistics and performance metrics
- Academic-style formatting suitable for research purposes

## 📚 More Information

- 📄 **Paper**: [ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild](https://arxiv.org/abs/2407.04172)
- 🤗 **Model Page**: https://huggingface.co/ahmed-masry/chartgemma
- 🎮 **Online Demo**: [ChartGemma Web Demo](https://huggingface.co/spaces/ahmed-masry/ChartGemma)

## 📝 Example Usage Code

```python
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Load model
model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma")
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

# Analyze chart
image = Image.open("your_chart.png")
question = "What are the main trends and values in this chart?"
inputs = processor(text=question, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## 🎯 Use Cases for Academic Research

- 📊 **Business Report Analysis**: Automated extraction of insights from financial charts
- 📈 **Scientific Data Visualization**: Understanding complex research data visualizations  
- 📋 **Automated Report Generation**: Converting charts to structured text descriptions
- 🤖 **Intelligent Data Analysis Assistant**: Interactive chart exploration and analysis
- 📚 **Educational Material Creation**: Generating explanations for instructional charts
- 🔬 **Literature Review Automation**: Extracting data from paper figures
- 📖 **Accessibility Enhancement**: Converting visual data to text for screen readers

## 🏆 Why This Academic Version?

Traditional chart analysis tools focus on basic description. Our enhanced version provides:

✅ **Intelligent Chart Type Recognition**  
✅ **Academic-Quality Question Sets**  
✅ **Specialized Analysis for Research Contexts**  
✅ **Professional Reporting Format**  
✅ **Batch Processing for Multiple Charts**  
✅ **Performance Metrics and Statistics**

Perfect for researchers, academics, and professionals who need detailed, systematic chart analysis!
