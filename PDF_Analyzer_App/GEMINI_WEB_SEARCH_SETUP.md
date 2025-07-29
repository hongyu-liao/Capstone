# 🔮 Gemini Web Search Setup Guide

This guide will help you set up native Google Search integration for Gemini models.

## 📦 Installation

### 1. Install Required Package
```bash
pip install google-genai>=0.2.0
```

### 2. Verify Installation
```python
# Test in Python
try:
    from google import genai
    from google.genai import types
    print("✅ google-genai installed successfully")
except ImportError as e:
    print(f"❌ Installation failed: {e}")
```

## 🔑 API Key Setup

### 1. Get Google AI API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### 2. Configure in Application
1. Select "Google (Gemini)" as AI provider
2. Choose a Gemini model (e.g., `gemini-2.5-flash`)
3. Enter your API key
4. Click "🔍 Test Gemini Web Search" to verify

## 🌐 Web Search Features

### What You Get
- **🔍 Real-time Search**: Live Google Search integrated into AI analysis
- **📚 Source Attribution**: Direct links to information sources
- **🎯 Smart Queries**: AI automatically determines what to search for
- **📑 Text Mapping**: See which parts of analysis came from which sources

### How It Works
```python
# Example of what happens internally:
response = gemini.analyze_with_web_search(
    image=scientific_diagram,
    prompt="Analyze this climate change visualization",
    tools=[GoogleSearch()]
)

# Result includes:
# - AI analysis enhanced with current web information
# - List of sources used
# - Mapping of text segments to sources
```

## 🔧 Troubleshooting

### ❌ "google-genai not available"
**Solution:**
```bash
pip install --upgrade google-genai
```

### ❌ "Gemini client failed to initialize"
**Possible causes:**
1. **Invalid API Key**: Check your Google AI API key
2. **Network Issues**: Verify internet connection
3. **Model Access**: Ensure your API key has access to the selected model

**Solutions:**
```bash
# Try reinstalling
pip uninstall google-genai
pip install google-genai>=0.2.0

# Test API key
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
```

### ⚠️ "All images skipped"
**Common causes:**
1. **Model name format**: Use `gemini-2.5-flash`, not `models/gemini-2.5-flash`
2. **API permissions**: Ensure multimodal access is enabled
3. **Package version**: Update to latest google-genai version

### 🔄 "Fallback to regular analysis"
This is normal behavior when:
- Web search is not needed for the specific image
- Network issues prevent web search
- API rate limits are reached

## 📊 Expected Behavior

### ✅ Working Correctly
```
🌐 AI Native Web Search Results:
🔗 This analysis includes real-time web search results directly integrated by the AI model.
Search Queries: climate change impacts, sea level rise data
Sources Used: 3 web sources integrated into analysis
```

### ⚠️ Fallback Mode
```
⚠️ Native web search failed, analysis completed without web enhancement
```

### 🦆 DuckDuckGo Fallback
```
🦆 DuckDuckGo Web Search Context:
Keywords: climate change, carbon cycle
Summary: [AI-generated summary of search results]
```

## 🎯 Best Practices

### 1. **Model Selection**
- Use latest models: `gemini-2.5-flash`, `gemini-2.5-pro`
- Avoid older models that may have limited web search capabilities

### 2. **Image Types**
- Web search works best with **conceptual images**:
  - Maps and geographic visualizations
  - Process diagrams and flowcharts
  - Methodological illustrations
  - Scientific framework diagrams

### 3. **Network Considerations**
- Ensure stable internet connection
- Web search adds ~2-5 seconds to analysis time
- Fallback to regular analysis if search fails

## 🔐 Security Notes

- API keys are stored only in session memory
- No search data is retained by the application
- All communication uses HTTPS encryption
- Google's privacy policy applies to web search results

## 📞 Support

If you continue to experience issues:

1. **Check the diagnostic**: Use "🔍 Test Gemini Web Search" button
2. **Verify installation**: Ensure `google-genai>=0.2.0` is installed
3. **Test API key**: Confirm access to Gemini models
4. **Check logs**: Look for detailed error messages in the terminal

---

**🚀 Ready to experience AI-powered web-enhanced image analysis!**