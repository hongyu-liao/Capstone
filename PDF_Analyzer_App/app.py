import streamlit as st
import logging
import os
import json
import base64
import zipfile
import io
import time
import tempfile
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import custom modules
from pdf_processor import PDFProcessor
from image_analyzer import ImageAnalyzer
from config import Config
from api_manager import APIManager

# Page configuration
st.set_page_config(
    page_title="PDF Image Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .error {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .info {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .image-analysis {
        border: 1px solid #ddd;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
    .stTextInput > div > div > input[type="password"] {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []

def configure_sidebar():
    """Configure sidebar with AI provider settings"""
    st.sidebar.header("🤖 AI Configuration")
    
    # AI Provider selection
    ai_provider = st.sidebar.selectbox(
        "AI Provider",
        ["LM Studio (Local)", "OpenAI (ChatGPT)", "Google (Gemini)", "Anthropic (Claude)"],
        key="ai_provider"
    )
    
    # Model selection based on provider
    if ai_provider == "LM Studio (Local)":
        st.sidebar.info("🏠 Using local LM Studio server")
        lm_studio_url = st.sidebar.text_input(
            "LM Studio URL",
            value="http://localhost:1234/v1/chat/completions",
            key="lm_studio_url"
        )
        model_name = st.sidebar.text_input(
            "Model Name",
            value="",
            placeholder="Enter model name from LM Studio",
            key="model_name"
        )
        # No API key needed for local LM Studio
        # Create hidden API key field to maintain session state consistency
        api_key = st.sidebar.text_input("API Key", value="", type="password", 
                                       key="api_key", disabled=True, 
                                       help="Not required for local LM Studio")
        
    elif ai_provider == "OpenAI (ChatGPT)":
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "🔧 Custom Model"]
        selected_model = st.sidebar.selectbox("Model", models)
        
        if selected_model == "🔧 Custom Model":
            model_name = st.sidebar.text_input("Custom Model Name", key="model_name")
        else:
            model_name = selected_model
        
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key")
        
    elif ai_provider == "Google (Gemini)":
        models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", 
                 "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "🔧 Custom Model"]
        selected_model = st.sidebar.selectbox("Model", models)
        
        if selected_model == "🔧 Custom Model":
            model_name = st.sidebar.text_input("Custom Model Name", key="model_name")
        else:
            model_name = selected_model
        
        api_key = st.sidebar.text_input("Google AI API Key", type="password", key="api_key")
        
    elif ai_provider == "Anthropic (Claude)":
        models = ["claude-4-sonnet-latest", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-latest",
                 "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "🔧 Custom Model"]
        selected_model = st.sidebar.selectbox("Model", models)
        
        if selected_model == "🔧 Custom Model":
            model_name = st.sidebar.text_input("Custom Model Name", key="model_name")
        else:
            model_name = selected_model
        
        api_key = st.sidebar.text_input("Anthropic API Key", type="password", key="api_key")
    
    # Processing options
    st.sidebar.subheader("🔧 Processing Options")
    enable_web_search = st.sidebar.checkbox(
        "Enable Web Search",
        value=True,
        help="Perform web search for conceptual images (native or DuckDuckGo based on provider)",
        key="enable_web_search"
    )
    
    generate_nlp_ready = st.sidebar.checkbox(
        "Generate NLP-Ready JSON",
        value=False,
        help="Create a version without image data for NLP processing",
        key="generate_nlp_ready"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=200,
        max_value=2000,
        value=700,
        step=50,
        help="Maximum tokens for AI responses",
        key="max_tokens"
    )

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">📄 PDF Image Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Add tabs for different modes
    tab1, tab2, tab3 = st.tabs(["🔄 Process Files", "📊 Results Dashboard", "📈 Analytics"])
    
    return tab1, tab2, tab3

def display_sidebar():
    """Display the sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # AI Provider Selection
    st.sidebar.subheader("🤖 AI Provider")
    ai_provider = st.sidebar.selectbox(
        "Choose AI Provider",
        ["LM Studio (Local)", "OpenAI (ChatGPT)", "Google (Gemini)", "Anthropic (Claude)"],
        help="Select your preferred AI service provider"
    )
    
    # Provider-specific configuration
    if ai_provider == "LM Studio (Local)":
        lm_studio_url = st.sidebar.text_input(
            "LM Studio URL", 
            value="http://localhost:1234/v1/chat/completions",
            help="URL of your LM Studio server"
        )
        
        model_name = st.sidebar.text_input(
            "Model Name", 
            value="google/gemma-3-12b-it-gguf",
            help="Name of the model loaded in LM Studio"
        )
        
        api_key = None
        
    else:
        # External API configuration
        lm_studio_url = None
        
        if ai_provider == "OpenAI (ChatGPT)":
            # Latest OpenAI models
            predefined_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "🔧 Custom Model"
            ]
            
            selected_model = st.sidebar.selectbox(
                "Model", 
                predefined_models,
                help="Choose OpenAI model or select Custom to enter your own"
            )
            
            if selected_model == "🔧 Custom Model":
                model_name = st.sidebar.text_input(
                    "Custom Model Name",
                    placeholder="e.g., gpt-4o-2024-08-06",
                    help="Enter exact model identifier from OpenAI"
                )
                if not model_name:
                    st.sidebar.warning("⚠️ Please enter a custom model name")
            else:
                model_name = selected_model
                
            api_url = "https://api.openai.com/v1/chat/completions"
            
        elif ai_provider == "Google (Gemini)":
            # Latest Gemini models
            predefined_models = [
                "gemini-2.5-pro",
                "gemini-2.5-flash", 
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest",
                "🔧 Custom Model"
            ]
            
            selected_model = st.sidebar.selectbox(
                "Model", 
                predefined_models,
                help="Choose Gemini model or select Custom to enter your own"
            )
            
            if selected_model == "🔧 Custom Model":
                model_name = st.sidebar.text_input(
                    "Custom Model Name",
                    placeholder="e.g., gemini-2.0-flash-thinking-exp",
                    help="Enter exact model identifier from Google AI"
                )
                if not model_name:
                    st.sidebar.warning("⚠️ Please enter a custom model name")
            else:
                model_name = selected_model
                
            api_url = "https://generativelanguage.googleapis.com/v1beta/models"
            
        elif ai_provider == "Anthropic (Claude)":
            # Latest Claude models
            predefined_models = [
                "claude-4-sonnet-latest",
                "claude-3-5-sonnet-20241022",
                "claude-3-7-sonnet-latest",
                "claude-3-5-haiku-20241022", 
                "claude-3-opus-20240229",
                "🔧 Custom Model"
            ]
            
            selected_model = st.sidebar.selectbox(
                "Model", 
                predefined_models,
                help="Choose Claude model or select Custom to enter your own"
            )
            
            if selected_model == "🔧 Custom Model":
                model_name = st.sidebar.text_input(
                    "Custom Model Name",
                    placeholder="e.g., claude-4-sonnet-latest",
                    help="Enter exact model identifier from Anthropic"
                )
                if not model_name:
                    st.sidebar.warning("⚠️ Please enter a custom model name")
            else:
                model_name = selected_model
                
            api_url = "https://api.anthropic.com/v1/messages"
        
        # API Key input
        api_key_placeholder = f"Enter your {ai_provider.split('(')[0].strip()} API key"
        api_key = st.sidebar.text_input(
            "🔑 API Key", 
            type="password",
            placeholder=api_key_placeholder,
            help="Your API key will not be stored after the session ends"
        )
        
        if api_key:
            st.session_state.api_keys[ai_provider] = api_key
        
        # Model information
        if model_name and model_name != "🔧 Custom Model":
            with st.sidebar.expander("ℹ️ Model Information"):
                model_info = get_model_information(ai_provider, model_name)
                
                st.caption(f"**Provider:** {ai_provider}")
                st.caption(f"**Model:** {model_name}")
                
                if "description" in model_info:
                    st.caption(f"**Description:** {model_info['description']}")
                
                if "context_window" in model_info:
                    st.caption(f"**Context Window:** {model_info['context_window']:,} tokens")
                
                # Note about costs
                st.caption("💰 **Costs:** Pricing varies by provider and usage. Please check provider's website for current rates.")
        
        # Web search capabilities
        st.sidebar.subheader("🌐 Web Search Features")
        if ai_provider == "LM Studio (Local)":
            st.sidebar.info("🦆 **DuckDuckGo Search:** Available for conceptual images")
            st.sidebar.caption("Local models use external DuckDuckGo search with AI summarization")
        elif ai_provider == "Google (Gemini)":
            st.sidebar.success("🌐 **Native Google Search:** Real-time web integration")
            st.sidebar.caption("Gemini models include live web search directly in analysis")
            
            # Gemini diagnostic tool
            if st.sidebar.button("🔍 Test Gemini Web Search", help="Quick test of Gemini's web search capability"):
                if api_key:
                    with st.sidebar.spinner("Testing Gemini web search..."):
                        try:
                            # Test Gemini web search with a simple query
                            from api_manager import APIManager
                            config = {
                                'ai_provider': 'Google (Gemini)',
                                'model_name': model_name,
                                'api_key': api_key,
                                'lm_studio_url': 'https://generativelanguage.googleapis.com/v1beta/models'
                            }
                            api_manager = APIManager(config)
                            
                            # Simple test without image
                            if hasattr(api_manager, 'genai_client') and api_manager.genai_client:
                                st.sidebar.success("✅ Gemini client initialized successfully")
                                st.sidebar.info("🔗 Native web search should work for image analysis")
                            else:
                                st.sidebar.error("❌ Gemini client failed to initialize")
                                st.sidebar.warning("Install: pip install google-genai")
                                
                        except Exception as e:
                            st.sidebar.error(f"❌ Test failed: {str(e)}")
                            st.sidebar.info("💡 Check your API key and model name")
                else:
                    st.sidebar.warning("⚠️ Please enter API key first")
                    
        elif ai_provider == "OpenAI (ChatGPT)":
            if model_name in ["gpt-4o", "gpt-4-turbo"]:
                st.sidebar.success("🌐 **Native Web Browsing:** Available (when enabled)")
            else:
                st.sidebar.info("🦆 **DuckDuckGo Search:** Fallback search available")
            st.sidebar.caption("Some ChatGPT models support web browsing capabilities")
        elif ai_provider == "Anthropic (Claude)":
            st.sidebar.info("🦆 **DuckDuckGo Search:** Available for conceptual images")
            st.sidebar.caption("Claude uses external DuckDuckGo search with AI summarization")
        else:
            st.sidebar.warning("❓ **Web Search:** Status unknown")
    
    # Processing options
    st.sidebar.subheader("🔧 Processing Options")
    enable_web_search = st.sidebar.checkbox(
        "Enable Web Search", 
        value=True,
        help="Perform web search for conceptual images (native or DuckDuckGo based on provider)"
    )
    
    generate_nlp_ready = st.sidebar.checkbox(
        "Generate NLP-Ready JSON", 
        value=False,
        help="Remove images and create NLP-optimized version"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens for AI Analysis", 
        min_value=200, 
        max_value=1000, 
        value=600,
        help="Maximum tokens for AI image analysis"
    )
    
    # Advanced options
    with st.sidebar.expander("🎛️ Advanced Options"):
        save_intermediate = st.checkbox("Save Intermediate Results", value=True, help="Keep intermediate processing files")
        parallel_processing = st.checkbox("Parallel Image Analysis", help="Analyze multiple images simultaneously")
        
    return {
        'ai_provider': ai_provider,
        'lm_studio_url': lm_studio_url if ai_provider == "LM Studio (Local)" else api_url,
        'model_name': model_name,
        'api_key': api_key,
        'enable_web_search': enable_web_search,
        'generate_nlp_ready': generate_nlp_ready,
        'max_tokens': max_tokens,
        'save_intermediate': save_intermediate,
        'parallel_processing': parallel_processing
    }

def get_model_information(provider: str, model_name: str) -> dict:
    """Get information about a specific model"""
    
    model_database = {
        "OpenAI (ChatGPT)": {
            "gpt-4o": {
                "description": "Latest GPT-4 Omni model with vision capabilities",
                "context_window": 128000
            },
            "gpt-4o-mini": {
                "description": "Smaller, faster GPT-4 Omni model with vision",
                "context_window": 128000
            },
            "gpt-4-turbo": {
                "description": "High-performance GPT-4 with larger context window",
                "context_window": 128000
            },
            "gpt-4": {
                "description": "Original GPT-4 model",
                "context_window": 8192
            },
            "gpt-3.5-turbo": {
                "description": "Fast and efficient model for most tasks",
                "context_window": 16385
            }
        },
        "Google (Gemini)": {
            "gemini-2.5-pro": {
                "description": "Latest Gemini 2.5 Pro with enhanced multimodal capabilities",
                "context_window": 2000000
            },
            "gemini-2.5-flash": {
                "description": "Gemini 2.5 Flash - Optimized for speed and efficiency",
                "context_window": 1000000
            },
            "gemini-2.5-flash-lite": {
                "description": "Lightweight Gemini 2.5 Flash for simple tasks",
                "context_window": 1000000
            },
            "gemini-2.0-flash": {
                "description": "Gemini 2.0 Flash with improved vision capabilities",
                "context_window": 1000000
            },
            "gemini-1.5-pro-latest": {
                "description": "Most capable Gemini model with 2M token context",
                "context_window": 2000000
            },
            "gemini-1.5-flash-latest": {
                "description": "Fast Gemini model optimized for speed",
                "context_window": 1000000
            }
        },
        "Anthropic (Claude)": {
            "claude-4-sonnet-latest": {
                "description": "Next-generation Claude 4 Sonnet with advanced reasoning",
                "context_window": 500000
            },
            "claude-3-5-sonnet-20241022": {
                "description": "Latest Claude 3.5 Sonnet with enhanced capabilities",
                "context_window": 200000
            },
            "claude-3-7-sonnet-latest": {
                "description": "Claude 3.7 Sonnet - Advanced reasoning and analysis",
                "context_window": 200000
            },
            "claude-3-5-haiku-20241022": {
                "description": "Fast and efficient Claude 3.5 model",
                "context_window": 200000
            },
            "claude-3-opus-20240229": {
                "description": "Most powerful Claude 3 model",
                "context_window": 200000
            }
        }
    }
    
    return model_database.get(provider, {}).get(model_name, {})

def upload_files_section():
    """Handle file upload section"""
    st.subheader("📁 File Upload")
    
    # File type selection
    file_type_option = st.radio("Select file type:", ["PDF", "JSON"])
    
    if file_type_option == "PDF":
        file_type = "pdf"
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process. You can upload a single file or multiple files for batch processing."
        )
    else:  # JSON
        file_type = "json"
        uploaded_files = st.file_uploader(
            "Choose JSON files",
            type=['json'],
            accept_multiple_files=True,
            help="Upload one or more JSON files from previous Docling conversion"
        )
    
    if uploaded_files:
        # Handle both single and multiple files uniformly
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        # Create temporary files
        file_list = []
        total_size = 0
        
        for uploaded_file in uploaded_files:
            if file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
            else:  # JSON
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w+b') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
            
            file_list.append({
                'name': uploaded_file.name,
                'path': temp_path,
                'size': uploaded_file.size
            })
            total_size += uploaded_file.size
        
        # Display upload summary
        if len(file_list) == 1:
            st.success(f"✅ File uploaded: {file_list[0]['name']}")
            st.info(f"📊 File size: {file_list[0]['size'] / 1024 / 1024:.2f} MB")
        else:
            st.success(f"✅ {len(file_list)} files uploaded successfully!")
            st.info(f"📊 Total size: {total_size / 1024 / 1024:.2f} MB")
            
            # Show file list
            with st.expander("📋 View uploaded files"):
                for i, file_info in enumerate(file_list, 1):
                    st.write(f"{i}. **{file_info['name']}** ({file_info['size'] / 1024 / 1024:.2f} MB)")
        
        return file_type, file_list
    
    return None, None



def output_settings_section():
    """Configure output settings"""
    st.subheader("📁 Output Settings")
    
    # Base output directory
    output_dir = st.text_input(
        "Output Directory",
        value="./output",
        help="Directory where processed files will be saved"
    )
    
    # Create directory button
    if st.button("📁 Create Directory"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            st.success(f"✅ Directory created: {output_dir}")
        except Exception as e:
            st.error(f"❌ Failed to create directory: {e}")
    
    st.info("💡 Files will use simplified naming: filename.json, filename_report.html, etc.")
    return output_dir



def process_batch_files(file_type, file_list, base_output_dir, config):
    """Process files with unified naming scheme"""
    try:
        if len(file_list) == 1:
            st.header("🔄 Processing File")
        else:
            st.header(f"🔄 Processing {len(file_list)} Files")
        
        # Validate API configuration
        if config['ai_provider'] != "LM Studio (Local)" and not config.get('api_key'):
            st.warning(f"⚠️ Please provide an API key for {config['ai_provider']} in the sidebar")
            return
        
        # Test connection for external APIs
        if config['ai_provider'] != "LM Studio (Local)":
            with st.spinner("Testing API connection..."):
                try:
                    from api_manager import APIManager
                    api_manager = APIManager(config)
                    success, message = api_manager.test_connection()
                    if not success:
                        st.error(f"❌ API Connection Failed: {message}")
                        st.info("💡 Please check your model name and API key, then try again.")
                        return
                    else:
                        st.success(message)
                except Exception as e:
                    st.error(f"❌ API Connection Error: {str(e)}")
                    st.info("💡 Please check your model name and API key, then try again.")
                    return
        
        # Initialize results storage
        batch_results = []
        total_files = len(file_list)
        
        # Create base output directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Overall progress
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        # Process each file
        for i, file_info in enumerate(file_list):
            file_path = file_info['path']
            file_name = file_info['name']
            
            overall_status.text(f"Processing file {i+1}/{total_files}: {file_name}")
            
            # Create expandable section for this file's processing
            with st.expander(f"📄 Processing: {file_name}", expanded=True):
                try:
                    result = process_single_file_batch(
                        file_type, file_path, file_name, base_output_dir, config
                    )
                    
                    if result:
                        result['file_index'] = i + 1
                        batch_results.append(result)
                        st.success(f"✅ Completed: {file_name}")
                    else:
                        st.error(f"❌ Failed: {file_name}")
                        batch_results.append({
                            'file_name': file_name,
                            'file_index': i + 1,
                            'success': False,
                            'error': 'Processing failed'
                        })
                
                except Exception as e:
                    st.error(f"❌ Error processing {file_name}: {str(e)}")
                    batch_results.append({
                        'file_name': file_name,
                        'file_index': i + 1,
                        'success': False,
                        'error': str(e)
                    })
            
            # Update overall progress
            overall_progress.progress((i + 1) / total_files)
        
        # Store results in session state
        st.session_state.batch_results = batch_results
        st.session_state.batch_processing_complete = True
        st.session_state.batch_base_output_dir = base_output_dir
        
        # Summary
        successful_files = sum(1 for r in batch_results if r.get('success', False))
        overall_status.text("✅ Batch processing complete!")
        
        st.success(f"🎉 Batch Processing Complete! {successful_files}/{total_files} files processed successfully.")
        
        # Cleanup temporary files
        for file_info in file_list:
            try:
                os.unlink(file_info['path'])
            except:
                pass
        
    except Exception as e:
        st.error(f"❌ Batch processing failed: {str(e)}")
        logging.error(f"Batch processing error: {e}", exc_info=True)

def process_single_file_batch(file_type, file_path, file_name, output_dir, config):
    """Process a single file as part of batch processing"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get base filename (without extension)
        original_name = Path(file_name).stem
        
        # Initialize processors
        if file_type == "pdf":
            pdf_processor = PDFProcessor(config)
            image_analyzer = ImageAnalyzer(config)
            
            # Step 1: Convert PDF to JSON
            status_text.text("Step 1/4: Converting PDF to JSON...")
            progress_bar.progress(25)
            
            # Use original filename for JSON output (without _step1_docling suffix)
            json_result = pdf_processor.convert_pdf_to_json(file_path, output_dir, original_name)
            
            if not json_result:
                return None
            
            # Rename the generated JSON file to match original PDF name exactly
            generated_json_path = json_result['json_path']
            target_json_path = os.path.join(output_dir, f"{original_name}.json")
            
            # Move the file if names are different
            if generated_json_path != target_json_path:
                import shutil
                shutil.move(generated_json_path, target_json_path)
            
            json_path = target_json_path
            base_filename = original_name
            
        else:  # JSON file
            json_path = file_path
            image_analyzer = ImageAnalyzer(config)
            
            # Use original filename (without extension) as base filename
            # This preserves the complete original name without stripping suffixes
            base_filename = Path(file_name).stem
            
            # Copy JSON to output directory if it's not already there
            # Use original filename for the copy to avoid conflicts with enhanced versions
            target_json_path = os.path.join(output_dir, file_name)
            if json_path != target_json_path:
                import shutil
                shutil.copy2(json_path, target_json_path)
                json_path = target_json_path
            
            status_text.text("Step 1/4: Loading JSON file...")
            progress_bar.progress(25)
        
        # Step 2: Extract and analyze images
        status_text.text("Step 2/4: Extracting and analyzing images...")
        progress_bar.progress(50)
        
        analysis_results = image_analyzer.analyze_images_from_json(
            json_path, 
            config['enable_web_search']
        )
        
        if not analysis_results:
            status_text.text("⚠️ No informative images found")
            progress_bar.progress(100)
            return {
                'success': True,
                'file_name': file_name,
                'file_type': file_type,
                'base_filename': base_filename,
                'original_json': json_path,
                'enhanced_json': None,
                'nlp_ready_json': None,
                'analysis_results': [],
                'output_dir': output_dir,
                'ai_provider': config['ai_provider'],
                'model_name': config['model_name'],
                'timestamp': datetime.now(),
                'message': 'No informative images found'
            }
        
        # Step 3: Generate enhanced JSON
        status_text.text("Step 3/4: Generating enhanced JSON...")
        progress_bar.progress(75)
        
        enhanced_json_path = image_analyzer.create_enhanced_json(
            json_path, 
            analysis_results, 
            output_dir, 
            base_filename,
            batch_mode=True  # Use simplified naming for batch processing
        )
        
        # Step 4: Optional NLP-ready version
        nlp_ready_path = None
        if config['generate_nlp_ready']:
            status_text.text("Step 4/4: Creating NLP-ready version...")
            progress_bar.progress(90)
            
            nlp_ready_path = image_analyzer.create_nlp_ready_version(
                enhanced_json_path, 
                output_dir, 
                base_filename,
                batch_mode=True  # Use simplified naming for batch processing
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        return {
            'success': True,
            'file_name': file_name,
            'file_type': file_type,
            'base_filename': base_filename,
            'original_json': json_path,
            'enhanced_json': enhanced_json_path,
            'nlp_ready_json': nlp_ready_path,
            'analysis_results': analysis_results,
            'output_dir': output_dir,
            'ai_provider': config['ai_provider'],
            'model_name': config['model_name'],
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        return None

def display_results_section():
    """Display processing results"""
    if not st.session_state.processing_complete or not st.session_state.processed_data:
        st.info("📭 No processing results yet. Please process a file first.")
        return
    
    data = st.session_state.processed_data
    st.header("📊 Processing Results")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Processed", data['file_name'])
    with col2:
        st.metric("🖼️ Images Analyzed", len(data['analysis_results']))
    with col3:
        st.metric("🤖 AI Provider", data['ai_provider'])
    
    # File information
    with st.expander("📁 Generated Files", expanded=True):
        st.write("**📄 Original JSON:**", data['original_json'])
        st.write("**📊 Enhanced JSON:**", data['enhanced_json'])
        if data.get('nlp_ready_json'):
            st.write("**📝 NLP-Ready JSON:**", data['nlp_ready_json'])
        
        # Download buttons for generated files
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📥 Download Original JSON"):
                download_file(data['original_json'])
        
        with col2:
            if st.button("📥 Download Enhanced JSON"):
                download_file(data['enhanced_json'])
        
        with col3:
            if data.get('nlp_ready_json') and st.button("📥 Download NLP-Ready JSON"):
                download_file(data['nlp_ready_json'])
        
        with col4:
            # Add HTML report generation buttons
            if st.button("📊 Generate HTML Reports", type="primary"):
                generate_html_reports(data)
    
    # HTML Report Export Section
    if st.session_state.get('html_reports_generated'):
        with st.expander("📋 HTML Evaluation Reports", expanded=True):
            st.success("✅ HTML reports have been generated!")
            
            html_reports = st.session_state.get('html_reports', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Image Analysis Report")
                st.info("🖼️ Detailed view of all image recognition results")
                if 'image_analysis_report' in html_reports:
                    st.write(f"**File:** {Path(html_reports['image_analysis_report']).name}")
                    if st.button("🌐 Open Image Analysis Report", key="open_img_report"):
                        open_html_file(html_reports['image_analysis_report'])
            
            with col2:
                st.subheader("📋 Complete Evaluation Report")
                st.info("📄 Combined text extraction + image analysis for evaluation")
                if 'complete_evaluation_report' in html_reports:
                    st.write(f"**File:** {Path(html_reports['complete_evaluation_report']).name}")
                    if st.button("🌐 Open Complete Evaluation Report", key="open_complete_report"):
                        open_html_file(html_reports['complete_evaluation_report'])
            
            st.markdown("---")
            st.markdown("**💡 Evaluation Tips:**")
            st.markdown("- **Image Analysis Report**: Review AI recognition accuracy and descriptions")
            st.markdown("- **Complete Evaluation Report**: Compare original PDF with extracted content")
            st.markdown("- Use the interactive tabs to systematically evaluate extraction quality")
    
    # Image analysis results
    st.subheader("🖼️ Image Analysis Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_type = st.selectbox("Filter by Type", 
                                 ["All", "CONCEPTUAL", "DATA_VISUALIZATION"])
    with col2:
        filter_search = st.selectbox("Filter by Search", 
                                   ["All", "With Web Search", "Without Web Search"])
    with col3:
        sort_by = st.selectbox("Sort by", 
                             ["Picture Number", "Image Type", "Analysis Quality"])
    
    # Apply filters
    filtered_results = apply_filters(data['analysis_results'], filter_type, filter_search, sort_by)
    
    if not filtered_results:
        st.warning("No results match the current filters.")
        return
    
    # Display filtered results
    for i, result in enumerate(filtered_results):
        display_image_analysis(result, result['picture_number'])
    
    # Processing statistics
    create_summary_report(data)

def display_analytics_section():
    """Display analytics and processing history"""
    st.header("📈 Analytics & History")
    
    # Processing history
    if st.session_state.processing_history:
        st.subheader("📊 Processing History")
        
        # Convert to DataFrame for analysis
        import pandas as pd
        
        history_data = []
        for item in st.session_state.processing_history:
            history_data.append({
                'File': item['file_name'],
                'Type': item['file_type'].upper(),
                'Images': len(item['analysis_results']),
                'Provider': item['ai_provider'],
                'Model': item['model_name'],
                'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M')
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", len(st.session_state.processing_history))
        
        with col2:
            total_images = sum(len(item['analysis_results']) for item in st.session_state.processing_history)
            st.metric("Images Processed", total_images)
        
        with col3:
            providers = df['Provider'].value_counts()
            most_used = providers.index[0] if len(providers) > 0 else "None"
            st.metric("Most Used Provider", most_used)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.processing_history = []
            st.rerun()
    
    else:
        st.info("📭 No processing history yet. Process some files to see analytics here.")
    
    # Batch results history
    if st.session_state.batch_results:
        st.subheader("📚 Batch Processing History")
        
        successful_batch = [r for r in st.session_state.batch_results if r.get('success', False)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch Files", len(st.session_state.batch_results))
        with col2:
            st.metric("Successful", len(successful_batch))
        with col3:
            batch_images = sum(len(r.get('analysis_results', [])) for r in successful_batch)
            st.metric("Batch Images", batch_images)

def display_image_analysis(result, image_num):
    """Display individual image analysis result with adaptive resolution"""
    
    # Display image if available
    if 'image_data' in result:
        try:
            import io
            from PIL import Image
            
            # Decode image data
            image_data = base64.b64decode(result['image_data'])
            
            # Get original image dimensions
            image_pil = Image.open(io.BytesIO(image_data))
            original_width, original_height = image_pil.size
            
            # Calculate appropriate display width based on original resolution
            if original_width <= 400:
                # Small images: display at original size or slightly larger
                display_width = min(original_width * 1.2, 500)
            elif original_width <= 800:
                # Medium images: display at reasonable size
                display_width = min(original_width, 600)
            else:
                # Large images: scale down to fit
                display_width = 700
            
            display_width = int(display_width)
            
            # Create expandable image display
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Image info
                st.markdown(f"**📸 Image {image_num}** | Original: {original_width}×{original_height}px | Format: {result.get('image_format', 'unknown').upper()}")
                
                # Normal display
                st.image(
                    image_data, 
                    caption=f"Image {image_num} - Click 'Enlarge' to view full size",
                    width=display_width,
                    use_column_width=False  # 🔧 Don't force stretch
                )
            
            with col2:
                # Enlarge button
                if st.button(f"🔍 Enlarge", key=f"enlarge_{image_num}", help="View image at full resolution"):
                    st.session_state[f'enlarged_{image_num}'] = not st.session_state.get(f'enlarged_{image_num}', False)
                
                # Image quality indicator
                if original_width < 500 or original_height < 500:
                    st.caption("📱 Low Resolution")
                elif original_width > 1500 or original_height > 1500:
                    st.caption("🖥️ High Resolution") 
                else:
                    st.caption("📺 Medium Resolution")
            
            # Show enlarged version if requested
            if st.session_state.get(f'enlarged_{image_num}', False):
                with st.expander("🔍 **Full Size Image**", expanded=True):
                    # Calculate appropriate full-size display
                    max_display_width = min(original_width, 1200)  # Cap at 1200px for UI
                    
                    st.image(
                        image_data,
                        caption=f"Full Size: {original_width}×{original_height}px",
                        width=max_display_width,
                        use_column_width=False
                    )
                    
                    # Additional image info
                    file_size = len(image_data) / 1024  # KB
                    st.caption(f"📊 Size: {file_size:.1f} KB | Aspect Ratio: {original_width/original_height:.2f}")
                    
                    # Download button for the image
                    img_format = result.get('image_format', 'png')
                    st.download_button(
                        label=f"💾 Download Image",
                        data=image_data,
                        file_name=f"image_{image_num}.{img_format}",
                        mime=f"image/{img_format}",
                        key=f"download_img_{image_num}"
                    )
        
        except Exception as e:
            st.error(f"❌ Could not display image: {e}")
            
            # Fallback display without PIL
            try:
                image_data = base64.b64decode(result['image_data'])
                st.image(
                    image_data, 
                    caption=f"Image {image_num} (Fallback Mode)",
                    width=400,  # Safe fallback width
                    use_column_width=False
                )
            except:
                st.warning("⚠️ Image data could not be processed")
    
    # Display analysis
    st.markdown("**🤖 AI Analysis:**")
    st.write(result.get('detailed_description', 'No description available'))
    
    # Display image type with color coding
    image_type = result.get('image_type', 'UNKNOWN')
    if image_type == 'CONCEPTUAL':
        st.markdown("🧠 **Type:** <span style='color: #2E8B57'>Conceptual/Methodological</span>", unsafe_allow_html=True)
    elif image_type == 'DATA_VISUALIZATION':
        st.markdown("📊 **Type:** <span style='color: #4169E1'>Data Visualization</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"❓ **Type:** <span style='color: #708090'>{image_type}</span>", unsafe_allow_html=True)
    
    # Display original caption if available
    if result.get('original_caption') and result['original_caption'] != "No caption":
        st.markdown(f"**📝 Original Caption:** {result['original_caption']}")
    
    # Display web search results if available
    if 'web_context' in result:
        web_context = result['web_context']
        search_method = web_context.get('search_method', 'unknown')
        
        # Display appropriate header based on search method
        if search_method == 'native':
            st.markdown("**🌐 AI Native Web Search Results:**")
            st.info("🔗 This analysis includes real-time web search results directly integrated by the AI model.")
        elif search_method == 'duckduckgo':
            st.markdown("**🦆 DuckDuckGo Web Search Context:**")
        else:
            st.markdown("**🌐 Web Search Context:**")
        
        # Show search keywords if available (for DuckDuckGo or fallback)
        if search_method != 'native' and 'search_keywords' in web_context:
            st.markdown(f"**Keywords:** `{', '.join(web_context['search_keywords'])}`")
        
        if search_method == 'native':
            # For native search, the results are already integrated in the main description
            sources_count = web_context.get('sources_count', 0)
            fallback_used = web_context.get('fallback_used', False)
            
            if fallback_used:
                st.warning("⚠️ Native web search failed, analysis completed without web enhancement")
            elif sources_count > 0:
                st.markdown(f"**Sources Used:** {sources_count} web sources integrated into analysis")
                
                # Show actual search queries performed by Gemini
                search_queries = web_context.get('search_queries', [])
                if search_queries:
                    st.markdown(f"**Search Queries:** `{', '.join(search_queries)}`")
                
                # Show native sources if available
                if 'sources' in web_context and web_context['sources']:
                    with st.expander(f"🔍 View {len(web_context['sources'])} Native Search Sources"):
                        for j, source in enumerate(web_context['sources'], 1):
                            st.markdown(f"**Source {j}:** [{source.get('title', 'Untitled')}]({source.get('url', '#')})")
                            if source.get('snippet'):
                                st.write(source['snippet'])
                            if j < len(web_context['sources']):
                                st.markdown("---")
                
                # Show grounding supports if available (which text segments came from which sources)
                grounding_supports = web_context.get('grounding_supports', [])
                if grounding_supports:
                    with st.expander("📑 View Text-Source Mappings"):
                        st.caption("These show which parts of the analysis came from which web sources:")
                        for support in grounding_supports:
                            if support.get('text'):
                                text_preview = support['text'][:100] + "..." if len(support['text']) > 100 else support['text']
                                chunk_indices = support.get('chunk_indices', [])
                                if chunk_indices:
                                    source_nums = [str(i+1) for i in chunk_indices if i < len(web_context.get('sources', []))]
                                    st.markdown(f"**Text:** {text_preview}")
                                    st.markdown(f"**Sources:** {', '.join(source_nums) if source_nums else 'Unknown'}")
                                    st.markdown("---")
            else:
                st.info("🔍 Web search was available but no external sources were needed for this analysis")
        else:
            # For DuckDuckGo search, show traditional summary
            if 'ai_summary' in web_context:
                st.markdown("**Summary:**")
                st.info(web_context['ai_summary'])
            
            if 'sources' in web_context and web_context['sources']:
                with st.expander(f"📚 View {len(web_context['sources'])} DuckDuckGo Sources"):
                    for j, source in enumerate(web_context['sources'], 1):
                        st.markdown(f"**Source {j}:** [{source.get('title', 'Untitled')}]({source.get('url', '#')})")
                        if source.get('body'):
                            st.write(source['body'][:200] + "..." if len(source['body']) > 200 else source['body'])
                        if j < len(web_context['sources']):
                            st.markdown("---")
        
        # Show search method and timestamp
        search_timestamp = web_context.get('search_timestamp', 'Unknown')
        method_emoji = "🌐" if search_method == 'native' else "🦆"
        st.caption(f"{method_emoji} Search method: {search_method.title()} | Searched at: {search_timestamp}")

def generate_html_reports(data):
    """Generate HTML reports for evaluation"""
    try:
        with st.spinner("🔄 Generating HTML reports..."):
            from html_report_generator import HTMLReportGenerator
            
            generator = HTMLReportGenerator()
            base_filename = data['base_filename']
            output_dir = data['output_dir']
            analysis_results = data['analysis_results']
            original_json = data['original_json']
            
            # Generate image analysis report
            image_report_path = generator.generate_image_analysis_report(
                analysis_results, base_filename, output_dir
            )
            
            # Generate complete evaluation report
            complete_report_path = generator.generate_complete_evaluation_report(
                original_json, analysis_results, base_filename, output_dir
            )
            
            # Store report paths in session state
            st.session_state.html_reports = {
                'image_analysis_report': image_report_path,
                'complete_evaluation_report': complete_report_path
            }
            st.session_state.html_reports_generated = True
            
            st.success("✅ HTML reports generated successfully!")
            st.rerun()  # Refresh to show the reports section
            
    except Exception as e:
        st.error(f"❌ Failed to generate HTML reports: {str(e)}")
        logging.error(f"HTML report generation failed: {e}", exc_info=True)

def open_html_file(file_path):
    """Open HTML file in default browser"""
    try:
        import webbrowser
        import os
        
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Open in default browser
        webbrowser.open(f"file://{abs_path}")
        st.success(f"🌐 Opened {Path(file_path).name} in your default browser")
        
    except Exception as e:
        st.error(f"❌ Failed to open HTML file: {str(e)}")
        st.info(f"📁 File location: {file_path}")

def apply_filters(analysis_results, filter_type, filter_search, sort_by):
    """Apply filters and sorting to analysis results"""
    filtered_results = []
    
    for result in analysis_results:
        # Skip non-informative images (already filtered out in processing)
        if result.get('is_non_informative', False):
            continue
        
        # Apply type filter
        if filter_type != "All":
            if result.get('image_type', 'UNKNOWN') != filter_type:
                continue
        
        # Apply search filter
        has_web_search = 'web_context' in result
        if filter_search == "With Web Search" and not has_web_search:
            continue
        elif filter_search == "Without Web Search" and has_web_search:
            continue
        
        filtered_results.append(result)
    
    # Apply sorting
    if sort_by == "Image Type":
        filtered_results.sort(key=lambda x: x.get('image_type', 'UNKNOWN'))
    elif sort_by == "Analysis Quality":
        # Sort by description length as a proxy for analysis quality
        filtered_results.sort(key=lambda x: len(x.get('detailed_description', '')), reverse=True)
    # Default: sort by picture number (already in order)
    
    return filtered_results

def download_file(file_path):
    """Handle file download"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Use st.download_button would be ideal, but we need to trigger it from a regular button
            # Instead, we'll show the file path and suggest manual download
            st.success(f"📁 File ready for download: {file_path}")
            st.info("💡 You can copy this path and access the file directly from your file explorer")
            
            # Also provide a download button
            file_name = Path(file_path).name
            if file_path.endswith('.json'):
                mime_type = "application/json"
            else:
                mime_type = "application/octet-stream"
            
            st.download_button(
                label=f"💾 Download {file_name}",
                data=file_content,
                file_name=file_name,
                mime=mime_type,
                key=f"download_{file_name}_{time.time()}"  # Unique key
            )
        else:
            st.error(f"❌ File not found: {file_path}")
    
    except Exception as e:
        st.error(f"❌ Failed to prepare download: {str(e)}")

def create_summary_report(data):
    """Create and display processing summary"""
    st.subheader("📈 Processing Summary")
    
    analysis_results = data['analysis_results']
    
    # Calculate statistics
    total_images = len(analysis_results)
    conceptual_count = sum(1 for r in analysis_results if r.get('image_type') == 'CONCEPTUAL')
    data_viz_count = sum(1 for r in analysis_results if r.get('image_type') == 'DATA_VISUALIZATION')
    web_search_count = sum(1 for r in analysis_results if 'web_context' in r)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Images", total_images)
    
    with col2:
        st.metric("🗺️ Conceptual", conceptual_count)
    
    with col3:
        st.metric("📈 Data Viz", data_viz_count)
    
    with col4:
        st.metric("🌐 Web Enhanced", web_search_count)
    
    # Processing details
    with st.expander("🔍 Processing Details"):
        st.write(f"**AI Provider:** {data['ai_provider']}")
        st.write(f"**Model:** {data['model_name']}")
        st.write(f"**Processed:** {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Output Directory:** {data['output_dir']}")
        
        # File sizes
        try:
            original_size = os.path.getsize(data['original_json']) / 1024 / 1024
            enhanced_size = os.path.getsize(data['enhanced_json']) / 1024 / 1024
            st.write(f"**Original JSON Size:** {original_size:.2f} MB")
            st.write(f"**Enhanced JSON Size:** {enhanced_size:.2f} MB")
            
            if data.get('nlp_ready_json') and os.path.exists(data['nlp_ready_json']):
                nlp_size = os.path.getsize(data['nlp_ready_json']) / 1024 / 1024
                st.write(f"**NLP-Ready JSON Size:** {nlp_size:.2f} MB")
        except:
            pass

def create_images_zip(analysis_results):
    """Create a ZIP file containing all extracted images"""
    try:
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(analysis_results):
                if result.get('is_non_informative', False) or 'image_data' not in result:
                    continue
                
                try:
                    image_data = base64.b64decode(result['image_data'])
                    image_format = result.get('image_format', 'png')
                    filename = f"image_{result.get('picture_number', i+1)}.{image_format}"
                    
                    zip_file.writestr(filename, image_data)
                    
                    # Also add analysis as text file
                    analysis_text = f"""Image Analysis Report
{'='*30}

Image Number: {result.get('picture_number', i+1)}
Type: {result.get('image_type', 'UNKNOWN')}
Original Caption: {result.get('original_caption', 'No caption')}

AI Analysis:
{result.get('detailed_description', 'No description available')}

"""
                    
                    if 'web_context' in result:
                        analysis_text += f"""
Web Search Results:
Keywords: {', '.join(result['web_context'].get('search_keywords', []))}
Summary: {result['web_context'].get('ai_summary', 'No summary')}
"""
                    
                    analysis_filename = f"analysis_{result.get('picture_number', i+1)}.txt"
                    zip_file.writestr(analysis_filename, analysis_text)
                    
                except Exception as e:
                    logging.error(f"Failed to add image {i+1} to ZIP: {e}")
                    continue
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        logging.error(f"Failed to create ZIP file: {e}")
        return None

def display_batch_results():
    """Display batch processing results with individual tabs for each file"""
    st.header("📚 Batch Processing Results")
    
    batch_results = st.session_state.batch_results
    successful_results = [r for r in batch_results if r.get('success', False)]
    failed_results = [r for r in batch_results if not r.get('success', False)]
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Total Files", len(batch_results))
    with col2:
        st.metric("✅ Successful", len(successful_results))
    with col3:
        st.metric("❌ Failed", len(failed_results))
    with col4:
        total_images = sum(len(r.get('analysis_results', [])) for r in successful_results)
        st.metric("🖼️ Images Analyzed", total_images)
    
    # Batch actions
    st.subheader("🔧 Batch Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate All HTML Reports", type="primary"):
            generate_batch_html_reports(successful_results)
    
    with col2:
        if st.button("📥 Download Batch Summary"):
            create_batch_summary_report(batch_results)
    
    with col3:
        if st.button("🗜️ Create Batch ZIP"):
            create_batch_zip_download(successful_results)
    
    # Show failed files if any
    if failed_results:
        with st.expander("❌ Failed Files", expanded=False):
            for result in failed_results:
                st.error(f"**{result['file_name']}**: {result.get('error', 'Unknown error')}")
    
    # Individual file results in tabs
    if successful_results:
        st.subheader("📊 Individual Results")
        
        # Create tabs for each successful file
        tab_names = [f"{r['file_index']}. {Path(r['file_name']).stem}" for r in successful_results]
        tabs = st.tabs(tab_names)
        
        for tab, result in zip(tabs, successful_results):
            with tab:
                display_single_batch_result(result)

def display_single_batch_result(result):
    """Display results for a single file from batch processing"""
    # File information
    st.markdown(f"### 📄 {result['file_name']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🖼️ Images Found", len(result.get('analysis_results', [])))
    with col2:
        st.metric("🤖 AI Provider", result.get('ai_provider', 'Unknown'))
    with col3:
        st.metric("⏱️ Processed", result['timestamp'].strftime('%H:%M:%S'))
    
    # File paths
    with st.expander("📁 Generated Files", expanded=False):
        if result.get('original_json'):
            st.write(f"**📄 Original JSON:** {result['original_json']}")
        if result.get('enhanced_json'):
            st.write(f"**📊 Enhanced JSON:** {result['enhanced_json']}")
        if result.get('nlp_ready_json'):
            st.write(f"**📝 NLP-Ready JSON:** {result['nlp_ready_json']}")
        
        # Individual file actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📊 Generate HTML Reports", key=f"html_{result['file_index']}"):
                generate_single_html_reports(result)
        with col2:
            if st.button(f"📁 Open Output Folder", key=f"folder_{result['file_index']}"):
                open_output_folder(result['output_dir'])
    
    # Show analysis results if any
    analysis_results = result.get('analysis_results', [])
    if analysis_results:
        # Filter and display options
        col1, col2 = st.columns(2)
        with col1:
            show_images = st.checkbox(f"Show Images", value=False, key=f"show_img_{result['file_index']}")
        with col2:
            max_display = st.slider(f"Max Images to Display", 1, len(analysis_results), 
                                  min(3, len(analysis_results)), key=f"max_img_{result['file_index']}")
        
        # Display filtered results
        for i, img_result in enumerate(analysis_results[:max_display]):
            with st.expander(f"🖼️ Image {img_result['picture_number']}: {img_result.get('image_type', 'UNKNOWN')}", 
                           expanded=False):
                if show_images:
                    display_image_analysis(img_result, img_result['picture_number'])
                else:
                    # Show text summary only
                    st.write(f"**Type:** {img_result.get('image_type', 'UNKNOWN')}")
                    st.write(f"**Description:** {img_result.get('detailed_description', 'No description')[:200]}...")
                    if 'web_context' in img_result:
                        st.write(f"**Web Enhanced:** ✅")
        
        if len(analysis_results) > max_display:
            st.info(f"📝 Showing {max_display} of {len(analysis_results)} images. Adjust slider to see more.")
    else:
        if result.get('message'):
            st.info(f"ℹ️ {result['message']}")
        else:
            st.info("ℹ️ No images were analyzed for this file.")

def generate_batch_html_reports(successful_results):
    """Generate HTML reports for all successfully processed files"""
    try:
        with st.spinner(f"🔄 Generating HTML reports for {len(successful_results)} files..."):
            from html_report_generator import HTMLReportGenerator
            generator = HTMLReportGenerator()
            
            generated_reports = []
            
            for result in successful_results:
                if result.get('analysis_results'):  # Only generate if there are analysis results
                    try:
                        base_filename = result['base_filename']
                        output_dir = result['output_dir']
                        analysis_results = result['analysis_results']
                        original_json = result['original_json']
                        
                        # Generate reports with batch mode naming
                        image_report = generator.generate_image_analysis_report(
                            analysis_results, base_filename, output_dir, batch_mode=True
                        )
                        complete_report = generator.generate_complete_evaluation_report(
                            original_json, analysis_results, base_filename, output_dir, batch_mode=True
                        )
                        
                        generated_reports.append({
                            'file_name': result['file_name'],
                            'image_report': image_report,
                            'complete_report': complete_report
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Failed to generate reports for {result['file_name']}: {str(e)}")
            
            st.success(f"✅ Generated HTML reports for {len(generated_reports)} files!")
            
            # Show generated reports
            with st.expander("📋 Generated HTML Reports", expanded=True):
                for report_info in generated_reports:
                    st.write(f"**{report_info['file_name']}:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🌐 Open Image Report", key=f"open_img_{report_info['file_name']}"):
                            open_html_file(report_info['image_report'])
                    with col2:
                        if st.button(f"🌐 Open Complete Report", key=f"open_complete_{report_info['file_name']}"):
                            open_html_file(report_info['complete_report'])
            
    except Exception as e:
        st.error(f"❌ Failed to generate batch HTML reports: {str(e)}")

def generate_single_html_reports(result):
    """Generate HTML reports for a single file"""
    try:
        if not result.get('analysis_results'):
            st.warning("⚠️ No analysis results available for HTML report generation")
            return
            
        with st.spinner("🔄 Generating HTML reports..."):
            from html_report_generator import HTMLReportGenerator
            generator = HTMLReportGenerator()
            
            base_filename = result['base_filename']
            output_dir = result['output_dir']
            analysis_results = result['analysis_results']
            original_json = result['original_json']
            
            # Generate reports with batch mode naming
            image_report = generator.generate_image_analysis_report(
                analysis_results, base_filename, output_dir, batch_mode=True
            )
            complete_report = generator.generate_complete_evaluation_report(
                original_json, analysis_results, base_filename, output_dir, batch_mode=True
            )
            
            st.success("✅ HTML reports generated!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌐 Open Image Report", key=f"single_img_{result['file_index']}"):
                    open_html_file(image_report)
            with col2:
                if st.button("🌐 Open Complete Report", key=f"single_complete_{result['file_index']}"):
                    open_html_file(complete_report)
    
    except Exception as e:
        st.error(f"❌ Failed to generate HTML reports: {str(e)}")

def open_output_folder(output_dir):
    """Open output folder in file explorer"""
    try:
        import subprocess
        import platform
        
        abs_path = os.path.abspath(output_dir)
        
        if platform.system() == "Windows":
            subprocess.run(["explorer", abs_path])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", abs_path])
        else:  # Linux
            subprocess.run(["xdg-open", abs_path])
        
        st.success(f"📁 Opened folder: {abs_path}")
    
    except Exception as e:
        st.error(f"❌ Failed to open folder: {str(e)}")
        st.info(f"📁 Folder location: {output_dir}")

def create_batch_summary_report(batch_results):
    """Create and download a summary report for batch processing"""
    try:
        successful_results = [r for r in batch_results if r.get('success', False)]
        failed_results = [r for r in batch_results if not r.get('success', False)]
        
        # Generate summary report
        report_content = f"""BATCH PDF PROCESSING SUMMARY REPORT
{'='*60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Files: {len(batch_results)}
Successful: {len(successful_results)}
Failed: {len(failed_results)}

PROCESSING SUMMARY:
{'='*30}
"""
        
        for result in successful_results:
            analysis_count = len(result.get('analysis_results', []))
            report_content += f"""
File: {result['file_name']}
Images Analyzed: {analysis_count}
AI Provider: {result.get('ai_provider', 'Unknown')}
Model: {result.get('model_name', 'Unknown')}
Processed: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Output: {result['output_dir']}
"""
        
        if failed_results:
            report_content += f"\n\nFAILED FILES:\n{'='*30}\n"
            for result in failed_results:
                report_content += f"File: {result['file_name']}\nError: {result.get('error', 'Unknown error')}\n\n"
        
        # Provide download
        st.download_button(
            label="📥 Download Batch Summary",
            data=report_content,
            file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="batch_summary_download"
        )
        
    except Exception as e:
        st.error(f"❌ Failed to create batch summary: {str(e)}")

def create_batch_zip_download(successful_results):
    """Create a ZIP file containing all processed files"""
    try:
        with st.spinner("🔄 Creating batch ZIP file..."):
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for result in successful_results:
                    safe_name = "".join(c for c in Path(result['file_name']).stem if c.isalnum() or c in (' ', '-', '_'))
                    
                    # Add JSON files
                    if result.get('enhanced_json') and os.path.exists(result['enhanced_json']):
                        zip_file.write(result['enhanced_json'], f"{safe_name}/{Path(result['enhanced_json']).name}")
                    
                    if result.get('nlp_ready_json') and os.path.exists(result['nlp_ready_json']):
                        zip_file.write(result['nlp_ready_json'], f"{safe_name}/{Path(result['nlp_ready_json']).name}")
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="🗜️ Download Batch ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                key="batch_zip_download"
            )
            
    except Exception as e:
        st.error(f"❌ Failed to create batch ZIP: {str(e)}")

def get_processing_config():
    """Get processing configuration from session state"""
    ai_provider = st.session_state.get('ai_provider', 'LM Studio (Local)')
    
    # Handle API key based on provider
    if ai_provider == "LM Studio (Local)":
        api_key = ""  # No API key needed for local LM Studio
    else:
        api_key = st.session_state.get('api_key', '')
    
    return {
        'ai_provider': ai_provider,
        'model_name': st.session_state.get('model_name', ''),
        'api_key': api_key,
        'lm_studio_url': st.session_state.get('lm_studio_url', 'http://localhost:1234/v1/chat/completions'),
        'enable_web_search': st.session_state.get('enable_web_search', True),
        'generate_nlp_ready': st.session_state.get('generate_nlp_ready', False),
        'max_tokens': st.session_state.get('max_tokens', 700)
    }

def main():
    """Main application function"""
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'batch_processing_complete' not in st.session_state:
        st.session_state.batch_processing_complete = False
    
    # Configure sidebar UI components
    configure_sidebar()
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔄 Process", "📊 Results", "📈 Analytics"])
    
    with tab1:
        # Main processing tab
        file_type, file_list = upload_files_section()
        
        if file_list:
            # Output settings
            output_dir = output_settings_section()
            config = get_processing_config()
            
            # Display processing summary
            if len(file_list) == 1:
                st.info(f"📄 Ready to process: **{file_list[0]['name']}**")
            else:
                st.info(f"📚 Ready to process **{len(file_list)} files** in batch")
            
            # Start processing button
            if st.button("🚀 Start Processing", type="primary"):
                process_batch_files(file_type, file_list, output_dir, config)
    
    with tab2:
        # Results display
        if st.session_state.batch_processing_complete and st.session_state.batch_results:
            display_batch_results()
        else:
            st.info("📭 No processing results yet. Please process files first.")
    
    with tab3:
        # Analytics section
        display_analytics_section()
    
    # Footer
    st.markdown("---")
    st.markdown("**Made with** ❤️ **and Streamlit**")

if __name__ == "__main__":
    main()