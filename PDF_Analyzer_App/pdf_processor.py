import logging
import json
import time
from typing import Optional, Dict
import os
from pathlib import Path

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, PdfPipelineOptions, ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

class PDFProcessor:
    """Handles PDF to JSON conversion using Docling and LM Studio"""
    
    def __init__(self, config):
        """
        Initialize PDF processor
        
        Args:
            config (dict): Configuration dictionary containing LM Studio settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def convert_pdf_to_json(self, pdf_path: str, output_dir: str, output_filename: str = None, use_api: bool = False) -> Optional[Dict]:
        """
        Convert PDF to JSON using Docling with VLM Pipeline or API
        
        Args:
            pdf_path (str): Path to input PDF file
            output_dir (str): Output directory
            output_filename (str): Optional custom filename (if None, use original PDF name)
            use_api (bool): Whether to use remote API instead of local model
            
        Returns:
            Optional[Dict]: Result metadata or None if failed
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename based on original PDF name
            pdf_path_obj = Path(pdf_path)
            if output_filename:
                base_name = output_filename
            else:
                base_name = pdf_path_obj.stem
            
            # Generate full output path with step1 suffix
            json_output_path = os.path.join(output_dir, f"{base_name}_step1_docling.json")
            
            self.logger.info(f"📄 Converting PDF to JSON: {pdf_path_obj.name}")
            self.logger.info(f"📁 Output will be saved as: {Path(json_output_path).name}")
            
            start_time = time.time()
            
            # Check processing method based on configuration
            ai_provider = self.config.get('ai_provider', 'LM Studio (Local)')
            
            if use_api and ai_provider != 'LM Studio (Local)':
                self.logger.info(f"⚙️ Configuring VLM Pipeline to use '{ai_provider}' API...")
                pipeline_options = self._configure_remote_api_pipeline()
                if not pipeline_options:
                    self.logger.error("❌ Failed to configure remote API pipeline")
                    return None
            elif self.config.get('lm_studio_url') and self.config.get('model_name'):
                self.logger.info(f"⚙️ Configuring VLM Pipeline to use '{self.config['model_name']}' on LM Studio...")
                
                # Configure VLM Pipeline Options (matching notebook logic)
                pipeline_options = VlmPipelineOptions(
                    url=self.config.get('lm_studio_url', "http://localhost:1234/v1/chat/completions"),
                    model=self.config.get('model_name', "google/gemma-3-12b-it-gguf"),
                    prompt="Parse the document.",
                    params={"max_tokens": 16384},
                    generate_page_images=True  # Essential for image extraction
                )
                
                # Initialize DocumentConverter with VLM Pipeline
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline,
                            pipeline_options=pipeline_options,
                        ),
                    },
                )
                
                if use_api and ai_provider != 'LM Studio (Local)':
                    self.logger.info(f"🌐 Using {ai_provider} API for enhanced extraction")
                else:
                    self.logger.info("🤖 Using LM Studio VLM pipeline for enhanced extraction")
            else:
                # Fallback to standard pipeline
                self.logger.info("📋 Using standard Docling pipeline")
                converter = DocumentConverter()
            
            # Convert PDF
            self.logger.info(f"🚀 Starting PDF conversion for: {pdf_path_obj.name}")
            conv_result = converter.convert(pdf_path)
            document = conv_result.document
            self.logger.info("✅ PDF conversion complete.")
            
            # Extract metadata
            conversion_time = time.time() - start_time
            metadata = self._extract_metadata(document, pdf_path, conversion_time)
            
            # Save JSON using the document's method (matching notebook logic)
            document.save_as_json(json_output_path)
            self.logger.info(f"💾 Saved JSON: {Path(json_output_path).name}")
            
            # Update metadata with output path
            metadata['json_path'] = json_output_path
            metadata['conversion_status'] = 'success'
            
            self.logger.info(f"🎉 JSON file generated successfully in {conversion_time:.2f} seconds!")
            return metadata
            
        except Exception as e:
            self.logger.error(f"PDF conversion failed: {str(e)}")
            if "LM Studio" in str(e) or "connection" in str(e).lower():
                self.logger.error(f"❌ Is the LM Studio server running with model '{self.config.get('model_name', 'N/A')}' loaded?")
            return None
    
    def _configure_remote_api_pipeline(self) -> Optional[VlmPipelineOptions]:
        """Configure VLM Pipeline for remote API providers"""
        try:
            ai_provider = self.config.get('ai_provider')
            api_key = self.config.get('api_key')
            model_name = self.config.get('model_name')
            
            if not api_key:
                self.logger.error(f"❌ API key required for {ai_provider}")
                return None
            
            # Note: For actual API usage, we fall back to LM Studio format
            # as Docling's ApiVlmOptions may not support all providers directly
            self.logger.warning(f"⚠️ API provider {ai_provider} configured, but using LM Studio format for Docling compatibility")
            
            if ai_provider == "OpenAI (ChatGPT)":
                # Use LM Studio-compatible format but will be handled by API manager
                api_options = ApiVlmOptions(
                    url="https://api.openai.com/v1/chat/completions",
                    params=dict(model=model_name, Authorization=f"Bearer {api_key}"),
                    prompt="Parse the document and extract all text content and images.",
                    timeout=90.0,
                    response_format=ResponseFormat.MARKDOWN
                )
                
            elif ai_provider == "Google (Gemini)":
                api_options = ApiVlmOptions(
                    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
                    params=dict(model=model_name),
                    prompt="Parse the document and extract all text content and images.",
                    timeout=90.0,
                    response_format=ResponseFormat.MARKDOWN
                )
                
            elif ai_provider == "Anthropic (Claude)":
                api_options = ApiVlmOptions(
                    url="https://api.anthropic.com/v1/messages",
                    params=dict(model=model_name, max_tokens=16384, **{"x-api-key": api_key, "anthropic-version": "2023-06-01"}),
                    prompt="Parse the document and extract all text content and images.",
                    timeout=90.0,
                    response_format=ResponseFormat.MARKDOWN
                )
            else:
                self.logger.error(f"❌ Unsupported API provider: {ai_provider}")
                return None
            
            return VlmPipelineOptions(
                api_vlm_options=api_options,
                generate_page_images=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to configure remote API: {e}")
            return None
    
    def _test_lm_studio_connection(self) -> bool:
        """Test LM Studio connection"""
        try:
            import requests
            
            test_payload = {
                "model": self.config.get('model_name', 'test'),
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.config.get('lm_studio_url', 'http://localhost:1234/v1/chat/completions'),
                json=test_payload,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception:
            return False
    
    def _extract_metadata(self, document, pdf_path: str, conversion_time: float) -> Dict:
        """
        Extract metadata from converted document
        
        Args:
            document: Converted document object
            pdf_path (str): Source PDF path
            conversion_time (float): Time taken for conversion
            
        Returns:
            Dict: Metadata dictionary
        """
        try:
            metadata = {
                'source_file': pdf_path,
                'conversion_time': conversion_time,
                'total_pages': 0,
                'total_images': 0,
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Extract page count
            if hasattr(document, 'pages') and document.pages:
                metadata['total_pages'] = len(document.pages)
            
            if hasattr(document, 'pictures') and document.pictures:
                metadata['total_images'] = len(document.pictures)
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Could not extract metadata: {e}")
            return {
                'source_file': pdf_path,
                'conversion_time': conversion_time,
                'total_pages': 0,
                'total_images': 0,
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }