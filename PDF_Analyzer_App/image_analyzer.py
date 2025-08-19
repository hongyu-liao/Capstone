import logging
from pathlib import Path
import json
import base64
import requests
import re
import time
from typing import List, Dict, Optional, Any
import os # Added for os.path.join

# Try to import web search package, fallback gracefully
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False
        DDGS = None
        logging.warning("Web search package not available. Web search will be disabled.")

from api_manager import APIManager

# Import chart extraction dependencies
try:
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    from PIL import Image
    import io
    HAS_DEPLOT = True
except ImportError:
    HAS_DEPLOT = False
    logging.warning("Chart extraction dependencies not available. DePlot functionality will be disabled.")

# Import ChartGemma dependencies
try:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    import torch
    HAS_CHARTGEMMA = True
except ImportError:
    HAS_CHARTGEMMA = False
    logging.warning("ChartGemma dependencies not available. Advanced chart analysis will be disabled.")

class ImageAnalyzer:
    """Handles image extraction, analysis, and web search enhancement"""
    
    def __init__(self, config):
        """
        Initialize image analyzer
        
        Args:
            config (dict): Configuration dictionary containing AI provider and processing settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_manager = APIManager(config)
        
        # ChartGemma components
        self.chartgemma_model = None
        self.chartgemma_processor = None
        self.chartgemma_device = None
        
    def _safe_get_caption(self, pic_data: Dict) -> str:
        """
        Safely extract caption from picture data
        
        Args:
            pic_data (Dict): Picture data from JSON
            
        Returns:
            str: Caption text or "No caption" if not available
        """
        try:
            captions = pic_data.get("captions", [])
            if captions and len(captions) > 0 and isinstance(captions[0], dict):
                return captions[0].get("text", "No caption")
            else:
                return "No caption"
        except (IndexError, TypeError, AttributeError):
            return "No caption"
        
    def analyze_images_from_json(self, json_path: str, enable_web_search: bool = True, enable_chart_extraction: bool = True, enable_chartgemma: bool = True) -> List[Dict]:
        """
        Analyze all images from a Docling-generated JSON file
        
        Args:
            json_path (str): Path to the JSON file
            enable_web_search (bool): Whether to perform web search for conceptual images
            enable_chart_extraction (bool): Whether to extract chart data for data visualization images
            
        Returns:
            List[Dict]: List of analysis results for each image
        """
        json_path = Path(json_path)
        
        if not json_path.is_file():
            self.logger.error(f"JSON file not found: {json_path}")
            return []
        
        self.logger.info(f"Analyzing images from: {json_path.name}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {e}")
            return []
        
        pictures = data.get("pictures", [])
        if not pictures:
            self.logger.warning("No pictures found in JSON file")
            return []
        
        self.logger.info(f"Found {len(pictures)} pictures to analyze")
        
        analysis_results = []
        
        for i, pic_data in enumerate(pictures):
            self.logger.info(f"Analyzing image {i+1}/{len(pictures)}")
            
            try:
                result = self._analyze_single_image(pic_data, i+1, enable_web_search, enable_chart_extraction, enable_chartgemma)
                
                # Skip non-informative images completely (like original notebook)
                if result is None:
                    self.logger.info(f"   ❌ Skipping non-informative image #{i+1} (logo/watermark)")
                    continue
                
                analysis_results.append(result)
                    
                # Small delay to be respectful to APIs
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze image {i+1}: {e}")
                # Create a basic result for failed analysis with safe caption access
                failed_result = {
                    'picture_number': i+1,
                    'is_non_informative': False,
                    'self_ref': pic_data.get('self_ref', f'picture_{i+1}'),
                    'original_caption': self._safe_get_caption(pic_data),  # 🔧 Safe caption access
                    'image_type': 'ANALYSIS_FAILED',
                    'detailed_description': f'Failed to analyze image: {str(e)}',
                    'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_used': self.config.get('model_name', 'Unknown')
                }
                
                # Try to extract image data for display
                try:
                    image_uri = pic_data.get("image", {}).get("uri")
                    if image_uri and image_uri.startswith("data:image"):
                        match = re.match(r'data:image/([^;]+);base64,(.+)', image_uri)
                        if match:
                            failed_result['image_data'] = match.group(2)
                            failed_result['image_format'] = match.group(1)
                except:
                    pass
                
                analysis_results.append(failed_result)
                continue
        
        self.logger.info(f"Completed analysis of {len(analysis_results)} informative images (skipped non-informative ones)")
        return analysis_results
    
    def _analyze_single_image(self, pic_data: Dict, pic_number: int, enable_web_search: bool, enable_chart_extraction: bool = True, enable_chartgemma: bool = True) -> Optional[Dict]:
        """
        Analyze a single image using the configured AI provider
        
        Args:
            pic_data (Dict): Picture data from JSON
            pic_number (int): Picture number for logging
            enable_web_search (bool): Whether to perform web search
            enable_chart_extraction (bool): Whether to extract chart data for data visualization images
            
        Returns:
            Optional[Dict]: Analysis result or None if image is non-informative (logo/watermark)
        """
        # Extract image data
        image_uri = pic_data.get("image", {}).get("uri")
        if not image_uri or not image_uri.startswith("data:image"):
            self.logger.warning(f"No valid image data for picture {pic_number}")
            return None
        
        # Extract base64 data for potential display
        try:
            match = re.match(r'data:image/([^;]+);base64,(.+)', image_uri)
            if match:
                image_format = match.group(1)
                base64_data = match.group(2)
            else:
                self.logger.warning(f"Invalid image URI format for picture {pic_number}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to parse image URI for picture {pic_number}: {e}")
            return None
        
        # Check if AI provider supports native web search
        supports_native_search = self.api_manager.supports_native_web_search()
        
        # Get AI analysis - use web search if supported and enabled for conceptual images
        if enable_web_search and supports_native_search:
            self.logger.info(f"🌐 Using {self.config.get('ai_provider')} native web search for picture {pic_number}")
            ai_analysis = self._get_ai_analysis_with_native_search(image_uri, pic_number)
        else:
            ai_analysis = self._get_ai_analysis(image_uri, pic_number)
        
        if not ai_analysis:
            return None
        
        # Check if image is non-informative - if so, return None to skip completely
        if ai_analysis.get('is_non_informative', False):
            self.logger.info(f"Picture {pic_number} identified as non-informative")
            return None  # 🔧 Return None instead of creating a record
        
        # Build result - ALWAYS include image data for display
        result = {
            'picture_number': pic_number,
            'is_non_informative': False,
            'self_ref': pic_data.get('self_ref', f'picture_{pic_number}'),
            'original_caption': self._safe_get_caption(pic_data),  # 🔧 Safe caption access
            'image_type': ai_analysis.get('image_type', 'UNKNOWN'),
            'detailed_description': ai_analysis.get('detailed_description', ''),
            'image_data': base64_data,  # Always include for display
            'image_format': image_format,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': self.config.get('model_name', 'Unknown'),
            'ai_provider': self.config.get('ai_provider', 'Unknown')
        }
        
        # Handle chart data extraction for DATA_VISUALIZATION images
        if enable_chart_extraction and ai_analysis.get('image_type') == 'DATA_VISUALIZATION':
            self.logger.info(f"📊 Extracting chart data for picture {pic_number}")
            
            # Step 1: Get raw DePlot output
            chart_data = self._extract_chart_data_with_deplot(image_uri, pic_number)
            
            if chart_data and chart_data.get("raw_table"):
                # Step 2: Use image + DePlot verification for better accuracy
                self.logger.info(f"🔍 Verifying chart data with image analysis for picture {pic_number}")
                verified_chart = self._extract_chart_with_image_verification(
                    image_uri, chart_data["raw_table"], pic_number
                )
                
                if verified_chart and verified_chart.get("parsing_success"):
                    result['chart_data_extraction'] = verified_chart
                    self.logger.info(f"✅ Successfully verified chart data for picture {pic_number}")
                else:
                    # Fallback to raw DePlot output
                    self.logger.info(f"⚠️ Image verification failed, using raw DePlot for picture {pic_number}")
                    result['chart_data_extraction'] = {
                        "extraction_success": False,
                        "raw_table": chart_data["raw_table"],
                        "extraction_method": "deplot_only",
                        "extraction_timestamp": chart_data.get("extraction_timestamp")
                    }
            else:
                # DePlot extraction failed
                result['chart_data_extraction'] = {
                    "extraction_success": False,
                    "extraction_method": "deplot_failed",
                    "extraction_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }

        # Handle web search results
        if 'web_sources' in ai_analysis and ai_analysis['web_sources']:
            # Native web search was used and returned sources
            result['web_context'] = {
                'search_method': 'native',
                'search_keywords': ai_analysis.get('search_keywords', []),
                'search_queries': ai_analysis.get('search_queries', []),  # Actual queries performed by Gemini
                'sources': ai_analysis['web_sources'],
                'sources_count': len(ai_analysis['web_sources']),
                'ai_summary': "Integrated web search results included in analysis above.",
                'search_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'grounding_supports': ai_analysis.get('grounding_supports', []),  # Text segments with sources
                'fallback_used': ai_analysis.get('fallback_used', False)
            }
            result['search_keywords'] = ai_analysis.get('search_keywords', [])
            
        elif (enable_web_search and 
              not supports_native_search and
              ai_analysis.get('image_type') == 'CONCEPTUAL' and 
              ai_analysis.get('search_keywords')):
            
            # Fallback to DuckDuckGo search for local models
            self.logger.info(f"🦆 Using DuckDuckGo search for conceptual image {pic_number}")
            web_context = self._perform_web_search(
                ai_analysis['search_keywords'],
                ai_analysis['detailed_description'],
                pic_number
            )
            
            if web_context:
                result['web_context'] = web_context
                result['search_keywords'] = ai_analysis['search_keywords']
        
        # ChartGemma analysis for DATA_VISUALIZATION images
        if (enable_chartgemma and 
            ai_analysis.get('image_type') == 'DATA_VISUALIZATION' and
            HAS_CHARTGEMMA and self.chartgemma_model is not None):
            
            chart_type = result.get('chart_data_extraction', {}).get('chart_type', 'unknown')
            chartgemma_result = self.analyze_chart_with_chartgemma(image_uri, chart_type, pic_number)
            
            if chartgemma_result:
                result['chartgemma_analysis'] = chartgemma_result
                self.logger.info(f"✅ Added ChartGemma analysis for picture {pic_number}")
        
        return result
    
    def _get_ai_analysis(self, image_uri: str, pic_number: int) -> Optional[Dict]:
        """
        Get AI analysis of an image using the configured AI provider
        
        Args:
            image_uri (str): Data URI of the image
            pic_number (int): Picture number for logging
            
        Returns:
            Optional[Dict]: Analysis result or None if failed
        """
        try:
            enhanced_prompt = """You are an expert scientific analyst. Analyze the provided image and follow these steps:

1. First, determine if this is:
   - INFORMATIVE: A meaningful scientific element (graphs, charts, diagrams, maps, flowcharts, etc.)
   - NON-INFORMATIVE: Publisher logos, watermarks, decorative icons, etc.

2. If NON-INFORMATIVE, respond with exactly: "N/A"

3. If INFORMATIVE, classify the image type:
   - DATA_VISUALIZATION: Charts, graphs, plots with actual data points, statistical visualizations
   - CONCEPTUAL: Flowcharts, process diagrams, maps, conceptual frameworks, schematic diagrams, methodological illustrations

4. CRITICAL: Provide a comprehensive text description that can completely replace the image in a document. This description will be used instead of the image for NLP processing. Include all important details, relationships, data patterns, labels, and contextual information that a reader would need to understand what the image conveyed.

5. If the image is CONCEPTUAL type, also provide 2-3 specific search keywords that would help find background information about the concepts, methods, or geographic locations shown in the image.

Format your response as:
TYPE: [DATA_VISUALIZATION/CONCEPTUAL]
DETAILED_DESCRIPTION: [Your comprehensive replacement text description that captures all essential information from the image]
SEARCH_KEYWORDS: [keyword1, keyword2, keyword3] (only if CONCEPTUAL type)"""

            # Use APIManager to make the call
            ai_response = self.api_manager.analyze_image(
                image_uri, 
                enhanced_prompt, 
                self.config.get('max_tokens', 700)
            )
            
            if not ai_response:
                self.logger.error(f"No response from AI provider for picture {pic_number}")
                return None
            
            # Check for N/A response (non-informative image)
            if ai_response.strip() == "N/A":
                return {'is_non_informative': True}
            
            # Parse structured response
            result = {
                'is_non_informative': False,
                'image_type': 'UNKNOWN',
                'detailed_description': ai_response,
                'search_keywords': []
            }
            
            try:
                lines = ai_response.split('\n')
                for line in lines:
                    if line.startswith('TYPE:'):
                        result['image_type'] = line.replace('TYPE:', '').strip()
                    elif line.startswith('DETAILED_DESCRIPTION:'):
                        result['detailed_description'] = line.replace('DETAILED_DESCRIPTION:', '').strip()
                    elif line.startswith('SEARCH_KEYWORDS:'):
                        keywords_text = line.replace('SEARCH_KEYWORDS:', '').strip()
                        keywords_text = keywords_text.strip('[]')
                        result['search_keywords'] = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            except Exception as e:
                self.logger.warning(f"Could not parse structured response for picture {pic_number}: {e}")
            
            return result
            
        except ValueError as e:
            # Model validation or API errors that should be shown to user
            self.logger.error(f"❌ API Error for picture {pic_number}: {e}")
            raise e  # Re-raise ValueError to show to user
            
        except Exception as e:
            self.logger.error(f"AI analysis failed for picture {pic_number}: {e}")
            return None
    
    def _get_ai_analysis_with_native_search(self, image_uri: str, pic_number: int) -> Optional[Dict]:
        """
        Get AI analysis using provider's native web search capabilities
        First analyze image to get keywords, then search with those keywords
        
        Args:
            image_uri (str): Data URI of the image
            pic_number (int): Picture number for logging
            
        Returns:
            Optional[Dict]: Analysis result or None if failed
        """
        try:
            # Step 1: First get regular AI analysis to determine search keywords
            regular_analysis = self._get_ai_analysis(image_uri, pic_number)
            
            if not regular_analysis or regular_analysis.get('is_non_informative', False):
                return regular_analysis  # Return early for non-informative images
            
            # Step 2: If it's a conceptual image with search keywords, enhance with web search
            if (regular_analysis.get('image_type') == 'CONCEPTUAL' and 
                regular_analysis.get('search_keywords')):
                
                self.logger.info(f"🔍 Gemini will search for: {regular_analysis['search_keywords']}")
                
                # Enhanced prompt that includes the keywords for targeted searching
                search_keywords = regular_analysis['search_keywords']
                enhanced_prompt = f"""You are an expert scientific analyst. Analyze the provided image and provide a comprehensive description.

This image has been identified as a CONCEPTUAL type showing: {regular_analysis.get('detailed_description', 'scientific concepts')}

Please enhance your analysis by searching the web for current information about these specific topics: {', '.join(search_keywords)}

Provide a detailed analysis that integrates both the visual information and relevant web-sourced background information about these concepts.

Format your response as a comprehensive description that combines visual analysis with current contextual information."""

                # Use APIManager's web search enabled analysis with targeted keywords
                search_result = self.api_manager.analyze_image_with_web_search(
                    image_uri, 
                    enhanced_prompt,
                    search_query=' '.join(search_keywords),  # Pass keywords as search query
                    max_tokens=self.config.get('max_tokens', 700)
                )
                
                if search_result and search_result.get('response'):
                    # Combine the original analysis structure with web-enhanced content
                    enhanced_analysis = regular_analysis.copy()
                    enhanced_analysis['detailed_description'] = search_result['response']
                    enhanced_analysis['web_sources'] = search_result.get('web_sources')
                    enhanced_analysis['search_queries'] = search_result.get('search_queries', search_keywords)
                    enhanced_analysis['grounding_supports'] = search_result.get('grounding_supports')
                    
                    self.logger.info(f"✅ Enhanced analysis with web search for keywords: {search_keywords}")
                    return enhanced_analysis
                else:
                    self.logger.warning(f"⚠️ Web search failed, using regular analysis for picture {pic_number}")
                    return regular_analysis
            
            else:
                # For DATA_VISUALIZATION or images without keywords, return regular analysis
                self.logger.info(f"📊 Data visualization or no keywords - skipping web search for picture {pic_number}")
                return regular_analysis
                
        except ValueError as e:
            # Model validation or API errors that should be shown to user
            self.logger.error(f"❌ API Error for picture {pic_number}: {e}")
            raise e  # Re-raise ValueError to show to user
            
        except Exception as e:
            self.logger.error(f"AI analysis with web search failed for picture {pic_number}: {e}")
            # Fallback to regular analysis
            try:
                fallback_analysis = self._get_ai_analysis(image_uri, pic_number)
                if fallback_analysis:
                    fallback_analysis['fallback_used'] = True
                return fallback_analysis
            except:
                return None
    
    def _perform_web_search(self, search_keywords: List[str], image_description: str, pic_number: int) -> Optional[Dict]:
        """
        Perform DuckDuckGo web search and get AI summary (fallback for local models)
        
        Args:
            search_keywords (List[str]): Keywords to search for
            image_description (str): Description of the image
            pic_number (int): Picture number for logging
            
        Returns:
            Optional[Dict]: Web search results and summary or None if failed
        """
        if not HAS_DDGS or DDGS is None:
            self.logger.warning(f"⚠️ Web search package not available for picture {pic_number}. Please install: pip install ddgs")
            return None
        
        try:
            search_query = " ".join(search_keywords[:2])
            
            # Perform search
            ddgs = DDGS()
            results = ddgs.text(search_query, max_results=5)
            
            if not results:
                self.logger.warning(f"No search results for picture {pic_number}")
                return None
            
            # Compile search results
            sources_info = []
            sources_text = ""
            
            for i, result in enumerate(results, 1):
                title = result.get('title', f'Source {i}')
                body = result.get('body', '')
                href = result.get('href', '')
                
                sources_text += f"\nSource {i} - {title}:\n{body}\n"
                if href:
                    sources_text += f"URL: {href}\n"
                sources_text += "-" * 50 + "\n"
                
                sources_info.append({
                    'title': title,
                    'body': body,
                    'url': href
                })
            
            # Generate AI summary using APIManager
            ai_summary = self._generate_search_summary(search_query, sources_text)
            
            return {
                'search_method': 'duckduckgo',
                'search_query': search_query,
                'search_keywords': search_keywords,
                'sources_count': len(sources_info),
                'sources': sources_info,
                'ai_summary': ai_summary,
                'search_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed for picture {pic_number}: {e}")
            return None
    
    def _generate_search_summary(self, search_query: str, sources_text: str) -> str:
        """
        Generate AI summary of search results using the configured AI provider
        
        Args:
            search_query (str): Original search query
            sources_text (str): Compiled search results text
            
        Returns:
            str: AI-generated summary
        """
        try:
            summary_prompt = f"""Based on the following search results about "{search_query}", provide a concise one-paragraph summary focusing on the essential concepts and definitions (maximum 6-8 sentences):

{sources_text}

Please synthesize this information into a brief, coherent explanation.

Summary:"""

            # Use APIManager for text generation
            summary = self.api_manager.generate_text(summary_prompt, 500)
            
            if summary:
                return summary.strip()
            else:
                return "Failed to generate summary from search results."
            
        except Exception as e:
            self.logger.error(f"Failed to generate search summary: {e}")
            return "Failed to generate summary from search results."
    
    def _extract_chart_data_with_deplot(self, image_uri: str, pic_number: int) -> Optional[Dict]:
        """
        Extract chart data using DePlot model
        
        Args:
            image_uri (str): Base64 image data URI
            pic_number (int): Picture number for logging
            
        Returns:
            Optional[Dict]: Chart extraction results or None if failed
        """
        if not HAS_DEPLOT:
            self.logger.warning(f"DePlot not available for chart extraction on picture {pic_number}")
            return None
            
        try:
            if not image_uri or not image_uri.startswith("data:image"):
                self.logger.warning(f"Invalid image URI format for picture {pic_number}")
                return None

            header, encoded = image_uri.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data))

            self.logger.info(f"Extracting chart data from picture {pic_number} using DePlot...")
            processor = Pix2StructProcessor.from_pretrained("google/deplot")
            model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

            prompt = "Generate underlying data table of the figure below:"
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            predictions = model.generate(**inputs, max_new_tokens=512)
            table_string = processor.decode(predictions[0], skip_special_tokens=True)

            return {
                "raw_table": table_string,
                "extraction_method": "google/deplot",
                "extraction_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Chart data extraction failed for picture {pic_number}: {e}")
            return None
    
    def _extract_chart_with_image_verification(self, image_uri: str, raw_deplot_output: str, pic_number: int) -> Optional[Dict]:
        """
        Enhanced chart extraction using image + DePlot verification
        
        Args:
            image_uri (str): Base64 image data URI
            raw_deplot_output (str): Raw DePlot extraction result
            pic_number (int): Picture number for logging
            
        Returns:
            Optional[Dict]: Verified chart data or None if failed
        """
        if not image_uri or not raw_deplot_output:
            return None
            
        try:
            # Enhanced prompt for image + DePlot verification
            enhanced_prompt = f"""You are analyzing a chart image. I have already extracted some data using DePlot model, but the format might not be perfect. Please look at the image and the DePlot result below, then provide a clean, structured table.

DePlot extracted data:
{raw_deplot_output}

Instructions:
1. Look at the chart image carefully
2. Identify the X-axis labels (these might be categorical like model names, or numeric values)
3. Identify all the data series/lines in the chart
4. Create a structured table where each row represents one X-axis point and columns represent different metrics/series

Please return ONLY a JSON object with this exact structure:
{{
    "x_axis_label": "name of x-axis (e.g., 'model', 'time', etc.)",
    "x_categories": ["list", "of", "x-axis", "labels"] (if categorical) or null (if numeric),
    "series_names": ["metric1", "metric2", "metric3", ...],
    "data_table": [
        {{"x": "x_value_1", "metric1": value1, "metric2": value2, ...}},
        {{"x": "x_value_2", "metric1": value1, "metric2": value2, ...}},
        ...
    ],
    "chart_type": "bar_chart" or "line_chart",
    "total_data_points": number
}}

Focus on accuracy and make sure the X-axis values and series data correspond correctly to what you see in the image."""

            # Use APIManager to analyze image with enhanced prompt
            ai_response = self.api_manager.analyze_image(image_uri, enhanced_prompt, 1000)
            
            if not ai_response:
                return None
            
            # Extract JSON from response
            start = ai_response.find("{")
            end = ai_response.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = ai_response[start : end + 1]
                parsed = json.loads(json_str)
                
                # Convert to internal format
                x_label = parsed.get("x_axis_label", "X")
                x_categories = parsed.get("x_categories")
                series_names = parsed.get("series_names", [])
                data_table = parsed.get("data_table", [])
                chart_type = parsed.get("chart_type", "unknown")
                
                # Build datasets in expected format
                datasets = {}
                total_points = 0
                
                for series_name in series_names:
                    datasets[series_name] = []
                    
                for i, row in enumerate(data_table):
                    x_val = i  # Use index as X coordinate for categorical data
                    for series_name in series_names:
                        if series_name in row:
                            try:
                                y_val = float(row[series_name])
                                datasets[series_name].append((x_val, y_val))
                                total_points += 1
                            except (ValueError, TypeError):
                                continue
                
                # Remove empty datasets
                datasets = {k: v for k, v in datasets.items() if v}
                
                if not datasets:
                    self.logger.warning(f"No valid datasets extracted from picture {pic_number}")
                    return None
                    
                result = {
                    "chart_type": chart_type,
                    "x_axis_label": x_label,
                    "y_axis_labels": list(datasets.keys()),
                    "datasets": datasets,
                    "data_points_count": total_points,
                    "x_categories": x_categories,
                    "raw_table": raw_deplot_output,
                    "extraction_method": "image+deplot_verification",
                    "parsing_success": True,
                    "extraction_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.info(f"Successfully extracted chart data for picture {pic_number}: {len(datasets)} series, {total_points} points")
                return result
                
        except Exception as e:
            self.logger.error(f"Image verification failed for picture {pic_number}: {e}")
            return None
    
    def create_enhanced_json(self, original_json_path: str, analysis_results: List[Dict], 
                            output_dir: str, output_filename: str = None, 
                            batch_mode: bool = False) -> str:
        """
        Create enhanced JSON with AI analysis results
        
        Args:
            original_json_path (str): Path to original JSON file
            analysis_results (List[Dict]): Image analysis results
            output_dir (str): Output directory
            output_filename (str): Optional custom filename (if None, derive from original)
            batch_mode (bool): If True, use simplified naming for batch processing
            
        Returns:
            str: Path to enhanced JSON file
        """
        try:
            # Load original JSON
            with open(original_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Generate output filename based on original JSON name
            original_path_obj = Path(original_json_path)
            if output_filename:
                base_name = output_filename
            else:
                # Extract base name from original file, removing step1 suffix if present
                base_name = original_path_obj.stem
                if base_name.endswith('_step1_docling'):
                    base_name = base_name[:-14]  # Remove '_step1_docling'
            
            # Generate enhanced JSON path with appropriate suffix
            if batch_mode:
                enhanced_json_path = os.path.join(output_dir, f"{base_name}_enhanced.json")
            else:
                enhanced_json_path = os.path.join(output_dir, f"{base_name}_step2_enhanced.json")
            
            self.logger.info(f"📊 Creating enhanced JSON: {Path(enhanced_json_path).name}")
            
            # Create enhanced data structure
            enhanced_data = original_data.copy()
            
            # Add analysis metadata
            enhanced_data['step2_metadata'] = {
                'version': '2.0',
                'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ai_provider': self.config.get('ai_provider', 'Unknown'),
                'model_name': self.config.get('model_name', 'Unknown'),
                'total_images_analyzed': len(analysis_results),
                'web_search_enabled': self.config.get('enable_web_search', False),
                'processing_notes': 'Enhanced with AI image analysis and optional web search context'
            }
            
            # Update pictures with AI analysis
            if 'pictures' in enhanced_data:
                # Create mapping of analyzed images
                analysis_map = {result['picture_number']: result for result in analysis_results}
                
                updated_pictures = []
                for i, pic in enumerate(enhanced_data['pictures'], 1):
                    if i in analysis_map:
                        # Add AI analysis to the picture
                        pic['ai_analysis'] = analysis_map[i]
                        updated_pictures.append(pic)
                    # Note: Non-informative images are completely excluded
                
                enhanced_data['pictures'] = updated_pictures
                
                # Update metadata counts
                enhanced_data['step2_metadata']['original_images_count'] = len(original_data.get('pictures', []))
                enhanced_data['step2_metadata']['informative_images_count'] = len(updated_pictures)
                enhanced_data['step2_metadata']['non_informative_images_removed'] = len(original_data.get('pictures', [])) - len(updated_pictures)
            
            # Save enhanced JSON
            with open(enhanced_json_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Enhanced JSON created: {enhanced_json_path}")
            return enhanced_json_path
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced JSON: {e}")
            raise

    def load_chartgemma_model(self):
        """Load ChartGemma model for advanced chart analysis"""
        if not HAS_CHARTGEMMA:
            self.logger.warning("ChartGemma dependencies not available")
            return False
            
        try:
            self.logger.info("🤖 Loading ChartGemma model...")
            
            # Determine device
            if torch.cuda.is_available():
                self.chartgemma_device = torch.device("cuda")
            else:
                self.chartgemma_device = torch.device("cpu")
            
            self.chartgemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
                "ahmed-masry/chartgemma", 
                torch_dtype=torch.float16 if self.chartgemma_device.type == "cuda" else torch.float32
            ).to(self.chartgemma_device)
            
            self.chartgemma_processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
            
            self.logger.info("✅ ChartGemma model loaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ ChartGemma model loading failed: {e}")
            return False

    def analyze_chart_with_chartgemma(self, image_uri: str, chart_type: str = "unknown", pic_number: int = 1) -> Optional[Dict]:
        """Analyze chart using ChartGemma model"""
        if not HAS_CHARTGEMMA or self.chartgemma_model is None:
            return None
            
        try:
            # Extract base64 data and decode
            if image_uri.startswith("data:image"):
                base64_data = image_uri.split(",")[1]
            else:
                base64_data = image_uri
                
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Get appropriate question based on chart type
            question = self._get_chartgemma_question(chart_type)
            
            self.logger.info(f"📊 Analyzing chart {pic_number} with ChartGemma...")
            
            # Process inputs
            inputs = self.chartgemma_processor(text=question, images=image, return_tensors="pt")
            prompt_length = inputs['input_ids'].shape[1]
            inputs = {k: v.to(self.chartgemma_device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.chartgemma_model.generate(
                    **inputs, 
                    num_beams=4, 
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode output
            output_text = self.chartgemma_processor.batch_decode(
                generate_ids[:, prompt_length:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return {
                "question": question,
                "response": output_text.strip(),
                "chart_type_detected": chart_type,
                "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"❌ ChartGemma analysis failed for picture {pic_number}: {e}")
            return {"error": str(e), "chart_type_detected": chart_type}

    def _get_chartgemma_question(self, chart_type: str = "unknown") -> str:
        """Get appropriate question based on chart type"""
        if "line" in chart_type.lower():
            return "Describe this line chart in detail. Identify each line/trend, their patterns over time, maximum and minimum values, and overall trend directions."
        elif "bar" in chart_type.lower():
            return "Describe this bar chart in detail. List all categories and their values, identify highest and lowest values, and describe any patterns."
        elif "pie" in chart_type.lower():
            return "Describe this pie chart in detail. List each segment with its percentage, identify the largest and smallest portions."
        elif "scatter" in chart_type.lower():
            return "Describe this scatter plot in detail. Analyze the correlation pattern, identify any outliers, and describe the overall trend."
        else:
            return "Describe this chart in detail, including all visible elements, data patterns, trends, and key insights. Extract all numerical values and relationships."

    def create_nlp_ready_version(self, enhanced_json_path: str, output_dir: str, 
                                output_filename: str = None, batch_mode: bool = False) -> str:
        """
        Create NLP-ready JSON by removing image data and keeping only text descriptions
        
        Args:
            enhanced_json_path (str): Path to enhanced JSON file
            output_dir (str): Output directory
            output_filename (str): Optional custom filename (if None, derive from enhanced)
            batch_mode (bool): If True, use simplified naming for batch processing
            
        Returns:
            str: Path to NLP-ready JSON file
        """
        try:
            # Load enhanced JSON
            with open(enhanced_json_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
            
            # Generate output filename
            enhanced_path_obj = Path(enhanced_json_path)
            if output_filename:
                base_name = output_filename
            else:
                # Extract base name from enhanced file, removing suffix if present
                base_name = enhanced_path_obj.stem
                if base_name.endswith('_step2_enhanced'):
                    base_name = base_name[:-15]  # Remove '_step2_enhanced'
                elif base_name.endswith('_enhanced'):
                    base_name = base_name[:-9]  # Remove '_enhanced'
            
            # Generate NLP-ready JSON path with appropriate suffix
            if batch_mode:
                nlp_ready_path = os.path.join(output_dir, f"{base_name}_nlp_ready.json")
            else:
                nlp_ready_path = os.path.join(output_dir, f"{base_name}_step3_nlp_ready.json")
            
            self.logger.info(f"📝 Creating NLP-ready JSON: {Path(nlp_ready_path).name}")
            
            # Create NLP-ready copy
            nlp_data = enhanced_data.copy()
            
            # Add NLP-ready metadata
            nlp_data['step3_metadata'] = {
                'version': '3.0',
                'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'nlp_ready': True,
                'non_informative_images_already_removed': True,
                'image_data_removed': True,
                'ai_descriptions_preserved': True,
                'web_context_preserved': True,
                'processing_notes': 'NLP-ready version with image data removed, AI descriptions and web context preserved'
            }
            
            # Thoroughly remove all image data while preserving AI analysis
            removed_images_count = 0
            
            # Define comprehensive keys to remove
            keys_to_strip = [
                "image",           # Main Docling image container with uri field
                "image_data",      # Alternative image data field
                "imageData",       # CamelCase variant
                "thumbnail",       # Thumbnail images
                "page_image",      # Page-level image references
                "pageImage",       # CamelCase page image
                "uri",            # Direct URI fields (including base64 data URIs)
                "data",           # Raw image data fields
                "base64",         # Explicit base64 fields
                "picture",        # Alternative picture containers
                "img",            # Short image references
                "imageUri",       # URI-specific image fields
                "image_uri",      # Snake case URI fields
            ]
            
            def remove_keys_recursive(obj: Any, keys_to_remove: List[str]) -> tuple[Any, int]:
                """Recursively remove any keys in keys_to_remove from nested dict/list structures."""
                removed = 0
                if isinstance(obj, dict):
                    new_dict = {}
                    for k, v in obj.items():
                        if k in keys_to_remove:
                            removed += 1
                            continue
                        sanitized_v, rem = remove_keys_recursive(v, keys_to_remove)
                        removed += rem
                        new_dict[k] = sanitized_v
                    return new_dict, removed
                if isinstance(obj, list):
                    new_list = []
                    for item in obj:
                        sanitized_item, rem = remove_keys_recursive(item, keys_to_remove)
                        removed += rem
                        new_list.append(sanitized_item)
                    return new_list, removed
                return obj, 0
            
            def clean_base64_data_uris(obj: Any) -> Any:
                """Recursively find and clean base64 data URIs from any string fields"""
                if isinstance(obj, dict):
                    cleaned_dict = {}
                    for k, v in obj.items():
                        cleaned_v = clean_base64_data_uris(v)
                        # Check if this is a data URI field and replace with placeholder
                        if isinstance(cleaned_v, str) and cleaned_v.startswith("data:image"):
                            cleaned_dict[k] = "data:image/placeholder;base64,REMOVED_FOR_NLP"
                        else:
                            cleaned_dict[k] = cleaned_v
                    return cleaned_dict
                elif isinstance(obj, list):
                    return [clean_base64_data_uris(item) for item in obj]
                else:
                    return obj
            
            if 'pictures' in nlp_data:
                nlp_ready_pictures = []
                for pic in nlp_data['pictures']:
                    # Remove nested image-related keys recursively
                    sanitized_pic, removed = remove_keys_recursive(pic, keys_to_strip)
                    removed_images_count += removed
                    
                    # Mark images as removed in AI analysis
                    if "ai_analysis" not in sanitized_pic:
                        sanitized_pic["ai_analysis"] = {
                            "description": "Image was removed - no AI description available",
                            "image_type": "REMOVED",
                            "will_replace_image": True,
                            "images_removed_in_step2": True,
                        }
                    else:
                        # Mark existing AI analysis to indicate images were removed
                        sanitized_pic["ai_analysis"]["images_removed_in_step2"] = True
                        sanitized_pic["ai_analysis"]["will_replace_image"] = True
                        
                        # Update description to note image removal if it doesn't mention it
                        current_desc = sanitized_pic["ai_analysis"].get("description", "")
                        if current_desc and "removed" not in current_desc.lower():
                            sanitized_pic["ai_analysis"]["description"] = f"{current_desc} (Original image data removed for NLP processing)"
                    
                    nlp_ready_pictures.append(sanitized_pic)
                
                nlp_data['pictures'] = nlp_ready_pictures
            
            # Remove root-level image containers
            root_keys_to_remove = [
                "resources", "page_images", "pageImages", "images", 
                "figures", "pictures_raw", "media", "assets",
                "embedded_images", "image_resources"
            ]
            for root_key in root_keys_to_remove:
                if root_key in nlp_data:
                    try:
                        del nlp_data[root_key]
                        self.logger.info(f"Removed root-level image container: {root_key}")
                    except Exception:
                        pass
            
            # Apply base64 cleaning to the entire data structure
            nlp_data = clean_base64_data_uris(nlp_data)
            
            # Update metadata with removal statistics
            nlp_data['step3_metadata'].update({
                'removed_images_count': removed_images_count,
                'base64_data_uris_removed': True,
                'image_data_thoroughly_cleaned': True,
                'removal_method': 'enhanced_recursive_removal'
            })
            
            # Save NLP-ready JSON
            with open(nlp_ready_path, 'w', encoding='utf-8') as f:
                json.dump(nlp_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ NLP-ready JSON created: {nlp_ready_path}")
            return nlp_ready_path
            
        except Exception as e:
            self.logger.error(f"Failed to create NLP-ready JSON: {e}")
            raise
    
    def validate_ai_provider(self) -> tuple[bool, str]:
        """
        Validate that the AI provider is properly configured and accessible
        
        Returns:
            tuple: (success, message)
        """
        try:
            success, message = self.api_manager.test_connection()
            return success, message
        except Exception as e:
            return False, f"❌ Validation failed: {str(e)}"
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current AI provider
        
        Returns:
            Dict: Provider information including rate limits and cost estimates
        """
        try:
            return {
                'provider': self.config.get('ai_provider', 'Unknown'),
                'model': self.config.get('model_name', 'Unknown'),
                'rate_limits': self.api_manager.get_rate_limits(),
                'test_connection': self.api_manager.test_connection()
            }
        except Exception as e:
            return {
                'provider': self.config.get('ai_provider', 'Unknown'),
                'model': self.config.get('model_name', 'Unknown'),
                'error': str(e)
            }