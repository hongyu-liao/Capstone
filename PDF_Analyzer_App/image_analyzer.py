import logging
from pathlib import Path
import json
import base64
import requests
import re
import time
from typing import List, Dict, Optional, Any
import os # Added for os.path.join

# Try to import duckduckgo_search, fallback gracefully
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logging.warning("duckduckgo_search not available. Web search will be disabled.")

from api_manager import APIManager

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
        
    def analyze_images_from_json(self, json_path: str, enable_web_search: bool = True) -> List[Dict]:
        """
        Analyze all images from a Docling-generated JSON file
        
        Args:
            json_path (str): Path to the JSON file
            enable_web_search (bool): Whether to perform web search for conceptual images
            
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
                result = self._analyze_single_image(pic_data, i+1, enable_web_search)
                
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
    
    def _analyze_single_image(self, pic_data: Dict, pic_number: int, enable_web_search: bool) -> Optional[Dict]:
        """
        Analyze a single image using the configured AI provider
        
        Args:
            pic_data (Dict): Picture data from JSON
            pic_number (int): Picture number for logging
            enable_web_search (bool): Whether to perform web search
            
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
        if not HAS_DDGS:
            self.logger.warning("DuckDuckGo search not available")
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
    
    def create_enhanced_json(self, original_json_path: str, analysis_results: List[Dict], 
                            output_dir: str, output_filename: str = None) -> str:
        """
        Create enhanced JSON with AI analysis results
        
        Args:
            original_json_path (str): Path to original JSON file
            analysis_results (List[Dict]): Image analysis results
            output_dir (str): Output directory
            output_filename (str): Optional custom filename (if None, derive from original)
            
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
            
            # Generate enhanced JSON path with step2 suffix
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

    def create_nlp_ready_version(self, enhanced_json_path: str, output_dir: str, output_filename: str = None) -> str:
        """
        Create NLP-ready JSON by removing image data and keeping only text descriptions
        
        Args:
            enhanced_json_path (str): Path to enhanced JSON file
            output_dir (str): Output directory
            output_filename (str): Optional custom filename (if None, derive from enhanced)
            
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
                # Extract base name from enhanced file, removing step2 suffix if present
                base_name = enhanced_path_obj.stem
                if base_name.endswith('_step2_enhanced'):
                    base_name = base_name[:-15]  # Remove '_step2_enhanced'
            
            # Generate NLP-ready JSON path with step3 suffix
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
            
            # Remove image data while preserving AI analysis
            if 'pictures' in nlp_data:
                for pic in nlp_data['pictures']:
                    # Remove binary image data
                    if 'image' in pic:
                        del pic['image']
                    
                    # Keep AI analysis and web context for NLP processing
                    # These contain the text descriptions that replace the images
            
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