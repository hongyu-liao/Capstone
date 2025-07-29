import logging
import requests
import json
import base64
from typing import Dict, Any, Optional, List
import time
from pathlib import Path

# Try to import Google GenAI for enhanced Gemini features
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    logging.warning("google-genai not available. Gemini will use basic HTTP API without web search.")

class APIManager:
    """Manages API calls to different AI providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize API manager
        
        Args:
            config (Dict): Configuration containing provider, model, and API key
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.provider = config.get('ai_provider', 'LM Studio (Local)')
        self.model_name = config.get('model_name')
        self.api_key = config.get('api_key')
        self.base_url = config.get('lm_studio_url')
        
        # Initialize Google GenAI client if available
        self.genai_client = None
        if self.provider == "Google (Gemini)" and HAS_GOOGLE_GENAI and self.api_key:
            try:
                self.genai_client = genai.Client(api_key=self.api_key)
                self.logger.info("🔗 Initialized Google GenAI client with web search support")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google GenAI client: {e}")
                self.genai_client = None
        
        # Validate and normalize model name
        if self.provider == "Google (Gemini)":
            self.model_name = self._normalize_gemini_model_name(self.model_name)
    
    def _normalize_gemini_model_name(self, model_name: str) -> str:
        """
        Normalize Gemini model name to remove models/ prefix if present
        
        Args:
            model_name (str): Raw model name
            
        Returns:
            str: Normalized model name
        """
        if not model_name:
            return model_name
        
        # Remove models/ prefix if present
        if model_name.startswith("models/"):
            normalized = model_name[7:]  # Remove "models/" prefix
            self.logger.info(f"Normalized Gemini model name from '{model_name}' to '{normalized}'")
            return normalized
        
        return model_name
    
    def _validate_model_exists(self) -> tuple[bool, str]:
        """
        Validate if the specified model exists for the provider
        
        Returns:
            tuple: (exists, error_message)
        """
        known_models = {
            "OpenAI (ChatGPT)": [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
            ],
            "Google (Gemini)": [
                "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", 
                "gemini-2.0-flash", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"
            ],
            "Anthropic (Claude)": [
                "claude-4-sonnet-latest", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-latest",
                "claude-3-5-haiku-20241022", "claude-3-opus-20240229"
            ]
        }
        
        if self.provider == "LM Studio (Local)":
            return True, ""  # Local models are not validated
        
        provider_models = known_models.get(self.provider, [])
        if self.model_name not in provider_models:
            return False, f"Model '{self.model_name}' not found for {self.provider}. Please check the model name."
        
        return True, ""
    
    def supports_native_web_search(self) -> bool:
        """
        Check if the current provider supports native web search
        
        Returns:
            bool: True if native web search is supported
        """
        if self.provider == "Google (Gemini)" and self.genai_client:
            return True
        elif self.provider == "OpenAI (ChatGPT)":
            # ChatGPT with web browsing (only certain models and tiers)
            return self.model_name in ["gpt-4o", "gpt-4-turbo"] and self.api_key
        else:
            return False
    
    def analyze_image_with_web_search(self, image_uri: str, prompt: str, search_query: str = None, max_tokens: int = 600) -> Optional[Dict]:
        """
        Analyze image with built-in web search capabilities
        
        Args:
            image_uri (str): Data URI of the image
            prompt (str): Analysis prompt
            search_query (str): Optional search query for web enhancement
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            Optional[Dict]: Result with 'response' and optionally 'web_sources'
        """
        if not self.supports_native_web_search():
            # Fallback to regular analysis
            response = self.analyze_image(image_uri, prompt, max_tokens)
            return {'response': response, 'web_sources': None} if response else None
        
        try:
            if self.provider == "Google (Gemini)":
                return self._gemini_analyze_with_web_search(image_uri, prompt, search_query, max_tokens)
            elif self.provider == "OpenAI (ChatGPT)":
                return self._openai_analyze_with_web_search(image_uri, prompt, search_query, max_tokens)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Web search analysis failed for {self.provider}: {e}")
            # Fallback to regular analysis
            response = self.analyze_image(image_uri, prompt, max_tokens)
            return {'response': response, 'web_sources': None} if response else None
    
    def _gemini_analyze_with_web_search(self, image_uri: str, prompt: str, search_query: str, max_tokens: int) -> Optional[Dict]:
        """Analyze image with Gemini's native web search using specified search keywords"""
        if not self.genai_client:
            return None
        
        try:
            # Extract base64 data for Gemini
            if image_uri.startswith("data:image"):
                mime_type = image_uri.split(";")[0].split(":")[1]
                base64_data = image_uri.split(",")[1]
            else:
                mime_type = "image/png"
                base64_data = image_uri
            
            # If specific search keywords are provided, guide the search more precisely
            if search_query and search_query.strip():
                self.logger.info(f"🎯 Directing Gemini to search for: {search_query}")
                guided_prompt = f"""{prompt}

IMPORTANT: Focus your web search on these specific topics: {search_query}

Search the web for current, relevant information about these concepts to enhance your analysis."""
            else:
                # Fallback to general web search guidance
                guided_prompt = f"""{prompt}

Additionally, if this image shows concepts, methodologies, or geographic locations that would benefit from current context or background information, please search the web to provide relevant additional information that enhances understanding of the image content."""
            
            # Create content with image
            image_part = types.Part.from_bytes(
                data=base64.b64decode(base64_data),
                mime_type=mime_type
            )
            
            # Define grounding tool for web search
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            # Configure generation with web search
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.1,
                max_output_tokens=max_tokens
            )
            
            # Make the request
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=[guided_prompt, image_part],
                config=config
            )
            
            # Extract web sources if available - using correct structure from official docs
            web_sources = []
            web_search_queries = []
            
            # Parse the response to extract grounding metadata
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check for groundingMetadata in the candidate
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_metadata = candidate.grounding_metadata
                    
                    # Extract web search queries that were actually performed
                    if hasattr(grounding_metadata, 'web_search_queries'):
                        web_search_queries = list(grounding_metadata.web_search_queries)
                    
                    # Extract grounding chunks (web sources)
                    if hasattr(grounding_metadata, 'grounding_chunks'):
                        for chunk in grounding_metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                web_source = {
                                    'title': getattr(chunk.web, 'title', 'Unknown'),
                                    'url': getattr(chunk.web, 'uri', ''),
                                    'snippet': ''  # Gemini doesn't provide snippets in this structure
                                }
                                if web_source['url']:  # Only add if we have a URL
                                    web_sources.append(web_source)
                    
                    # Extract grounding supports for additional context
                    grounding_supports = []
                    if hasattr(grounding_metadata, 'grounding_supports'):
                        for support in grounding_metadata.grounding_supports:
                            if hasattr(support, 'segment') and support.segment:
                                grounding_supports.append({
                                    'text': getattr(support.segment, 'text', ''),
                                    'start_index': getattr(support.segment, 'start_index', 0),
                                    'end_index': getattr(support.segment, 'end_index', 0),
                                    'chunk_indices': list(getattr(support, 'grounding_chunk_indices', []))
                                })
            
            # Log search results
            if search_query:
                self.logger.info(f"✅ Gemini targeted search completed. Keywords: '{search_query}' → Queries: {len(web_search_queries)}, Sources: {len(web_sources)}")
            else:
                self.logger.info(f"✅ Gemini general search completed. Queries: {len(web_search_queries)}, Sources: {len(web_sources)}")
            
            # Log the queries that were performed for debugging
            if web_search_queries:
                self.logger.debug(f"🔍 Gemini performed searches: {web_search_queries}")
            
            return {
                'response': response.text,
                'web_sources': web_sources if web_sources else None,
                'search_queries': web_search_queries,
                'grounding_supports': grounding_supports if 'grounding_supports' in locals() else None,
                'guided_keywords': search_query if search_query else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gemini web search analysis failed: {e}")
            self.logger.error(f"   Error details: {type(e).__name__}: {str(e)}")
            
            # Try to fallback to regular Gemini analysis without web search
            try:
                self.logger.info("🔄 Falling back to regular Gemini analysis without web search")
                regular_response = self._call_gemini(image_uri, prompt, max_tokens)
                if regular_response:
                    return {
                        'response': regular_response,
                        'web_sources': None,
                        'search_queries': [],
                        'fallback_used': True
                    }
            except Exception as fallback_error:
                self.logger.error(f"❌ Gemini fallback also failed: {fallback_error}")
            
            raise  # Re-raise original error if both attempts fail
    
    def _openai_analyze_with_web_search(self, image_uri: str, prompt: str, search_query: str, max_tokens: int) -> Optional[Dict]:
        """Analyze image with OpenAI's web browsing (if available)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Enhanced prompt for web-enabled analysis
            enhanced_prompt = f"""{prompt}

Additionally, if this image shows concepts, methodologies, or topics that would benefit from current information or background context, please use web browsing to provide relevant additional information that enhances understanding of the image content."""
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_uri}
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "tools": [
                    {
                        "type": "browser"  # Enable web browsing if available
                    }
                ]
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response and any web sources
            message_content = result["choices"][0]["message"]["content"]
            
            # Note: OpenAI's web browsing integration may vary
            # This is a placeholder for when the feature becomes available
            web_sources = None
            
            self.logger.info("✅ OpenAI web browsing analysis completed")
            
            return {
                'response': message_content,
                'web_sources': web_sources
            }
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI web browsing analysis failed: {e}")
            # Fallback to regular analysis without web browsing
            response = self._call_openai(image_uri, prompt, max_tokens)
            return {'response': response, 'web_sources': None} if response else None
        
    def analyze_image(self, image_uri: str, prompt: str, max_tokens: int = 600) -> Optional[str]:
        """
        Analyze image using the configured AI provider
        
        Args:
            image_uri (str): Data URI of the image
            prompt (str): Analysis prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            Optional[str]: AI response or None if failed
        """
        # Validate model first
        if self.provider != "LM Studio (Local)":
            is_valid, error_msg = self._validate_model_exists()
            if not is_valid:
                self.logger.error(f"Model validation failed: {error_msg}")
                raise ValueError(error_msg)
        
        try:
            self.logger.info(f"🤖 Calling {self.provider} with model: {self.model_name}")
            
            if self.provider == "LM Studio (Local)":
                return self._call_lm_studio(image_uri, prompt, max_tokens)
            elif self.provider == "OpenAI (ChatGPT)":
                return self._call_openai(image_uri, prompt, max_tokens)
            elif self.provider == "Google (Gemini)":
                return self._call_gemini(image_uri, prompt, max_tokens)
            elif self.provider == "Anthropic (Claude)":
                return self._call_claude(image_uri, prompt, max_tokens)
            else:
                self.logger.error(f"Unsupported AI provider: {self.provider}")
                return None
                
        except Exception as e:
            self.logger.error(f"API call failed for {self.provider} ({self.model_name}): {e}")
            raise  # Re-raise to allow upper level handling
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Generate text using the configured AI provider (for web search summaries)
        
        Args:
            prompt (str): Text generation prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            Optional[str]: AI response or None if failed
        """
        try:
            if self.provider == "LM Studio (Local)":
                return self._call_lm_studio_text(prompt, max_tokens)
            elif self.provider == "OpenAI (ChatGPT)":
                return self._call_openai_text(prompt, max_tokens)
            elif self.provider == "Google (Gemini)":
                return self._call_gemini_text(prompt, max_tokens)
            elif self.provider == "Anthropic (Claude)":
                return self._call_claude_text(prompt, max_tokens)
            else:
                self.logger.error(f"Unsupported AI provider: {self.provider}")
                return None
                
        except Exception as e:
            self.logger.error(f"Text generation failed for {self.provider}: {e}")
            return None
    
    def _call_lm_studio(self, image_uri: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Call LM Studio API for image analysis"""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(self.base_url, json=payload, timeout=300)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_lm_studio_text(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call LM Studio API for text generation"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        response = requests.post(self.base_url, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_openai(self, image_uri: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Call OpenAI API for image analysis"""
        # Convert data URI to base64 for OpenAI
        if image_uri.startswith("data:image"):
            # Extract base64 part
            base64_data = image_uri.split(",")[1]
        else:
            base64_data = image_uri
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_uri}
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_openai_text(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call OpenAI API for text generation"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name if self.model_name.startswith("gpt") else "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_gemini(self, image_uri: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Google Gemini API for image analysis"""
        try:
            # Extract base64 data
            if image_uri.startswith("data:image"):
                mime_type = image_uri.split(";")[0].split(":")[1]
                base64_data = image_uri.split(",")[1]
            else:
                mime_type = "image/png"
                base64_data = image_uri
            
            # Build URL - ensure model name is properly formatted
            url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            self.logger.debug(f"🔗 Gemini API URL: {url}")
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.1
                }
            }
            
            self.logger.debug(f"📤 Gemini request payload keys: {list(payload.keys())}")
            response = requests.post(url, json=payload, timeout=300)
            
            # Enhanced error handling
            if response.status_code != 200:
                error_text = response.text
                self.logger.error(f"❌ Gemini API error (status {response.status_code}): {error_text}")
                
                if "404" in str(response.status_code):
                    raise ValueError(f"Model '{self.model_name}' not found. Please check the model name.")
                elif "400" in str(response.status_code):
                    raise ValueError(f"Invalid request to Gemini API: {error_text}")
                else:
                    response.raise_for_status()
            
            result = response.json()
            self.logger.debug(f"📥 Gemini response keys: {list(result.keys())}")
            
            # Check if response has expected structure
            if "candidates" not in result:
                self.logger.error(f"❌ Unexpected Gemini response structure: {result}")
                raise ValueError("Invalid response from Gemini API - no candidates found")
            
            if not result["candidates"]:
                self.logger.error("❌ Gemini returned empty candidates list")
                raise ValueError("Gemini API returned no candidates")
            
            candidate = result["candidates"][0]
            if "content" not in candidate:
                self.logger.error(f"❌ Gemini candidate missing content: {candidate}")
                raise ValueError("Gemini candidate missing content")
            
            content = candidate["content"]
            if "parts" not in content or not content["parts"]:
                self.logger.error(f"❌ Gemini content missing parts: {content}")
                raise ValueError("Gemini content missing parts")
            
            response_text = content["parts"][0]["text"]
            self.logger.debug(f"✅ Gemini response: {response_text[:100]}...")
            
            return response_text
            
        except requests.RequestException as e:
            self.logger.error(f"❌ Gemini network error: {e}")
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"❌ Gemini response parsing error: {e}")
            raise ValueError(f"Failed to parse Gemini response: {e}")
        except Exception as e:
            self.logger.error(f"❌ Gemini unexpected error: {e}")
            raise
    
    def _call_gemini_text(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Google Gemini API for text generation"""
        try:
            # Use text-only model for text generation
            text_model = "gemini-1.5-flash-latest" if "vision" in self.model_name else self.model_name
            url = f"{self.base_url}/{text_model}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.3
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            self.logger.error(f"❌ Gemini text generation error: {e}")
            raise
    
    def _call_claude(self, image_uri: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Anthropic Claude API for image analysis"""
        # Extract base64 data and media type
        if image_uri.startswith("data:image"):
            media_type = image_uri.split(";")[0].split(":")[1]
            base64_data = image_uri.split(",")[1]
        else:
            media_type = "image/png"
            base64_data = image_uri
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def _call_claude_text(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Anthropic Claude API for text generation"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test connection to the AI provider
        
        Returns:
            tuple: (success, message)
        """
        try:
            # Validate model first
            if self.provider != "LM Studio (Local)":
                is_valid, error_msg = self._validate_model_exists()
                if not is_valid:
                    return False, f"❌ {error_msg}"
            
            test_prompt = "Hello! Please respond with 'Connection successful' to confirm the API is working."
            
            response = self.generate_text(test_prompt, max_tokens=50)
            
            if response:
                web_search_status = ""
                
                # Special check for Gemini web search capability
                if self.provider == "Google (Gemini)":
                    if HAS_GOOGLE_GENAI and self.genai_client:
                        web_search_status = "with native Google Search ✨"
                        self.logger.info("🔗 Gemini client initialized with web search capabilities")
                    elif HAS_GOOGLE_GENAI:
                        web_search_status = "Google GenAI available but client failed to initialize"
                        self.logger.warning("⚠️ Google GenAI SDK available but client initialization failed")
                    else:
                        web_search_status = "without web search (install google-genai package)"
                        self.logger.warning("⚠️ google-genai package not available")
                elif self.supports_native_web_search():
                    web_search_status = "with native web search"
                else:
                    web_search_status = "with DuckDuckGo fallback search"
                
                return True, f"✅ {self.provider} connection successful ({web_search_status})"
            else:
                return False, f"❌ {self.provider} returned empty response"
                
        except Exception as e:
            return False, f"❌ {self.provider} connection failed: {str(e)}"
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limit information for the provider
        
        Returns:
            Dict: Rate limit information
        """
        rate_limits = {
            "LM Studio (Local)": {
                "requests_per_minute": "Unlimited (local)",
                "tokens_per_minute": "Depends on hardware",
                "concurrent_requests": "1-4 (typical)"
            },
            "OpenAI (ChatGPT)": {
                "requests_per_minute": "500-10000 (tier dependent)",
                "tokens_per_minute": "30K-2M (tier dependent)",
                "concurrent_requests": "100-1000"
            },
            "Google (Gemini)": {
                "requests_per_minute": "60-1500 (tier dependent)",
                "tokens_per_minute": "2M-4M (model dependent)",
                "concurrent_requests": "1000"
            },
            "Anthropic (Claude)": {
                "requests_per_minute": "50-1000 (tier dependent)",
                "tokens_per_minute": "40K-400K (tier dependent)",
                "concurrent_requests": "5-100"
            }
        }
        
        return rate_limits.get(self.provider, {
            "requests_per_minute": "Unknown",
            "tokens_per_minute": "Unknown",
            "concurrent_requests": "Unknown"
        })