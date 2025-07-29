"""
Configuration settings for PDF Image Analyzer
"""

import os
from pathlib import Path

class Config:
    """Configuration class for PDF Image Analyzer"""
    
    # Default LM Studio settings
    DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    DEFAULT_MODEL_NAME = "google/gemma-3-12b-it-gguf"
    
    # Processing settings
    DEFAULT_MAX_TOKENS = 600
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TIMEOUT = 300  # seconds
    
    # Output settings
    DEFAULT_OUTPUT_DIR = "output"
    
    # Web search settings
    DEFAULT_SEARCH_RESULTS = 5
    DEFAULT_SEARCH_TIMEOUT = 120  # seconds
    
    # File size limits (in MB)
    MAX_PDF_SIZE_MB = 100
    
    # Supported file formats
    SUPPORTED_PDF_EXTENSIONS = ['.pdf']
    SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'gif', 'bmp']
    
    @classmethod
    def get_default_config(cls):
        """Get default configuration dictionary"""
        return {
            'lm_studio_url': cls.DEFAULT_LM_STUDIO_URL,
            'model_name': cls.DEFAULT_MODEL_NAME,
            'max_tokens': cls.DEFAULT_MAX_TOKENS,
            'temperature': cls.DEFAULT_TEMPERATURE,
            'timeout': cls.DEFAULT_TIMEOUT,
            'output_dir': cls.DEFAULT_OUTPUT_DIR,
            'search_results': cls.DEFAULT_SEARCH_RESULTS,
            'search_timeout': cls.DEFAULT_SEARCH_TIMEOUT,
            'enable_web_search': True,
            'generate_nlp_ready': False
        }
    
    @classmethod
    def validate_config(cls, config):
        """
        Validate configuration settings
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            tuple: (is_valid, error_message)
        """
        required_keys = ['lm_studio_url', 'model_name']
        
        for key in required_keys:
            if key not in config or not config[key]:
                return False, f"Missing required configuration: {key}"
        
        # Validate URL format
        if not config['lm_studio_url'].startswith(('http://', 'https://')):
            return False, "LM Studio URL must start with http:// or https://"
        
        # Validate numeric values
        numeric_fields = {
            'max_tokens': (10, 2000),
            'temperature': (0.0, 2.0),
            'timeout': (10, 600)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                try:
                    value = float(config[field])
                    if not (min_val <= value <= max_val):
                        return False, f"{field} must be between {min_val} and {max_val}"
                except (ValueError, TypeError):
                    return False, f"{field} must be a valid number"
        
        return True, "Configuration is valid"
    
    @classmethod
    def create_output_directory(cls, output_dir):
        """
        Create output directory if it doesn't exist
        
        Args:
            output_dir (str): Output directory path
            
        Returns:
            bool: True if successful
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    @classmethod
    def get_environment_config(cls):
        """
        Get configuration from environment variables
        
        Returns:
            dict: Configuration from environment variables
        """
        env_config = {}
        
        # Map environment variables to config keys
        env_mapping = {
            'LM_STUDIO_URL': 'lm_studio_url',
            'LM_STUDIO_MODEL': 'model_name',
            'MAX_TOKENS': 'max_tokens',
            'OUTPUT_DIR': 'output_dir'
        }
        
        for env_key, config_key in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value:
                # Convert numeric values
                if config_key in ['max_tokens', 'temperature', 'timeout']:
                    try:
                        env_config[config_key] = float(env_value)
                    except ValueError:
                        pass  # Skip invalid values
                else:
                    env_config[config_key] = env_value
        
        return env_config

# Application metadata
APP_INFO = {
    'name': 'PDF Image Analyzer',
    'version': '1.0.0',
    'description': 'AI-powered PDF image extraction and analysis tool',
    'author': 'PDF Analyzer Team',
    'license': 'MIT'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}