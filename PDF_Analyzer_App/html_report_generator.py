"""
HTML Report Generator for PDF Analysis Results

This module creates comprehensive HTML reports for evaluating PDF extraction and 
image analysis effectiveness.
"""

import logging
import json
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Try to import Docling for text extraction HTML conversion
try:
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.base import ImageRefMode
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logging.warning("Docling not available. Text extraction HTML will use fallback method.")

class HTMLReportGenerator:
    """Generates HTML reports for PDF analysis evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_image_analysis_report(self, analysis_results: List[Dict], 
                                     base_filename: str, output_dir: str) -> str:
        """
        Generate HTML report showing all image analysis results
        
        Args:
            analysis_results (List[Dict]): Image analysis results
            base_filename (str): Base filename for output
            output_dir (str): Output directory
            
        Returns:
            str: Path to generated HTML file
        """
        try:
            # Generate output path
            html_path = Path(output_dir) / f"{base_filename}_step2_image_analysis_report.html"
            
            self.logger.info(f"📊 Generating image analysis HTML report: {html_path.name}")
            
            # Create HTML content
            html_content = self._create_image_analysis_html(analysis_results, base_filename)
            
            # Save HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ Image analysis report generated: {html_path}")
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate image analysis report: {e}")
            raise
    
    def generate_complete_evaluation_report(self, original_json_path: str, 
                                          analysis_results: List[Dict],
                                          base_filename: str, output_dir: str) -> str:
        """
        Generate complete HTML evaluation report combining text extraction and image analysis
        
        Args:
            original_json_path (str): Path to original Docling JSON
            analysis_results (List[Dict]): Image analysis results  
            base_filename (str): Base filename for output
            output_dir (str): Output directory
            
        Returns:
            str: Path to generated HTML file
        """
        try:
            # Generate output path
            html_path = Path(output_dir) / f"{base_filename}_complete_evaluation_report.html"
            
            self.logger.info(f"📋 Generating complete evaluation HTML report: {html_path.name}")
            
            # Create HTML content combining text and image analysis
            html_content = self._create_complete_evaluation_html(
                original_json_path, analysis_results, base_filename
            )
            
            # Save HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ Complete evaluation report generated: {html_path}")
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate complete evaluation report: {e}")
            raise
    
    def _create_image_analysis_html(self, analysis_results: List[Dict], base_filename: str) -> str:
        """Create HTML content for image analysis results"""
        
        # Generate statistics
        total_images = len(analysis_results)
        conceptual_count = sum(1 for r in analysis_results if r.get('image_type') == 'CONCEPTUAL')
        data_viz_count = sum(1 for r in analysis_results if r.get('image_type') == 'DATA_VISUALIZATION')
        web_search_count = sum(1 for r in analysis_results if 'web_context' in r)
        
        # Create timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate image analysis HTML sections
        image_sections = ""
        for i, result in enumerate(analysis_results, 1):
            image_sections += self._create_image_section_html(result, i)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Analysis Report - {base_filename}</title>
    <style>
        {self._get_image_analysis_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Image Analysis Report</h1>
            <h2>{base_filename}</h2>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <div class="summary">
            <h3>📈 Analysis Summary</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_images}</div>
                    <div class="stat-label">Total Images Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{conceptual_count}</div>
                    <div class="stat-label">Conceptual Images</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{data_viz_count}</div>
                    <div class="stat-label">Data Visualizations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{web_search_count}</div>
                    <div class="stat-label">Web Enhanced</div>
                </div>
            </div>
        </div>
        
        <div class="analysis-results">
            <h3>🖼️ Detailed Image Analysis</h3>
            {image_sections}
        </div>
        
        <footer>
            <p>Report generated by PDF Image Analyzer Pro</p>
        </footer>
    </div>
</body>
</html>
"""
        return html_content
    
    def _create_complete_evaluation_html(self, original_json_path: str, 
                                       analysis_results: List[Dict], 
                                       base_filename: str) -> str:
        """Create HTML content for complete evaluation report"""
        
        # Load and convert text extraction results
        text_html = self._convert_docling_json_to_html(original_json_path)
        
        # Create image analysis section
        image_analysis_html = self._create_image_analysis_section(analysis_results)
        
        # Generate statistics
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete PDF Evaluation Report - {base_filename}</title>
    <style>
        {self._get_complete_evaluation_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 Complete PDF Evaluation Report</h1>
            <h2>{base_filename}</h2>
            <p class="timestamp">Generated: {timestamp}</p>
            <div class="evaluation-note">
                <strong>📝 Evaluation Purpose:</strong> This report helps assess the effectiveness of PDF text extraction and image analysis. 
                Review the text content for completeness and accuracy, and examine image analysis results for proper recognition and description.
            </div>
        </header>
        
        <nav class="nav-tabs">
            <button class="tab-button active" onclick="showTab('text-extraction')">📄 Text Extraction</button>
            <button class="tab-button" onclick="showTab('image-analysis')">🖼️ Image Analysis</button>
            <button class="tab-button" onclick="showTab('comparison')">⚖️ Comparison</button>
        </nav>
        
        <div id="text-extraction" class="tab-content active">
            <h3>📄 Extracted Text Content</h3>
            <div class="evaluation-instructions">
                <p><strong>💡 Evaluation Tips:</strong></p>
                <ul>
                    <li>Check if all text from the original PDF is present</li>
                    <li>Verify proper formatting and structure preservation</li>
                    <li>Look for missing headers, paragraphs, or special content</li>
                    <li>Ensure mathematical formulas and citations are captured</li>
                </ul>
            </div>
            <div class="text-content">
                {text_html}
            </div>
        </div>
        
        <div id="image-analysis" class="tab-content">
            <h3>🖼️ Image Recognition Results</h3>
            <div class="evaluation-instructions">
                <p><strong>💡 Evaluation Tips:</strong></p>
                <ul>
                    <li>Verify images are correctly classified (data visualization vs conceptual)</li>
                    <li>Check if AI descriptions accurately describe image content</li>
                    <li>Ensure logos and decorative elements are properly filtered out</li>
                    <li>Review web search enhancements for conceptual images</li>
                </ul>
            </div>
            {image_analysis_html}
        </div>
        
        <div id="comparison" class="tab-content">
            <h3>⚖️ Extraction vs Recognition Comparison</h3>
            <div class="comparison-section">
                <p>This section helps you compare the original PDF against the AI processing results:</p>
                
                <div class="comparison-cards">
                    <div class="comparison-card">
                        <h4>📄 Text Extraction Quality</h4>
                        <div class="checklist">
                            <label><input type="checkbox"> All paragraphs captured</label>
                            <label><input type="checkbox"> Headers and formatting preserved</label>
                            <label><input type="checkbox"> Tables and lists accurate</label>
                            <label><input type="checkbox"> Special characters correct</label>
                            <label><input type="checkbox"> Citations and references intact</label>
                        </div>
                    </div>
                    
                    <div class="comparison-card">
                        <h4>🖼️ Image Analysis Quality</h4>
                        <div class="checklist">
                            <label><input type="checkbox"> All meaningful images detected</label>
                            <label><input type="checkbox"> Image types correctly classified</label>
                            <label><input type="checkbox"> Descriptions are accurate and detailed</label>
                            <label><input type="checkbox"> Logos/decorative elements filtered</label>
                            <label><input type="checkbox"> Web context enhances understanding</label>
                        </div>
                    </div>
                </div>
                
                <div class="overall-assessment">
                    <h4>📊 Overall Assessment</h4>
                    <textarea placeholder="Write your evaluation notes here...&#10;&#10;Text Extraction:&#10;- &#10;&#10;Image Analysis:&#10;- &#10;&#10;Recommendations:&#10;- " rows="10" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;"></textarea>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Report generated by PDF Image Analyzer Pro - Evaluation Mode</p>
        </footer>
    </div>
    
    <script>
        {self._get_tab_script()}
    </script>
</body>
</html>
"""
        return html_content
    
    def _convert_docling_json_to_html(self, json_path: str) -> str:
        """Convert Docling JSON to HTML using the same method as json2html.ipynb"""
        try:
            if HAS_DOCLING:
                # Use Docling's built-in HTML conversion (preferred method)
                with open(json_path, "r", encoding="utf-8") as f:
                    json_str = f.read()
                
                document = DoclingDocument.model_validate_json(json_str)
                
                # Convert to HTML string (similar to save_as_html but return string)
                # This is a simplified version - in practice you'd use document.export_to_html()
                return self._docling_to_html_string(document)
            else:
                # Fallback: simple JSON to HTML conversion
                return self._simple_json_to_html(json_path)
                
        except Exception as e:
            self.logger.warning(f"Docling HTML conversion failed: {e}, using fallback")
            return self._simple_json_to_html(json_path)
    
    def _docling_to_html_string(self, document) -> str:
        """Convert DoclingDocument to HTML string"""
        try:
            # Extract main text content from Docling document
            main_text = ""
            
            # Get body content if available
            if hasattr(document, 'body') and document.body:
                for element in document.body:
                    if hasattr(element, 'text') and element.text:
                        main_text += element.text + "\n\n"
            
            # Fallback to raw text if body parsing fails
            if not main_text and hasattr(document, 'text'):
                main_text = document.text
            
            # Convert to formatted HTML
            formatted_text = main_text.replace('\n\n', '</p><p>').replace('\n', '<br>')
            return f"<p>{formatted_text}</p>"
            
        except Exception as e:
            self.logger.error(f"Failed to convert Docling document to HTML: {e}")
            return "<p>Error converting document content to HTML</p>"
    
    def _simple_json_to_html(self, json_path: str) -> str:
        """Simple fallback JSON to HTML conversion"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract main text content
            main_text = ""
            
            # Try different possible text fields
            text_fields = ['main_text', 'text', 'content', 'body']
            for field in text_fields:
                if field in data and data[field]:
                    main_text = data[field]
                    break
            
            # If no direct text field, try to extract from body/content structure
            if not main_text and 'body' in data:
                body_texts = []
                if isinstance(data['body'], list):
                    for item in data['body']:
                        if isinstance(item, dict) and 'text' in item:
                            body_texts.append(item['text'])
                        elif isinstance(item, str):
                            body_texts.append(item)
                main_text = '\n\n'.join(body_texts)
            
            # Format as HTML
            if main_text:
                formatted_text = main_text.replace('\n\n', '</p><p>').replace('\n', '<br>')
                return f"<p>{formatted_text}</p>"
            else:
                return "<p>No text content found in JSON file</p>"
                
        except Exception as e:
            self.logger.error(f"Failed to convert JSON to HTML: {e}")
            return f"<p>Error loading text content: {e}</p>"
    
    def _create_image_section_html(self, result: Dict, image_num: int) -> str:
        """Create HTML section for a single image analysis result"""
        
        # Get image data
        image_data = result.get('image_data', '')
        image_format = result.get('image_format', 'png')
        image_type = result.get('image_type', 'UNKNOWN')
        description = result.get('detailed_description', 'No description available')
        original_caption = result.get('original_caption', 'No caption')
        
        # Get web context if available
        web_context_html = ""
        if 'web_context' in result:
            web_context_html = self._create_web_context_html(result['web_context'])
        
        # Create image display
        image_html = ""
        if image_data:
            image_src = f"data:image/{image_format};base64,{image_data}"
            image_html = f"""
            <div class="image-display">
                <img src="{image_src}" alt="Image {image_num}" onclick="enlargeImage(this)">
                <p class="image-info">Format: {image_format.upper()} | Click to enlarge</p>
            </div>
            """
        
        return f"""
        <div class="image-analysis">
            <h4>📸 Image {image_num} - {image_type}</h4>
            <div class="image-content">
                {image_html}
                <div class="analysis-details">
                    <div class="detail-section">
                        <h5>📝 Original Caption</h5>
                        <p class="caption">{original_caption}</p>
                    </div>
                    
                    <div class="detail-section">
                        <h5>🤖 AI Analysis</h5>
                        <p class="ai-description">{description}</p>
                    </div>
                    
                    {web_context_html}
                </div>
            </div>
        </div>
        """
    
    def _create_web_context_html(self, web_context: Dict) -> str:
        """Create HTML for web search context"""
        search_method = web_context.get('search_method', 'unknown')
        
        if search_method == 'native':
            return self._create_native_search_html(web_context)
        else:
            return self._create_duckduckgo_search_html(web_context)
    
    def _create_native_search_html(self, web_context: Dict) -> str:
        """Create HTML for native web search results"""
        search_queries = web_context.get('search_queries', [])
        sources = web_context.get('sources', [])
        
        sources_html = ""
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', '#')
            sources_html += f'<li><a href="{url}" target="_blank">{title}</a></li>'
        
        return f"""
        <div class="detail-section web-context">
            <h5>🌐 Native Web Search Enhancement</h5>
            <p><strong>Search Queries:</strong> {', '.join(search_queries)}</p>
            <p><strong>Sources Used:</strong></p>
            <ul class="sources-list">{sources_html}</ul>
            <p class="note">Web search results are integrated directly into the AI analysis above.</p>
        </div>
        """
    
    def _create_duckduckgo_search_html(self, web_context: Dict) -> str:
        """Create HTML for DuckDuckGo search results"""
        keywords = web_context.get('search_keywords', [])
        ai_summary = web_context.get('ai_summary', 'No summary available')
        sources = web_context.get('sources', [])
        
        sources_html = ""
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', '#')
            body = source.get('body', '')[:150] + "..." if len(source.get('body', '')) > 150 else source.get('body', '')
            sources_html += f"""
            <li>
                <a href="{url}" target="_blank">{title}</a>
                <p class="source-snippet">{body}</p>
            </li>
            """
        
        return f"""
        <div class="detail-section web-context">
            <h5>🦆 DuckDuckGo Search Context</h5>
            <p><strong>Keywords:</strong> {', '.join(keywords)}</p>
            <div class="ai-summary">
                <h6>AI Summary:</h6>
                <p>{ai_summary}</p>
            </div>
            <details>
                <summary>📚 View Sources ({len(sources)})</summary>
                <ul class="sources-list">{sources_html}</ul>
            </details>
        </div>
        """
    
    def _create_image_analysis_section(self, analysis_results: List[Dict]) -> str:
        """Create image analysis section for complete report"""
        sections = ""
        for i, result in enumerate(analysis_results, 1):
            sections += self._create_image_section_html(result, i)
        
        return f"""
        <div class="image-analysis-container">
            {sections}
        </div>
        """
    
    def _get_image_analysis_css(self) -> str:
        """CSS styles for image analysis report"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #007bff;
        }
        
        h1 {
            color: #007bff;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #6c757d;
            margin-top: 0;
        }
        
        .timestamp {
            color: #6c757d;
            font-style: italic;
        }
        
        .summary {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        
        .image-analysis {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 30px;
            overflow: hidden;
        }
        
        .image-analysis h4 {
            background: #007bff;
            color: white;
            margin: 0;
            padding: 15px;
        }
        
        .image-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            padding: 20px;
        }
        
        .image-display img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .image-display img:hover {
            transform: scale(1.05);
        }
        
        .image-info {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 10px;
        }
        
        .detail-section {
            margin-bottom: 20px;
        }
        
        .detail-section h5 {
            color: #495057;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        
        .caption {
            font-style: italic;
            color: #6c757d;
        }
        
        .ai-description {
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }
        
        .web-context {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #2196f3;
        }
        
        .sources-list {
            margin: 10px 0;
        }
        
        .sources-list li {
            margin-bottom: 10px;
        }
        
        .sources-list a {
            color: #007bff;
            text-decoration: none;
        }
        
        .sources-list a:hover {
            text-decoration: underline;
        }
        
        .source-snippet {
            font-size: 0.9em;
            color: #6c757d;
            margin: 5px 0 0 20px;
        }
        
        .ai-summary {
            background: white;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .note {
            font-size: 0.9em;
            color: #6c757d;
            font-style: italic;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #6c757d;
        }
        
        @media (max-width: 768px) {
            .image-content {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_complete_evaluation_css(self) -> str:
        """CSS styles for complete evaluation report"""
        return self._get_image_analysis_css() + """
        
        .evaluation-note {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            color: #856404;
        }
        
        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-radius: 8px 8px 0 0;
            margin-top: 20px;
        }
        
        .tab-button {
            flex: 1;
            padding: 15px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        
        .tab-button:hover {
            background: #e9ecef;
        }
        
        .tab-button.active {
            background: #007bff;
            color: white;
        }
        
        .tab-content {
            display: none;
            padding: 30px;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 8px 8px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .evaluation-instructions {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .evaluation-instructions ul {
            margin: 10px 0 0 0;
        }
        
        .text-content {
            background: white;
            padding: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .comparison-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        
        .comparison-cards {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .comparison-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        
        .comparison-card h4 {
            color: #007bff;
            margin-top: 0;
        }
        
        .checklist {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .checklist label {
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
        }
        
        .checklist input[type="checkbox"] {
            transform: scale(1.2);
        }
        
        .overall-assessment {
            margin-top: 30px;
        }
        
        .overall-assessment h4 {
            color: #007bff;
        }
        
        @media (max-width: 768px) {
            .comparison-cards {
                grid-template-columns: 1fr;
            }
            
            .nav-tabs {
                flex-direction: column;
            }
        }
        """
    
    def _get_tab_script(self) -> str:
        """JavaScript for tab functionality"""
        return """
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Set active button
            event.target.classList.add('active');
        }
        
        function enlargeImage(img) {
            // Create overlay
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                cursor: pointer;
            `;
            
            // Create enlarged image
            const enlargedImg = document.createElement('img');
            enlargedImg.src = img.src;
            enlargedImg.style.cssText = `
                max-width: 90%;
                max-height: 90%;
                object-fit: contain;
            `;
            
            overlay.appendChild(enlargedImg);
            document.body.appendChild(overlay);
            
            // Close on click
            overlay.onclick = () => document.body.removeChild(overlay);
        }
        """