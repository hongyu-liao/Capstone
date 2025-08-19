#!/usr/bin/env python3
"""
ChartGemma Test Script for Academic Chart Analysis
Based on https://huggingface.co/ahmed-masry/chartgemma

This script analyzes charts in the Sample Line Chart folder using ChartGemma model
with specialized prompts for different chart types commonly found in academic papers.
"""

import os
import sys
from pathlib import Path
import json
import time
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartGemmaProcessor:
    def __init__(self):
        """Initialize ChartGemma processor"""
        self.model = None
        self.processor = None
        self.device = None
        
    def load_model(self):
        """Load ChartGemma model"""
        try:
            logger.info("🤖 Loading ChartGemma model...")
            
            # Check CUDA availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load model and processor
            logger.info("📥 Downloading model files...")
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                "ahmed-masry/chartgemma", 
                torch_dtype=torch.float16
            )
            self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
            
            # Move to device
            self.model = self.model.to(self.device)
            logger.info("✅ ChartGemma model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def analyze_chart(self, image_path: str, question: str) -> str:
        """
        Analyze chart image
        
        Args:
            image_path: Path to image file
            question: Question to ask about the chart
            
        Returns:
            Model's response
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"📊 Analyzing chart: {Path(image_path).name}")
            
            # Process inputs
            inputs = self.processor(text=question, images=image, return_tensors="pt")
            prompt_length = inputs['input_ids'].shape[1]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs, 
                    num_beams=4, 
                    max_new_tokens=512,
                    do_sample=False  # Ensure consistent results
                )
            
            # Decode output
            output_text = self.processor.batch_decode(
                generate_ids[:, prompt_length:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Chart analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    def detect_chart_type(self, image_path: str) -> str:
        """
        Detect the type of chart to apply appropriate questions
        
        Args:
            image_path: Path to image file
            
        Returns:
            Detected chart type
        """
        try:
            detection_question = "What type of chart is this? Answer with one of: line chart, bar chart, pie chart, scatter plot, histogram, box plot, area chart, or other."
            chart_type_response = self.analyze_chart(image_path, detection_question)
            
            # Simple keyword matching to determine chart type
            response_lower = chart_type_response.lower()
            if any(keyword in response_lower for keyword in ['line', 'trend', 'time series']):
                return "line_chart"
            elif any(keyword in response_lower for keyword in ['bar', 'column', 'vertical', 'horizontal']):
                return "bar_chart" 
            elif any(keyword in response_lower for keyword in ['pie', 'circular', 'percentage', 'proportion']):
                return "pie_chart"
            elif any(keyword in response_lower for keyword in ['scatter', 'correlation', 'dots']):
                return "scatter_plot"
            elif any(keyword in response_lower for keyword in ['histogram', 'distribution', 'frequency']):
                return "histogram"
            elif any(keyword in response_lower for keyword in ['box', 'whisker', 'quartile']):
                return "box_plot"
            elif any(keyword in response_lower for keyword in ['area', 'filled']):
                return "area_chart"
            else:
                return "other"
                
        except Exception as e:
            logger.error(f"❌ Chart type detection failed: {e}")
            return "other"
    
    def get_specialized_questions(self, chart_type: str) -> list:
        """
        Get specialized questions based on chart type
        
        Args:
            chart_type: Detected chart type
            
        Returns:
            List of specialized questions for the chart type
        """
        base_questions = [
            "Describe this chart in detail, including all visible elements.",
            "What is the main message or insight this chart conveys?"
        ]
        
        if chart_type == "line_chart":
            specialized_questions = [
                "Identify each line/trend in this chart and describe their patterns over time.",
                "What are the maximum and minimum values for each line series?",
                "Describe the overall trend direction for each line (increasing, decreasing, stable, or fluctuating).",
                "Are there any notable intersections, peaks, or valleys in the data?",
                "What time period or range does this chart cover?",
                "Calculate the approximate rate of change for the steepest trend."
            ]
        elif chart_type == "bar_chart":
            specialized_questions = [
                "List all categories shown and their corresponding values.",
                "Which category has the highest value and which has the lowest?",
                "What is the approximate ratio between the highest and lowest values?",
                "Are the bars arranged in any particular order (ascending, descending, categorical)?",
                "What is the total sum of all values if applicable?",
                "Identify any significant differences or patterns between categories."
            ]
        elif chart_type == "pie_chart":
            specialized_questions = [
                "List each segment with its percentage or proportion of the total.",
                "Which segment represents the largest portion and what percentage?",
                "Which segment represents the smallest portion and what percentage?",
                "Are there any segments that are approximately equal in size?",
                "What do the different colors or patterns represent?",
                "Calculate the combined percentage of the top 3 largest segments."
            ]
        elif chart_type == "scatter_plot":
            specialized_questions = [
                "Describe the correlation pattern between the x and y variables.",
                "Are there any outliers or unusual data points?",
                "What is the general trend direction (positive, negative, or no correlation)?",
                "Identify any clusters or groupings in the data points.",
                "What are the approximate ranges for both x and y axes?",
                "Is there evidence of linear or non-linear relationships?"
            ]
        elif chart_type == "histogram":
            specialized_questions = [
                "Describe the distribution shape (normal, skewed, bimodal, etc.).",
                "What is the range of values shown on the x-axis?",
                "Which bin or interval has the highest frequency?",
                "Are there any gaps or unusual patterns in the distribution?",
                "What can you infer about the central tendency of this data?",
                "Estimate the total number of observations represented."
            ]
        elif chart_type == "box_plot":
            specialized_questions = [
                "Identify the median, quartiles, and any outliers shown.",
                "What is the interquartile range (IQR) for each box?",
                "Compare the distributions if multiple boxes are shown.",
                "Are there any extreme outliers beyond the whiskers?",
                "Describe the skewness of each distribution.",
                "What insights can be drawn about data variability?"
            ]
        elif chart_type == "area_chart":
            specialized_questions = [
                "Describe how each area/category contributes to the total over time.",
                "Which category shows the most growth or decline?",
                "Are there any periods where certain categories dominate?",
                "What is the total value at different time points?",
                "Identify any significant changes in proportional relationships.",
                "Describe the stacking order and what it represents."
            ]
        else:  # other chart types
            specialized_questions = [
                "What type of data visualization technique is being used?",
                "What are the key variables or dimensions being displayed?",
                "Identify any patterns, trends, or relationships in the data.",
                "What conclusions can be drawn from this visualization?",
                "Are there any notable features or anomalies?",
                "How effective is this chart type for displaying this kind of data?"
            ]
        
        return base_questions + specialized_questions

def main():
    """Main function"""
    logger.info("🚀 Starting ChartGemma Academic Chart Analysis")
    
    # Define chart folder and output folder
    chart_folder = Path("Sample Line Chart")
    output_folder = Path("chartgemma_results")
    output_folder.mkdir(exist_ok=True)
    
    # Check if chart folder exists
    if not chart_folder.exists():
        logger.error(f"❌ Chart folder does not exist: {chart_folder}")
        return
    
    # Get all image files
    image_files = list(chart_folder.glob("*.png")) + list(chart_folder.glob("*.jpg")) + list(chart_folder.glob("*.jpeg"))
    
    if not image_files:
        logger.error(f"❌ No image files found in {chart_folder}")
        return
    
    logger.info(f"📁 Found {len(image_files)} image files")
    
    # Initialize ChartGemma processor
    processor = ChartGemmaProcessor()
    
    try:
        # Load model
        processor.load_model()
        
        # Analyze each image
        all_results = {}
        
        for image_file in image_files:
            logger.info(f"\n📊 Processing image: {image_file.name}")
            
            # Detect chart type first
            logger.info("🔍 Detecting chart type...")
            chart_type = processor.detect_chart_type(str(image_file))
            logger.info(f"📈 Detected chart type: {chart_type}")
            
            # Get specialized questions based on chart type
            questions = processor.get_specialized_questions(chart_type)
            
            image_results = {
                "image_path": str(image_file),
                "image_name": image_file.name,
                "detected_chart_type": chart_type,
                "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "questions_and_answers": []
            }
            
            # Analyze with specialized questions
            for i, question in enumerate(questions, 1):
                logger.info(f"   🤔 Question {i}/{len(questions)}: {question[:60]}{'...' if len(question) > 60 else ''}")
                
                start_time = time.time()
                answer = processor.analyze_chart(str(image_file), question)
                end_time = time.time()
                
                qa_result = {
                    "question": question,
                    "answer": answer,
                    "processing_time_seconds": round(end_time - start_time, 2)
                }
                
                image_results["questions_and_answers"].append(qa_result)
                
                logger.info(f"   ✅ Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
                logger.info(f"   ⏱️ Processing time: {qa_result['processing_time_seconds']}s")
                
                # Brief delay to avoid overload
                time.sleep(0.3)
            
            all_results[image_file.name] = image_results
        
        # Save results to JSON file
        output_file = output_folder / f"chartgemma_analysis_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Analysis results saved to: {output_file}")
        
        # Create HTML report
        create_html_report(all_results, output_folder)
        
        logger.info("🎉 ChartGemma analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Program execution failed: {e}")
        raise

def create_html_report(results: dict, output_folder: Path):
    """Create HTML analysis report"""
    try:
        logger.info("📄 Generating HTML report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>ChartGemma Academic Chart Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .chart-section {{ background: #f8f9fa; padding: 25px; margin-bottom: 25px; border-radius: 10px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .chart-type {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #1976d2; }}
                .question {{ background: #f3e5f5; padding: 12px; margin: 12px 0; border-left: 4px solid #7b1fa2; border-radius: 5px; }}
                .answer {{ background: #fff; padding: 12px; margin: 8px 0; border: 1px solid #e0e0e0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .meta {{ color: #666; font-size: 0.85em; font-style: italic; }}
                .image-container {{ text-align: center; margin: 25px 0; }}
                .image-container img {{ max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .stats {{ background: #fff8e1; padding: 15px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #ff8f00; }}
                .question-number {{ color: #1976d2; font-weight: bold; }}
                .chart-type-badge {{ background: #4caf50; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
                h1 {{ margin: 0; font-size: 2.2em; }}
                h2 {{ color: #1976d2; border-bottom: 2px solid #e3f2fd; padding-bottom: 10px; }}
                .summary {{ background: #e8f5e8; padding: 15px; border-radius: 6px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 ChartGemma Academic Chart Analysis Report</h1>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Charts Analyzed:</strong> {len(results)}</p>
                <p><strong>Model:</strong> <a href="https://huggingface.co/ahmed-masry/chartgemma" target="_blank" style="color: #fff; text-decoration: underline;">ahmed-masry/chartgemma</a></p>
                <p><strong>Analysis Type:</strong> Specialized Academic Chart Analysis with Chart Type Detection</p>
            </div>
        """
        
        for image_name, data in results.items():
            # Calculate statistics
            processing_times = [qa["processing_time_seconds"] for qa in data["questions_and_answers"]]
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            total_time = sum(processing_times)
            chart_type = data.get('detected_chart_type', 'unknown')
            
            html_content += f"""
            <div class="chart-section">
                <h2>📊 {image_name}</h2>
                
                <div class="chart-type">
                    <strong>🔍 Detected Chart Type:</strong> 
                    <span class="chart-type-badge">{chart_type.replace('_', ' ').title()}</span>
                </div>
                
                <div class="image-container">
                    <img src="../{data['image_path']}" alt="{image_name}">
                </div>
                
                <div class="stats">
                    <strong>📈 Processing Statistics:</strong><br>
                    <strong>Total Questions:</strong> {len(data['questions_and_answers'])}<br>
                    <strong>Total Processing Time:</strong> {total_time:.2f} seconds<br>
                    <strong>Average Response Time:</strong> {avg_time:.2f} seconds<br>
                    <strong>Analysis Timestamp:</strong> {data['analysis_timestamp']}<br>
                    <strong>Question Set:</strong> Specialized for {chart_type.replace('_', ' ')} analysis
                </div>
            """
            
            for i, qa in enumerate(data["questions_and_answers"], 1):
                html_content += f"""
                <div class="question">
                    <span class="question-number">Q{i}:</span> {qa['question']}
                    <div class="meta">Processing time: {qa['processing_time_seconds']} seconds</div>
                </div>
                <div class="answer">
                    <strong>🤖 ChartGemma Response:</strong><br>
                    {qa['answer'].replace('\n', '<br>')}
                </div>
                """
            
            html_content += "</div>"
        
        # Add summary section
        chart_types_found = set(data.get('detected_chart_type', 'unknown') for data in results.values())
        total_questions = sum(len(data['questions_and_answers']) for data in results.values())
        total_processing_time = sum(
            sum(qa['processing_time_seconds'] for qa in data['questions_and_answers']) 
            for data in results.values()
        )
        
        html_content += f"""
            <div class="summary">
                <h3>📋 Analysis Summary</h3>
                <p><strong>Chart Types Detected:</strong> {', '.join(sorted(chart_types_found))}</p>
                <p><strong>Total Questions Asked:</strong> {total_questions}</p>
                <p><strong>Total Processing Time:</strong> {total_processing_time:.2f} seconds</p>
                <p><strong>Average Time per Question:</strong> {total_processing_time/total_questions:.2f} seconds</p>
            </div>
            
            <div class="header">
                <h3>📖 About ChartGemma</h3>
                <p>This report demonstrates ChartGemma's capabilities for academic chart analysis with specialized question sets.</p>
                <p><strong>ChartGemma</strong> is a state-of-the-art chart understanding model based on PaliGemma, specifically designed for visual instruction-tuning on chart reasoning tasks.</p>
                <p><strong>Key Features:</strong></p>
                <ul style="color: white;">
                    <li>Automatic chart type detection</li>
                    <li>Specialized question sets for different chart types</li>
                    <li>Academic paper-focused analysis</li>
                    <li>Detailed numerical and trend analysis</li>
                </ul>
                <p><strong>More Information:</strong> <a href="https://huggingface.co/ahmed-masry/chartgemma" target="_blank" style="color: #fff; text-decoration: underline;">ChartGemma Model Page</a></p>
                <p><strong>Paper:</strong> <a href="https://arxiv.org/abs/2407.04172" target="_blank" style="color: #fff; text-decoration: underline;">ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild</a></p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = output_folder / f"chartgemma_academic_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📄 HTML report saved to: {html_file}")
        
    except Exception as e:
        logger.error(f"❌ HTML report generation failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️ 用户中断程序")
    except Exception as e:
        logger.error(f"❌ 程序异常退出: {e}")
        sys.exit(1)
