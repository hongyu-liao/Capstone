import logging
from pathlib import Path
import json
import os
import time
from typing import List, Dict, Any, Optional
import concurrent.futures
from datetime import datetime
import shutil

from pdf_processor import PDFProcessor
from image_analyzer import ImageAnalyzer

class BatchProcessor:
    """Handles batch processing of multiple PDF files and advanced operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batch processor
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pdf_processor = PDFProcessor(config)
        self.image_analyzer = ImageAnalyzer(config)
        
    def process_multiple_files(self, file_paths: List[str], output_dir: str, 
                             parallel: bool = False, progress_callback=None) -> List[Dict]:
        """
        Process multiple PDF or JSON files
        
        Args:
            file_paths (List[str]): List of file paths to process
            output_dir (str): Output directory
            parallel (bool): Whether to process files in parallel
            progress_callback: Function to call with progress updates
            
        Returns:
            List[Dict]: Results for each processed file
        """
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"Starting batch processing of {total_files} files")
        
        if parallel and total_files > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, total_files)) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, output_dir, i): file_path 
                    for i, file_path in enumerate(file_paths)
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if progress_callback:
                            progress_callback(len(results), total_files, file_path)
                            
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {e}")
                        results.append({
                            'file_path': file_path,
                            'success': False,
                            'error': str(e),
                            'timestamp': datetime.now()
                        })
        else:
            # Sequential processing
            for i, file_path in enumerate(file_paths):
                try:
                    result = self._process_single_file(file_path, output_dir, i)
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files, file_path)
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    results.append({
                        'file_path': file_path,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
        
        self.logger.info(f"Batch processing completed. {len([r for r in results if r.get('success')])} successful, {len([r for r in results if not r.get('success')])} failed")
        
        return results
    
    def _process_single_file(self, file_path: str, output_dir: str, index: int) -> Dict:
        """
        Process a single file
        
        Args:
            file_path (str): Path to the file
            output_dir (str): Output directory
            index (int): File index in batch
            
        Returns:
            Dict: Processing result
        """
        file_path_obj = Path(file_path)
        file_type = "pdf" if file_path_obj.suffix.lower() == ".pdf" else "json"
        
        # Create unique output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"batch_{index:03d}_{file_path_obj.stem}_{timestamp}"
        
        try:
            start_time = time.time()
            
            if file_type == "pdf":
                # Convert PDF to JSON
                json_result = self.pdf_processor.convert_pdf_to_json(
                    file_path, output_dir, output_filename
                )
                if not json_result:
                    return {
                        'file_path': file_path,
                        'file_type': file_type,
                        'success': False,
                        'error': 'PDF conversion failed',
                        'timestamp': datetime.now()
                    }
                json_path = json_result['json_path']
            else:
                json_path = file_path
            
            # Analyze images
            analysis_results = self.image_analyzer.analyze_images_from_json(
                json_path,
                self.config.get('enable_web_search', True)
            )
            
            # Create enhanced JSON
            enhanced_json_path = self.image_analyzer.create_enhanced_json(
                json_path,
                analysis_results,
                output_dir,
                output_filename
            )
            
            # Create NLP-ready version if requested
            nlp_ready_path = None
            if self.config.get('generate_nlp_ready', False):
                nlp_ready_path = self.image_analyzer.create_nlp_ready_version(
                    enhanced_json_path,
                    output_dir,
                    output_filename
                )
            
            processing_time = time.time() - start_time
            
            return {
                'file_path': file_path,
                'file_type': file_type,
                'file_name': file_path_obj.name,
                'success': True,
                'processing_time': processing_time,
                'enhanced_json': enhanced_json_path,
                'nlp_ready_json': nlp_ready_path,
                'analysis_results': analysis_results,
                'statistics': self._calculate_statistics(analysis_results),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return {
                'file_path': file_path,
                'file_type': file_type,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _calculate_statistics(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for analysis results"""
        total_images = len(analysis_results)
        informative_images = sum(1 for img in analysis_results if not img.get('is_non_informative', False))
        conceptual_images = sum(1 for img in analysis_results if img.get('image_type') == 'CONCEPTUAL')
        data_viz_images = sum(1 for img in analysis_results if img.get('image_type') == 'DATA_VISUALIZATION')
        failed_images = sum(1 for img in analysis_results if img.get('image_type') == 'ANALYSIS_FAILED')
        web_search_count = sum(1 for img in analysis_results if 'web_context' in img)
        
        return {
            'total_images': total_images,
            'informative_images': informative_images,
            'conceptual_images': conceptual_images,
            'data_visualization_images': data_viz_images,
            'failed_analysis': failed_images,
            'web_search_enhanced': web_search_count,
            'non_informative_images': total_images - informative_images - failed_images
        }
    
    def compare_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compare results across multiple processed files
        
        Args:
            results (List[Dict]): List of processing results
            
        Returns:
            Dict: Comparison analysis
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful results to compare'}
        
        # Aggregate statistics
        total_stats = {
            'files_processed': len(successful_results),
            'total_images': 0,
            'total_informative': 0,
            'total_conceptual': 0,
            'total_data_viz': 0,
            'total_failed': 0,
            'total_web_search': 0,
            'total_processing_time': 0
        }
        
        file_comparisons = []
        
        for result in successful_results:
            stats = result.get('statistics', {})
            total_stats['total_images'] += stats.get('total_images', 0)
            total_stats['total_informative'] += stats.get('informative_images', 0)
            total_stats['total_conceptual'] += stats.get('conceptual_images', 0)
            total_stats['total_data_viz'] += stats.get('data_visualization_images', 0)
            total_stats['total_failed'] += stats.get('failed_analysis', 0)
            total_stats['total_web_search'] += stats.get('web_search_enhanced', 0)
            total_stats['total_processing_time'] += result.get('processing_time', 0)
            
            file_comparisons.append({
                'file_name': result.get('file_name', 'Unknown'),
                'file_type': result.get('file_type', 'Unknown'),
                'processing_time': result.get('processing_time', 0),
                'images_found': stats.get('total_images', 0),
                'informative_ratio': stats.get('informative_images', 0) / max(stats.get('total_images', 1), 1),
                'conceptual_ratio': stats.get('conceptual_images', 0) / max(stats.get('informative_images', 1), 1),
                'data_viz_ratio': stats.get('data_visualization_images', 0) / max(stats.get('informative_images', 1), 1)
            })
        
        # Calculate averages
        num_files = len(successful_results)
        averages = {
            'avg_images_per_file': total_stats['total_images'] / num_files,
            'avg_informative_per_file': total_stats['total_informative'] / num_files,
            'avg_processing_time': total_stats['total_processing_time'] / num_files,
            'avg_informative_ratio': total_stats['total_informative'] / max(total_stats['total_images'], 1),
            'avg_conceptual_ratio': total_stats['total_conceptual'] / max(total_stats['total_informative'], 1),
            'avg_data_viz_ratio': total_stats['total_data_viz'] / max(total_stats['total_informative'], 1)
        }
        
        return {
            'summary': total_stats,
            'averages': averages,
            'file_comparisons': file_comparisons,
            'failed_files': [r for r in results if not r.get('success', False)]
        }
    
    def create_batch_report(self, results: List[Dict], output_dir: str) -> str:
        """
        Create a comprehensive batch processing report
        
        Args:
            results (List[Dict]): Processing results
            output_dir (str): Output directory
            
        Returns:
            str: Path to the report file
        """
        comparison = self.compare_results(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(output_dir) / f"batch_report_{timestamp}.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("BATCH PROCESSING REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"AI Provider: {self.config.get('ai_provider', 'Unknown')}\n")
                f.write(f"Model: {self.config.get('model_name', 'Unknown')}\n")
                f.write(f"Web Search Enabled: {self.config.get('enable_web_search', False)}\n\n")
                
                # Summary
                if 'summary' in comparison:
                    summary = comparison['summary']
                    f.write("SUMMARY STATISTICS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Files Successfully Processed: {summary['files_processed']}\n")
                    f.write(f"Total Images Found: {summary['total_images']}\n")
                    f.write(f"Informative Images: {summary['total_informative']}\n")
                    f.write(f"Conceptual Images: {summary['total_conceptual']}\n")
                    f.write(f"Data Visualization Images: {summary['total_data_viz']}\n")
                    f.write(f"Failed Analyses: {summary['total_failed']}\n")
                    f.write(f"Web Search Enhanced: {summary['total_web_search']}\n")
                    f.write(f"Total Processing Time: {summary['total_processing_time']:.2f} seconds\n\n")
                
                # Averages
                if 'averages' in comparison:
                    averages = comparison['averages']
                    f.write("AVERAGE METRICS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Images per File: {averages['avg_images_per_file']:.1f}\n")
                    f.write(f"Informative Images per File: {averages['avg_informative_per_file']:.1f}\n")
                    f.write(f"Processing Time per File: {averages['avg_processing_time']:.2f} seconds\n")
                    f.write(f"Informative Ratio: {averages['avg_informative_ratio']:.2%}\n")
                    f.write(f"Conceptual Ratio: {averages['avg_conceptual_ratio']:.2%}\n")
                    f.write(f"Data Visualization Ratio: {averages['avg_data_viz_ratio']:.2%}\n\n")
                
                # File-by-file breakdown
                if 'file_comparisons' in comparison:
                    f.write("FILE-BY-FILE BREAKDOWN\n")
                    f.write("-" * 30 + "\n")
                    for file_info in comparison['file_comparisons']:
                        f.write(f"\nFile: {file_info['file_name']}\n")
                        f.write(f"  Type: {file_info['file_type'].upper()}\n")
                        f.write(f"  Processing Time: {file_info['processing_time']:.2f}s\n")
                        f.write(f"  Images Found: {file_info['images_found']}\n")
                        f.write(f"  Informative Ratio: {file_info['informative_ratio']:.2%}\n")
                        f.write(f"  Conceptual Ratio: {file_info['conceptual_ratio']:.2%}\n")
                        f.write(f"  Data Viz Ratio: {file_info['data_viz_ratio']:.2%}\n")
                
                # Failed files
                if 'failed_files' in comparison and comparison['failed_files']:
                    f.write("\n\nFAILED FILES\n")
                    f.write("-" * 30 + "\n")
                    for failed in comparison['failed_files']:
                        f.write(f"\nFile: {Path(failed['file_path']).name}\n")
                        f.write(f"  Error: {failed.get('error', 'Unknown error')}\n")
                        f.write(f"  Timestamp: {failed.get('timestamp', 'Unknown')}\n")
            
            self.logger.info(f"Batch report saved: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create batch report: {e}")
            return None
    
    def export_combined_results(self, results: List[Dict], output_dir: str, 
                              format_type: str = "json") -> str:
        """
        Export combined results from multiple files
        
        Args:
            results (List[Dict]): Processing results
            output_dir (str): Output directory
            format_type (str): Export format ("json", "csv", "xlsx")
            
        Returns:
            str: Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if format_type == "json":
                export_path = Path(output_dir) / f"combined_results_{timestamp}.json"
                
                combined_data = {
                    'metadata': {
                        'export_timestamp': datetime.now().isoformat(),
                        'ai_provider': self.config.get('ai_provider', 'Unknown'),
                        'model': self.config.get('model_name', 'Unknown'),
                        'total_files': len(results),
                        'successful_files': len([r for r in results if r.get('success', False)])
                    },
                    'comparison_analysis': self.compare_results(results),
                    'individual_results': results
                }
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False, default=str)
            
            elif format_type == "csv":
                import pandas as pd
                export_path = Path(output_dir) / f"combined_results_{timestamp}.csv"
                
                # Flatten results for CSV
                flattened_data = []
                for result in results:
                    if result.get('success', False):
                        stats = result.get('statistics', {})
                        row = {
                            'file_name': result.get('file_name', ''),
                            'file_type': result.get('file_type', ''),
                            'processing_time': result.get('processing_time', 0),
                            'total_images': stats.get('total_images', 0),
                            'informative_images': stats.get('informative_images', 0),
                            'conceptual_images': stats.get('conceptual_images', 0),
                            'data_viz_images': stats.get('data_visualization_images', 0),
                            'failed_analysis': stats.get('failed_analysis', 0),
                            'web_search_enhanced': stats.get('web_search_enhanced', 0),
                            'timestamp': result.get('timestamp', '')
                        }
                        flattened_data.append(row)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(export_path, index=False)
            
            elif format_type == "xlsx":
                import pandas as pd
                export_path = Path(output_dir) / f"combined_results_{timestamp}.xlsx"
                
                # Create multiple sheets
                with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                    # Summary sheet
                    comparison = self.compare_results(results)
                    if 'summary' in comparison:
                        summary_df = pd.DataFrame([comparison['summary']])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # File comparisons sheet
                    if 'file_comparisons' in comparison:
                        comparisons_df = pd.DataFrame(comparison['file_comparisons'])
                        comparisons_df.to_excel(writer, sheet_name='File_Comparisons', index=False)
                    
                    # Failed files sheet
                    if 'failed_files' in comparison and comparison['failed_files']:
                        failed_df = pd.DataFrame(comparison['failed_files'])
                        failed_df.to_excel(writer, sheet_name='Failed_Files', index=False)
            
            self.logger.info(f"Combined results exported: {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export combined results: {e}")
            return None
    
    def cleanup_intermediate_files(self, results: List[Dict], keep_enhanced: bool = True) -> int:
        """
        Clean up intermediate files from batch processing
        
        Args:
            results (List[Dict]): Processing results
            keep_enhanced (bool): Whether to keep enhanced JSON files
            
        Returns:
            int: Number of files cleaned up
        """
        cleaned_count = 0
        
        for result in results:
            if not result.get('success', False):
                continue
            
            try:
                # Remove original JSON if it was created from PDF
                if result.get('file_type') == 'pdf':
                    # The original JSON was created during processing
                    original_json = result.get('enhanced_json', '').replace('_enhanced.json', '.json')
                    if os.path.exists(original_json):
                        os.remove(original_json)
                        cleaned_count += 1
                
                # Optionally remove enhanced JSON files
                if not keep_enhanced:
                    enhanced_json = result.get('enhanced_json')
                    if enhanced_json and os.path.exists(enhanced_json):
                        os.remove(enhanced_json)
                        cleaned_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Failed to clean up files for {result.get('file_name', 'unknown')}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} intermediate files")
        return cleaned_count