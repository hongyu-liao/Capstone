#!/usr/bin/env python3
"""
Simplified ChartGemma Test Script
For quick testing of single charts with academic-focused analysis
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

def test_single_chart(image_path: str, question: str = "Describe this chart in detail, including all visible elements and data patterns."):
    """
    Test single chart with ChartGemma
    
    Args:
        image_path: Path to image file
        question: Question to ask about the chart
    """
    try:
        print(f"🤖 Loading ChartGemma model...")
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            "ahmed-masry/chartgemma", 
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
        model = model.to(device)
        
        print(f"✅ Model loaded successfully")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"📊 Analyzing chart: {Path(image_path).name}")
        print(f"🤔 Question: {question}")
        
        # Process inputs
        inputs = processor(text=question, images=image, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
        
        # Decode output
        output_text = processor.batch_decode(
            generate_ids[:, prompt_length:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\n💬 ChartGemma Response:")
        print("-" * 60)
        print(output_text.strip())
        print("-" * 60)
        
        return output_text.strip()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_chartgemma_simple.py <image_path> [question]")
        print("Example: python run_chartgemma_simple.py 'Sample Line Chart/3.png'")
        print("\nPredefined academic questions you can try:")
        print("- 'What type of chart is this and what are its main components?'")
        print("- 'Identify and describe each data series or trend shown.'")
        print("- 'What are the maximum and minimum values in this chart?'")
        print("- 'Describe the overall patterns and relationships in the data.'")
        print("- 'What insights can be drawn from this visualization?'")
        
        # If no arguments, try to analyze the first image
        sample_images = list(Path("Sample Line Chart").glob("*.png"))
        if sample_images:
            image_path = str(sample_images[0])
            print(f"\n📊 Using sample image: {image_path}")
        else:
            print("❌ Please provide an image path")
            return
    else:
        image_path = sys.argv[1]
    
    # Get question
    if len(sys.argv) >= 3:
        question = sys.argv[2]
    else:
        question = "Describe this chart in detail, including all visible elements and data patterns."
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image file does not exist: {image_path}")
        return
    
    # Run analysis
    result = test_single_chart(image_path, question)
    
    if result:
        print("\n🎉 Analysis completed!")
        print("\n💡 Tip: Try running the full analysis script with:")
        print("   python test_chartgemma.py")
    else:
        print("\n❌ Analysis failed")

if __name__ == "__main__":
    main()
