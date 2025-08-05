#!/bin/bash

echo "=== PDF Image Analyzer - Docker Deployment ==="
echo "Scanning for PDF files in /app/input..."

# Check if input directory has any PDF files
if ls /app/input/*.pdf 1> /dev/null 2>&1; then
    echo "Found PDF files to process:"
    ls -la /app/input/*.pdf
    
    # Process each PDF file
    for pdf_file in /app/input/*.pdf; do
        echo ""
        echo "Processing: $(basename "$pdf_file")"
        echo "========================================"
        
        # Run the main processing script
        python main.py "$pdf_file" --output-dir /app/output
        
        # Check if processing was successful
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $(basename "$pdf_file")"
        else
            echo "❌ Failed to process: $(basename "$pdf_file")"
        fi
        echo ""
    done
    
    echo "=== Processing Complete ==="
    echo "Results are available in /app/output/"
    ls -la /app/output/
    
else
    echo "❌ No PDF files found in /app/input/"
    echo "Please mount a directory containing PDF files to /app/input"
    echo ""
    echo "Usage example:"
    echo "docker run -v /path/to/pdfs:/app/input -v /path/to/output:/app/output pdf-analyzer"
fi

# Keep container running for debugging if needed
echo ""
echo "Container will exit now. Check /app/output for results."