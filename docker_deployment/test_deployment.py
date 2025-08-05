#!/usr/bin/env python3
"""
Test script for Docker deployment
Verify that all components are working correctly
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def test_docker_available():
    """Test if Docker is available and running"""
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available and running")
            return True
        else:
            print("❌ Docker is not running")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def test_nvidia_gpu():
    """Test if NVIDIA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(f"GPU Info: {result.stdout.split('|')[1].strip() if '|' in result.stdout else 'GPU available'}")
            return True
        else:
            print("⚠️  NVIDIA GPU not detected, will use CPU")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found, will use CPU")
        return False

def test_directories():
    """Create and test input/output directories"""
    try:
        os.makedirs('input', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        print("✅ Input/output directories created")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def test_build_image():
    """Test building the Docker image"""
    print("🔨 Building Docker image (this may take a while)...")
    try:
        result = subprocess.run(['docker', 'build', '-t', 'pdf-analyzer', '.'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker image built successfully")
            return True
        else:
            print("❌ Docker build failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return False

def create_test_json():
    """Create a simple test JSON file to verify processing"""
    test_data = {
        "pictures": [
            {
                "prov": [{"page": {"page": 1}}],
                "image": {
                    "uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                }
            }
        ],
        "texts": [
            {"text": "Test document content"}
        ]
    }
    
    test_file = Path('input/test_document.json')
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Created test file: {test_file}")
    return test_file

def test_simple_run():
    """Test a simple container run"""
    print("🧪 Testing simple container run...")
    
    # Create test JSON file
    test_file = create_test_json()
    
    # Determine GPU support
    gpu_flag = "--gpus all" if test_nvidia_gpu() else ""
    
    # Run container
    cmd = f"docker run --rm {gpu_flag} -v {os.getcwd()}/input:/app/input -v {os.getcwd()}/output:/app/output pdf-analyzer python main.py /app/input/test_document.json --output-dir /app/output"
    
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Container run successful")
            
            # Check output files
            output_files = list(Path('output').glob('*'))
            if output_files:
                print(f"✅ Output files generated: {[f.name for f in output_files]}")
                return True
            else:
                print("⚠️  No output files found")
                return False
        else:
            print("❌ Container run failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Container run timed out (this may be normal for first run)")
        return False
    except Exception as e:
        print(f"❌ Container run error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== PDF Image Analyzer Docker Deployment Test ===")
    print()
    
    tests = [
        ("Docker availability", test_docker_available),
        ("NVIDIA GPU", test_nvidia_gpu),
        ("Directories", test_directories),
        ("Docker image build", test_build_image),
        ("Simple container run", test_simple_run)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        results[test_name] = test_func()
    
    print("\n=== Test Results Summary ===")
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your deployment is ready.")
        print("\nTo use the system:")
        print("1. Place PDF files in the input/ directory")
        print("2. Run: ./deploy.sh (Linux/Mac) or deploy.bat (Windows)")
    elif passed >= total - 1:
        print("\n⚠️  Most tests passed. Your deployment should work.")
        print("Check any failed tests above.")
    else:
        print("\n❌ Several tests failed. Please check your setup.")
        print("Common issues:")
        print("- Docker Desktop not running")
        print("- Insufficient disk space")
        print("- Network connectivity issues")

if __name__ == "__main__":
    main()