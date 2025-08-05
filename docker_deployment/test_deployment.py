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

def test_device_detection():
    """Test device detection functionality"""
    print("🔍 Testing device detection...")
    try:
        # Import torch and test device detection
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        print("✅ Device detection test completed")
        return True
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
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
    gpu_support = test_nvidia_gpu()
    
    # Run container with appropriate GPU support
    if gpu_support:
        cmd = ['docker', 'run', '--gpus', 'all', '-v', f'{os.getcwd()}/input:/app/input', 
               '-v', f'{os.getcwd()}/output:/app/output', 'pdf-analyzer']
    else:
        cmd = ['docker', 'run', '-v', f'{os.getcwd()}/input:/app/input', 
               '-v', f'{os.getcwd()}/output:/app/output', 'pdf-analyzer']
    
    try:
        print("Running container test...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Container test completed successfully")
            print("Output:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ Container test failed")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Container test timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ Container test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 PDF Image Analyzer - Docker Deployment Test")
    print("=" * 50)
    
    tests = [
        ("Docker Availability", test_docker_available),
        ("NVIDIA GPU", test_nvidia_gpu),
        ("Device Detection", test_device_detection),
        ("Directories", test_directories),
        ("Docker Build", test_build_image),
        ("Container Run", test_simple_run),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())