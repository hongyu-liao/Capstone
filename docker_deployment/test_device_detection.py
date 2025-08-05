#!/usr/bin/env python3
"""
Test script for device detection functionality
"""

import torch
import sys

def test_device_detection():
    """Test the device detection functionality"""
    print("🔍 Testing Device Detection...")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
        
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # List all GPU devices
        print("\n📊 Available GPU Devices:")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"\nCurrent GPU Device: {current_device}")
        
        # Test device selection
        print("\n🎯 Testing Device Selection:")
        print("1. CPU")
        print("2. GPU")
        
        # Simulate device selection
        test_device = torch.device("cuda" if gpu_count > 0 else "cpu")
        print(f"Selected device: {test_device}")
        
        # Test model loading on selected device
        try:
            print(f"\n🤖 Testing model loading on {test_device}...")
            # Create a simple tensor to test device
            test_tensor = torch.randn(2, 2).to(test_device)
            print(f"✅ Successfully created tensor on {test_device}")
            print(f"Tensor device: {test_tensor.device}")
        except Exception as e:
            print(f"❌ Failed to create tensor on {test_device}: {e}")
            return False
            
    else:
        print("⚠️  CUDA not available. Testing CPU only...")
        test_device = torch.device("cpu")
        
        try:
            print(f"\n🤖 Testing model loading on {test_device}...")
            test_tensor = torch.randn(2, 2).to(test_device)
            print(f"✅ Successfully created tensor on {test_device}")
            print(f"Tensor device: {test_tensor.device}")
        except Exception as e:
            print(f"❌ Failed to create tensor on {test_device}: {e}")
            return False
    
    print("\n🎉 Device detection test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_device_detection()
    sys.exit(0 if success else 1) 