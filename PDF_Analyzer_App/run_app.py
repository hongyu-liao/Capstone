#!/usr/bin/env python3
"""
Launcher script for PDF Image Analyzer
This script helps users start the application with proper error checking
"""

import sys
import subprocess
import pkg_resources
import logging
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'docling',
        'requests',
        'PIL'  # Pillow
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package} is installed")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("💡 Please install them with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_app_files():
    """Check if required application files exist"""
    required_files = [
        'app.py',
        'pdf_processor.py',
        'image_analyzer.py',
        'config.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"❌ {file} is missing")
        else:
            print(f"✅ {file} found")
    
    if missing_files:
        print(f"\n📄 Missing files: {', '.join(missing_files)}")
        print("💡 Please ensure all application files are in the current directory.")
        return False
    
    return True

def test_lm_studio_connection():
    """Test connection to LM Studio (optional check)"""
    import requests
    
    try:
        response = requests.get("http://localhost:1234", timeout=5)
        print("✅ LM Studio server appears to be running")
        return True
    except requests.exceptions.RequestException:
        print("⚠️  Warning: Cannot connect to LM Studio at http://localhost:1234")
        print("💡 Please ensure LM Studio is running with a vision model loaded")
        print("   You can still start the app and configure the connection later")
        return False

def run_streamlit_app():
    """Launch the Streamlit application"""
    try:
        print("\n🚀 Starting PDF Image Analyzer...")
        print("📱 The web interface will open in your browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application\n")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install requirements.txt")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function to run all checks and start the app"""
    print("=" * 60)
    print("📄 PDF Image Analyzer - Startup Checker")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Application Files", check_app_files),
        ("LM Studio Connection", test_lm_studio_connection)
    ]
    
    all_critical_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        result = check_func()
        
        # Only LM Studio check is optional
        if not result and check_name != "LM Studio Connection":
            all_critical_passed = False
    
    print("\n" + "=" * 60)
    
    if all_critical_passed:
        print("🎉 All critical checks passed!")
        
        # Ask user if they want to continue
        try:
            response = input("\n▶️  Start the application now? (y/n): ").lower().strip()
            if response in ['y', 'yes', '']:
                run_streamlit_app()
            else:
                print("👋 Startup cancelled by user")
        except KeyboardInterrupt:
            print("\n👋 Startup cancelled by user")
    else:
        print("❌ Some critical checks failed. Please fix the issues above and try again.")
        print("\n💡 Quick fix commands:")
        print("   pip install -r requirements.txt")
        print("   python run_app.py")

if __name__ == "__main__":
    main()