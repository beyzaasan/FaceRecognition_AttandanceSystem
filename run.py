#!/usr/bin/env python3
"""
Biometric Access System Runner Script
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found!")
        if os.path.exists('.env.example'):
            print("📝 Creating .env file from template...")
            shutil.copy('.env.example', '.env')
            print("✅ .env file created. Please edit it with your database credentials.")
            print("📝 Edit .env file and run this script again.")
            return False
        else:
            print("❌ .env.example file not found!")
            return False
    
    # Check if app directory exists
    if not os.path.exists('app'):
        print("❌ app directory not found!")
        return False
    
    # Check if biometric_app.py exists
    if not os.path.exists('app/biometric_app.py'):
        print("❌ biometric_app.py not found in app directory!")
        return False
    
    print("✅ All requirements met!")
    return True

def install_dependencies():
    """Install dependencies if needed"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_app():
    """Run the Streamlit application"""
    print("🚀 Starting Biometric Access System...")
    
    # Run streamlit from app directory
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app/biometric_app.py'], 
                      check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")

def main():
    """Main function"""
    print("🔐 Biometric Access System Runner")
    print("=" * 40)
    
    # Check if user wants to install dependencies
    if len(sys.argv) > 1 and sys.argv[1] == '--install':
        if not install_dependencies():
            sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main() 