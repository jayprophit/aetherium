#!/usr/bin/env python3
"""
Debug deployment issues and start servers manually
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_files():
    """Check if required files exist"""
    print("🔍 Checking required files...")
    
    backend_main = Path("backend/main.py")
    frontend_package = Path("frontend/package.json")
    requirements = Path("backend/requirements.txt")
    
    print(f"✓ Backend main.py: {'EXISTS' if backend_main.exists() else 'MISSING'}")
    print(f"✓ Frontend package.json: {'EXISTS' if frontend_package.exists() else 'MISSING'}")
    print(f"✓ Requirements.txt: {'EXISTS' if requirements.exists() else 'MISSING'}")
    
    return backend_main.exists()

def start_backend_simple():
    """Start backend server with simple approach"""
    print("\n🚀 Starting Backend Server...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    main_py = backend_dir / "main.py"
    if not main_py.exists():
        print("❌ main.py not found!")
        return False
    
    try:
        # Change to backend directory and start server
        os.chdir(backend_dir)
        print(f"📁 Changed to directory: {os.getcwd()}")
        
        # Try to start the server
        print("🔄 Starting FastAPI server...")
        subprocess.run([sys.executable, "main.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend startup failed: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 Backend server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main troubleshooting function"""
    print("🔧 AETHERIUM DEPLOYMENT TROUBLESHOOTER")
    print("=" * 45)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Check if files exist
    if not check_files():
        print("\n❌ Required files missing! Please ensure you're in the correct directory.")
        return
    
    print("\n🚀 Attempting to start backend server...")
    print("   (Press Ctrl+C to stop)")
    
    start_backend_simple()

if __name__ == "__main__":
    main()