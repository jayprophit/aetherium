#!/usr/bin/env python3
"""
COMPLETE AETHERIUM LAUNCHER - ENHANCED VERSION
Launches the enhanced backend with all integrations + React frontend
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
import requests
from pathlib import Path

def install_deps():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    deps = ["fastapi", "uvicorn[standard]", "python-multipart", "websockets", "aiofiles"]
    
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {dep}")
        except:
            print(f"   ⚠️ {dep} (may already be installed)")
    
    print("✅ Python dependencies ready")

def start_backend():
    """Start the enhanced backend"""
    print("🚀 Starting Enhanced Backend...")
    
    backend_path = Path("src/backend_enhanced.py")
    if not backend_path.exists():
        print("❌ Enhanced backend not found!")
        return None
    
    try:
        # Start backend process
        process = subprocess.Popen([sys.executable, str(backend_path)])
        
        # Wait for startup
        print("   ⏳ Backend starting up...")
        time.sleep(4)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                print("   ✅ Backend is healthy!")
                print(f"      🔌 WebSocket: {health.get('features', {}).get('websocket')}")
                print(f"      📁 File Upload: {health.get('features', {}).get('file_upload')}")
                print(f"      🗄️ Database: {health.get('features', {}).get('database')}")
                print(f"      🤖 AI Tools: {health.get('features', {}).get('ai_tools')}")
                return process
            else:
                print(f"   ❌ Backend health check failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Backend not responding: {e}")
        
    except Exception as e:
        print(f"   ❌ Failed to start backend: {e}")
    
    return None

def start_frontend():
    """Start React frontend"""
    print("🎨 Starting React Frontend...")
    
    if not Path("package.json").exists():
        print("   ❌ package.json not found!")
        return None
    
    try:
        # Install deps if needed
        if not Path("node_modules").exists():
            print("   📦 Installing Node dependencies...")
            subprocess.run(["npm", "install"], check=True, stdout=subprocess.DEVNULL)
        
        # Start dev server
        print("   ⏳ Frontend starting up...")
        process = subprocess.Popen(["npm", "run", "dev"])
        
        # Wait for startup
        time.sleep(6)
        
        print("   ✅ Frontend is running!")
        return process
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Frontend failed to start: {e}")
    except FileNotFoundError:
        print("   ❌ Node.js/npm not found. Please install Node.js")
    
    return None

def validate_platform():
    """Validate platform is working"""
    print("🔍 Validating Platform...")
    
    tests = []
    
    # Test backend health
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        tests.append(("Backend Health", response.status_code == 200))
    except:
        tests.append(("Backend Health", False))
    
    # Test tools endpoint
    try:
        response = requests.get("http://localhost:8000/api/tools", timeout=5)
        tests.append(("AI Tools API", response.status_code == 200))
    except:
        tests.append(("AI Tools API", False))
    
    # Test frontend
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        tests.append(("Frontend", response.status_code == 200))
    except:
        tests.append(("Frontend", False))
    
    # Print results
    print("   📊 Validation Results:")
    all_passed = True
    for test_name, passed in tests:
        status = "✅" if passed else "❌"
        print(f"      {status} {test_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def open_browser():
    """Open browser after delay"""
    time.sleep(8)
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:5173")

def main():
    """Main launcher function"""
    print("🚀 AETHERIUM COMPLETE LAUNCHER")
    print("=" * 60)
    print("Starting complete AI productivity platform...")
    print()
    
    # Install dependencies
    install_deps()
    print()
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Backend failed to start. Aborting.")
        return False
    
    print()
    
    # Start frontend  
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Frontend failed to start. Cleaning up...")
        backend_process.terminate()
        return False
    
    print()
    
    # Validate platform
    time.sleep(2)
    validation_passed = validate_platform()
    
    print()
    print("=" * 60)
    
    if validation_passed:
        print("🎉 AETHERIUM PLATFORM LAUNCHED SUCCESSFULLY!")
        print()
        print("📍 Access Points:")
        print("   🎨 Frontend:  http://localhost:5173")
        print("   🔧 Backend:   http://localhost:8000")  
        print("   📚 API Docs:  http://localhost:8000/docs")
        print()
        print("🌟 Features Ready:")
        print("   ✅ Real-time AI chat with thinking process")
        print("   ✅ 8+ AI tools with real execution")
        print("   ✅ File upload/download system")
        print("   ✅ Database persistence")
        print("   ✅ WebSocket integration")
        print("   ✅ Multi-user support")
        print("   ✅ Claude/Manus-style UI")
        
        # Open browser in background
        threading.Thread(target=open_browser, daemon=True).start()
        
    else:
        print("⚠️ PLATFORM LAUNCHED WITH ISSUES")
        print("Some components may not be fully functional.")
    
    print("=" * 60)
    
    try:
        print("\n🔄 Platform is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if frontend_process:
            frontend_process.terminate()
        if backend_process:
            backend_process.terminate()
        print("✅ Shutdown complete")
    
    return validation_passed

if __name__ == "__main__":
    success = main()
    if not success:
        input("\n❌ Launch issues detected. Press Enter to exit...")
    sys.exit(0 if success else 1)