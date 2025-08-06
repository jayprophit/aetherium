#!/usr/bin/env python3
"""
AETHERIUM - EXECUTE NOW COMPLETE
Direct execution of the fully integrated platform
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path

def log(message, level="INFO"):
    """Simple logging"""
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "LAUNCH": "🚀"}
    print(f"{icons.get(level, '📝')} {message}")

def execute_platform():
    """Execute the complete Aetherium platform"""
    project_root = Path(__file__).parent
    
    log("🚀 EXECUTING AETHERIUM AI PLATFORM", "LAUNCH")
    log("=" * 50)
    
    try:
        os.chdir(project_root)
        
        # Check if dependencies need installation
        if not (project_root / "node_modules").exists():
            log("Installing dependencies...")
            result = subprocess.run(["npm", "install"], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                log(f"Dependency installation failed: {result.stderr}", "ERROR")
                return False
            log("Dependencies installed", "SUCCESS")
        
        # Start the development server
        log("Starting Aetherium Platform...")
        
        # Use subprocess.Popen to start the dev server
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root
        )
        
        # Give it time to start
        time.sleep(8)
        
        # Check if process is still running
        if process.poll() is None:
            log("Platform server started successfully!", "SUCCESS")
            
            # Open browser
            def open_browser_delayed():
                time.sleep(3)
                try:
                    webbrowser.open("http://localhost:5173")
                    log("Browser opened to http://localhost:5173", "SUCCESS")
                except:
                    log("Please manually open: http://localhost:5173")
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
            
            log("=" * 50)
            log("✅ AETHERIUM PLATFORM IS NOW RUNNING!", "SUCCESS")
            log("🌐 Access at: http://localhost:5173")
            log("=" * 50)
            log("")
            log("🎯 FEATURES AVAILABLE:")
            log("   ✅ Real-time Chat with AI")
            log("   ✅ 80+ Interactive AI Tools")
            log("   ✅ Quantum AI Models")
            log("   ✅ Advanced UI/UX (Manus/Claude style)")
            log("   ✅ System Status Monitoring")
            log("   ✅ Dark/Light Theme Toggle")
            log("   ✅ Cascading Sidebar Navigation")
            log("   ✅ Mobile Responsive Design")
            log("")
            log("🎉 ENJOY YOUR QUANTUM AI PLATFORM!")
            log("Press Ctrl+C to stop...")
            
            try:
                # Keep running
                while process.poll() is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                log("Shutting down...")
                process.terminate()
            
            return True
        else:
            # Process failed
            stdout, stderr = process.communicate()
            log(f"Server failed to start: {stderr}", "ERROR")
            return False
            
    except Exception as e:
        log(f"Execution failed: {e}", "ERROR")
        return False

if __name__ == "__main__":
    success = execute_platform()
    if not success:
        log("❌ EXECUTION FAILED")
        input("Press Enter to exit...")
    sys.exit(0 if success else 1)