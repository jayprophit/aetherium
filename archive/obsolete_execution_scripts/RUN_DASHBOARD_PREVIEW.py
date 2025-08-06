#!/usr/bin/env python3
"""
RUN DASHBOARD PREVIEW - Execute the working Aetherium dashboard
"""
import subprocess
import sys
import os

def main():
    print("🚀 LAUNCHING AETHERIUM DASHBOARD PREVIEW")
    print("=" * 50)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {script_dir}")
    print("🎯 Launching AETHERIUM_UPDATED_DASHBOARD.py...")
    print("=" * 50)
    
    try:
        # Execute the working dashboard script
        subprocess.run([sys.executable, "AETHERIUM_UPDATED_DASHBOARD.py"], 
                      capture_output=False, 
                      text=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying alternative dashboard script...")
        try:
            subprocess.run([sys.executable, "AETHERIUM_MAIN_DASHBOARD.py"], 
                          capture_output=False, 
                          text=True)
        except Exception as e2:
            print(f"❌ Alternative failed: {e2}")

if __name__ == "__main__":
    main()