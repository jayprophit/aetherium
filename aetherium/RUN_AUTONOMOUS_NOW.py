#!/usr/bin/env python3
"""
RUN AUTONOMOUS EXECUTION NOW
===========================
Execute the direct autonomous implementation immediately.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_now():
    print("🤖 RUNNING AUTONOMOUS EXECUTION NOW...")
    print("="*60)
    
    script_dir = Path("C:/Users/jpowe/CascadeProjects/github/aetherium/aetherium")
    os.chdir(script_dir)
    
    # Execute the direct autonomous script
    script_path = script_dir / "DIRECT_AUTONOMOUS_EXECUTION.py"
    
    print(f"🚀 Executing: {script_path}")
    
    try:
        result = subprocess.run([
            sys.executable, 
            str(script_path)
        ], 
        cwd=str(script_dir),
        capture_output=False,
        text=True)
        
        print(f"📊 Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("\n🎉 AUTONOMOUS EXECUTION COMPLETED SUCCESSFULLY!")
            return True
        else:
            print(f"\n⚠️ Exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_now()
    sys.exit(0 if success else 1)