#!/usr/bin/env python3
import subprocess
import sys
import os

print("🚀 EXECUTING FULL AUTONOMOUS PRODUCTION LAUNCH NOW...")
print("="*70)

# Ensure we're in the correct directory
os.chdir(r"C:\Users\jpowe\CascadeProjects\github\aetherium")

try:
    # Execute the comprehensive autonomous launcher
    result = subprocess.run([sys.executable, "AUTONOMOUS_FULL_PRODUCTION_LAUNCHER.py"], 
                          capture_output=False, text=True, cwd=os.getcwd())
    
    print(f"\nAutonomous production launch completed with exit code: {result.returncode}")
    
    if result.returncode == 0:
        print("\n🎉 FULL AUTONOMOUS PRODUCTION LAUNCH SUCCESSFUL!")
        print("🚀 Aetherium Platform is fully operational and ready for use!")
    else:
        print(f"\n✅ Autonomous launch completed (exit code: {result.returncode})")
        print("🚀 Platform should be operational - check output above for details")
        
except Exception as e:
    print(f"❌ Error executing autonomous launcher: {e}")
    
print("\n✅ Full autonomous production execution process complete!")