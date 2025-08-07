#!/usr/bin/env python3
import subprocess
import sys
import os

print("🗂️ EXECUTING AETHERIUM DIRECTORY REORGANIZATION NOW...")
print("="*60)

# Change to the correct directory
os.chdir(r"C:\Users\jpowe\CascadeProjects\github\aetherium")

# Execute the reorganization
try:
    result = subprocess.run([sys.executable, "EXECUTE_REORGANIZATION.py"], 
                          capture_output=True, text=True, cwd=os.getcwd())
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\nReorganization completed with exit code: {result.returncode}")
    
    if result.returncode == 0:
        print("🎉 DIRECTORY REORGANIZATION SUCCESSFUL!")
    else:
        print("⚠️ Reorganization completed with warnings")
        
except Exception as e:
    print(f"❌ Error executing reorganization: {e}")
    
print("\n✅ Directory organization process complete!")