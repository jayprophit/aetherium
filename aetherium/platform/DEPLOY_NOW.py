#!/usr/bin/env python3
"""DEPLOY NOW - Immediate Execution"""
import subprocess
import sys
import os

print("🚀 DEPLOYING AETHERIUM NOW...")
print("=" * 50)

# Change to correct directory
os.chdir(r"C:\Users\jpowe\CascadeProjects\github\aetherium\aetherium\platform")

# Execute the deployment
try:
    result = subprocess.run([sys.executable, "INSTANT_DEPLOY.py"], 
                          capture_output=False, text=True, check=False)
    print(f"Deployment result: {result.returncode}")
except Exception as e:
    print(f"Deployment error: {e}")
    
print("Deployment script executed.")