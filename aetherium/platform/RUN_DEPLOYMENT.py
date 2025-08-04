#!/usr/bin/env python3
"""RUN DEPLOYMENT - Execute for User"""
import os
import subprocess
import sys

print("🚀 RUNNING AETHERIUM DEPLOYMENT FOR USER")
print("=" * 50)

# Change to platform directory
platform_dir = r"C:\Users\jpowe\CascadeProjects\github\aetherium\aetherium\platform"
os.chdir(platform_dir)
print(f"Working in: {platform_dir}")

# Try to run INSTANT_DEPLOY.py
try:
    print("Executing INSTANT_DEPLOY.py...")
    result = subprocess.run([sys.executable, "INSTANT_DEPLOY.py"], 
                          capture_output=False, text=True)
    print(f"Deployment completed with return code: {result.returncode}")
except Exception as e:
    print(f"Error running INSTANT_DEPLOY.py: {e}")
    
    # Fallback - try a simpler deployment
    print("Trying fallback deployment...")
    try:
        result = subprocess.run([sys.executable, "WORKING_LAUNCH.py"], 
                              capture_output=False, text=True)
        print(f"Fallback deployment completed: {result.returncode}")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        print("Will create simple working version...")

print("✅ Deployment execution attempted")
print("🌐 Check your browser for the platform")