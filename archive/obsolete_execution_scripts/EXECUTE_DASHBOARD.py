#!/usr/bin/env python3
import subprocess
import sys
import os

# Execute the dashboard
try:
    print("🚀 Launching Aetherium Dashboard...")
    result = subprocess.run([sys.executable, "QUICK_DASHBOARD_RUN.py"], 
                          cwd=os.path.dirname(os.path.abspath(__file__)),
                          timeout=None)
except Exception as e:
    print(f"Error: {e}")