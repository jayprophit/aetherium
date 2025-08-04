#!/usr/bin/env python3
"""
Quick status checker for Aetherium deployment
"""

import os
import requests
import subprocess
from pathlib import Path

print("🔍 AETHERIUM DEPLOYMENT STATUS CHECK")
print("=" * 40)

# Check if files were created
backend_exists = Path("backend/main.py").exists()
frontend_exists = Path("frontend_simple/index.html").exists()

print(f"📁 Backend files: {'✅ EXISTS' if backend_exists else '❌ MISSING'}")
print(f"📁 Frontend files: {'✅ EXISTS' if frontend_exists else '❌ MISSING'}")

# Check if backend is running
try:
    response = requests.get("http://localhost:8000", timeout=3)
    if response.status_code == 200:
        print("🚀 Backend server: ✅ RUNNING")
        data = response.json()
        print(f"   Message: {data.get('message', 'N/A')}")
        print(f"   AI Tools: {data.get('ai_tools', 'N/A')}")
    else:
        print(f"🚀 Backend server: ⚠️ RESPONDING ({response.status_code})")
except requests.exceptions.ConnectRefused:
    print("🚀 Backend server: ❌ NOT RUNNING")
except requests.exceptions.Timeout:
    print("🚀 Backend server: ⏳ TIMEOUT")
except Exception as e:
    print(f"🚀 Backend server: ❌ ERROR ({e})")

# Check processes
try:
    result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
    if ':8000' in result.stdout:
        print("🌐 Port 8000: ✅ IN USE")
    else:
        print("🌐 Port 8000: ❌ NOT IN USE")
except:
    print("🌐 Port check: ❌ FAILED")

print("\n" + "=" * 40)

# Provide next steps
if backend_exists and frontend_exists:
    print("✅ Files created successfully!")
    try:
        response = requests.get("http://localhost:8000", timeout=2)
        print("🎉 PLATFORM IS RUNNING!")
        print("\n🌐 Access your platform:")
        print("   • Frontend: Open frontend_simple/index.html in browser")
        print("   • Backend: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
    except:
        print("⚠️ Platform created but backend not responding")
        print("\n🔧 Try starting backend manually:")
        print("   cd backend")
        print("   python main.py")
else:
    print("❌ Deployment incomplete")
    print("\n🔧 Try running deployment again:")
    print("   python simple_deploy.py")

print("\n🛠️ For manual start:")
print("   1. cd backend && python main.py")
print("   2. Open frontend_simple/index.html in browser")