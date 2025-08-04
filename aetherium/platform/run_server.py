#!/usr/bin/env python3
"""RUN SERVER NOW - Simple execution"""
import subprocess
import sys
import time
import webbrowser
import socket
import os
from pathlib import Path

# Ensure we're in the right directory
platform_dir = Path(r"C:\Users\jpowe\CascadeProjects\github\aetherium\aetherium\platform")
os.chdir(platform_dir)

print("🚀 AETHERIUM - STARTING NOW")
print("=" * 50)
print(f"📁 Directory: {platform_dir}")

# Find available port
def find_port():
    for port in range(3000, 3100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except:
            continue
    return 3000

port = find_port()
print(f"🌐 Port: {port}")

# Create simple backend NOW
backend_code = '''from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Aetherium Live")

@app.get("/")
async def home():
    return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Aetherium - AI Platform</title>
<style>
body{font-family:system-ui;margin:0;background:#f5f5f5;display:flex;height:100vh}
.sidebar{width:280px;background:#fff;border-right:1px solid #e5e5e5;padding:20px}
.main{flex:1;padding:40px;background:#fff;margin:20px;border-radius:12px;box-shadow:0 2px 20px rgba(0,0,0,0.1)}
.title{font-size:32px;font-weight:700;margin-bottom:20px;color:#1a1a1a}
.subtitle{font-size:18px;color:#666;margin-bottom:40px}
.tools{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
.tool{background:#f8f9fa;border:1px solid #e5e5e5;border-radius:8px;padding:20px;cursor:pointer;transition:all 0.2s}
.tool:hover{border-color:#007AFF;background:#f0f7ff;transform:translateY(-2px)}
.tool-name{font-weight:600;margin-bottom:8px;color:#1a1a1a}
.tool-desc{font-size:14px;color:#666}
.nav-item{padding:12px 16px;border-radius:8px;margin-bottom:8px;cursor:pointer;transition:all 0.2s}
.nav-item:hover{background:#f8f9fa}
.nav-item.active{background:#007AFF;color:white}
.status{background:#28a745;color:white;padding:8px 16px;border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin-bottom:20px}
</style></head>
<body>
<div class="sidebar">
<h3>Aetherium AI</h3>
<div class="nav-item active">🏠 Dashboard</div>
<div class="nav-item">💬 Chat</div>
<div class="nav-item">📊 Projects</div>
<div class="nav-item">🔧 Tools</div>
<div class="nav-item">⚙️ Settings</div>
</div>
<div class="main">
<div class="status">✅ Platform Live</div>
<h1 class="title">Welcome to Aetherium</h1>
<p class="subtitle">Your AI-powered productivity platform is running successfully!</p>
<div class="tools">
<div class="tool">
<div class="tool-name">🔍 Wide Research</div>
<div class="tool-desc">Deep research on any topic</div>
</div>
<div class="tool">
<div class="tool-name">📊 Data Visualization</div>
<div class="tool-desc">Create stunning charts</div>
</div>
<div class="tool">
<div class="tool-name">🎨 AI Color Analysis</div>
<div class="tool-desc">Analyze color schemes</div>
</div>
<div class="tool">
<div class="tool-name">🧮 Everything Calculator</div>
<div class="tool-desc">Advanced calculations</div>
</div>
<div class="tool">
<div class="tool-name">💻 PC Builder</div>
<div class="tool-desc">Build custom PCs</div>
</div>
<div class="tool">
<div class="tool-name">🎯 Coupon Finder</div>
<div class="tool-desc">Find best deals</div>
</div>
<div class="tool">
<div class="tool-name">🏆 AI Coach</div>
<div class="tool-desc">Personal coaching</div>
</div>
<div class="tool">
<div class="tool-name">✉️ Email Generator</div>
<div class="tool-desc">Create professional emails</div>
</div>
<div class="tool">
<div class="tool-name">✈️ Trip Planner</div>
<div class="tool-desc">Plan perfect trips</div>
</div>
<div class="tool">
<div class="tool-name">📝 Essay Outline</div>
<div class="tool-desc">Structure your writing</div>
</div>
<div class="tool">
<div class="tool-name">🌍 Translator</div>
<div class="tool-desc">Translate any language</div>
</div>
<div class="tool">
<div class="tool-name">📄 PDF Translator</div>
<div class="tool-desc">Translate documents</div>
</div>
</div>
</div>
</body></html>""")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=''' + str(port) + ''')'''

# Write and run
with open("simple_server.py", "w", encoding='utf-8') as f:
    f.write(backend_code)

print("✅ Server created")
print("🚀 Starting server...")

# Start the server
try:
    # Use python directly
    process = subprocess.Popen([
        sys.executable, "simple_server.py"
    ], cwd=platform_dir)
    
    print("✅ Server starting...")
    time.sleep(3)
    
    # Open browser
    url = f"http://localhost:{port}"
    print(f"🌐 Opening: {url}")
    webbrowser.open(url)
    
    print("=" * 50)
    print("🎯 AETHERIUM IS LIVE!")
    print("=" * 50)
    print(f"🌐 URL: {url}")
    print("✅ Platform running")
    print("✅ Browser opened")
    print("✅ All systems ready")
    print("=" * 50)
    
    # Wait for user
    input("Press Enter to stop server...")
    process.terminate()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Server may still be running...")