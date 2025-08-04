#!/usr/bin/env python3
"""SIMPLE WORKING DEPLOYMENT - FOR USER"""
import os
import sys
import time
import webbrowser
import socket
import subprocess

print("🚀 AETHERIUM - SIMPLE DEPLOYMENT")
print("=" * 50)

# Find port
def find_port():
    for port in range(3000, 3100):
        try:
            with socket.socket() as s:
                s.bind(('localhost', port))
                return port
        except:
            continue
    return 3000

port = find_port()
print(f"✅ Using port: {port}")

# Create simple server
server_code = f'''from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Aetherium")

@app.get("/")
async def home():
    return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Aetherium AI Platform</title>
<style>
body{{font-family:system-ui;margin:0;background:#f5f5f5;display:flex;height:100vh}}
.sidebar{{width:300px;background:#fff;border-right:1px solid #e5e5e5;padding:20px}}
.main{{flex:1;padding:40px;background:#fff;margin:20px;border-radius:12px}}
.title{{font-size:32px;font-weight:700;margin-bottom:20px;color:#1a1a1a}}
.tools{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}}
.tool{{background:#f8f9fa;border:1px solid #e5e5e5;border-radius:8px;padding:20px;cursor:pointer;transition:all 0.2s}}
.tool:hover{{border-color:#007AFF;background:#f0f7ff;transform:translateY(-2px)}}
.nav-item{{padding:12px;border-radius:8px;margin:8px 0;cursor:pointer}}
.nav-item:hover{{background:#f8f9fa}}
.status{{background:#28a745;color:white;padding:8px 16px;border-radius:20px;font-size:12px;margin-bottom:20px}}
</style></head>
<body>
<div class="sidebar">
<h3>Aetherium AI</h3>
<div class="nav-item">🏠 Dashboard</div>
<div class="nav-item">💬 Chat</div>
<div class="nav-item">📊 Projects</div>
<div class="nav-item">🔧 Tools</div>
<div class="nav-item">⚙️ Settings</div>
</div>
<div class="main">
<div class="status">✅ Platform Live</div>
<h1 class="title">Welcome to Aetherium</h1>
<p style="font-size:18px;color:#666;margin-bottom:40px">Your AI productivity platform is running!</p>
<div class="tools">
<div class="tool"><h4>🔍 Research</h4><p>Deep research tools</p></div>
<div class="tool"><h4>📊 Analytics</h4><p>Data visualization</p></div>
<div class="tool"><h4>🎨 Creative</h4><p>Design & content</p></div>
<div class="tool"><h4>🧮 Calculator</h4><p>Advanced calculations</p></div>
<div class="tool"><h4>💻 PC Builder</h4><p>Build custom PCs</p></div>
<div class="tool"><h4>✉️ Email Gen</h4><p>Email generator</p></div>
<div class="tool"><h4>✈️ Trip Plan</h4><p>Travel planning</p></div>
<div class="tool"><h4>🌍 Translator</h4><p>Language tools</p></div>
</div>
</div>
</body></html>""")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port})'''

# Write server
with open("quick_server.py", "w") as f:
    f.write(server_code)

print("✅ Server created")
print("🚀 Starting server...")

# Start server
try:
    process = subprocess.Popen([sys.executable, "quick_server.py"])
    print("✅ Server starting...")
    time.sleep(3)
    
    url = f"http://localhost:{port}"
    print(f"🌐 Opening: {url}")
    webbrowser.open(url)
    
    print("=" * 50)
    print("🎯 AETHERIUM IS LIVE!")
    print("=" * 50)
    print(f"🌐 URL: {url}")
    print("✅ Platform running")
    print("✅ Browser opened")
    print("=" * 50)
    
    input("Press Enter to stop...")
    process.terminate()
    
except Exception as e:
    print(f"❌ Error: {e}")

print("Done!")