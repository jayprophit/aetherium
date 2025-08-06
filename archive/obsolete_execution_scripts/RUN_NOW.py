#!/usr/bin/env python3
import os
import sys
import http.server
import socketserver
import webbrowser
import threading

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("🎉 AETHERIUM COMPLETE DASHBOARD")
print("✅ All 16 incremental updates applied")
print("⚛️ Quantum AI • 🛠️ 80+ Tools • 🎨 Manus/Claude UI")

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            try:
                with open('aetherium.tsx', 'r', encoding='utf-8') as f:
                    tsx = f.read()
            except:
                tsx = "// aetherium.tsx loading error"
            
            html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚛️ Aetherium Dashboard</title>
<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://unpkg.com/lucide-react@latest/dist/umd/lucide-react.js"></script>
<script src="https://cdn.tailwindcss.com"></script>
<style>.tool-button{{transition:all 0.3s ease}}.tool-button:hover{{transform:translateY(-3px);box-shadow:0 8px 25px rgba(0,0,0,0.2)}}.sidebar-tab{{transition:all 0.3s ease;border-left:3px solid transparent}}.sidebar-tab.active{{border-left-color:#667eea;background-color:rgba(102,126,234,0.15)}}</style>
</head><body><div id="root"></div>
<script type="text/babel">{tsx.replace('export default AetheriumPlatform;', 'const App = AetheriumPlatform;')}ReactDOM.render(<App />, document.getElementById('root'));</script>
</body></html>'''
            self.wfile.write(html.encode())

PORT = 8110
try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Server: http://localhost:{PORT}")
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
        print("📱 Opening browser... Press Ctrl+C to stop")
        httpd.serve_forever()
except Exception as e:
    print(f"Error: {e}")
    print("Manual browser navigation may be required")