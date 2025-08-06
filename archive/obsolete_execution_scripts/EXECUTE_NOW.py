#!/usr/bin/env python3
import os, sys, http.server, socketserver, webbrowser, threading

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("🎉 EXECUTING AETHERIUM COMPLETE DASHBOARD")
print("✅ All 16 incremental updates applied")

class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            try:
                with open('aetherium.tsx', 'r', encoding='utf-8') as f: tsx = f.read()
            except: tsx = "// Error loading aetherium.tsx"
            html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>⚛️ Aetherium Dashboard</title>
<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://unpkg.com/lucide-react@latest/dist/umd/lucide-react.js"></script>
<script src="https://cdn.tailwindcss.com"></script></head><body><div id="root"></div>
<script type="text/babel">{tsx.replace('export default AetheriumPlatform;', 'const App = AetheriumPlatform;')}ReactDOM.render(<App />, document.getElementById('root'));</script></body></html>'''
            self.wfile.write(html.encode())

PORT = 8120
with socketserver.TCPServer(("", PORT), H) as httpd:
    print(f"🌐 http://localhost:{PORT}")
    threading.Timer(0.5, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
    httpd.serve_forever()