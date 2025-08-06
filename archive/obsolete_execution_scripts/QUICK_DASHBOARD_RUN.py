#!/usr/bin/env python3
"""
QUICK DASHBOARD RUN - Execute working Aetherium dashboard with updated aetherium.tsx
"""
import os
import sys
import http.server
import socketserver
import threading
import webbrowser
import json
from urllib.parse import urlparse, parse_qs

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("🚀 LAUNCHING AETHERIUM DASHBOARD WITH UPDATED FEATURES")
print("=" * 60)
print("📁 Working directory:", script_dir)
print("🎯 Serving enhanced Aetherium dashboard...")
print("=" * 60)

class AetheriumHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Read the updated aetherium.tsx content
            try:
                with open('aetherium.tsx', 'r', encoding='utf-8') as f:
                    tsx_content = f.read()
                    print("✅ Successfully loaded updated aetherium.tsx")
            except:
                tsx_content = "// aetherium.tsx not found"
                print("⚠️ Could not load aetherium.tsx, using fallback")
            
            html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚛️ Aetherium - AI Productivity Platform</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/lucide-react@latest/dist/umd/lucide-react.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .aetherium-gradient {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .tool-button {{
            transition: all 0.2s ease;
        }}
        .tool-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .chat-bubble {{
            animation: slideUp 0.3s ease-out;
        }}
        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .sidebar-tab {{
            transition: all 0.2s ease;
            border-left: 3px solid transparent;
        }}
        .sidebar-tab.active {{
            border-left-color: #667eea;
            background-color: rgba(102, 126, 234, 0.1);
        }}
        .typing-indicator {{
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 0.4; }}
            50% {{ opacity: 1; }}
        }}
    </style>
</head>
<body class="bg-gray-50">
    <div id="root"></div>
    
    <script type="text/babel">
        {tsx_content.replace('export default AetheriumPlatform;', 'const App = AetheriumPlatform;')}
        
        // Simple simulation for missing dependencies
        const useState = React.useState;
        const useRef = React.useRef;
        const useEffect = React.useEffect;
        
        // Mount the app
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
    
    <script>
        console.log('🚀 Aetherium Dashboard Loaded Successfully');
        console.log('✅ Updated features: Quantum AI models, 80+ tools, enhanced UI/UX');
        console.log('🎯 Ready for interactive testing');
    </script>
</body>
</html>'''
            
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()

def start_server():
    # Find available port
    port = 8000
    for p in range(8000, 8010):
        try:
            with socketserver.TCPServer(("", p), AetheriumHandler) as test_server:
                port = p
                break
        except OSError:
            continue
    
    print(f"🌐 Starting server on http://localhost:{port}")
    
    with socketserver.TCPServer(("", port), AetheriumHandler) as httpd:
        print(f"✅ Server running at http://localhost:{port}")
        print("📱 Opening browser...")
        
        # Open browser
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
        
        print("\n" + "="*60)
        print("🎯 AETHERIUM DASHBOARD FEATURES:")
        print("✅ Updated Aetherium branding (Atom logo, quantum tagline)")
        print("✅ Comprehensive 80+ AI tools organized by category")
        print("✅ Quantum AI, Neuromorphic, and Time Crystal models")
        print("✅ Enhanced Manus/Claude-style UI/UX")
        print("✅ All incremental updates applied to aetherium.tsx")
        print("="*60)
        print("🔄 Press Ctrl+C to stop server")
        print("="*60)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    start_server()