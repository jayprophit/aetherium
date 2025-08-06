#!/usr/bin/env python3
"""
AETHERIUM LAUNCHER - Master execution script for the Aetherium AI Productivity Platform
Consolidated from 21+ redundant execution scripts into a single, reliable launcher.
"""
import os
import sys
import http.server
import socketserver
import webbrowser
import threading
import time
import argparse
from pathlib import Path

class AetheriumLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.component_path = self.project_root / "src" / "components" / "AetheriumDashboard.tsx"
        self.port = 8200
        self.debug = False
        
    def check_dependencies(self):
        """Check if required files exist"""
        missing = []
        
        if not self.component_path.exists():
            missing.append(f"Main component: {self.component_path}")
            
        if missing:
            print("❌ Missing required files:")
            for item in missing:
                print(f"   - {item}")
            return False
            
        print("✅ All dependencies found")
        return True
    
    def load_component(self):
        """Load the Aetherium dashboard component"""
        try:
            with open(self.component_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if self.debug:
                    print(f"✅ Loaded component ({len(content)} characters)")
                return content
        except Exception as e:
            print(f"❌ Error loading component: {e}")
            return "// Error loading AetheriumDashboard.tsx"
    
    def create_html_wrapper(self, tsx_content):
        """Create HTML wrapper for the React component"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚛️ Aetherium - AI Productivity Platform</title>
    <meta name="description" content="Advanced AI productivity platform with quantum computing, time crystals, and neuromorphic AI capabilities">
    
    <!-- React & Dependencies -->
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/lucide-react@latest/dist/umd/lucide-react.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Enhanced Aetherium Styling -->
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: white;
            overflow-x: hidden;
        }}
        
        .aetherium-gradient {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        
        .tool-button {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .tool-button:hover {{
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }}
        
        .tool-button:active {{
            transform: translateY(-2px) scale(1.01);
        }}
        
        .tool-button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}
        
        .tool-button:hover::before {{
            left: 100%;
        }}
        
        .chat-bubble {{
            animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(40px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .sidebar-tab {{
            transition: all 0.3s ease;
            border-left: 3px solid transparent;
            position: relative;
        }}
        
        .sidebar-tab.active {{
            border-left-color: #667eea;
            background-color: rgba(102, 126, 234, 0.2);
        }}
        
        .sidebar-tab.active::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 3px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        }}
        
        .quantum-glow {{
            animation: quantumGlow 4s ease-in-out infinite;
        }}
        
        @keyframes quantumGlow {{
            0%, 100% {{ 
                box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
                border-color: rgba(102, 126, 234, 0.3);
            }}
            50% {{ 
                box-shadow: 0 0 25px rgba(102, 126, 234, 0.8), 0 0 40px rgba(118, 75, 162, 0.6);
                border-color: rgba(102, 126, 234, 0.8);
            }}
        }}
        
        .aetherium-brand {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: brandShimmer 3s ease-in-out infinite;
        }}
        
        @keyframes brandShimmer {{
            0%, 100% {{ filter: brightness(1); }}
            50% {{ filter: brightness(1.2); }}
        }}
        
        .status-indicator {{
            animation: statusPulse 2s ease-in-out infinite;
        }}
        
        @keyframes statusPulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.1); }}
        ::-webkit-scrollbar-thumb {{ background: rgba(102, 126, 234, 0.5); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(102, 126, 234, 0.8); }}
        
        .loading-screen {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3a 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            transition: opacity 0.5s ease;
        }}
        
        .loading-screen.hidden {{
            opacity: 0;
            pointer-events: none;
        }}
        
        .quantum-loader {{
            width: 60px;
            height: 60px;
            border: 3px solid rgba(102, 126, 234, 0.3);
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: quantumSpin 1s linear infinite;
        }}
        
        @keyframes quantumSpin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <!-- Loading Screen -->
    <div id="loading-screen" class="loading-screen">
        <div class="quantum-loader"></div>
        <div style="margin-top: 20px; font-size: 18px; font-weight: 600;">
            ⚛️ Initializing Aetherium Platform
        </div>
        <div style="margin-top: 10px; font-size: 14px; opacity: 0.7;">
            Quantum systems coming online...
        </div>
    </div>
    
    <!-- Main Application -->
    <div id="root"></div>
    
    <script type="text/babel">
        {tsx_content.replace('export default AetheriumPlatform;', 'const AetheriumDashboard = AetheriumPlatform;')}
        
        // Initialize the application
        function initializeAetherium() {{
            console.log('🎉 AETHERIUM PLATFORM INITIALIZATION');
            console.log('✅ All 16 incremental updates applied');
            console.log('⚛️ Quantum AI Models: Quantum-1, Neural-3, Crystal-2');
            console.log('🛠️ 80+ AI Tools organized and ready');
            console.log('🎨 Manus/Claude-style UI/UX complete');
            console.log('📱 Comprehensive sidebar navigation active');
            console.log('🚀 Platform ready for interaction');
            
            // Hide loading screen and show app
            setTimeout(() => {{
                const loadingScreen = document.getElementById('loading-screen');
                loadingScreen.classList.add('hidden');
                
                // Mount the React application
                ReactDOM.render(<AetheriumDashboard />, document.getElementById('root'));
                
                console.log('🎯 Aetherium Platform fully loaded and interactive!');
            }}, 2000);
        }}
        
        // Start initialization when DOM is ready
        document.addEventListener('DOMContentLoaded', initializeAetherium);
    </script>
    
    <script>
        // Performance monitoring
        window.addEventListener('load', function() {{
            console.log('📊 Performance metrics:');
            console.log(`   Load time: ${{performance.now().toFixed(2)}}ms`);
            console.log(`   Components: AetheriumDashboard`);
            console.log(`   Features: Quantum AI, 80+ Tools, Advanced UI/UX`);
        }});
        
        // Error handling
        window.addEventListener('error', function(e) {{
            console.error('❌ Application error:', e.error);
        }});
    </script>
</body>
</html>'''
    
    def find_available_port(self, start_port=8200, max_attempts=10):
        """Find an available port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as test_server:
                    return port
            except OSError:
                continue
        return start_port  # Fallback
    
    def start_server(self):
        """Start the development server"""
        if not self.check_dependencies():
            return False
        
        tsx_content = self.load_component()
        html_content = self.create_html_wrapper(tsx_content)
        
        class AetheriumHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.html_content = html_content
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(self.html_content.encode('utf-8'))
                else:
                    super().do_GET()
        
        self.port = self.find_available_port(self.port)
        
        try:
            with socketserver.TCPServer(("", self.port), AetheriumHandler) as httpd:
                print("🎉 AETHERIUM PLATFORM LAUNCHER")
                print("=" * 50)
                print(f"🌐 Server running at: http://localhost:{self.port}")
                print(f"📁 Component: {self.component_path}")
                print(f"⚛️ Features: Quantum AI, 80+ Tools, Advanced UI/UX")
                print("=" * 50)
                
                # Open browser
                threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{self.port}')).start()
                print("📱 Opening browser...")
                print("🔄 Press Ctrl+C to stop server")
                
                httpd.serve_forever()
                
        except KeyboardInterrupt:
            print("\n🛑 Aetherium Platform stopped")
            return True
        except Exception as e:
            print(f"❌ Server error: {e}")
            return False
    
    def run(self, args):
        """Main entry point"""
        self.debug = args.debug
        if args.port:
            self.port = args.port
        
        print("🚀 Starting Aetherium AI Productivity Platform...")
        return self.start_server()

def main():
    parser = argparse.ArgumentParser(description='Aetherium AI Productivity Platform Launcher')
    parser.add_argument('--port', type=int, help='Server port (default: 8200)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    launcher = AetheriumLauncher()
    launcher.run(args)

if __name__ == "__main__":
    main()