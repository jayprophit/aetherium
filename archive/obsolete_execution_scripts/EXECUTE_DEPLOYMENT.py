#!/usr/bin/env python3
"""
EXECUTE DEPLOYMENT - Direct Platform Launch
"""
import subprocess
import sys
import os

def main():
    print("🚀 EXECUTING AETHERIUM PLATFORM DEPLOYMENT")
    print("=" * 50)
    
    # Change to correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {script_dir}")
    print("🎯 Launching WORKING_DEPLOYMENT.py...")
    print("=" * 50)
    
    try:
        # Execute the deployment script
        result = subprocess.run([sys.executable, "WORKING_DEPLOYMENT.py"], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("✅ Platform deployed successfully!")
        else:
            print(f"⚠️ Platform exit code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Platform stopped by user")
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        print("\n🔄 Trying alternative launch method...")
        
        # Fallback method
        try:
            import http.server
            import socketserver
            import webbrowser
            import threading
            import time
            
            port = 8080
            print(f"🌐 Starting fallback server on port {port}...")
            
            def open_browser():
                time.sleep(2)
                webbrowser.open(f"http://localhost:{port}")
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            with socketserver.TCPServer(("localhost", port), http.server.SimpleHTTPRequestHandler) as httpd:
                print(f"✅ Fallback server running: http://localhost:{port}")
                print("🌐 Opening browser...")
                httpd.serve_forever()
                
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")

if __name__ == "__main__":
    main()