#!/usr/bin/env python3
"""
Simple backend starter - bypasses complex deployment logic
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

print("🚀 Starting Aetherium Backend Server...")
print(f"📁 Backend path: {backend_path}")
print(f"🐍 Python path: {sys.path[0]}")

try:
    # Change to backend directory
    os.chdir(backend_path)
    print(f"✓ Changed to: {os.getcwd()}")
    
    # Import and start the FastAPI app
    print("📦 Importing FastAPI app...")
    
    # Create minimal FastAPI app if main.py has issues
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(title="Aetherium AI Platform", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "🎉 Aetherium AI Productivity Suite is running!",
            "status": "operational",
            "version": "1.0.0"
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "Aetherium Backend"}
    
    # Try to import AI productivity suite if available
    try:
        print("🤖 Loading AI Productivity Suite...")
        from ai_productivity_suite.suite_manager import AISuiteManager
        suite_manager = AISuiteManager()
        
        @app.get("/api/suite/status")
        async def suite_status():
            return await suite_manager.get_suite_status()
        
        @app.get("/api/suite/health")
        async def suite_health():
            return await suite_manager.health_check()
            
        print("✅ AI Productivity Suite loaded successfully!")
    except Exception as e:
        print(f"⚠️ AI Productivity Suite not available: {e}")
    
    # Start the server
    print("\n🌟 Starting server on http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Installing required dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    print("✅ Dependencies installed. Please run the script again.")
except Exception as e:
    print(f"❌ Error starting backend: {e}")
    import traceback
    traceback.print_exc()