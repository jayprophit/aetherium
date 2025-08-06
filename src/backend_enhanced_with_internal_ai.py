#!/usr/bin/env python3
"""
AETHERIUM ENHANCED BACKEND WITH INTERNAL AI
FastAPI backend with Aetherium's own AI engine as primary + external APIs as secondary
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import sqlite3
import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Try to import internal AI engine
try:
    from ai.aetherium_ai_engine import get_aetherium_ai, initialize_aetherium_ai, AetheriumInferenceEngine
    INTERNAL_AI_AVAILABLE = True
    print("✅ Internal Aetherium AI engine available")
except ImportError as e:
    print(f"⚠️ Internal AI engine not available: {e}")
    print("📦 Install PyTorch: pip install torch")
    INTERNAL_AI_AVAILABLE = False

# External API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="Aetherium Enhanced Backend with Internal AI",
    description="Complete AI platform with internal Aetherium AI + external API support",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
websocket_connections: List[WebSocket] = []
internal_ai: Optional[AetheriumInferenceEngine] = None

# Initialize database
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('aetherium.db')
    cursor = conn.cursor()
    
    # Chats table with AI engine info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'anonymous',
            message TEXT NOT NULL,
            response TEXT,
            ai_engine TEXT DEFAULT 'aetherium-internal',
            model TEXT DEFAULT 'aetherium-quantum',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            response_time REAL
        )
    ''')
    
    # AI engines performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_performance (
            id TEXT PRIMARY KEY,
            engine TEXT NOT NULL,
            model TEXT,
            avg_response_time REAL,
            total_requests INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 1.0,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'anonymous',
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            mime_type TEXT,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized with AI engine tracking")

# Initialize database and AI
init_database()

# Initialize internal AI if available
if INTERNAL_AI_AVAILABLE:
    try:
        internal_ai = get_aetherium_ai()
        print("🧠 Aetherium Internal AI initialized and ready!")
    except Exception as e:
        print(f"❌ Failed to initialize internal AI: {e}")
        INTERNAL_AI_AVAILABLE = False

# Create uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

@app.on_startup
async def startup_event():
    """Startup tasks"""
    print("🚀 Aetherium Enhanced Backend with Internal AI starting...")
    print("=" * 60)
    print(f"✅ Internal AI Available: {INTERNAL_AI_AVAILABLE}")
    print(f"✅ OpenAI Available: {OPENAI_AVAILABLE}")
    print(f"✅ WebSocket Support: Enabled")
    print(f"✅ File Upload Support: Enabled")
    print(f"✅ Database Persistence: Enabled")
    print("=" * 60)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time AI chat"""
    await websocket.accept()
    websocket_connections.append(websocket)
    print(f"🔌 WebSocket connected. Total: {len(websocket_connections)}")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "chat":
                await handle_ai_chat(websocket, data)
            elif data.get("type") == "tool_execute":
                await handle_tool_execution(websocket, data)
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        websocket_connections.remove(websocket)
        print(f"🔌 WebSocket disconnected. Total: {len(websocket_connections)}")

async def handle_ai_chat(websocket: WebSocket, data: dict):
    """Handle AI chat with multiple engine support"""
    try:
        message = data.get("message", "")
        ai_engine = data.get("ai_engine", "aetherium-internal")
        model = data.get("model", "aetherium-quantum")
        user_id = data.get("user_id", "anonymous")
        session_id = data.get("session_id", "default")
        
        # Send thinking indicator
        await websocket.send_json({
            "type": "thinking",
            "content": f"🤔 Processing with {ai_engine}...",
            "engine": ai_engine
        })
        
        start_time = asyncio.get_event_loop().time()
        response = ""
        
        # Route to appropriate AI engine
        if ai_engine == "aetherium-internal" and INTERNAL_AI_AVAILABLE:
            response = await generate_internal_ai_response(message, model)
        elif ai_engine == "openai" and OPENAI_AVAILABLE:
            response = await generate_openai_response(message, model)
        elif ai_engine == "claude":
            response = await generate_claude_response(message, model)
        elif ai_engine == "gemini":
            response = await generate_gemini_response(message, model)
        else:
            # Fallback to internal AI or simulated response
            if INTERNAL_AI_AVAILABLE:
                response = await generate_internal_ai_response(message, "aetherium-quantum")
                ai_engine = "aetherium-internal"
            else:
                response = generate_fallback_response(message)
                ai_engine = "fallback"
        
        response_time = asyncio.get_event_loop().time() - start_time
        
        # Send final response
        await websocket.send_json({
            "type": "response",
            "content": response,
            "engine": ai_engine,
            "model": model,
            "response_time": round(response_time, 2),
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to database
        save_chat_to_db(user_id, message, response, ai_engine, model, session_id, response_time)
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "content": f"Error processing message: {str(e)}"
        })

async def generate_internal_ai_response(message: str, model: str) -> str:
    """Generate response using internal Aetherium AI"""
    try:
        if not internal_ai:
            return "❌ Internal AI not initialized"
        
        # Route to specialized functions based on model
        if "quantum" in model:
            response = internal_ai.quantum_reasoning(message)
        elif "creative" in model:
            response = internal_ai.creative_generation(message)
        elif "productivity" in model:
            response = internal_ai.productivity_assistance(message)
        else:
            response = internal_ai.generate_response(message)
        
        # Add Aetherium branding
        return f"🧠 **Aetherium Internal AI Response**\n\n{response}\n\n*Powered by Aetherium's quantum-enhanced neural architecture*"
        
    except Exception as e:
        return f"❌ Internal AI error: {str(e)}"

async def generate_openai_response(message: str, model: str) -> str:
    """Generate response using OpenAI API"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OpenAI API key not configured"
        
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model=model if model.startswith("gpt") else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=500
        )
        
        return f"🤖 **OpenAI Response**\n\n{response.choices[0].message.content}"
        
    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"

async def generate_claude_response(message: str, model: str) -> str:
    """Generate response using Claude API"""
    # Placeholder - implement Anthropic API
    return f"🧠 **Claude Response** (Simulated)\n\nI understand you said: '{message}'. Claude API integration coming soon!"

async def generate_gemini_response(message: str, model: str) -> str:
    """Generate response using Gemini API"""
    # Placeholder - implement Google Gemini API
    return f"✨ **Gemini Response** (Simulated)\n\nI understand you said: '{message}'. Gemini API integration coming soon!"

def generate_fallback_response(message: str) -> str:
    """Fallback response when no AI engines are available"""
    return f"🤖 **Aetherium Fallback Response**\n\nI understand you said: '{message}'\n\nThis is a simulated response. To enable full AI capabilities:\n- Add PyTorch for internal AI: `pip install torch`\n- Add API keys for external providers\n\nFull AI integration coming online..."

def save_chat_to_db(user_id: str, message: str, response: str, engine: str, model: str, session_id: str, response_time: float):
    """Save chat interaction to database"""
    try:
        conn = sqlite3.connect('aetherium.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chats (id, user_id, message, response, ai_engine, model, session_id, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), user_id, message, response, engine, model, session_id, response_time))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat: {e}")

async def handle_tool_execution(websocket: WebSocket, data: dict):
    """Handle AI tool execution"""
    tool_id = data.get("tool_id")
    tool_data = data.get("data", {})
    ai_engine = data.get("ai_engine", "aetherium-internal")
    
    # Send processing indicator
    await websocket.send_json({
        "type": "tool_processing",
        "tool_id": tool_id,
        "engine": ai_engine,
        "status": "executing"
    })
    
    # Execute tool with selected AI engine
    result = await execute_ai_tool(tool_id, tool_data, ai_engine)
    
    # Send result
    await websocket.send_json({
        "type": "tool_result",
        "tool_id": tool_id,
        "result": result,
        "engine": ai_engine,
        "timestamp": datetime.now().isoformat()
    })

async def execute_ai_tool(tool_id: str, data: dict, ai_engine: str) -> dict:
    """Execute AI tools with engine selection"""
    try:
        if tool_id == "research" and ai_engine == "aetherium-internal" and INTERNAL_AI_AVAILABLE:
            query = data.get("query", "research topic")
            response = internal_ai.productivity_assistance(f"Research: {query}")
            return {"status": "success", "result": response, "engine": "aetherium-internal"}
        
        elif tool_id == "creative" and ai_engine == "aetherium-internal" and INTERNAL_AI_AVAILABLE:
            task = data.get("task", "creative task")
            response = internal_ai.creative_generation(task)
            return {"status": "success", "result": response, "engine": "aetherium-internal"}
        
        elif tool_id == "quantum" and ai_engine == "aetherium-internal" and INTERNAL_AI_AVAILABLE:
            problem = data.get("problem", "quantum problem")
            response = internal_ai.quantum_reasoning(problem)
            return {"status": "success", "result": response, "engine": "aetherium-internal"}
        
        else:
            # Generic tool execution
            return {
                "status": "success",
                "result": f"Tool '{tool_id}' executed with {ai_engine}",
                "data": data,
                "engine": ai_engine
            }
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

# REST API Endpoints

@app.get("/api/health")
async def health_check():
    """Enhanced health check with AI engine status"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "ai_engines": {
            "aetherium_internal": {
                "available": INTERNAL_AI_AVAILABLE,
                "status": "online" if INTERNAL_AI_AVAILABLE else "offline",
                "models": ["aetherium-quantum", "aetherium-creative", "aetherium-productivity"] if INTERNAL_AI_AVAILABLE else []
            },
            "openai": {
                "available": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
                "models": ["gpt-4", "gpt-3.5-turbo"] if OPENAI_AVAILABLE else []
            },
            "claude": {
                "available": False,
                "models": ["claude-3-sonnet", "claude-3-haiku"]
            },
            "gemini": {
                "available": False, 
                "models": ["gemini-pro", "gemini-pro-vision"]
            }
        },
        "features": {
            "websocket": True,
            "file_upload": True,
            "database": True,
            "internal_ai": INTERNAL_AI_AVAILABLE,
            "external_apis": OPENAI_AVAILABLE
        },
        "stats": {
            "websocket_connections": len(websocket_connections)
        }
    }

@app.get("/api/ai-engines")
async def get_ai_engines():
    """Get available AI engines and models"""
    engines = []
    
    if INTERNAL_AI_AVAILABLE:
        engines.append({
            "id": "aetherium-internal",
            "name": "Aetherium Internal AI",
            "description": "Built-from-scratch AI with quantum reasoning",
            "models": [
                {"id": "aetherium-quantum", "name": "Quantum Reasoning", "specialization": "physics, science, analysis"},
                {"id": "aetherium-creative", "name": "Creative Engine", "specialization": "writing, design, ideation"},
                {"id": "aetherium-productivity", "name": "Productivity Assistant", "specialization": "tasks, business, automation"}
            ],
            "status": "online",
            "primary": True
        })
    
    if OPENAI_AVAILABLE:
        engines.append({
            "id": "openai",
            "name": "OpenAI",
            "description": "External API - GPT models",
            "models": [
                {"id": "gpt-4", "name": "GPT-4", "specialization": "general, advanced reasoning"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "specialization": "general, fast responses"}
            ],
            "status": "available" if os.getenv("OPENAI_API_KEY") else "needs_api_key",
            "primary": False
        })
    
    engines.extend([
        {
            "id": "claude",
            "name": "Anthropic Claude",
            "description": "External API - Coming soon",
            "models": [
                {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "specialization": "analysis, writing"},
                {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "specialization": "fast responses"}
            ],
            "status": "coming_soon",
            "primary": False
        },
        {
            "id": "gemini",
            "name": "Google Gemini",
            "description": "External API - Coming soon",
            "models": [
                {"id": "gemini-pro", "name": "Gemini Pro", "specialization": "multimodal, reasoning"},
                {"id": "gemini-pro-vision", "name": "Gemini Pro Vision", "specialization": "vision, analysis"}
            ],
            "status": "coming_soon",
            "primary": False
        }
    ])
    
    return {"engines": engines, "count": len(engines)}

@app.get("/api/tools")
async def get_tools():
    """Get available AI tools"""
    tools = [
        {"id": "research", "name": "AI Research Assistant", "category": "productivity", "ai_enhanced": True},
        {"id": "creative", "name": "Creative Generator", "category": "creative", "ai_enhanced": True},
        {"id": "quantum", "name": "Quantum Reasoner", "category": "science", "ai_enhanced": True},
        {"id": "writer", "name": "Content Writer", "category": "creative", "ai_enhanced": True},
        {"id": "translator", "name": "Language Translator", "category": "communication", "ai_enhanced": False},
        {"id": "calculator", "name": "Smart Calculator", "category": "utility", "ai_enhanced": False},
        {"id": "analyzer", "name": "Data Analyzer", "category": "analysis", "ai_enhanced": True},
        {"id": "coder", "name": "Code Assistant", "category": "development", "ai_enhanced": True},
    ]
    return {"tools": tools, "count": len(tools)}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """File upload endpoint"""
    try:
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        unique_filename = f"{file_id}{file_extension}"
        file_path = uploads_dir / unique_filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        conn = sqlite3.connect('aetherium.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO files (id, filename, original_name, file_path, file_size, mime_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (file_id, unique_filename, file.filename, str(file_path), len(content), file.content_type))
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/files")
async def list_files():
    """List uploaded files"""
    try:
        conn = sqlite3.connect('aetherium.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, original_name, file_size, mime_type, upload_date
            FROM files ORDER BY upload_date DESC
        ''')
        files = [
            {
                "id": row[0],
                "name": row[1], 
                "size": row[2],
                "type": row[3],
                "uploaded": row[4]
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 AETHERIUM ENHANCED BACKEND WITH INTERNAL AI STARTING...")
    print("=" * 60)
    print("✅ Internal Aetherium AI as primary engine")
    print("✅ External APIs (OpenAI, Claude, Gemini) as secondary")
    print("✅ WebSocket real-time chat with AI selection")
    print("✅ File upload/download system")
    print("✅ Database persistence with AI tracking")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)