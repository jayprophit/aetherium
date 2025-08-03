"""
FastAPI backend for Knowledge Base AI Assistant
- Chat endpoint (RAG over docs)
- Search endpoint
- Plugin/tool execution endpoint (stub)
"""
from fastapi import FastAPI, Request, APIRouter
from pydantic import BaseModel
from typing import List, Optional
import os
from plugins import run_tool, list_plugins, run_plugin_chain
import faiss
import openai
import numpy as np
import json
import socketio
from fastapi.middleware.cors import CORSMiddleware

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app_sio = socketio.ASGIApp(sio, app)
api_router = APIRouter()

# --- Models ---
class ChatRequest(BaseModel):
    user: str
    message: str
    history: List[str] = []

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ToolRequest(BaseModel):
    tool: str
    args: dict

# --- In-memory doc index (to be replaced by vector DB) ---
doc_index = {}

# --- Utility: Load all docs into memory (for demo) ---
def load_docs():
    docs_roots = [
        '/workspaces/knowledge-base/resources/documentation/docs',
        '/workspaces/knowledge-base/knowledge',
    ]
    for docs_root in docs_roots:
        for root, dirs, files in os.walk(docs_root):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8') as f:
                        doc_index[path] = f.read()
load_docs()

EMBED_DIM = 1536
openai.api_key = os.getenv('OPENAI_API_KEY', 'sk-...')  # Set your key or use env var

# --- Load FAISS index and mapping ---
FAISS_PATH = '/workspaces/knowledge-base/backend/doc_faiss.index'
MAPPING_PATH = '/workspaces/knowledge-base/backend/doc_faiss_mapping.json'
faiss_index = None
faiss_mapping = []
if os.path.exists(FAISS_PATH) and os.path.exists(MAPPING_PATH):
    faiss_index = faiss.read_index(FAISS_PATH)
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        faiss_mapping = json.load(f)

def get_embedding(text):
    resp = openai.Embedding.create(
        input=text.replace("\n", " "),
        model="text-embedding-ada-002"
    )
    return np.array(resp['data'][0]['embedding'], dtype=np.float32)

def semantic_search(query, top_k=3):
    if not faiss_index or not faiss_mapping:
        return []
    q_emb = get_embedding(query).reshape(1, -1)
    D, I = faiss_index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(faiss_mapping):
            path = faiss_mapping[idx]
            snippet = doc_index.get(path, '')[:200]
            results.append({'path': path, 'snippet': snippet})
    return results

# --- Endpoints ---
@api_router.post('/chat')
def chat(req: ChatRequest):
    # Use semantic search and LLM for response
    context_docs = semantic_search(req.message, top_k=3)
    context = '\n'.join([d['snippet'] for d in context_docs])
    prompt = f"Context:\n{context}\n\nUser: {req.message}\nAI:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=256
    )
    return {"response": response['choices'][0]['text'].strip()}

@api_router.post('/search')
def search(req: SearchRequest):
    # Simple keyword search
    results = []
    for path, content in doc_index.items():
        if req.query.lower() in content.lower():
            results.append({"path": path, "snippet": content[:200]})
            if len(results) >= req.top_k:
                break
    return {"results": results}

@api_router.post('/tool')
def tool(req: ToolRequest):
    # Call the plugin/tool system
    result = run_tool(req.tool, req.args)
    return {"result": result}

@api_router.post('/semantic_search')
def semantic_search_endpoint(req: SearchRequest):
    results = semantic_search(req.query, req.top_k)
    return {"results": results}

@api_router.get('/plugin_marketplace')
def plugin_marketplace():
    # List all available plugins with quantum properties and descriptions
    return {"plugins": list_plugins()}

@api_router.post('/run_plugin')
def run_plugin(req: ToolRequest):
    result = run_tool(req.tool, req.args)
    return {"result": result}

@api_router.post('/run_plugin_chain')
async def run_plugin_chain_endpoint(request: Request):
    data = await request.json()
    chain = data['chain']
    results = run_plugin_chain(chain)
    return {"results": results}

@api_router.post('/quantum_ai')
def quantum_ai_endpoint(req: ChatRequest):
    # Stub: Quantum AI with time crystals (theoretical)
    return {"response": "[Quantum AI] This feature is under active research. Time crystals and quantum effects will be simulated using available scientific models and your knowledge-base."}

app.include_router(api_router, prefix="/api")

# --- Helper ---
def retrieve_relevant_docs(query):
    # TODO: Replace with vector search (e.g., FAISS, Chroma, Haystack)
    for path, content in doc_index.items():
        if query.lower() in content.lower():
            return content[:1000]
    return "No relevant docs found."

# --- Real-time graph state (in-memory, per-room for demo) ---
graph_state = {'nodes': [], 'edges': []}

@sio.event
def connect(sid, environ):
    print('Client connected:', sid)
    sio.emit('graph', graph_state, to=sid)

@sio.event
def update_graph(sid, data):
    graph_state['nodes'] = data.get('nodes', [])
    graph_state['edges'] = data.get('edges', [])
    sio.emit('graph', graph_state, skip_sid=sid)

@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app_sio", host="0.0.0.0", port=4000, reload=True)
