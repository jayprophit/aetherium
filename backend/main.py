"""
FastAPI backend for Knowledge Base AI Assistant
- Chat endpoint (RAG over docs)
- Search endpoint
- Plugin/tool execution endpoint (stub)
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import os
from plugins import run_tool

app = FastAPI()

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

# --- Endpoints ---
@app.post('/chat')
def chat(req: ChatRequest):
    # TODO: Replace with vector search and LLM call
    context = retrieve_relevant_docs(req.message)
    response = f"[AI] (context: {context[:200]}...)\nYou said: {req.message}"
    return {"response": response}

@app.post('/search')
def search(req: SearchRequest):
    # Simple keyword search
    results = []
    for path, content in doc_index.items():
        if req.query.lower() in content.lower():
            results.append({"path": path, "snippet": content[:200]})
            if len(results) >= req.top_k:
                break
    return {"results": results}

@app.post('/tool')
def tool(req: ToolRequest):
    # Call the plugin/tool system
    result = run_tool(req.tool, req.args)
    return {"result": result}

# --- Helper ---
def retrieve_relevant_docs(query):
    # TODO: Replace with vector search (e.g., FAISS, Chroma, Haystack)
    for path, content in doc_index.items():
        if query.lower() in content.lower():
            return content[:1000]
    return "No relevant docs found."
