# Knowledge Base AI Assistant Backend

## Features
- FastAPI endpoints for chat, search, and tool/plugin execution
- Loads and indexes all documentation from `/resources/documentation/docs`
- Ready for RAG (Retrieval-Augmented Generation) with LLMs
- Easily extensible for plugins/tools

## Usage

### 1. Install dependencies
```bash
pip install fastapi uvicorn pydantic
```

### 2. Index documentation
```bash
python index_docs.py
```

### 3. Run the backend
```bash
uvicorn main:app --reload
```

## Next Steps
- Integrate vector search (FAISS, Chroma, Haystack, etc.)
- Connect to OpenAI, Qwen, or other LLM APIs
- Add authentication, user management, and plugin execution
