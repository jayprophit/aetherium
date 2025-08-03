"""
Script to index all documentation in /resources/documentation/docs and /knowledge for RAG.
Builds a FAISS vector index using OpenAI embeddings for semantic search.
"""
import os
import json
import faiss
import openai
from pathlib import Path

EMBED_DIM = 1536  # OpenAI text-embedding-ada-002
openai.api_key = os.getenv('OPENAI_API_KEY', 'sk-...')  # Set your key or use env var

def get_embedding(text):
    # Call OpenAI API for embedding
    resp = openai.Embedding.create(
        input=text.replace("\n", " "),
        model="text-embedding-ada-002"
    )
    return resp['data'][0]['embedding']

def index_docs(docs_roots, out_path, faiss_path, mapping_path):
    index = {}
    texts = []
    paths = []
    for docs_root in docs_roots:
        for root, dirs, files in os.walk(docs_root):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        index[path] = content
                        texts.append(content[:2000])  # Truncate for embedding
                        paths.append(path)
    # Build embeddings
    vectors = [get_embedding(t) for t in texts]
    xb = np.array(vectors).astype('float32')
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    faiss_index.add(xb)
    faiss.write_index(faiss_index, faiss_path)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(paths, f)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(index, f)
    print(f"Indexed {len(index)} docs to {out_path} and built FAISS index.")

if __name__ == '__main__':
    import numpy as np
    docs_roots = [
        '/workspaces/knowledge-base/resources/documentation/docs',
        '/workspaces/knowledge-base/knowledge',
    ]
    out_path = '/workspaces/knowledge-base/backend/doc_index.json'
    faiss_path = '/workspaces/knowledge-base/backend/doc_faiss.index'
    mapping_path = '/workspaces/knowledge-base/backend/doc_faiss_mapping.json'
    index_docs(docs_roots, out_path, faiss_path, mapping_path)
