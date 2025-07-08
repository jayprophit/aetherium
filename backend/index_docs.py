"""
Script to index all documentation in /resources/documentation/docs for RAG.
This can be extended to use FAISS, Chroma, Haystack, etc.
"""
import os
import json
from pathlib import Path

def index_docs(docs_root, out_path):
    index = {}
    for root, dirs, files in os.walk(docs_root):
        for file in files:
            if file.endswith('.md'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    index[path] = f.read()
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(index, f)
    print(f"Indexed {len(index)} docs to {out_path}")

if __name__ == '__main__':
    docs_root = '/workspaces/knowledge-base/resources/documentation/docs'
    out_path = '/workspaces/knowledge-base/backend/doc_index.json'
    index_docs(docs_root, out_path)
