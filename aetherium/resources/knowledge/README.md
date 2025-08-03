# Knowledge Base: Automated Knowledge Ingestion

- All knowledge and documentation in `/knowledge` and `/resources/documentation/docs` is now auto-indexed for RAG and search.
- Example entries for anime, films, books, comics, and TV series are provided.
- To add new knowledge, simply create a Markdown file in the appropriate subdirectory and run:

```bash
bash scripts/reindex_knowledge.sh
```

The backend will automatically load all indexed knowledge for advanced retrieval and chat.

---
