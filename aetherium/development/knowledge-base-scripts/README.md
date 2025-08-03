# Knowledge Ingestion Automation

This script and workflow will automatically index all new knowledge and documentation in `/knowledge` and `/resources/documentation/docs` for use in RAG and search. Run this script after adding or updating knowledge files:

```bash
bash scripts/reindex_knowledge.sh
```

This ensures the backend always has the latest knowledge indexed and ready for advanced retrieval.
