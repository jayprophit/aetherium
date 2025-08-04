# Deployment Guide

## Local Deployment
1. Ensure Python 3.9+, Node.js, and Docker are installed.
2. Re-index knowledge:
   ```bash
   bash scripts/reindex_knowledge.sh
   ```
3. Start backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
4. Start frontend:
   ```bash
   cd frontend
   npm install && npm start
   ```

## Remote Deployment (GitHub)
- Push to GitHub triggers CI/CD (see `.github/workflows/` for details)
- Docker Compose and cloud deployment supported

---
