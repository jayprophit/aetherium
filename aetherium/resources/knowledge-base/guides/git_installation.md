---
title: Git Installation
date: 2025-07-08
---

# Git Installation

---
category: resources
date: '2025-07-08'
tags: []
title: Git Installation
---

# Git Installation Guide

This guide is for contributors who want to set up the knowledge base for development.

## Prerequisites

- Git
- Docker (optional)

## Steps

1. Fork the repository on GitHub.

2. Clone your fork:

   ```bash
   git clone https://github.com/your-username/knowledge-base.git
   ```

3. Add the original repository as a remote:

   ```bash
   git remote add upstream https://github.com/jayprophit/knowledge-base.git
   ```

4. Create a new branch for your changes:

   ```bash
   git checkout -b your-branch-name
   ```

5. Make your changes and commit them:

   ```bash
   git commit -m "Your commit message"
   ```

6. Push your changes to your fork:

   ```bash
   git push origin your-branch-name
   ```

7. Create a pull request from your fork to the original repository.

## Deployment Guide

For deployment instructions, see the [Deployment Guide](../deployment/deployment_guide.md).

## Docker Setup

If you prefer to use Docker, you can build and run the application with:

```bash
docker build -t knowledge-base .
docker run -p 8000:8000 knowledge-base
