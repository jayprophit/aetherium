---
title: Local Installation
date: 2025-07-08
---

# Local Installation

---
category: resources
date: '2025-07-08'
tags: []
title: Local Installation
---

# Local Installation Guide

This guide will walk you through installing the knowledge base locally.

## Prerequisites
- Python 3.8 or higher
- Git

## Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/jayprophit/knowledge-base.git
   ```
2. Navigate to the project directory:
   ```bash
   cd knowledge-base
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source .venv/bin/activate
     ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the application:
   ```bash
   python main.py
   ```

## Troubleshooting
If you encounter any issues, please check the [FAQ](#faq) or open an issue on GitHub.
