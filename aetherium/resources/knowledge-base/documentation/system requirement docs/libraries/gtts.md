---
title: Gtts
date: 2025-07-08
---

# Gtts

---
author: Knowledge Base Automation System
created_at: '2025-07-04'
description: Documentation on Gtts for libraries/gtts.md
title: Gtts
updated_at: '2025-07-04'
version: 1.0.0
---

# gTTS (Google Text-to-Speech) Library

## Overview
[gTTS](https://pypi.org/project/gTTS/) is a Python library and CLI tool to interface with Google Text-to-Speech API. It converts text to spoken mp3 audio using Google’s TTS service.

## Installation
```sh
pip install gTTS
```

## Example Usage
```python
from gtts import gTTS
tts = gTTS(text='Hello, world!', lang='en')
tts.save('hello.mp3')
```

## Integration Notes
- Used for cloud-based text-to-speech in the assistant.
- Requires internet connection.

## Cross-links
- [virtual_assistant_book.md](../virtual_assistant_book.md)
- [ai_agents.md](../ai_agents.md)

---
_Last updated: July 3, 2025_
