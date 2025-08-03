# Sample backend test for knowledge ingestion
import os
import pytest

def test_knowledge_indexing():
    knowledge_root = '/workspaces/knowledge-base/knowledge'
    found = False
    for root, dirs, files in os.walk(knowledge_root):
        for file in files:
            if file.endswith('.md'):
                found = True
    assert found, 'No markdown files found in /knowledge (should have example entries)'
