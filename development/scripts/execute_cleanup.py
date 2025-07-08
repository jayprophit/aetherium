#!/usr/bin/env python3
"""
Knowledge Base Cleanup Script
============================

This script implements fixes for issues identified in the deep scan.
"""

import os
import shutil
from pathlib import Path
import yaml
import re

class KnowledgeBaseCleanup:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.fixes_applied = []
        
    def cleanup_all(self):
        """Execute all cleanup operations"""
        self.fix_directory_structure()
        self.add_frontmatter()
        self.consolidate_todos()
        self.generate_report()
        
    def fix_directory_structure(self):
        """Fix directory structure issues"""
        # Fix duplicate nested directories
        duplicates = [
            ('operations/infrastructure/infrastructure', 'operations/infrastructure'),
            ('operations/security/security', 'operations/security')
        ]
        
        for old, new in duplicates:
            old_path = self.root_path / old
            new_path = self.root_path / new
            if old_path.exists():
                # Move contents up one level
                for item in old_path.iterdir():
                    shutil.move(str(item), str(new_path))
                os.rmdir(old_path)
                self.fixes_applied.append(f"Consolidated directory {old} into {new}")
                
    def add_frontmatter(self):
        """Add frontmatter to markdown files that need it"""
        for path in self.root_path.rglob('*.md'):
            if path.is_file():
                content = path.read_text(encoding='utf-8')
                if not content.startswith('---\n'):
                    # Get relative path for the title
                    rel_path = path.relative_to(self.root_path)
                    title = rel_path.stem.replace('_', ' ').title()
                    
                    # Create frontmatter
                    frontmatter = {
                        'title': title,
                        'date': '2025-07-08',
                        'category': str(rel_path.parent).split('/')[0],
                        'tags': []
                    }
                    
                    # Add frontmatter to content
                    new_content = f"---\n{yaml.dump(frontmatter)}---\n\n{content}"
                    path.write_text(new_content, encoding='utf-8')
                    self.fixes_applied.append(f"Added frontmatter to {rel_path}")
                    
    def consolidate_todos(self):
        """Consolidate TODO and FIXME files into task lists"""
        todo_content = "# Consolidated Task List\n\n"
        fixme_content = "# Consolidated Fix List\n\n"
        
        # Process TODO.md
        todo_path = self.root_path / 'mcp-instructions' / 'TODO.md'
        if todo_path.exists():
            todo_content += todo_path.read_text(encoding='utf-8')
            todo_path.unlink()
            self.fixes_applied.append("Consolidated TODO.md")
            
        # Process FIXME.md
        fixme_path = self.root_path / 'mcp-instructions' / 'FIXME.md'
        if fixme_path.exists():
            fixme_content += fixme_path.read_text(encoding='utf-8')
            fixme_path.unlink()
            self.fixes_applied.append("Consolidated FIXME.md")
            
        # Create task_list.md with consolidated content
        task_list_path = self.root_path / 'mcp-instructions' / 'task_list.md'
        task_list_content = f"{todo_content}\n\n{fixme_content}"
        task_list_path.write_text(task_list_content, encoding='utf-8')
        
    def generate_report(self):
        """Generate cleanup report"""
        report = f"""# Knowledge Base Cleanup Report

## Date
2025-07-08

## Changes Applied
"""
        for fix in self.fixes_applied:
            report += f"- {fix}\n"
            
        report_path = self.root_path / 'reports' / 'cleanup_report.md'
        report_path.write_text(report, encoding='utf-8')

def main():
    cleanup = KnowledgeBaseCleanup('/workspaces/knowledge-base')
    cleanup.cleanup_all()
    print("Cleanup completed. Check reports/cleanup_report.md for results.")

if __name__ == '__main__':
    main()
