#!/usr/bin/env python3
"""
Deep Scan for Duplicates and Stubs
==================================

This script performs a comprehensive scan of the knowledge base to:
1. Identify duplicate folders and files
2. Find and categorize stub files
3. Generate merge recommendations
4. Create cleanup action plans

Author: Knowledge Base Automation System
Date: 2025-07-06
"""

import os
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

class DeepScanner:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.duplicates = defaultdict(list)
        self.stubs = []
        self.empty_dirs = []
        self.similar_dirs = defaultdict(list)
        self.file_hashes = {}
        
    def calculate_file_hash(self, filepath):
        """Calculate MD5 hash of file content"""
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            print(f"Error hashing {filepath}: {e}")
            return None
    
    def is_stub_file(self, filepath):
        """Identify if a file is a stub based on content patterns"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Common stub indicators
            stub_patterns = [
                r'auto-generated stub',
                r'This is a stub',
                r'TODO.*implement',
                r'Feature \d+.*Feature \d+.*Feature \d+',  # Generic feature lists
                r'Example code.*import module.*result = module\.function\(\)',
                r'This module provides functionality for\.\.\.',
                r'# Placeholder',
                r'# TODO',
                r'# FIXME',
            ]
            
            content_lower = content.lower()
            
            # Check for stub patterns
            for pattern in stub_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    return True
            
            # Check for very short files with generic content
            if len(content.strip()) < 200:
                generic_phrases = [
                    'overview', 'features', 'usage', 'example',
                    'module provides', 'functionality for'
                ]
                phrase_count = sum(1 for phrase in generic_phrases if phrase in content_lower)
                if phrase_count >= 3:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return False
    
    def find_similar_directory_names(self):
        """Find directories with similar names that might be duplicates"""
        dirs = []
        for root, dirnames, filenames in os.walk(self.root_path):
            for dirname in dirnames:
                dirs.append((dirname.lower(), os.path.join(root, dirname)))
        
        # Group similar names
        name_groups = defaultdict(list)
        for name, path in dirs:
            # Remove common prefixes/suffixes and group
            clean_name = re.sub(r'[_-]*(docs?|documentation|system|module|lib)s?[_-]*', '', name)
            clean_name = re.sub(r'[_-]+', '_', clean_name).strip('_')
            if clean_name:
                name_groups[clean_name].append(path)
        
        # Keep only groups with multiple directories
        for clean_name, paths in name_groups.items():
            if len(paths) > 1:
                self.similar_dirs[clean_name] = paths
    
    def scan_for_duplicates(self):
        """Scan for duplicate files and folders"""
        print("🔍 Scanning for duplicate files...")
        
        # Scan all files
        for root, dirs, files in os.walk(self.root_path):
            # Skip certain directories
            skip_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.devcontainer'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            # Check for empty directories
            if not dirs and not files:
                self.empty_dirs.append(root)
            
            for file in files:
                filepath = os.path.join(root, file)
                file_size = os.path.getsize(filepath)
                
                # Skip very large files or system files
                if file_size > 10 * 1024 * 1024 or file.startswith('.'):
                    continue
                
                # Calculate hash
                file_hash = self.calculate_file_hash(filepath)
                if file_hash:
                    key = f"{file_hash}_{file_size}"
                    self.duplicates[key].append(filepath)
                    self.file_hashes[filepath] = file_hash
                
                # Check if it's a stub file
                if file.endswith(('.md', '.py', '.js', '.html')):
                    if self.is_stub_file(filepath):
                        self.stubs.append(filepath)
        
        # Remove non-duplicates
        self.duplicates = {k: v for k, v in self.duplicates.items() if len(v) > 1}
        
        # Find similar directory names
        self.find_similar_directory_names()
    
    def generate_report(self):
        """Generate comprehensive scan report"""
        report = {
            'scan_date': datetime.now().isoformat(),
            'summary': {
                'duplicate_file_groups': len(self.duplicates),
                'total_duplicate_files': sum(len(files) for files in self.duplicates.values()),
                'stub_files': len(self.stubs),
                'empty_directories': len(self.empty_dirs),
                'similar_directory_groups': len(self.similar_dirs)
            },
            'duplicates': {},
            'stubs': self.stubs,
            'empty_directories': self.empty_dirs,
            'similar_directories': dict(self.similar_dirs)
        }
        
        # Format duplicates for report
        for i, (key, files) in enumerate(self.duplicates.items()):
            report['duplicates'][f'group_{i+1}'] = {
                'hash_size_key': key,
                'files': files,
                'recommendation': self.get_merge_recommendation(files)
            }
        
        return report
    
    def get_merge_recommendation(self, files):
        """Generate merge recommendation for duplicate files"""
        # Analyze file paths to suggest which to keep
        priorities = []
        for file in files:
            score = 0
            path_lower = file.lower()
            
            # Prefer files in main documentation areas
            if '/docs/' in path_lower or '/documentation/' in path_lower:
                score += 10
            
            # Prefer files in resources
            if '/resources/' in path_lower:
                score += 5
            
            # Prefer shorter paths (closer to root)
            score -= path_lower.count('/')
            
            # Prefer non-backup/temp names
            if 'backup' in path_lower or 'temp' in path_lower or 'old' in path_lower:
                score -= 20
            
            priorities.append((score, file))
        
        priorities.sort(reverse=True)
        keep_file = priorities[0][1]
        remove_files = [f for _, f in priorities[1:]]
        
        return {
            'action': 'merge',
            'keep': keep_file,
            'remove': remove_files,
            'reason': 'Keep file in preferred location, remove duplicates'
        }
    
    def save_report(self, output_file):
        """Save scan report to file"""
        report = self.generate_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also create a readable text report
        text_report_file = output_file.replace('.json', '.txt')
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("KNOWLEDGE BASE DEEP SCAN REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Scan Date: {report['scan_date']}\n\n")
            
            # Summary
            f.write("SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for key, value in report['summary'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Duplicates
            if report['duplicates']:
                f.write("DUPLICATE FILES:\n")
                f.write("-" * 20 + "\n")
                for group_name, group_data in report['duplicates'].items():
                    f.write(f"\n{group_name.upper()}:\n")
                    f.write(f"  Files:\n")
                    for file in group_data['files']:
                        f.write(f"    - {file}\n")
                    f.write(f"  Recommendation: {group_data['recommendation']['action']}\n")
                    f.write(f"  Keep: {group_data['recommendation']['keep']}\n")
                    f.write(f"  Remove: {', '.join(group_data['recommendation']['remove'])}\n")
                f.write("\n")
            
            # Stubs
            if report['stubs']:
                f.write("STUB FILES (Need Content):\n")
                f.write("-" * 30 + "\n")
                for stub in report['stubs']:
                    f.write(f"  - {stub}\n")
                f.write("\n")
            
            # Similar directories
            if report['similar_directories']:
                f.write("SIMILAR DIRECTORIES (Potential Duplicates):\n")
                f.write("-" * 45 + "\n")
                for name, paths in report['similar_directories'].items():
                    f.write(f"\n{name.upper()}:\n")
                    for path in paths:
                        f.write(f"  - {path}\n")
                f.write("\n")
            
            # Empty directories
            if report['empty_directories']:
                f.write("EMPTY DIRECTORIES:\n")
                f.write("-" * 20 + "\n")
                for empty_dir in report['empty_directories']:
                    f.write(f"  - {empty_dir}\n")
        
        print(f"📊 Reports saved:")
        print(f"  - JSON: {output_file}")
        print(f"  - Text: {text_report_file}")
        
        return report

def main():
    root_path = r"c:\Users\jpowe\CascadeProjects\github\knowledge-base"
    output_file = r"c:\Users\jpowe\CascadeProjects\github\knowledge-base\development\scripts\deep_scan_report.json"
    
    print("🚀 Starting Knowledge Base Deep Scan...")
    print("=" * 50)
    
    scanner = DeepScanner(root_path)
    scanner.scan_for_duplicates()
    
    report = scanner.save_report(output_file)
    
    print("\n📋 SCAN COMPLETE!")
    print("=" * 20)
    print(f"Duplicate file groups: {report['summary']['duplicate_file_groups']}")
    print(f"Total duplicate files: {report['summary']['total_duplicate_files']}")
    print(f"Stub files found: {report['summary']['stub_files']}")
    print(f"Empty directories: {report['summary']['empty_directories']}")
    print(f"Similar directory groups: {report['summary']['similar_directory_groups']}")
    print(f"\n📄 Full reports available in development/scripts/")

if __name__ == "__main__":
    main()
