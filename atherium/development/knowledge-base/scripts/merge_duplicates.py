#!/usr/bin/env python3
"""
Merge Duplicate Files Script
===========================

This script automatically merges duplicate files based on the deep scan report recommendations.
It safely removes duplicate files while preserving the best copy of each.

Author: Knowledge Base Automation System
Date: 2025-07-06
"""

import os
import json
import shutil
from pathlib import Path
import argparse

class DuplicateMerger:
    def __init__(self, report_file, dry_run=True):
        self.report_file = report_file
        self.dry_run = dry_run
        self.merged_count = 0
        self.removed_count = 0
        self.errors = []
        
    def load_report(self):
        """Load the deep scan report"""
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading report: {e}")
            return None
    
    def merge_duplicates(self):
        """Merge duplicate files based on report recommendations"""
        report = self.load_report()
        if not report:
            return
        
        print(f"{'🔍 DRY RUN:' if self.dry_run else '🚀 EXECUTING:'} Merging duplicate files...")
        print("=" * 60)
        
        duplicates = report.get('duplicates', {})
        
        for group_name, group_data in duplicates.items():
            print(f"\n📁 Processing {group_name}:")
            
            recommendation = group_data.get('recommendation', {})
            keep_file = recommendation.get('keep')
            remove_files = recommendation.get('remove', [])
            
            if not keep_file or not remove_files:
                print(f"  ⚠️  Skipping {group_name} - incomplete recommendation")
                continue
            
            # Verify keep file exists
            if not os.path.exists(keep_file):
                print(f"  ❌ Keep file not found: {keep_file}")
                self.errors.append(f"Keep file not found: {keep_file}")
                continue
            
            print(f"  ✅ Keeping: {keep_file}")
            
            # Remove duplicate files
            for remove_file in remove_files:
                if os.path.exists(remove_file):
                    try:
                        if not self.dry_run:
                            os.remove(remove_file)
                            self.removed_count += 1
                        print(f"  🗑️  {'Would remove' if self.dry_run else 'Removed'}: {remove_file}")
                    except Exception as e:
                        error_msg = f"Error removing {remove_file}: {e}"
                        print(f"  ❌ {error_msg}")
                        self.errors.append(error_msg)
                else:
                    print(f"  ⚠️  File already removed: {remove_file}")
            
            if not self.dry_run:
                self.merged_count += 1
    
    def remove_empty_directories(self, report):
        """Remove empty directories"""
        empty_dirs = report.get('empty_directories', [])
        
        if empty_dirs:
            print(f"\n📂 {'Would remove' if self.dry_run else 'Removing'} {len(empty_dirs)} empty directories:")
            
            for empty_dir in empty_dirs:
                if os.path.exists(empty_dir) and not os.listdir(empty_dir):
                    try:
                        if not self.dry_run:
                            os.rmdir(empty_dir)
                        print(f"  🗑️  {'Would remove' if self.dry_run else 'Removed'}: {empty_dir}")
                    except Exception as e:
                        error_msg = f"Error removing directory {empty_dir}: {e}"
                        print(f"  ❌ {error_msg}")
                        self.errors.append(error_msg)
    
    def run(self):
        """Run the merge process"""
        report = self.load_report()
        if not report:
            return False
        
        print(f"📋 DUPLICATE MERGE REPORT")
        print("=" * 30)
        print(f"Report date: {report.get('scan_date', 'Unknown')}")
        print(f"Duplicate groups: {report['summary']['duplicate_file_groups']}")
        print(f"Total duplicate files: {report['summary']['total_duplicate_files']}")
        
        # Merge duplicates
        self.merge_duplicates()
        
        # Remove empty directories
        self.remove_empty_directories(report)
        
        # Summary
        print(f"\n{'🔍 DRY RUN SUMMARY:' if self.dry_run else '✅ MERGE COMPLETE:'}")
        print("=" * 30)
        if not self.dry_run:
            print(f"Merged duplicate groups: {self.merged_count}")
            print(f"Files removed: {self.removed_count}")
        else:
            print(f"Would merge {len(report.get('duplicates', {}))} duplicate groups")
            print(f"Would remove ~{report['summary']['total_duplicate_files'] - report['summary']['duplicate_file_groups']} files")
        
        if self.errors:
            print(f"\n⚠️  Errors encountered: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        return len(self.errors) == 0

def main():
    parser = argparse.ArgumentParser(description='Merge duplicate files based on deep scan report')
    parser.add_argument('--execute', action='store_true', help='Execute the merge (default is dry run)')
    parser.add_argument('--report', default='deep_scan_report.json', help='Path to deep scan report')
    
    args = parser.parse_args()
    
    report_path = os.path.join(os.path.dirname(__file__), args.report)
    
    merger = DuplicateMerger(report_path, dry_run=not args.execute)
    success = merger.run()
    
    if not args.execute:
        print(f"\n💡 This was a dry run. Use --execute to perform actual merges.")
        print(f"   Command: python {__file__} --execute")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
