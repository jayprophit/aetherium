#!/usr/bin/env python3
"""
AETHERIUM DEEP DIRECTORY SCAN AND REORGANIZER
Analyzes entire directory structure, identifies issues, and performs intelligent cleanup
"""

import os
import shutil
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import difflib
from datetime import datetime

class AetheriumDirectoryAnalyzer:
    """Comprehensive directory analysis and reorganization tool"""
    
    def __init__(self, root_path: str = "C:/Users/jpowe/CascadeProjects/github/aetherium"):
        self.root_path = Path(root_path)
        self.analysis_results = {}
        self.recommendations = []
        self.file_hashes = {}
        self.duplicate_groups = []
        self.obsolete_files = []
        self.logger = self.setup_logging()
        
        # Define structure standards
        self.ideal_structure = {
            'src/': 'Core source code',
            'src/ai/': 'AI engine and models',
            'src/agents/': 'Multi-agent system',
            'src/services/': 'Service implementations',
            'src/networking/': 'Networking components',
            'src/frontend/': 'Frontend components',
            'src/backend/': 'Backend API',
            'src/utils/': 'Utility functions',
            'docs/': 'Documentation',
            'resources/': 'Static resources and data',
            'resources/knowledge_base/': 'Knowledge base content',
            'tests/': 'Test files',
            'scripts/': 'Automation scripts',
            'config/': 'Configuration files',
            'deployment/': 'Deployment configurations',
            'archive/': 'Archived/deprecated files'
        }
        
        # File patterns to identify as obsolete
        self.obsolete_patterns = [
            'LAUNCH_*.py',
            'COMPLETE_*.py',
            'EXECUTE_*.py',
            'AUTO_*.py',
            'WORKING_*.py',
            'EXACT_*.py',
            'INSTANT_*.py',
            'DIRECT_*.py',
            'FINAL_*.py',
            'START_*.bat',
            'simple_*.py',
            'demo_*.py',
            'test_*.html'
        ]
        
    def setup_logging(self) -> logging.Logger:
        """Setup logging for analysis"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('directory_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not hash {file_path}: {e}")
            return ""
    
    def scan_directory_structure(self) -> Dict[str, Any]:
        """Comprehensive directory structure analysis"""
        self.logger.info("🔍 Starting deep directory scan...")
        
        analysis = {
            'total_files': 0,
            'total_directories': 0,
            'file_types': defaultdict(int),
            'large_files': [],
            'empty_directories': [],
            'directory_sizes': {},
            'potential_duplicates': [],
            'obsolete_files': [],
            'structure_issues': []
        }
        
        # Scan all files and directories
        for item in self.root_path.rglob('*'):
            if item.is_file():
                analysis['total_files'] += 1
                
                # File type analysis
                suffix = item.suffix.lower()
                analysis['file_types'][suffix] += 1
                
                # Size analysis
                try:
                    size = item.stat().st_size
                    if size > 10 * 1024 * 1024:  # Files larger than 10MB
                        analysis['large_files'].append((str(item), size))
                    
                    # Calculate hash for duplicate detection
                    if size > 0 and suffix in ['.py', '.js', '.ts', '.md', '.json']:
                        file_hash = self.calculate_file_hash(item)
                        if file_hash:
                            if file_hash in self.file_hashes:
                                self.file_hashes[file_hash].append(str(item))
                            else:
                                self.file_hashes[file_hash] = [str(item)]
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {item}: {e}")
                
                # Identify obsolete files
                if self.is_obsolete_file(item):
                    analysis['obsolete_files'].append(str(item))
            
            elif item.is_dir():
                analysis['total_directories'] += 1
                
                # Check for empty directories
                try:
                    if not any(item.iterdir()):
                        analysis['empty_directories'].append(str(item))
                except:
                    pass
        
        # Find duplicates
        for file_hash, files in self.file_hashes.items():
            if len(files) > 1:
                analysis['potential_duplicates'].append(files)
        
        self.analysis_results = analysis
        return analysis
    
    def is_obsolete_file(self, file_path: Path) -> bool:
        """Check if file matches obsolete patterns"""
        filename = file_path.name
        
        # Check against obsolete patterns
        for pattern in self.obsolete_patterns:
            if pattern.replace('*', '') in filename:
                return True
        
        # Additional checks for specific obsolete files
        obsolete_names = [
            'aetherium_sidebar_backend.py',
            'aetherium_working.py',
            'ai_complete_backend.py',
            'AI_COMPLETE_UI.py',
            'auto_deploy_complete.py',
            'launch_full_interactive.py',
            'REPOSITORY_CLEANUP_AND_ENHANCEMENT.py'
        ]
        
        return filename in obsolete_names
    
    def analyze_code_similarity(self) -> List[Dict]:
        """Analyze code files for similarity and potential merging"""
        similar_files = []
        python_files = list(self.root_path.rglob('*.py'))
        
        self.logger.info(f"Analyzing {len(python_files)} Python files for similarity...")
        
        for i, file1 in enumerate(python_files):
            if i % 50 == 0:
                self.logger.info(f"Processed {i}/{len(python_files)} files...")
            
            try:
                with open(file1, 'r', encoding='utf-8') as f1:
                    content1 = f1.readlines()
                
                for file2 in python_files[i+1:]:
                    try:
                        with open(file2, 'r', encoding='utf-8') as f2:
                            content2 = f2.readlines()
                        
                        # Calculate similarity
                        similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
                        
                        if similarity > 0.8:  # 80% similar
                            similar_files.append({
                                'file1': str(file1),
                                'file2': str(file2),
                                'similarity': similarity,
                                'action': 'merge_or_deduplicate'
                            })
                    except Exception as e:
                        continue
            except Exception as e:
                continue
        
        return similar_files
    
    def generate_reorganization_plan(self) -> Dict[str, Any]:
        """Generate comprehensive reorganization plan"""
        self.logger.info("📋 Generating reorganization plan...")
        
        plan = {
            'actions': [],
            'moves': [],
            'deletions': [],
            'merges': [],
            'directory_structure': {},
            'estimated_space_saved': 0,
            'cleanup_summary': {}
        }
        
        # 1. Archive obsolete files
        if self.analysis_results.get('obsolete_files'):
            archive_dir = self.root_path / 'archive' / 'obsolete_launchers'
            plan['actions'].append(f"Create archive directory: {archive_dir}")
            
            for obsolete_file in self.analysis_results['obsolete_files']:
                plan['moves'].append({
                    'from': obsolete_file,
                    'to': str(archive_dir / Path(obsolete_file).name),
                    'reason': 'obsolete_launcher_script'
                })
        
        # 2. Handle duplicates
        for duplicate_group in self.analysis_results.get('potential_duplicates', []):
            if len(duplicate_group) > 1:
                # Keep the one in the most appropriate location
                primary_file = self.select_primary_duplicate(duplicate_group)
                for dup_file in duplicate_group:
                    if dup_file != primary_file:
                        plan['deletions'].append({
                            'file': dup_file,
                            'reason': f'duplicate_of_{Path(primary_file).name}',
                            'primary': primary_file
                        })
        
        # 3. Remove empty directories
        for empty_dir in self.analysis_results.get('empty_directories', []):
            plan['deletions'].append({
                'file': empty_dir,
                'reason': 'empty_directory'
            })
        
        # 4. Standardize structure
        plan['directory_structure'] = self.ideal_structure
        
        # 5. Clean up root directory
        root_files = [f for f in self.root_path.iterdir() if f.is_file()]
        python_executables = [f for f in root_files if f.suffix == '.py' and 'LAUNCH' in f.name.upper()]
        
        if len(python_executables) > 3:  # Keep only essential launchers
            essential_launchers = [
                'COMPREHENSIVE_AETHERIUM_COMPLETE_LAUNCHER.py'
            ]
            
            for py_file in python_executables:
                if py_file.name not in essential_launchers:
                    plan['moves'].append({
                        'from': str(py_file),
                        'to': str(self.root_path / 'archive' / 'redundant_launchers' / py_file.name),
                        'reason': 'redundant_launcher'
                    })
        
        return plan
    
    def select_primary_duplicate(self, duplicate_files: List[str]) -> str:
        """Select the primary file to keep from duplicates"""
        # Prefer files in src/ over others
        for file_path in duplicate_files:
            if '/src/' in file_path:
                return file_path
        
        # Prefer shorter paths (closer to root)
        return min(duplicate_files, key=len)
    
    def execute_reorganization(self, plan: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """Execute the reorganization plan"""
        self.logger.info(f"🚀 {'Simulating' if dry_run else 'Executing'} reorganization plan...")
        
        results = {
            'moves_completed': 0,
            'deletions_completed': 0,
            'directories_created': 0,
            'errors': [],
            'space_freed': 0
        }
        
        if not dry_run:
            # Create necessary directories
            for directory in plan.get('directory_structure', {}):
                dir_path = self.root_path / directory
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    results['directories_created'] += 1
            
            # Execute moves
            for move_action in plan.get('moves', []):
                try:
                    src = Path(move_action['from'])
                    dst = Path(move_action['to'])
                    
                    # Create destination directory if needed
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    if src.exists():
                        shutil.move(str(src), str(dst))
                        results['moves_completed'] += 1
                        self.logger.info(f"Moved: {src} → {dst}")
                
                except Exception as e:
                    results['errors'].append(f"Move failed {move_action['from']}: {e}")
            
            # Execute deletions
            for delete_action in plan.get('deletions', []):
                try:
                    file_path = Path(delete_action['file'])
                    if file_path.exists():
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            file_path.unlink()
                            results['space_freed'] += size
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                        
                        results['deletions_completed'] += 1
                        self.logger.info(f"Deleted: {file_path}")
                
                except Exception as e:
                    results['errors'].append(f"Deletion failed {delete_action['file']}: {e}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("🔍 AETHERIUM DIRECTORY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary statistics
        if self.analysis_results:
            report.append("📊 DIRECTORY STATISTICS")
            report.append("-" * 40)
            report.append(f"Total Files: {self.analysis_results['total_files']:,}")
            report.append(f"Total Directories: {self.analysis_results['total_directories']:,}")
            report.append(f"File Types: {len(self.analysis_results['file_types'])}")
            report.append(f"Large Files (>10MB): {len(self.analysis_results['large_files'])}")
            report.append(f"Empty Directories: {len(self.analysis_results['empty_directories'])}")
            report.append(f"Potential Duplicates: {len(self.analysis_results['potential_duplicates'])}")
            report.append(f"Obsolete Files: {len(self.analysis_results['obsolete_files'])}")
            report.append("")
            
            # File type breakdown
            report.append("📁 FILE TYPE DISTRIBUTION")
            report.append("-" * 40)
            for ext, count in sorted(self.analysis_results['file_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"{ext or 'no extension':15} {count:4} files")
            report.append("")
            
            # Issues found
            if self.analysis_results['obsolete_files']:
                report.append("🗑️ OBSOLETE FILES TO ARCHIVE/REMOVE")
                report.append("-" * 40)
                for obsolete in self.analysis_results['obsolete_files'][:10]:
                    report.append(f"• {Path(obsolete).name}")
                if len(self.analysis_results['obsolete_files']) > 10:
                    report.append(f"... and {len(self.analysis_results['obsolete_files']) - 10} more")
                report.append("")
            
            if self.analysis_results['potential_duplicates']:
                report.append("🔄 DUPLICATE FILES DETECTED")
                report.append("-" * 40)
                for i, dup_group in enumerate(self.analysis_results['potential_duplicates'][:5]):
                    report.append(f"Group {i+1}:")
                    for dup_file in dup_group:
                        report.append(f"  • {Path(dup_file).name}")
                    report.append("")
        
        # Recommendations
        report.append("💡 REORGANIZATION RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Archive obsolete launcher scripts to archive/obsolete_launchers/")
        report.append("2. Remove duplicate files and keep primary versions")
        report.append("3. Standardize directory structure according to best practices")
        report.append("4. Clean up root directory by moving scripts to scripts/")
        report.append("5. Remove empty directories")
        report.append("6. Consolidate similar functionality into single files")
        report.append("")
        
        return "\n".join(report)

# Main execution function
def main():
    """Main analysis and reorganization function"""
    
    print("🚀 Starting Aetherium Deep Directory Analysis...")
    
    analyzer = AetheriumDirectoryAnalyzer()
    
    # 1. Perform comprehensive scan
    print("📋 Step 1: Scanning directory structure...")
    analysis = analyzer.scan_directory_structure()
    
    # 2. Analyze code similarity
    print("🔍 Step 2: Analyzing code similarity...")
    similar_files = analyzer.analyze_code_similarity()
    
    # 3. Generate reorganization plan
    print("📋 Step 3: Generating reorganization plan...")
    plan = analyzer.generate_reorganization_plan()
    
    # 4. Generate report
    print("📄 Step 4: Generating analysis report...")
    report = analyzer.generate_report()
    
    # Save report
    with open('AETHERIUM_DIRECTORY_ANALYSIS.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save detailed results
    with open('directory_analysis_detailed.json', 'w') as f:
        json.dump({
            'analysis': analysis,
            'similar_files': similar_files,
            'reorganization_plan': plan
        }, f, indent=2, default=str)
    
    # 5. Execute reorganization (dry run first)
    print("🧪 Step 5: Simulating reorganization...")
    dry_run_results = analyzer.execute_reorganization(plan, dry_run=True)
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 Found {analysis['total_files']} files in {analysis['total_directories']} directories")
    print(f"🗑️ Identified {len(analysis['obsolete_files'])} obsolete files")
    print(f"🔄 Found {len(analysis['potential_duplicates'])} duplicate groups")
    print(f"📋 Generated plan with {len(plan['moves'])} moves and {len(plan['deletions'])} deletions")
    print(f"📄 Report saved to: AETHERIUM_DIRECTORY_ANALYSIS.txt")
    print("="*60)
    
    # Ask for confirmation to execute
    response = input("\n🚀 Execute reorganization plan? (y/N): ")
    if response.lower() == 'y':
        print("🔧 Executing reorganization plan...")
        results = analyzer.execute_reorganization(plan, dry_run=False)
        print(f"✅ Completed: {results['moves_completed']} moves, {results['deletions_completed']} deletions")
        print(f"💾 Space freed: {results['space_freed'] / (1024*1024):.2f} MB")
        if results['errors']:
            print(f"⚠️ Errors encountered: {len(results['errors'])}")

if __name__ == "__main__":
    main()
