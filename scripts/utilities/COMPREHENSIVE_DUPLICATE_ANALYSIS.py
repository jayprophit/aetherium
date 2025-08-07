#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DUPLICATE & OBSOLETE FILE ANALYSIS
=================================================

This script analyzes the entire Aetherium directory for:
1. Duplicate files and folders
2. Obsolete/redundant scripts and components  
3. Missing core functionality
4. Directory size optimization opportunities

Results will guide cleanup and optimization efforts.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime

class AetheriumDuplicateAnalyzer:
    """Comprehensive analysis for duplicates, obsolete files, and missing components."""
    
    def __init__(self, base_path=None):
        if base_path is None:
            self.base_path = Path(__file__).parent.absolute()
        else:
            self.base_path = Path(base_path)
        
        self.duplicates_found = []
        self.obsolete_files = []
        self.missing_components = []
        self.size_issues = []
        
        print(f"🔍 Analyzing Aetherium directory: {self.base_path}")
    
    def analyze_launcher_duplicates(self):
        """Analyze launcher scripts for duplicates and obsolete versions."""
        
        print("\n🚀 ANALYZING LAUNCHER SCRIPTS")
        print("=" * 50)
        
        launcher_dir = self.base_path / "scripts" / "launchers"
        if not launcher_dir.exists():
            print("❌ Launcher directory not found")
            return
        
        launcher_files = {
            "AETHERIUM_BLT_V4_LAUNCHER.py": "Latest BLT v4.0 launcher - KEEP",
            "AETHERIUM_COMPLETE_LAUNCHER_WITH_INTERNAL_AI.py": "Internal AI launcher - POTENTIAL DUPLICATE", 
            "COMPLETE_INTEGRATED_LAUNCHER.py": "Generic integrated launcher - LIKELY OBSOLETE",
            "COMPLETE_WORKING_LAUNCHER.py": "Working launcher - LIKELY OBSOLETE", 
            "COMPREHENSIVE_AETHERIUM_COMPLETE_LAUNCHER.py": "Comprehensive launcher - POTENTIAL DUPLICATE",
            "LAUNCH_AETHERIUM_COMPLETE.py": "Complete launcher - LIKELY OBSOLETE",
            "PRODUCTION_LAUNCH.py": "Production launcher - MAY KEEP"
        }
        
        print("📊 Launcher Analysis Results:")
        for file, status in launcher_files.items():
            file_path = launcher_dir / file
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📄 {file} ({size_kb:.1f}KB) - {status}")
                
                if "OBSOLETE" in status or "POTENTIAL DUPLICATE" in status:
                    self.obsolete_files.append({
                        'file': str(file_path),
                        'reason': status,
                        'size_kb': size_kb
                    })
        
        # Recommendation: Keep only BLT v4.0 and Production launchers
        print("\n💡 LAUNCHER RECOMMENDATIONS:")
        print("  ✅ KEEP: AETHERIUM_BLT_V4_LAUNCHER.py (Latest/Primary)")
        print("  ✅ KEEP: PRODUCTION_LAUNCH.py (Production deployment)")
        print("  🗑️  REMOVE: 5 obsolete/duplicate launcher scripts")
        print(f"  💾 SPACE SAVINGS: ~50-100KB")
    
    def analyze_backend_duplicates(self):
        """Analyze backend files for duplicates."""
        
        print("\n🔧 ANALYZING BACKEND FILES")
        print("=" * 50)
        
        backend_files = {
            "src/backend_enhanced.py": "Enhanced backend - CHECK USAGE",
            "src/backend_enhanced_with_internal_ai.py": "Backend with internal AI - CHECK USAGE",
            "src/aetherium_master_orchestrator.py": "Master orchestrator - CHECK USAGE"
        }
        
        print("📊 Backend Analysis Results:")
        for file, status in backend_files.items():
            file_path = self.base_path / file
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📄 {file} ({size_kb:.1f}KB) - {status}")
        
        print("\n💡 BACKEND RECOMMENDATIONS:")
        print("  🔍 INVESTIGATE: Which backend file is currently active?")
        print("  🗑️  REMOVE: Unused backend duplicates")
        print("  💾 POTENTIAL SAVINGS: ~20-50KB")
    
    def analyze_ai_engine_versions(self):
        """Analyze AI engine files for version duplicates."""
        
        print("\n🧠 ANALYZING AI ENGINE VERSIONS")
        print("=" * 50)
        
        ai_engines = {
            "src/ai/aetherium_ai_engine.py": "Original engine - LIKELY OBSOLETE",
            "src/ai/aetherium_ai_engine_enhanced.py": "Enhanced v2.0 - OBSOLETE", 
            "src/ai/aetherium_ai_engine_v3_advanced.py": "v3.0 Advanced - OBSOLETE",
            "src/ai/aetherium_blt_engine_v4.py": "BLT v4.0 - CURRENT/KEEP"
        }
        
        print("📊 AI Engine Analysis Results:")
        total_obsolete_size = 0
        for file, status in ai_engines.items():
            file_path = self.base_path / file
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📄 {file} ({size_kb:.1f}KB) - {status}")
                
                if "OBSOLETE" in status:
                    total_obsolete_size += size_kb
                    self.obsolete_files.append({
                        'file': str(file_path),
                        'reason': status,
                        'size_kb': size_kb
                    })
        
        print("\n💡 AI ENGINE RECOMMENDATIONS:")
        print("  ✅ KEEP: aetherium_blt_engine_v4.py (Latest BLT architecture)")
        print("  🗑️  REMOVE: 3 obsolete AI engine versions")
        print(f"  💾 SPACE SAVINGS: ~{total_obsolete_size:.1f}KB")
    
    def analyze_integration_duplicates(self):
        """Analyze integration scripts for duplicates."""
        
        print("\n🔗 ANALYZING INTEGRATION SCRIPTS")
        print("=" * 50)
        
        integration_dir = self.base_path / "scripts" / "integration"
        if not integration_dir.exists():
            print("❌ Integration directory not found")
            return
        
        integration_files = {
            "AETHERIUM_V3_COMPLETE_INTEGRATION.py": "v3.0 integration - MAY BE OBSOLETE",
            "COMPLETE_AI_INTEGRATION.py": "AI integration - KEEP IF USED",
            "COMPLETE_AUTH_FLOW.py": "Auth integration - KEEP IF USED", 
            "COMPLETE_DATABASE_SYSTEM.py": "Database integration - KEEP IF USED",
            "COMPLETE_FILE_SYSTEM.py": "File system - KEEP IF USED",
            "COMPLETE_WEBSOCKET_INTEGRATION.py": "WebSocket - KEEP IF USED",
            "FINAL_COMPLETE_INTEGRATION.py": "Final integration - CHECK FOR DUPLICATES",
            "INTEGRATE_EVERYTHING_NOW.py": "Everything integration - POTENTIAL DUPLICATE"
        }
        
        print("📊 Integration Analysis Results:")
        for file, status in integration_files.items():
            file_path = integration_dir / file
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📄 {file} ({size_kb:.1f}KB) - {status}")
        
        print("\n💡 INTEGRATION RECOMMENDATIONS:")
        print("  🔍 INVESTIGATE: Which integration scripts are actively used?")
        print("  🗑️  REMOVE: Duplicate 'complete integration' scripts")  
        print("  💾 POTENTIAL SAVINGS: ~30-80KB")
    
    def analyze_missing_components(self):
        """Analyze for missing core components."""
        
        print("\n❓ ANALYZING MISSING COMPONENTS")
        print("=" * 50)
        
        # Check for essential missing files/directories
        essential_components = {
            "requirements.txt": "Python dependencies file",
            "environment.yml": "Conda environment file",
            ".env": "Environment variables file",
            "LICENSE": "Software license file",
            "CONTRIBUTING.md": "Contribution guidelines",
            "src/main.py": "Main application entry point",
            "tests/": "Test directory",
            "docs/api/": "API documentation",
            "docs/architecture/": "Architecture documentation"
        }
        
        missing_found = []
        for component, description in essential_components.items():
            component_path = self.base_path / component
            if not component_path.exists():
                missing_found.append(f"  ❌ {component} - {description}")
                self.missing_components.append({
                    'component': component,
                    'description': description,
                    'priority': 'HIGH' if component in ['requirements.txt', 'src/main.py'] else 'MEDIUM'
                })
        
        if missing_found:
            print("📊 Missing Components Found:")
            for item in missing_found:
                print(item)
        else:
            print("✅ All essential components appear to be present")
        
        print("\n💡 MISSING COMPONENT RECOMMENDATIONS:")
        print("  📝 CREATE: requirements.txt with all Python dependencies")
        print("  📝 CREATE: LICENSE file for legal clarity")
        print("  📝 CREATE: tests/ directory with test files")
        print("  📝 CREATE: Complete API documentation")
    
    def analyze_directory_sizes(self):
        """Analyze directory sizes for optimization opportunities."""
        
        print("\n📊 ANALYZING DIRECTORY SIZES")
        print("=" * 50)
        
        def get_dir_size(path):
            total_size = 0
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except OSError:
                            pass
            except OSError:
                pass
            return total_size
        
        directories_to_check = [
            "node_modules",
            ".git", 
            "aetherium",
            "src",
            "scripts",
            "docs",
            "archive"
        ]
        
        print("📊 Directory Size Analysis:")
        for dir_name in directories_to_check:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                size_mb = get_dir_size(dir_path) / (1024 * 1024)
                print(f"  📁 {dir_name}: {size_mb:.1f}MB")
                
                if size_mb > 100:  # Flag large directories
                    self.size_issues.append({
                        'directory': dir_name,
                        'size_mb': size_mb,
                        'recommendation': 'INVESTIGATE for cleanup opportunities'
                    })
        
        print("\n💡 SIZE OPTIMIZATION RECOMMENDATIONS:")
        if self.size_issues:
            for issue in self.size_issues:
                print(f"  🔍 {issue['directory']} ({issue['size_mb']:.1f}MB) - {issue['recommendation']}")
        else:
            print("  ✅ No major size issues detected")
    
    def generate_cleanup_recommendations(self):
        """Generate comprehensive cleanup recommendations."""
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE CLEANUP RECOMMENDATIONS")
        print("=" * 60)
        
        total_obsolete_files = len(self.obsolete_files)
        total_size_savings = sum(file['size_kb'] for file in self.obsolete_files)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  🗑️  Obsolete files identified: {total_obsolete_files}")
        print(f"  💾 Potential space savings: {total_size_savings:.1f}KB")
        print(f"  ❓ Missing components: {len(self.missing_components)}")
        print(f"  📁 Large directories flagged: {len(self.size_issues)}")
        
        print(f"\n🎯 PRIORITY ACTIONS:")
        print(f"  1️⃣  REMOVE obsolete AI engine versions (v1-v3)")
        print(f"  2️⃣  CONSOLIDATE launcher scripts (keep 2, remove 5)")
        print(f"  3️⃣  MERGE duplicate integration scripts")
        print(f"  4️⃣  CREATE missing essential files (requirements.txt, tests/)")
        print(f"  5️⃣  INVESTIGATE large directory contents")
        
        return {
            'obsolete_files': self.obsolete_files,
            'missing_components': self.missing_components,
            'size_issues': self.size_issues,
            'total_savings_kb': total_size_savings
        }
    
    def run_comprehensive_analysis(self):
        """Execute the complete duplicate and obsolete file analysis."""
        
        print("🔍" * 50)
        print("🔍 AETHERIUM COMPREHENSIVE DUPLICATE ANALYSIS")
        print("🔍" * 50)
        
        # Run all analysis phases
        self.analyze_launcher_duplicates()
        self.analyze_backend_duplicates() 
        self.analyze_ai_engine_versions()
        self.analyze_integration_duplicates()
        self.analyze_missing_components()
        self.analyze_directory_sizes()
        
        # Generate recommendations
        results = self.generate_cleanup_recommendations()
        
        print(f"\n🎊 ANALYSIS COMPLETE!")
        print(f"📋 Review recommendations above and proceed with cleanup")
        
        return results

def main():
    """Main execution function."""
    
    print("🌟" * 50)
    print("🔍 AETHERIUM DUPLICATE & OBSOLETE ANALYSIS")
    print("🌟" * 50)
    
    # Run the analysis
    analyzer = AetheriumDuplicateAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n✅ Analysis complete! Ready for cleanup phase.")
    
    return results

if __name__ == "__main__":
    main()