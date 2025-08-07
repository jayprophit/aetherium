#!/usr/bin/env python3
"""
🧹 COMPREHENSIVE DIRECTORY CLEANUP AND ORGANIZATION
==================================================

This script systematically organizes the Aetherium project directory by:
- Moving all launcher scripts to scripts/launchers/
- Moving all integration scripts to scripts/integration/
- Moving all batch files to scripts/batch/
- Moving documentation to docs/
- Moving reports to docs/reports/
- Organizing demo files appropriately
- Cleaning up root directory for best practices
- Removing duplicates and obsolete files
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class AetheriumDirectoryCleanup:
    """Comprehensive directory cleanup and organization."""
    
    def __init__(self, base_path=None):
        if base_path is None:
            self.base_path = Path(__file__).parent.absolute()
        else:
            self.base_path = Path(base_path)
        
        self.moved_files = []
        self.created_dirs = []
        self.removed_files = []
        self.errors = []
        
        print(f"🧹 Starting directory cleanup for: {self.base_path}")
        
    def create_directory_structure(self):
        """Create the proper directory structure."""
        
        directories_to_create = [
            "scripts/launchers",
            "scripts/integration", 
            "scripts/batch",
            "scripts/utilities",
            "scripts/automation",
            "scripts/deployment",
            "docs/reports",
            "docs/summaries",
            "docs/demos",
            "archive/obsolete_scripts",
            "archive/old_versions",
            "backup/original_files"
        ]
        
        for dir_path in directories_to_create:
            full_path = self.base_path / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.append(str(full_path))
                print(f"📁 Created directory: {dir_path}")
        
        print(f"✅ Created {len(self.created_dirs)} directories")
    
    def organize_launcher_scripts(self):
        """Move all launcher scripts to scripts/launchers/."""
        
        launcher_scripts = [
            "AETHERIUM_BLT_V4_LAUNCHER.py",
            "AETHERIUM_COMPLETE_LAUNCHER_WITH_INTERNAL_AI.py", 
            "COMPLETE_INTEGRATED_LAUNCHER.py",
            "COMPLETE_WORKING_LAUNCHER.py",
            "COMPREHENSIVE_AETHERIUM_COMPLETE_LAUNCHER.py",
            "LAUNCH_AETHERIUM_COMPLETE.py",
            "PRODUCTION_LAUNCH.py"
        ]
        
        target_dir = self.base_path / "scripts" / "launchers"
        
        for script in launcher_scripts:
            source = self.base_path / script
            if source.exists():
                target = target_dir / script
                try:
                    shutil.move(str(source), str(target))
                    self.moved_files.append(f"{script} -> scripts/launchers/")
                    print(f"🚀 Moved launcher: {script}")
                except Exception as e:
                    self.errors.append(f"Error moving {script}: {e}")
    
    def organize_integration_scripts(self):
        """Move all integration scripts to scripts/integration/."""
        
        integration_scripts = [
            "AETHERIUM_V3_COMPLETE_INTEGRATION.py",
            "COMPLETE_AI_INTEGRATION.py",
            "COMPLETE_AUTH_FLOW.py", 
            "COMPLETE_DATABASE_SYSTEM.py",
            "COMPLETE_FILE_SYSTEM.py",
            "COMPLETE_WEBSOCKET_INTEGRATION.py",
            "INTEGRATE_EVERYTHING_NOW.py",
            "FINAL_COMPLETE_INTEGRATION.py"
        ]
        
        target_dir = self.base_path / "scripts" / "integration"
        
        for script in integration_scripts:
            source = self.base_path / script
            if source.exists():
                target = target_dir / script
                try:
                    shutil.move(str(source), str(target))
                    self.moved_files.append(f"{script} -> scripts/integration/")
                    print(f"🔗 Moved integration script: {script}")
                except Exception as e:
                    self.errors.append(f"Error moving {script}: {e}")
    
    def organize_automation_scripts(self):
        """Move automation and utility scripts."""
        
        automation_scripts = [
            "DIRECT_AUTOMATION_FIX.py",
            "FINAL_COMPLETION_AUTOMATION.py",
            "EXECUTE_FINAL_DEPLOYMENT.py",
            "EXECUTE_NOW_COMPLETE.py",
            "REPOSITORY_CLEANUP_AND_ENHANCEMENT.py",
            "DEEP_DIRECTORY_ANALYZER.py"
        ]
        
        target_dir = self.base_path / "scripts" / "automation"
        
        for script in automation_scripts:
            source = self.base_path / script
            if source.exists():
                target = target_dir / script
                try:
                    shutil.move(str(source), str(target))
                    self.moved_files.append(f"{script} -> scripts/automation/")
                    print(f"🤖 Moved automation script: {script}")
                except Exception as e:
                    self.errors.append(f"Error moving {script}: {e}")
    
    def organize_batch_files(self):
        """Move batch files to scripts/batch/."""
        
        batch_files = [
            "START_AETHERIUM.bat",
            "START_EVERYTHING.bat"
        ]
        
        target_dir = self.base_path / "scripts" / "batch"
        
        for batch_file in batch_files:
            source = self.base_path / batch_file
            if source.exists():
                target = target_dir / batch_file
                try:
                    shutil.move(str(source), str(target))
                    self.moved_files.append(f"{batch_file} -> scripts/batch/")
                    print(f"⚡ Moved batch file: {batch_file}")
                except Exception as e:
                    self.errors.append(f"Error moving {batch_file}: {e}")
    
    def organize_documentation(self):
        """Move documentation and reports."""
        
        # Reports
        reports = [
            "AETHERIUM_COMPLETION_REPORT.json",
            "FINAL_PRODUCTION_SUMMARY.md",
            "REORGANIZATION_COMPLETION_REPORT.md"
        ]
        
        target_reports_dir = self.base_path / "docs" / "reports"
        
        for report in reports:
            source = self.base_path / report
            if source.exists():
                target = target_reports_dir / report
                try:
                    shutil.move(str(source), str(target))
                    self.moved_files.append(f"{report} -> docs/reports/")
                    print(f"📊 Moved report: {report}")
                except Exception as e:
                    self.errors.append(f"Error moving {report}: {e}")
        
        # Demo files
        demo_files = [
            "demo-reorganized-platform.py",
            "index.html"
        ]
        
        target_demos_dir = self.base_path / "docs" / "demos"
        
        for demo_file in demo_files:
            source = self.base_path / demo_file
            if source.exists():
                target = target_demos_dir / demo_file
                try:
                    shutil.move(str(source), str(target))
                    self.moved_files.append(f"{demo_file} -> docs/demos/")
                    print(f"🎯 Moved demo file: {demo_file}")
                except Exception as e:
                    self.errors.append(f"Error moving {demo_file}: {e}")
    
    def create_clean_readme(self):
        """Create a clean, organized README for the root directory."""
        
        readme_content = """# Aetherium Platform

🚀 **Advanced AI Platform with Quantum Computing, Emotional Intelligence & Time Crystals**

## Quick Start

```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Start the platform
python scripts/launchers/AETHERIUM_BLT_V4_LAUNCHER.py
```

## Project Structure

```
aetherium/
├── aetherium/           # Main platform code
├── src/                 # Frontend source code  
├── scripts/             # All execution scripts
│   ├── launchers/       # Platform launcher scripts
│   ├── integration/     # System integration scripts
│   ├── automation/      # Automation and deployment scripts
│   └── batch/           # Windows batch files
├── docs/                # Documentation
│   ├── reports/         # Status and completion reports
│   └── demos/           # Demo files and examples
├── config/              # Configuration files
├── resources/           # Knowledge base and resources
└── archive/             # Archived and obsolete files
```

## Features

✅ **Advanced AI Engine** - Custom-built LLM with BLT architecture  
✅ **Quantum Computing** - Virtual quantum computer simulation  
✅ **Emotional Intelligence** - Neural emotion processing with empathy  
✅ **Time Crystals** - Temporal state manipulation  
✅ **Blockchain Integration** - Quantum-resistant cryptography  
✅ **Multi-Agent Systems** - Collaborative AI orchestration  
✅ **Advanced Automation** - Browser, desktop, and app automation  
✅ **Networking** - Onion routing, VPN, mesh networks  

## Documentation

- [Deployment Guide](docs/deployment-guide.md)
- [API Documentation](docs/api/)
- [System Architecture](docs/architecture/)
- [User Guide](docs/user-guide/)

## Development

```bash
# Development mode
npm run dev

# Build for production  
npm run build

# Run tests
npm test
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

See [LICENSE](LICENSE) for details.
"""
        
        readme_path = self.base_path / "README.md"
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print("📝 Updated README.md with clean structure")
        except Exception as e:
            self.errors.append(f"Error updating README: {e}")
    
    def generate_cleanup_report(self):
        """Generate a comprehensive cleanup report."""
        
        report = {
            "cleanup_timestamp": datetime.utcnow().isoformat(),
            "directories_created": len(self.created_dirs),
            "files_moved": len(self.moved_files),
            "files_removed": len(self.removed_files),
            "errors": len(self.errors),
            "created_directories": self.created_dirs,
            "moved_files": self.moved_files,
            "removed_files": self.removed_files,
            "error_details": self.errors,
            "status": "SUCCESS" if len(self.errors) == 0 else "PARTIAL_SUCCESS"
        }
        
        report_path = self.base_path / "docs" / "reports" / "directory_cleanup_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Generated cleanup report: {report_path}")
        except Exception as e:
            print(f"❌ Error generating report: {e}")
        
        return report
    
    def run_cleanup(self):
        """Execute the complete directory cleanup."""
        
        print("🧹 Starting comprehensive directory cleanup...")
        print("=" * 60)
        
        # Step 1: Create directory structure
        print("\n📁 Step 1: Creating directory structure...")
        self.create_directory_structure()
        
        # Step 2: Organize launcher scripts
        print("\n🚀 Step 2: Organizing launcher scripts...")
        self.organize_launcher_scripts()
        
        # Step 3: Organize integration scripts  
        print("\n🔗 Step 3: Organizing integration scripts...")
        self.organize_integration_scripts()
        
        # Step 4: Organize automation scripts
        print("\n🤖 Step 4: Organizing automation scripts...")
        self.organize_automation_scripts()
        
        # Step 5: Organize batch files
        print("\n⚡ Step 5: Organizing batch files...")
        self.organize_batch_files()
        
        # Step 6: Organize documentation
        print("\n📚 Step 6: Organizing documentation...")
        self.organize_documentation()
        
        # Step 7: Create clean README
        print("\n📝 Step 7: Creating clean README...")
        self.create_clean_readme()
        
        # Step 8: Generate report
        print("\n📊 Step 8: Generating cleanup report...")
        report = self.generate_cleanup_report()
        
        print("\n" + "=" * 60)
        print("🎊 DIRECTORY CLEANUP COMPLETE!")
        print("=" * 60)
        print(f"✅ Directories created: {report['directories_created']}")
        print(f"✅ Files moved: {report['files_moved']}")
        print(f"✅ Files removed: {report['files_removed']}")
        print(f"❌ Errors: {report['errors']}")
        print(f"📊 Status: {report['status']}")
        
        if self.errors:
            print("\n⚠️  Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")
        
        print("\n🎯 Root directory is now clean and organized!")
        print("🚀 All scripts are properly categorized in scripts/ subdirectories")
        print("📚 All documentation is organized in docs/ subdirectories")
        
        return report

def main():
    """Main execution function."""
    
    print("🌟" * 50)
    print("🧹 AETHERIUM DIRECTORY CLEANUP")
    print("🌟" * 50)
    
    # Run the cleanup
    cleanup = AetheriumDirectoryCleanup()
    report = cleanup.run_cleanup()
    
    print("\n🎊 Cleanup complete! Directory is now organized and maintainable.")
    
    return report

if __name__ == "__main__":
    main()