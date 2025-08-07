#!/usr/bin/env python3
"""
AETHERIUM DIRECTORY REORGANIZATION EXECUTOR
==========================================
Comprehensive file and folder organization with automated execution.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    """Execute comprehensive directory reorganization"""
    print("🗂️ EXECUTING AETHERIUM DIRECTORY REORGANIZATION...")
    print("=" * 60)
    print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = Path.cwd()
    print(f"📍 Working Directory: {base_dir}")
    
    try:
        # PHASE 1: CREATE NEW DIRECTORY STRUCTURE
        print("\n📁 PHASE 1: CREATING ORGANIZED DIRECTORY STRUCTURE...")
        print("-" * 50)
        
        # Define new directory structure
        new_structure = {
            "platform": {
                "frontend": {"components", "pages", "services", "hooks", "utils", "assets"},
                "shared": {"types", "constants", "utils"}
                # backend already exists and is organized
            },
            "ai-systems": {
                "core": {},
                "models": {},
                "training": {}
            },
            "automation": {
                "launchers": {},
                "deployment": {},
                "utilities": {}
            },
            "infrastructure": {
                "config": {},
                "docker": {},
                "deployment": {}
            },
            "tests": {
                "unit": {},
                "integration": {},
                "e2e": {}
            },
            "docs": {
                "api": {},
                "guides": {},
                "architecture": {}
            },
            "resources": {
                "knowledge-base": {},
                "assets": {},
                "data": {}
            },
            "scripts": {
                "build": {},
                "dev": {},
                "utils": {}
            }
        }
        
        def create_directories(base_path, structure):
            """Recursively create directory structure"""
            for name, subdirs in structure.items():
                dir_path = base_path / name
                dir_path.mkdir(exist_ok=True)
                print(f"   ✅ Created: {dir_path.relative_to(base_dir)}")
                
                if isinstance(subdirs, dict):
                    create_directories(dir_path, subdirs)
                elif isinstance(subdirs, set):
                    for subdir in subdirs:
                        subdir_path = dir_path / subdir
                        subdir_path.mkdir(exist_ok=True)
                        print(f"      └─ {subdir}")
        
        create_directories(base_dir, new_structure)
        
        # PHASE 2: MOVE AUTOMATION SCRIPTS
        print("\n🤖 PHASE 2: ORGANIZING AUTOMATION SCRIPTS...")
        print("-" * 50)
        
        # Define automation script mappings
        automation_moves = {
            "launchers": [
                "AETHERIUM_PLATFORM_LAUNCHER.py",
                "IMPLEMENT_MISSING_COMPONENTS_NOW.py",
                "aetherium/RUN_AUTOMATION_NOW.py",
                "aetherium/RUN_AUTONOMOUS_NOW.py", 
                "aetherium/RUN_IMMEDIATE.py",
                "scripts/aetherium-launcher.py"
            ],
            "deployment": [
                "aetherium/AUTOMATED_MISSING_COMPONENTS_FIX.py",
                "aetherium/AUTONOMOUS_EXECUTION_COMPLETE.py",
                "aetherium/DIRECT_AUTONOMOUS_EXECUTION.py",
                "aetherium/DIRECT_EXECUTION.py",
                "scripts/production-deploy.py"
            ],
            "utilities": [
                "aetherium/EXECUTE_AUTOMATED_FIX.py",
                "aetherium/EXECUTE_AUTONOMOUS_NOW.py",
                "aetherium/EXECUTE_MISSING_COMPONENTS.py",
                "aetherium/EXECUTE_NOW_AUTONOMOUS.py",
                "aetherium/MISSING_COMPONENTS_IMPLEMENTATION.py",
                "MISSING_COMPONENTS_ANALYSIS.py",
                "scripts/validate-platform.py"
            ]
        }
        
        def move_files_to_category(category, file_list):
            """Move files to automation category"""
            target_dir = base_dir / "automation" / category
            moved_count = 0
            
            for file_path_str in file_list:
                source_path = base_dir / file_path_str
                if source_path.exists():
                    target_path = target_dir / source_path.name
                    
                    # Avoid overwriting - add suffix if needed
                    counter = 1
                    while target_path.exists():
                        name_parts = source_path.name.split('.')
                        if len(name_parts) > 1:
                            new_name = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                        else:
                            new_name = f"{source_path.name}_{counter}"
                        target_path = target_dir / new_name
                        counter += 1
                    
                    shutil.move(str(source_path), str(target_path))
                    print(f"   📦 Moved: {source_path.name} → automation/{category}/")
                    moved_count += 1
            
            return moved_count
        
        total_moved = 0
        for category, files in automation_moves.items():
            moved = move_files_to_category(category, files)
            total_moved += moved
            print(f"   ✅ {category}: {moved} files organized")
        
        print(f"   📊 Total automation files moved: {total_moved}")
        
        # PHASE 3: ORGANIZE FRONTEND FILES
        print("\n⚛️ PHASE 3: ORGANIZING FRONTEND FILES...")
        print("-" * 50)
        
        # Move React files from src/ to platform/frontend/
        src_dir = base_dir / "src"
        frontend_dir = base_dir / "platform" / "frontend"
        
        if src_dir.exists():
            frontend_moves = {
                "components": ["components"],
                "services": ["services"],
                "hooks": ["hooks"],
                "utils": ["utils"],
                "pages": [],  # Create manually if needed
                "assets": []  # Create manually if needed
            }
            
            # Move main React files to root of frontend
            react_files = ["App.tsx", "main.tsx", "index.css"]
            for file_name in react_files:
                source = src_dir / file_name
                if source.exists():
                    target = frontend_dir / file_name
                    shutil.move(str(source), str(target))
                    print(f"   📦 Moved: {file_name} → platform/frontend/")
            
            # Move directories
            for target_name, source_dirs in frontend_moves.items():
                for source_name in source_dirs:
                    source_path = src_dir / source_name
                    if source_path.exists() and source_path.is_dir():
                        target_path = frontend_dir / target_name
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        shutil.move(str(source_path), str(target_path))
                        print(f"   📁 Moved: {source_name}/ → platform/frontend/{target_name}/")
        
        # PHASE 4: ORGANIZE AI SYSTEMS
        print("\n🤖 PHASE 4: ORGANIZING AI SYSTEMS...")
        print("-" * 50)
        
        # Move AI-related directories
        ai_moves = {
            "core": ["aetherium/ai-systems", "src/ai"],
            "models": [],  # Will be created as needed
            "training": []  # Will be created as needed
        }
        
        ai_target = base_dir / "ai-systems"
        for target_name, source_dirs in ai_moves.items():
            for source_name in source_dirs:
                source_path = base_dir / source_name
                if source_path.exists() and source_path.is_dir():
                    target_path = ai_target / target_name
                    
                    # If target exists, merge contents
                    if target_path.exists():
                        for item in source_path.iterdir():
                            item_target = target_path / item.name
                            if item.is_dir():
                                shutil.move(str(item), str(item_target))
                            else:
                                shutil.move(str(item), str(item_target))
                        shutil.rmtree(source_path)
                    else:
                        shutil.move(str(source_path), str(target_path))
                    
                    print(f"   🧠 Moved: {source_name} → ai-systems/{target_name}/")
        
        # PHASE 5: ORGANIZE DOCUMENTATION
        print("\n📚 PHASE 5: ORGANIZING DOCUMENTATION...")
        print("-" * 50)
        
        docs_target = base_dir / "docs"
        
        # Move documentation files
        doc_files = [
            ("aetherium/DEPLOYMENT_GUIDE.md", "guides"),
            ("aetherium/FINAL_COMPLETION_REPORT.md", "guides"),
            ("aetherium/README.md", "guides"),
            ("DIRECTORY_REORGANIZATION_PLAN.md", "architecture")
        ]
        
        for source_file, category in doc_files:
            source_path = base_dir / source_file
            if source_path.exists():
                target_dir = docs_target / category
                target_path = target_dir / source_path.name
                shutil.move(str(source_path), str(target_path))
                print(f"   📄 Moved: {source_path.name} → docs/{category}/")
        
        # Move existing docs directories
        existing_docs = ["aetherium/docs", "docs"]  # Note: docs already exists at root
        for doc_dir_name in existing_docs:
            source_dir = base_dir / doc_dir_name
            if source_dir.exists() and source_dir != docs_target:
                # Merge contents
                for item in source_dir.iterdir():
                    if item.is_file():
                        target = docs_target / "guides" / item.name
                        shutil.move(str(item), str(target))
                    elif item.is_dir():
                        target = docs_target / item.name
                        if target.exists():
                            # Merge directory contents
                            for subitem in item.iterdir():
                                sub_target = target / subitem.name
                                shutil.move(str(subitem), str(sub_target))
                            shutil.rmtree(item)
                        else:
                            shutil.move(str(item), str(target))
                
                if source_dir != docs_target:
                    shutil.rmtree(source_dir)
                print(f"   📚 Merged: {doc_dir_name} → docs/")
        
        # PHASE 6: ORGANIZE RESOURCES
        print("\n📦 PHASE 6: ORGANIZING RESOURCES...")
        print("-" * 50)
        
        resources_target = base_dir / "resources"
        
        # Move existing resources
        resource_dirs = ["aetherium/resources", "resources"]  # Note: resources exists at root
        for res_dir_name in resource_dirs:
            source_dir = base_dir / res_dir_name
            if source_dir.exists() and source_dir != resources_target:
                # Move knowledge-base specifically
                kb_source = source_dir / "knowledge"
                kb_source2 = source_dir / "knowledge-base"
                
                if kb_source.exists():
                    kb_target = resources_target / "knowledge-base"
                    if kb_target.exists():
                        shutil.rmtree(kb_target)
                    shutil.move(str(kb_source), str(kb_target))
                    print(f"   🧠 Moved: knowledge → resources/knowledge-base/")
                
                if kb_source2.exists():
                    kb_target = resources_target / "knowledge-base"
                    if kb_target.exists():
                        # Merge contents
                        for item in kb_source2.iterdir():
                            target = kb_target / item.name
                            shutil.move(str(item), str(target))
                        shutil.rmtree(kb_source2)
                    else:
                        shutil.move(str(kb_source2), str(kb_target))
                    print(f"   🧠 Moved: knowledge-base → resources/knowledge-base/")
                
                # Move other resource files
                for item in source_dir.iterdir():
                    if item.name not in ["knowledge", "knowledge-base"]:
                        target = resources_target / "data" / item.name
                        shutil.move(str(item), str(target))
                        print(f"   📦 Moved: {item.name} → resources/data/")
                
                if source_dir != resources_target:
                    if source_dir.exists() and not any(source_dir.iterdir()):
                        shutil.rmtree(source_dir)
        
        # PHASE 7: ORGANIZE INFRASTRUCTURE
        print("\n🏗️ PHASE 7: ORGANIZING INFRASTRUCTURE...")
        print("-" * 50)
        
        infra_target = base_dir / "infrastructure"
        
        # Move configuration files
        config_files = [
            ("aetherium/aetherium-config.yaml", "config"),
            ("aetherium/.env", "config"),
            ("aetherium/Dockerfile", "docker"),
            (".env.example", "config")
        ]
        
        for source_file, category in config_files:
            source_path = base_dir / source_file
            if source_path.exists():
                target_dir = infra_target / category
                target_path = target_dir / source_path.name
                shutil.move(str(source_path), str(target_path))
                print(f"   ⚙️ Moved: {source_path.name} → infrastructure/{category}/")
        
        # Move deployment directory
        deploy_dirs = ["aetherium/deployment", "deployment"]
        for deploy_dir_name in deploy_dirs:
            source_dir = base_dir / deploy_dir_name
            if source_dir.exists():
                target_dir = infra_target / "deployment"
                if target_dir.exists():
                    # Merge contents
                    for item in source_dir.iterdir():
                        target = target_dir / item.name
                        shutil.move(str(item), str(target))
                    shutil.rmtree(source_dir)
                else:
                    shutil.move(str(source_dir), str(target_dir))
                print(f"   🚀 Moved: {deploy_dir_name} → infrastructure/deployment/")
        
        # PHASE 8: ORGANIZE TESTS
        print("\n🧪 PHASE 8: ORGANIZING TESTS...")
        print("-" * 50)
        
        tests_target = base_dir / "tests"
        
        # Move existing test files
        test_files = ["test_platform.py"]
        for test_file in test_files:
            source_path = tests_target / test_file  # Already exists in tests/
            if source_path.exists():
                target_path = tests_target / "integration" / test_file
                shutil.move(str(source_path), str(target_path))
                print(f"   🧪 Moved: {test_file} → tests/integration/")
        
        # Move test directories from src/
        src_test_dir = base_dir / "src" / "test"
        if src_test_dir.exists():
            target_dir = tests_target / "unit"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(src_test_dir), str(target_dir))
            print(f"   🧪 Moved: src/test → tests/unit/")
        
        # PHASE 9: ORGANIZE SCRIPTS
        print("\n📜 PHASE 9: ORGANIZING SCRIPTS...")
        print("-" * 50)
        
        scripts_target = base_dir / "scripts"
        
        # Move remaining script directories
        if (base_dir / "scripts").exists():
            for item in (base_dir / "scripts").iterdir():
                if item.is_dir() and item.name not in ["build", "dev", "utils"]:
                    target = scripts_target / "utils" / item.name
                    shutil.move(str(item), str(target))
                    print(f"   📜 Moved: scripts/{item.name} → scripts/utils/{item.name}")
        
        # PHASE 10: CLEAN UP REMAINING FILES
        print("\n🧹 PHASE 10: CLEANING UP REMAINING FILES...")
        print("-" * 50)
        
        # Remove empty aetherium subdirectory if it exists
        aetherium_subdir = base_dir / "aetherium"
        if aetherium_subdir.exists():
            remaining_files = list(aetherium_subdir.iterdir())
            print(f"   📋 Remaining files in aetherium/: {len(remaining_files)}")
            
            # Move any remaining important files
            for item in remaining_files:
                if item.is_file():
                    if item.suffix == '.py':
                        target = base_dir / "automation" / "utilities" / item.name
                        shutil.move(str(item), str(target))
                        print(f"   📦 Moved: {item.name} → automation/utilities/")
                    elif item.suffix in ['.md', '.txt']:
                        target = base_dir / "docs" / "guides" / item.name
                        shutil.move(str(item), str(target))
                        print(f"   📄 Moved: {item.name} → docs/guides/")
                    else:
                        target = base_dir / "resources" / "data" / item.name
                        shutil.move(str(item), str(target))
                        print(f"   📦 Moved: {item.name} → resources/data/")
                elif item.is_dir() and not any(item.iterdir()):
                    shutil.rmtree(item)
                    print(f"   🗑️ Removed empty: {item.name}/")
            
            # Remove aetherium directory if empty
            if not any(aetherium_subdir.iterdir()):
                shutil.rmtree(aetherium_subdir)
                print(f"   🗑️ Removed empty directory: aetherium/")
        
        # Remove empty src directory if it exists
        src_dir = base_dir / "src"
        if src_dir.exists() and not any(src_dir.iterdir()):
            shutil.rmtree(src_dir)
            print(f"   🗑️ Removed empty directory: src/")
        
        # PHASE 11: CREATE SUMMARY AND VALIDATION
        print("\n📊 PHASE 11: SUMMARY AND VALIDATION...")
        print("-" * 50)
        
        # Create final directory summary
        final_structure = {}
        for item in base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                subdir_count = len([x for x in item.iterdir() if x.is_dir()])
                file_count = len([x for x in item.iterdir() if x.is_file()])
                final_structure[item.name] = {"dirs": subdir_count, "files": file_count}
        
        print("\n🏁 REORGANIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📋 FINAL DIRECTORY STRUCTURE:")
        print("┌─────────────────────────────────────────────┐")
        print("│              ORGANIZED STRUCTURE            │")
        print("├─────────────────────────────────────────────┤")
        
        structure_icons = {
            "platform": "🏛️",
            "ai-systems": "🤖", 
            "automation": "⚙️",
            "infrastructure": "🏗️",
            "tests": "🧪",
            "docs": "📚",
            "resources": "📦",
            "scripts": "📜",
            "backend": "💾",
            "archive": "📁",
            "node_modules": "📦"
        }
        
        for dir_name, counts in sorted(final_structure.items()):
            icon = structure_icons.get(dir_name, "📁")
            print(f"│ {icon} {dir_name:<20} │ {counts['dirs']:2d} dirs, {counts['files']:2d} files   │")
        
        print("└─────────────────────────────────────────────┘")
        
        print("\n✅ ORGANIZATION BENEFITS:")
        print("   • Clear separation of concerns")
        print("   • Logical grouping of related files")
        print("   • Reduced duplication and confusion")
        print("   • Professional project structure")
        print("   • Easier navigation and maintenance")
        
        return True
        
    except Exception as e:
        print(f"\n❌ REORGANIZATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🗂️ AETHERIUM DIRECTORY REORGANIZATION")
    print("Executing comprehensive file and folder organization...")
    
    success = main()
    
    if success:
        print("\n🎉 REORGANIZATION COMPLETED SUCCESSFULLY!")
        print("Directory structure is now clean and organized!")
        exit_code = 0
    else:
        print("\n❌ REORGANIZATION ENCOUNTERED ERRORS")
        print("Some files may need manual organization.")
        exit_code = 1
    
    print(f"\nExiting with code: {exit_code}")
    exit(exit_code)