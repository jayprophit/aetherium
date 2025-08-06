#!/usr/bin/env python3
"""
AETHERIUM PLATFORM VALIDATOR
Comprehensive validation script to test all features after reorganization
"""
import os
import sys
from pathlib import Path
import subprocess
import webbrowser
import threading
import time

class AetheriumValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.validation_results = []
        
    def log_result(self, test_name, status, message=""):
        """Log validation result"""
        status_symbol = "✅" if status else "❌"
        result = f"{status_symbol} {test_name}"
        if message:
            result += f": {message}"
        self.validation_results.append((test_name, status, message))
        print(result)
        
    def validate_directory_structure(self):
        """Validate the new directory structure"""
        print("\n🔍 VALIDATING DIRECTORY STRUCTURE")
        print("=" * 50)
        
        required_dirs = [
            "src/components",
            "scripts", 
            "deployment",
            "docs",
            "archive",
            "aetherium/platform",
            "aetherium/ai-systems"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            self.log_result(f"Directory: {dir_path}", exists)
            
    def validate_essential_files(self):
        """Validate essential project files"""
        print("\n📁 VALIDATING ESSENTIAL FILES")
        print("=" * 50)
        
        essential_files = [
            "src/components/AetheriumDashboard.tsx",
            "scripts/aetherium-launcher.py",
            "package.json",
            "README.md",
            ".gitignore",
            "aetherium/aetherium-config.yaml"
        ]
        
        for file_path in essential_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            self.log_result(f"File: {file_path}", exists)
            
    def validate_main_component(self):
        """Validate the main dashboard component"""
        print("\n⚛️ VALIDATING MAIN COMPONENT")
        print("=" * 50)
        
        component_path = self.project_root / "src/components/AetheriumDashboard.tsx"
        
        if not component_path.exists():
            self.log_result("Component file exists", False)
            return
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key features
            feature_checks = [
                ("Component export", "export default" in content or "const AetheriumPlatform" in content),
                ("React imports", "import React" in content or "from 'react'" in content),
                ("Sidebar navigation", "sidebar" in content.lower()),
                ("AI tools integration", "tools" in content.lower() and "ai" in content.lower()),
                ("Chat interface", "chat" in content.lower()),
                ("Quantum features", "quantum" in content.lower()),
                ("Component length", len(content) > 10000)  # Should be substantial
            ]
            
            for check_name, passed in feature_checks:
                self.log_result(check_name, passed)
                
        except Exception as e:
            self.log_result("Component file readable", False, str(e))
            
    def validate_launcher_script(self):
        """Validate the consolidated launcher script"""
        print("\n🚀 VALIDATING LAUNCHER SCRIPT")
        print("=" * 50)
        
        launcher_path = self.project_root / "scripts/aetherium-launcher.py"
        
        if not launcher_path.exists():
            self.log_result("Launcher script exists", False)
            return
            
        try:
            with open(launcher_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key functionality
            launcher_checks = [
                ("Python shebang", "#!/usr/bin/env python3" in content),
                ("Main class", "class AetheriumLauncher" in content),
                ("Server functionality", "start_server" in content),
                ("Dependency checking", "check_dependencies" in content),
                ("HTML wrapper", "create_html_wrapper" in content),
                ("Port management", "find_available_port" in content),
                ("Error handling", "try:" in content and "except" in content)
            ]
            
            for check_name, passed in launcher_checks:
                self.log_result(check_name, passed)
                
        except Exception as e:
            self.log_result("Launcher script readable", False, str(e))
            
    def validate_archived_files(self):
        """Validate that obsolete files are properly archived"""
        print("\n📦 VALIDATING ARCHIVED FILES")
        print("=" * 50)
        
        archive_dir = self.project_root / "archive/obsolete_execution_scripts"
        
        if not archive_dir.exists():
            self.log_result("Archive directory exists", False)
            return
            
        archived_files = list(archive_dir.glob("*.py"))
        archived_count = len(archived_files)
        
        self.log_result("Archive directory exists", True)
        self.log_result(f"Archived scripts count", archived_count > 15, f"Found {archived_count} archived scripts")
        
        # Check root directory is clean
        root_py_files = list(self.project_root.glob("*.py"))
        root_clean = len(root_py_files) == 0
        
        self.log_result("Root directory clean", root_clean, f"Found {len(root_py_files)} Python files in root")
        
    def test_launcher_execution(self):
        """Test if the launcher can be executed"""
        print("\n🧪 TESTING LAUNCHER EXECUTION")
        print("=" * 50)
        
        launcher_path = self.project_root / "scripts/aetherium-launcher.py"
        
        if not launcher_path.exists():
            self.log_result("Launcher execution test", False, "Launcher script not found")
            return
            
        try:
            # Test if the script can be imported (syntax check)
            result = subprocess.run([
                sys.executable, "-c", 
                f"import sys; sys.path.append('{launcher_path.parent}'); "
                f"exec(open('{launcher_path}').read()); print('✅ Syntax OK')"
            ], capture_output=True, text=True, timeout=5)
            
            syntax_ok = result.returncode == 0 and "✅ Syntax OK" in result.stdout
            self.log_result("Launcher syntax check", syntax_ok, result.stderr if result.stderr else "")
            
        except subprocess.TimeoutExpired:
            self.log_result("Launcher syntax check", False, "Script execution timeout")
        except Exception as e:
            self.log_result("Launcher syntax check", False, str(e))
            
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for _, status, _ in self.validation_results if status)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, status, message in self.validation_results:
                if not status:
                    print(f"   • {test_name}" + (f": {message}" if message else ""))
                    
        print("\n" + "=" * 50)
        
        if success_rate >= 90:
            print("🎉 VALIDATION SUCCESSFUL - Platform ready for deployment!")
            return True
        elif success_rate >= 75:
            print("⚠️ VALIDATION MOSTLY SUCCESSFUL - Minor issues to address")
            return True
        else:
            print("⚠️ VALIDATION ISSUES - Please review failed tests")
            return False
            
    def run_validation(self):
        """Run complete validation suite"""
        print("🎉 AETHERIUM PLATFORM VALIDATION")
        print("🔍 Comprehensive post-reorganization testing")
        print("=" * 50)
        
        # Run all validation tests
        self.validate_directory_structure()
        self.validate_essential_files()
        self.validate_main_component()
        self.validate_launcher_script()
        self.validate_archived_files()
        self.test_launcher_execution()
        
        # Generate final report
        return self.generate_validation_report()

def main():
    """Main validation entry point"""
    validator = AetheriumValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🚀 Ready to launch platform with:")
        print("   python scripts/aetherium-launcher.py")
    else:
        print("\n🔧 Please address validation issues before deployment")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())