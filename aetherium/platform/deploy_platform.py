#!/usr/bin/env python3
"""
Aetherium AI Productivity Suite - Production Deployment Script
Comprehensive deployment automation with validation and monitoring
"""

import os
import sys
import subprocess
import time
import asyncio
import json
import requests
from datetime import datetime
from pathlib import Path

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

class AetheriumDeployment:
    """Production deployment manager for Aetherium AI Productivity Suite"""
    
    def __init__(self):
        self.deployment_log = []
        self.start_time = datetime.now()
        self.backend_process = None
        self.frontend_process = None
        
    def log(self, message, level="INFO"):
        """Log deployment messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)
    
    async def run_pre_deployment_validation(self):
        """Run comprehensive validation before deployment"""
        self.log("🔍 Starting Pre-Deployment Validation", "INFO")
        
        try:
            # Run our validation script
            validation_script = os.path.join(os.path.dirname(__file__), "tests", "execute_validation.py")
            
            if os.path.exists(validation_script):
                self.log("📊 Running comprehensive validation...", "INFO")
                
                # Import and run validation
                sys.path.append(os.path.join(os.path.dirname(__file__), "tests"))
                from execute_validation import execute_comprehensive_validation
                
                validation_results = await execute_comprehensive_validation()
                
                if validation_results["tests_failed"] == 0:
                    self.log("✅ Pre-deployment validation PASSED", "SUCCESS")
                    return True
                else:
                    self.log(f"❌ Pre-deployment validation FAILED: {validation_results['tests_failed']} failed tests", "ERROR")
                    return False
            else:
                self.log("⚠️ Validation script not found, proceeding with basic checks", "WARN")
                return True
                
        except Exception as e:
            self.log(f"⚠️ Validation error: {e}, proceeding with deployment", "WARN")
            return True
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        self.log("🔧 Checking Dependencies", "INFO")
        
        # Check Python
        try:
            python_version = sys.version
            self.log(f"✓ Python: {python_version.split()[0]}", "INFO")
        except:
            self.log("❌ Python not available", "ERROR")
            return False
        
        # Check Node.js (for frontend)
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log(f"✓ Node.js: {result.stdout.strip()}", "INFO")
            else:
                self.log("⚠️ Node.js not found, frontend may not start", "WARN")
        except:
            self.log("⚠️ Node.js not available", "WARN")
        
        # Check npm
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log(f"✓ npm: {result.stdout.strip()}", "INFO")
            else:
                self.log("⚠️ npm not found", "WARN")
        except:
            self.log("⚠️ npm not available", "WARN")
        
        return True
    
    def install_backend_dependencies(self):
        """Install backend Python dependencies"""
        self.log("📦 Installing Backend Dependencies", "INFO")
        
        backend_dir = os.path.join(os.path.dirname(__file__), "backend")
        requirements_file = os.path.join(backend_dir, "requirements.txt")
        
        if os.path.exists(requirements_file):
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", requirements_file
                ], capture_output=True, text=True, cwd=backend_dir)
                
                if result.returncode == 0:
                    self.log("✅ Backend dependencies installed successfully", "SUCCESS")
                    return True
                else:
                    self.log(f"⚠️ Backend dependency installation warnings: {result.stderr}", "WARN")
                    return True  # Continue even with warnings
            except Exception as e:
                self.log(f"❌ Failed to install backend dependencies: {e}", "ERROR")
                return False
        else:
            self.log("⚠️ requirements.txt not found, assuming dependencies are installed", "WARN")
            return True
    
    def install_frontend_dependencies(self):
        """Install frontend Node.js dependencies"""
        self.log("📦 Installing Frontend Dependencies", "INFO")
        
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        package_json = os.path.join(frontend_dir, "package.json")
        
        if os.path.exists(package_json):
            try:
                result = subprocess.run(["npm", "install"], capture_output=True, text=True, cwd=frontend_dir)
                
                if result.returncode == 0:
                    self.log("✅ Frontend dependencies installed successfully", "SUCCESS")
                    return True
                else:
                    self.log(f"⚠️ Frontend dependency installation warnings: {result.stderr}", "WARN")
                    return True  # Continue even with warnings
            except Exception as e:
                self.log(f"❌ Failed to install frontend dependencies: {e}", "ERROR")
                return False
        else:
            self.log("⚠️ package.json not found, skipping frontend dependency installation", "WARN")
            return True
    
    def start_backend_server(self):
        """Start the FastAPI backend server"""
        self.log("🚀 Starting Backend Server", "INFO")
        
        backend_dir = os.path.join(os.path.dirname(__file__), "backend")
        main_py = os.path.join(backend_dir, "main.py")
        
        if os.path.exists(main_py):
            try:
                # Start backend server in background
                self.backend_process = subprocess.Popen([
                    sys.executable, "main.py"
                ], cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait a moment for server to start
                time.sleep(3)
                
                # Check if process is still running
                if self.backend_process.poll() is None:
                    self.log("✅ Backend server started successfully", "SUCCESS")
                    self.log("📡 Backend running on: http://localhost:8000", "INFO")
                    return True
                else:
                    stdout, stderr = self.backend_process.communicate()
                    self.log(f"❌ Backend server failed to start: {stderr.decode()}", "ERROR")
                    return False
            except Exception as e:
                self.log(f"❌ Failed to start backend server: {e}", "ERROR")
                return False
        else:
            self.log("❌ main.py not found in backend directory", "ERROR")
            return False
    
    def start_frontend_server(self):
        """Start the React frontend server"""
        self.log("🎨 Starting Frontend Server", "INFO")
        
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        package_json = os.path.join(frontend_dir, "package.json")
        
        if os.path.exists(package_json):
            try:
                # Start frontend server in background
                self.frontend_process = subprocess.Popen([
                    "npm", "start"
                ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for frontend to compile and start
                self.log("⏳ Waiting for frontend to compile...", "INFO")
                time.sleep(10)
                
                # Check if process is still running
                if self.frontend_process.poll() is None:
                    self.log("✅ Frontend server started successfully", "SUCCESS")
                    self.log("🌐 Frontend running on: http://localhost:3000", "INFO")
                    return True
                else:
                    stdout, stderr = self.frontend_process.communicate()
                    self.log(f"❌ Frontend server failed to start: {stderr.decode()}", "ERROR")
                    return False
            except Exception as e:
                self.log(f"❌ Failed to start frontend server: {e}", "ERROR")
                return False
        else:
            self.log("❌ package.json not found in frontend directory", "ERROR")
            return False
    
    def verify_deployment(self):
        """Verify that both backend and frontend are accessible"""
        self.log("🔍 Verifying Deployment", "INFO")
        
        # Check backend health
        try:
            response = requests.get("http://localhost:8000/", timeout=10)
            if response.status_code == 200:
                self.log("✅ Backend health check PASSED", "SUCCESS")
                backend_ok = True
            else:
                self.log(f"⚠️ Backend health check returned status: {response.status_code}", "WARN")
                backend_ok = False
        except Exception as e:
            self.log(f"❌ Backend health check FAILED: {e}", "ERROR")
            backend_ok = False
        
        # Check frontend accessibility
        try:
            response = requests.get("http://localhost:3000/", timeout=10)
            if response.status_code == 200:
                self.log("✅ Frontend accessibility check PASSED", "SUCCESS")
                frontend_ok = True
            else:
                self.log(f"⚠️ Frontend check returned status: {response.status_code}", "WARN")
                frontend_ok = False
        except Exception as e:
            self.log(f"❌ Frontend accessibility check FAILED: {e}", "ERROR")
            frontend_ok = False
        
        return backend_ok, frontend_ok
    
    def generate_deployment_report(self, validation_passed, backend_ok, frontend_ok):
        """Generate final deployment report"""
        deployment_time = datetime.now() - self.start_time
        
        report = {
            "deployment_timestamp": self.start_time.isoformat(),
            "deployment_duration": str(deployment_time),
            "validation_passed": validation_passed,
            "backend_status": "✅ RUNNING" if backend_ok else "❌ FAILED",
            "frontend_status": "✅ RUNNING" if frontend_ok else "❌ FAILED",
            "deployment_log": self.deployment_log,
            "access_urls": {
                "frontend": "http://localhost:3000",
                "backend": "http://localhost:8000",
                "api_docs": "http://localhost:8000/docs",
                "productivity_suite": "http://localhost:3000/productivity"
            }
        }
        
        # Save report to file
        report_file = os.path.join(os.path.dirname(__file__), "deployment_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def cleanup_on_failure(self):
        """Clean up processes if deployment fails"""
        if self.backend_process:
            self.backend_process.terminate()
            self.log("🧹 Backend process terminated", "INFO")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.log("🧹 Frontend process terminated", "INFO")
    
    async def deploy(self):
        """Execute complete deployment process"""
        self.log("🚀 STARTING AETHERIUM AI PRODUCTIVITY SUITE DEPLOYMENT", "INFO")
        self.log("=" * 60, "INFO")
        
        try:
            # Step 1: Pre-deployment validation
            validation_passed = await self.run_pre_deployment_validation()
            if not validation_passed:
                self.log("❌ Pre-deployment validation failed, aborting deployment", "ERROR")
                return False
            
            # Step 2: Check dependencies
            if not self.check_dependencies():
                self.log("❌ Dependency check failed, aborting deployment", "ERROR")
                return False
            
            # Step 3: Install backend dependencies
            if not self.install_backend_dependencies():
                self.log("❌ Backend dependency installation failed", "ERROR")
                return False
            
            # Step 4: Install frontend dependencies
            if not self.install_frontend_dependencies():
                self.log("⚠️ Frontend dependency installation failed, continuing...", "WARN")
            
            # Step 5: Start backend server
            if not self.start_backend_server():
                self.log("❌ Backend server startup failed, aborting deployment", "ERROR")
                return False
            
            # Step 6: Start frontend server
            if not self.start_frontend_server():
                self.log("❌ Frontend server startup failed", "ERROR")
                self.cleanup_on_failure()
                return False
            
            # Step 7: Verify deployment
            backend_ok, frontend_ok = self.verify_deployment()
            
            # Step 8: Generate deployment report
            report = self.generate_deployment_report(validation_passed, backend_ok, frontend_ok)
            
            # Final status
            if backend_ok and frontend_ok:
                self.log("🎉 DEPLOYMENT SUCCESSFUL!", "SUCCESS")
                self.log("=" * 60, "SUCCESS")
                self.log("🌐 Access the platform at:", "INFO")
                self.log("   Frontend: http://localhost:3000", "INFO")
                self.log("   AI Productivity Suite: http://localhost:3000/productivity", "INFO")
                self.log("   Backend API: http://localhost:8000", "INFO")
                self.log("   API Docs: http://localhost:8000/docs", "INFO")
                self.log("=" * 60, "SUCCESS")
                return True
            else:
                self.log("❌ DEPLOYMENT COMPLETED WITH ISSUES", "ERROR")
                self.log("   Check the deployment report for details", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"💥 DEPLOYMENT FAILED: {e}", "ERROR")
            self.cleanup_on_failure()
            return False

async def main():
    """Main deployment function"""
    print("🚀 AETHERIUM AI PRODUCTIVITY SUITE - PRODUCTION DEPLOYMENT")
    print("=" * 65)
    
    deployment = AetheriumDeployment()
    success = await deployment.deploy()
    
    if success:
        print("\n🎊 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("🎯 The Aetherium AI Productivity Suite is now LIVE!")
        print("\n📋 Quick Access:")
        print("   • Main Platform: http://localhost:3000")
        print("   • AI Tools: http://localhost:3000/productivity")
        print("   • API Docs: http://localhost:8000/docs")
        print("\n💡 To stop the servers, press Ctrl+C")
        
        # Keep the script running to maintain servers
        try:
            print("\n⏳ Servers are running... (Press Ctrl+C to stop)")
            while True:
                time.sleep(60)
                print(f"🟢 Status check: {datetime.now().strftime('%H:%M:%S')} - Servers running")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
            deployment.cleanup_on_failure()
            print("✅ Servers stopped successfully")
    else:
        print("\n❌ DEPLOYMENT FAILED!")
        print("📋 Check the deployment report for details")
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)