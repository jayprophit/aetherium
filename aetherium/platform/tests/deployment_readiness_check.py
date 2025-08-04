#!/usr/bin/env python3
"""
Deployment Readiness Check for Aetherium AI Productivity Suite
Comprehensive validation of all components before production deployment
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

class DeploymentReadinessChecker:
    """Comprehensive deployment readiness checker"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
            "summary": {},
            "recommendations": []
        }
    
    async def run_all_checks(self):
        """Run all deployment readiness checks"""
        print("🚀 AETHERIUM AI PRODUCTIVITY SUITE - DEPLOYMENT READINESS CHECK")
        print("=" * 70)
        
        # Check 1: Core Components
        await self._check_core_components()
        
        # Check 2: AI Productivity Suite
        await self._check_ai_productivity_suite()
        
        # Check 3: API Endpoints
        await self._check_api_endpoints()
        
        # Check 4: Frontend Integration
        await self._check_frontend_integration()
        
        # Check 5: Configuration
        await self._check_configuration()
        
        # Check 6: Documentation
        await self._check_documentation()
        
        # Generate final assessment
        self._generate_final_assessment()
        
        return self.results
    
    async def _check_core_components(self):
        """Check core platform components"""
        print("🔧 Checking Core Components...")
        
        checks = {}
        
        try:
            # Check main FastAPI app
            from main import app
            checks["fastapi_app"] = {"status": "✅", "details": "FastAPI app imports successfully"}
            
            # Check suite manager
            from ai_productivity_suite.suite_manager import AISuiteManager
            checks["suite_manager"] = {"status": "✅", "details": "Suite manager imports successfully"}
            
            # Check individual services
            services_to_check = [
                "ai_productivity_suite.services.communication_service",
                "ai_productivity_suite.services.analysis_service", 
                "ai_productivity_suite.services.creative_service",
                "ai_productivity_suite.services.shopping_service",
                "ai_productivity_suite.services.automation_service"
            ]
            
            for service_module in services_to_check:
                try:
                    __import__(service_module)
                    service_name = service_module.split('.')[-1]
                    checks[service_name] = {"status": "✅", "details": f"{service_name} imports successfully"}
                except ImportError as e:
                    checks[service_name] = {"status": "❌", "details": f"Import error: {e}"}
            
        except Exception as e:
            checks["core_import_error"] = {"status": "❌", "details": f"Core import error: {e}"}
        
        self.results["checks"]["core_components"] = checks
        print(f"   📋 Core Components: {len([c for c in checks.values() if c['status'] == '✅'])}/{len(checks)} passed")
    
    async def _check_ai_productivity_suite(self):
        """Check AI Productivity Suite functionality"""
        print("🤖 Checking AI Productivity Suite...")
        
        checks = {}
        
        try:
            # Initialize suite manager
            from ai_productivity_suite.suite_manager import AISuiteManager
            suite_manager = AISuiteManager()
            
            # Check service initialization
            services = await suite_manager.list_available_services()
            checks["service_count"] = {
                "status": "✅" if len(services) == 5 else "⚠️",
                "details": f"Found {len(services)} services: {', '.join(services)}"
            }
            
            # Check health status
            health = await suite_manager.health_check()
            checks["health_check"] = {
                "status": "✅" if health["overall_status"] == "healthy" else "⚠️",
                "details": f"Health status: {health['overall_status']}"
            }
            
            # Check suite status
            status = await suite_manager.get_suite_status()
            checks["suite_status"] = {
                "status": "✅" if status["suite_status"] == "operational" else "⚠️",
                "details": f"Suite operational with {status['total_tools']} tools"
            }
            
            # Test key functionality from each service
            service_tests = []
            
            # Communication Service Test
            try:
                comm_service = await suite_manager.get_service('communication')
                result = await comm_service.write_email(
                    email_type="test", 
                    recipient_info={"name": "Test"}, 
                    subject="Test", 
                    key_points=["Test"]
                )
                service_tests.append(("communication", result.success))
            except Exception as e:
                service_tests.append(("communication", False))
            
            # Analysis Service Test
            try:
                analysis_service = await suite_manager.get_service('analysis')
                result = await analysis_service.create_data_visualization(
                    data={"test": [1, 2, 3]}, 
                    chart_type="bar"
                )
                service_tests.append(("analysis", result.success))
            except Exception as e:
                service_tests.append(("analysis", False))
            
            # Creative Service Test
            try:
                creative_service = await suite_manager.get_service('creative')
                result = await creative_service.convert_sketch_to_photo(
                    sketch_data={"format": "test"}, 
                    conversion_preferences={"quality": "high"}
                )
                service_tests.append(("creative", result.success))
            except Exception as e:
                service_tests.append(("creative", False))
            
            # Shopping Service Test
            try:
                shopping_service = await suite_manager.get_service('shopping')
                result = await shopping_service.find_coupons_and_discounts(
                    product_info={"category": "test"}, 
                    search_preferences={"discount_threshold": 10}
                )
                service_tests.append(("shopping", result.success))
            except Exception as e:
                service_tests.append(("shopping", False))
            
            # Automation Service Test
            try:
                automation_service = await suite_manager.get_service('automation')
                result = await automation_service.create_ai_agent(
                    agent_config={"name": "Test"}, 
                    capabilities=["test"], 
                    behavior_settings={"proactive": True}
                )
                service_tests.append(("automation", result.success))
            except Exception as e:
                service_tests.append(("automation", False))
            
            passed_tests = sum(1 for _, success in service_tests if success)
            checks["service_functionality"] = {
                "status": "✅" if passed_tests == len(service_tests) else "⚠️",
                "details": f"Service tests: {passed_tests}/{len(service_tests)} passed"
            }
            
        except Exception as e:
            checks["suite_error"] = {"status": "❌", "details": f"Suite error: {e}"}
        
        self.results["checks"]["ai_productivity_suite"] = checks
        print(f"   🤖 AI Suite: {len([c for c in checks.values() if c['status'] == '✅'])}/{len(checks)} passed")
    
    async def _check_api_endpoints(self):
        """Check API endpoint availability"""
        print("🌐 Checking API Endpoints...")
        
        checks = {}
        
        try:
            # Check if API routes can be imported
            from api.productivity_suite_routes import router as productivity_router
            checks["productivity_routes"] = {"status": "✅", "details": "Productivity routes import successfully"}
            
            # Check other API routes
            api_modules = [
                "api.quantum_routes",
                "api.time_crystal_routes", 
                "api.neuromorphic_routes",
                "api.ai_ml_routes",
                "api.iot_routes"
            ]
            
            for module in api_modules:
                try:
                    __import__(module)
                    module_name = module.split('.')[-1]
                    checks[module_name] = {"status": "✅", "details": f"{module_name} imports successfully"}
                except ImportError as e:
                    module_name = module.split('.')[-1]
                    checks[module_name] = {"status": "⚠️", "details": f"Import warning: {e}"}
            
        except Exception as e:
            checks["api_error"] = {"status": "❌", "details": f"API error: {e}"}
        
        self.results["checks"]["api_endpoints"] = checks
        print(f"   🌐 API Endpoints: {len([c for c in checks.values() if c['status'] == '✅'])}/{len(checks)} passed")
    
    async def _check_frontend_integration(self):
        """Check frontend integration files"""
        print("🎨 Checking Frontend Integration...")
        
        checks = {}
        
        # Check if frontend files exist
        frontend_files = [
            "../frontend/src/App.tsx",
            "../frontend/src/components/Sidebar/Sidebar.tsx",
            "../frontend/src/pages/ProductivitySuite/ProductivitySuite.tsx",
            "../frontend/src/pages/ProductivitySuite/ProductivitySuite.css",
            "../frontend/package.json"
        ]
        
        for file_path in frontend_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            file_name = os.path.basename(file_path)
            
            if os.path.exists(full_path):
                checks[file_name] = {"status": "✅", "details": f"{file_name} exists"}
            else:
                checks[file_name] = {"status": "❌", "details": f"{file_name} missing"}
        
        self.results["checks"]["frontend_integration"] = checks
        print(f"   🎨 Frontend: {len([c for c in checks.values() if c['status'] == '✅'])}/{len(checks)} files found")
    
    async def _check_configuration(self):
        """Check configuration files"""
        print("⚙️ Checking Configuration...")
        
        checks = {}
        
        # Check configuration files
        config_files = [
            "../../aetherium-config.yaml",
            "../requirements.txt",
            "../../Dockerfile",
            "../../README.md"
        ]
        
        for file_path in config_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            file_name = os.path.basename(file_path)
            
            if os.path.exists(full_path):
                checks[file_name] = {"status": "✅", "details": f"{file_name} exists"}
            else:
                checks[file_name] = {"status": "⚠️", "details": f"{file_name} missing"}
        
        self.results["checks"]["configuration"] = checks
        print(f"   ⚙️ Configuration: {len([c for c in checks.values() if c['status'] == '✅'])}/{len(checks)} files found")
    
    async def _check_documentation(self):
        """Check documentation files"""
        print("📚 Checking Documentation...")
        
        checks = {}
        
        # Check documentation files
        doc_files = [
            "../../docs/AETHERIUM_ARCHITECTURE.md",
            "../../docs/api/api-documentation.md",
            "../../docs/deployment/deployment-guide.md"
        ]
        
        for file_path in doc_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            file_name = os.path.basename(file_path)
            
            if os.path.exists(full_path):
                checks[file_name] = {"status": "✅", "details": f"{file_name} exists"}
            else:
                checks[file_name] = {"status": "⚠️", "details": f"{file_name} missing"}
        
        self.results["checks"]["documentation"] = checks
        print(f"   📚 Documentation: {len([c for c in checks.values() if c['status'] == '✅'])}/{len(checks)} files found")
    
    def _generate_final_assessment(self):
        """Generate final deployment readiness assessment"""
        print("\n" + "=" * 70)
        print("📊 DEPLOYMENT READINESS ASSESSMENT")
        print("=" * 70)
        
        total_checks = 0
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        
        for category, checks in self.results["checks"].items():
            for check_name, check_result in checks.items():
                total_checks += 1
                if check_result["status"] == "✅":
                    passed_checks += 1
                elif check_result["status"] == "⚠️":
                    warning_checks += 1
                else:
                    failed_checks += 1
        
        # Determine overall status
        if failed_checks == 0 and warning_checks <= 2:
            overall_status = "READY"
            status_emoji = "🎉"
        elif failed_checks <= 1:
            overall_status = "MOSTLY_READY"
            status_emoji = "⚠️"
        else:
            overall_status = "NOT_READY"
            status_emoji = "❌"
        
        self.results["overall_status"] = overall_status
        self.results["summary"] = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "warnings": warning_checks,
            "failed": failed_checks,
            "success_rate": f"{(passed_checks/total_checks*100):.1f}%"
        }
        
        print(f"{status_emoji} OVERALL STATUS: {overall_status}")
        print(f"📈 Success Rate: {self.results['summary']['success_rate']}")
        print(f"✅ Passed: {passed_checks}")
        print(f"⚠️ Warnings: {warning_checks}")
        print(f"❌ Failed: {failed_checks}")
        
        # Generate recommendations
        recommendations = []
        
        if overall_status == "READY":
            recommendations.extend([
                "🚀 All systems are ready for deployment!",
                "💡 Start the backend server: python main.py",
                "💡 Launch the frontend: npm start",
                "💡 Access the productivity suite at /productivity",
                "💡 Monitor system health after deployment"
            ])
        elif overall_status == "MOSTLY_READY":
            recommendations.extend([
                "⚠️ Most systems are ready with minor issues",
                "💡 Review warnings before deployment",
                "💡 Consider fixing non-critical issues",
                "💡 Deploy to staging environment first"
            ])
        else:
            recommendations.extend([
                "❌ Critical issues detected - deployment not recommended",
                "💡 Fix failed checks before proceeding",
                "💡 Review error logs and resolve issues",
                "💡 Re-run readiness check after fixes"
            ])
        
        self.results["recommendations"] = recommendations
        
        print("\n📋 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
        
        print("=" * 70)

async def main():
    """Main function to run deployment readiness check"""
    checker = DeploymentReadinessChecker()
    results = await checker.run_all_checks()
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), "deployment_readiness_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return results["overall_status"] in ["READY", "MOSTLY_READY"]

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)