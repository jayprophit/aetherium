#!/usr/bin/env python3
"""
Execute validation for Aetherium AI Productivity Suite
Runs comprehensive checks and provides deployment readiness assessment
"""

import sys
import os
import asyncio
import importlib.util

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

async def run_comprehensive_validation():
    """Run comprehensive validation of the AI Productivity Suite"""
    print("🚀 STARTING AETHERIUM AI PRODUCTIVITY SUITE VALIDATION")
    print("=" * 65)
    
    validation_results = {
        "suite_manager_test": False,
        "service_functionality_test": False,
        "api_integration_test": False,
        "overall_health": False
    }
    
    try:
        # Step 1: Test Suite Manager
        print("🔧 STEP 1: Testing Suite Manager...")
        from ai_productivity_suite.suite_manager import AISuiteManager
        
        suite_manager = AISuiteManager()
        print("   ✓ Suite Manager initialized successfully")
        
        # Get available services
        services = await suite_manager.list_available_services()
        print(f"   ✓ Found {len(services)} services: {', '.join(services)}")
        
        if len(services) == 5:
            validation_results["suite_manager_test"] = True
            print("   ✅ Suite Manager Test: PASSED")
        else:
            print("   ❌ Suite Manager Test: FAILED - Expected 5 services")
        
        # Step 2: Test Service Functionality
        print("\n🛠️ STEP 2: Testing Service Functionality...")
        
        service_test_results = []
        
        # Test Communication Service
        try:
            comm_service = await suite_manager.get_service('communication')
            result = await comm_service.write_email(
                email_type="professional",
                recipient_info={"name": "Test User", "company": "Test Corp"},
                subject="Test Email",
                key_points=["Point 1", "Point 2"]
            )
            service_test_results.append(("Communication", result.success))
            print(f"   ✓ Communication Service: {'PASSED' if result.success else 'FAILED'}")
        except Exception as e:
            service_test_results.append(("Communication", False))
            print(f"   ❌ Communication Service: FAILED - {e}")
        
        # Test Analysis Service
        try:
            analysis_service = await suite_manager.get_service('analysis')
            result = await analysis_service.create_data_visualization(
                data={"sales": [100, 150, 200], "months": ["Jan", "Feb", "Mar"]},
                chart_type="line"
            )
            service_test_results.append(("Analysis", result.success))
            print(f"   ✓ Analysis Service: {'PASSED' if result.success else 'FAILED'}")
        except Exception as e:
            service_test_results.append(("Analysis", False))
            print(f"   ❌ Analysis Service: FAILED - {e}")
        
        # Test Creative Service
        try:
            creative_service = await suite_manager.get_service('creative')
            result = await creative_service.convert_sketch_to_photo(
                sketch_data={"format": "base64", "style": "realistic"},
                conversion_preferences={"quality": "high"}
            )
            service_test_results.append(("Creative", result.success))
            print(f"   ✓ Creative Service: {'PASSED' if result.success else 'FAILED'}")
        except Exception as e:
            service_test_results.append(("Creative", False))
            print(f"   ❌ Creative Service: FAILED - {e}")
        
        # Test Shopping Service
        try:
            shopping_service = await suite_manager.get_service('shopping')
            result = await shopping_service.find_coupons_and_discounts(
                product_info={"name": "Test Product", "category": "electronics"},
                search_preferences={"discount_threshold": 10}
            )
            service_test_results.append(("Shopping", result.success))
            print(f"   ✓ Shopping Service: {'PASSED' if result.success else 'FAILED'}")
        except Exception as e:
            service_test_results.append(("Shopping", False))
            print(f"   ❌ Shopping Service: FAILED - {e}")
        
        # Test Automation Service
        try:
            automation_service = await suite_manager.get_service('automation')
            result = await automation_service.create_ai_agent(
                agent_config={"name": "TestAgent", "role": "assistant"},
                capabilities=["research", "analysis"],
                behavior_settings={"proactive": True}
            )
            service_test_results.append(("Automation", result.success))
            print(f"   ✓ Automation Service: {'PASSED' if result.success else 'FAILED'}")
        except Exception as e:
            service_test_results.append(("Automation", False))
            print(f"   ❌ Automation Service: FAILED - {e}")
        
        passed_services = sum(1 for _, success in service_test_results if success)
        total_services = len(service_test_results)
        
        if passed_services == total_services:
            validation_results["service_functionality_test"] = True
            print(f"   ✅ Service Functionality Test: PASSED ({passed_services}/{total_services})")
        else:
            print(f"   ⚠️ Service Functionality Test: PARTIAL ({passed_services}/{total_services})")
        
        # Step 3: Test API Integration
        print("\n🌐 STEP 3: Testing API Integration...")
        
        try:
            # Test API routes import
            from api.productivity_suite_routes import router
            print("   ✓ Productivity Suite API routes import successfully")
            
            # Test main app integration
            from main import app
            print("   ✓ Main FastAPI app imports successfully")
            
            validation_results["api_integration_test"] = True
            print("   ✅ API Integration Test: PASSED")
            
        except Exception as e:
            print(f"   ❌ API Integration Test: FAILED - {e}")
        
        # Step 4: Overall Health Check
        print("\n🏥 STEP 4: Overall Health Check...")
        
        try:
            health_status = await suite_manager.health_check()
            suite_status = await suite_manager.get_suite_status()
            
            print(f"   ✓ Overall Health: {health_status['overall_status'].upper()}")
            print(f"   ✓ Suite Status: {suite_status['suite_status'].upper()}")
            print(f"   ✓ Total Tools Available: {suite_status['total_tools']}")
            
            if (health_status['overall_status'] == 'healthy' and 
                suite_status['suite_status'] == 'operational'):
                validation_results["overall_health"] = True
                print("   ✅ Overall Health Check: PASSED")
            else:
                print("   ⚠️ Overall Health Check: DEGRADED")
                
        except Exception as e:
            print(f"   ❌ Overall Health Check: FAILED - {e}")
        
        # Final Assessment
        print("\n" + "=" * 65)
        print("📊 FINAL VALIDATION RESULTS")
        print("=" * 65)
        
        passed_tests = sum(validation_results.values())
        total_tests = len(validation_results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"✅ Passed Tests: {passed_tests}/{total_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 100:
            status = "🎉 FULLY OPERATIONAL"
            recommendations = [
                "🚀 System is ready for production deployment!",
                "💡 Start backend: python main.py",
                "💡 Start frontend: npm start",
                "💡 Access suite at: /productivity"
            ]
        elif success_rate >= 75:
            status = "✅ MOSTLY OPERATIONAL"
            recommendations = [
                "⚠️ System is mostly ready with minor issues",
                "💡 Review any failed components",
                "💡 Consider staging deployment first"
            ]
        else:
            status = "❌ REQUIRES ATTENTION" 
            recommendations = [
                "⚠️ Critical issues detected",
                "💡 Fix failed components before deployment",
                "💡 Re-run validation after fixes"
            ]
        
        print(f"\n🏆 OVERALL STATUS: {status}")
        print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
        
        print("\n📋 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
        
        print("\n🔧 AVAILABLE TOOLS SUMMARY:")
        for service_name in services:
            tools = await suite_manager.get_service_tools(service_name)
            print(f"   🛠️ {service_name.title()}: {len(tools)} tools")
        
        print("=" * 65)
        
        return success_rate >= 75
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Aetherium AI Productivity Suite Validation...\n")
    success = asyncio.run(run_comprehensive_validation())
    
    if success:
        print("\n🎊 VALIDATION COMPLETED SUCCESSFULLY!")
        print("The Aetherium AI Productivity Suite is ready for use!")
    else:
        print("\n⚠️ VALIDATION COMPLETED WITH ISSUES")
        print("Please review the results above and address any failures.")
    
    sys.exit(0 if success else 1)