#!/usr/bin/env python3
"""
Quick validation script for Aetherium AI Productivity Suite
Tests core functionality without complex dependencies
"""

import sys
import os
import asyncio

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

async def validate_suite():
    """Validate the AI Productivity Suite"""
    print("🔍 AETHERIUM AI PRODUCTIVITY SUITE - QUICK VALIDATION")
    print("=" * 55)
    
    try:
        # Test 1: Import Suite Manager
        print("✅ Testing Suite Manager Import...")
        from ai_productivity_suite.suite_manager import AISuiteManager
        
        # Test 2: Initialize Suite Manager
        print("✅ Initializing Suite Manager...")
        suite_manager = AISuiteManager()
        
        # Test 3: Check Services
        print("✅ Checking Available Services...")
        services = await suite_manager.list_available_services()
        print(f"   📋 Found {len(services)} services: {', '.join(services)}")
        
        # Test 4: Health Check
        print("✅ Running Health Check...")
        health = await suite_manager.health_check()
        print(f"   🏥 Overall Status: {health['overall_status'].upper()}")
        
        # Test 5: Suite Status
        print("✅ Getting Suite Status...")
        status = await suite_manager.get_suite_status()
        print(f"   📊 Total Services: {status['total_services']}")
        print(f"   🔧 Total Tools: {status['total_tools']}")
        print(f"   ⚡ Suite Status: {status['suite_status'].upper()}")
        
        # Test 6: Service Tool Discovery
        print("✅ Testing Service Tool Discovery...")
        for service_name in services:
            tools = await suite_manager.get_service_tools(service_name)
            print(f"   🛠️  {service_name.title()}: {len(tools)} tools")
        
        # Test 7: Individual Service Tests
        print("✅ Testing Individual Services...")
        
        # Communication Service
        print("   📞 Testing Communication Service...")
        communication_service = await suite_manager.get_service('communication')
        email_result = await communication_service.write_email(
            email_type="professional",
            recipient_info={"name": "Test User"},
            subject="Test Email",
            key_points=["Test point 1", "Test point 2"]
        )
        assert email_result.success, "Communication service failed"
        
        # Analysis Service
        print("   📊 Testing Analysis Service...")
        analysis_service = await suite_manager.get_service('analysis')
        viz_result = await analysis_service.create_data_visualization(
            data={"test": [1, 2, 3]},
            chart_type="bar"
        )
        assert viz_result.success, "Analysis service failed"
        
        # Creative Service
        print("   🎨 Testing Creative Service...")
        creative_service = await suite_manager.get_service('creative')
        sketch_result = await creative_service.convert_sketch_to_photo(
            sketch_data={"format": "test"},
            conversion_preferences={"quality": "high"}
        )
        assert sketch_result.success, "Creative service failed"
        
        # Shopping Service
        print("   🛒 Testing Shopping Service...")
        shopping_service = await suite_manager.get_service('shopping')
        coupon_result = await shopping_service.find_coupons_and_discounts(
            product_info={"category": "electronics"},
            search_preferences={"discount_threshold": 10}
        )
        assert coupon_result.success, "Shopping service failed"
        
        # Automation Service
        print("   🤖 Testing Automation Service...")
        automation_service = await suite_manager.get_service('automation')
        agent_result = await automation_service.create_ai_agent(
            agent_config={"name": "TestAgent"},
            capabilities=["test"],
            behavior_settings={"proactive": True}
        )
        assert agent_result.success, "Automation service failed"
        
        print("\n" + "=" * 55)
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Aetherium AI Productivity Suite is fully operational")
        print("=" * 55)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Some modules may not be available")
        return False
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(validate_suite())
    if success:
        print("\n🚀 Ready to launch! Start the servers and access /productivity")
    else:
        print("\n⚠️  Some issues detected. Check the logs above.")
    
    sys.exit(0 if success else 1)