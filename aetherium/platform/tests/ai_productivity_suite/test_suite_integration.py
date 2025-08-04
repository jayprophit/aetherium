"""
Comprehensive Integration Tests for Aetherium AI Productivity Suite
Tests all services, tools, and API endpoints to ensure full functionality
"""

import pytest
import asyncio
import json
from typing import Dict, Any
import sys
import os

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai_productivity_suite.suite_manager import AISuiteManager
from ai_productivity_suite.services.communication_service import CommunicationService
from ai_productivity_suite.services.analysis_service import AnalysisService
from ai_productivity_suite.services.creative_service import CreativeService
from ai_productivity_suite.services.shopping_service import ShoppingService
from ai_productivity_suite.services.automation_service import AutomationService


class TestAIProductivitySuiteIntegration:
    """Comprehensive integration tests for the AI Productivity Suite"""
    
    @pytest.fixture
    async def suite_manager(self):
        """Create and initialize suite manager for testing"""
        manager = AISuiteManager()
        return manager
    
    @pytest.fixture
    async def test_services(self):
        """Create individual service instances for testing"""
        return {
            'communication': CommunicationService(),
            'analysis': AnalysisService(),
            'creative': CreativeService(),
            'shopping': ShoppingService(),
            'automation': AutomationService()
        }
    
    async def test_suite_manager_initialization(self, suite_manager):
        """Test that suite manager initializes correctly"""
        assert suite_manager is not None
        assert len(suite_manager.services) == 5
        
        expected_services = ['communication', 'analysis', 'creative', 'shopping', 'automation']
        available_services = await suite_manager.list_available_services()
        
        for service in expected_services:
            assert service in available_services
    
    async def test_suite_health_check(self, suite_manager):
        """Test suite health check functionality"""
        health_status = await suite_manager.health_check()
        
        assert health_status['overall_status'] in ['healthy', 'degraded']
        assert 'services' in health_status
        assert 'timestamp' in health_status
        
        # Verify all services report status
        for service_name in await suite_manager.list_available_services():
            assert service_name in health_status['services']
            assert 'status' in health_status['services'][service_name]
    
    async def test_communication_service_tools(self, test_services):
        """Test Communication & Voice Service tools"""
        service = test_services['communication']
        
        # Test Email Writer
        email_result = await service.write_email(
            email_type="professional",
            recipient_info={"name": "John Doe", "company": "Tech Corp"},
            subject="Project Update",
            key_points=["Progress report", "Next steps", "Timeline"]
        )
        assert email_result.success
        assert "subject" in email_result.data
        assert "body" in email_result.data
        
        # Test Voice Generator
        voice_result = await service.generate_voice(
            text="Hello, this is a test message",
            voice_preferences={"style": "professional", "speed": "normal"}
        )
        assert voice_result.success
        assert "audio_data" in voice_result.data
        
        # Test Smart Notifications
        notification_result = await service.setup_smart_notifications(
            notification_rules=[{"type": "email", "priority": "high"}],
            delivery_preferences={"channels": ["email", "sms"]}
        )
        assert notification_result.success
        assert "notification_id" in notification_result.data
    
    async def test_analysis_service_tools(self, test_services):
        """Test Analysis & Research Service tools"""
        service = test_services['analysis']
        
        # Test Data Visualization
        viz_result = await service.create_data_visualization(
            data={"sales": [100, 150, 200, 175], "months": ["Jan", "Feb", "Mar", "Apr"]},
            chart_type="line",
            style_preferences={"theme": "modern", "colors": ["blue", "green"]}
        )
        assert viz_result.success
        assert "chart_data" in viz_result.data
        
        # Test YouTube Viral Analysis
        youtube_result = await service.analyze_youtube_viral_potential(
            video_data={"title": "Amazing Tech Demo", "description": "Cool new technology"},
            analysis_preferences={"metrics": ["engagement", "shareability"]}
        )
        assert youtube_result.success
        assert "viral_score" in youtube_result.data
        
        # Test Fact Checker
        fact_result = await service.check_facts(
            content="The Earth is round and orbits the Sun",
            verification_preferences={"sources": ["scientific", "educational"]}
        )
        assert fact_result.success
        assert "verification_results" in fact_result.data
    
    async def test_creative_service_tools(self, test_services):
        """Test Creative & Design Service tools"""
        service = test_services['creative']
        
        # Test Sketch to Photo
        sketch_result = await service.convert_sketch_to_photo(
            sketch_data={"format": "base64", "style": "realistic"},
            conversion_preferences={"quality": "high", "style": "photorealistic"}
        )
        assert sketch_result.success
        assert "generated_image" in sketch_result.data
        
        # Test AI Video Generator
        video_result = await service.generate_ai_video(
            video_concept={"theme": "nature", "duration": 30},
            style_preferences={"quality": "HD", "fps": 30}
        )
        assert video_result.success
        assert "video_data" in video_result.data
        
        # Test Meme Maker
        meme_result = await service.create_meme(
            meme_template="distracted_boyfriend",
            text_content={"top": "Old Technology", "bottom": "New AI"}
        )
        assert meme_result.success
        assert "meme_image" in meme_result.data
    
    async def test_shopping_service_tools(self, test_services):
        """Test Shopping & Comparison Service tools"""
        service = test_services['shopping']
        
        # Test Price Tracker
        price_result = await service.track_price_changes(
            product_url="https://example.com/product",
            target_price=99.99,
            notification_preferences={"email": True, "threshold": 10}
        )
        assert price_result.success
        assert "tracking_id" in price_result.data
        
        # Test Deal Analyzer
        deal_result = await service.analyze_deals_and_offers(
            product_category="electronics",
            budget_range={"min": 100, "max": 500},
            deal_preferences={"discount_threshold": 20}
        )
        assert deal_result.success
        assert "deals" in deal_result.data
        
        # Test Budget Optimizer
        budget_result = await service.optimize_shopping_budget(
            budget_constraints={"total": 1000, "categories": {"electronics": 500}},
            shopping_goals={"priority": "best_value", "quality": "high"}
        )
        assert budget_result.success
        assert "optimization_plan" in budget_result.data
    
    async def test_automation_service_tools(self, test_services):
        """Test Automation & AI Agents Service tools"""
        service = test_services['automation']
        
        # Test AI Agent Creation
        agent_result = await service.create_ai_agent(
            agent_config={"name": "TestAgent", "role": "assistant"},
            capabilities=["research", "analysis"],
            behavior_settings={"proactive": True, "learning": True}
        )
        assert agent_result.success
        assert "agent_id" in agent_result.data
        
        # Test Project Manager
        project_result = await service.manage_project(
            project_details={"name": "Test Project", "deadline": "2024-12-31"},
            team_members=[{"name": "Alice", "role": "developer"}],
            management_preferences={"methodology": "agile"}
        )
        assert project_result.success
        assert "project_plan" in project_result.data
        
        # Test Data Pipeline
        pipeline_result = await service.setup_data_pipeline(
            pipeline_config={"name": "TestPipeline", "schedule": "daily"},
            data_sources=[{"type": "csv", "location": "/data/test.csv"}],
            processing_requirements={"format": "json", "validation": True}
        )
        assert pipeline_result.success
        assert "pipeline_id" in pipeline_result.data
    
    async def test_service_tool_discovery(self, suite_manager):
        """Test that all services and tools can be discovered dynamically"""
        for service_name in await suite_manager.list_available_services():
            tools = await suite_manager.get_service_tools(service_name)
            assert len(tools) > 0
            
            # Test getting info for each tool
            for tool_name in tools:
                tool_info = await suite_manager.get_tool_info(service_name, tool_name)
                assert tool_info['service'] == service_name
                assert tool_info['tool'] == tool_name
                assert tool_info['available'] == True
    
    async def test_suite_status_comprehensive(self, suite_manager):
        """Test comprehensive suite status reporting"""
        status = await suite_manager.get_suite_status()
        
        assert status['suite_status'] == 'operational'
        assert status['total_services'] == 5
        assert status['total_tools'] > 30  # We have 40+ tools total
        assert 'services' in status
        assert 'initialized_at' in status
        
        # Verify each service reports tools
        for service_name in ['communication', 'analysis', 'creative', 'shopping', 'automation']:
            assert service_name in status['services']
            assert status['services'][service_name]['status'] == 'active'
            assert status['services'][service_name]['tools_count'] > 0
    
    async def test_error_handling(self, suite_manager):
        """Test error handling for invalid requests"""
        # Test invalid service
        with pytest.raises(ValueError, match="Service 'invalid_service' not found"):
            await suite_manager.get_service('invalid_service')
        
        # Test invalid tool
        with pytest.raises(ValueError, match="Tool 'invalid_tool' not found"):
            await suite_manager.execute_tool('communication', 'invalid_tool')
    
    async def test_cross_service_integration(self, test_services):
        """Test integration between different services"""
        # Example: Use analysis service to analyze content created by creative service
        creative = test_services['creative']
        analysis = test_services['analysis']
        
        # Create content with creative service
        meme_result = await creative.create_meme(
            meme_template="success_kid",
            text_content={"top": "Successfully integrated", "bottom": "AI Productivity Suite"}
        )
        assert meme_result.success
        
        # Analyze the content with analysis service (simulated)
        # In a real scenario, this could involve image analysis, content analysis, etc.
        # For now, we'll test that both services work independently
        fact_result = await analysis.check_facts(
            content="AI Productivity Suite integration successful",
            verification_preferences={"sources": ["technical"]}
        )
        assert fact_result.success


# Helper function to run async tests
def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting Aetherium AI Productivity Suite Integration Tests...")
    
    async def run_all_tests():
        test_instance = TestAIProductivitySuiteIntegration()
        
        # Create fixtures
        suite_manager = AISuiteManager()
        test_services = {
            'communication': CommunicationService(),
            'analysis': AnalysisService(),
            'creative': CreativeService(),
            'shopping': ShoppingService(),
            'automation': AutomationService()
        }
        
        try:
            print("✅ Testing Suite Manager Initialization...")
            await test_instance.test_suite_manager_initialization(suite_manager)
            
            print("✅ Testing Suite Health Check...")
            await test_instance.test_suite_health_check(suite_manager)
            
            print("✅ Testing Communication Service...")
            await test_instance.test_communication_service_tools(test_services)
            
            print("✅ Testing Analysis Service...")
            await test_instance.test_analysis_service_tools(test_services)
            
            print("✅ Testing Creative Service...")
            await test_instance.test_creative_service_tools(test_services)
            
            print("✅ Testing Shopping Service...")
            await test_instance.test_shopping_service_tools(test_services)
            
            print("✅ Testing Automation Service...")
            await test_instance.test_automation_service_tools(test_services)
            
            print("✅ Testing Tool Discovery...")
            await test_instance.test_service_tool_discovery(suite_manager)
            
            print("✅ Testing Suite Status...")
            await test_instance.test_suite_status_comprehensive(suite_manager)
            
            print("✅ Testing Error Handling...")
            await test_instance.test_error_handling(suite_manager)
            
            print("✅ Testing Cross-Service Integration...")
            await test_instance.test_cross_service_integration(test_services)
            
            print("\n🎉 All Integration Tests Passed Successfully!")
            print("📊 Suite Status:")
            status = await suite_manager.get_suite_status()
            print(f"   • Total Services: {status['total_services']}")
            print(f"   • Total Tools: {status['total_tools']}")
            print(f"   • Suite Status: {status['suite_status'].upper()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration Test Failed: {e}")
            return False
    
    return asyncio.run(run_all_tests())


if __name__ == "__main__":
    success = run_integration_tests()
    if not success:
        exit(1)