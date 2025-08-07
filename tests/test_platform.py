"""Comprehensive Test Suite for Aetherium Platform"""
import asyncio
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def test_authentication():
    """Test authentication system comprehensively"""
    try:
        from security.auth_service import auth_service
        
        print("🔐 Testing Authentication System...")
        
        # Test admin login
        user_id = auth_service.authenticate_user("admin", "admin123")
        assert user_id is not None, "Admin authentication failed"
        print("   ✅ Admin authentication successful")
        
        # Test wrong password
        invalid_user = auth_service.authenticate_user("admin", "wrongpassword")
        assert invalid_user is None, "Invalid authentication should fail"
        print("   ✅ Invalid password rejection working")
        
        # Test token generation
        token = auth_service.generate_token(user_id)
        assert token is not None and len(token) > 20, "Token generation failed"
        print("   ✅ Token generation successful")
        
        # Test token verification
        verified_user_id = auth_service.verify_token(token)
        assert verified_user_id == user_id, "Token verification failed"
        print("   ✅ Token verification successful")
        
        # Test user registration
        reg_result = auth_service.register_user("testuser", "test@aetherium.com", "password123")
        assert reg_result["status"] == "success", "User registration failed"
        print("   ✅ User registration successful")
        
        # Test get user info
        user_info = auth_service.get_user_info(user_id)
        assert user_info is not None and user_info["username"] == "admin", "User info retrieval failed"
        print("   ✅ User info retrieval successful")
        
        print("✅ Authentication System: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Authentication System: FAILED - {e}")
        return False

def test_database():
    """Test database operations comprehensively"""
    try:
        from database.models import data_store
        
        print("🗄️ Testing Database System...")
        
        # Test user creation
        user = data_store.create_user("testuser_db", "testdb@aetherium.com")
        assert user["username"] == "testuser_db", "User creation failed"
        print("   ✅ User creation successful")
        
        # Test user retrieval
        retrieved_user = data_store.get_user(user["id"])
        assert retrieved_user is not None and retrieved_user["email"] == "testdb@aetherium.com", "User retrieval failed"
        print("   ✅ User retrieval successful")
        
        # Test user retrieval by username
        user_by_name = data_store.get_user_by_username("testuser_db")
        assert user_by_name is not None and user_by_name["id"] == user["id"], "User retrieval by username failed"
        print("   ✅ User retrieval by username successful")
        
        # Test chat session creation
        session = data_store.create_chat_session(user["id"], "Test Session")
        assert session["title"] == "Test Session" and session["user_id"] == user["id"], "Session creation failed"
        print("   ✅ Chat session creation successful")
        
        # Test message addition
        message = data_store.add_chat_message(session["id"], "user", "Hello, Aetherium!", "aetherium_quantum")
        assert message["content"] == "Hello, Aetherium!" and message["role"] == "user", "Message creation failed"
        print("   ✅ Chat message creation successful")
        
        # Test message retrieval
        messages = data_store.get_chat_messages(session["id"])
        assert len(messages) == 1 and messages[0]["content"] == "Hello, Aetherium!", "Message retrieval failed"
        print("   ✅ Chat message retrieval successful")
        
        # Test tool usage logging
        data_store.log_tool_usage(
            user["id"], 
            "calculator", 
            {"operation": "add", "a": 5, "b": 3},
            {"status": "completed", "result": 8},
            0.05
        )
        print("   ✅ Tool usage logging successful")
        
        # Test system metrics
        metrics = data_store.get_system_metrics()
        assert metrics["total_users"] > 0 and metrics["total_messages"] > 0, "System metrics failed"
        print(f"   ✅ System metrics: {metrics['total_users']} users, {metrics['total_messages']} messages")
        
        print("✅ Database System: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Database System: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_engine():
    """Test AI engine comprehensively"""
    try:
        from ai_ml.ai_engine import ai_engine, AetheriumAIModel
        
        print("🤖 Testing AI Engine System...")
        
        # Test model information
        models = ai_engine.get_models()
        assert len(models) == 3, f"Expected 3 models, got {len(models)}"
        print(f"   ✅ AI models loaded: {[m['name'] for m in models]}")
        
        # Test response generation for each model
        test_prompts = [
            ("Create a website", AetheriumAIModel.QUANTUM),
            ("Analyze data trends", AetheriumAIModel.NEURAL), 
            ("Plan a project", AetheriumAIModel.CRYSTAL)
        ]
        
        async def test_ai_responses():
            for prompt, model in test_prompts:
                print(f"   🧪 Testing {model.value} with: '{prompt[:30]}...'")
                response_chunks = []
                
                async for chunk in ai_engine.generate_response(prompt, model, "test_user", "test_session"):
                    response_chunks.append(chunk)
                    if len(response_chunks) > 10:  # Limit to avoid spam
                        break
                
                assert len(response_chunks) > 0, f"No response from {model.value}"
                print(f"      ✅ Generated {len(response_chunks)} response chunks")
        
        # Run async test
        asyncio.run(test_ai_responses())
        
        # Test usage statistics
        stats = ai_engine.get_usage_stats()
        assert stats["total_requests"] > 0, "Usage stats not updating"
        print(f"   ✅ Usage stats: {stats['total_requests']} total requests")
        
        print("✅ AI Engine System: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ AI Engine System: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tools_registry():
    """Test AI tools registry comprehensively"""
    try:
        from tools.tools_registry import tools_registry
        
        print("🛠️ Testing Tools Registry System...")
        
        # Test getting all tools
        all_tools = tools_registry.get_all_tools()
        assert len(all_tools) >= 10, f"Expected at least 10 tools, got {len(all_tools)}"
        print(f"   ✅ Tools loaded: {len(all_tools)} tools available")
        
        # Test categories
        categories = tools_registry.get_categories()
        assert len(categories) > 5, f"Expected multiple categories, got {len(categories)}"
        print(f"   ✅ Categories: {[c['name'] for c in categories]}")
        
        # Test tool execution
        test_tools = ["calculator", "data_visualization", "market_research", "video_generator", "website_builder"]
        
        async def test_tool_execution():
            for tool_name in test_tools:
                print(f"   🧪 Testing tool: {tool_name}")
                
                result = await tools_registry.execute_tool(
                    tool_name, 
                    {"test_param": "value", "operation": "test"}, 
                    "test_user"
                )
                
                assert result["status"] == "completed", f"Tool {tool_name} execution failed: {result}"
                assert "tool_name" in result, f"Tool {tool_name} missing name in result"
                print(f"      ✅ {tool_name}: {result['status']}")
        
        # Run async test
        asyncio.run(test_tool_execution())
        
        # Test tools by category
        business_tools = tools_registry.get_tools_by_category("Business")
        assert len(business_tools) > 0, "No business tools found"
        print(f"   ✅ Business tools: {len(business_tools)} tools")
        
        # Test analytics
        analytics = tools_registry.get_usage_analytics()
        assert analytics["total_tools"] == len(all_tools), "Analytics mismatch"
        print(f"   ✅ Analytics: {analytics['total_executions']} executions recorded")
        
        print("✅ Tools Registry System: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Tools Registry System: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test system integration and workflows"""
    try:
        print("🔗 Testing System Integration...")
        
        # Import all systems
        from security.auth_service import auth_service
        from ai_ml.ai_engine import ai_engine, AetheriumAIModel
        from tools.tools_registry import tools_registry
        from database.models import data_store
        
        # Test complete workflow: User → Auth → AI → Tools → Database
        print("   🧪 Testing complete workflow...")
        
        # 1. User authentication
        user_id = auth_service.authenticate_user("admin", "admin123")
        assert user_id, "Integration test: Authentication failed"
        
        # 2. Create user in database
        db_user = data_store.get_user_by_username("admin") or data_store.create_user("admin", "admin@aetherium.com")
        session = data_store.create_chat_session(db_user["id"], "Integration Test Session")
        
        # 3. AI interaction
        async def integration_test():
            response_generated = False
            async for chunk in ai_engine.generate_response(
                "Create a calculator for my business", 
                AetheriumAIModel.QUANTUM,
                db_user["id"],
                session["id"]
            ):
                if not response_generated:  # Only check first chunk
                    assert len(chunk) > 0, "AI generated empty response"
                    response_generated = True
                    break
            
            # 4. Tool execution
            calc_result = await tools_registry.execute_tool("calculator", {"expression": "5 + 3"}, db_user["id"])
            assert calc_result["status"] == "completed", "Tool execution failed"
            
            # 5. Log everything in database
            data_store.add_chat_message(session["id"], "user", "Create a calculator for my business", "aetherium_quantum")
            data_store.add_chat_message(session["id"], "assistant", "Calculator created successfully", "aetherium_quantum")
            data_store.log_tool_usage(db_user["id"], "calculator", {"expression": "5 + 3"}, calc_result)
            
            return True
        
        # Run integration test
        integration_success = asyncio.run(integration_test())
        assert integration_success, "Integration workflow failed"
        
        # Verify final state
        final_metrics = data_store.get_system_metrics()
        ai_stats = ai_engine.get_usage_stats()
        tools_analytics = tools_registry.get_usage_analytics()
        
        print(f"      ✅ Final state - Users: {final_metrics['total_users']}, Messages: {final_metrics['total_messages']}")
        print(f"      ✅ AI requests: {ai_stats['total_requests']}, Tool executions: {tools_analytics['total_executions']}")
        
        print("✅ System Integration: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ System Integration: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test_suite():
    """Run the complete Aetherium platform test suite"""
    print("🧪 AETHERIUM PLATFORM COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Testing Platform Components at {Path.cwd()}")
    print("-" * 60)
    
    # Define test suite
    test_suite = [
        ("Authentication System", test_authentication),
        ("Database System", test_database),
        ("AI Engine System", test_ai_engine), 
        ("Tools Registry System", test_tools_registry),
        ("System Integration", test_integration)
    ]
    
    # Run all tests
    passed = 0
    total = len(test_suite)
    failed_tests = []
    
    for test_name, test_function in test_suite:
        print(f"\n🔍 TESTING: {test_name}")
        print("-" * 40)
        
        try:
            if test_function():
                passed += 1
                print(f"🎉 {test_name}: PASSED\n")
            else:
                failed_tests.append(test_name)
                print(f"💥 {test_name}: FAILED\n")
        except Exception as e:
            failed_tests.append(test_name)
            print(f"💥 {test_name}: CRASHED - {e}\n")
    
    # Final results
    print("=" * 60)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PLATFORM IS PRODUCTION READY!")
        print(f"✅ {passed}/{total} test suites successful")
        print("🚀 Aetherium platform is fully operational and ready for deployment!")
        return True
    else:
        print(f"⚠️ SOME TESTS FAILED: {passed}/{total} test suites passed")
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print("🔧 Please review and fix the failing components before deployment.")
        return False

if __name__ == "__main__":
    print("🚀 Starting Aetherium Platform Test Suite...")
    
    # Set up environment
    print(f"📍 Working directory: {Path.cwd()}")
    print(f"📍 Backend path: {backend_path}")
    
    # Run comprehensive tests
    success = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"\n🏁 Test suite completed with exit code: {exit_code}")
    
    sys.exit(exit_code)