#!/usr/bin/env python3
"""
Simple test runner for Aetherium AI Productivity Suite
Validates all services, tools, and integrations
"""

import sys
import os
import asyncio
import traceback

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def main():
    """Main test runner function"""
    print("=" * 60)
    print("🚀 AETHERIUM AI PRODUCTIVITY SUITE - INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Import the test runner from our integration test file
        from ai_productivity_suite.test_suite_integration import run_integration_tests
        
        # Run all integration tests
        success = run_integration_tests()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - AETHERIUM AI SUITE IS READY!")
            print("=" * 60)
            print("🎯 Next Steps:")
            print("   • Start the backend server: python main.py")
            print("   • Launch the frontend: npm start")
            print("   • Access the productivity suite at /productivity")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED - CHECK LOGS ABOVE")
            print("=" * 60)
            return 1
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required modules are available")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)