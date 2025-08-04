#!/usr/bin/env python3
# Ultra-minimal test to diagnose the issue
import sys

print("🔍 TESTING PYTHON EXECUTION")
print("=" * 40)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("✅ Basic Python execution working")

# Test imports one by one
try:
    import http.server
    print("✅ http.server import: OK")
except Exception as e:
    print(f"❌ http.server import: {e}")

try:
    import socketserver
    print("✅ socketserver import: OK")
except Exception as e:
    print(f"❌ socketserver import: {e}")

try:
    import webbrowser
    print("✅ webbrowser import: OK")
except Exception as e:
    print(f"❌ webbrowser import: {e}")

try:
    import threading
    print("✅ threading import: OK")
except Exception as e:
    print(f"❌ threading import: {e}")

try:
    import socket
    print("✅ socket import: OK")
except Exception as e:
    print(f"❌ socket import: {e}")

print("=" * 40)
print("🎯 DIAGNOSTIC COMPLETE")
input("Press Enter to continue...")