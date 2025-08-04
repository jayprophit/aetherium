#!/usr/bin/env python3
"""EXECUTE DEPLOYMENT - IMMEDIATE LAUNCH"""
import os
import subprocess
import sys
import time
import threading

def execute_deployment():
    print("🚀 EXECUTING AETHERIUM DEPLOYMENT...")
    print("=" * 50)
    
    # Change to platform directory
    platform_dir = r"C:\Users\jpowe\CascadeProjects\github\aetherium\aetherium\platform"
    os.chdir(platform_dir)
    print(f"Working directory: {platform_dir}")
    
    # Execute INSTANT_DEPLOY.py
    print("Launching INSTANT_DEPLOY.py...")
    
    try:
        # Use Popen to run in background and capture output
        process = subprocess.Popen(
            [sys.executable, "INSTANT_DEPLOY.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=platform_dir
        )
        
        print("✅ Deployment process started!")
        print("Process ID:", process.pid)
        
        # Read output in real-time
        def read_output():
            for line in iter(process.stdout.readline, ''):
                print(f"[DEPLOY] {line.strip()}")
        
        def read_errors():
            for line in iter(process.stderr.readline, ''):
                print(f"[ERROR] {line.strip()}")
        
        # Start threads to read output
        output_thread = threading.Thread(target=read_output)
        error_thread = threading.Thread(target=read_errors)
        output_thread.daemon = True
        error_thread.daemon = True
        output_thread.start()
        error_thread.start()
        
        print("✅ Deployment script is running...")
        print("✅ Check your browser - it should open automatically!")
        print("✅ Platform will be available at http://localhost:3000-3100")
        
        # Wait a bit then return
        time.sleep(2)
        print("✅ DEPLOYMENT INITIATED SUCCESSFULLY!")
        
        return process
        
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return None

if __name__ == "__main__":
    process = execute_deployment()
    if process:
        print("\n🎯 DEPLOYMENT STATUS: RUNNING")
        print("🌐 Your Aetherium platform is deploying...")
        print("📱 Browser should open automatically")
        print("✅ All features will be tested automatically")
    else:
        print("\n❌ DEPLOYMENT FAILED")