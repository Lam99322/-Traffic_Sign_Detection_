#!/usr/bin/env python3
"""
Simple test script to validate the web interface and check for common issues
"""
import os
import requests
import time
from pathlib import Path

def test_server_endpoints():
    """Test basic server endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Web Interface...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Server Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False
    
    # Test 2: Check API endpoints
    endpoints_to_test = [
        "/api/health",
        "/api/upload-image",
        "/api/upload-video"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            # Use HEAD request to avoid actually uploading
            response = requests.head(f"{base_url}{endpoint}", timeout=5)
            print(f"✅ Endpoint {endpoint}: Available")
        except Exception as e:
            print(f"⚠️  Endpoint {endpoint}: {e}")
    
    # Test 3: Check static files
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if "Traffic Sign Detection" in response.text:
            print("✅ HTML Template: Loading correctly")
        else:
            print("⚠️  HTML Template: May have issues")
    except Exception as e:
        print(f"❌ HTML Template: {e}")
    
    print("\n📊 Test Summary:")
    print("- Server is running on http://localhost:8000")
    print("- Web interface should be accessible")
    print("- Check browser console for JavaScript errors")
    print("\n🌐 Open http://localhost:8000 in your browser to test the interface")
    
    return True

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 Checking File Structure...")
    print("=" * 50)
    
    required_files = [
        "server/app_demo.py",
        "templates/index.html",
        "src/yolo_detector.py",
        "src/data_preprocessing.py"
    ]
    
    base_path = Path(".")
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")

def main():
    """Main test function"""
    print("🚦 Traffic Sign Detection System - Web Interface Test")
    print("=" * 60)
    
    # Check file structure first
    check_file_structure()
    
    # Test server if it's running
    print("\nWaiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    test_server_endpoints()
    
    print("\n" + "=" * 60)
    print("💡 If you see JavaScript errors in browser console:")
    print("   1. Refresh the page (F5)")
    print("   2. Clear browser cache (Ctrl+Shift+R)")
    print("   3. Check browser developer tools (F12)")
    
    print("\n🎯 System Features to Test:")
    print("   - 📁 Image Upload (tab 1)")
    print("   - 📹 Video Upload (tab 2)")
    print("   - 📷 Webcam Detection (tab 3)")
    print("   - 🔴 Real-time Stream (tab 4)")
    print("   - 📊 Analytics View (tab 5)")

if __name__ == "__main__":
    main()