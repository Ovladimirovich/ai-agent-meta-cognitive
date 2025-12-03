#!/usr/bin/env python3
"""Debug script for API testing"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"

    print("🔍 Testing AI Agent Meta-Cognitive API")
    print("=" * 50)

    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        r = requests.get(f"{base_url}/", timeout=5)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   Response: {r.json()}")
        else:
            print(f"   Error: {r.text}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Health endpoint
    print("\n2. Testing health endpoint...")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Health Score: {data.get('health_score')}")
            print(f"   Issues: {data.get('issues_count')}")
        else:
            print(f"   Error: {r.text}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Agent process endpoint
    print("\n3. Testing agent/process endpoint...")
    try:
        payload = {"query": "Привет, как дела?"}
        print(f"   Sending: {payload}")
        r = requests.post(f"{base_url}/agent/process",
                         json=payload,
                         timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Response ID: {data.get('id')}")
            print(f"   Content: {data.get('content')[:100]}...")
            print(f"   Confidence: {data.get('confidence')}")
        else:
            print(f"   Error response: {r.text[:200]}")
            print(f"   Headers: {dict(r.headers)}")
    except requests.exceptions.Timeout:
        print("   Timeout: Request took too long")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: System info
    print("\n4. Testing system/info endpoint...")
    try:
        r = requests.get(f"{base_url}/system/info", timeout=5)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   Response: {r.json()}")
        else:
            print(f"   Error: {r.text}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("🔍 API testing completed")

if __name__ == "__main__":
    test_api()
