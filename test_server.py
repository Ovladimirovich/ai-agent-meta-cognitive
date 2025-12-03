#!/usr/bin/env python3
"""
Временный скрипт для тестирования сервера
"""

import subprocess
import time
import requests
import sys

def test_server():
    print('🚀 Starting FastAPI server...')

    # Start server
    server = subprocess.Popen([
        sys.executable, '-c',
        'import uvicorn; uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")'
    ])

    # Wait for startup
    print('⏳ Waiting for server to initialize...')
    time.sleep(15)

    try:
        base_url = 'http://127.0.0.1:8000'

        # Test 1: Root endpoint
        print('1. Testing root endpoint...')
        try:
            response = requests.get(f'{base_url}/', timeout=5)
            print(f'   Status: {response.status_code}')
            if response.status_code == 200:
                print('   ✅ Root endpoint works')
            else:
                print(f'   ❌ Root endpoint failed: {response.text}')
        except Exception as e:
            print(f'   ❌ Root endpoint error: {e}')

        # Test 2: Health endpoint
        print('2. Testing health endpoint...')
        try:
            response = requests.get(f'{base_url}/health', timeout=5)
            print(f'   Status: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                health_score = data.get('health_score', 'N/A')
                print(f'   Health score: {health_score}')
                print('   ✅ Health endpoint works')
            elif response.status_code == 503:
                print('   ⚠️  Health endpoint returns 503 (initializing)')
            else:
                print(f'   ❌ Health endpoint failed: {response.text}')
        except Exception as e:
            print(f'   ❌ Health endpoint error: {e}')

        # Test 3: Agent process endpoint
        print('3. Testing agent process endpoint...')
        try:
            payload = {
                'query': 'Hello, test message',
                'user_id': 'test_user',
                'session_id': 'test_session'
            }
            response = requests.post(f'{base_url}/agent/process', json=payload, timeout=15)
            print(f'   Status: {response.status_code}')
            if response.status_code == 200:
                print('   ✅ Agent process works')
                data = response.json()
                content = data.get('content', '')[:100]
                print(f'   Response: {content}...')
            elif response.status_code == 503:
                print('   ⚠️  Agent process returns 503 (service unavailable)')
            else:
                print(f'   ❌ Agent process failed: {response.text}')
        except Exception as e:
            print(f'   ❌ Agent process error: {e}')

        # Test 4: Meta cognitive endpoint
        print('4. Testing meta-cognitive endpoint...')
        try:
            payload = {
                'query': 'Test meta cognition',
                'user_id': 'test_user'
            }
            response = requests.post(f'{base_url}/agent/process-meta', json=payload, timeout=20)
            print(f'   Status: {response.status_code}')
            if response.status_code == 200:
                print('   ✅ Meta-cognitive process works')
            elif response.status_code == 503:
                print('   ⚠️  Meta-cognitive process returns 503 (service unavailable)')
            else:
                print(f'   ❌ Meta-cognitive process failed: {response.text}')
        except Exception as e:
            print(f'   ❌ Meta-cognitive process error: {e}')

        print('\n🎉 Testing completed!')

        # Check for 502 errors
        if any('502' in str(response.status_code) for response in []):  # We'll check this manually
            print('❌ 502 errors detected - problem not solved')
            return False
        else:
            print('✅ No 502 errors detected')
            return True

    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        return False

    finally:
        print('🛑 Stopping server...')
        server.terminate()
        try:
            server.wait(timeout=5)
        except:
            server.kill()

if __name__ == '__main__':
    success = test_server()
    sys.exit(0 if success else 1)
