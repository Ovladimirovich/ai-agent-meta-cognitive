#!/usr/bin/env python3
"""
Простой тест debug эндпоинта
"""

import requests

try:
    response = requests.get('http://localhost:8000/debug/test')
    print(f'Debug test status: {response.status_code}')
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print('Error:', response.text)
except Exception as e:
    print('Exception:', e)
