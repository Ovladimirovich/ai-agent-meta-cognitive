#!/usr/bin/env python3
"""Fix syntax error in api/main.py"""

with open('api/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the syntax error
content = content.replace('@app_test.post("/agent/process"))', '@app_test.post("/agent/process")')

with open('api/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed syntax error')
