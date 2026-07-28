#!/usr/bin/env python3
"""Spawn priority agents in tmux."""

import subprocess
import time

# Priority departments
depts = [
    ('dept-plasticity', 'gemini'),
    ('dept-lens', 'gemini'),
    ('dept-macro', 'claude'),
    ('dept-attention-routing', 'claude'),
    ('dept-super', 'claude'),
    ('dept-cluster', 'claude'),
]

for dept, model in depts:
    session = f'{dept}_agent'
    # Check if already exists
    result = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
    if session in result.stdout:
        print(f'SKIP: {dept} already running')
        continue
    
    # Create session
    try:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-x', '120', '-y', '40'], check=True)
        print(f'Created: {session}')
        time.sleep(0.5)
    except Exception as e:
        print(f'Failed {dept}: {e}')
