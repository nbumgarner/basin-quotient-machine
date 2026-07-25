#!/usr/bin/env python3
"""
Coherency test: Compare L0 vs L1 attention patterns and output quality.
"""
import json
import urllib.request
import time
import sys

BASE_URL = 'http://localhost:8082'

def chat(msg, max_tokens=50):
    """Send chat request and return response."""
    payload = {
        'model': 'bqsm',
        'messages': [{'role': 'user', 'content': msg}],
        'temperature': 0.7,
        'max_tokens': max_tokens,
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f'{BASE_URL}/v1/chat/completions',
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode('utf-8'))
    elapsed = time.time() - start
    return d, elapsed

def extract_attention_stats(response):
    """Extract attention weight statistics from response."""
    content = response['choices'][0]['message']['content']
    lines = content.split('\n')

    # Find attention weights lines (format: "  Q-head 0 (KV-head 0, local): [0.50 0.00 0.50]")
    l0_weights = []
    l1_weights = []

    for line in lines:
        # Match lines like "  Q-head 0 (KV-head 0, local): [0.50 0.00 0.50]"
        if 'Q-head' in line and '[' in line and ']' in line:
            # This is layer 0 (first occurrence after "═══ Attention Weights (layer 0")
            parts = line.split('[')
            if len(parts) > 1:
                weights_str = parts[1].rstrip(']').split()
                weights = [float(w) for w in weights_str]
                l0_weights.append(weights)
        elif 'Q-head' in line and '[' in line and ']' in line:
            # Check if we're in layer 1 section
            # Look for "═══ Attention Weights (layer 1" before this line
            pass

    # Simpler approach: just return all attention lines
    return l0_weights, l1_weights

def extract_prediction(response):
    """Extract prediction token from response."""
    content = response['choices'][0]['message']['content']
    lines = content.split('\n')
    for line in reversed(lines):
        if 'Prediction:' in line:
            return line
    return 'No prediction found'

def test_coherency():
    """Run coherency tests."""
    tests = [
        'Hello world',
        'The cat sat on the mat',
        'Say hello',
        '2 + 2 = ',
    ]

    print('=== COHERENCY TEST: L0 vs L1 ===\n')

    for test in tests:
        print(f'Input: "{test}"')
        d, t = chat(test, max_tokens=10)

        # Extract stats
        l0_weights, l1_weights = extract_attention_stats(d)
        prediction = extract_prediction(d)

        # Analyze L0 attention
        if l0_weights:
            l0_variability = sum(len(w) for w in l0_weights) / len(l0_weights)
            l0_entropy = -sum(w * (w + 1e-10) for w in l0_weights[0]) if l0_weights else 0
            print(f'  L0: {len(l0_weights)} heads, avg span={l0_variability:.1f}')
        else:
            print(f'  L0: No attention data')

        # Analyze L1 attention
        if l1_weights:
            l1_variability = sum(len(w) for w in l1_weights) / len(l1_weights)
            print(f'  L1: {len(l1_weights)} heads, avg span={l1_variability:.1f}')
        else:
            print(f'  L1: No attention data')

        print(f'  Prediction: {prediction}')
        print(f'  Time: {t:.2f}s')
        print()

def test_speed():
    """Test inference speed."""
    print('=== SPEED TEST ===\n')

    times = []
    for i in range(10):
        d, t = chat('Hello', max_tokens=10)
        times.append(t)
        print(f'Run {i+1}: {t:.3f}s')

    avg = sum(times) / len(times)
    print(f'\nAverage: {avg:.3f}s')
    print(f'Throughput: {1/avg:.2f} req/sec')

if __name__ == '__main__':
    test_coherency()
    test_speed()
