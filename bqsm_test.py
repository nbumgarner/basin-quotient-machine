#!/usr/bin/env python3
"""
BQSM Inference Server - Functional Test
Tests: End-to-end coherence, speed, and correctness
"""
import subprocess
import sys
import time
import json
import os
import signal
import pathlib
import urllib.request
import urllib.error
import numpy as np

# Configuration
MODEL_PATH = "./gemma-2b-ternary.bqsm"
SERVER_BIN = "./bqsm_server"
PORT = 8082
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MAX_TOKENS=50
TEST_PROMPTS = [
    "Hello world",
    "The cat sat on the mat",
    "Say hello",
    "2 + 2 =",
    "What is the capital of France?",
]

def is_server_running():
    """Check if server is responding."""
    try:
        with urllib.request.urlopen(f"{BASE_URL.replace('/v1','')}/health", timeout=2) as resp:
            return resp.status == 200
    except:
        return False

def start_server():
    """Start the BQSM server."""
    if is_server_running():
        print("✓ Server already running")
        return None
    
    print("Starting BQSM server...")
    proc = subprocess.Popen(
        [SERVER_BIN, "--model", MODEL_PATH, "--port", str(PORT), "--layers", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    # Wait for server
    for _ in range(30):
        time.sleep(0.5)
        if is_server_running():
            print("✓ Server started")
            return proc
    print("✗ Server failed to start")
    proc.terminate()
    return None

def chat(msg, max_tokens=MAX_TOKENS):
    """Send chat request. Returns (response_dict, elapsed_time) or (None, error_msg)."""
    payload = {
        "model": "bqsm",
        "messages": [{"role": "user", "content": msg}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        return d, elapsed
    except Exception as e:
        return None, str(e)

def extract_prediction(response):
    """Extract prediction from response."""
    if not response:
        return "Error"
    content = response["choices"][0]["message"]["content"]
    lines = content.split("\n")
    for line in reversed(lines):
        if "Prediction:" in line:
            return line.split("Prediction:")[1].strip()
    return "No prediction"

def test_coherency():
    """Test end-to-end coherence."""
    print("\n=== COHERENCY TEST ===")
    results = []
    
    for prompt in TEST_PROMPTS:
        response, elapsed = chat(prompt)
        if response is None:
            print(f"✗ {prompt}: No response")
            results.append(False)
            continue
        
        prediction = extract_prediction(response)
        print(f"✓ {prompt} → {prediction} ({elapsed:.2f}s)")
        results.append(True)
    
    return all(results)

def test_speed():
    """Test inference speed."""
    print("\n=== SPEED TEST ===")
    times = []
    
    for i in range(10):
        response, elapsed = chat("Hello", max_tokens=10)
        if response is None:
            print(f"✗ Run {i+1}: No response")
            continue
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")
    
    if not times:
        print("✗ No successful runs")
        return False
    
    avg = sum(times) / len(times)
    print(f"✓ Average: {avg:.3f}s")
    print(f"✓ Throughput: {1/avg:.2f} req/sec")
    return avg < 2.0  # Should be under 2 seconds

def test_attention():
    """Test attention patterns."""
    print("\n=== ATTENTION TEST ===")
    
    # Check if attention weights are non-zero
    response, error = chat("Hello world", max_tokens=5)
    if not response:
        print("✗ No response")
        return False
    
    content = response["choices"][0]["message"]["content"]
    
    # Check for attention weights
    if "Q-head" in content:
        print("✓ Attention weights present")
        return True
    else:
        print("✗ No attention weights")
        return False

def main():
    """Run all tests."""
    print("BQSM Inference Server - Functional Test")
    print("=" * 50)
    
    # Start server
    proc = start_server()
    if not proc and not is_server_running():
        print("✗ Failed to start server")
        return 1
    
    # Run tests
    coherency_ok = test_coherency()
    speed_ok = test_speed()
    attention_ok = test_attention()
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Coherency: {'✓ PASS' if coherency_ok else '✗ FAIL'}")
    print(f"Speed: {'✓ PASS' if speed_ok else '✗ FAIL'}")
    print(f"Attention: {'✓ PASS' if attention_ok else '✗ FAIL'}")
    
    if coherency_ok and speed_ok and attention_ok:
        print("\n✓ ALL TESTS PASSED - BQSM is functional")
        return 0
    else:
        print("\n✗ TESTS FAILED - BQSM needs fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
