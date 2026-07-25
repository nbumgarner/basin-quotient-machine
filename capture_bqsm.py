#!/usr/bin/env python3
"""capture_bqsm.py — Run one request, read checkpoint files, compute cosine ladder."""
import subprocess, time, os, numpy as np, requests, json

SERVER = "/home/compunerd/Downloads/basin-quotient-machine/bqsm_server"
MODEL = "/home/compunerd/Downloads/gemma-2b-ternary.bqsm"
PORT = 8081

# Kill any existing server
os.system(f"fuser -k {PORT}/tcp 2>/dev/null")
time.sleep(1)

# Launch server
proc = subprocess.Popen([SERVER, "--model", MODEL, "--port", str(PORT), "--layers", "2"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

# Make request
try:
    r = requests.post(f"http://localhost:{PORT}/v1/chat/completions",
                      json={"model":"bqsm","messages":[{"role":"user","content":"the cat"}]},
                      timeout=60)
    print(f"Server response: {r.status_code}")
    d = r.json()
    c = d['choices'][0]['message']['content']
    for l in c.split('\n'):
        if 'hidden=' in l:
            print(l[:120])
except Exception as e:
    print(f"Error: {e}")

# Kill server
proc.terminate()
time.sleep(1)

# Read checkpoints
checkpoints = {}
for name in ['embed', 'L0_pre', 'L0_attn', 'L0_ffn', 'L1_pre', 'L1_attn', 'L1_ffn']:
    fn = f"/tmp/bqsm_{name}.bin"
    if os.path.exists(fn):
        data = np.fromfile(fn, dtype=np.int8)
        checkpoints[name] = data.astype(np.float32)
        print(f"BQSM {name}: {len(data)} bytes, mean={data.mean():.1f}, std={data.std():.1f}")
    else:
        print(f"MISSING: {fn}")

# Load float checkpoints
if os.path.exists('/tmp/float_checkpoints.npz'):
    fc = np.load('/tmp/float_checkpoints.npz', allow_pickle=True)
    print("\nFloat checkpoints loaded")
    
    # Compute cosine ladder
    # Map BQSM checkpoint names to float checkpoint names
    # BQSM embed = embedding for last token (token 2 = 'cat')
    # Float: embed[2] = third embedding
    float_map = {
        'embed': 'embed[2]',
        'L0_pre': 'L0_in[2]',     # x before attention
        'L0_attn': 'L0_attn[2]',  # after attention
        'L0_ffn': 'L0_ffn[2]',    # after FFN
        'L1_pre': 'L1_in[2]',
        'L1_attn': 'L1_attn[2]',
        'L1_ffn': 'L1_ffn[2]',
    }
    
    print("\n═══ COSINE LADDER ═══")
    for bqsm_name, float_name in float_map.items():
        if bqsm_name in checkpoints and float_name in fc:
            bqsm_vec = checkpoints[bqsm_name]
            float_vec = fc[float_name].flatten()
            # Take min length
            n = min(len(bqsm_vec), len(float_vec))
            bv = bqsm_vec[:n]
            fv = float_vec[:n]
            cos = np.dot(bv, fv) / (np.linalg.norm(bv) * np.linalg.norm(fv) + 1e-8)
            print(f"  {bqsm_name:10s}: cosine = {cos:+.4f}  |bqsm|={np.linalg.norm(bv):.1f}  |float|={np.linalg.norm(fv):.1f}")
