#!/usr/bin/env python3
"""Restart agent sessions properly with hermes chat."""

import subprocess
import time
import os

# Kill existing sessions
subprocess.run(['tmux', 'kill-server'], capture_output=True)
time.sleep(1)

# Priority departments with hermes commands
agents = [
    ('dept-plasticity', 'gemini', 'Implement Hebbian plasticity engine per PLASTICITY_RULE_DESIGN.md — Oja + competition in harmonic basis. Target: single-ring convergence < 1000 chunks. Report progress to .heartbeat_state/dept-plasticity_progress.txt'),
    ('dept-lens', 'gemini', 'Design pulse-receptive lens profiles for macro aggregation fix. Current: 25% harmonic detection → target >90%. Characterize pulse→harmonic transfer function. Report progress to .heartbeat_state/dept-lens_progress.txt'),
    ('dept-macro', 'claude', 'Implement macro aggregation with optimized destination lenses. Test cross-vQPU feature routing. Report progress to .heartbeat_state/dept-macro_progress.txt'),
    ('dept-attention-routing', 'claude', 'Implement Hebbian attention weight learning on macro harmonic correlations. Current routing: 50% chance → target >85%. Report progress to .heartbeat_state/dept-attention-routing_progress.txt'),
    ('dept-super', 'claude', 'Define cross-vQPU super↔super protocol. Implement feedback modulation to macro rings. Report progress to .heartbeat_state/dept-super_progress.txt'),
    ('dept-cluster', 'claude', 'Run 16-vQPU scaling validation. Produce FPGA pin/BRAM/power projection. Start sensor fusion demo. Report progress to .heartbeat_state/dept-cluster_progress.txt'),
]

os.chdir('/home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module')

for dept, model, prompt in agents:
    session = f'{dept}_agent'
    # Create tmux session with hermes chat (no dimensions to avoid issues)
    cmd = f'hermes chat -q "{prompt}" --worktree'
    try:
        subprocess.run(
            ['tmux', 'new-session', '-d', '-s', session],
            check=True
        )
        subprocess.run(
            ['tmux', 'send-keys', '-t', session, cmd, 'Enter'],
            check=True
        )
        print(f'Created: {session}')
        time.sleep(1)
    except Exception as e:
        print(f'Failed {dept}: {e}')

print('\nAll hermes agent sessions created.')
