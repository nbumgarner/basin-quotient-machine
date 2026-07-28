#!/usr/bin/env python3
"""Send work prompts to agent sessions."""

import subprocess
import time

# Department prompts
prompts = {
    'dept-plasticity_agent': """You are working on the Hebbian plasticity engine for the BQSM cluster module.

TASK: Implement Oja's rule + competition in harmonic basis for single-ring convergence.

GOAL: Achieve convergence in < 1000 chunks for plasticity learning.

APPROACH:
1. Read PLASTICITY_RULE_DESIGN.md for the full specification
2. Implement Oja's rule in the harmonic basis (not raw spike space)
3. Add lateral competition between adjacent harmonic bands
4. Test convergence on a single vQPU ring
5. Report: convergence rate, stability metrics, any issues

Use the cluster-module C codebase. Write results to .heartbeat_state/{dept}_progress.txt""",

    'dept-lens_agent': """You are working on pulse-receptive lens profiles for macro aggregation.

TASK: Design lens profiles that maximize harmonic detection.

CURRENT: ~25% harmonic detection rate
TARGET: >90% harmonic detection rate

APPROACH:
1. Read LENS_DESIGN.md and MACRO_DESIGN.md
2. Characterize the pulse→harmonic transfer function
3. Design lens profiles optimized for pulse reception
4. Test with the existing simulation framework
5. Report: detection rate improvements, optimal profile parameters

Use the cluster-module C codebase. Write results to .heartbeat_state/{dept}_progress.txt""",

    'dept-macro_agent': """You are working on macro aggregation with optimized destination lenses.

TASK: Implement macro aggregation that routes features cross-vQPU.

APPROACH:
1. Read MACRO_DESIGN.md for aggregation protocol
2. Implement destination lens optimization
3. Test cross-vQPU feature routing
4. Measure aggregation fidelity vs. ring count
5. Report: aggregation accuracy, latency, bandwidth utilization

Use the cluster-module C codebase. Write results to .heartbeat_state/{dept}_progress.txt""",

    'dept-attention-routing_agent': """You are working on Hebbian attention weight learning.

TASK: Implement attention routing on macro harmonic correlations.

CURRENT: 50% routing accuracy
TARGET: >85% routing accuracy

APPROACH:
1. Read ATTENTION_ROUTING_DESIGN.md
2. Implement Hebbian weight updates on macro correlations
3. Add competition for attention (winner-take-all)
4. Test routing accuracy on synthetic feature sets
5. Report: accuracy vs. training steps, saturation behavior

Use the cluster-module C codebase. Write results to .heartbeat_state/{dept}_progress.txt""",

    'dept-super_agent': """You are working on cross-vQPU super↔super protocol.

TASK: Define and implement feedback modulation between super-rings.

APPROACH:
1. Read SUPER_RING_DESIGN.md
2. Define the super↔super handshake protocol
3. Implement feedback modulation to macro rings
4. Test cross-vQPU coordination
5. Report: protocol latency, coordination fidelity, scalability

Use the cluster-module C codebase. Write results to .heartbeat_state/{dept}_progress.txt""",

    'dept-cluster_agent': """You are working on 16-vQPU scaling validation.

TASK: Run scaling tests and produce FPGA projections.

APPROACH:
1. Read CLUSTER_DESIGN.md
2. Run 16-vQPU scaling validation (or extrapolate from smaller tests)
3. Produce FPGA pin count, BRAM usage, power projection
4. Start sensor fusion demo setup
5. Report: scaling curves, resource estimates, demo status

Use the cluster-module C codebase. Write results to .heartbeat_state/{dept}_progress.txt""",
}

# Send prompts to each session
for session, prompt in prompts.items():
    try:
        # Send the prompt
        subprocess.run(['tmux', 'send-keys', '-t', session, prompt, 'Enter'], check=True)
        print(f'Sent prompt to {session}')
        time.sleep(1)  # Brief pause between sends
    except Exception as e:
        print(f'Failed to send to {session}: {e}')

print('\nAll prompts sent. Agents should be working now.')
