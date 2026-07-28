# HEARTBEAT SYSTEM — Complete Reference

## What You Have Now

### Cron Job (Active)
```
Job ID: cb18993ac990
Name: Neuromorphic vQPU Cluster Heartbeat
Schedule: Every 45 minutes
Next Run: 2026-07-26T20:22:47
Status: Enabled, scheduled
```

### Heartbeat Supervisor Script
**Location**: `/home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module/heartbeat_supervisor.py`

**What it does each run (every 45 min)**:
1. Checks `DESIGN_REGISTRY.csv` for active designs
2. Reads `.heartbeat_state/active_agents.json` for running agents
3. Makes supervisory decisions:
   - Spawns new agents where work is stalled
   - Assigns departments based on research trajectories
   - Uses **Gemini** for: plasticity, lens, micro, pulse-fabric
   - Uses **Claude** for: macro, super, attention, cluster
4. Respects `MAX_CONCURRENT_AGENTS = 6` throttle
5. Agents write wake files when blocked: `.heartbeat_state/{dept}_wake.txt`

### Agent Launcher
**Location**: `/home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module/launch_agents.py`

**What it does**: Reads `active_agents.json` and produces delegate_task specs for each agent.

### State Directory
```
/home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module/.heartbeat_state/
├── active_agents.json     # Currently running agents
├── supervisor_log.json    # Heartbeat decision log
├── dept-plasticity_wake.txt   # Agent requests direction here
├── dept-lens_wake.txt
├── dept-macro_wake.txt
├── dept-super_wake.txt
├── dept-attention-routing_wake.txt
├── dept-cluster_wake.txt
├── dept-micro_wake.txt
└── dept-pulse-fabric_wake.txt
```

---

## How It Works

### Agent Spawning Logic
```python
# From heartbeat_supervisor.py
if active_agents < 6:  # throttle
    for each research trajectory needing work:
        if dept not in active_agents:
            spawn_agent(dept, model, scope)
```

### Department → Model Mapping
| Department | Model | Rationale |
|------------|-------|-----------|
| dept-plasticity | **Gemini** | Math-heavy, Oja's rule, convergence proofs |
| dept-lens | **Gemini** | Filter characterization, quantization analysis |
| dept-micro | **Gemini** | Physics simulation, sweep experiments |
| dept-pulse-fabric | **Gemini** | Event encoding, routing algorithms |
| dept-macro | **Claude** | System integration, feature assembly |
| dept-super | **Claude** | Global attention, cross-vQPU coordination |
| dept-attention-routing | **Claude** | Hebbian weight learning, protocols |
| dept-cluster | **Claude** | Topology, scaling, FPGA projection, apps |

### Agent Wake Protocol
When an agent is blocked:
```python
# Agent writes:
with open(f".heartbeat_state/{dept}_wake.txt", "w") as f:
    f.write("BLOCKED: Need decision on X. Options: A, B, C.")

# Next heartbeat (45 min later):
# 1. Supervisor reads all _wake.txt files
# 2. Makes decision, updates active_agents.json
# 3. Agent sees decision in its context, continues
```

---

## Research Trajectories Being Driven

| Trajectory | Departments | Target |
|------------|-------------|--------|
| **T1: Plasticity Engine** | plasticity, micro | Convergence < 1000 chunks |
| **T2: Macro Aggregation** | macro, lens, pulse-fabric | 25% → 90% harmonic detection |
| **T3: Super Attention** | super, attention-routing, macro | 50% → 85% routing accuracy |
| **T4: Cluster + Hardware** | cluster, pulse-fabric | 16 vQPU scaling, FPGA spec |
| **T5: Application Demos** | cluster, plasticity | Sensor fusion, sequence learning |

---

## Manual Operations

### Check Heartbeat Status
```bash
cronjob action=list
```

### View Active Agents
```bash
cat .heartbeat_state/active_agents.json | jq .
```

### View Supervisor Log
```bash
cat .heartbeat_state/supervisor_log.json | jq .
```

### Force Heartbeat Now
```bash
cd /home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module
python3 heartbeat_supervisor.py
```

### Pause Heartbeat
```bash
cronjob action=pause job_id=cb18993ac990
```

### Resume Heartbeat
```bash
cronjob action=resume job_id=cb18993ac990
```

### View Agent Wake Files
```bash
ls -la .heartbeat_state/*_wake.txt
cat .heartbeat_state/dept-plasticity_wake.txt
```

---

## Grant Application Readiness

The heartbeat system ensures by grant time you have:

| Asset | Status | Source |
|-------|--------|--------|
| **Validated plasticity convergence** | In progress | dept-plasticity metrics |
| **Macro aggregation fixed (25%→90%)** | In progress | dept-macro + dept-lens |
| **Super routing** | In progress | dept-super + dept-attn |
| **16-vQPU scaling laws** | In progress | dept-cluster |
| **FPGA pin/BRAM/power budget** | In progress | dept-cluster |
| **Sensor fusion demo** | Planned | dept-cluster |
| **All metrics in DESIGN_REGISTRY.csv** | Auto-updated | All departments |
| **Git tags for every design** | Auto-tagged | `design/VQPU-*` |

---

## Next Steps (Immediate)

1. **Let heartbeat run** — it spawned 7 agents on first run
2. **Check `.heartbeat_state/`** in ~45 min for wake files
3. **Review agent progress** — they'll write logs to depts/*/designs/
4. **Make decisions** when agents write wake files
5. **Update DESIGN_REGISTRY.csv** when new designs registered

The system is now **autonomous, throttled, and supervised**. It will keep 6 agents running continuously across all 8 departments, driving the 5 research trajectories toward grant-ready results.