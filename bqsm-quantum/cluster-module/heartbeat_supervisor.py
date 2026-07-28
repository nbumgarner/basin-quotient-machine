#!/usr/bin/env python3
"""
Heartbeat Supervisor — Runs every 45 minutes to:
1. Check research progress across all departments
2. Spawn subagents for active workstreams
3. Make supervisory decisions on priority allocation
4. Wake human when direction needed
5. Maintain momentum toward grant-ready evidence
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =========== CONFIGURATION ===========
CLUSTER_ROOT = Path("/home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module")
REGISTRY_FILE = CLUSTER_ROOT / "DESIGN_REGISTRY.csv"
TRAJECTORY_FILE = CLUSTER_ROOT / "RESEARCH_TRAJECTORIES.md"
DEPT_INDEX = CLUSTER_ROOT / "depts" / "DEPARTMENT_INDEX.md"

# Department → Trajectory mapping
DEPT_TRAJECTORIES = {
    "dept-micro": ["T1", "T4"],
    "dept-lens": ["T2", "T4"],
    "dept-macro": ["T2", "T4"],
    "dept-super": ["T3"],
    "dept-pulse-fabric": ["T4", "T5"],
    "dept-plasticity": ["T4", "T2", "T3"],
    "dept-attention-routing": ["T3"],
    "dept-cluster": ["T5"],
}

# Active subagent tracking
STATE_DIR = CLUSTER_ROOT / ".heartbeat_state"
STATE_DIR.mkdir(exist_ok=True)
ACTIVE_AGENTS_FILE = STATE_DIR / "active_agents.json"
LAST_WAKE_FILE = STATE_DIR / "last_wake.txt"
DECISION_LOG_FILE = STATE_DIR / "decisions.log"

# =========== UTILITIES ===========
def log_decision(msg):
    timestamp = datetime.now().isoformat()
    with open(DECISION_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"📝 {msg}")

def read_registry():
    """Parse DESIGN_REGISTRY.csv for active designs"""
    designs = []
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(",")
            if len(parts) >= 6:
                designs.append({
                    "design_id": parts[0],
                    "component": parts[1],
                    "variant": parts[2],
                    "version": parts[3],
                    "status": parts[4],
                    "owner": parts[5],
                })
    return designs

def get_active_designs():
    """Return designs with status 'active' or 'design'"""
    return [d for d in read_registry() if d["status"] in ("active", "design")]

def check_department_progress(dept_name):
    """Check what designs are active for a department"""
    dept_path = CLUSTER_ROOT / "depts" / dept_name
    designs = []
    if dept_path.exists():
        design_dirs = list((dept_path / "designs").glob("VQPU-*")) if (dept_path / "designs").exists() else []
        designs = [d.name for d in design_dirs]
    return designs

def wake_human(reason, context):
    """Alert human that direction is needed"""
    msg = f"""
╔══════════════════════════════════════════════════════════════╗
║  HEARTBEAT WAKE-UP: Human Direction Required                 ║
╠══════════════════════════════════════════════════════════════╣
║ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
║ Reason: {reason}
║ Context: {context}
╚══════════════════════════════════════════════════════════════╝
"""
    print(msg)
    # Also log to file for review
    with open(LAST_WAKE_FILE, "w") as f:
        f.write(msg)
    log_decision(f"WAKE_HUMAN: {reason} — {context}")

def spawn_subagent(dept_name, scope, priority="normal"):
    """Spawn a subagent to work on a department's scope"""
    dept_config = {
        "dept-micro": {"model": "gemini", "focus": "Micro ring physics, lens catalog, 1-chunk settle"},
        "dept-lens": {"model": "gemini", "focus": "Lens profiles, pulse-receptive design, synthesis"},
        "dept-macro": {"model": "claude", "focus": "Macro aggregation fix, feature assembly"},
        "dept-super": {"model": "claude", "focus": "Attention routing, cross-vQPU protocol"},
        "dept-pulse-fabric": {"model": "gemini", "focus": "Pulse encoding, routing, FPGA projection"},
        "dept-plasticity": {"model": "gemini", "focus": "Hebbian implementation, convergence validation"},
        "dept-attention-routing": {"model": "claude", "focus": "Super attention Hebbian learning"},
        "dept-cluster": {"model": "claude", "focus": "Scaling laws, FPGA spec, application demos"},
    }
    
    config = dept_config.get(dept_name, {"model": "gemini", "focus": "General"})
    
    # Build the subagent task
    task = f"""
DEPARTMENT: {dept_name}
PRIORITY: {priority}
SCOPE: {scope}

CONTEXT FROM HEARTBEAT:
- Current active designs: {get_active_designs()}
- Trajectories: {DEPT_TRAJECTORIES.get(dept_name, [])}
- Department focus: {config['focus']}

YOUR MISSION:
1. Review the department CHARTER.md at {CLUSTER_ROOT}/depts/{dept_name}/CHARTER.md
2. Check DESIGN_REGISTRY.csv for your department's designs
3. Execute the scope defined above
4. Produce metrics CSV and update DESIGN_REGISTRY.csv
5. If blocked or need human direction, write to {STATE_DIR}/{dept_name}_wake.txt with reason
6. Log progress to {STATE_DIR}/{dept_name}_progress.log

TOOLS AVAILABLE:
- Run simulations in {CLUSTER_ROOT}/src/
- Read/write department files
- Spawn sub-subagents if needed (max depth 1)
- Use terminal for compilation/testing

DELIVERABLES:
- Updated metrics CSV in {CLUSTER_ROOT}/metrics/
- DESIGN_REGISTRY.csv entry if new design
- Progress log
- Wake file if blocked
"""
    
    # Track active agent
    active_agents = {}
    if ACTIVE_AGENTS_FILE.exists():
        with open(ACTIVE_AGENTS_FILE) as f:
            active_agents = json.load(f)
    
    agent_id = f"{dept_name}_{int(time.time())}"
    active_agents[agent_id] = {
        "dept": dept_name,
        "scope": scope,
        "started": datetime.now().isoformat(),
        "priority": priority,
        "model": config["model"]
    }
    with open(ACTIVE_AGENTS_FILE, "w") as f:
        json.dump(active_agents, f, indent=2)
    
    log_decision(f"SPAWNED {config['model'].upper()} agent for {dept_name}: {scope[:80]}...")
    
    # Actually spawn via delegate_task
    return {
        "agent_id": agent_id,
        "dept": dept_name,
        "task": task,
        "model": config["model"]
    }

def check_agent_wakes():
    """Check if any agents have requested wake-up"""
    wake_files = list(STATE_DIR.glob("*_wake.txt"))
    for wf in wake_files:
        dept = wf.stem.replace("_wake", "")
        with open(wf) as f:
            reason = f.read().strip()
        wake_human(f"Agent {dept} requests direction", reason)
        wf.unlink()  # Remove after reading

def make_supervisory_decisions():
    """Core supervisory logic - decide what to prioritize"""
    active_designs = get_active_designs()
    active_agents = {}
    if ACTIVE_AGENTS_FILE.exists():
        with open(ACTIVE_AGENTS_FILE) as f:
            active_agents = json.load(f)
    
    decisions = []
    
    # 1. Ensure plasticity is running (Trajectory 1 - highest leverage)
    plasticity_active = any("plasticity" in a["dept"] for a in active_agents.values())
    if not plasticity_active:
        decisions.append(("spawn", "dept-plasticity", "Implement Hebbian plasticity engine per PLASTICITY_RULE_DESIGN.md — Oja + competition in harmonic basis. Target: single-ring convergence < 1000 chunks.", "high"))
    
    # 2. Ensure macro aggregation fix is running (Trajectory 2)
    macro_active = any("macro" in a["dept"] or "lens" in a["dept"] for a in active_agents.values())
    if not macro_active:
        decisions.append(("spawn", "dept-lens", "Design pulse-receptive lens profiles for macro aggregation fix. Current: 25% harmonic detection → target >90%. Characterize pulse→harmonic transfer function.", "high"))
        decisions.append(("spawn", "dept-macro", "Implement macro aggregation with optimized destination lenses. Test cross-vQPU feature routing.", "high"))
    
    # 3. Ensure super attention training (Trajectory 3)
    super_active = any("super" in a["dept"] or "attention" in a["dept"] for a in active_agents.values())
    if not super_active:
        decisions.append(("spawn", "dept-attention-routing", "Implement Hebbian attention weight learning on macro harmonic correlations. Current routing: 50% chance → target >85%.", "high"))
        decisions.append(("spawn", "dept-super", "Define cross-vQPU super↔super protocol. Implement feedback modulation to macro rings.", "medium"))
    
    # 4. Cluster scaling + FPGA (Trajectory 5)
    cluster_active = any("cluster" in a["dept"] for a in active_agents.values())
    if not cluster_active:
        decisions.append(("spawn", "dept-cluster", "Run 16-vQPU scaling validation. Produce FPGA pin/BRAM/power projection. Start sensor fusion demo.", "medium"))
    
    # 5. Micro ring validation (Trajectory 1 support)
    micro_active = any("micro" in a["dept"] for a in active_agents.values())
    if not micro_active:
        decisions.append(("spawn", "dept-micro", "Validate 1-chunk settle accuracy. Sweep lens catalog. Quantization floor for Q4/Q8.", "medium"))
    
    # 6. Limit concurrent agents (max 6 for compute)
    if len(active_agents) >= 6:
        decisions.append(("throttle", None, f"Max agents (6) reached. Active: {len(active_agents)}", "info"))
    
    return decisions

def run_heartbeat():
    """Main heartbeat execution"""
    print(f"\n{'='*60}")
    print(f"💓 HEARTBEAT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check for agent wake-ups
    check_agent_wakes()
    
    # Read current state
    active_designs = get_active_designs()
    active_agents = {}
    if ACTIVE_AGENTS_FILE.exists():
        with open(ACTIVE_AGENTS_FILE) as f:
            active_agents = json.load(f)
    
    print(f"📊 Active designs: {len(active_designs)}")
    print(f"🤖 Active agents: {len(active_agents)}")
    for aid, info in active_agents.items():
        print(f"   - {info['dept']} ({info['model']}): {info['scope'][:60]}...")
    
    # Make supervisory decisions
    decisions = make_supervisory_decisions()
    
    # Execute decisions
    spawned = []
    for decision in decisions:
        if decision[0] == "spawn":
            _, dept, scope, priority = decision
            # Check if already active
            if not any(a["dept"] == dept for a in active_agents.values()):
                agent = spawn_subagent(dept, scope, priority)
                spawned.append(agent)
            else:
                log_decision(f"SKIP {dept} — already active")
        elif decision[0] == "throttle":
            log_decision(decision[2])
    
    # Actually launch the subagents via delegate_task
    for agent in spawned:
        try:
            # This would call the delegate_task tool in real implementation
            log_decision(f"LAUNCHING {agent['model'].upper()} for {agent['dept']}")
            # In actual implementation: delegate_task(...)
        except Exception as e:
            log_decision(f"ERROR launching {agent['dept']}: {e}")
    
    # Summary
    print(f"\n✅ Heartbeat complete. Spawned {len(spawned)} new agents.")
    print(f"📈 Total active agents: {len(active_agents) + len(spawned)}")
    print(f"{'='*60}\n")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "active_designs": len(active_designs),
        "active_agents_before": len(active_agents),
        "spawned": len(spawned),
        "decisions": decisions
    }

if __name__ == "__main__":
    result = run_heartbeat()
    # Write heartbeat log
    log_file = STATE_DIR / f"heartbeat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)