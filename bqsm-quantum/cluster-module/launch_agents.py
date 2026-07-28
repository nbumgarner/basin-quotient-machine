#!/usr/bin/env python3
"""
Launch actual subagents via delegate_task for the spawned agents.
This is called by the heartbeat supervisor after making decisions.
"""

import json
from pathlib import Path
from datetime import datetime

CLUSTER_ROOT = Path("/home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module")
STATE_DIR = CLUSTER_ROOT / ".heartbeat_state"
ACTIVE_AGENTS_FILE = STATE_DIR / "active_agents.json"

DEPT_CONFIG = {
    "dept-plasticity": {
        "model": "gemini",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-plasticity" / "CHARTER.md"
    },
    "dept-lens": {
        "model": "gemini", 
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-lens" / "CHARTER.md"
    },
    "dept-macro": {
        "model": "claude",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-macro" / "CHARTER.md"
    },
    "dept-super": {
        "model": "claude",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-super" / "CHARTER.md"
    },
    "dept-pulse-fabric": {
        "model": "gemini",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-pulse-fabric" / "CHARTER.md"
    },
    "dept-attention-routing": {
        "model": "claude",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-attention-routing" / "CHARTER.md"
    },
    "dept-cluster": {
        "model": "claude",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-cluster" / "CHARTER.md"
    },
    "dept-micro": {
        "model": "gemini",
        "toolsets": ["terminal", "file", "skills"],
        "charter": CLUSTER_ROOT / "depts" / "dept-micro" / "CHARTER.md"
    },
}

def load_active_agents():
    if ACTIVE_AGENTS_FILE.exists():
        with open(ACTIVE_AGENTS_FILE) as f:
            return json.load(f)
    return {}

def save_active_agents(agents):
    with open(ACTIVE_AGENTS_FILE, "w") as f:
        json.dump(agents, f, indent=2)

def launch_agent(agent_id, agent_info):
    """Launch a subagent via delegate_task"""
    dept = agent_info["dept"]
    config = DEPT_CONFIG.get(dept, {})
    
    if not config:
        print(f"❌ No config for {dept}")
        return False
    
    # Read charter for context
    charter_text = ""
    if config["charter"].exists():
        with open(config["charter"]) as f:
            charter_text = f.read()
    
    # Build the task
    task = f"""
DEPARTMENT: {dept}
AGENT ID: {agent_id}
MODEL: {config['model'].upper()}

CHARTER:
{charter_text}

YOUR MISSION:
{agent_info['scope']}

CONTEXT:
- Cluster module: {CLUSTER_ROOT}
- Simulator: {CLUSTER_ROOT}/src/cluster_validate.c
- Design registry: {CLUSTER_ROOT}/DESIGN_REGISTRY.csv
- State dir: {STATE_DIR}

REQUIREMENTS:
1. Work in {CLUSTER_ROOT}/depts/{dept}/designs/
2. Produce metrics CSV in {CLUSTER_ROOT}/metrics/
3. Update DESIGN_REGISTRY.csv if new design created
4. Log progress to {STATE_DIR}/{dept}_progress.log
5. If BLOCKED or need human direction: write reason to {STATE_DIR}/{dept}_wake.txt
6. When complete, remove yourself from active_agents.json

TOOLS: terminal, file, skills (hermes-agent for subagents if needed)
WORKDIR: {CLUSTER_ROOT}
"""
    
    # This would be called via the delegate_task tool
    # For now, just return the task spec
    return {
        "role": "leaf",
        "goal": task,
        "toolsets": config["toolsets"],
        "context": f"Department: {dept}, Model: {config['model']}"
    }

if __name__ == "__main__":
    agents = load_active_agents()
    print(f"📋 Launching {len(agents)} agents...")
    
    for agent_id, info in agents.items():
        if "launched" not in info:
            spec = launch_agent(agent_id, info)
            print(f"🚀 Would launch {agent_id} ({info['dept']}) with {info['model']}")
            info["launched"] = True
            info["launched_at"] = datetime.now().isoformat()
    
    save_active_agents(agents)
    print("✅ Agent specs saved. In real implementation, call delegate_task for each.")