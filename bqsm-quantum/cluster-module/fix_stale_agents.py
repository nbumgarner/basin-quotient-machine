#!/usr/bin/env python3
"""Clean up stale agents and re-spawn priority ones."""

import json
import os
import subprocess
import time
from pathlib import Path

HEARTBEAT_DIR = Path(".heartbeat_state")
ACTIVE_FILE = HEARTBEAT_DIR / "active_agents.json"
DECISIONS_LOG = HEARTBEAT_DIR / "decisions.log"

# Priority order for re-spawning
PRIORITY_ORDER = [
    "dept-plasticity",   # Trajectory 1: Plasticity engine
    "dept-macro",        # Trajectory 2: Macro aggregation fix
    "dept-attention-routing",  # Trajectory 3: Super attention routing
    "dept-lens",         # Support for macro
    "dept-super",        # Trajectory 3 continuation
    "dept-micro",        # Trajectory 5: Foundation
    "dept-cluster",      # Trajectory 5: Scaling
]

def log_decision(msg):
    """Append to decisions log."""
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    with open(DECISIONS_LOG, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def check_tmux_sessions():
    """Check which agents are actually running in tmux."""
    result = subprocess.run(
        ["tmux", "list-sessions"],
        capture_output=True,
        text=True
    )
    running = set()
    for line in result.stdout.strip().split("\n"):
        if line:
            session_name = line.split(":")[0]
            running.add(session_name)
    return running

def get_agent_scope(dept):
    """Get the scope for a department from DESIGN_REGISTRY."""
    registry_file = Path("DESIGN_REGISTRY.csv")
    if not registry_file.exists():
        return f"Work on {dept} trajectory"
    
    with open(registry_file) as f:
        for line in f:
            if dept in line:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    return parts[2]  # scope column
    return f"Continue {dept} work"

def spawn_agent(dept, model="claude"):
    """Spawn a tmux agent for the given department."""
    scope = get_agent_scope(dept)
    session_name = f"{dept}_agent"
    
    # Check if already running
    running = check_tmux_sessions()
    if session_name in running:
        log_decision(f"SKIP {dept}: already running in tmux")
        return False
    
    # Build the hermes command
    cmd = [
        "hermes",
        "chat",
        "-q",
        f"Work on the {dept} trajectory. Scope: {scope}. Report progress every 5 minutes.",
        "--worktree",
        "--profile",
        model
    ]
    
    # Create tmux session
    tmux_cmd = [
        "tmux", "new-session", "-d",
        "-s", session_name,
        "-x", "120",
        "-y", "40",
        ";",
        ";".join(cmd)
    ]
    
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", session_name, "-x", "120", "-y", "40"],
            check=True
        )
        time.sleep(2)
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "Enter"],
            check=True
        )
        log_decision(f"SPAWNED {model.upper()} agent for {dept}")
        return True
    except Exception as e:
        log_decision(f"FAILED to spawn {dept}: {e}")
        return False

def main():
    # Clean up stale agents
    if ACTIVE_FILE.exists():
        with open(ACTIVE_FILE) as f:
            agents = json.load(f)
        
        # Clear all stale entries
        running = check_tmux_sessions()
        stale = [k for k in agents.keys() if k.split("_")[0] not in running]
        
        if stale:
            log_decision(f"CLEANUP: {len(stale)} stale agents detected, clearing")
            for key in stale:
                dept = agents[key]["dept"]
                log_decision(f"  - Removed stale {dept}")
            agents.clear()
            with open(ACTIVE_FILE, "w") as f:
                json.dump(agents, f, indent=2)
    
    # Re-spawn priority agents (max 6)
    spawned = 0
    for dept in PRIORITY_ORDER:
        if spawned >= 6:
            break
        # Alternate between claude and gemini
        model = "gemini" if dept in ["dept-plasticity", "dept-lens", "dept-micro"] else "claude"
        if spawn_agent(dept, model):
            spawned += 1
    
    log_decision(f"HEARTBEAT COMPLETE: {spawned} agents spawned")
    print(f"Spawned {spawned} agents for priority trajectories")

if __name__ == "__main__":
    main()
