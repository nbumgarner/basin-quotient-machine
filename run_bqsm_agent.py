#!/usr/bin/env python3
"""
run_bqsm_agent.py

A tiny wrapper that:
  • Starts the local BQSM server (if not already running)
  • Provides a chat/agent REPL that talks to the server via the OpenAI‑compatible API
  • Maintains an MCP folder with goal/priority files for contextual hierarchy
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

# ------------------- CONFIG -------------------
MODEL_PATH = "./gemma-2b-ternary.bqsm"
SERVER_BIN = "./bqsm_server"
PORT = 8082                     # change if you prefer another port
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MCP_DIR = pathlib.Path("./mcp")
MCP_FILES = ["LTgoals.txt", "STgoals.txt", "Pri0.txt", "Pri1.txt", "Pri2.txt"]

# ------------------- HELPERS -------------------
def ensure_mcp_dir():
    """Create MCP directory and default files if they don't exist."""
    MCP_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MCP_FILES:
        fpath = MCP_DIR / fname
        if not fpath.exists():
            fpath.write_text(f"# {fname}\n\n", encoding="utf-8")

def read_mcp():
    """Read all MCP files and return a dict of their contents."""
    data = {}
    for fname in MCP_FILES:
        fpath = MCP_DIR / fname
        if fpath.exists():
            data[fname] = fpath.read_text(encoding="utf-8").strip()
        else:
            data[fname] = ""
    return data

def write_mcp_file(fname, content):
    """Write content to an MCP file."""
    fpath = MCP_DIR / fname
    fpath.write_text(content + "\n", encoding="utf-8")

def server_running():
    """Check if the BQSM server is responding on /health."""
    try:
        with urllib.request.urlopen(f"{BASE_URL.replace('/v1','')}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def start_server():
    """Launch the BQSM server as a background process."""
    if server_running():
        print(f"✓ Server already running on port {PORT}")
        return None
    print(f"Starting BQSM server on port {PORT}...")
    # Use Popen with same environment; redirect output to a log file
    log_file = open("bqsm_server.log", "a")
    proc = subprocess.Popen(
        [SERVER_BIN, "--model", MODEL_PATH, "--port", str(PORT), "--layers", "2"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Wait for server to become ready
    for _ in range(30):
        time.sleep(0.5)
        if server_running():
            print("✓ Server started and healthy")
            return proc
    print("✗ Server failed to start within timeout")
    proc.terminate()
    return None

def chat_completion(messages, temperature=0.7, max_tokens=256):
    """Send a chat completion request to the local BQSM server."""
    payload = {
        "model": "bqsm",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except Exception as e:
        return {"error": str(e)}

def build_system_prompt(mcp_data):
    """Construct a system prompt that includes the MCP hierarchy."""
    lines = [
        "You are an AI assistant with a structured priority hierarchy for context.",
        "The following files define your goals and priorities (higher priority = more important):",
        "",
    ]
    # Order: Pri0, Pri1, Pri2, STgoals, LTgoals (highest to lowest priority)
    priority_order = ["Pri0.txt", "Pri1.txt", "Pri2.txt", "STgoals.txt", "LTgoals.txt"]
    for fname in priority_order:
        content = mcp_data.get(fname, "").strip()
        if content:
            lines.append(f"=== {fname} ===")
            lines.append(content)
            lines.append("")
    lines.append("Use this hierarchy to guide your responses and decision‑making.")
    return "\n".join(lines)

def repl_loop(server_proc):
    """Main interactive loop."""
    print("\n=== BQSM Agent REPL ===")
    print("Commands:")
    print("  /show [file]      – display MCP file(s)")
    print("  /edit <file>      – edit an MCP file (opens $EDITOR or prompts)")
    print("  /reset <file>     – clear an MCP file")
    print("  /mcp              – show all MCP files")
    print("  /quit             – exit")
    print("  Anything else is sent as a chat message.")
    print()

    conversation = []  # list of {"role": "...", "content": "..."}

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # ----- Command handling -----
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                break

            elif cmd == "/show":
                mcp = read_mcp_files()
                if arg:
                    fname = arg if arg.endswith(".txt") else arg + ".txt"
                    if fname in mcp:
                        print(f"\n--- {fname} ---")
                        print(mcp[fname] or "(empty)")
                    else:
                        print(f"Unknown file: {fname}")
                else:
                    for fname, content in mcp.items():
                        print(f"\n--- {fname} ---")
                        print(content or "(empty)")
                continue

            elif cmd == "/mcp":
                mcp = read_mcp_files()
                for fname, content in mcp.items():
                    print(f"\n--- {fname} ---")
                    print(content or "(empty)")
                continue

            elif cmd == "/reset":
                if arg:
                    fname = arg if arg.endswith(".txt") else arg + ".txt"
                    if fname in MCP_FILES:
                        write_mcp_file(fname, "")
                        print(f"Cleared {fname}")
                    else:
                        print(f"Invalid file: {fname}")
                else:
                    print("Usage: /reset <file>")
                continue

            elif cmd == "/edit":
                if not arg:
                    print("Usage: /edit <file>")
                    continue
                fname = arg if arg.endswith(".txt") else arg + ".txt"
                if fname not in MCP_FILES:
                    print(f"Invalid file: {fname}")
                    continue
                # Simple inline edit: read current, prompt for new content
                current = read_mcp_files().get(fname, "")
                print(f"Current content of {fname}:")
                print(current or "(empty)")
                print("Enter new content (end with a line containing only 'END'):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                new_content = "\n".join(lines)
                write_mcp_file(fname, new_content)
                print(f"Updated {fname}")
                continue

            else:
                print(f"Unknown command: {cmd}")
                continue

        # ----- Regular chat message -----
        # Refresh MCP data each turn so edits take effect immediately
        mcp_data = read_mcp()
        system_prompt = build_system_prompt(mcp_data)

        # Build messages for the API
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_input})

        print("... thinking ...")
        resp = chat_completion(messages)
        if "error" in resp:
            print(f"Error: {resp['error']}")
            continue
        try:
            assistant_msg = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            assistant_msg = "(no response)"

        print(f"\n{assistant_msg}\n")

        # Update conversation history
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": assistant_msg})
        # Keep history bounded (last 10 exchanges)
        if len(conversation) > 20:
            conversation = conversation[-20:]

    # Cleanup
    if server_proc:
        print("Stopping server...")
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    ensure_mcp_dir()
    proc = start_server()
    try:
        repl_loop(proc)
    finally:
        if proc:
            proc.terminate()