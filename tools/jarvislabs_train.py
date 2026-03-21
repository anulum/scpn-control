#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — JarvisLabs Remote PPO Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Launch PPO training on JarvisLabs, download results, destroy instance.

Usage:
    python tools/jarvislabs_train.py

Lifecycle:
    1. Create cheapest CPU-capable instance
    2. Wait for SSH ready
    3. Upload repo via scp/rsync
    4. Run training (3 seeds x 500K steps)
    5. Download weights + metrics + benchmark
    6. Destroy instance
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── JarvisLabs setup ──────────────────────────────────────────────


def setup_jarvislabs(token: str):
    """Configure jlclient with token."""
    from jlclient import jarvisclient

    jarvisclient.token = token
    return jarvisclient


def get_balance(jarvisclient):
    from jlclient.jarvisclient import User

    balance = User.get_balance()
    print(f"JarvisLabs balance: {balance}")
    return balance


def list_instances(jarvisclient):
    from jlclient.jarvisclient import User

    instances = User.get_instances()
    print(f"Active instances: {len(instances)}")
    for inst in instances:
        print(f"  - {inst.name} | {inst.status} | {inst.machine_id}")
    return instances


def create_instance(jarvisclient, name: str = "scpn-ppo-train"):
    """Create a cheap GPU instance (smallest available)."""
    from jlclient.jarvisclient import Instance

    print(f"Creating instance '{name}'...")
    instance = Instance.create(
        "GPU",
        gpu_type="A5000",
        num_gpus=1,
        storage=20,
        name=name,
        framework_id=10,  # PyTorch (includes Python)
    )
    print(f"Instance created: machine_id={instance.machine_id}")
    print(f"Status: {instance.status}")
    return instance


def wait_for_ready(instance, timeout: int = 300):
    """Poll until instance is running and SSH is available."""
    from jlclient.jarvisclient import Instance

    print("Waiting for instance to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            # Refresh instance info
            inst = Instance.get_by_id(instance.machine_id)
            if hasattr(inst, "status") and inst.status == "Running":
                ssh_str = getattr(inst, "ssh_str", None)
                if ssh_str:
                    print(f"Instance ready! SSH: {ssh_str}")
                    return inst
        except Exception as e:
            print(f"  Polling... ({e})")
        time.sleep(10)

    raise TimeoutError(f"Instance not ready after {timeout}s")


def run_ssh_command(ssh_str: str, command: str, timeout: int = 1800):
    """Run a command on the remote instance via SSH."""
    # Parse ssh_str: typically "ssh -p PORT user@host"
    parts = ssh_str.strip().split()
    ssh_args = parts[1:]  # skip the 'ssh' prefix

    full_cmd = ["ssh"] + ssh_args + ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", command]
    print(f"[SSH] {command[:100]}...")
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout[-500:])
    if result.returncode != 0:
        print(f"[SSH ERROR] rc={result.returncode}")
        if result.stderr:
            print(result.stderr[-500:])
    return result


def scp_upload(ssh_str: str, local_path: str, remote_path: str):
    """Upload file/dir to remote instance."""
    # Extract port and host from ssh_str
    parts = ssh_str.strip().split()
    port = "22"
    user_host = parts[-1]
    for i, p in enumerate(parts):
        if p == "-p" and i + 1 < len(parts):
            port = parts[i + 1]

    cmd = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no", "-r", local_path, f"{user_host}:{remote_path}"]
    print(f"[SCP] {local_path} -> {remote_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[SCP ERROR] {result.stderr[:300]}")
    return result


def scp_download(ssh_str: str, remote_path: str, local_path: str):
    """Download file from remote instance."""
    parts = ssh_str.strip().split()
    port = "22"
    user_host = parts[-1]
    for i, p in enumerate(parts):
        if p == "-p" and i + 1 < len(parts):
            port = parts[i + 1]

    cmd = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no", "-r", f"{user_host}:{remote_path}", local_path]
    print(f"[SCP] {remote_path} -> {local_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[SCP ERROR] {result.stderr[:300]}")
    return result


def destroy_instance(instance):
    """Destroy the instance to stop billing."""
    print(f"Destroying instance {instance.machine_id}...")
    try:
        instance.destroy()
        print("Instance destroyed.")
    except Exception as e:
        print(f"WARNING: destroy failed: {e}")
        print("MANUAL ACTION REQUIRED: destroy instance via jarvislabs.ai dashboard!")


def main():
    token = os.environ.get("JARVISLABS_TOKEN", "")
    if not token:
        print("Set JARVISLABS_TOKEN environment variable")
        sys.exit(1)

    jc = setup_jarvislabs(token)
    get_balance(jc)
    list_instances(jc)

    instance = None
    try:
        # 1. Create instance
        instance = create_instance(jc)
        instance = wait_for_ready(instance)
        ssh = instance.ssh_str

        # 2. Setup environment on remote
        setup_cmds = [
            "pip install stable-baselines3 gymnasium numpy scipy click",
            "git clone https://github.com/anulum/scpn-control.git || true",
            "cd scpn-control && pip install -e '.[rl,dev]'",
        ]
        for cmd in setup_cmds:
            run_ssh_command(ssh, cmd, timeout=600)

        # 3. Upload local changes (reward shaping, training script)
        # Upload the modified files that aren't on GitHub yet
        files_to_upload = [
            "src/scpn_control/control/gym_tokamak_env.py",
            "tools/train_rl_tokamak.py",
            "tools/train_rl_upcloud.sh",
            "benchmarks/rl_vs_classical.py",
        ]
        for f in files_to_upload:
            local = str(REPO_ROOT / f)
            remote = f"scpn-control/{f}"
            scp_upload(ssh, local, remote)

        # 4. Run training
        train_cmd = "cd scpn-control && export CUDA_VISIBLE_DEVICES='' && bash tools/train_rl_upcloud.sh 500000 50"
        result = run_ssh_command(ssh, train_cmd, timeout=3600)

        # 5. Download results
        downloads = [
            ("scpn-control/weights/ppo_tokamak.zip", str(REPO_ROOT / "weights" / "ppo_tokamak.zip")),
            ("scpn-control/weights/ppo_tokamak.metrics.json", str(REPO_ROOT / "weights" / "ppo_tokamak.metrics.json")),
            ("scpn-control/benchmarks/rl_vs_classical.json", str(REPO_ROOT / "benchmarks" / "rl_vs_classical.json")),
        ]
        # Also download per-seed files
        for seed in [42, 123, 456]:
            downloads.append(
                (
                    f"scpn-control/weights/ppo_tokamak_seed{seed}.zip",
                    str(REPO_ROOT / "weights" / f"ppo_tokamak_seed{seed}.zip"),
                )
            )
            downloads.append(
                (
                    f"scpn-control/weights/ppo_tokamak_seed{seed}.metrics.json",
                    str(REPO_ROOT / "weights" / f"ppo_tokamak_seed{seed}.metrics.json"),
                )
            )

        for remote_path, local_path in downloads:
            scp_download(ssh, remote_path, local_path)

        # 6. Verify downloads
        print("\n=== Download verification ===")
        for _, local_path in downloads:
            p = Path(local_path)
            if p.exists():
                print(f"  OK: {p.name} ({p.stat().st_size} bytes)")
            else:
                print(f"  MISSING: {p.name}")

        # 7. Print results
        bench_path = REPO_ROOT / "benchmarks" / "rl_vs_classical.json"
        if bench_path.exists():
            print("\n=== Benchmark Results ===")
            results = json.loads(bench_path.read_text())
            for name in ["mpc", "pid", "ppo"]:
                d = results[name]
                print(
                    f"  {name.upper():>4}: reward={d['mean_reward']:8.1f} +/- {d['std_reward']:5.1f}  "
                    f"disruption={d['disruption_rate'] * 100:.0f}%"
                )

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 8. ALWAYS destroy instance
        if instance is not None:
            destroy_instance(instance)
            print("\nInstance destroyed. Billing stopped.")

        # Final balance check
        try:
            get_balance(jc)
        except Exception:
            pass


if __name__ == "__main__":
    main()
