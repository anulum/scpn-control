#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JarvisLabs training workflow
"""Orchestrate an explicitly authorised JarvisLabs PPO training campaign.

Importing performs no cloud action. Running main requires JARVISLABS_TOKEN,
optional JLClient, existing upload sources/parents, fresh local artifacts and
recorded benchmark custody before any SDK request. The recipe requests one
A5000 GPU but forces CPU training. The shell now forwards seeds42/123/456; retained legacy seed-labelled files
predate that correction and do not prove independent experiments.

Transport failures stop the recipe. Cleanup is attempted for every returned
instance handle; explicit SDK acknowledgement does not verify stopped billing.
Success requires all fresh downloads and the producer's uppercase metric keys,
not independent physics validity or weight authentication. Cloud execution,
provider availability and successful training require separate qualification.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_control.benchmark_records import CAMPAIGN_ENV, require_recorded_campaign
from tools.jarvislabs_client import (
    create_instance,
    destroy_instance,
    get_balance,
    list_instances,
    setup_jarvislabs,
    wait_for_ready,
)
from tools.jarvislabs_transport import run_ssh_command, scp_download, scp_upload
from tools.rl_training_results import load_benchmark

__all__ = [
    "create_instance",
    "destroy_instance",
    "get_balance",
    "list_instances",
    "main",
    "run_ssh_command",
    "scp_download",
    "scp_upload",
    "setup_jarvislabs",
    "wait_for_ready",
]

UPLOAD_FILES = (
    "src/scpn_control/control/gym_tokamak_env.py",
    "tools/train_rl_tokamak.py",
    "tools/train_rl_upcloud.sh",
    "benchmarks/rl_vs_classical.py",
    "tools/rl_training_config.py",
    "tools/rl_tokamak_evaluation.py",
    "tools/rl_training_results.py",
)
DOWNLOAD_FILES = (
    "weights/ppo_tokamak.zip",
    "weights/ppo_tokamak.metrics.json",
    "benchmarks/rl_vs_classical.json",
    *(f"weights/ppo_tokamak_seed{seed}{suffix}" for seed in (42, 123, 456) for suffix in (".zip", ".metrics.json")),
)


def _download_paths(output_dir: Path | None) -> list[Path]:
    """Resolve legacy defaults or nine flat files in an existing candidate directory."""
    return [REPO_ROOT / name if output_dir is None else output_dir / Path(name).name for name in DOWNLOAD_FILES]


def _check_artifact_paths(output_dir: Path | None = None) -> str:
    """Require seven sources, fresh outputs and canonical remote-report custody."""
    sources = {str((REPO_ROOT / name).resolve()) for name in UPLOAD_FILES}
    if not all((REPO_ROOT / name).is_file() for name in UPLOAD_FILES):
        raise FileNotFoundError("A required training upload source is unavailable")
    outputs: set[str] = set()
    for target in _download_paths(output_dir):
        resolved = str(target.resolve())
        if resolved in sources or resolved in outputs or target.exists() or target.is_symlink():
            raise FileExistsError("Training download targets must be fresh and distinct")
        if not target.parent.is_dir():
            raise NotADirectoryError("A training download parent is unavailable")
        outputs.add(resolved)
    campaign = require_recorded_campaign(REPO_ROOT / "benchmarks/rl_vs_classical.json", repository_root=REPO_ROOT)
    if campaign is None:
        raise RuntimeError("Training requires recorded benchmark custody")
    return campaign


def _report_downloads(output_dir: Path | None = None) -> None:
    """Require nine nonempty fresh files and the actual full benchmark metric schema."""
    paths = _download_paths(output_dir)
    for target in paths:
        if not target.is_file() or target.is_symlink() or target.stat().st_size == 0:
            raise RuntimeError("A required training download is unavailable")
    for name, metrics in load_benchmark(paths[2]).items():
        print(
            f"{name}: reward={metrics['mean_reward']:.1f} +/- {metrics['std_reward']:.1f}, disruption={metrics['disruption_rate'] * 100:.0f}%"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the guarded cloud recipe and return zero only for verified file delivery.

    Missing credentials return 1 before SDK import. Upload/output/campaign checks
    precede all provider requests. Setup, create, readiness, SSH/SCP, artifact or
    metric failures print one fixed sentence and return 1. A returned handle is
    retained through readiness failure and always offered to destroy_instance;
    unresolved cleanup also returns 1. Ctrl-C still executes cleanup and then
    propagates. Credentials, provider errors and remote output are not printed.

    This function can provision billable resources and install/train remotely.
    --output-dir selects nine flat candidate downloads in an existing directory;
    no flag retains legacy local paths. Remote candidates use a fresh campaign
    directory. A successful return does not establish independent model acceptance,
    remote checkout reproducibility or stopped billing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="existing fresh candidate directory for all nine downloads")
    args = parser.parse_args([] if argv is None else argv)
    token = os.environ.get("JARVISLABS_TOKEN", "")
    if not token.strip():
        print("Set JARVISLABS_TOKEN environment variable")
        return 1
    instance: object | None = None
    result = 1
    try:
        campaign = _check_artifact_paths(args.output_dir)
        client = setup_jarvislabs(token)
        get_balance(client)
        list_instances(client)
        instance = create_instance(client)
        ready = wait_for_ready(instance)
        ssh: object = getattr(ready, "ssh_str", None)
        if not isinstance(ssh, str):
            raise RuntimeError("JarvisLabs did not return an SSH endpoint")
        setup_commands = (
            "pip install stable-baselines3 gymnasium numpy scipy click",
            "test -d scpn-control/.git || git clone https://github.com/anulum/scpn-control.git",
            "cd scpn-control && pip install -e '.[rl,dev]'",
        )
        for command in setup_commands:
            run_ssh_command(ssh, command, timeout=600)
        for name in UPLOAD_FILES:
            scp_upload(ssh, str(REPO_ROOT / name), f"scpn-control/{name}")
        remote_directory = f"artifacts/rl/{campaign}"
        train_command = (
            f"cd scpn-control && export {CAMPAIGN_ENV}={shlex.quote(campaign)} && "
            f"export CUDA_VISIBLE_DEVICES='' && bash tools/train_rl_upcloud.sh --output-dir {shlex.quote(remote_directory)} 500000 50"
        )
        run_ssh_command(ssh, train_command, timeout=3600)
        for name, target in zip(DOWNLOAD_FILES, _download_paths(args.output_dir), strict=True):
            scp_download(ssh, f"scpn-control/{remote_directory}/{Path(name).name}", str(target))
        _report_downloads(args.output_dir)
        result = 0
    except Exception:
        print("JarvisLabs training workflow failed; results are not accepted.")
    finally:
        if instance is not None and not destroy_instance(instance):
            result = 1
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
