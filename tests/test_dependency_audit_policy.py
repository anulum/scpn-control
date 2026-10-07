# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blocking dependency lock audit workflow contracts.

"""Bind every committed dependency lock to an unfiltered required CI audit."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from tools.ci_workflow_inventory import REPOSITORY_ROOT, load_ci_workflow_policy, workflow_path_for_job


def _workflow(path: Path) -> dict[str, Any]:
    """Read a physical workflow through the maintained distributed inventory."""
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def test_audit_jobs_are_unconditional_and_reach_the_required_gate() -> None:
    """Keep all audit failures inside the required security category result."""
    policy = load_ci_workflow_policy()
    category = next(item for item in policy["categories"] if item["id"] == "security-supply-chain")
    coordinator = _workflow(REPOSITORY_ROOT / policy["coordinator"])
    gate = coordinator["jobs"][policy["required_gate"]]
    caller = coordinator["jobs"][category["id"]]
    assert category["id"] in gate["needs"]
    assert gate["if"] == "always()"
    assert caller == {"uses": "./" + category["workflow"]}
    workflow = _workflow(REPOSITORY_ROOT / category["workflow"])
    for name in ("python-audit", "studio-web-audit", "rust-audit"):
        assert name in category["jobs"] and name in policy["job_order"]
        job = workflow["jobs"][name]
        assert "if" not in job and "continue-on-error" not in job
        assert "shell" not in job.get("defaults", {}).get("run", {})
        for step in job["steps"]:
            assert "if" not in step and "continue-on-error" not in step and "shell" not in step


def test_python_audit_pins_universal_lock_export_and_strict_scan() -> None:
    """Audit all optional, Python-version and platform pins without re-resolution."""
    job = _workflow(workflow_path_for_job("python-audit"))["jobs"]["python-audit"]
    commands = {step["name"]: step["run"] for step in job["steps"] if "run" in step}
    assert commands["Run pip audit"] == "python -m pip_audit"
    assert commands["Audit every locked Python package version"] == (
        "set -euo pipefail\n"
        'audit_dir="$(mktemp -d "${RUNNER_TEMP}/python-lock-audit.XXXXXX")"\n'
        'python tools/export_dependency_audit_lock.py --output "${audit_dir}/pylock.toml"\n'
        'python -m pip_audit --locked --strict "${audit_dir}"\n'
    )
    assert "pip install --require-hashes -r requirements/ci-audit.txt" in commands["Install project and audit tools"]


def test_frontend_audit_uses_matching_pinned_tools_and_the_entire_lock() -> None:
    """Retain a lock-only pnpm audit without severity, dependency or error filters."""
    audit = _workflow(workflow_path_for_job("studio-web-audit"))["jobs"]["studio-web-audit"]
    product = _workflow(workflow_path_for_job("studio-web"))["jobs"]["studio-web"]
    assert audit["defaults"] == {"run": {"working-directory": "studio-web"}}
    commands = {step["name"]: step["run"] for step in audit["steps"] if "run" in step}
    assert commands == {
        "Enable pnpm": "corepack enable && corepack prepare pnpm@11.25.0 --activate",
        "Audit the committed frontend lock": "pnpm audit",
    }
    audit_node = next(step for step in audit["steps"] if step.get("uses", "").startswith("actions/setup-node@"))
    product_node = next(step for step in product["steps"] if step.get("uses", "").startswith("actions/setup-node@"))
    assert audit_node == product_node


def test_rust_audit_checks_both_committed_locks_without_suppressions() -> None:
    """Check the fuzz lock as well as the workspace lock using the pinned auditor."""
    job = _workflow(workflow_path_for_job("rust-audit"))["jobs"]["rust-audit"]
    assert job["defaults"] == {"run": {"working-directory": "scpn-control-rs"}}
    commands = {step["name"]: step["run"] for step in job["steps"] if "run" in step}
    assert commands == {
        "Install cargo-audit": "cargo install --locked cargo-audit --version 0.22.2",
        "Audit": "cargo audit",
        "Audit the committed fuzz lock": "cargo audit --file fuzz/Cargo.lock",
    }


def test_committed_lock_inventory_requires_audit_ownership_for_every_lock() -> None:
    """Refuse a newly committed ecosystem lock until its audit ownership is added."""
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPOSITORY_ROOT).decode().split("\0")
    names = {"uv.lock", "Cargo.lock", "pnpm-lock.yaml", "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock"}
    locks = {
        path
        for path in tracked
        if Path(path).name in names
        or (path.startswith("requirements/") and path.endswith(".txt"))
        or (Path(path).name.startswith("pylock") and path.endswith(".toml"))
    }
    python_locks = {
        "uv.lock",
        *[p.relative_to(REPOSITORY_ROOT).as_posix() for p in (REPOSITORY_ROOT / "requirements").glob("ci-*.txt")],
    }
    assert locks <= python_locks | {
        "scpn-control-rs/Cargo.lock",
        "scpn-control-rs/fuzz/Cargo.lock",
        "studio-web/pnpm-lock.yaml",
    }
