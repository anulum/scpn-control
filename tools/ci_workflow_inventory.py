# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Distributed CI workflow inventory.

"""Read the distributed CI graph through its versioned ownership policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_POLICY = REPOSITORY_ROOT / "tools/ci_workflow_policy.json"


class WorkflowCategory(TypedDict):
    """One reusable CI workflow and its exclusive ownership surface."""

    id: str
    workflow: str
    caller_needs: list[str]
    secrets: list[str]
    jobs: list[str]


class WorkflowLimits(TypedDict):
    """Repository-local workflow size and count limits."""

    coordinator_max_lines: int
    coordinator_max_bytes: int
    reusable_max_lines: int
    reusable_max_bytes: int
    max_reusable_workflows: int


class EventPolicy(TypedDict):
    """Coordinator trigger, permission and concurrency contract."""

    push_branches: list[str]
    pull_request_branches: list[str]
    permissions: dict[str, str]
    concurrency_group: str
    cancel_in_progress: bool


class ArtifactContract(TypedDict):
    """One artifact producer and its declared consumers."""

    name: str
    producer: str
    consumers: list[str]


class WorkflowPolicy(TypedDict):
    """Complete distributed CI ownership and semantic-parity contract."""

    schema_version: int
    coordinator: str
    required_gate: str
    stable_required_jobs: list[str]
    event_policy: EventPolicy
    limits: WorkflowLimits
    categories: list[WorkflowCategory]
    job_order: list[str]
    dependency_graph: dict[str, list[str]]
    artifacts: list[ArtifactContract]
    conditional_steps: dict[str, list[str]]
    step_output_consumers: dict[str, list[str]]


def load_ci_workflow_policy() -> WorkflowPolicy:
    """Load the versioned distributed CI policy."""
    payload = json.loads(CI_WORKFLOW_POLICY.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CI workflow policy must be a JSON object")
    return cast(WorkflowPolicy, payload)


def ci_workflow_paths(policy: WorkflowPolicy | None = None) -> tuple[Path, ...]:
    """Return the coordinator followed by reusable workflows in policy order."""
    resolved = load_ci_workflow_policy() if policy is None else policy
    paths = [REPOSITORY_ROOT / resolved["coordinator"]]
    paths.extend(REPOSITORY_ROOT / category["workflow"] for category in resolved["categories"])
    return tuple(paths)


def _job_blocks(workflow: str) -> dict[str, str]:
    """Extract top-level job blocks without normalising executable YAML."""
    lines = workflow.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    jobs_seen = False
    for index, line in enumerate(lines):
        if line.rstrip("\n") == "jobs:":
            jobs_seen = True
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if jobs_seen and match:
            starts.append((index, match.group(1)))
    blocks: dict[str, str] = {}
    for position, (start, job_id) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        if job_id in blocks:
            raise ValueError(f"CI job appears multiple times in one workflow: {job_id}")
        blocks[job_id] = "".join(lines[start:end]).strip("\n")
    return blocks


def read_ci_workflow_source() -> str:
    """Return all executable CI jobs as one ordered compatibility view.

    The view is reconstructed from real workflow files and contains no stored
    duplicate of the former coordinator. It exists for contract readers that
    need to inspect job bodies independently of physical category ownership.
    """
    policy = load_ci_workflow_policy()
    coordinator_path = REPOSITORY_ROOT / policy["coordinator"]
    coordinator = coordinator_path.read_text(encoding="utf-8")
    prefix, separator, _jobs = coordinator.partition("jobs:\n")
    if not separator:
        raise ValueError("CI coordinator is missing jobs")
    blocks: dict[str, str] = {}
    for path in ci_workflow_paths(policy)[1:]:
        for job_id, block in _job_blocks(path.read_text(encoding="utf-8")).items():
            if job_id in blocks:
                raise ValueError(f"CI job appears in multiple reusable workflows: {job_id}")
            blocks[job_id] = block
    coordinator_blocks = _job_blocks(coordinator)
    gate_id = policy["required_gate"]
    if gate_id not in coordinator_blocks:
        raise ValueError(f"CI coordinator is missing required gate {gate_id}")
    missing = set(policy["job_order"]) - blocks.keys()
    if missing:
        raise ValueError(f"CI workflow inventory is missing jobs: {sorted(missing)}")
    ordered = [blocks[job_id] for job_id in policy["job_order"]]
    ordered.append(coordinator_blocks[gate_id])
    return prefix + "jobs:\n" + "\n\n".join(ordered) + "\n"


def workflow_path_for_job(job_id: str) -> Path:
    """Resolve the reusable workflow that exclusively owns the job."""
    policy = load_ci_workflow_policy()
    for category in policy["categories"]:
        if job_id in category["jobs"]:
            return REPOSITORY_ROOT / category["workflow"]
    if job_id == policy["required_gate"]:
        return REPOSITORY_ROOT / policy["coordinator"]
    raise KeyError(job_id)


def category_for_job(job_id: str) -> str:
    """Return the category identifier that owns the job."""
    policy = load_ci_workflow_policy()
    for category in policy["categories"]:
        if job_id in category["jobs"]:
            return category["id"]
    raise KeyError(job_id)


__all__ = [
    "CI_WORKFLOW_POLICY",
    "REPOSITORY_ROOT",
    "ArtifactContract",
    "EventPolicy",
    "WorkflowCategory",
    "WorkflowLimits",
    "WorkflowPolicy",
    "category_for_job",
    "ci_workflow_paths",
    "load_ci_workflow_policy",
    "read_ci_workflow_source",
    "workflow_path_for_job",
]
