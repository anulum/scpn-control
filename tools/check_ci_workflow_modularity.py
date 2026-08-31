# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Distributed CI modularity audit.

"""Reject incomplete, ambiguous, oversized, or non-fail-closed CI ownership."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ci_workflow_inventory import REPOSITORY_ROOT, WorkflowPolicy, load_ci_workflow_policy

_PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_STEP_OUTPUT = re.compile(r"steps\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)")
_LEGACY_COORDINATOR_LITERAL = ".github/workflows/ci.yml"
_ALLOWED_COORDINATOR_REFERENCES = {
    "tests/test_ci_workflow_modularity.py",
    "tests/test_public_surface_hygiene.py",
    "tests/test_studio_offline_sealing.py",
    "tests/test_tools/test_capability_manifest.py",
    "tools/check_ci_workflow_modularity.py",
    "tools/ci_workflow_inventory.py",
}


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load workflow YAML while preserving the on key as a string."""
    loader = yaml.BaseLoader(path.read_text(encoding="utf-8"))
    try:
        payload = loader.get_single_data()
    finally:
        loader.dispose()
    if not isinstance(payload, dict):
        raise ValueError(f"workflow must be a mapping: {path}")
    return payload


def _needs(job: dict[str, Any]) -> list[str]:
    """Normalise a scalar or sequence needs declaration."""
    value = job.get("needs", [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"job needs must be a string or string list: {value!r}")


def _measure(path: Path) -> tuple[int, int]:
    """Return physical line and UTF-8 byte counts."""
    text = path.read_text(encoding="utf-8")
    return len(text.splitlines()), len(text.encode("utf-8"))


def _job_text(path: Path, job_id: str) -> str:
    """Return the raw YAML block for one top-level job."""
    match = re.search(
        rf"^  {re.escape(job_id)}:\s*$.*?(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"workflow job not found: {job_id}")
    return match.group(0)


def _check_action_pins(workflow: dict[str, Any], path: Path, errors: list[str]) -> None:
    """Require immutable SHAs for every third-party action."""
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        errors.append(f"{path}: jobs must be a mapping")
        return
    for job_id, raw_job in jobs.items():
        if not isinstance(raw_job, dict):
            continue
        references: list[str] = []
        job_use = raw_job.get("uses")
        if isinstance(job_use, str):
            references.append(job_use)
        steps = raw_job.get("steps", [])
        if isinstance(steps, list):
            references.extend(
                step["uses"] for step in steps if isinstance(step, dict) and isinstance(step.get("uses"), str)
            )
        for reference in references:
            if reference.startswith("./"):
                continue
            candidate = reference.split(" #", maxsplit=1)[0]
            if _PINNED_ACTION.fullmatch(candidate) is None:
                errors.append(f"{path}: job {job_id} has unpinned action {reference}")


def _check_no_direct_coordinator_readers(errors: list[str]) -> None:
    """Prevent contracts from recoupling to the former monolithic coordinator."""
    for root_name in ("tests", "tools", "scripts"):
        root = REPOSITORY_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            if relative in _ALLOWED_COORDINATOR_REFERENCES:
                continue
            if _LEGACY_COORDINATOR_LITERAL in path.read_text(encoding="utf-8"):
                errors.append(f"{relative}: inspect the distributed CI inventory, not the coordinator")


def _artifact_inventory(workflows: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    """Return artifact producers and consumers from executable steps."""
    inventory: dict[str, dict[str, object]] = {}
    for workflow in workflows:
        jobs = cast(dict[str, Any], workflow["jobs"])
        for job_id, raw_job in jobs.items():
            if not isinstance(raw_job, dict):
                continue
            job = cast(dict[str, Any], raw_job)
            for raw_step in cast(list[Any], job.get("steps", [])):
                if not isinstance(raw_step, dict):
                    continue
                reference = raw_step.get("uses")
                options = raw_step.get("with", {})
                if not isinstance(reference, str) or not isinstance(options, dict):
                    continue
                name = options.get("name")
                if not isinstance(name, str):
                    continue
                record = inventory.setdefault(name, {"producer": None, "consumers": []})
                if reference.startswith("actions/upload-artifact@"):
                    record["producer"] = job_id
                elif reference.startswith("actions/download-artifact@"):
                    cast(list[str], record["consumers"]).append(job_id)
    return inventory


def _conditional_steps(workflows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return named steps carrying execution conditions."""
    result: dict[str, list[str]] = {}
    for workflow in workflows:
        jobs = cast(dict[str, Any], workflow["jobs"])
        for job_id, raw_job in jobs.items():
            if not isinstance(raw_job, dict):
                continue
            job = cast(dict[str, Any], raw_job)
            names = [
                cast(str, step["name"])
                for step in cast(list[Any], job.get("steps", []))
                if isinstance(step, dict) and "if" in step and isinstance(step.get("name"), str)
            ]
            if names:
                result[job_id] = names
    return result


def _step_output_consumers(workflows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return step-output references used by each job."""
    result: dict[str, list[str]] = {}
    for workflow in workflows:
        jobs = cast(dict[str, Any], workflow["jobs"])
        for job_id, raw_job in jobs.items():
            if not isinstance(raw_job, dict):
                continue
            references = sorted({f"{step}.{output}" for step, output in _STEP_OUTPUT.findall(json.dumps(raw_job))})
            if references:
                result[job_id] = references
    return result


def audit_ci_workflow_modularity(policy: WorkflowPolicy | None = None) -> list[str]:
    """Return deterministic violations of the distributed CI contract."""
    resolved = load_ci_workflow_policy() if policy is None else policy
    errors: list[str] = []
    limits = resolved["limits"]
    categories = resolved["categories"]
    if resolved["schema_version"] != 1:
        errors.append("CI workflow policy schema_version must equal 1")
    if len(categories) > limits["max_reusable_workflows"]:
        errors.append("CI reusable workflow count exceeds repository policy")

    workflow_root = REPOSITORY_ROOT / ".github/workflows"
    expected_paths = {REPOSITORY_ROOT / category["workflow"] for category in categories}
    if set(workflow_root.glob("ci-*.yml")) != expected_paths:
        errors.append("physical CI category workflows do not match the versioned policy")

    coordinator_path = REPOSITORY_ROOT / resolved["coordinator"]
    coordinator_lines, coordinator_bytes = _measure(coordinator_path)
    if coordinator_lines > limits["coordinator_max_lines"]:
        errors.append(f"{coordinator_path}: {coordinator_lines} lines exceed {limits['coordinator_max_lines']}")
    if coordinator_bytes > limits["coordinator_max_bytes"]:
        errors.append(f"{coordinator_path}: {coordinator_bytes} bytes exceed {limits['coordinator_max_bytes']}")

    coordinator = _load_workflow(coordinator_path)
    event_policy = resolved["event_policy"]
    trigger = coordinator.get("on")
    if not isinstance(trigger, dict):
        errors.append("CI coordinator trigger must be a mapping")
    else:
        push = trigger.get("push")
        pull_request = trigger.get("pull_request")
        if not isinstance(push, dict) or push.get("branches") != event_policy["push_branches"]:
            errors.append("CI coordinator push trigger differs from policy")
        if not isinstance(pull_request, dict) or pull_request.get("branches") != event_policy["pull_request_branches"]:
            errors.append("CI coordinator pull-request trigger differs from policy")
        if set(trigger) != {"push", "pull_request"}:
            errors.append("CI coordinator exposes undeclared triggers")
    if coordinator.get("permissions") != event_policy["permissions"]:
        errors.append("CI coordinator permissions differ from policy")
    concurrency = coordinator.get("concurrency")
    if not isinstance(concurrency, dict) or concurrency != {
        "group": event_policy["concurrency_group"],
        "cancel-in-progress": str(event_policy["cancel_in_progress"]).lower(),
    }:
        errors.append("CI coordinator concurrency differs from policy")

    coordinator_jobs = coordinator.get("jobs", {})
    if not isinstance(coordinator_jobs, dict):
        return [*errors, f"{coordinator_path}: jobs must be a mapping"]
    category_ids = [category["id"] for category in categories]
    expected_coordinator_jobs = {*category_ids, resolved["required_gate"]}
    if set(coordinator_jobs) != expected_coordinator_jobs:
        errors.append("CI coordinator jobs do not match category calls plus the required gate")

    owned_jobs: dict[str, str] = {}
    executable_workflows: list[dict[str, Any]] = []
    category_jobs: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_id = category["id"]
        path = REPOSITORY_ROOT / category["workflow"]
        lines, size = _measure(path)
        if lines > limits["reusable_max_lines"]:
            errors.append(f"{path}: {lines} lines exceed {limits['reusable_max_lines']}")
        if size > limits["reusable_max_bytes"]:
            errors.append(f"{path}: {size} bytes exceed {limits['reusable_max_bytes']}")
        workflow = _load_workflow(path)
        workflow_trigger = workflow.get("on")
        if not isinstance(workflow_trigger, dict) or set(workflow_trigger) != {"workflow_call"}:
            errors.append(f"{path}: reusable category must expose only workflow_call")
        else:
            call = workflow_trigger["workflow_call"]
            declared_secrets: list[str] = []
            if isinstance(call, dict):
                raw_secrets = call.get("secrets", {})
                if isinstance(raw_secrets, dict):
                    declared_secrets = list(raw_secrets)
                    if any(
                        not isinstance(value, dict) or value.get("required") != "false"
                        for value in raw_secrets.values()
                    ):
                        errors.append(f"{path}: workflow_call secrets must be optional")
            if declared_secrets != category["secrets"]:
                errors.append(f"{path}: workflow_call secrets differ from policy")
        if workflow.get("permissions") != event_policy["permissions"]:
            errors.append(f"{path}: reusable category permissions differ from policy")
        jobs = workflow.get("jobs", {})
        if not isinstance(jobs, dict):
            errors.append(f"{path}: jobs must be a mapping")
            continue
        executable_workflows.append(workflow)
        category_jobs[category_id] = jobs
        if list(jobs) != category["jobs"]:
            errors.append(f"{path}: job order or ownership differs from policy")
        for job_id, raw_job in jobs.items():
            if job_id in owned_jobs:
                errors.append(f"CI job {job_id} is duplicated in {owned_jobs[job_id]} and {category_id}")
            owned_jobs[job_id] = category_id
            if not isinstance(raw_job, dict):
                errors.append(f"{path}: job {job_id} must be a mapping")

        caller = coordinator_jobs.get(category_id)
        if not isinstance(caller, dict):
            errors.append(f"CI coordinator is missing reusable call {category_id}")
        else:
            if caller.get("uses") != f"./{category['workflow']}":
                errors.append(f"CI coordinator call {category_id} targets the wrong workflow")
            if _needs(caller) != category["caller_needs"]:
                errors.append(f"CI coordinator call {category_id} has incorrect dependencies")
            expected_secret_values = {secret: f"${{{{ secrets.{secret} }}}}" for secret in category["secrets"]}
            if caller.get("secrets", {}) != expected_secret_values:
                errors.append(f"CI coordinator call {category_id} has incorrect secret surface")
            if set(caller) - {"uses", "needs", "secrets"}:
                errors.append(f"CI coordinator call {category_id} contains executable job fields")
        _check_action_pins(workflow, path, errors)

    declared_jobs = [job for category in categories for job in category["jobs"]]
    if set(owned_jobs) != set(resolved["job_order"]) or set(declared_jobs) != set(resolved["job_order"]):
        errors.append("distributed CI job inventory is incomplete or contains undeclared jobs")
    if not set(resolved["stable_required_jobs"]).issubset(owned_jobs):
        errors.append("stable branch-protection job inventory is incomplete")

    dependency_graph = resolved["dependency_graph"]
    category_index = {category["id"]: index for index, category in enumerate(categories)}
    for category in categories:
        path = REPOSITORY_ROOT / category["workflow"]
        jobs = category_jobs.get(category["id"])
        if jobs is None:
            continue
        cross_categories: set[str] = set()
        for job_id, raw_job in jobs.items():
            if not isinstance(raw_job, dict):
                continue
            expected = dependency_graph.get(job_id, [])
            internal = [dependency for dependency in expected if owned_jobs.get(dependency) == category["id"]]
            actual = _needs(cast(dict[str, Any], raw_job))
            if actual != internal:
                errors.append(f"{path}: job {job_id} dependencies differ from policy")
            cross_categories.update(
                owned_jobs[dependency] for dependency in expected if owned_jobs.get(dependency) != category["id"]
            )
        ordered_cross = sorted(cross_categories, key=category_index.__getitem__)
        if ordered_cross != category["caller_needs"]:
            errors.append(f"{path}: lifted cross-category dependencies differ from policy")

    actual_artifacts = _artifact_inventory(executable_workflows)
    expected_artifacts = {
        artifact["name"]: {
            "producer": artifact["producer"],
            "consumers": artifact["consumers"],
        }
        for artifact in resolved["artifacts"]
    }
    if actual_artifacts != expected_artifacts:
        errors.append("CI artifact producers or consumers differ from policy")
    if _conditional_steps(executable_workflows) != resolved["conditional_steps"]:
        errors.append("CI conditional-step inventory differs from policy")
    if _step_output_consumers(executable_workflows) != resolved["step_output_consumers"]:
        errors.append("CI step-output dependency inventory differs from policy")

    gate_id = resolved["required_gate"]
    gate = coordinator_jobs.get(gate_id)
    if not isinstance(gate, dict):
        errors.append(f"CI coordinator is missing required gate {gate_id}")
    else:
        if _needs(gate) != category_ids:
            errors.append("required CI gate does not aggregate every category exactly once")
        if gate.get("if") != "always()":
            errors.append("required CI gate must run with if: always()")
        gate_text = _job_text(coordinator_path, gate_id)
        if "toJSON(needs)" not in gate_text or 'value["result"] != "success"' not in gate_text:
            errors.append("required CI gate must fail closed over every category result")
    _check_action_pins(coordinator, coordinator_path, errors)
    _check_no_direct_coordinator_readers(errors)
    return errors


def main() -> int:
    """Print modularity violations and return a shell-compatible status."""
    errors = audit_ci_workflow_modularity()
    if errors:
        print("FAIL: distributed CI workflow contract")
        for error in errors:
            print(f"  - {error}")
        return 1
    policy = load_ci_workflow_policy()
    print(
        "PASS: distributed CI owns "
        f"{len(policy['job_order'])} jobs in {len(policy['categories'])} categories "
        f"with stable {policy['required_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_ci_workflow_modularity", "main"]
