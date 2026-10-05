# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native coverage declaration decoding.

"""Decode coverage YAML/TOML and bind producer/consumer dependency declarations."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Hashable
from pathlib import Path
from typing import NoReturn, cast

import yaml
from yaml.nodes import MappingNode, Node

from tools.native_coverage_contracts._steps import _active, _combined, _mapping, _producer

_RUST_TESTS = (
    "tests/test_aer_observation_rust_parity.py",
    "tests/test_boris_pyo3_bridge.py",
    "tests/test_capacitor_bank_state_pyo3.py",
    "tests/test_controller_advanced_paths.py",
    "tests/test_fusion_neural_mpc_pulsed_adapter_rust_parity.py",
    "tests/test_multi_shot_campaign.py",
    "tests/test_multi_shot_campaign_pyo3.py",
    "tests/test_pyo3_control_bridge.py",
    "tests/test_rust_compat_wrapper.py",
    "tests/test_rust_python_parity.py",
    "tests/test_rust_realtime_parity.py",
    "tests/test_snn_pyo3_bridge.py",
)
_BOOLEAN_RULE = ("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"))


class _UniqueLoader(yaml.SafeLoader):
    """Decode GitHub booleans and unique explicit keys, retaining YAML merges."""

    yaml_implicit_resolvers = {
        initial: [(tag, pattern) for tag, pattern in resolvers if tag != _BOOLEAN_RULE[0]]
        + ([_BOOLEAN_RULE] if initial in {"t", "T", "f", "F"} else [])
        for initial, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node: Node, deep: bool = False) -> dict[Hashable, object]:
        """Build one safely dispatched YAML map after checking its explicit keys."""
        if not isinstance(node, MappingNode):
            raise yaml.YAMLError("workflow mapping requires a mapping node")
        mapping = node
        seen: set[Hashable] = set()
        for key_node, _value_node in mapping.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key: object = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable) or key in seen:
                raise yaml.YAMLError("workflow mappings require unique hashable keys")
            seen.add(key)
        return cast(dict[Hashable, object], super().construct_mapping(mapping, deep=deep))


def _load_config(text: str) -> dict[str, object]:
    """Decode a safe string-keyed mapping with unique explicit YAML/JSON keys."""
    try:
        value: object = yaml.load(text, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError("coverage workflow must be valid unique-key YAML") from exc
    return _mapping(value)


def _load_yaml(text: str) -> dict[str, object]:
    """Read safe unique-key YAML with a nonempty job mapping, without executing it."""
    root = _load_config(text)
    if not _mapping(root.get("jobs")):
        raise ValueError("coverage workflow must declare jobs")
    return root


def _needs(value: object, expected: tuple[str, ...]) -> bool:
    """Compare declared dependency IDs as a unique unordered exact set."""
    items = [value] if isinstance(value, str) else value
    return (
        isinstance(items, list)
        and all(isinstance(item, str) for item in items)
        and len(items) == len(expected)
        and set(items) == set(expected)
    )


def _context_job(root: dict[str, object], job: object) -> dict[str, object]:
    """Copy a job with inherited workflow environment and run defaults."""
    value = _mapping(job)
    return {
        **value,
        "env": {**_mapping(root.get("env")), **_mapping(value.get("env"))},
        "defaults": {
            "run": {
                **_mapping(_mapping(root.get("defaults")).get("run")),
                **_mapping(_mapping(value.get("defaults")).get("run")),
            }
        },
    }


def _invalid_json_constant(_token: str) -> NoReturn:
    """Reject the non-finite constants admitted by Python's permissive decoder."""
    raise ValueError("coverage policy requires finite standard JSON values")


def _distributed_jobs(policy_path: Path) -> tuple[dict[str, dict[str, object]], bool]:
    """Read physical reusable owners and their enabled exact coordinator calls."""
    text = policy_path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=_invalid_json_constant)
    policy = _load_config(text)
    repository_root = policy_path.resolve().parent.parent
    categories = policy.get("categories")
    coordinator_path = policy.get("coordinator")
    if (
        not isinstance(categories, list)
        or type(policy.get("schema_version")) is not int
        or policy.get("schema_version") != 1
        or not isinstance(coordinator_path, str)
    ):
        raise ValueError("coverage policy requires its coordinator and categories")
    coordinator_file = (repository_root / coordinator_path).resolve()
    if not coordinator_file.is_relative_to(repository_root / ".github/workflows") or coordinator_file.suffix not in {
        ".yml",
        ".yaml",
    }:
        raise ValueError("coverage coordinator must be a repository workflow file")
    coordinator = _load_yaml(coordinator_file.read_text(encoding="utf-8"))
    calls = _mapping(coordinator["jobs"])
    jobs: dict[str, dict[str, object]] = {}
    required = {
        "python-quality": ("python-tests", ()),
        "native-polyglot": ("rust-python-interop", ()),
        "native-coverage": ("native-coverage-combine", ("python-quality", "native-polyglot")),
    }
    seen: set[str] = set()
    needs_ok = _needs(
        _mapping(policy.get("dependency_graph")).get("native-coverage-combine"),
        ("python-tests", "rust-python-interop"),
    )
    for entry in categories:
        category = _mapping(entry)
        category_id = category.get("id")
        if not isinstance(category_id, str):
            raise ValueError("coverage category IDs must be strings")
        if category_id not in required:
            continue
        if category_id in seen:
            raise ValueError("coverage categories require unique ownership")
        seen.add(category_id)
        job_id, expected = required[category_id]
        workflow = f".github/workflows/ci-{category_id}.yml"
        owners = category.get("jobs")
        if category.get("workflow") != workflow or not isinstance(owners, list) or job_id not in owners:
            raise ValueError("coverage category must bind its physical job owner")
        root = _load_yaml((repository_root / workflow).read_text(encoding="utf-8"))
        call = _mapping(calls.get(category_id))
        enabled = (
            _active(call)
            and _needs(call.get("needs", []), expected)
            and _needs(category.get("caller_needs"), expected)
            and call.get("uses") == "./" + workflow
        )
        jobs[job_id] = _context_job(root, _mapping(root["jobs"]).get(job_id)) if enabled else {}
        if category_id == "native-coverage":
            needs_ok = needs_ok and enabled
    if seen != set(required):
        raise ValueError("coverage policy is missing a required category")
    return jobs, needs_ok


def _workflow_checks(text: str, *, policy_path: Path | None) -> tuple[bool, bool, bool]:
    """Return producer/consumer declaration verdicts for native coverage.

    Malformed YAML, shell declarations, input IO or policy data may raise;
    the public guard converts these failures to authored findings. A passing
    declaration verdict does not establish hosted execution or collected data.
    """
    if policy_path is not None:
        jobs, needs = _distributed_jobs(policy_path)
    else:
        root = _load_yaml(text)
        jobs = {name: _context_job(root, job) for name, job in _mapping(root["jobs"]).items()}
        needs = _needs(jobs.get("native-coverage-combine", {}).get("needs"), ("python-tests", "rust-python-interop"))
    combined = jobs.get("native-coverage-combine", {})
    return (
        _producer(_mapping(jobs.get("python-tests"))),
        _producer(_mapping(jobs.get("rust-python-interop")), _RUST_TESTS),
        _combined(combined, needs),
    )


def _threshold100(text: str) -> bool:
    """Require the parsed TOML coverage report threshold to be numeric 100."""
    config = tomllib.loads(text)
    tool = _mapping(config.get("tool"))
    report = _mapping(_mapping(tool.get("coverage")).get("report"))
    value = report.get("fail_under")
    return type(value) in {int, float} and value == 100
