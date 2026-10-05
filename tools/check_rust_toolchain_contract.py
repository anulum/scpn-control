#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reproducible Rust toolchain contract gate
"""Check exact local Rust pins and real YAML action-input declarations.

This read-only repository policy checks one fixed TOML table and seven fixed
workflow files. Workflow actions count only inside jobs' step sequences and
their toolchain/components values come only from each action's own ``with``
mapping. YAML nodes are composed without constructing tags. The policy validates
declarations, not Rust installation, hosted execution, action authenticity,
conditional reachability, compiler behaviour, or scientific/native admission.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from typing import Final, cast

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

ROOT: Final = Path(__file__).resolve().parents[1]
STABLE_TOOLCHAIN: Final = "1.98.0"
NIGHTLY_TOOLCHAIN: Final = "nightly-2026-08-18"
EXPECTED_WORKFLOWS: Final = {
    ".github/workflows/ci-native-polyglot.yml": (STABLE_TOOLCHAIN, 2, "rustfmt, clippy"),
    ".github/workflows/ci-rust-benchmark.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/ci-security-supply-chain.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/ci-static-governance.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/pre-commit.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/benchmark-nightly.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/fuzz-nightly.yml": (NIGHTLY_TOOLCHAIN, 2, None),
}
_FULL_SHA_RE: Final = re.compile(r"[0-9a-f]{40}")


class _WorkflowContractError(ValueError):
    """Represent malformed or ambiguous workflow pin declarations."""


def _mapping(node: Node | None, label: str) -> dict[str, Node]:
    """Read one ordinary mapping with unique string keys and no merge ambiguity.

    Parameters
    ----------
    node : yaml.nodes.Node or None
        Composed YAML node; custom constructors are never invoked.
    label : str
        Structural location for authored diagnostics.

    Returns
    -------
    dict[str, yaml.nodes.Node]
        Child nodes keyed by literal strings, in declaration order.

    Raises
    ------
    _WorkflowContractError
        The node is not an ordinary mapping, has a non-string key, repeats a
        key, or contains a merge key. BaseLoader leaves implicit ``on`` and
        version scalars as strings; explicitly tagged keys remain constrained.
    """
    if not isinstance(node, MappingNode) or node.tag != "tag:yaml.org,2002:map":
        raise _WorkflowContractError(f"{label} requires a mapping")
    fields: dict[str, Node] = {}
    for key, value in node.value:
        if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
            raise _WorkflowContractError(f"{label} requires string keys")
        if key.value == "<<" or key.value in fields:
            raise _WorkflowContractError(f"{label} has an ambiguous key {key.value!r}")
        fields[key.value] = value
    return fields


def _scalar(node: Node, label: str) -> str:
    """Read one literal string scalar without coercion or tag construction.

    Parameters
    ----------
    node : yaml.nodes.Node
        Composed action field.
    label : str
        Structural location for diagnostics.

    Returns
    -------
    str
        Exact scalar value, including significant whitespace.

    Raises
    ------
    _WorkflowContractError
        A relevant action field is a collection or an explicitly non-string tag.
    """
    if not isinstance(node, ScalarNode) or node.tag != "tag:yaml.org,2002:str":
        raise _WorkflowContractError(f"{label} requires a string")
    return cast(str, node.value)


def _toolchain_steps(path: Path) -> list[tuple[str, str | None, str | None]]:
    """Collect Rust actions from real job steps and each action's own inputs.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 workflow file. Named steps and ordinary aliases are supported.

    Returns
    -------
    list[tuple[str, str or None, str or None]]
        Literal action suffix, toolchain, and components in job/step order.
        Missing input fields remain None. Run strings and env mappings do not
        supply actions or inputs; unrelated custom tags are not constructed.

    Raises
    ------
    OSError
        The workflow cannot be read.
    UnicodeError
        The file is not UTF-8.
    yaml.YAMLError
        YAML parsing fails, including multiple documents or undefined aliases.
    _WorkflowContractError
        Relevant mappings, step sequences, or scalar fields are ambiguous or
        malformed. This policy does not validate the complete Actions schema.
    """
    document = yaml.compose(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    workflow = _mapping(document, "workflow")
    jobs = _mapping(workflow.get("jobs"), "jobs")
    steps: list[tuple[str, str | None, str | None]] = []
    for name, job_node in jobs.items():
        job = _mapping(job_node, f"job {name!r}")
        sequence = job.get("steps")
        if sequence is None:
            continue
        if not isinstance(sequence, SequenceNode) or sequence.tag != "tag:yaml.org,2002:seq":
            raise _WorkflowContractError(f"job {name!r} steps requires a sequence")
        for index, step_node in enumerate(sequence.value):
            label = f"job {name!r} step {index}"
            step = _mapping(step_node, label)
            uses = step.get("uses")
            if uses is None:
                continue
            action = _scalar(uses, label + " uses")
            if not action.startswith("dtolnay/rust-toolchain@"):
                continue
            settings_node = step.get("with")
            settings = {} if settings_node is None else _mapping(settings_node, label + " with")
            toolchain_node = settings.get("toolchain")
            components_node = settings.get("components")
            toolchain = None if toolchain_node is None else _scalar(toolchain_node, label + " toolchain")
            components = None if components_node is None else _scalar(components_node, label + " components")
            steps.append((action.removeprefix("dtolnay/rust-toolchain@"), toolchain, components))
    return steps


def check_rust_toolchain_contract(root: Path = ROOT) -> list[str]:
    """Compare one repository's Rust declarations with the fixed local policy.

    Parameters
    ----------
    root : pathlib.Path, default ROOT
        Repository root. Relative roots resolve from caller cwd; the default
        resolves from this script. Symlinks retain ordinary pathlib behaviour.

    Returns
    -------
    list[str]
        Fresh findings in local-table then declared-workflow/step order. The
        stable channel is 1.98.0, ordered local components are clippy/rustfmt,
        profile is minimal, and no other TOML table/key is allowed. Seven fixed
        workflows require exact counts, literal channels/component strings and
        40 lowercase hexadecimal action refs. SHA syntax is not authentication.
        Missing/unreadable inputs and TOML/YAML/structure errors are findings.

    Raises
    ------
    UnicodeError
        A present input is not UTF-8. No native installation or file write occurs.

    Notes
    -----
    This API checks declarations and returns caller-owned strings/list. It has
    no numerical units, array shapes, simulation clock, controller state, remote
    lookup, or compiler execution. It does not prove steps run or Rust parity.
    """
    errors: list[str] = []
    toolchain_path = root / "rust-toolchain.toml"
    try:
        payload = tomllib.loads(toolchain_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"cannot read rust-toolchain.toml: {exc}"]
    toolchain = payload.get("toolchain")
    if not isinstance(toolchain, dict):
        return ["rust-toolchain.toml requires a [toolchain] table"]
    typed_toolchain = cast(dict[str, object], toolchain)
    expected_table: dict[str, object] = {
        "channel": STABLE_TOOLCHAIN,
        "components": ["clippy", "rustfmt"],
        "profile": "minimal",
    }
    if typed_toolchain != expected_table:
        errors.append(f"rust-toolchain.toml drift: expected {expected_table!r}, got {typed_toolchain!r}")
    if set(payload) != {"toolchain"}:
        errors.append("rust-toolchain.toml may contain only the [toolchain] table")

    for relative_path, (expected_toolchain, expected_count, expected_components) in EXPECTED_WORKFLOWS.items():
        path = root / relative_path
        try:
            steps = _toolchain_steps(path)
        except OSError as exc:
            errors.append(f"cannot read {relative_path}: {exc}")
            continue
        except (yaml.YAMLError, _WorkflowContractError) as exc:
            errors.append(f"cannot parse {relative_path}: {exc}")
            continue
        if len(steps) != expected_count:
            errors.append(f"{relative_path}: expected {expected_count} Rust toolchain steps, found {len(steps)}")
        for ref, actual_toolchain, actual_components in steps:
            if _FULL_SHA_RE.fullmatch(ref) is None:
                errors.append(f"{relative_path}: rust-toolchain action is not pinned to a full SHA: {ref}")
            if actual_toolchain != expected_toolchain:
                errors.append(
                    f"{relative_path}: expected toolchain {expected_toolchain}, got {actual_toolchain or 'implicit'}"
                )
            if actual_components != expected_components:
                errors.append(
                    f"{relative_path}: expected components {expected_components or 'none'}, "
                    f"got {actual_components or 'none'}"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Inspect the selected real repository through the maintained CLI entry.

    Parameters
    ----------
    argv : list[str] or None, default None
        Argparse arguments; None reads process arguments. ``--root`` selects a
        pathlib repository root, with the resolved script root as its default.

    Returns
    -------
    int
        Zero for matching declarations; one after printing ``FAIL:`` findings.

    Raises
    ------
    SystemExit
        Argparse help exits zero and malformed/unknown arguments exit two.
    UnicodeError
        A required present file is not UTF-8.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    errors = check_rust_toolchain_contract(args.root)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"Rust toolchain contract passed: stable={STABLE_TOOLCHAIN} nightly={NIGHTLY_TOOLCHAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
