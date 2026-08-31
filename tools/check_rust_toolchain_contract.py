#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reproducible Rust toolchain contract gate
"""Fail closed when local and hosted Rust toolchain pins drift."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from typing import Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
STABLE_TOOLCHAIN: Final = "1.98.0"
NIGHTLY_TOOLCHAIN: Final = "nightly-2026-08-18"
EXPECTED_WORKFLOWS: Final = {
    ".github/workflows/ci.yml": (STABLE_TOOLCHAIN, 5, "rustfmt, clippy"),
    ".github/workflows/pre-commit.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/benchmark-nightly.yml": (STABLE_TOOLCHAIN, 1, "rustfmt, clippy"),
    ".github/workflows/fuzz-nightly.yml": (NIGHTLY_TOOLCHAIN, 2, None),
}
_ACTION_RE: Final = re.compile(r"^(?P<indent>\s*)-\s+uses:\s+dtolnay/rust-toolchain@(?P<ref>\S+?)(?:\s+#.*)?$")
_TOOLCHAIN_RE: Final = re.compile(r"^\s+toolchain:\s*[\"']?(?P<toolchain>[^\s\"']+)[\"']?(?:\s+#.*)?$")
_COMPONENTS_RE: Final = re.compile(r"^\s+components:\s*[\"']?(?P<components>[^\"']+?)[\"']?\s*(?:#.*)?$")
_FULL_SHA_RE: Final = re.compile(r"[0-9a-f]{40}")


def _toolchain_steps(path: Path) -> list[tuple[str, str | None, str | None]]:
    """Return action refs, toolchains, and components from one workflow."""
    lines = path.read_text(encoding="utf-8").splitlines()
    steps: list[tuple[str, str | None, str | None]] = []
    for index, line in enumerate(lines):
        match = _ACTION_RE.match(line)
        if match is None:
            continue
        indent = match.group("indent")
        toolchain: str | None = None
        components: str | None = None
        for candidate in lines[index + 1 :]:
            if candidate.startswith(f"{indent}- "):
                break
            toolchain_match = _TOOLCHAIN_RE.match(candidate)
            if toolchain_match is not None:
                toolchain = toolchain_match.group("toolchain")
            components_match = _COMPONENTS_RE.match(candidate)
            if components_match is not None:
                components = components_match.group("components")
        steps.append((match.group("ref"), toolchain, components))
    return steps


def check_rust_toolchain_contract(root: Path = ROOT) -> list[str]:
    """Return deterministic contract violations for a repository root."""
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
    """CLI entry point for the Rust toolchain contract gate."""
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
