# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public inventory and coverage claim tests.
"""Regression tests for public inventory and coverage claims."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST: Final = ROOT / "docs" / "_generated" / "capability_manifest.json"


def _manifest_counts() -> dict[str, int]:
    """Return generated capability inventory counts."""

    raw: Any = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = cast(dict[str, Any], raw)
    return cast(dict[str, int], manifest["counts"])


def _coverage_gate() -> int:
    """Return the configured package coverage gate from ``pyproject.toml``."""

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^fail_under = (?P<gate>\d+)$", pyproject)
    assert match is not None
    return int(match.group("gate"))


def test_pitch_inventory_claim_matches_generated_manifest() -> None:
    """The pitch inventory summary must match generated capability counts."""

    counts = _manifest_counts()
    pitch = (ROOT / "docs" / "pitch.md").read_text(encoding="utf-8")

    assert f"{counts['source_module_count']} Python control/physics modules" in pitch
    assert f"{counts['rust_source_file_count']} Rust source files" in pitch
    assert f"{counts['python_test_file_count']} Python test files" in pitch
    assert f"{counts['workflow_count']} GitHub Actions workflows" in pitch


def test_readme_tree_test_count_matches_generated_manifest() -> None:
    """The README tree summary must match generated test-file inventory."""

    counts = _manifest_counts()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"tests/                 # {counts['python_test_file_count']} Python test files" in readme


def test_public_coverage_gate_claims_match_configuration() -> None:
    """Current public coverage-gate claims must match ``pyproject.toml``."""

    gate = _coverage_gate()
    pitch = (ROOT / "docs" / "pitch.md").read_text(encoding="utf-8")
    paper = (ROOT / "paper.md").read_text(encoding="utf-8")

    assert f"{gate}% configured coverage gate" in pitch
    assert f"{gate}% package-coverage gate" in paper


def test_current_surfaces_do_not_use_stale_live_inventory_counts() -> None:
    """Live public inventory summaries must not repeat stale current counts."""

    checked_paths = (
        ROOT / "README.md",
        ROOT / "docs" / "pitch.md",
        ROOT / "paper.md",
    )
    stale_fragments = (
        "153 Python modules",
        "4,000+ collected Python tests",
        "20 CI jobs",
        "3,700+ collected Python tests",
        "170+ more test files",
        "99% coverage gate",
        "99% package-coverage gate",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for fragment in stale_fragments:
            assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment}"


def test_safe_rl_paper_claim_matches_implementation_surface() -> None:
    """Paper safe-RL wording must match the implemented controller surface."""

    expected = "safe RL (CPO-formulated Lagrangian constraints with control barrier functions, Ames 2017)"
    stale = "safe RL (PPO with MHD constraint veto)"
    checked_paths = (
        ROOT / "paper.md",
        ROOT / "docs" / "joss_paper.md",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8").replace("\n  ", " ")
        assert expected in text, f"{path.relative_to(ROOT)} missing safe-RL implementation wording"
        assert stale not in text, f"{path.relative_to(ROOT)} contains stale safe-RL veto wording"


def test_neural_equilibrium_latency_claim_is_not_uncited() -> None:
    """Public neural-equilibrium docs must not repeat the uncited 0.39 ms claim."""

    checked_paths = (
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "pitch.md",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert "0.39 ms" not in text
        assert "0.39ms" not in text

    pitch = (ROOT / "docs" / "pitch.md").read_text(encoding="utf-8")
    assert "validation/reports/neural_equilibrium_pretraining.json" in pitch
    assert "real EFIT/P-EFIT" in pitch
    assert "latency or accuracy claims remain blocked" in pitch
