# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — the clean wheel installation stays complete

"""The clean-install lock carries exactly the package's runtime dependencies.

``build-package / package-quality`` installs the built wheel with ``--no-deps``
into an environment holding only ``requirements/ci-wheel-smoke.txt`` and then
runs ``pip check``. That proves completeness only while the lock's input names
the same requirements as ``[project].dependencies``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci-build-package.yml"


def _requirements(lines: list[str]) -> set[str]:
    """Return normalised requirement strings, ignoring comments and options."""
    return {
        str(Requirement(line.strip())) for line in lines if line.strip() and not line.lstrip().startswith(("#", "-"))
    }


def test_lock_input_names_exactly_the_runtime_dependencies() -> None:
    """Keep the clean-install lock input equal to ``[project].dependencies``."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = _requirements(list(project["dependencies"]))
    locked = _requirements((ROOT / "requirements" / "ci-wheel-smoke.in").read_text(encoding="utf-8").splitlines())
    assert locked == declared


def test_every_runtime_dependency_is_pinned_with_hashes() -> None:
    """Pin every runtime dependency in the hashed lock."""
    lock = (ROOT / "requirements" / "ci-wheel-smoke.txt").read_text(encoding="utf-8")
    pinned = {line.split("==", 1)[0] for line in lock.splitlines() if "==" in line and not line.startswith(" ")}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert {Requirement(item).name for item in project["dependencies"]} <= pinned
    assert "--hash=sha256:" in lock


def test_package_quality_runs_the_wheel_outside_the_checkout() -> None:
    """Install with ``--no-deps``, prove completeness and run from outside the tree."""
    text = WORKFLOW.read_text(encoding="utf-8")
    for fragment in (
        "unset PYTHONPATH",
        "-m pip install --require-hashes -r requirements/ci-wheel-smoke.txt",
        "-m pip install --no-deps dist/scpn_control-*.whl",
        "-m pip check",
        'cd "$RUNNER_TEMP"',
        'test "$installed" = "$expected"',
        '"$venv/bin/scpn-control" --help',
    ):
        assert fragment in text
