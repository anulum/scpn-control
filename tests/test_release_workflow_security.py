# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Release workflow security regression tests.

"""Exercise production-publish guards and changelog extraction refusal."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict[str, object]:
    """Load one committed workflow as its parsed mapping."""
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_production_pypi_step_requires_tag_for_push_and_dispatch() -> None:
    """Keep manually dispatched branch builds out of production PyPI."""
    workflow = _workflow("publish-pypi.yml")
    steps = workflow["jobs"]["publish"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish to PyPI")
    condition = publish["if"]

    assert "startsWith(github.ref, 'refs/tags/v')" in condition
    assert "github.event_name == 'push'" in condition
    assert "github.event.inputs.repository == 'pypi'" in condition


@pytest.mark.parametrize(
    ("ref", "accepted"),
    [
        ("refs/tags/v0.23.0", True),
        ("refs/tags/v0.99.0", False),
        ("refs/tags/v0.23.00", False),
        ("refs/tags/v0.23.0/extra", False),
        ('refs/tags/v0.23.0") { system("id") }', False),
        ("refs/heads/main", False),
    ],
)
def test_release_changelog_extraction_refuses_unsafe_or_missing_tags(tmp_path: Path, ref: str, accepted: bool) -> None:
    """Run the actual release step with both valid and adversarial refs."""
    workflow = _workflow("release.yml")
    steps = workflow["jobs"]["release"]["steps"]
    script = next(step["run"] for step in steps if step.get("id") == "changelog")
    output = tmp_path / "github_output"
    environment = {**os.environ, "GITHUB_REF": ref, "GITHUB_OUTPUT": str(output), "RUNNER_TEMP": str(tmp_path)}

    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode == 0) is accepted, result.stderr
    if accepted:
        notes = (tmp_path / "release_notes.md").read_text(encoding="utf-8")
        assert "### Added" in notes
        assert "## [0.22.1]" not in notes
        assert len(notes) > 1000
        assert output.read_text(encoding="utf-8") == f"notes_file={tmp_path}/release_notes.md\n"
    else:
        assert not output.exists()
