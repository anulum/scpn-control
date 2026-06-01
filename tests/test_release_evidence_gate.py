# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Release Evidence Gate Workflow Tests

"""Workflow contract tests for release-evidence admission automation.

The release-evidence workflow must run the top-level validation command in CI
and persist its JSON output as an auditable artifact. This cross-surface test is
allowed because it covers the named release-evidence workflow contract.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_publishes_top_level_release_evidence_report():
    """CI must run top-level validation and upload the release evidence report."""
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "release-evidence-gate:" in workflow
    assert "python -m scpn_control.cli validate --json-out > artifacts/release_evidence_report.json" in workflow
    assert (
        "python validation/validate_release_evidence.py artifacts/release_evidence_report.json "
        "--json-out > artifacts/release_evidence_admission.json"
    ) in workflow
    assert "name: release-evidence-report" in workflow
    assert "path: |" in workflow
    assert "artifacts/release_evidence_report.json" in workflow
    assert "artifacts/release_evidence_admission.json" in workflow
