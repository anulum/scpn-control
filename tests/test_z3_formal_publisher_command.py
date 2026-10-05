# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal validator declaration and public command evidence

"""Run actual installed Z3 bounded proof publisher into isolated output paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.scpn.z3_formal_report import load_z3_formal_report
from validation.validate_scpn_z3_formal import publish_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["api", "script"])
def test_actual_bounded_z3_publisher_preserves_proof_scope(tmp_path: Path, entry: str) -> None:
    """Installed Z3 proves the actual two-step reference net through public entry points."""
    target = tmp_path / "nested/z3.json"
    markdown = tmp_path / "other/z3.md"
    if entry == "api":
        result = publish_report(json_path=target, markdown_path=markdown, require_z3=True)
        assert result == {"status": "pass", "backend": "z3", "holds": True, "max_depth": 2}
    else:
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "validation/validate_scpn_z3_formal.py"),
                "--json-out",
                str(target),
                "--markdown-out",
                str(markdown),
                "--require-z3",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads(process.stdout) == {"status": "pass", "backend": "z3", "holds": True, "max_depth": 2}
    payload = load_z3_formal_report(target)
    assert payload["status"] == "pass" and payload["holds"] is True and payload["max_depth"] == 2
    assert payload["scope"] == "bounded SMT evidence for compiled Petri-net control logic"
    assert payload["claim_boundary"] == "not hardware timing evidence, PCS certification, or unbounded liveness proof"
    assert payload["solver"].startswith("z3-solver ")
    assert not payload["safety"]["violations"] and not payload["temporal"]["violations"]
    assert "move_eventually_fires" in payload["checked_specs"]
    assert "move_marks_sink" in payload["checked_specs"]
    assert "exclusive_source_sink" in payload["checked_specs"]
    assert target.read_bytes().endswith(b"\n") and markdown.read_bytes().endswith(b"\n")
    assert "bounded" in markdown.read_text(encoding="utf-8")
