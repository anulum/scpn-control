# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap model persistence and rendering tests
"""Exercise complete declared models through real matrix construction and storage."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

from tools.evidence_gap_matrix import ROOT, build_evidence_gap_matrix


def test_matrix_models_keep_existing_pickle_addresses_and_declarations(tmp_path: Path) -> None:
    """A persisted real matrix reloads through the public facade with complete declarations."""
    matrix = build_evidence_gap_matrix(ROOT / "validation/physics_traceability.json")
    for model in [matrix, matrix.entries[0], matrix.trackers[0], matrix.work_packages[0]]:
        assert type(model).__module__ == "tools.evidence_gap_matrix"
    stored = tmp_path / "declared-matrix.pickle"
    stored.write_bytes(pickle.dumps(matrix))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle,sys,json;from pathlib import Path;m=pickle.loads(Path(sys.argv[1]).read_bytes());print(json.dumps(m.to_dict(),sort_keys=True))",
            str(stored),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == matrix.to_dict()
    assert pickle.loads(stored.read_bytes()) == matrix


def test_work_package_rendering_retains_real_requirements_and_deduplicates_paths() -> None:
    """Actual full registry rendering preserves first occurrence ordering and package counts."""
    matrix = build_evidence_gap_matrix(ROOT / "validation/physics_traceability.json")
    for package in matrix.work_packages:
        rendered = package.to_dict()
        assert rendered["components"] == [entry.component for entry in package.entries]
        assert rendered["module_paths"] == list(dict.fromkeys(entry.module_path for entry in package.entries))
        assert rendered["claim_admission_requirements"] == list(
            dict.fromkeys(action for entry in package.entries for action in entry.claim_admission_requirements)
        )
        assert package.open_fidelity_gaps == sum(entry.open_fidelity_gap for entry in package.entries)
    assert matrix.to_markdown().endswith("\n")
