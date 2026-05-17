# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — validation reference configuration tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for documented validation reference configurations."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_README = REPO_ROOT / "validation" / "reference_data" / "README.md"


def _documented_top_level_validation_configs() -> list[Path]:
    text = REFERENCE_README.read_text(encoding="utf-8")
    matches = re.findall(r"`\.\./([^`]+\.json)`", text)
    return [REPO_ROOT / "validation" / name for name in matches]


def test_documented_top_level_validation_configs_exist() -> None:
    configs = _documented_top_level_validation_configs()

    assert configs, "reference-data README should document top-level validation configs"
    missing = [path.name for path in configs if not path.is_file()]
    assert missing == []


def test_documented_iter_configs_have_solver_schema() -> None:
    for path in _documented_top_level_validation_configs():
        data = json.loads(path.read_text(encoding="utf-8"))

        assert isinstance(data["reactor_name"], str)
        assert len(data["grid_resolution"]) == 2
        assert all(int(n) >= 3 for n in data["grid_resolution"])
        assert {"R_min", "R_max", "Z_min", "Z_max"} <= set(data["dimensions"])
        assert data["dimensions"]["R_min"] < data["dimensions"]["R_max"]
        assert data["dimensions"]["Z_min"] < data["dimensions"]["Z_max"]
        assert data["physics"]["plasma_current_target"] > 0.0
        assert data["coils"], f"{path.name} must define at least one coil"
        assert {"max_iterations", "convergence_threshold"} <= set(data["solver"])
