# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ITPA Reference Generator

"""Generate the checked-in ITPA gyro-Bohm reference payload."""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE_LAST_MODIFIED = "2026-06-03"


def generate_gyro_bohm_reference() -> None:
    """Generate ITPA Gyro-Bohm scaling coefficients JSON reference data."""
    target_dir = Path("validation/reference_data/itpa")
    target_file = target_dir / "gyro_bohm_coefficients.json"
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "c_gB": 0.0424,
        "metadata": {
            "itpa_version": "2026.1",
            "confinement_regime_targets": ["H-mode", "J-mode"],
            "last_modified": REFERENCE_LAST_MODIFIED,
        },
        "scaling_parameters": {
            "c_gB_nominal": 0.0424,
            "alpha_Te": 1.5,
            "alpha_B": -2.0,
            "normalized_radius_bounds": {
                "rho_tor_min": 0.1,
                "rho_tor_max": 0.95,
            },
        },
        "thresholds": {
            "h_mode_transition_critical_gradient": 12.85,
            "j_mode_suppression_factor": 0.761,
        },
    }

    with target_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    print(f"SUCCESS: Generated ITPA reference data at -> {target_file.resolve()}")


if __name__ == "__main__":
    generate_gyro_bohm_reference()
