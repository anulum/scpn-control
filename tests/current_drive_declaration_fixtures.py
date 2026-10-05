# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Typed declaration fixtures for current-drive command tests

"""Caller-authored current-drive test metadata; no authenticated physical reference or computation."""

from __future__ import annotations


def declaration() -> dict[str, object]:
    """Return fresh caller-authored metadata for JSON command boundaries; no solver, external artifact or facility evidence."""
    payload: dict[str, object] = {
        "source": "ray_tracing_benchmark",
        "reference_dataset_id": "current-drive-ray-tracing-fixture-v1",
        "reference_artifact_sha256": "d" * 64,
        "reference_case_count": 2,
        "units": {
            "power": "W",
            "current": "A",
            "current_density": "A/m^2",
            "density": "10^19 m^-3",
            "temperature": "keV",
            "rho": "1",
            "time": "s",
            "energy": "keV",
        },
        "metrics": {
            "total_power_relative_error": 0.01,
            "total_current_relative_error": 0.03,
            "deposition_centroid_abs_error": 0.01,
            "peak_current_density_relative_error": 0.05,
            "nbi_slowing_down_relative_error": 0.02,
        },
        "tolerances": {
            "total_power_relative_error": 0.03,
            "total_current_relative_error": 0.1,
            "deposition_centroid_abs_error": 0.05,
            "peak_current_density_relative_error": 0.15,
            "nbi_slowing_down_relative_error": 0.1,
        },
    }
    payload.update(
        schema_version="1.0",
        model_id="DECLARATION_ONLY_TEST",
        model_version="test",
        executed_at="DECLARATION_ONLY_TEST",
        external_code="TORBEAM",
        reference_artifact_uri="UNAUTHENTICATED_DECLARATION_ONLY_TEST",
        source_metadata={"total_power_W": 1.0, "rho_points": 2.5, "rho_min": 0.1, "rho_max": 1.0},
    )
    return payload
