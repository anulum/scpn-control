# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity validation tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_jax_gk_parity import validate_jax_gk_parity


def _valid_parity_report() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "case": "cyclone_base_case",
        "backend": "cpu",
        "jax_version": "0.5.0",
        "jaxlib_version": "0.5.0",
        "platform": "linux-x86_64",
        "device_kind": "cpu",
        "dtype": "float32",
        "x64_enabled": False,
        "executed_at": "2026-05-18T06:30:00Z",
        "native_gamma_max_cs_over_a": 0.21,
        "jax_gamma_max_cs_over_a": 0.209,
        "native_omega_r_cs_over_a": -0.42,
        "jax_omega_r_cs_over_a": -0.419,
        "gamma_relative_tolerance": 0.05,
        "omega_absolute_tolerance": 0.02,
    }


def test_strict_jax_parity_gate_requires_persisted_artifacts(tmp_path: Path) -> None:
    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["parity_artifacts"] == 0
    assert report["errors"][0]["error"] == "no JAX GK parity artifacts found"


def test_jax_parity_gate_accepts_backend_metadata_and_tolerances(tmp_path: Path) -> None:
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(_valid_parity_report()), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "pass"
    assert report["parity_artifacts"] == 1
    assert report["entries"][0]["case"] == "cyclone_base_case"
    assert report["entries"][0]["backend"] == "cpu"


def test_jax_parity_gate_rejects_missing_backend_metadata(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["jaxlib_version"] = ""
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "jaxlib_version"


def test_jax_parity_gate_rejects_out_of_tolerance_artifact(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["jax_gamma_max_cs_over_a"] = 0.1
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "gamma_max_cs_over_a"
