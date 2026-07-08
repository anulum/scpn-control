# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence claim-evidence COV-1 tests.
"""Tests for neural-turbulence reference admission guard behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_control.core.neural_turbulence import neural_turbulence_claim_evidence


def _validation_result() -> dict[str, Any]:
    """Return the minimal local validation metrics needed to build evidence."""
    return {
        "n_samples": 16,
        "Q_i_rmse_gB": 0.04,
        "Q_e_rmse_gB": 0.03,
        "Gamma_e_rmse_gB": 0.02,
        "flux_relative_mae": 0.05,
        "critical_gradient_accuracy": 0.94,
    }


def _write_reference_artifact(
    tmp_path: Path,
    *,
    source: str = "documented_public_reference",
    sample_count: int = 96,
) -> tuple[Path, Path]:
    """Write a reference artifact and its matching weights fixture."""
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"bounded neural turbulence weights")
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "source": source,
        "model_id": "neural_turbulence_qlknn_facade",
        "model_version": "cov1",
        "trained_weights_sha256": weights_sha256,
        "reference_dataset_id": "bounded-gk-fixture",
        "reference_artifact_sha256": "e" * 64,
        "executed_at": "2026-05-31T00:00:00Z",
        "reference_url": "https://example.invalid/gk-reference",
        "feature_schema": [
            "R_LTi",
            "R_LTe",
            "R_Ln",
            "q",
            "s_hat",
            "alpha_MHD",
            "Ti_Te",
            "nu_star",
            "Z_eff",
            "epsilon",
        ],
        "units": {
            "Q_i": "gyroBohm",
            "Q_e": "gyroBohm",
            "Gamma_e": "gyroBohm",
            "input_gradients": "dimensionless",
        },
        "reference_sample_count": sample_count,
        "metrics": {
            "Q_i_rmse_gB": 0.04,
            "Q_e_rmse_gB": 0.03,
            "Gamma_e_rmse_gB": 0.02,
            "flux_relative_mae": 0.05,
            "critical_gradient_accuracy": 0.94,
        },
        "tolerances": {
            "Q_i_rmse_gB": 0.08,
            "Q_e_rmse_gB": 0.08,
            "Gamma_e_rmse_gB": 0.06,
            "flux_relative_mae": 0.10,
            "critical_gradient_accuracy_min": 0.90,
        },
    }
    artifact = tmp_path / "reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    return weights, artifact


def _force_reference_validator_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force validator success so claim-evidence defence-in-depth guards run."""

    def passing_validator(_artifact_path: Path, *, require_reference_artifacts: bool) -> dict[str, object]:
        assert require_reference_artifacts is True
        return {"status": "pass"}

    monkeypatch.setattr(
        "validation.validate_neural_turbulence_reference.validate_neural_turbulence_reference",
        passing_validator,
    )


def test_claim_evidence_rejects_inadmissible_reference_source_after_validator_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim boundary still rejects unsupported sources if validation is relaxed."""
    _force_reference_validator_pass(monkeypatch)
    weights, artifact = _write_reference_artifact(tmp_path, source="synthetic")

    with pytest.raises(ValueError, match="source is not admissible"):
        neural_turbulence_claim_evidence(
            _validation_result(),
            source="documented_public_reference",
            source_id="tests/test_neural_turbulence_claim_evidence_cov1.py::source",
            weights_path=weights,
            reference_artifact_path=artifact,
        )


def test_claim_evidence_rejects_nonpositive_reference_sample_count_after_validator_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim boundary still rejects zero samples if validation is relaxed."""
    _force_reference_validator_pass(monkeypatch)
    weights, artifact = _write_reference_artifact(tmp_path, sample_count=0)

    with pytest.raises(ValueError, match="reference_sample_count must be positive"):
        neural_turbulence_claim_evidence(
            _validation_result(),
            source="documented_public_reference",
            source_id="tests/test_neural_turbulence_claim_evidence_cov1.py::sample_count",
            weights_path=weights,
            reference_artifact_path=artifact,
        )
