# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption predictor claim-boundary tests.
"""Claim-boundary tests for the disruption-risk heuristic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_control._typing import FloatArray
import scpn_control.control.disruption_predictor as disruption_module
from scpn_control.control.disruption_predictor import (
    DISRUPTION_FEATURE_CONTRACT,
    DISRUPTION_HEURISTIC_SCORE_SOURCE,
    DisruptionRiskClaimBoundary,
    _linear_percentile,
    _perturb_toroidal_observables,
    build_disruption_feature_vector,
    disruption_risk_claim_boundary,
    load_or_train_predictor,
    predict_disruption_risk_safe,
    run_anomaly_alarm_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalise_text(text: str) -> str:
    """Collapse rendered Markdown or JSON text to single-space content."""

    return " ".join(text.split())


def _claim_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract the disruption claim-boundary metadata from a model response."""

    claim_boundary = metadata.get("claim_boundary")
    assert isinstance(claim_boundary, dict)
    return cast(dict[str, Any], claim_boundary)


def test_feature_contract_matches_boundary_metadata() -> None:
    """Feature construction and metadata expose the same ordered contract."""

    features = build_disruption_feature_vector(
        np.array([0.1, 0.2, 0.5], dtype=np.float64),
        {"toroidal_n1_amp": 0.3, "toroidal_n2_amp": 0.1},
    )
    boundary = disruption_risk_claim_boundary()
    metadata = boundary.to_metadata()

    assert features.shape == (len(DISRUPTION_FEATURE_CONTRACT),)
    assert boundary.feature_contract == DISRUPTION_FEATURE_CONTRACT
    assert metadata["feature_contract"] == list(DISRUPTION_FEATURE_CONTRACT)
    assert metadata["score_source"] == DISRUPTION_HEURISTIC_SCORE_SOURCE
    assert metadata["public_claim_allowed"] is False
    assert metadata["facility_roc_validated"] is False


def test_safe_fallback_metadata_carries_claim_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit fallback inference returns the no-real-ROC claim boundary."""
    import scpn_control.control.disruption_checkpoint as checkpoint_mod

    # load_or_train lives on the checkpoint leaf after R7-S4.
    monkeypatch.setattr(checkpoint_mod, "torch", None)
    monkeypatch.setattr(disruption_module, "torch", None)

    risk, metadata = predict_disruption_risk_safe(
        np.ones(16, dtype=np.float64),
        allow_legacy_fallback=True,
    )
    claim_boundary = _claim_metadata(metadata)

    assert 0.0 <= risk <= 1.0
    assert metadata["mode"] == "fallback"
    assert metadata["risk_source"] == "predict_disruption_risk"
    assert claim_boundary["public_claim_allowed"] is False
    assert claim_boundary["facility_roc_validated"] is False
    assert "no_real_disruption_database_fit" in str(claim_boundary["training_provenance"])


def test_claim_boundary_rejects_facility_claim_widening() -> None:
    """Boundary construction fails if a caller widens the real-database claim."""

    boundary = disruption_risk_claim_boundary()
    with pytest.raises(ValueError, match="public_claim_allowed"):
        DisruptionRiskClaimBoundary(
            predictor_id=boundary.predictor_id,
            score_source=boundary.score_source,
            feature_contract=boundary.feature_contract,
            training_provenance=boundary.training_provenance,
            validation_provenance=boundary.validation_provenance,
            public_claim_allowed=True,
            facility_roc_validated=False,
            required_action=boundary.required_action,
        )
    with pytest.raises(ValueError, match="facility_roc_validated"):
        DisruptionRiskClaimBoundary(
            predictor_id=boundary.predictor_id,
            score_source=boundary.score_source,
            feature_contract=boundary.feature_contract,
            training_provenance=boundary.training_provenance,
            validation_provenance=boundary.validation_provenance,
            public_claim_allowed=False,
            facility_roc_validated=True,
            required_action=boundary.required_action,
        )


def test_claim_boundary_rejects_empty_contract_fields() -> None:
    """Boundary construction rejects empty identifiers and feature names."""

    boundary = disruption_risk_claim_boundary()
    with pytest.raises(ValueError, match="predictor_id"):
        DisruptionRiskClaimBoundary(
            predictor_id=" ",
            score_source=boundary.score_source,
            feature_contract=boundary.feature_contract,
            training_provenance=boundary.training_provenance,
            validation_provenance=boundary.validation_provenance,
            public_claim_allowed=False,
            facility_roc_validated=False,
            required_action=boundary.required_action,
        )
    with pytest.raises(ValueError, match="feature_contract"):
        DisruptionRiskClaimBoundary(
            predictor_id=boundary.predictor_id,
            score_source=boundary.score_source,
            feature_contract=("mean", ""),
            training_provenance=boundary.training_provenance,
            validation_provenance=boundary.validation_provenance,
            public_claim_allowed=False,
            facility_roc_validated=False,
            required_action=boundary.required_action,
        )


def test_feature_builder_rejects_contract_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature construction fails if the declared contract length is tampered."""

    import scpn_control.control.disruption_physics_proxies as proxies_module
    import scpn_control.control.disruption_risk_claims as claims_module

    # Contract is owned by the claims leaf and consumed by physics proxies.
    monkeypatch.setattr(claims_module, "DISRUPTION_FEATURE_CONTRACT", ("mean",))
    monkeypatch.setattr(proxies_module, "DISRUPTION_FEATURE_CONTRACT", ("mean",))
    monkeypatch.setattr(disruption_module, "DISRUPTION_FEATURE_CONTRACT", ("mean",))
    with pytest.raises(RuntimeError, match="feature vector length"):
        build_disruption_feature_vector(np.ones(3, dtype=np.float64))


def test_local_percentile_handles_edges_without_numpy_reductions() -> None:
    """Local percentile helper handles singleton, interpolation, and empty inputs."""

    assert _linear_percentile([2.0], 95.0) == 2.0
    assert _linear_percentile([4.0, 8.0], 0.0) == 4.0
    assert _linear_percentile([0.0, 10.0], 50.0) == 5.0
    with pytest.raises(ValueError, match="at least one sample"):
        _linear_percentile([], 95.0)


def test_toroidal_perturbation_scales_observables() -> None:
    """Toroidal sigma-point perturbation scales every supplied observable."""

    perturbed = _perturb_toroidal_observables({"toroidal_n1_amp": 2.0, "toroidal_n2_amp": 1.0}, 1.0)

    assert perturbed is not None
    assert perturbed["toroidal_n1_amp"] == pytest.approx(2.16)
    assert perturbed["toroidal_n2_amp"] == pytest.approx(1.08)


def test_anomaly_campaign_counts_false_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe synthetic shots that alarm are counted as false positives."""

    def mock_safe_tearing(
        steps: int = 1000,
        *,
        rng: np.random.Generator | None = None,
        mode: str = "ntm",
    ) -> tuple[FloatArray, int, int]:
        del rng, mode
        return np.ones(steps, dtype=np.float64) * 12.0, 0, -1

    monkeypatch.setattr(disruption_module, "simulate_tearing_mode", mock_safe_tearing)

    result = run_anomaly_alarm_campaign(seed=7, episodes=2, window=32, threshold=0.01)

    assert result["false_positive_rate"] == 1.0
    assert result["true_positive_rate"] == 0.0


def test_load_or_train_predictor_rejects_implicit_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback loading requires both explicit fallback switches."""
    import scpn_control.control.disruption_checkpoint as checkpoint_mod

    monkeypatch.setattr(checkpoint_mod, "torch", None)
    monkeypatch.setattr(disruption_module, "torch", None)
    with pytest.raises(ValueError, match="allow_legacy_fallback"):
        load_or_train_predictor(allow_fallback=True)
    with pytest.raises(RuntimeError, match="Torch is required"):
        load_or_train_predictor(allow_fallback=False)


def test_public_surfaces_keep_disruption_predictor_boundary() -> None:
    """Docs and studio evidence retain the synthetic-only disruption boundary."""

    readme = _normalise_text((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    competitive = _normalise_text((REPO_ROOT / "docs" / "competitive_analysis.md").read_text(encoding="utf-8"))
    traceability = _normalise_text((REPO_ROOT / "docs" / "physics_traceability.md").read_text(encoding="utf-8"))
    deficiencies = _normalise_text((REPO_ROOT / "docs" / "validation_deficiencies.md").read_text(encoding="utf-8"))
    studio_evidence = _normalise_text(
        (REPO_ROOT / "src" / "scpn_control" / "studio" / "evidence.py").read_text(encoding="utf-8")
    )

    assert "Default score is a deterministic fixed-weight heuristic" in readme
    assert "Not validated on experimental disruption databases" in readme
    assert "Disruption prediction (heuristic)" in competitive
    assert "Disruption prediction (ML)" not in competitive
    assert "not a model trained or fitted on a real disruption database" in traceability
    assert "Synthetic-only heuristic replay" in deficiencies
    assert "fixed-weight baseline" in studio_evidence
    assert "ROC-validated" in studio_evidence
    assert "real disruption database" in studio_evidence
