# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption ensemble tests
"""Exercise the public disruption ensemble and evidence API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_control.control.halo_re_physics import (
    DisruptionMitigationClaimEvidence,
    assert_disruption_mitigation_claim_admissible,
    disruption_mitigation_claim_evidence,
    run_disruption_ensemble,
    save_disruption_mitigation_claim_evidence,
)


class TestDisruptionEnsemble:
    """Check ensemble determinism and evidence admission."""

    def test_basic_run(self) -> None:
        """Basic run."""
        report = run_disruption_ensemble(ensemble_runs=5, seed=42)
        assert report.ensemble_runs == 5
        assert 0.0 <= report.prevention_rate <= 1.0
        assert len(report.per_run_details) == 5

    def test_all_details_have_required_keys(self) -> None:
        """All details have required keys."""
        report = run_disruption_ensemble(ensemble_runs=3, seed=0)
        for d in report.per_run_details:
            assert "halo_peak_ma" in d
            assert "re_peak_ma" in d
            assert "prevented" in d
            assert "tpf_product" in d

    def test_rejects_zero_runs(self) -> None:
        """Rejects zero runs."""
        with pytest.raises(ValueError, match="ensemble_runs"):
            run_disruption_ensemble(ensemble_runs=0)

    def test_reproducibility(self) -> None:
        """Reproducibility."""
        r1 = run_disruption_ensemble(ensemble_runs=5, seed=123)
        r2 = run_disruption_ensemble(ensemble_runs=5, seed=123)
        assert r1.prevention_rate == r2.prevention_rate
        assert r1.mean_halo_peak_ma == pytest.approx(r2.mean_halo_peak_ma)

    def test_halo_peaks_finite(self) -> None:
        """Halo peaks finite."""
        report = run_disruption_ensemble(ensemble_runs=10, seed=42)
        assert np.isfinite(report.mean_halo_peak_ma)
        assert np.isfinite(report.p95_halo_peak_ma)
        assert np.isfinite(report.mean_re_peak_ma)
        assert np.isfinite(report.p95_re_peak_ma)

    def test_claim_evidence_records_bounded_ensemble_boundary(self, tmp_path: Path) -> None:
        """Claim evidence records bounded ensemble boundary."""
        report = run_disruption_ensemble(ensemble_runs=4, seed=7)

        evidence = disruption_mitigation_claim_evidence(
            report,
            source="synthetic_regression_reference",
            source_id="tests/test_halo_re_physics.py::bounded_ensemble",
            ensemble_seed=7,
        )

        assert isinstance(evidence, DisruptionMitigationClaimEvidence)
        assert evidence.mitigation_claim_allowed is False
        assert evidence.reference_source == "none"
        assert evidence.ensemble_runs == 4
        assert evidence.ensemble_seed == 7
        assert evidence.prevention_rate == pytest.approx(report.prevention_rate)
        assert evidence.mean_halo_peak_ma == pytest.approx(report.mean_halo_peak_ma)
        assert evidence.p95_re_peak_ma == pytest.approx(report.p95_re_peak_ma)
        with pytest.raises(ValueError, match="blocked without matched reference"):
            assert_disruption_mitigation_claim_admissible(evidence)

        out = tmp_path / "disruption_claim.json"
        save_disruption_mitigation_claim_evidence(evidence, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["claim_status"].startswith("bounded halo/runaway ensemble evidence only")

    def test_reference_metadata_cannot_self_admit_disruption_claim(self, tmp_path: Path) -> None:
        """Reference metadata cannot self admit disruption claim."""
        report = run_disruption_ensemble(ensemble_runs=4, seed=8)
        artifact = tmp_path / "reference.json"
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "source": "documented_public_reference",
            "model_id": "halo_runaway_disruption_mitigation",
            "model_version": "test",
            "reference_dataset_id": "bounded-disruption-reference",
            "reference_artifact_sha256": "a" * 64,
            "executed_at": "2026-05-31T00:00:00Z",
            "reference_url": "https://example.invalid/disruption-reference",
            "reference_case_count": 5,
            "signal_window": {
                "sample_count": 32,
                "sample_period_s": 0.001,
                "pre_disruption_duration_s": 0.05,
                "current_quench_duration_ms": 12.0,
                "thermal_quench_duration_ms": 1.0,
            },
            "mitigation_metadata": {
                "neon_quantity_mol": 0.1,
                "argon_quantity_mol": 0.01,
                "xenon_quantity_mol": 0.0,
                "total_impurity_mol": 0.11,
                "mitigation_strength": 0.8,
                "tbr_reference": 1.0,
            },
            "units": {
                "time": "s",
                "quench_time": "ms",
                "current": "MA",
                "energy": "MJ",
                "impurity_inventory": "mol",
                "risk": "1",
                "tbr": "1",
            },
            "metrics": {
                "risk_after_abs_error": 0.01,
                "detection_lead_time_abs_error_ms": 2.0,
                "halo_current_relative_error": 0.05,
                "runaway_beam_relative_error": 0.04,
                "tbr_abs_error": 0.02,
            },
            "tolerances": {
                "risk_after_abs_error": 0.05,
                "detection_lead_time_abs_error_ms": 5.0,
                "halo_current_relative_error": 0.10,
                "runaway_beam_relative_error": 0.10,
                "tbr_abs_error": 0.05,
            },
        }
        artifact.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="independently verified reference comparison"):
            disruption_mitigation_claim_evidence(
                report,
                source="documented_public_reference",
                source_id="tests/test_halo_re_physics.py::reference_admission",
                ensemble_seed=8,
                reference_artifact_path=artifact,
            )

        payload["metrics"]["halo_current_relative_error"] = 0.50
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="failed strict validation"):
            disruption_mitigation_claim_evidence(
                report,
                source="documented_public_reference",
                source_id="tests/test_halo_re_physics.py::reference_admission",
                ensemble_seed=8,
                reference_artifact_path=artifact,
            )


def test_run_disruption_ensemble_verbose_logging() -> None:
    """Run disruption ensemble verbose logging."""
    report = run_disruption_ensemble(ensemble_runs=2, seed=3, verbose=True)
    assert report.ensemble_runs == 2
