# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference-evidence validation campaign tests.
"""Claim-admission and CLI regressions for the reference-evidence campaign."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import scpn_control.control.disruption_predictor as disruption_predictor
from validation.validate_real_shots import (
    build_campaign_report,
    main,
    render_markdown,
    run_campaign,
    validate_disruption,
    validate_transport,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "validate_real_shots.py"


def _passing_lane(evidence_class: str) -> dict[str, object]:
    """Return a minimal computationally successful lane fixture."""
    return {
        "evidence_class": evidence_class,
        "data_provenance_pass": True,
        "computational_pass": True,
    }


def test_help_is_side_effect_free(tmp_path: Path) -> None:
    """Asking for help must not run validation or create output artifacts."""
    json_out = tmp_path / "must-not-exist.json"
    markdown_out = tmp_path / "must-not-exist.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-json",
            str(json_out),
            "--output-markdown",
            str(markdown_out),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "reference-evidence" in completed.stdout.lower()
    assert not json_out.exists()
    assert not markdown_out.exists()
    assert list(tmp_path.iterdir()) == []


def test_mixed_evidence_cannot_admit_real_shot_or_facility_validation() -> None:
    """Green public-reference/synthetic calculations stay below physics claims."""
    report = build_campaign_report(
        {
            "equilibrium_public_reference": _passing_lane("public_reference"),
            "equilibrium_synthetic": _passing_lane("synthetic"),
            "transport_public_reference": _passing_lane("public_reference"),
            "disruption_synthetic": _passing_lane("synthetic"),
        },
        generated_at="2026-08-27T00:00:00+00:00",
        runtime_s=1.25,
    )

    assert report["computational_success"] is True
    assert report["data_provenance_pass"] is True
    assert report["physics_validation_admitted"] is False
    assert report["real_shot_validation_admitted"] is False
    assert report["facility_validation_admitted"] is False
    assert report["public_claim_allowed"] is False
    assert report["production_claim_allowed"] is False
    assert report["evidence_classes"] == ["public_reference", "synthetic"]

    rendered = render_markdown(report)
    assert "Reference-Evidence Validation Report" in rendered
    assert "**Real-shot validation admitted**: **NO**" in rendered
    assert "**Facility validation admitted**: **NO**" in rendered
    assert "OVERALL: PASS" not in rendered


def test_unknown_evidence_class_fails_closed() -> None:
    """An undeclared evidence class cannot enter the serialized campaign."""
    report = build_campaign_report(
        {"equilibrium": _passing_lane("mixed")},
        generated_at="2026-08-27T00:00:00+00:00",
        runtime_s=0.0,
    )

    assert report["computational_success"] is False
    assert report["data_provenance_pass"] is False
    assert report["public_claim_allowed"] is False
    assert report["claim_admission_errors"] == ["equilibrium: unsupported evidence_class 'mixed'"]


def test_repository_campaign_writes_bounded_public_reports(tmp_path: Path) -> None:
    """The real repository CLI path emits the bounded schema and no claim promotion."""
    json_out = tmp_path / "campaign.json"
    markdown_out = tmp_path / "campaign.md"

    exit_code = main(
        [
            "--reference-root",
            str(ROOT / "validation" / "reference_data"),
            "--output-json",
            str(json_out),
            "--output-markdown",
            str(markdown_out),
        ]
    )

    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema"] == "scpn-control.reference-evidence-validation.v1"
    assert report["computational_success"] is True
    assert report["real_shot_validation_admitted"] is False
    assert report["facility_validation_admitted"] is False
    assert report["production_claim_allowed"] is False
    assert "Reference-Evidence Validation Report" in markdown_out.read_text(encoding="utf-8")


def test_missing_reference_root_fails_closed_without_claim_admission(tmp_path: Path) -> None:
    """An absent reference corpus reports missing lanes instead of succeeding."""
    report = run_campaign(tmp_path)

    assert report["data_provenance_pass"] is False
    assert report["computational_success"] is False
    assert report["real_shot_validation_admitted"] is False
    assert report["lanes"]["transport_public_reference"]["error"] == "ITPA CSV not found"
    assert report["lanes"]["disruption_synthetic"]["error"] == "No disruption NPZ files"


def test_empty_transport_reference_fails_provenance(tmp_path: Path) -> None:
    """A header-only public-reference table cannot pass the transport lane."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("Ip_MA,BT_T,ne19_1e19m3,Ploss_MW,R_m,a_m,kappa,M_AMU,tau_E_s\n", encoding="utf-8")

    report = validate_transport(csv_path)

    assert report["n_shots"] == 0
    assert report["data_provenance_pass"] is False
    assert report["computational_pass"] is False


def test_disruption_lane_covers_no_data_and_missing_signal(tmp_path: Path) -> None:
    """Missing files and signal arrays fail provenance without exceptions."""
    empty_report = validate_disruption(tmp_path)
    assert empty_report["computational_pass"] is False

    np.savez(tmp_path / "missing-signal.npz", is_disruption=False)
    report = validate_disruption(tmp_path)
    assert report["data_provenance_pass"] is False
    assert report["shots"][0]["error"] == "No signal data"


def test_disruption_lane_counts_false_negative_and_false_positive(tmp_path: Path, monkeypatch) -> None:
    """Undetected disruptions and detected safe shots enter separate confusion cells."""
    signal = np.zeros(130, dtype=np.float64)
    np.savez(
        tmp_path / "disruption.npz",
        is_disruption=True,
        disruption_time_idx=129,
        dBdt_gauss_per_s=signal,
    )
    monkeypatch.setattr(disruption_predictor, "predict_disruption_risk", lambda _window, _toroidal: 0.0)
    false_negative = validate_disruption(tmp_path)
    assert false_negative["false_negatives"] == 1

    (tmp_path / "disruption.npz").unlink()
    np.savez(tmp_path / "safe.npz", is_disruption=False, dBdt_gauss_per_s=signal)
    monkeypatch.setattr(disruption_predictor, "predict_disruption_risk", lambda _window, _toroidal: 1.0)
    false_positive = validate_disruption(tmp_path)
    assert false_positive["false_positives"] == 1


def test_disruption_detection_without_time_axis_uses_declared_fallback(tmp_path: Path, monkeypatch) -> None:
    """A detected disruption without timestamps uses the documented index scale."""
    signal = np.zeros(132, dtype=np.float64)
    np.savez(
        tmp_path / "disruption.npz",
        is_disruption=True,
        disruption_time_idx=131,
        dBdt_gauss_per_s=signal,
    )
    monkeypatch.setattr(disruption_predictor, "predict_disruption_risk", lambda _window, _toroidal: 1.0)

    report = validate_disruption(tmp_path)

    assert report["true_positives"] == 1
    assert report["shots"][0]["detection_lead_ms"] == 9.0


def test_unlabelled_disruption_time_does_not_enter_confusion_counts(tmp_path: Path, monkeypatch) -> None:
    """A disruption-labelled fixture without a valid event index stays unscored."""
    signal = np.zeros(130, dtype=np.float64)
    np.savez(
        tmp_path / "unscored.npz",
        is_disruption=True,
        disruption_time_idx=-1,
        dBdt_gauss_per_s=signal,
    )
    monkeypatch.setattr(disruption_predictor, "predict_disruption_risk", lambda _window, _toroidal: 0.0)

    report = validate_disruption(tmp_path)

    assert report["n_disruptions"] == 0
    assert report["n_safe"] == 0


def test_explicit_real_admissions_are_all_required() -> None:
    """Even real evidence promotes only when every independent admission is explicit."""
    lane = {
        "evidence_class": "real",
        "data_provenance_pass": True,
        "computational_pass": True,
        "physics_validation_admitted": True,
        "real_shot_validation_admitted": True,
        "facility_validation_admitted": True,
        "public_claim_allowed": True,
        "production_claim_allowed": True,
    }
    report = build_campaign_report({"measured": lane}, generated_at="2026-08-27T00:00:00Z", runtime_s=0.0)

    assert report["physics_validation_admitted"] is True
    assert report["real_shot_validation_admitted"] is True
    assert report["facility_validation_admitted"] is True
    assert report["public_claim_allowed"] is True
    assert report["production_claim_allowed"] is True
