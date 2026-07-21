# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent ballooning cross-validation tests
"""Offline tests for :mod:`validation.validate_stability_mhd_ballooning`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation import validate_stability_mhd_ballooning as mod


def test_crosscheck_passes_on_production() -> None:
    report = mod.validate_stability_mhd_ballooning()
    assert report["status"] == "pass"
    assert report["cases"] == len(mod._SHEAR_VALUES) == 5
    assert report["findings"] == []
    assert report["public_claims"]["bounded_ballooning_boundary_independently_verified"] is True
    assert report["public_claims"]["ideal_mhd_stability_public_claim"] is False


def test_analytic_tracks_numerical_within_tolerance() -> None:
    report = mod.validate_stability_mhd_ballooning()
    # The analytic fit stays within a few percent of the numerical boundary.
    assert report["max_relative_error"] < mod._REL_TOLERANCE
    assert report["max_relative_error"] < 0.06
    for entry in report["entries"]:
        assert entry["agrees"] is True
        assert entry["alpha_crit_numeric"] > 0.0
        # unit-shear numerical marginal boundary is ~0.61
        if entry["shear"] == 1.0:
            assert entry["alpha_crit_numeric"] == pytest.approx(0.612, abs=0.02)
            assert entry["alpha_crit_analytic"] == pytest.approx(0.6, abs=1e-9)


def test_module_alpha_crit_reads_production_formula() -> None:
    # The analytic branch: s(1-s/2) for s<1, 0.6 s for s>=1.
    values = mod._module_alpha_crit((0.5, 1.0, 2.0))
    assert values[0] == pytest.approx(0.5 * (1 - 0.25))
    assert values[1] == pytest.approx(0.6)
    assert values[2] == pytest.approx(1.2)


def test_crosscheck_detects_corrupted_analytic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_SHEAR_VALUES", (0.7, 1.0, 1.5))  # >=3 for a valid q-profile
    real = mod._module_alpha_crit

    def corrupt(shear_values: tuple[float, ...]) -> list[float]:
        return [3.0 * v for v in real(shear_values)]  # gross overestimate

    monkeypatch.setattr(mod, "_module_alpha_crit", corrupt)
    report = mod.validate_stability_mhd_ballooning()
    assert report["status"] == "fail"
    assert any("diverges from numerical" in f for f in report["findings"])
    assert report["public_claims"]["bounded_ballooning_boundary_independently_verified"] is False


def test_payload_sha256_is_deterministic() -> None:
    a = mod.validate_stability_mhd_ballooning()["payload_sha256"]
    b = mod.validate_stability_mhd_ballooning()["payload_sha256"]
    assert a == b and isinstance(a, str) and len(a) == 64


def test_report_schema_and_metadata() -> None:
    report = mod.validate_stability_mhd_ballooning()
    assert report["schema_version"] == "scpn-control.stability-ballooning-independent-crosscheck.v1"
    assert report["model_reference"].startswith("Connor")
    assert report["tolerance"] == {"relative": mod._REL_TOLERANCE}


def test_main_default_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "pass" in out and "cases=5" in out


def test_main_json_out(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main(["--json-out"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"


def test_main_writes_output_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "ballooning_crosscheck.json"
    assert mod.main(["--output-json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["cases"] == 5
    assert payload["payload_sha256"] is not None


def test_main_returns_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(mod, "_SHEAR_VALUES", (0.7, 1.0, 1.5))
    real = mod._module_alpha_crit
    monkeypatch.setattr(mod, "_module_alpha_crit", lambda sv: [5.0 * v for v in real(sv)])
    assert mod.main([]) == 1
    assert "FINDING" in capsys.readouterr().err
