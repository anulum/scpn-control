# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Miller geometry cross-validation tests
"""Offline tests for :mod:`validation.validate_gk_geometry_independent`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation import validate_gk_geometry_independent as mod


def test_crosscheck_passes_on_production() -> None:
    report = mod.validate_gk_geometry_independent()
    assert report["status"] == "pass"
    assert report["cases"] == len(mod._CASES) == 5
    assert report["findings"] == []
    assert report["public_claims"]["bounded_local_miller_geometry_independently_verified"] is True


def test_every_case_agrees_within_tolerance() -> None:
    report = mod.validate_gk_geometry_independent()
    active = {e["case"]: e for e in report["entries"]}
    assert {"positive_shaping_shear", "negative_shaping_shear"} <= set(active)
    for entry in report["entries"]:
        assert entry["status"] == "pass"
        for field in mod._COMPARE_FIELDS:
            result = entry["fields"][field]
            assert result["agrees"] is True
            assert result["max_abs_error"] < 1e-6
    # The two shaping-shear cases must be flagged as exercising s_kappa/s_delta.
    assert active["positive_shaping_shear"]["shaping_shear_active"] is True
    assert active["circular_cyclone_limit"]["shaping_shear_active"] is False


def test_field_agreement_flags_mismatch() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    perturbed = ref + np.array([0.0, 0.0, 0.5])
    max_abs, max_rel, agrees = mod._field_agreement(perturbed, ref)
    assert agrees is False
    assert max_abs == pytest.approx(0.5)
    assert max_rel == pytest.approx(0.5 / 3.0)


def test_field_agreement_all_zero_reference() -> None:
    ref = np.zeros(4)
    max_abs, max_rel, agrees = mod._field_agreement(ref, ref)
    assert agrees is True
    assert max_abs == 0.0
    assert max_rel == 0.0


def test_crosscheck_detects_corrupted_production(monkeypatch: pytest.MonkeyPatch) -> None:
    real = mod.miller_geometry

    def corrupt(**kwargs: object) -> object:
        geom = real(**kwargs)
        geom.g_rr = geom.g_rr * 1.5  # inject a gross metric error
        return geom

    monkeypatch.setattr(mod, "miller_geometry", corrupt)
    report = mod.validate_gk_geometry_independent()
    assert report["status"] == "fail"
    assert any("g_rr" in f for f in report["findings"])
    assert report["public_claims"]["bounded_local_miller_geometry_independently_verified"] is False
    assert any(e["status"] == "fail" for e in report["entries"])


def test_payload_sha256_is_deterministic() -> None:
    a = mod.validate_gk_geometry_independent()["payload_sha256"]
    b = mod.validate_gk_geometry_independent()["payload_sha256"]
    assert a == b and isinstance(a, str) and len(a) == 64


def test_report_schema_and_metadata() -> None:
    report = mod.validate_gk_geometry_independent()
    assert report["schema_version"] == "scpn-control.gk-geometry-independent-crosscheck.v1"
    assert report["model_reference"].startswith("Miller")
    assert report["tolerances"] == {"absolute": mod._ABS_TOLERANCE, "relative": mod._REL_TOLERANCE}
    assert set(report["units"]) == set(mod._COMPARE_FIELDS)  # all compared fields carry a unit


def test_main_default_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "pass" in out and "cases=5" in out


def test_main_json_out(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main(["--json-out"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"


def test_main_writes_output_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "crosscheck.json"
    assert mod.main(["--output-json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["cases"] == 5
    assert payload["payload_sha256"] is not None


def test_main_returns_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    real = mod.miller_geometry

    def corrupt(**kwargs: object) -> object:
        geom = real(**kwargs)
        geom.g_tt = geom.g_tt + 1.0
        return geom

    monkeypatch.setattr(mod, "miller_geometry", corrupt)
    assert mod.main([]) == 1
    assert "FINDING" in capsys.readouterr().err
