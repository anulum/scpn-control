# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Fokker-Planck collision cross-validation tests
"""Offline tests for :mod:`validation.validate_gk_collision_independent`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation import validate_gk_collision_independent as mod


def test_crosscheck_passes_on_production() -> None:
    report = mod.validate_gk_collision_independent()
    assert report["status"] == "pass"
    assert report["cases"] == len(mod._CASES) == 5
    assert report["findings"] == []
    assert report["public_claims"]["bounded_collision_scaling_independently_verified"] is True
    assert report["public_claims"]["quantitative_collisional_damping"] is False


def test_scaling_is_exactly_constant_across_cases() -> None:
    report = mod.validate_gk_collision_independent()
    agg = report["aggregate"]
    # A constant production/reference ratio across density, temperature, Z_eff,
    # and species mass proves the functional scaling matches to machine precision.
    assert agg["deflection_ratio_spread"] < 1e-12
    assert agg["braginskii_ratio_spread"] < 1e-12
    assert agg["scaling_exact"] is True
    # The bounded O(1) prefactors match the pinned physics.
    assert agg["deflection_prefactor"] == pytest.approx(0.2710231582, rel=1e-6)
    assert agg["braginskii_prefactor"] == pytest.approx(0.5, rel=1e-9)
    assert agg["prefactor_bounded"] is True


def test_energy_channel_matches_elastic_identity() -> None:
    report = mod.validate_gk_collision_independent()
    for entry in report["entries"]:
        assert entry["energy_structure_exact"] is True
        assert entry["energy_structure_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_spread_handles_zero_mean() -> None:
    assert mod._spread([0.0, 0.0]) == float("inf")
    assert mod._spread([2.0, 2.0]) == pytest.approx(0.0)


def test_crosscheck_detects_broken_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    real = mod.collision_frequencies

    def scaling_break(species: object, *args: object, **kwargs: object) -> tuple[float, float]:
        nu_d, nu_e = real(species, *args, **kwargs)
        # Inject a temperature-dependent factor (args[1] is T_e_keV) so the
        # production/reference ratio is no longer constant across cases — a
        # genuine scaling break the constancy test must catch.
        scale = 1.0 + float(args[1])  # type: ignore[arg-type]
        return nu_d * scale, nu_e * scale

    monkeypatch.setattr(mod, "collision_frequencies", scaling_break)
    report = mod.validate_gk_collision_independent()
    assert report["status"] == "fail"
    assert report["aggregate"]["scaling_exact"] is False
    assert any("functional scaling diverges" in f for f in report["findings"])
    assert report["public_claims"]["bounded_collision_scaling_independently_verified"] is False


def test_crosscheck_detects_out_of_band_prefactor(monkeypatch: pytest.MonkeyPatch) -> None:
    real = mod.collision_frequencies

    def inflated(species: object, *args: object, **kwargs: object) -> tuple[float, float]:
        nu_d, nu_e = real(species, *args, **kwargs)
        # Constant blow-up keeps scaling exact but pushes the prefactor out of band.
        return nu_d * 100.0, nu_e * 100.0

    monkeypatch.setattr(mod, "collision_frequencies", inflated)
    report = mod.validate_gk_collision_independent()
    assert report["status"] == "fail"
    assert report["aggregate"]["scaling_exact"] is True
    assert report["aggregate"]["prefactor_bounded"] is False
    assert any("outside the documented" in f for f in report["findings"])


def test_crosscheck_detects_broken_energy_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    real = mod.collision_frequencies

    def broken_energy(species: object, *args: object, **kwargs: object) -> tuple[float, float]:
        nu_d, nu_e = real(species, *args, **kwargs)
        return nu_d, nu_e * 1.5  # break the elastic energy-transfer identity

    monkeypatch.setattr(mod, "collision_frequencies", broken_energy)
    report = mod.validate_gk_collision_independent()
    assert report["status"] == "fail"
    assert any("energy-relaxation channel deviates" in f for f in report["findings"])
    assert all(not e["energy_structure_exact"] for e in report["entries"])


def test_payload_sha256_is_deterministic() -> None:
    a = mod.validate_gk_collision_independent()["payload_sha256"]
    b = mod.validate_gk_collision_independent()["payload_sha256"]
    assert a == b and isinstance(a, str) and len(a) == 64


def test_report_schema_and_metadata() -> None:
    report = mod.validate_gk_collision_independent()
    assert report["schema_version"] == "scpn-control.gk-collision-independent-crosscheck.v1"
    assert "Helander" in report["model_reference"]
    assert report["tolerances"]["prefactor_band"] == list(mod._PREFACTOR_BAND)
    assert set(report["units"]) >= {"deflection_ratio", "braginskii_ratio", "energy_structure_ratio"}


def test_main_default_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "pass" in out and "cases=5" in out


def test_main_json_out(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main(["--json-out"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"


def test_main_writes_output_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "collision_crosscheck.json"
    assert mod.main(["--output-json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["cases"] == 5
    assert payload["payload_sha256"] is not None


def test_main_returns_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    real = mod.collision_frequencies

    def broken(species: object, *args: object, **kwargs: object) -> tuple[float, float]:
        nu_d, nu_e = real(species, *args, **kwargs)
        return nu_d, nu_e * 2.0

    monkeypatch.setattr(mod, "collision_frequencies", broken)
    assert mod.main([]) == 1
    assert "FINDING" in capsys.readouterr().err
