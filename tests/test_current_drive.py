# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Current-drive tests.
"""Module-specific tests for auxiliary current-drive source contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.core.current_drive import (
    CurrentDriveMix,
    ECCDSource,
    LHCDSource,
    NBISource,
    assert_current_drive_external_claim_admissible,
    current_drive_claim_evidence,
    eccd_efficiency,
    nbi_critical_energy,
    nbi_slowing_down_time,
    save_current_drive_claim_evidence,
)


def _profiles(n: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.linspace(0.0, 1.0, n)
    ne = 7.0 - 2.0 * rho**2
    te = 12.0 - 6.0 * rho**1.5
    ti = 10.0 - 5.0 * rho**1.4
    return rho, ne, te, ti


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.ndim != 1 or x_arr.ndim != 1:
        raise ValueError("test trapezoid helper expects one-dimensional arrays")
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("test trapezoid helper values and grid lengths must match")
    if x_arr.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y_arr[1:] + y_arr[:-1]) * np.diff(x_arr)))


def test_current_drive_sources_conserve_grid_normalised_power() -> None:
    rho, ne, te, ti = _profiles()
    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.35, sigma_rho=0.08)
    lhcd = LHCDSource(P_lh_MW=4.0, rho_dep=0.65, sigma_rho=0.12)
    nbi = NBISource(P_nbi_MW=16.0, E_beam_keV=1000.0, rho_tangency=0.45, sigma_rho=0.15)

    assert _trapezoid(eccd.P_absorbed(rho), rho) == pytest.approx(10.0e6)
    assert _trapezoid(lhcd.P_absorbed(rho), rho) == pytest.approx(4.0e6)
    assert _trapezoid(nbi.P_heating(rho), rho) == pytest.approx(16.0e6)

    mix = CurrentDriveMix(a=2.0)
    mix.add_source(eccd)
    mix.add_source(lhcd)
    mix.add_source(nbi)
    assert _trapezoid(mix.total_heating_power(rho), rho) == pytest.approx(30.0e6)
    assert mix.total_driven_current(rho, ne, te, ti) > 0.0


def test_efficiency_and_nbi_scalings_are_physical() -> None:
    eta_low = eccd_efficiency(Te_keV=5.0, Z_eff=2.0, N_parallel=0.6)
    eta_high = eccd_efficiency(Te_keV=15.0, Z_eff=2.0, N_parallel=0.6)
    assert eta_high > eta_low > 0.0

    tau_low_density = nbi_slowing_down_time(Te_keV=10.0, ne_19=2.0)
    tau_high_density = nbi_slowing_down_time(Te_keV=10.0, ne_19=8.0)
    assert tau_low_density > tau_high_density > 0.0
    assert nbi_critical_energy(Te_keV=12.0) > nbi_critical_energy(Te_keV=6.0)


def test_current_drive_rejects_invalid_domains() -> None:
    rho, ne, te, ti = _profiles(8)
    with pytest.raises(ValueError, match="rho_dep must be finite and within"):
        ECCDSource(P_ec_MW=1.0, rho_dep=1.2, sigma_rho=0.1)
    with pytest.raises(ValueError, match="E_beam_keV must be finite and > 0"):
        NBISource(P_nbi_MW=1.0, E_beam_keV=0.0, rho_tangency=0.5)
    with pytest.raises(ValueError, match="rho grid must be strictly increasing"):
        ECCDSource(1.0, 0.5, 0.1).j_cd(np.array([0.0, 0.4, 0.4]), ne[:3], te[:3])
    with pytest.raises(ValueError, match="ne_19 must contain finite positive values"):
        CurrentDriveMix(a=2.0).total_driven_current(rho, np.zeros_like(ne), te, ti)


def test_current_drive_claim_evidence_records_bounded_boundary(tmp_path) -> None:
    rho, ne, te, ti = _profiles()
    mix = CurrentDriveMix(a=2.0)
    mix.add_source(ECCDSource(P_ec_MW=8.0, rho_dep=0.35, sigma_rho=0.08))
    mix.add_source(LHCDSource(P_lh_MW=3.0, rho_dep=0.65, sigma_rho=0.12))
    mix.add_source(NBISource(P_nbi_MW=14.0, E_beam_keV=1000.0, rho_tangency=0.45, sigma_rho=0.15))

    evidence = current_drive_claim_evidence(
        mix,
        rho=rho,
        ne_19=ne,
        Te_keV=te,
        Ti_keV=ti,
        source="repository_current_drive_regression",
        source_id="current-drive-regression-v1",
    )
    assert evidence.claim_status == "bounded_current_drive_evidence"
    assert evidence.external_claim_allowed is False
    assert evidence.grid_normalised_power is True
    assert evidence.total_driven_current_A > 0.0
    with pytest.raises(ValueError, match="current-drive claim requires matched"):
        assert_current_drive_external_claim_admissible(evidence)

    output = tmp_path / "current_drive_claim.json"
    save_current_drive_claim_evidence(evidence, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["claim_status"] == "bounded_current_drive_evidence"


def test_current_drive_external_claim_requires_reference_artifact() -> None:
    rho, ne, te, ti = _profiles(32)
    mix = CurrentDriveMix(a=2.0)
    mix.add_source(ECCDSource(P_ec_MW=8.0, rho_dep=0.35, sigma_rho=0.08))
    artifact = {
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
            "total_current_relative_error": 0.10,
            "deposition_centroid_abs_error": 0.05,
            "peak_current_density_relative_error": 0.15,
            "nbi_slowing_down_relative_error": 0.10,
        },
    }
    evidence = current_drive_claim_evidence(
        mix,
        rho=rho,
        ne_19=ne,
        Te_keV=te,
        Ti_keV=ti,
        source="ray_tracing_benchmark",
        source_id="current-drive-ray-tracing-fixture-v1",
        reference_artifact=artifact,
    )
    assert evidence.external_claim_allowed is True
    assert assert_current_drive_external_claim_admissible(evidence) is evidence

    bad_artifact = dict(artifact)
    bad_artifact["metrics"] = dict(artifact["metrics"])
    bad_artifact["metrics"]["total_current_relative_error"] = 0.5
    with pytest.raises(ValueError, match="total_current_relative_error exceeds declared tolerance"):
        current_drive_claim_evidence(
            mix,
            rho=rho,
            ne_19=ne,
            Te_keV=te,
            Ti_keV=ti,
            source="ray_tracing_benchmark",
            source_id="current-drive-ray-tracing-fixture-v1",
            reference_artifact=bad_artifact,
        )


def _valid_reference_artifact() -> dict:
    return {
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
            "total_current_relative_error": 0.10,
            "deposition_centroid_abs_error": 0.05,
            "peak_current_density_relative_error": 0.15,
            "nbi_slowing_down_relative_error": 0.10,
        },
    }


class TestScalarAndProfileValidators:
    def test_require_nonnegative_scalar(self) -> None:
        from scpn_control.core.current_drive import _require_nonnegative_scalar

        assert _require_nonnegative_scalar("x", 0.0) == 0.0
        with pytest.raises(ValueError, match="finite and >= 0"):
            _require_nonnegative_scalar("x", -1.0)

    def test_require_positive_profile_shape(self) -> None:
        from scpn_control.core.current_drive import _require_positive_profile

        with pytest.raises(ValueError, match=r"must have shape"):
            _require_positive_profile("x", np.ones(3), (4,))
        with pytest.raises(ValueError, match="finite positive"):
            _require_positive_profile("x", np.array([1.0, -2.0, 3.0]))

    def test_require_current_drive_profiles_grid(self) -> None:
        from scpn_control.core.current_drive import _require_current_drive_profiles

        ne = np.ones(3)
        with pytest.raises(ValueError, match="one-dimensional"):
            _require_current_drive_profiles(np.zeros((2, 2)), ne, ne)
        with pytest.raises(ValueError, match="rho grid must be finite"):
            _require_current_drive_profiles(np.array([0.0, np.nan, 1.0]), ne, ne)

    def test_require_current_drive_grid(self) -> None:
        from scpn_control.core.current_drive import _require_current_drive_grid

        with pytest.raises(ValueError, match="one-dimensional"):
            _require_current_drive_grid(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="rho grid must be finite"):
            _require_current_drive_grid(np.array([0.0, np.inf]))
        with pytest.raises(ValueError, match="strictly increasing"):
            _require_current_drive_grid(np.array([0.0, 0.5, 0.5]))

    def test_trapezoid_integral_guards(self) -> None:
        from scpn_control.core.current_drive import _trapezoid_integral

        with pytest.raises(ValueError, match="one-dimensional"):
            _trapezoid_integral(np.zeros((2, 2)), np.zeros(2))
        with pytest.raises(ValueError, match="lengths must match"):
            _trapezoid_integral(np.zeros(3), np.zeros(4))
        assert _trapezoid_integral(np.array([1.0]), np.array([0.0])) == 0.0

    def test_normalised_radial_deposition_edge_cases(self) -> None:
        from scpn_control.core.current_drive import _normalised_radial_deposition

        empty = _normalised_radial_deposition(np.array([]), 1.0e6, 0.5, 0.1)
        assert empty.size == 0
        zeros = _normalised_radial_deposition(np.linspace(0.0, 1.0, 5), 0.0, 0.5, 0.1)
        assert np.all(zeros == 0.0)
        with pytest.raises(ValueError, match="cannot be normalised"):
            _normalised_radial_deposition(np.array([0.0, 1.0]), 1.0, 0.5, 1.0e-4)


class TestReferenceScalarValidators:
    def test_non_empty_text(self) -> None:
        from scpn_control.core.current_drive import _non_empty_text

        with pytest.raises(ValueError, match="non-empty string"):
            _non_empty_text("x", "  ")

    def test_positive_reference_scalar(self) -> None:
        from scpn_control.core.current_drive import _positive_reference_scalar

        assert _positive_reference_scalar("x", 2.0) == 2.0
        with pytest.raises(ValueError, match="finite and positive"):
            _positive_reference_scalar("x", True)
        with pytest.raises(ValueError, match="finite and positive"):
            _positive_reference_scalar("x", 0.0)

    def test_nonnegative_reference_scalar(self) -> None:
        from scpn_control.core.current_drive import _nonnegative_reference_scalar

        assert _nonnegative_reference_scalar("x", 0.0) == 0.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            _nonnegative_reference_scalar("x", "a")
        with pytest.raises(ValueError, match="finite and non-negative"):
            _nonnegative_reference_scalar("x", -1.0)

    def test_sha256_text(self) -> None:
        from scpn_control.core.current_drive import _sha256_text

        assert _sha256_text("x", "a" * 64) == "a" * 64
        with pytest.raises(ValueError, match="SHA-256 hex digest"):
            _sha256_text("x", "not-a-digest")


class TestReferenceArtifactRejection:
    def test_rejects_non_dict_artifact(self) -> None:
        from scpn_control.core.current_drive import _extract_current_drive_reference_artifact

        assert _extract_current_drive_reference_artifact(None) == (None, False)
        with pytest.raises(ValueError, match="must be a dictionary"):
            _extract_current_drive_reference_artifact(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_rejects_unadmissible_source(self) -> None:
        from scpn_control.core.current_drive import _extract_current_drive_reference_artifact

        artifact = _valid_reference_artifact()
        artifact["source"] = "homemade_guess"
        with pytest.raises(ValueError, match="source must be one of"):
            _extract_current_drive_reference_artifact(artifact)

    def test_rejects_bad_units(self) -> None:
        from scpn_control.core.current_drive import _extract_current_drive_reference_artifact

        artifact = _valid_reference_artifact()
        artifact["units"] = {"power": "kW"}
        with pytest.raises(ValueError, match="unit contracts"):
            _extract_current_drive_reference_artifact(artifact)

    def test_rejects_bad_case_count(self) -> None:
        from scpn_control.core.current_drive import _extract_current_drive_reference_artifact

        artifact = _valid_reference_artifact()
        artifact["reference_case_count"] = 0
        with pytest.raises(ValueError, match="positive integer"):
            _extract_current_drive_reference_artifact(artifact)

    def test_rejects_non_dict_metric_blocks(self) -> None:
        from scpn_control.core.current_drive import _extract_current_drive_reference_artifact

        artifact = _valid_reference_artifact()
        artifact["metrics"] = ["not", "a", "dict"]
        with pytest.raises(ValueError, match="metrics and tolerances must be dictionaries"):
            _extract_current_drive_reference_artifact(artifact)


class TestPhysicsEdgeCases:
    def test_eccd_efficiency_rejects_non_finite_n_parallel(self) -> None:
        with pytest.raises(ValueError, match="N_parallel must be finite"):
            eccd_efficiency(Te_keV=5.0, Z_eff=1.5, N_parallel=np.inf)

    def test_nbi_zero_power_drives_no_current(self) -> None:
        rho, ne, te, ti = _profiles(16)
        source = NBISource(P_nbi_MW=0.0, E_beam_keV=1000.0, rho_tangency=0.45, sigma_rho=0.15)
        j_cd = source.j_cd(rho, ne, te, ti)
        assert np.all(j_cd == 0.0)


class TestClaimEvidenceGuards:
    def test_rejects_unbounded_source(self) -> None:
        rho, ne, te, ti = _profiles(16)
        mix = CurrentDriveMix(a=2.0)
        mix.add_source(ECCDSource(P_ec_MW=8.0, rho_dep=0.35, sigma_rho=0.08))
        with pytest.raises(ValueError, match="source must be one of"):
            current_drive_claim_evidence(
                mix,
                rho=rho,
                ne_19=ne,
                Te_keV=te,
                Ti_keV=ti,
                source="homemade_guess",
                source_id="x",
            )

    def test_assert_and_save_reject_non_evidence(self) -> None:
        with pytest.raises(ValueError, match="must be CurrentDriveClaimEvidence"):
            assert_current_drive_external_claim_admissible(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be CurrentDriveClaimEvidence"):
            save_current_drive_claim_evidence(object(), "unused.json")  # type: ignore[arg-type]
