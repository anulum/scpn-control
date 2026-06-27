# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP model tests
from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scpn_control.control.rzip_model import (
    RZIPCalibrationEvidence,
    RZIPController,
    RZIPModel,
    VerticalStabilityAnalysis,
    _evidence_payload_digest,
    assert_rzip_facility_claim_admissible,
    rzip_calibration_evidence,
    save_rzip_calibration_evidence,
)
from scpn_control.core.vessel_model import VesselElement, VesselModel


@pytest.fixture
def simple_vessel():
    # Two loops above and below midplane to act as a wall
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e-3, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e-3, cross_section=0.1, inductance=1e-5),
    ]
    return VesselModel(elements)


@pytest.fixture
def active_coils():
    return [
        VesselElement(R=2.5, Z=1.0, resistance=1e-2, cross_section=0.05, inductance=5e-5),
        VesselElement(R=2.5, Z=-1.0, resistance=1e-2, cross_section=0.05, inductance=5e-5),
    ]


def test_stable_n_index(simple_vessel):
    # n_index > 0 means vertically stable
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=simple_vessel)
    gamma = rzip.vertical_growth_rate()
    assert gamma <= 0.0


def test_rzip_model_rejects_nonphysical_parameters(simple_vessel):
    invalid_kwargs = (
        {"R0": 0.0},
        {"a": -0.5},
        {"kappa": 0.0},
        {"Ip_MA": -1.0},
        {"B0": 0.0},
        {"n_index": np.nan},
    )

    for kwargs in invalid_kwargs:
        params = dict(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=simple_vessel)
        params.update(kwargs)
        with pytest.raises(ValueError, match="physical|finite|positive"):
            RZIPModel(**params)


def test_rzip_model_rejects_minor_radius_at_or_beyond_major_radius(simple_vessel):
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        RZIPModel(R0=2.0, a=2.0, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=simple_vessel)


def test_rzip_model_uses_declared_vertical_inertia(simple_vessel):
    light = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
        vertical_inertia_kg=1.0,
    )
    heavy = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
        vertical_inertia_kg=4.0,
    )

    assert heavy.M_eff == 4.0
    assert heavy.vertical_growth_rate() < light.vertical_growth_rate()


def test_rzip_calibration_evidence_records_bounded_local_claim(simple_vessel, tmp_path):
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
        vertical_inertia_kg=2.5,
    )

    evidence = rzip_calibration_evidence(
        rzip,
        source="local_regression_reference",
        source_id="tests/test_rzip_model.py::simple_vessel",
        wall_time_constant_s=0.01,
    )
    out = tmp_path / "rzip_calibration.json"
    save_rzip_calibration_evidence(evidence, out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert isinstance(evidence, RZIPCalibrationEvidence)
    assert evidence.vertical_inertia_kg == pytest.approx(2.5)
    assert evidence.growth_rate_s_inv == pytest.approx(rzip.vertical_growth_rate())
    assert evidence.reference_growth_rate_s_inv is None
    assert evidence.facility_claim_allowed is False
    assert len(evidence.evidence_payload_sha256) == 64
    assert evidence.claim_status.startswith("bounded local RZIP regression evidence")
    assert payload["schema_version"] == 1
    assert payload["facility_claim_allowed"] is False
    assert payload["evidence_payload_sha256"] == evidence.evidence_payload_sha256


def test_rzip_facility_claim_admission_requires_external_tolerance(simple_vessel):
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
        vertical_inertia_kg=1.0,
    )
    gamma = rzip.vertical_growth_rate()

    admitted = rzip_calibration_evidence(
        rzip,
        source="external_code_benchmark",
        source_id="reference_rzip_symmetric_wall_case",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=gamma * 1.01,
        growth_rate_relative_tolerance=0.02,
    )
    rejected = rzip_calibration_evidence(
        rzip,
        source="external_code_benchmark",
        source_id="reference_rzip_mismatched_case",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=gamma * 1.5,
        growth_rate_relative_tolerance=0.02,
    )

    assert assert_rzip_facility_claim_admissible(admitted) == admitted
    assert admitted.facility_claim_allowed is True
    assert admitted.growth_rate_relative_error == pytest.approx(abs(gamma - gamma * 1.01) / abs(gamma * 1.01))
    assert rejected.facility_claim_allowed is False
    with pytest.raises(ValueError, match="growth-rate tolerance"):
        assert_rzip_facility_claim_admissible(rejected)


def test_rzip_facility_claim_admission_rejects_tampered_local_evidence(simple_vessel):
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
    )
    local = rzip_calibration_evidence(
        rzip,
        source="local_regression_reference",
        source_id="tests/test_rzip_model.py::simple_vessel",
        wall_time_constant_s=0.01,
    )
    tampered = replace(
        local,
        facility_claim_allowed=True,
        claim_status="external RZIP reference admission passed for declared tolerance",
    )

    with pytest.raises(ValueError, match="facility reference source"):
        assert_rzip_facility_claim_admissible(tampered)


def test_rzip_facility_claim_admission_rejects_tampered_growth_error(simple_vessel):
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
    )
    gamma = rzip.vertical_growth_rate()
    admitted = rzip_calibration_evidence(
        rzip,
        source="external_code_benchmark",
        source_id="reference_rzip_symmetric_wall_case",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=gamma,
        growth_rate_relative_tolerance=0.02,
    )
    tampered = replace(admitted, growth_rate_relative_error=0.5)

    with pytest.raises(ValueError, match="growth-rate tolerance"):
        assert_rzip_facility_claim_admissible(tampered)


def test_rzip_facility_claim_admission_rejects_digest_tampering(simple_vessel):
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=simple_vessel,
    )
    gamma = rzip.vertical_growth_rate()
    admitted = rzip_calibration_evidence(
        rzip,
        source="external_code_benchmark",
        source_id="reference_rzip_symmetric_wall_case",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=gamma,
        growth_rate_relative_tolerance=0.02,
    )
    tampered = replace(admitted, source_id="changed-after-admission")

    with pytest.raises(ValueError, match="payload digest"):
        assert_rzip_facility_claim_admissible(tampered)


def test_rzip_calibration_evidence_rejects_invalid_reference_inputs(simple_vessel):
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel)

    with pytest.raises(ValueError, match="source"):
        rzip_calibration_evidence(rzip, source="mock", source_id="case", wall_time_constant_s=0.01)
    with pytest.raises(ValueError, match="source_id"):
        rzip_calibration_evidence(rzip, source="local_regression_reference", source_id="", wall_time_constant_s=0.01)
    with pytest.raises(ValueError, match="wall_time_constant_s"):
        rzip_calibration_evidence(rzip, source="local_regression_reference", source_id="case", wall_time_constant_s=0.0)
    with pytest.raises(ValueError, match="reference_growth_rate_s_inv"):
        rzip_calibration_evidence(
            rzip,
            source="external_code_benchmark",
            source_id="case",
            wall_time_constant_s=0.01,
            reference_growth_rate_s_inv=-1.0,
        )


def test_rzip_model_rejects_nonphysical_vertical_inertia(simple_vessel):
    with pytest.raises(ValueError, match="vertical_inertia_kg"):
        RZIPModel(
            R0=2.0,
            a=0.5,
            kappa=1.7,
            Ip_MA=1.0,
            B0=1.0,
            n_index=-1.0,
            vessel=simple_vessel,
            vertical_inertia_kg=0.0,
        )


def test_unstable_n_index(simple_vessel):
    # n_index < 0 means vertically unstable
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel)
    gamma = rzip.vertical_growth_rate()
    assert gamma > 0.0

    # Growth time should be realistic (e.g. 10s to 1000s of ms)
    tau = rzip.vertical_growth_time()
    assert 1.0 < tau < 10000.0


def test_wall_slows_growth():
    elements_highly_resistive = [
        VesselElement(R=2.0, Z=0.5, resistance=1e3, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e3, cross_section=0.1, inductance=1e-5),
    ]
    elements_conductive = [
        VesselElement(R=2.0, Z=0.5, resistance=1e-5, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e-5, cross_section=0.1, inductance=1e-5),
    ]
    vessel_res = VesselModel(elements_highly_resistive)
    vessel_cond = VesselModel(elements_conductive)

    rzip_res = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=vessel_res)
    rzip_cond = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=vessel_cond)

    # Conductive wall should lower the growth rate (slow down the instability)
    assert rzip_cond.vertical_growth_rate() < rzip_res.vertical_growth_rate()


def test_state_space_dimensions(simple_vessel, active_coils):
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel, active_coils=active_coils
    )
    A, B, C, D = rzip.build_state_space()

    n_wall = len(simple_vessel.elements)
    n_coils = len(active_coils)
    n_states = 2 + n_wall + n_coils

    assert A.shape == (n_states, n_states)
    assert B.shape == (n_states, n_coils)
    assert C.shape == (1, n_states)
    assert D.shape == (1, n_coils)


def test_feedback_stabilization(active_coils):
    # Use a resistive wall so it doesn't shield the active coils completely
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    resistive_vessel = VesselModel(elements)

    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=resistive_vessel, active_coils=active_coils
    )

    assert rzip.vertical_growth_rate() > 0.0

    ctrl = RZIPController(rzip, Kp=1.0, Kd=1e4)
    eigvals = ctrl.closed_loop_eigenvalues()

    max_real = np.max(np.real(eigvals))
    assert max_real < rzip.vertical_growth_rate()  # Growth rate should at least decrease


def test_build_state_space_rejects_singular_circuit_inductance_matrix(simple_vessel):
    # Create a one-circuit vessel with zero self-inductance to force singular M_mat.
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e-3, cross_section=0.1, inductance=0.0),
    ]
    vessel = VesselModel(elements)
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=vessel)
    with pytest.raises(ValueError, match="inductance matrix"):
        rzip.build_state_space()


def test_vertical_growth_time_stable(simple_vessel):
    """Stable vertical response reports infinite open-loop growth time."""
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=simple_vessel)
    tau = rzip.vertical_growth_time()
    assert tau == float("inf")


def test_stability_margin(simple_vessel):
    """Stability margin exposes the configured vertical field index."""
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=simple_vessel)
    assert rzip.stability_margin() == -0.5


def test_compute_n_index_from_flux_gradient():
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    r0 = 2.0
    target_n = 0.72
    b_axis = 3.1
    r_axis = np.linspace(1.4, 2.6, 121)
    z_axis = np.linspace(-0.35, 0.35, 41)
    rr, zz = np.meshgrid(r_axis, z_axis)

    psi = b_axis * ((1.0 + target_n) * rr**2 / 2.0 - target_n * rr**3 / (3.0 * r0))

    result = VerticalStabilityAnalysis.compute_n_index(psi, rr, zz, r0)

    assert result == pytest.approx(target_n, rel=2e-3, abs=2e-3)


def test_compute_n_index_rejects_degenerate_vertical_field():
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    r_axis = np.linspace(1.0, 3.0, 9)
    z_axis = np.linspace(-0.2, 0.2, 5)
    rr, zz = np.meshgrid(r_axis, z_axis)

    with pytest.raises(ValueError, match="vertical field"):
        VerticalStabilityAnalysis.compute_n_index(np.zeros_like(rr), rr, zz, 2.0)


def test_compute_n_index_rejects_non_finite_radial_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    original_interp = np.interp
    calls = 0

    def fake_interp(
        x: Any,
        xp: Any,
        fp: Any,
        left: Any = None,
        right: Any = None,
        period: Any = None,
    ) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            return float("nan")
        return float(original_interp(x, xp, fp, left=left, right=right, period=period))

    psi, r_axis, z_axis = _flux_grid()
    monkeypatch.setattr(np, "interp", fake_interp)

    with pytest.raises(ValueError, match="radial gradient is not finite"):
        VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, 2.0)

    assert calls == 2


def test_passive_stability_margin():
    """Line 146: VerticalStabilityAnalysis.passive_stability_margin returns n_index."""
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    assert VerticalStabilityAnalysis.passive_stability_margin(-0.3, 0.01) == -0.3


def test_passive_stability_margin_rejects_nonphysical_wall_time() -> None:
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    with pytest.raises(ValueError, match="n_index"):
        VerticalStabilityAnalysis.passive_stability_margin(float("nan"), 0.01)
    with pytest.raises(ValueError, match="tau_wall"):
        VerticalStabilityAnalysis.passive_stability_margin(0.3, 0.0)
    with pytest.raises(ValueError, match="tau_wall"):
        VerticalStabilityAnalysis.passive_stability_margin(0.3, float("inf"))


def test_required_feedback_gain():
    """Feedback gain follows the wall-normalised RZIP latency threshold."""
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    gain = VerticalStabilityAnalysis.required_feedback_gain(gamma=10.0, tau_wall=0.01, tau_controller=1e-4)

    assert gain == pytest.approx(0.1001)


def test_required_feedback_gain_rejects_nonphysical_inputs() -> None:
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    for kwargs in (
        {"gamma": -1.0, "tau_wall": 0.01, "tau_controller": 1e-4},
        {"gamma": 10.0, "tau_wall": 0.0, "tau_controller": 1e-4},
        {"gamma": 10.0, "tau_wall": 0.01, "tau_controller": -1e-4},
    ):
        with pytest.raises(ValueError, match="finite|non-negative|positive"):
            VerticalStabilityAnalysis.required_feedback_gain(**kwargs)


def test_controller_rejects_nonphysical_gains(active_coils):
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    vessel = VesselModel(elements)
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=vessel, active_coils=active_coils
    )

    with pytest.raises(ValueError, match="Kp"):
        RZIPController(rzip, Kp=float("nan"), Kd=1.0)
    with pytest.raises(ValueError, match="Kd"):
        RZIPController(rzip, Kp=1.0, Kd=-1.0)


def test_controller_step_rejects_nonpositive_timestep(active_coils):
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    vessel = VesselModel(elements)
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=vessel, active_coils=active_coils
    )
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)
    with pytest.raises(ValueError, match="dt"):
        ctrl.step(0.1, dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        ctrl.step(0.1, dt=-1e-3)


def test_controller_step_rejects_nonfinite_measurement_and_timestep(active_coils):
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    vessel = VesselModel(elements)
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=vessel, active_coils=active_coils
    )
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)

    with pytest.raises(ValueError, match="finite"):
        ctrl.step(float("nan"), dt=1e-3)

    with pytest.raises(ValueError, match="finite"):
        ctrl.step(0.1, dt=float("inf"))


def test_vertical_stability_analysis_rejects_nonrectilinear_grids() -> None:
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    psi = np.ones((3, 3), dtype=float)
    r_axis = np.array([2.0, 2.5, 3.0])
    z_axis = np.array([-0.5, 0.0, 0.5])
    R, Z = np.meshgrid(r_axis, z_axis)
    R[1, 1] += 0.1

    with pytest.raises(ValueError, match="rectilinear"):
        VerticalStabilityAnalysis.compute_n_index(psi, R, Z, R0=2.5)


class _NanGrowthRZIP:
    """Plant stub whose vertical growth rate is non-finite."""

    M_eff = 1.0

    @staticmethod
    def vertical_growth_rate() -> float:
        return float("nan")

    @staticmethod
    def vertical_growth_time() -> float:
        return 1.0


def _facility_evidence(simple_vessel):
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel)
    gamma = rzip.vertical_growth_rate()
    return rzip_calibration_evidence(
        rzip,
        source="external_code_benchmark",
        source_id="reference_rzip_symmetric_wall_case",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=gamma,
        growth_rate_relative_tolerance=0.02,
    )


def test_calibration_evidence_rejects_blank_model_id(simple_vessel):
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel)
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        rzip_calibration_evidence(
            rzip,
            source="local_regression_reference",
            source_id="case",
            wall_time_constant_s=0.01,
            model_id="   ",
        )


def test_calibration_evidence_rejects_non_finite_growth_rate():
    with pytest.raises(ValueError, match="RZIP growth rate must be finite"):
        rzip_calibration_evidence(
            _NanGrowthRZIP(),
            source="local_regression_reference",
            source_id="case",
            wall_time_constant_s=0.01,
        )


def test_calibration_evidence_flags_external_source_without_comparison(simple_vessel):
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel)
    evidence = rzip_calibration_evidence(
        rzip,
        source="external_code_benchmark",
        source_id="reference_without_reference_growth_rate",
        wall_time_constant_s=0.01,
    )
    assert evidence.facility_claim_allowed is False
    assert "comparison is missing" in evidence.claim_status


def test_facility_admission_rejects_non_evidence_object():
    with pytest.raises(ValueError, match="must be RZIPCalibrationEvidence"):
        assert_rzip_facility_claim_admissible({"not": "evidence"})


def test_facility_admission_rejects_unsupported_schema_version(simple_vessel):
    tampered = replace(_facility_evidence(simple_vessel), schema_version=99)
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        assert_rzip_facility_claim_admissible(tampered)


def test_facility_admission_rejects_blank_source_id(simple_vessel):
    tampered = replace(_facility_evidence(simple_vessel), source_id="   ")
    with pytest.raises(ValueError, match="non-empty source_id"):
        assert_rzip_facility_claim_admissible(tampered)


def test_facility_admission_rejects_blank_model_id(simple_vessel):
    tampered = replace(_facility_evidence(simple_vessel), model_id="   ")
    with pytest.raises(ValueError, match="non-empty model_id"):
        assert_rzip_facility_claim_admissible(tampered)


def test_facility_admission_requires_reference_growth_rate(simple_vessel):
    tampered = replace(_facility_evidence(simple_vessel), reference_growth_rate_s_inv=None)
    with pytest.raises(ValueError, match="requires a reference growth rate"):
        assert_rzip_facility_claim_admissible(tampered)


def test_facility_admission_requires_finite_comparison_error(simple_vessel):
    tampered = replace(_facility_evidence(simple_vessel), growth_rate_relative_error=None)
    with pytest.raises(ValueError, match="finite growth-rate comparison error"):
        assert_rzip_facility_claim_admissible(tampered)


def test_facility_admission_requires_finite_model_growth_rate(simple_vessel):
    tampered = replace(_facility_evidence(simple_vessel), growth_rate_s_inv=float("nan"))
    with pytest.raises(ValueError, match="finite model growth rate"):
        assert_rzip_facility_claim_admissible(tampered)


def test_facility_admission_rejects_resealed_disallowed_claim(simple_vessel):
    # Re-seal a payload whose facility flag is forced False while every other
    # admission gate (including the digest) still passes, isolating the final guard.
    admitted = _facility_evidence(simple_vessel)
    forced = replace(admitted, facility_claim_allowed=False)
    resealed = replace(forced, evidence_payload_sha256=_evidence_payload_digest(asdict(forced)))
    with pytest.raises(ValueError, match="not admissible"):
        assert_rzip_facility_claim_admissible(resealed)


def _flux_grid(target_n: float = 0.72, r0: float = 2.0, b_axis: float = 3.1):
    r_axis = np.linspace(1.4, 2.6, 121)
    z_axis = np.linspace(-0.35, 0.35, 41)
    rr, zz = np.meshgrid(r_axis, z_axis)
    psi = b_axis * ((1.0 + target_n) * rr**2 / 2.0 - target_n * rr**3 / (3.0 * r0))
    return psi, r_axis, z_axis


def test_compute_n_index_accepts_one_dimensional_axes():
    psi, r_axis, z_axis = _flux_grid()
    result = VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, 2.0)
    assert result == pytest.approx(0.72, rel=2e-3, abs=2e-3)


def test_compute_n_index_rejects_one_dimensional_axis_size_mismatch():
    psi = np.ones((4, 5), dtype=float)
    with pytest.raises(ValueError, match="must match psi shape"):
        VerticalStabilityAnalysis.compute_n_index(psi, np.linspace(1.0, 2.0, 9), np.linspace(-1.0, 1.0, 4), 1.5)


def test_compute_n_index_rejects_axes_that_are_not_axes_or_grids():
    psi = np.ones((5, 9), dtype=float)
    bad = np.ones((3, 3), dtype=float)
    with pytest.raises(ValueError, match="1-D axes or rectilinear 2-D grids"):
        VerticalStabilityAnalysis.compute_n_index(psi, bad, bad, 2.0)


def test_compute_n_index_rejects_too_few_grid_points():
    psi = np.ones((2, 5), dtype=float)
    with pytest.raises(ValueError, match="at least three R and Z points"):
        VerticalStabilityAnalysis.compute_n_index(psi, np.linspace(1.0, 2.0, 5), np.array([-0.5, 0.5]), 1.5)


def test_compute_n_index_rejects_non_finite_axes():
    psi = np.ones((3, 3), dtype=float)
    r_axis = np.array([2.0, 2.5, np.nan])
    z_axis = np.array([-0.5, 0.0, 0.5])
    with pytest.raises(ValueError, match="R/Z axes must be finite"):
        VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, 2.2)


def test_compute_n_index_rejects_non_increasing_axes():
    psi = np.ones((3, 3), dtype=float)
    r_axis = np.array([2.0, 2.0, 3.0])
    z_axis = np.array([-0.5, 0.0, 0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, 2.5)


def test_compute_n_index_rejects_non_positive_radius_axis():
    psi = np.ones((3, 3), dtype=float)
    r_axis = np.array([-1.0, 0.0, 1.0])
    z_axis = np.array([-0.5, 0.0, 0.5])
    with pytest.raises(ValueError, match="R axis must be positive"):
        VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, 0.5)


def test_compute_n_index_rejects_non_two_dimensional_psi():
    with pytest.raises(ValueError, match="psi must be a 2-D"):
        VerticalStabilityAnalysis.compute_n_index(np.ones(5), np.linspace(1.0, 2.0, 5), np.linspace(-1.0, 1.0, 5), 1.5)


def test_compute_n_index_rejects_non_finite_psi():
    psi = np.ones((3, 3), dtype=float)
    psi[1, 1] = np.inf
    with pytest.raises(ValueError, match="psi grid must be finite"):
        VerticalStabilityAnalysis.compute_n_index(psi, np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 1.0]), 2.0)


@pytest.mark.parametrize("r0", [0.0, float("nan"), -1.0])
def test_compute_n_index_rejects_non_physical_r0(r0):
    psi, r_axis, z_axis = _flux_grid()
    with pytest.raises(ValueError, match="R0 must be finite and positive"):
        VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, r0)


def test_compute_n_index_rejects_r0_outside_grid():
    psi, r_axis, z_axis = _flux_grid()
    with pytest.raises(ValueError, match="R0 must lie inside the R grid"):
        VerticalStabilityAnalysis.compute_n_index(psi, r_axis, z_axis, 100.0)


def test_required_feedback_gain_returns_zero_for_marginal_growth():
    assert VerticalStabilityAnalysis.required_feedback_gain(gamma=0.0, tau_wall=0.01, tau_controller=1e-4) == 0.0


def _active_feedback_rzip(active_coils: list[VesselElement]) -> RZIPModel:
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    return RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-0.5,
        vessel=VesselModel(elements),
        active_coils=active_coils,
    )


def test_controller_falls_back_to_zero_gain_when_riccati_fails(active_coils, monkeypatch):
    import scipy.linalg

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise np.linalg.LinAlgError("forced Riccati failure")

    monkeypatch.setattr(scipy.linalg, "solve_continuous_are", _boom)
    rzip = _active_feedback_rzip(active_coils)
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)
    A, B, _C, _D = rzip.build_state_space()
    assert ctrl.K_gain.shape == (B.shape[1], A.shape[1])
    assert np.all(ctrl.K_gain == 0.0)


def test_controller_uses_scipy_riccati_gain_when_available(active_coils, monkeypatch):
    import scipy.linalg

    rzip = _active_feedback_rzip(active_coils)
    A, B, _C, _D = rzip.build_state_space()

    def _identity_solution(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.eye(A.shape[0])

    monkeypatch.setattr(scipy.linalg, "solve_continuous_are", _identity_solution)
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)

    np.testing.assert_allclose(ctrl.K_gain, B.T)


def test_controller_falls_back_when_scipy_validation_raises_type_error(active_coils, monkeypatch):
    import scipy.linalg

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise TypeError("SciPy/NumPy validation rejected the ARE input")

    monkeypatch.setattr(scipy.linalg, "solve_continuous_are", _boom)
    rzip = _active_feedback_rzip(active_coils)
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)
    A, B, _C, _D = rzip.build_state_space()
    assert ctrl.K_gain.shape == (B.shape[1], A.shape[1])
    assert np.any(ctrl.K_gain != 0.0)
    assert np.max(np.real(ctrl.closed_loop_eigenvalues())) < rzip.vertical_growth_rate()


def test_controller_zero_gain_fallback_handles_nonfinite_discrete_riccati(active_coils, monkeypatch):
    import scipy.linalg

    def _care_boom(*_args: object, **_kwargs: object) -> None:
        raise TypeError("SciPy/NumPy validation rejected the ARE input")

    def _nonfinite_solve(_lhs: object, rhs: np.ndarray) -> np.ndarray:
        return np.full_like(rhs, np.inf, dtype=float)

    monkeypatch.setattr(scipy.linalg, "solve_continuous_are", _care_boom)
    monkeypatch.setattr(np.linalg, "solve", _nonfinite_solve)
    rzip = _active_feedback_rzip(active_coils)
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)
    A, B, _C, _D = rzip.build_state_space()
    assert ctrl.K_gain.shape == (B.shape[1], A.shape[1])
    assert np.all(ctrl.K_gain == 0.0)


def test_controller_zero_gain_fallback_handles_overflowing_discrete_covariance(active_coils, monkeypatch):
    import scipy.linalg

    def _care_boom(*_args: object, **_kwargs: object) -> None:
        raise TypeError("SciPy/NumPy validation rejected the ARE input")

    def _overflowing_solve(_lhs: object, rhs: np.ndarray) -> np.ndarray:
        return np.full_like(rhs, 1.0e308, dtype=float)

    monkeypatch.setattr(scipy.linalg, "solve_continuous_are", _care_boom)
    monkeypatch.setattr(np.linalg, "solve", _overflowing_solve)
    rzip = _active_feedback_rzip(active_coils)

    with np.errstate(over="ignore", invalid="ignore"):
        ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)

    A, B, _C, _D = rzip.build_state_space()
    assert ctrl.K_gain.shape == (B.shape[1], A.shape[1])
    assert np.all(ctrl.K_gain == 0.0)


def test_controller_zero_gain_fallback_remains_last_resort(active_coils, monkeypatch):
    import scipy.linalg

    def _care_boom(*_args: object, **_kwargs: object) -> None:
        raise TypeError("SciPy/NumPy validation rejected the ARE input")

    def _solve_boom(*_args: object, **_kwargs: object) -> None:
        raise np.linalg.LinAlgError("discrete Riccati fallback cannot solve this plant")

    monkeypatch.setattr(scipy.linalg, "solve_continuous_are", _care_boom)
    monkeypatch.setattr(np.linalg, "solve", _solve_boom)
    rzip = _active_feedback_rzip(active_coils)
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)
    A, B, _C, _D = rzip.build_state_space()
    assert ctrl.K_gain.shape == (B.shape[1], A.shape[1])
    assert np.all(ctrl.K_gain == 0.0)


def test_rzip_controller_step_is_antisymmetric_for_vertical_error(active_coils):
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1.0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1.0, cross_section=0.1, inductance=1e-5),
    ]
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-0.5,
        vessel=VesselModel(elements),
        active_coils=active_coils,
    )
    positive_controller = RZIPController(rzip, Kp=3.0, Kd=0.0)
    negative_controller = RZIPController(rzip, Kp=3.0, Kd=0.0)

    positive = positive_controller.step(0.1, 0.01)
    negative = negative_controller.step(-0.1, 0.01)

    np.testing.assert_allclose(positive, -negative)
