# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Kinetic Efit
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Kinetic EFIT Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.realtime_efit import MagneticDiagnostics
from scpn_control.core.kinetic_efit import (
    FastIonPressure,
    KineticConstraints,
    KineticEFIT,
    assert_kinetic_efit_facility_claim_admissible,
    kinetic_efit_claim_evidence,
    mse_pitch_angle,
    save_kinetic_efit_claim_evidence,
)


def reference_diagnostics() -> MagneticDiagnostics:
    return MagneticDiagnostics([(2.0, 1.0)], [(2.0, 1.0, "R")], rogowski_radius=3.0)


def reference_kin_constraints() -> KineticConstraints:
    return KineticConstraints(
        Te_points=[(6.0, 0.0, 10.0), (7.9, 0.0, 1.0)],
        ne_points=[(6.0, 0.0, 5.0), (7.9, 0.0, 0.5)],
        Ti_points=[(6.0, 0.0, 8.0), (7.9, 0.0, 0.8)],
        mse_points=[(6.5, 0.0, 5.0)],
    )


def test_fast_ion_pressure():
    fi = FastIonPressure(E_fast_keV=100.0, n_fast_frac=0.1, anisotropy_sigma=0.2)
    rho = np.linspace(0, 1, 10)
    ne = np.ones(10) * 5.0

    p_perp = fi.p_perp(rho, ne)
    p_par = fi.p_par(rho, ne)
    p_iso = fi.p_isotropic_equivalent(rho, ne)

    assert np.all(p_perp > p_par)
    assert np.allclose((2 * p_perp + p_par) / 3.0, p_iso)


def test_mse_pitch_angle():
    pitch = mse_pitch_angle(B_R=0.0, B_Z=0.5, B_phi=5.0, v_beam=1e6, R=6.0)
    assert 5.0 < pitch < 6.0  # arctan(0.1) ~ 5.7 deg


def test_kinetic_efit_isotropic():
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)

    res = kefit.reconstruct({})

    assert res.pressure_consistency == 0.0
    assert res.wall_time_ms < 200.0
    assert len(res.p_kinetic) == 50
    assert res.p_kinetic[0] > res.p_kinetic[-1]


def test_kinetic_efit_anisotropic():
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)

    res = kefit.reconstruct({})

    assert res.pressure_consistency > 0.0
    assert not np.allclose(res.p_equilibrium, res.p_kinetic)
    assert np.all(res.sigma_anisotropy == 0.2)
    assert res.beta_fast > 0.0


def test_mse_constraint_q_profile():
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.0, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)

    kefit_mse = KineticEFIT(diag, kin, fi, R, Z)

    kin_no_mse = reference_kin_constraints()
    kin_no_mse.mse_points = []
    kefit_no_mse = KineticEFIT(diag, kin_no_mse, fi, R, Z)

    res_mse = kefit_mse.reconstruct({})
    res_no_mse = kefit_no_mse.reconstruct({})

    # MSE should constrain q to be closer to 1.0 at axis
    assert res_mse.q_profile[0] < res_no_mse.q_profile[0]


def test_kinetic_efit_rejects_missing_ne_points():
    diag = reference_diagnostics()
    kin = KineticConstraints(Te_points=[(6.2, 0.0, 10.0)], ne_points=[], Ti_points=[], mse_points=[])
    fi = FastIonPressure(100.0, 0.0, 0.0)
    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)
    with pytest.raises(ValueError, match="ne_points"):
        kefit.reconstruct({})


def test_kinetic_efit_rejects_missing_te_points():
    diag = reference_diagnostics()
    kin = KineticConstraints(Te_points=[], ne_points=[(6.2, 0.0, 5.0)], Ti_points=[], mse_points=[])
    fi = FastIonPressure(100.0, 0.0, 0.0)
    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)
    with pytest.raises(ValueError, match="Te_points"):
        kefit.reconstruct({})


def test_kinetic_efit_uses_measured_ion_temperature_profile():
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    kin.Ti_points = [(6.0, 0.0, 2.0), (7.9, 0.0, 0.2)]
    fi = FastIonPressure(100.0, 0.0, 0.0)
    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit_cold_ions = KineticEFIT(diag, kin, fi, R, Z)

    hot_ion_constraints = reference_kin_constraints()
    hot_ion_constraints.Ti_points = [(6.0, 0.0, 12.0), (7.9, 0.0, 1.2)]
    kefit_hot_ions = KineticEFIT(diag, hot_ion_constraints, fi, R, Z)

    assert kefit_hot_ions.reconstruct({}).p_kinetic[0] > kefit_cold_ions.reconstruct({}).p_kinetic[0]


def test_kinetic_efit_claim_evidence_records_bounded_profile_provenance(tmp_path):
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)
    r_grid = np.linspace(4, 8, 33)
    z_grid = np.linspace(-3, 3, 33)
    result = KineticEFIT(diag, kin, fi, r_grid, z_grid).reconstruct({})

    evidence = kinetic_efit_claim_evidence(
        result,
        kin,
        fi,
        source="synthetic_regression_reference",
        source_id="kinetic-regression-v1",
        diagnostic_source="repository magnetic diagnostics fixture",
        profile_source="repository Thomson and ion-temperature fixture",
        fast_ion_source="repository anisotropic fast-ion fixture",
        mse_calibration_source="repository MSE pitch-angle fixture",
    )
    report_path = tmp_path / "kinetic_efit_claim.json"
    save_kinetic_efit_claim_evidence(evidence, report_path)

    assert evidence.facility_claim_allowed is False
    assert evidence.claim_status == "bounded_controller_evidence"
    assert evidence.interpolation_geometry == "normalised_elliptic_rho"
    assert evidence.n_te_points == len(kin.Te_points)
    assert evidence.n_ne_points == len(kin.ne_points)
    assert evidence.n_ti_points == len(kin.Ti_points)
    assert evidence.n_mse_points == len(kin.mse_points)
    assert evidence.pressure_consistency == pytest.approx(result.pressure_consistency)
    assert evidence.q_axis == pytest.approx(result.q_profile[0])
    assert evidence.q_edge == pytest.approx(result.q_profile[-1])
    assert '"facility_claim_allowed": false' in report_path.read_text(encoding="utf-8")


def test_kinetic_efit_facility_admission_requires_matched_pressure_q_and_anisotropy():
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)
    r_grid = np.linspace(4, 8, 33)
    z_grid = np.linspace(-3, 3, 33)
    result = KineticEFIT(diag, kin, fi, r_grid, z_grid).reconstruct({})

    matched = kinetic_efit_claim_evidence(
        result,
        kin,
        fi,
        source="p_efit_reference",
        source_id="matched-public-pefit-reference",
        diagnostic_source="documented magnetic diagnostics",
        profile_source="documented Thomson, charge-exchange, and density profiles",
        fast_ion_source="documented neutral-beam fast-ion model",
        mse_calibration_source="documented MSE calibration",
        reference_pressure=result.p_kinetic.copy(),
        reference_q_profile=result.q_profile.copy(),
        reference_anisotropy_sigma=0.2,
    )
    assert_kinetic_efit_facility_claim_admissible(matched)
    assert matched.facility_claim_allowed is True

    q_mismatch = kinetic_efit_claim_evidence(
        result,
        kin,
        fi,
        source="p_efit_reference",
        source_id="mismatched-q-reference",
        diagnostic_source="documented magnetic diagnostics",
        profile_source="documented Thomson, charge-exchange, and density profiles",
        fast_ion_source="documented neutral-beam fast-ion model",
        mse_calibration_source="documented MSE calibration",
        reference_pressure=result.p_kinetic.copy(),
        reference_q_profile=result.q_profile + 0.5,
        reference_anisotropy_sigma=0.2,
        q_profile_relative_tolerance=0.01,
    )
    with pytest.raises(ValueError, match="facility claim requires matched"):
        assert_kinetic_efit_facility_claim_admissible(q_mismatch)
    assert q_mismatch.facility_claim_allowed is False


def test_claim_evidence_rejects_blank_text_nonfinite_tolerance_and_bad_profiles() -> None:
    """Claim evidence fails closed on blank metadata, non-finite tolerance, bad fast-ions, bad profiles."""
    import dataclasses as _dc

    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)
    r_grid = np.linspace(4, 8, 33)
    z_grid = np.linspace(-3, 3, 33)
    result = KineticEFIT(diag, kin, fi, r_grid, z_grid).reconstruct({})

    # Blank source rejected by the non-empty-text guard.
    with pytest.raises(ValueError, match="must be a non-empty string"):
        kinetic_efit_claim_evidence(
            result,
            kin,
            fi,
            source="",
            source_id="id",
            diagnostic_source="d",
            profile_source="p",
            fast_ion_source="f",
            mse_calibration_source="m",
        )

    # Non-finite tolerance rejected by the finiteness guard.
    with pytest.raises(ValueError, match="must be finite"):
        kinetic_efit_claim_evidence(
            result,
            kin,
            fi,
            source="p_efit_reference",
            source_id="id",
            diagnostic_source="d",
            profile_source="p",
            fast_ion_source="f",
            mse_calibration_source="m",
            pressure_relative_tolerance=float("nan"),
        )

    # Non-physical fast-ion model rejected (zero energy).
    with pytest.raises(ValueError, match="fast-ion evidence requires"):
        kinetic_efit_claim_evidence(
            result,
            kin,
            FastIonPressure(0.0, 0.1, 0.2),
            source="p_efit_reference",
            source_id="id",
            diagnostic_source="d",
            profile_source="p",
            fast_ion_source="f",
            mse_calibration_source="m",
        )

    # Non-finite reconstructed profile rejected.
    bad_result = _dc.replace(result, q_profile=np.full(50, np.nan))
    with pytest.raises(ValueError, match="non-empty finite one-dimensional"):
        kinetic_efit_claim_evidence(
            bad_result,
            kin,
            fi,
            source="p_efit_reference",
            source_id="id",
            diagnostic_source="d",
            profile_source="p",
            fast_ion_source="f",
            mse_calibration_source="m",
        )


def test_reconstruct_rejects_nonfinite_constraint_coordinates() -> None:
    """A non-finite (R, Z) constraint coordinate fails the rho-mapping guard."""
    diag = reference_diagnostics()
    kin = KineticConstraints(
        Te_points=[(6.0, 0.0, 10.0)],
        ne_points=[(float("nan"), 0.0, 5.0)],
        Ti_points=[(6.0, 0.0, 8.0)],
        mse_points=[],
    )
    fi = FastIonPressure(100.0, 0.1, 0.0)
    r_grid = np.linspace(4, 8, 33)
    z_grid = np.linspace(-3, 3, 33)
    with pytest.raises(ValueError, match="coordinates must be finite"):
        KineticEFIT(diag, kin, fi, r_grid, z_grid).reconstruct({})


def test_reconstruct_rejects_negative_constraint_values() -> None:
    """A negative measured constraint value fails the profile-value guard."""
    diag = reference_diagnostics()
    kin = KineticConstraints(
        Te_points=[(6.0, 0.0, 10.0)],
        ne_points=[(6.0, 0.0, -5.0)],
        Ti_points=[(6.0, 0.0, 8.0)],
        mse_points=[],
    )
    fi = FastIonPressure(100.0, 0.1, 0.0)
    r_grid = np.linspace(4, 8, 33)
    z_grid = np.linspace(-3, 3, 33)
    with pytest.raises(ValueError, match="values must be finite and non-negative"):
        KineticEFIT(diag, kin, fi, r_grid, z_grid).reconstruct({})


def test_kinetic_efit_claim_evidence_rejects_invalid_reference_inputs():
    diag = reference_diagnostics()
    kin = reference_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)
    r_grid = np.linspace(4, 8, 33)
    z_grid = np.linspace(-3, 3, 33)
    result = KineticEFIT(diag, kin, fi, r_grid, z_grid).reconstruct({})

    with pytest.raises(ValueError, match="source must be one of"):
        kinetic_efit_claim_evidence(
            result,
            kin,
            fi,
            source="untracked_reference",
            source_id="bad-source",
            diagnostic_source="documented magnetic diagnostics",
            profile_source="documented profiles",
            fast_ion_source="documented fast-ion model",
            mse_calibration_source="documented MSE calibration",
        )

    with pytest.raises(ValueError, match="reference must be finite and match"):
        kinetic_efit_claim_evidence(
            result,
            kin,
            fi,
            source="p_efit_reference",
            source_id="bad-pressure-reference",
            diagnostic_source="documented magnetic diagnostics",
            profile_source="documented profiles",
            fast_ion_source="documented fast-ion model",
            mse_calibration_source="documented MSE calibration",
            reference_pressure=result.p_kinetic[:-1],
        )

    with pytest.raises(ValueError, match="reference tolerances must be positive"):
        kinetic_efit_claim_evidence(
            result,
            kin,
            fi,
            source="p_efit_reference",
            source_id="bad-tolerance",
            diagnostic_source="documented magnetic diagnostics",
            profile_source="documented profiles",
            fast_ion_source="documented fast-ion model",
            mse_calibration_source="documented MSE calibration",
            pressure_relative_tolerance=0.0,
        )
