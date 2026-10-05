# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference RMSE calculations.
"""Compute bounded comparison lanes from existing repository reference carriers.

CSV/JSON/GEQDSK inputs mix reference, calibration and design records.
Calculations retain their historical formulas and units; no held-out facility
accuracy is inferred. Historical optional CONTROL burn imports remain distinct
from the modern FUSION contract, whose beta conversion has no approval.
Unavailable models yield an explicitly skipped lane rather than target values.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.core.eqdsk import GEqdsk, read_geqdsk

try:
    from scpn_control.core.fusion_ignition_sim import FusionBurnPhysics
except ImportError:
    FusionBurnPhysics = None
try:
    from scpn_control.diagnostics.forward import generate_forward_channels
except ImportError:
    generate_forward_channels = None


def ipb98_tau_e(
    ip_ma: float,
    b_t: float,
    n_e19: float,
    p_loss_mw: float,
    r_m: float,
    kappa: float,
    epsilon: float,
    a_eff_amu: float = 2.5,
) -> float:
    """Evaluate the fixed IPB98(y,2) power law with coefficient 0.0562.

    Parameters
    ----------
    ip_ma, b_t, n_e19, p_loss_mw, r_m
        Positive plasma current (MA), field (T), density (1e19 m^-3), loss
        power (MW) and major radius (m). Loss power is not auxiliary power.
    kappa, epsilon
        Positive dimensionless elongation and minor/major radius ratio.
    a_eff_amu
        Positive effective isotope mass in atomic mass units, default 2.5.

    Returns
    -------
    float
        Confinement time in seconds using the stated fixed coefficients.
        This function performs no empirical admission or domain repair.

    Raises
    ------
    ZeroDivisionError, OverflowError, TypeError
        Arithmetic cannot produce a real scalar for the supplied inputs.
        Callers are responsible for finite positive operating points.
    """
    return float(
        0.0562
        * (ip_ma**0.93)
        * (b_t**0.15)
        * (n_e19**0.41)
        * (p_loss_mw**-0.69)
        * (r_m**1.97)
        * (kappa**0.78)
        * (epsilon**0.58)
        * (a_eff_amu**0.19)
    )


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Return sqrt(mean squared paired differences)) in the input quantity's units.

    Parameters
    ----------
    y_true, y_pred
        Nonempty equal-length scalar sequences in the same unit and order.
        Values are converted to float; finiteness is not checked here.

    Returns
    -------
    float
        Absolute unweighted RMSE. No normalisation or missing-value filtering
        is applied, and nonfinite values propagate through the arithmetic.

    Raises
    ------
    ValueError
        Sequences are empty, lengths differ or values cannot convert to float.
    OverflowError
        Squared differences exceed the representable range.

    Examples
    --------
    >>> rmse([1.0, 3.0], [2.0, 4.0])
    1.0
    """
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("RMSE requires non-empty lists of equal length.")
    return math.sqrt(statistics.mean((float(t) - float(p)) ** 2 for t, p in zip(y_true, y_pred)))


def load_json(path: Path) -> dict[str, Any]:
    """Decode a UTF-8 reference JSON object without model/provenance admission.

    File, decoding and JSON syntax errors propagate. Reference consumers check
    their own required fields; duplicate keys and numeric metadata are not
    validated by this legacy reader. A non-object root raises ValueError.
    """
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("reference JSON must be an object")
    return data


def compare_eq_axis(eq: GEqdsk) -> float:
    """Compare a GEQDSK grid flux extremum to its own declared axis, in metres.

    For simag < sibry use argmin(psirz), otherwise argmax. On equal extrema
    NumPy's first flat index wins. Coordinates use eq.r/eq.z; no sub-grid fit
    is performed. This is internal file consistency, not reconstruction
    error against an independent measured equilibrium. Input grids/shapes
    must be compatible; indexing failures propagate.
    """
    if eq.simag < eq.sibry:
        idx = int(np.argmin(eq.psirz))
    else:
        idx = int(np.argmax(eq.psirz))
    iz, ir = np.unravel_index(idx, eq.psirz.shape)
    r_psi = eq.r[ir]
    z_psi = eq.z[iz]
    return float(math.hypot(r_psi - eq.rmaxis, z_psi - eq.zmaxis))


def confinement_rmse_itpa(csv_path: Path) -> dict[str, Any]:
    """Compare CSV confinement seconds and H98 values against the fixed scaling.

    The tracked ITPA carrier mixes calibration/design/public reference inputs;
    a machine/shot label does not attest a measured shot. Rows supply Ip_MA,
    BT_T, ne19_1e19m3, Ploss_MW, R_m, a_m, kappa, M_AMU, tau_E_s and H98y2.
    Compute tau RMSE in seconds, mean absolute percentage error over positive
    reference tau and dimensionless H98 RMSE using tau_ref/tau_prediction.
    Return count and per-row values. Missing/empty files return skipped true,
    count zero and compatibility zero metrics, which the CI guard rejects.
    Malformed rows, invalid arithmetic and file errors otherwise propagate.
    """
    if not csv_path.exists():
        return {
            "count": 0,
            "tau_rmse_s": 0.0,
            "tau_mae_rel_pct": 0.0,
            "h98_rmse": 0.0,
            "rows": [],
            "skipped": True,
            "reason": f"missing input file: {csv_path}",
        }

    tau_true: list[float] = []
    tau_pred: list[float] = []
    h98_true: list[float] = []
    h98_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau_m = float(row["tau_E_s"])
            tau_p = ipb98_tau_e(
                ip_ma=float(row["Ip_MA"]),
                b_t=float(row["BT_T"]),
                n_e19=float(row["ne19_1e19m3"]),
                p_loss_mw=float(row["Ploss_MW"]),
                r_m=float(row["R_m"]),
                kappa=float(row["kappa"]),
                epsilon=float(row["a_m"]) / float(row["R_m"]),
                a_eff_amu=float(row["M_AMU"]),
            )
            h98_m = float(row["H98y2"])
            h98_p = tau_m / tau_p if tau_p > 0 else 0.0

            tau_true.append(tau_m)
            tau_pred.append(tau_p)
            h98_true.append(h98_m)
            h98_pred.append(h98_p)
            rows.append(
                {
                    "machine": row["machine"],
                    "shot": row["shot"],
                    "tau_measured_s": tau_m,
                    "tau_pred_s": tau_p,
                    "h98_measured": h98_m,
                    "h98_pred": h98_p,
                }
            )

    if not tau_true:
        return {
            "count": 0,
            "tau_rmse_s": 0.0,
            "tau_mae_rel_pct": 0.0,
            "h98_rmse": 0.0,
            "rows": [],
            "skipped": True,
            "reason": "no rows in ITPA confinement dataset",
        }

    tau_rel_abs_pct = [abs(t - p) / t * 100.0 for t, p in zip(tau_true, tau_pred) if t > 0]
    return {
        "count": len(tau_true),
        "tau_rmse_s": rmse(tau_true, tau_pred),
        "tau_mae_rel_pct": statistics.mean(tau_rel_abs_pct),
        "h98_rmse": rmse(h98_true, h98_pred),
        "rows": rows,
    }


def confinement_rmse_iter_sparc(reference_dir: Path) -> dict[str, Any]:
    """Compare fixed-law confinement seconds to ITER and SPARC design inputs.

    Read iter_reference.json and sparc_reference.json with explicit MA/T/MW/m
    units and density in 1e19 m^-3. Return two-row absolute tau RMSE in seconds,
    prediction/reference values and signed percentage errors. A zero reference
    has compatibility percentage zero; no held-out validation is inferred.
    Missing files, fields and invalid arithmetic propagate without fallback.
    """
    refs = [
        load_json(reference_dir / "iter_reference.json"),
        load_json(reference_dir / "sparc_reference.json"),
    ]
    tau_true: list[float] = []
    tau_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    for ref in refs:
        pred = ipb98_tau_e(
            ip_ma=float(ref["I_p_MA"]),
            b_t=float(ref["B_t_T"]),
            n_e19=float(ref["n_e_1e19"]),
            p_loss_mw=float(ref["P_loss_MW"]),
            r_m=float(ref["R_m"]),
            kappa=float(ref["kappa"]),
            epsilon=float(ref["a_m"]) / float(ref["R_m"]),
            a_eff_amu=float(ref["A_eff_amu"]),
        )
        obs = float(ref["tau_E_s"])
        tau_true.append(obs)
        tau_pred.append(pred)
        rows.append(
            {
                "scenario": ref["scenario"],
                "tau_measured_s": obs,
                "tau_pred_s": pred,
                "relative_error_pct": ((pred - obs) / obs * 100.0) if obs else 0.0,
            }
        )

    return {
        "count": len(tau_true),
        "tau_rmse_s": rmse(tau_true, tau_pred),
        "rows": rows,
    }


def estimate_beta_n_from_burn(
    reference: dict[str, Any],
    config_path: Path,
) -> tuple[float, dict[str, Any]]:
    """Estimate beta_N from dynamic burn model steady-state.

    Uses ``DynamicBurnModel`` which evolves temperature self-consistently
    with Bosch-Hale D-T reactivity and IPB98(y,2) confinement scaling,
    then converts the steady-state stored energy to ``beta_N`` via the
    Troyon-like definition:

        beta_N = 100 * beta_t * a * B_t / I_p

    A profile-peaking correction factor (``PROFILE_PEAKING_FACTOR``) is
    applied to account for the volume-averaged 0-D model underestimating
    peak pressure.  The factor was calibrated as the geometric mean of
    the per-machine corrections for ITER (target 1.8) and SPARC
    (target 1.0):

        c_ITER  = 1.8 / beta_n_raw_ITER  = 1.488
        c_SPARC = 1.0 / beta_n_raw_SPARC = 1.404
        PROFILE_PEAKING_FACTOR = sqrt(c_ITER * c_SPARC) ~= 1.446

    The adapter does not evolve a radial pressure profile or derive this
    correction from one; it is a historical design-target calibration.

    Falls back to an actual legacy ``FusionBurnPhysics`` run if the dynamic
    model fails. Both models are optional historical imports.

    Parameters
    ----------
    reference
        Design inputs: R_m/a_m in metres, B_t_T in tesla, I_p_MA in MA,
        P_aux_MW in MW, n_e_1e19 in 1e19 m^-3 and dimensionless kappa.
    config_path
        Legacy model equilibrium configuration, used only on that fallback.

    Returns
    -------
    tuple[float, dict[str, Any]]
        Dimensionless beta_N and actual model fusion power (MW), Q and stored
        energy (MJ). Calibration against these design targets prevents a
        held-out prediction claim.

    Raises
    ------
    RuntimeError
        Neither actual model can supply a result. Reference beta/Q/power are
        never substituted for unavailable predictions.
    KeyError, ValueError
        Required reference fields are missing or cannot be converted.
    """
    # Profile peaking correction: geometric mean calibration against
    # ITER (beta_N = 1.8) and SPARC (beta_N = 1.0) targets.
    PROFILE_PEAKING_FACTOR = 1.446

    r_m = float(reference["R_m"])
    a_m = float(reference["a_m"])
    kappa = float(reference["kappa"])
    b_t = float(reference["B_t_T"])
    i_p = float(reference["I_p_MA"])
    n_e20 = float(reference["n_e_1e19"]) / 10.0  # 10^19 -> 10^20
    p_aux = float(reference["P_aux_MW"])

    try:
        from scpn_control.core.fusion_ignition_sim import DynamicBurnModel

        model = DynamicBurnModel(
            R0=r_m,
            a=a_m,
            B_t=b_t,
            I_p=i_p,
            kappa=kappa,
            n_e20=n_e20,
            M_eff=2.5,
        )
        result = model.simulate(P_aux_mw=p_aux, duration_s=100.0, dt_s=0.01)

        # beta_N from steady-state W_thermal
        w_thermal_j = float(result["W_MJ"][-1]) * 1e6
        volume = model.V_plasma
        p_avg = w_thermal_j / (3.0 * volume) if volume > 0 else 0.0
        mu0 = 4.0 * math.pi * 1e-7
        beta_t = (2.0 * mu0 * p_avg / (b_t * b_t)) if b_t > 0 else 0.0
        beta_n = (100.0 * beta_t) * a_m * b_t / i_p if i_p > 0 else 0.0
        beta_n *= PROFILE_PEAKING_FACTOR

        metrics: dict[str, Any] = {
            "P_fusion_MW": result["P_fus_final_MW"],
            "Q": result["Q_final"],
            "W_MJ": result["W_MJ"][-1],
        }
        return beta_n, metrics
    except Exception as exc:
        if FusionBurnPhysics is None:
            raise RuntimeError("burn model unavailable; reference values cannot replace model predictions") from exc

        sim = FusionBurnPhysics(str(config_path))
        sim.solve_equilibrium()
        metrics = sim.calculate_thermodynamics(P_aux_MW=p_aux)

        w_thermal_j = float(metrics["W_MJ"]) * 1e6
        volume = 2.0 * math.pi * math.pi * r_m * a_m * a_m * kappa
        p_avg = w_thermal_j / (3.0 * volume) if volume > 0 else 0.0
        mu0 = 4.0 * math.pi * 1e-7
        beta_t_val = (2.0 * mu0 * p_avg / (b_t * b_t)) if b_t > 0 else 0.0
        beta_n = (100.0 * beta_t_val) * a_m * b_t / i_p if i_p > 0 else 0.0
        return beta_n, metrics


def beta_rmse_iter_sparc(reference_dir: Path, validation_dir: Path) -> dict[str, Any]:
    """Compare actual burn-model beta_N estimates to the two design targets.

    Read iter_reference.json and sparc_reference.json from reference_dir,
    pairing the legacy equilibrium configurations from validation_dir. A
    missing model returns count zero, null beta_n_rmse, no prediction rows,
    skipped true and a reason. Available results are calibrated design
    comparisons, not independent validation. File/field failures propagate.
    """
    pairs = [
        ("iter_reference.json", "iter_validated_config.json"),
        ("sparc_reference.json", "sparc_config.json"),
    ]
    beta_true: list[float] = []
    beta_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    for ref_name, cfg_name in pairs:
        ref = load_json(reference_dir / ref_name)
        beta_obs = float(ref["beta_N"])
        try:
            beta_est, metrics = estimate_beta_n_from_burn(ref, validation_dir / cfg_name)
        except RuntimeError as exc:
            return {"count": 0, "beta_n_rmse": None, "rows": [], "skipped": True, "reason": str(exc)}
        beta_true.append(beta_obs)
        beta_pred.append(beta_est)
        rows.append(
            {
                "scenario": ref["scenario"],
                "beta_n_measured": beta_obs,
                "beta_n_estimated": beta_est,
                "relative_error_pct": ((beta_est - beta_obs) / beta_obs * 100.0) if beta_obs else 0.0,
                "model_q": float(metrics["Q"]),
                "model_p_fusion_mw": float(metrics["P_fusion_MW"]),
            }
        )

    return {
        "count": len(beta_true),
        "beta_n_rmse": rmse(beta_true, beta_pred),
        "rows": rows,
    }


def sparc_axis_rmse(sparc_dir: Path) -> dict[str, Any]:
    """Aggregate self-file axis errors over sorted GEQDSK and EQDSK carriers.

    Return file count, per-file errors and RMSE in metres. Both extensions are
    read in sorted groups, GEQDSK first. These are grid-extremum consistency
    comparisons on mixed reference records, not facility reconstruction proof.
    Empty directories raise ValueError through rmse; malformed inputs propagate.
    """
    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for path in files:
        eq = read_geqdsk(path)
        err = compare_eq_axis(eq)
        errors.append(err)
        rows.append({"file": path.name, "axis_error_m": err})
    return {"count": len(errors), "axis_rmse_m": rmse(errors, [0.0] * len(errors)), "rows": rows}


def forward_diagnostics_rmse() -> dict[str, Any]:
    """Compare fixed synthetic Gaussian channels to deliberately biased profiles.

    On the fixed 33x33 R/Z grid, density is in m^-3, temperature in keV and
    neutron source in m^-3 s^-1. The second evaluation scales these profiles
    by 0.985/1.04/1.03 respectively. Return phase RMSE (rad), voltage RMSE (V)
    and neutron-rate relative error (percent); this checks synthetic channel
    plumbing, not detector calibration or measured facility accuracy. Missing
    optional forward code returns an explicit skipped lane and reason.
    """
    if generate_forward_channels is None:
        return {
            "count_interferometer_channels": 0,
            "phase_rmse_rad": 0.0,
            "neutron_rate_rel_error_pct": 0.0,
            "skipped": True,
            "reason": "forward diagnostics module unavailable",
        }

    r = np.linspace(4.0, 8.0, 33)
    z = np.linspace(-2.0, 2.0, 33)
    rr, zz = np.meshgrid(r, z)
    electron_density = 4.8e19 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 0.8)
    electron_temp = 12.0 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 1.0)
    neutron_source = 7.5e15 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 0.65)

    chords = [
        ((4.2, -0.8), (7.8, 0.8)),
        ((4.2, 0.0), (7.8, 0.0)),
        ((4.2, 0.8), (7.8, -0.8)),
    ]
    baseline = generate_forward_channels(
        electron_density_m3=electron_density,
        electron_temp_keV=electron_temp,
        neutron_source_m3_s=neutron_source,
        r_grid=r,
        z_grid=z,
        interferometer_chords=chords,
        volume_element_m3=float((r[1] - r[0]) * (z[1] - z[0])),
    )

    # Surrogate "prediction" lane with slight profile bias to quantify channel RMSE.
    pred = generate_forward_channels(
        electron_density_m3=electron_density * 0.985,
        electron_temp_keV=electron_temp * 1.04,
        neutron_source_m3_s=neutron_source * 1.03,
        r_grid=r,
        z_grid=z,
        interferometer_chords=chords,
        volume_element_m3=float((r[1] - r[0]) * (z[1] - z[0])),
    )

    phase_true = baseline.interferometer_phase_rad.tolist()
    phase_pred = pred.interferometer_phase_rad.tolist()
    phase_rmse = rmse(phase_true, phase_pred)
    rate_true = baseline.neutron_count_rate_hz
    rate_pred = pred.neutron_count_rate_hz
    rate_rel_pct = abs(rate_pred - rate_true) / max(rate_true, 1e-12) * 100.0
    thomson_true = baseline.thomson_scattering_voltage_v.tolist()
    thomson_pred = pred.thomson_scattering_voltage_v.tolist()
    thomson_rmse = rmse(thomson_true, thomson_pred)
    return {
        "count_interferometer_channels": len(chords),
        "count_thomson_channels": len(thomson_true),
        "phase_rmse_rad": phase_rmse,
        "neutron_rate_rel_error_pct": rate_rel_pct,
        "thomson_voltage_rmse_v": thomson_rmse,
        "rows": [
            {
                "channel": i,
                "phase_true_rad": t,
                "phase_pred_rad": p,
            }
            for i, (t, p) in enumerate(zip(phase_true, phase_pred))
        ],
    }
