#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — IPB98(y,2)-form confinement prefactor fit.
"""Fit the prefactor of an IPB98(y,2)-form confinement law to the ITPA H-mode CSV.

Despite the historical file name, this tool does not calibrate the local
gyro-Bohm coefficient ``c_gB`` of the transport solvers. That coefficient is
dimensionless and multiplies ``rho_s^2 * c_s / (a * q * R)``. The quantity
fitted here is the global, dimensional prefactor ``C`` of

    tau_E = C * Ip^0.93 * B^0.15 * n^0.41 * P^(-0.69)
              * R^1.97 * kappa^0.78 * epsilon^0.58 * M^0.19 * f_neo

in the units of the CSV columns (MA, T, 10^19 m^-3, MW, m, amu, s). The
exponents are the published IPB98(y,2) exponents (Nuclear Fusion 39, 1999,
2175) and are not refitted. ``f_neo`` in [0.5, 1] lowers ``tau_E`` when a
Chang-Hinton (1982) neoclassical ion diffusivity, evaluated at mid-radius
from a power-balance temperature estimate, is large compared with the
effective diffusivity implied by the prediction.

With the shipped CSV and the fitted ``C``, ``f_neo`` sits at its 0.5 floor
for 16 of the 20 rows, so for most rows the correction is a constant halving
and the fitted ``C`` is about twice the effective prefactor.

``C`` minimises the absolute RMSE of ``tau_E`` over the CSV rows with a
bounded scalar search in ``log10(C)``. Its one-sigma uncertainty comes from a
seeded bootstrap over the rows. The 20 rows are mixed machine and design
points, so the fit is an internal consistency check of this confinement form,
not a validation of any transport model.

Outputs
-------
Writes ``validation/reference_data/itpa/ipb98_prefactor_fit.json`` unless
another path is given. Writing to the solvers' coefficient file
``gyro_bohm_coefficients.json`` is refused, because a dimensional global
prefactor cannot stand in for the local coefficient. The command exits 0 when
the relative RMSE is below 15 %, 1 when it is not, and 2 when the input or
output path is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
ITPA_CSV = REPO_ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
OUTPUT_JSON = REPO_ROOT / "validation" / "reference_data" / "itpa" / "ipb98_prefactor_fit.json"
SOLVER_COEFFICIENT_JSON = REPO_ROOT / "validation" / "reference_data" / "itpa" / "gyro_bohm_coefficients.json"

# ── Physical constants ────────────────────────────────────────────────

E_CHARGE = 1.602176634e-19  # C
M_PROTON = 1.672621924e-27  # kg
EPS_0 = 8.854187812e-12  # F/m
E_KEV_J = 1.602176634e-16  # J per keV

#: Relative RMSE below which the command reports success.
RELATIVE_RMSE_TARGET = 0.15

#: Published IPB98(y,2) exponents applied to (Ip, B, n, P, R, kappa, epsilon, M).
IPB98_EXPONENTS: dict[str, float] = {
    "Ip": 0.93,
    "B": 0.15,
    "n": 0.41,
    "P": -0.69,
    "R": 1.97,
    "kappa": 0.78,
    "epsilon": 0.58,
    "M": 0.19,
}

_POSITIVE_COLUMNS = ("Ip_MA", "BT_T", "ne19_1e19m3", "Ploss_MW", "R_m", "a_m", "kappa", "M_AMU", "tau_E_s", "H98y2")


# ── Data container ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShotRecord:
    """One row of the ITPA H-mode confinement CSV.

    The CSV columns are machine, shot, Ip_MA, BT_T, ne19_1e19m3, Ploss_MW,
    R_m, a_m, kappa, delta, M_AMU, tau_E_s, H98y2 and source. Triangularity
    and H98y2 are carried but not used by the confinement form. The loader
    requires every numeric column except triangularity to be positive; the
    prediction functions assume it.
    """

    machine: str
    shot: str
    Ip_MA: float
    BT_T: float
    ne19: float  # line-averaged density [10^19 m^-3]
    Ploss_MW: float
    R_m: float  # major radius [m]
    a_m: float  # minor radius [m]
    kappa: float
    delta: float  # triangularity
    M_AMU: float  # effective ion mass [amu]
    tau_E_s: float  # measured or design confinement time [s]
    H98y2: float
    source: str


def _finite(row: dict[str, str], column: str, line: int) -> float:
    """Return one finite numeric cell, naming its CSV line on refusal."""
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"CSV line {line}: column {column} is missing or not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"CSV line {line}: column {column} is not finite")
    if column in _POSITIVE_COLUMNS and value <= 0.0:
        raise ValueError(f"CSV line {line}: column {column} must be positive")
    return value


def load_itpa_csv(path: Path = ITPA_CSV) -> list[ShotRecord]:
    """Load the ITPA H-mode confinement CSV.

    Parameters
    ----------
    path : pathlib.Path
        CSV with the columns listed on :class:`ShotRecord`.

    Returns
    -------
    list[ShotRecord]
        One record per data row, in file order.

    Raises
    ------
    ValueError
        If the file has no data rows, or a numeric cell is missing, not finite,
        or not positive where the confinement form takes a power of it.
    OSError
        If the file cannot be read.
    """
    records: list[ShotRecord] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            records.append(
                ShotRecord(
                    machine=row.get("machine") or "",
                    shot=row.get("shot") or "",
                    Ip_MA=_finite(row, "Ip_MA", line),
                    BT_T=_finite(row, "BT_T", line),
                    ne19=_finite(row, "ne19_1e19m3", line),
                    Ploss_MW=_finite(row, "Ploss_MW", line),
                    R_m=_finite(row, "R_m", line),
                    a_m=_finite(row, "a_m", line),
                    kappa=_finite(row, "kappa", line),
                    delta=_finite(row, "delta", line),
                    M_AMU=_finite(row, "M_AMU", line),
                    tau_E_s=_finite(row, "tau_E_s", line),
                    H98y2=_finite(row, "H98y2", line),
                    source=row.get("source") or "",
                )
            )
    if not records:
        raise ValueError(f"{path} has no data rows")
    logger.info("Loaded %d shots from %s", len(records), path.name)
    return records


# ── Chang-Hinton neoclassical chi ─────────────────────────────────────


def chang_hinton_chi_scalar(
    rho_norm: float,
    T_i_keV: float,
    n_e_19: float,
    q: float,
    R0: float,
    a: float,
    B0: float,
    A_ion: float = 2.0,
    Z_eff: float = 1.5,
) -> float:
    """Return the Chang-Hinton (1982) neoclassical ion diffusivity [m^2/s] at one point.

    Parameters
    ----------
    rho_norm : float
        Normalised minor radius; the inverse aspect ratio is ``rho_norm * a / R0``.
    T_i_keV : float
        Ion temperature [keV].
    n_e_19 : float
        Electron density [10^19 m^-3]; ion density is taken equal to it.
    q : float
        Safety factor.
    R0, a : float
        Major and minor radius [m].
    B0 : float
        Magnetic field [T].
    A_ion : float
        Ion mass number.
    Z_eff : float
        Effective charge.

    Returns
    -------
    float
        ``0.66 (1 + 1.54 eps) q^2 rho_i^2 nu_ii / (eps^1.5 (1 + 0.74 nu*^(2/3)))``
        with a Coulomb logarithm of 17, floored at 0.01. A non-positive radius,
        temperature, density, safety factor, geometry, field or mass, an inverse
        aspect ratio below 1e-6, or a non-finite result also returns 0.01.
    """
    if min(rho_norm, T_i_keV, n_e_19, q, R0, a, B0, A_ion) <= 0:
        return 0.01

    m_i = A_ion * M_PROTON
    epsilon = rho_norm * a / R0
    if epsilon < 1e-6:
        return 0.01

    T_J = T_i_keV * E_KEV_J
    v_ti = math.sqrt(2.0 * T_J / m_i)
    rho_i = m_i * v_ti / (E_CHARGE * B0)

    n_e = n_e_19 * 1e19
    ln_lambda = 17.0
    nu_ii = n_e * Z_eff**2 * E_CHARGE**4 * ln_lambda / (12.0 * math.pi**1.5 * EPS_0**2 * m_i**0.5 * T_J**1.5)

    eps32 = epsilon**1.5
    nu_star = nu_ii * q * R0 / (eps32 * v_ti)

    chi_val = 0.66 * (1.0 + 1.54 * epsilon) * q**2 * rho_i**2 * nu_ii / (eps32 * (1.0 + 0.74 * nu_star ** (2.0 / 3.0)))
    return max(chi_val, 0.01) if math.isfinite(chi_val) else 0.01


# ── Predict tau_E for one shot ────────────────────────────────────────


def ipb98_shape(shot: ShotRecord) -> float:
    """Return the IPB98(y,2) power-law product of one shot without its prefactor.

    Parameters
    ----------
    shot : ShotRecord
        One CSV row.

    Returns
    -------
    float
        ``Ip^0.93 B^0.15 n^0.41 P^-0.69 R^1.97 kappa^0.78 epsilon^0.58 M^0.19``
        in the CSV units, with ``epsilon = a / R``.
    """
    e = IPB98_EXPONENTS
    return float(
        shot.Ip_MA ** e["Ip"]
        * shot.BT_T ** e["B"]
        * shot.ne19 ** e["n"]
        * shot.Ploss_MW ** e["P"]
        * shot.R_m ** e["R"]
        * shot.kappa ** e["kappa"]
        * (shot.a_m / shot.R_m) ** e["epsilon"]
        * shot.M_AMU ** e["M"]
    )


def neoclassical_factor(shot: ShotRecord, tau_base: float) -> float:
    """Return the factor ``f_neo`` in [0.5, 1] that lowers a predicted ``tau_E``.

    The stored energy ``P_loss * tau_base`` and the plasma volume
    ``2 pi^2 R a^2 kappa`` give a volume-averaged temperature, clamped to
    [0.1, 50] keV. With the cylindrical safety factor
    ``max(5 a^2 B kappa / (R Ip), 2)``, the Chang-Hinton diffusivity
    ``chi_neo`` is evaluated at mid-radius. The effective diffusivity of the
    prediction is ``chi_eff = a^2 kappa^0.78 epsilon^0.58 / (2 tau_base)``.

    Parameters
    ----------
    shot : ShotRecord
        One CSV row.
    tau_base : float
        Prediction before the factor [s], positive.

    Returns
    -------
    float
        ``max(chi_eff / (chi_eff + chi_neo), 0.5)``.
    """
    R0, a, B0, kappa = shot.R_m, shot.a_m, shot.BT_T, shot.kappa
    volume = 2.0 * math.pi**2 * R0 * a**2 * kappa
    stored_energy_j = shot.Ploss_MW * 1e6 * tau_base
    t_avg_kev = stored_energy_j / (3.0 * shot.ne19 * 1e19 * volume * E_KEV_J)
    t_avg_kev = min(max(t_avg_kev, 0.1), 50.0)
    q_avg = max(5.0 * a**2 * B0 * kappa / (R0 * shot.Ip_MA), 2.0)
    chi_neo = chang_hinton_chi_scalar(0.5, t_avg_kev, shot.ne19, q_avg, R0, a, B0, shot.M_AMU)
    chi_eff = float(a**2 * kappa**0.78 * (a / R0) ** 0.58 / (2.0 * tau_base))
    return max(chi_eff / (chi_eff + chi_neo), 0.5)


def predict_tau_e(shot: ShotRecord, prefactor: float) -> float:
    """Predict the energy confinement time [s] of one shot.

    Parameters
    ----------
    shot : ShotRecord
        One CSV row.
    prefactor : float
        IPB98(y,2)-form prefactor ``C``, positive.

    Returns
    -------
    float
        ``C * ipb98_shape(shot) * neoclassical_factor(shot, C * ipb98_shape(shot))``,
        floored at 1e-4 s.
    """
    tau_base = prefactor * ipb98_shape(shot)
    return max(tau_base * neoclassical_factor(shot, tau_base), 1e-4)


# ── Fit metrics ───────────────────────────────────────────────────────


def compute_rmse(prefactor: float, shots: list[ShotRecord]) -> float:
    """Return the RMSE [s] of predicted against tabulated ``tau_E`` over ``shots``."""
    residuals = [(predict_tau_e(shot, prefactor) - shot.tau_E_s) ** 2 for shot in shots]
    return math.sqrt(sum(residuals) / len(residuals))


def compute_log_rmse(prefactor: float, shots: list[ShotRecord]) -> float:
    """Return the RMS of ``log10(predicted / tabulated)``, which weighs every row equally."""
    ratios = [math.log10(predict_tau_e(shot, prefactor) / shot.tau_E_s) ** 2 for shot in shots]
    return math.sqrt(sum(ratios) / len(ratios))


def objective(log_prefactor: float, shots: list[ShotRecord]) -> float:
    """Return :func:`compute_rmse` at ``C = 10**log_prefactor`` for the bounded search."""
    return compute_rmse(10.0**log_prefactor, shots)


def _fit(shots: list[ShotRecord], low: float, high: float, xatol: float) -> tuple[float, float]:
    """Minimise :func:`objective` over ``[low, high]`` in ``log10(C)``; return ``C`` and its RMSE."""
    result = minimize_scalar(
        objective, bounds=(low, high), args=(shots,), method="bounded", options={"xatol": xatol, "maxiter": 200}
    )
    return 10.0 ** float(result.x), float(result.fun)


def bootstrap_uncertainty(
    shots: list[ShotRecord],
    prefactor_best: float,
    n_bootstrap: int = 200,
    rng_seed: int = 42,
) -> float:
    """Return the bootstrap standard deviation of the fitted prefactor.

    Parameters
    ----------
    shots : list[ShotRecord]
        Rows of the full fit.
    prefactor_best : float
        Full-data prefactor; each resample is refitted within 1.5 decades of it.
    n_bootstrap : int
        Number of resamples with replacement.
    rng_seed : int
        Seed of ``numpy.random.default_rng``; equal seeds give equal results.

    Returns
    -------
    float
        Population standard deviation of the resampled prefactors.
    """
    rng = np.random.default_rng(rng_seed)
    centre = math.log10(prefactor_best)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(shots), size=len(shots))
        samples.append(_fit([shots[int(i)] for i in indices], centre - 1.5, centre + 1.5, 0.01)[0])
    return float(np.std(samples))


# ── Main fit routine ──────────────────────────────────────────────────


def _refuse_solver_coefficient_file(output_path: Path) -> None:
    """Refuse an output that would overwrite the solvers' local coefficient file."""
    if output_path.resolve() == SOLVER_COEFFICIENT_JSON.resolve():
        raise ValueError(
            f"refusing to write {SOLVER_COEFFICIENT_JSON.name}: the fitted IPB98(y,2) prefactor is not "
            "the local gyro-Bohm coefficient c_gB that the transport solvers read from it"
        )


def calibrate(
    csv_path: Path | None = None,
    output_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Fit the IPB98(y,2)-form prefactor and write the result as JSON.

    Parameters
    ----------
    csv_path : pathlib.Path, optional
        ITPA CSV; defaults to :data:`ITPA_CSV`.
    output_path : pathlib.Path, optional
        JSON destination; defaults to :data:`OUTPUT_JSON`. Its parent is
        created; an existing file is replaced.
    verbose : bool
        Print a per-row comparison table to stdout.

    Returns
    -------
    dict[str, object]
        The written record: ``ipb98_prefactor``, its bootstrap uncertainty,
        absolute, relative, logarithmic and mean-absolute-percentage errors,
        ``n_shots``, the UTC fit time, the form and its claim boundary.

    Raises
    ------
    ValueError
        If ``output_path`` is the solvers' coefficient file or the CSV is refused.
    OSError
        If the CSV cannot be read or the output cannot be written.
    """
    csv_path = csv_path or ITPA_CSV
    output_path = output_path or OUTPUT_JSON
    _refuse_solver_coefficient_file(output_path)

    shots = load_itpa_csv(csv_path)
    prefactor, rmse = _fit(shots, -2.0, 4.0, 0.005)
    mean_tau = sum(shot.tau_E_s for shot in shots) / len(shots)
    rmse_rel = rmse / mean_tau
    logger.info("IPB98(y,2)-form prefactor = %.6f, RMSE = %.4f s (%.2f %%)", prefactor, rmse, 100 * rmse_rel)

    if verbose:
        print("\n" + "=" * 90)
        print(f"{'Machine':<12} {'Shot':<12} {'tau_meas':>8} {'tau_pred':>8} {'error':>8} {'rel_err':>8}")
        print("-" * 90)
        for shot in shots:
            tau_pred = predict_tau_e(shot, prefactor)
            err = tau_pred - shot.tau_E_s
            print(
                f"{shot.machine:<12} {shot.shot:<12} {shot.tau_E_s:8.4f} {tau_pred:8.4f} {err:+8.4f} {err / shot.tau_E_s:+8.2%}"
            )
        print("=" * 90)

    uncertainty = bootstrap_uncertainty(shots, prefactor)
    mape = sum(abs(predict_tau_e(shot, prefactor) - shot.tau_E_s) / shot.tau_E_s for shot in shots) / len(shots)

    record: dict[str, object] = {
        "name": "ipb98_form_confinement_prefactor_fit",
        "quantity": (
            "Dimensional prefactor C of tau_E = C * Ip^0.93 * B^0.15 * n^0.41 * P^(-0.69) * R^1.97 "
            "* kappa^0.78 * epsilon^0.58 * M^0.19 * f_neo in the CSV units (MA, T, 1e19 m^-3, MW, m, amu, s)"
        ),
        "claim_boundary": (
            "Fit of a global confinement form to mixed machine and design rows; not the local gyro-Bohm "
            "coefficient c_gB of the transport solvers and not a transport-model validation"
        ),
        "ipb98_prefactor": round(prefactor, 6),
        "ipb98_prefactor_uncertainty": round(uncertainty, 6),
        "rmse_s": round(rmse, 6),
        "rmse_percent": round(100 * rmse_rel, 4),
        "rmse_relative": round(rmse_rel, 6),
        "log_rmse": round(compute_log_rmse(prefactor, shots), 4),
        "mape_percent": round(100 * mape, 2),
        "mean_tabulated_tau_s": round(mean_tau, 6),
        "n_shots": len(shots),
        "fit_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "scipy.optimize.minimize_scalar, bounded, in log10(C) over [-2, 4]; 200-sample seeded bootstrap",
        "exponents": dict(IPB98_EXPONENTS),
        "exponent_source": "IPB98(y,2), Nuclear Fusion 39 (1999) 2175",
        "neoclassical_model": "Chang-Hinton (1982), Phys. Fluids 25, 1493, at mid-radius",
        "reference": "ITPA H-mode confinement CSV (hmode_confinement.csv)",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved the fit to %s", output_path)
    return record


def main(argv: list[str] | None = None) -> int:
    """Run the fit from the command line.

    Parameters
    ----------
    argv : list[str], optional
        Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 when the relative RMSE is below :data:`RELATIVE_RMSE_TARGET`, 1 when
        it is not, and 2 when the CSV or the output path is refused.
    """
    parser = argparse.ArgumentParser(description="Fit the IPB98(y,2)-form confinement prefactor to the ITPA CSV.")
    parser.add_argument("--csv", type=Path, default=ITPA_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--quiet", action="store_true", help="omit the per-row table")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    try:
        record = calibrate(args.csv, args.output, verbose=not args.quiet)
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    rmse_relative = cast(float, record["rmse_relative"])
    if rmse_relative < RELATIVE_RMSE_TARGET:
        print(f"PASS: relative RMSE {100 * rmse_relative:.2f} % < {100 * RELATIVE_RMSE_TARGET:.0f} %")
        return 0
    print(f"FAIL: relative RMSE {100 * rmse_relative:.2f} % >= {100 * RELATIVE_RMSE_TARGET:.0f} %")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
