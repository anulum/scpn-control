# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Burn controller
"""D-T burn-control, alpha-heating, Lawson, and auxiliary-heating utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.uncertainty import bosch_hale_reactivity

# ── Fusion physics constants ──────────────────────────────────────────
# E_alpha = 3.52 MeV: DT alpha particle birth energy.
# ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Ch. 2.
E_ALPHA_J = 3.52e6 * 1.602176634e-19  # J

# E_fus = 17.6 MeV: total DT reaction energy release.
# Wesson 2011, "Tokamaks" 4th ed., Eq. 1.2.1.
E_FUS_J = 17.6e6 * 1.602176634e-19  # J

# Lawson criterion (ignition): n τ_E T > 3×10^21 m^-3 s keV.
# Lawson 1957, Proc. Phys. Soc. B 70, 6.
LAWSON_TRIPLE_PRODUCT = 3.0e21  # m^-3 s keV
_BURN_CLAIM_SCHEMA_VERSION = 1
_FACILITY_BURN_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "integrated_transport_benchmark", "measured_burn_replay"}
)
_BOUNDED_BURN_REFERENCE_SOURCES = frozenset({"repository_burn_regression", *_FACILITY_BURN_REFERENCE_SOURCES})

# Burn fraction reference: f_b ≈ a² n_DT <σv> / (4 v_th).
# Wesson 2011, Eq. 1.7.3.


def _require_positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scalar


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Return a finite non-negative scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _require_nonnegative_profile(name: str, values: AnyFloatArray, shape: tuple[int, ...] | None = None) -> FloatArray:
    """Return a finite non-negative profile with an optional exact shape."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must contain non-negative values")
    return arr


def _require_normalised_rho(rho: AnyFloatArray) -> FloatArray:
    rho_arr = _require_nonnegative_profile("rho", rho)
    if np.any(rho_arr > 1.0):
        raise ValueError("rho must stay within the normalised interval [0, 1]")
    if rho_arr.size > 1 and np.any(np.diff(rho_arr) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return rho_arr


@dataclass(frozen=True)
class BurnControlClaimEvidence:
    """Serialisable evidence for bounded or validated burn-control claims."""

    schema_version: int
    source: str
    source_id: str
    model_id: str
    R0_m: float
    a_m: float
    kappa: float
    profile_points: int
    ne_volume_average_1e20_m3: float
    Te_volume_average_keV: float
    Ti_volume_average_keV: float
    P_alpha_MW: float
    P_aux_MW: float
    Q: float
    lawson_triple_product: float
    lawson_margin: float
    burn_fraction: float
    reactivity_exponent: float
    thermally_stable: bool
    controller_Q_target: float
    controller_T_target_keV: float
    controller_P_aux_max_MW: float
    controller_command_MW: float
    reference_source: str | None
    reference_dataset_id: str | None
    reference_artifact_sha256: str | None
    reference_case_count: int | None
    P_alpha_relative_error: float | None
    Q_abs_error: float | None
    lawson_margin_abs_error: float | None
    burn_fraction_relative_error: float | None
    reactivity_exponent_abs_error: float | None
    P_alpha_relative_tolerance: float
    Q_abs_tolerance: float
    lawson_margin_abs_tolerance: float
    burn_fraction_relative_tolerance: float
    reactivity_exponent_abs_tolerance: float
    reactor_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _weighted_average(values: AnyFloatArray, rho: AnyFloatArray) -> float:
    rho_arr = _require_normalised_rho(rho)
    value_arr = _require_nonnegative_profile("values", values, rho_arr.shape)
    weights = rho_arr.copy()
    if np.all(weights == 0.0):
        return float(value_arr[0])
    _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    numerator = _trapz(value_arr * weights, rho_arr)
    denominator = _trapz(weights, rho_arr)
    return float(numerator / denominator)


def _sha256_text(name: str, value: object) -> str:
    text = _non_empty_text(name, str(value))
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return text


def _positive_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _extract_burn_reference_artifact(reference_artifact: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    if reference_artifact is None:
        return None, False
    if not isinstance(reference_artifact, dict):
        raise ValueError("reference_artifact must be a dictionary")
    source = _non_empty_text("reference_artifact.source", str(reference_artifact.get("source", "")))
    if source not in _FACILITY_BURN_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_FACILITY_BURN_REFERENCE_SOURCES))
        raise ValueError(f"reference_artifact.source must be one of: {allowed}")
    units = reference_artifact.get("units")
    expected_units = {
        "density": "m^-3",
        "temperature": "keV",
        "power": "MW",
        "time": "s",
        "reactivity": "m^3/s",
        "triple_product": "m^-3 s keV",
        "dimensionless": "1",
    }
    if not isinstance(units, dict) or any(units.get(key) != unit for key, unit in expected_units.items()):
        raise ValueError("reference_artifact.units must declare burn-control unit contracts")
    _sha256_text("reference_artifact.reference_artifact_sha256", reference_artifact.get("reference_artifact_sha256"))
    case_count = reference_artifact.get("reference_case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("reference_artifact.reference_case_count must be a positive integer")
    metrics = reference_artifact.get("metrics")
    tolerances = reference_artifact.get("tolerances")
    if not isinstance(metrics, dict) or not isinstance(tolerances, dict):
        raise ValueError("reference_artifact metrics and tolerances must be dictionaries")
    for metric in (
        "P_alpha_relative_error",
        "Q_abs_error",
        "lawson_margin_abs_error",
        "burn_fraction_relative_error",
        "reactivity_exponent_abs_error",
    ):
        observed = _nonnegative_reference_scalar(f"reference_artifact.metrics.{metric}", metrics.get(metric))
        tolerance = _positive_reference_scalar(f"reference_artifact.tolerances.{metric}", tolerances.get(metric))
        if observed > tolerance:
            raise ValueError(f"reference_artifact metric {metric} exceeds declared tolerance")
    return reference_artifact, True


def burn_control_claim_evidence(
    alpha_heating: AlphaHeating,
    controller: BurnController,
    *,
    rho: AnyFloatArray,
    ne_20: AnyFloatArray,
    Te_keV: AnyFloatArray,
    Ti_keV: AnyFloatArray,
    tau_E_s: float,
    P_aux_MW: float,
    source: str,
    source_id: str,
    model_id: str = "bounded_dt_burn_control",
    reference_artifact: dict[str, Any] | None = None,
    P_alpha_relative_tolerance: float = 0.05,
    Q_abs_tolerance: float = 0.5,
    lawson_margin_abs_tolerance: float = 0.2,
    burn_fraction_relative_tolerance: float = 0.10,
    reactivity_exponent_abs_tolerance: float = 0.25,
) -> BurnControlClaimEvidence:
    """Build fail-closed DT burn-control evidence from explicit profiles."""

    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_BURN_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_BURN_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    rho_arr = _require_normalised_rho(rho)
    ne_arr = _require_nonnegative_profile("ne_20", ne_20, rho_arr.shape)
    te_arr = _require_nonnegative_profile("Te_keV", Te_keV, rho_arr.shape)
    ti_arr = _require_nonnegative_profile("Ti_keV", Ti_keV, rho_arr.shape)
    tau_E = _require_positive_scalar("tau_E_s", tau_E_s)
    P_aux = _require_nonnegative_scalar("P_aux_MW", P_aux_MW)
    P_alpha = alpha_heating.power(ne_arr, te_arr, ti_arr, rho_arr)
    Q = alpha_heating.Q(P_alpha, P_aux)
    ne_avg_20 = _weighted_average(ne_arr, rho_arr)
    te_avg = _weighted_average(te_arr, rho_arr)
    ti_avg = _weighted_average(ti_arr, rho_arr)
    triple_product = lawson_triple_product(ne_avg_20 * 1.0e20, tau_E, ti_avg)
    lawson_margin = triple_product / LAWSON_TRIPLE_PRODUCT
    sigv = float(bosch_hale_reactivity(ti_avg)) if ti_avg > 0.0 else 0.0
    thermal_speed = max(float(np.sqrt(max(ti_avg, 0.0) * 1.0e3 * 1.602176634e-19 / (2.5 * 1.67262192369e-27))), 1.0)
    burn = burn_fraction(ne_avg_20 * 1.0e20, sigv, thermal_speed, alpha_heating.a)
    stability = BurnStabilityAnalysis(alpha_heating)
    reactivity_exp = stability.reactivity_exponent(ti_avg)
    command = controller.step(Q_meas=0.0 if not np.isfinite(Q) else Q, T_meas_keV=ti_avg, P_alpha_MW=P_alpha, dt=0.1)
    artifact, artifact_passed = _extract_burn_reference_artifact(reference_artifact)
    reactor_claim_allowed = bool(source_clean in _FACILITY_BURN_REFERENCE_SOURCES and artifact_passed)
    claim_status = "validated_burn_reference_matched" if reactor_claim_allowed else "bounded_burn_control_evidence"
    metrics = artifact.get("metrics", {}) if artifact else {}

    return BurnControlClaimEvidence(
        schema_version=_BURN_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        model_id=_non_empty_text("model_id", model_id),
        R0_m=float(alpha_heating.R0),
        a_m=float(alpha_heating.a),
        kappa=float(alpha_heating.kappa),
        profile_points=int(rho_arr.size),
        ne_volume_average_1e20_m3=ne_avg_20,
        Te_volume_average_keV=te_avg,
        Ti_volume_average_keV=ti_avg,
        P_alpha_MW=float(P_alpha),
        P_aux_MW=P_aux,
        Q=float(Q),
        lawson_triple_product=float(triple_product),
        lawson_margin=float(lawson_margin),
        burn_fraction=float(burn),
        reactivity_exponent=float(reactivity_exp),
        thermally_stable=bool(stability.is_thermally_stable(ti_avg)),
        controller_Q_target=float(controller.Q_target),
        controller_T_target_keV=float(controller.T_target),
        controller_P_aux_max_MW=float(controller.P_aux_max),
        controller_command_MW=float(command),
        reference_source=None if artifact is None else str(artifact["source"]),
        reference_dataset_id=None if artifact is None else str(artifact["reference_dataset_id"]),
        reference_artifact_sha256=None if artifact is None else str(artifact["reference_artifact_sha256"]),
        reference_case_count=None if artifact is None else int(artifact["reference_case_count"]),
        P_alpha_relative_error=None if artifact is None else float(metrics["P_alpha_relative_error"]),
        Q_abs_error=None if artifact is None else float(metrics["Q_abs_error"]),
        lawson_margin_abs_error=None if artifact is None else float(metrics["lawson_margin_abs_error"]),
        burn_fraction_relative_error=None if artifact is None else float(metrics["burn_fraction_relative_error"]),
        reactivity_exponent_abs_error=None if artifact is None else float(metrics["reactivity_exponent_abs_error"]),
        P_alpha_relative_tolerance=_positive_reference_scalar("P_alpha_relative_tolerance", P_alpha_relative_tolerance),
        Q_abs_tolerance=_positive_reference_scalar("Q_abs_tolerance", Q_abs_tolerance),
        lawson_margin_abs_tolerance=_positive_reference_scalar(
            "lawson_margin_abs_tolerance", lawson_margin_abs_tolerance
        ),
        burn_fraction_relative_tolerance=_positive_reference_scalar(
            "burn_fraction_relative_tolerance", burn_fraction_relative_tolerance
        ),
        reactivity_exponent_abs_tolerance=_positive_reference_scalar(
            "reactivity_exponent_abs_tolerance", reactivity_exponent_abs_tolerance
        ),
        reactor_claim_allowed=reactor_claim_allowed,
        claim_status=claim_status,
    )


def assert_burn_control_reactor_claim_admissible(evidence: BurnControlClaimEvidence) -> BurnControlClaimEvidence:
    """Raise when burn-control evidence is insufficient for reactor-control claims."""

    if not isinstance(evidence, BurnControlClaimEvidence):
        raise ValueError("evidence must be BurnControlClaimEvidence")
    if not evidence.reactor_claim_allowed:
        raise ValueError("reactor burn-control claim requires matched integrated or measured reference evidence")
    return evidence


def save_burn_control_claim_evidence(evidence: BurnControlClaimEvidence, path: str | Path) -> None:
    """Persist burn-control claim evidence as deterministic JSON."""

    if not isinstance(evidence, BurnControlClaimEvidence):
        raise ValueError("evidence must be BurnControlClaimEvidence")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AlphaHeating:
    """Alpha heating power from D-T fusion.

    P_α = n_D n_T <σv> E_α × V.
    ITER Physics Basis 1999, Nucl. Fusion 39, 2137.
    Reactivity <σv>_DT: Bosch & Hale 1992, Nucl. Fusion 32, 611.
    """

    def __init__(self, R0: float, a: float, kappa: float = 1.0):
        self.R0 = _require_positive_scalar("R0", R0)
        self.a = _require_positive_scalar("a", a)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        self.kappa = _require_positive_scalar("kappa", kappa)
        self.E_alpha_J = E_ALPHA_J

    def power_density(self, ne_20: AnyFloatArray, Te_keV: AnyFloatArray, Ti_keV: AnyFloatArray) -> FloatArray:
        """Alpha power density [MW/m^3] for 50:50 D-T mixture.

        p_α = n_D n_T <σv>(T_i) E_α.
        ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Eq. (2.2.1).
        """
        ne_arr = _require_nonnegative_profile("ne_20", ne_20)
        _require_nonnegative_profile("Te_keV", Te_keV, ne_arr.shape)
        ti_arr = _require_nonnegative_profile("Ti_keV", Ti_keV, ne_arr.shape)

        ne_m3 = ne_arr * 1e20
        nD = ne_m3 / 2.0
        nT = ne_m3 / 2.0

        # <σv>_DT: Bosch & Hale 1992, Nucl. Fusion 32, 611, Table IV
        positive_ti = ti_arr > 0.0
        sigv = np.zeros_like(ti_arr, dtype=float)
        if np.any(positive_ti):
            sigv[positive_ti] = np.asarray(bosch_hale_reactivity(ti_arr[positive_ti]), dtype=float)

        p_alpha_W = nD * nT * sigv * self.E_alpha_J
        return np.asarray(p_alpha_W / 1e6, dtype=float)

    def power(self, ne_20: AnyFloatArray, Te_keV: AnyFloatArray, Ti_keV: AnyFloatArray, rho: AnyFloatArray) -> float:
        """P_alpha [MW] integrated over plasma volume.

        dV = 4π² R₀ a² κ ρ dρ (torus shell element in normalised radius).
        """
        rho_arr = _require_normalised_rho(rho)
        ne_arr = _require_nonnegative_profile("ne_20", ne_20, rho_arr.shape)
        te_arr = _require_nonnegative_profile("Te_keV", Te_keV, rho_arr.shape)
        ti_arr = _require_nonnegative_profile("Ti_keV", Ti_keV, rho_arr.shape)
        p_dens = self.power_density(ne_arr, te_arr, ti_arr)

        dV = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.kappa * rho_arr

        _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        P_tot = _trapz(p_dens * dV, rho_arr)
        return float(P_tot)

    def Q(self, P_alpha_MW: float, P_aux_MW: float) -> float:
        """Fusion energy gain Q = P_fus / P_aux = 5 P_α / P_aux.

        P_fus = 5 P_α because α carries 1/5 of total DT energy (E_α/E_fus = 3.52/17.6).
        Delayed feedback for alpha-heating runaway:
        Mitarai & Muraoka 1999, Nucl. Fusion 39, 725.
        """
        P_alpha_MW = _require_nonnegative_scalar("P_alpha_MW", P_alpha_MW)
        P_aux_MW = _require_nonnegative_scalar("P_aux_MW", P_aux_MW)
        if P_aux_MW <= 0.0:
            return float("inf") if P_alpha_MW > 0 else 0.0
        return 5.0 * P_alpha_MW / P_aux_MW


def lawson_triple_product(ne_m3: float, tau_E_s: float, T_keV: float) -> float:
    """Return n τ_E T [m^-3 s keV].

    Ignition requires n τ_E T > 3×10^21 m^-3 s keV.
    Lawson 1957, Proc. Phys. Soc. B 70, 6.
    """
    ne_m3 = _require_nonnegative_scalar("ne_m3", ne_m3)
    tau_E_s = _require_positive_scalar("tau_E_s", tau_E_s)
    T_keV = _require_nonnegative_scalar("T_keV", T_keV)
    return ne_m3 * tau_E_s * T_keV


def burn_fraction(n_dt_m3: float, sigv: float, v_th_ms: float, a_m: float) -> float:
    """Approximate DT burn fraction.

    f_b ≈ a² n_DT <σv> / (4 v_th).
    Wesson 2011, "Tokamaks" 4th ed., Eq. 1.7.3.
    """
    n_dt_m3 = _require_nonnegative_scalar("n_dt_m3", n_dt_m3)
    sigv = _require_nonnegative_scalar("sigv", sigv)
    v_th_ms = _require_positive_scalar("v_th_ms", v_th_ms)
    a_m = _require_positive_scalar("a_m", a_m)
    return (a_m**2 * n_dt_m3 * sigv) / (4.0 * v_th_ms)


class BurnStabilityAnalysis:
    """Thermal burn stability based on reactivity exponent.

    Delayed alpha-heating feedback control:
    Mitarai & Muraoka 1999, Nucl. Fusion 39, 725.
    """

    def __init__(self, alpha_heating: AlphaHeating):
        self.alpha_heating = alpha_heating

    def reactivity_exponent(self, Ti_keV: float) -> float:
        """d(ln <σv>) / d(ln T), evaluated via finite difference.

        Stability requires this exponent < 2.
        Mitarai & Muraoka 1999, Nucl. Fusion 39, 725, Eq. (5).
        """
        Ti_keV = _require_nonnegative_scalar("Ti_keV", Ti_keV)
        if Ti_keV <= 0.1:
            return 10.0

        dT = 0.01 * Ti_keV
        sv_arr_plus = np.asarray(bosch_hale_reactivity(np.array([Ti_keV + dT])))
        sv_arr_minus = np.asarray(bosch_hale_reactivity(np.array([Ti_keV - dT])))
        sv_plus = sv_arr_plus[0]
        sv_minus = sv_arr_minus[0]

        if sv_minus <= 0 or sv_plus <= 0:
            return 10.0

        d_ln_sv = np.log(sv_plus) - np.log(sv_minus)
        d_ln_T = np.log(Ti_keV + dT) - np.log(Ti_keV - dT)

        return float(d_ln_sv / d_ln_T)

    def is_thermally_stable(self, Ti_keV: float) -> bool:
        """True if d(ln <σv>)/d(ln T) < 2 (Mitarai & Muraoka 1999)."""
        return self.reactivity_exponent(Ti_keV) < 2.0

    def stability_boundary_keV(self) -> float:
        """T where d(ln <σv>)/d(ln T) = 2 (bisection, 5–30 keV)."""
        T_low = 5.0
        T_high = 30.0

        for _ in range(20):
            T_mid = (T_low + T_high) / 2.0
            if self.reactivity_exponent(T_mid) > 2.0:
                T_low = T_mid
            else:
                T_high = T_mid

        return T_high


class BurnController:
    """PI burn controller targeting Q and T_i.

    Delayed feedback prevents alpha-heating runaway.
    Mitarai & Muraoka 1999, Nucl. Fusion 39, 725.
    P_aux_max = 73 MW matches ITER heating system capacity.
    ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Table I.
    """

    def __init__(self, Q_target: float = 10.0, T_target_keV: float = 20.0, P_aux_max_MW: float = 73.0):
        self.Q_target = _require_positive_scalar("Q_target", Q_target)
        self.T_target = _require_positive_scalar("T_target_keV", T_target_keV)
        self.P_aux_max = _require_positive_scalar("P_aux_max_MW", P_aux_max_MW)

        self.integral_T = 0.0

        self.K_T_p = -5.0  # MW/keV; proportional gain
        self.K_T_i = -1.0  # MW/(keV·s); integral gain

        self.last_P_aux = P_aux_max_MW / 2.0

    def step(self, Q_meas: float, T_meas_keV: float, P_alpha_MW: float, dt: float) -> float:
        """Advance the burn-temperature PI controller one step.

        Suppresses auxiliary heating above 30 keV to prevent alpha runaway
        (Mitarai & Muraoka 1999).

        Parameters
        ----------
        Q_meas
            Measured fusion gain; must be non-negative.
        T_meas_keV
            Measured ion temperature in keV; must be non-negative.
        P_alpha_MW
            Alpha-heating power in MW; must be non-negative.
        dt
            Time step in seconds; must be positive.

        Returns
        -------
        float
            The commanded auxiliary heating power in MW, clipped to
            ``[0, P_aux_max]``.
        """
        _require_nonnegative_scalar("Q_meas", Q_meas)
        T_meas_keV = _require_nonnegative_scalar("T_meas_keV", T_meas_keV)
        _require_nonnegative_scalar("P_alpha_MW", P_alpha_MW)
        dt = _require_positive_scalar("dt", dt)
        # > 30 keV: suppress heating to prevent alpha runaway (Mitarai & Muraoka 1999)
        if T_meas_keV > 30.0:
            self.last_P_aux = 0.0
            return 0.0

        e_T = T_meas_keV - self.T_target
        self.integral_T += e_T * dt

        # P_aux = P_ff + P_fb
        # If we just want to stabilize T:
        P_fb = self.K_T_p * e_T + self.K_T_i * self.integral_T

        # Base power to sustain target
        # Simplified: rely on integral to find it
        P_cmd = self.P_aux_max / 2.0 + P_fb

        P_cmd = np.clip(P_cmd, 0.0, self.P_aux_max)
        self.last_P_aux = P_cmd

        return float(P_cmd)


@dataclass
class BurnPoint:
    """A candidate burn operating point from a temperature scan.

    Attributes
    ----------
    Te_keV
        Electron temperature in keV.
    P_alpha_MW
        Alpha-heating power in MW.
    P_loss_MW
        Energy-loss power in MW.
    Q
        Fusion gain at this point.
    stable
        Whether the point is thermally stable.
    """

    Te_keV: float
    P_alpha_MW: float
    P_loss_MW: float
    Q: float
    stable: bool


class SubignitedBurnPoint:
    """Sub-ignited burn operating-point finder via 0-D power balance.

    Parameters
    ----------
    alpha_heating
        The alpha-heating model providing the geometry and P_α(T).
    """

    def __init__(self, alpha_heating: AlphaHeating):
        self.alpha = alpha_heating
        self.stability = BurnStabilityAnalysis(alpha_heating)

    def find_operating_point(self, ne_20: float, P_aux_MW: float, tau_E_s: float) -> list[BurnPoint]:
        """Scan T to find P_α(T) + P_aux = P_loss(T) intersections.

        P_loss = 3 n T V / τ_E (energy balance, 0-D model).
        Lawson criterion: n τ_E T > 3×10^21 m^-3 s keV for ignition.
        Lawson 1957, Proc. Phys. Soc. B 70, 6.
        """
        ne_20 = _require_positive_scalar("ne_20", ne_20)
        P_aux_MW = _require_nonnegative_scalar("P_aux_MW", P_aux_MW)
        tau_E_s = _require_positive_scalar("tau_E_s", tau_E_s)
        T_scan = np.linspace(1.0, 40.0, 400)
        points = []

        V = 2.0 * np.pi**2 * self.alpha.R0 * self.alpha.a**2 * self.alpha.kappa
        e_charge = 1.602e-19

        P_alphas = np.zeros_like(T_scan)
        P_losses = np.zeros_like(T_scan)

        for i, T in enumerate(T_scan):
            ne_arr = np.array([ne_20])
            T_arr = np.array([T])
            p_dens = self.alpha.power_density(ne_arr, T_arr, T_arr)[0]
            P_alphas[i] = p_dens * V

            # P_loss = 3 n T V / τ_E (Lawson 1957, Proc. Phys. Soc. B 70, 6)
            W_J = 3.0 * (ne_20 * 1e20) * (T * 1e3 * e_charge) * V
            P_losses[i] = (W_J / tau_E_s) / 1e6

        P_net = P_alphas + P_aux_MW - P_losses

        # Find zero crossings
        crossings = np.where(np.diff(np.sign(P_net)))[0]

        for idx in crossings:
            T_cross = T_scan[idx]
            P_a = P_alphas[idx]
            P_l = P_losses[idx]
            Q = self.alpha.Q(P_a, P_aux_MW)

            # Stable if dP_net/dT < 0 (loss grows faster than source)
            dP_net_dT = P_net[idx + 1] - P_net[idx]
            stable = dP_net_dT < 0

            points.append(
                BurnPoint(
                    Te_keV=float(T_cross), P_alpha_MW=float(P_a), P_loss_MW=float(P_l), Q=float(Q), stable=bool(stable)
                )
            )

        return points


__all__ = [
    "AlphaHeating",
    "BurnControlClaimEvidence",
    "BurnController",
    "BurnPoint",
    "BurnStabilityAnalysis",
    "E_ALPHA_J",
    "E_FUS_J",
    "LAWSON_TRIPLE_PRODUCT",
    "SubignitedBurnPoint",
    "assert_burn_control_reactor_claim_admissible",
    "burn_control_claim_evidence",
    "burn_fraction",
    "lawson_triple_product",
    "save_burn_control_claim_evidence",
]
