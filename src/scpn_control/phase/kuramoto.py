# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto-Sakaguchi phase dynamics
"""
Mean-field Kuramoto-Sakaguchi with exogenous global driver.

Equation:
    dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)

The ζ sin(Ψ−θ) term implements the reviewer's requested "intention as
carrier" injection.  Ψ is a Lagrangian pull parameter with no own
dynamics (no dotΨ equation) — it is resolved either from an external
value or from the mean-field phase.

Reference: arXiv:2004.06344 (generalized Kuramoto-Sakaguchi finite-size)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scpn_control._typing import FloatArray

logger = logging.getLogger(__name__)
KURAMOTO_RUNTIME_EVIDENCE_SCHEMA_VERSION = "scpn-control.kuramoto-runtime-evidence.v1"
KURAMOTO_RUNTIME_EVIDENCE_BOUNDED = "bounded_python_timestep_evidence_only"
KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED = "qualified_kuramoto_runtime_evidence"

# Rust fast-path (sub-ms for N > 1000)
try:
    from scpn_control_rs import kuramoto_step as _rust_step  # pragma: no cover

    RUST_KURAMOTO = True  # pragma: no cover
except ImportError:
    RUST_KURAMOTO = False


@dataclass(frozen=True)
class KuramotoRuntimeEvidence:
    """Tamper-evident phase-runtime parity and timestep-refinement evidence."""

    schema_version: str
    generated_utc: str
    target_id: str
    oscillator_count: int
    deployment_target_oscillators: int
    dt_s: float
    K: float
    alpha: float
    zeta: float
    psi_mode: str
    wrap: bool
    input_sha256: str
    python_reference_sha256: str
    rust_available: bool
    rust_parity_checked: bool
    rust_theta_max_abs_error_rad: float | None
    rust_order_parameter_abs_error: float | None
    parity_tolerance: float
    parity_passed: bool
    timestep_refinement_checked: bool
    timestep_refinement_error_rad: float
    timestep_refinement_tolerance: float
    timestep_refinement_passed: bool
    deployment_claim_allowed: bool
    claim_status: str
    payload_sha256: str


def _normalise_rust_step_result(rust_out: Any) -> tuple[FloatArray, float, float, float]:
    """Convert supported Rust binding return shapes to Python step fields."""
    if isinstance(rust_out, dict):
        return (
            np.asarray(rust_out["theta"], dtype=np.float64),
            float(rust_out["r"]),
            float(rust_out["psi_r"]),
            float(rust_out["psi_global"]),
        )

    th1, r_value, psi_r, psi_global = rust_out
    return (
        np.asarray(th1, dtype=np.float64),
        float(r_value),
        float(psi_r),
        float(psi_global),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in Kuramoto runtime evidence: {key}")
        seen.add(key)
        out[key] = value
    return out


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_nonnegative_float(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _nonnegative_float(name, value)


def _array_sha256(values: FloatArray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(_canonical_json({"shape": arr.shape, "values": arr.tolist()}).encode("utf-8")).hexdigest()


def _phase_max_abs_difference(a: FloatArray, b: FloatArray) -> float:
    return float(np.max(np.abs(wrap_phase(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))))


def _kuramoto_sakaguchi_step_numpy(
    th: FloatArray,
    om: FloatArray,
    *,
    dt: float,
    K: float,
    alpha: float,
    zeta: float,
    Psi: float,
    wrap: bool,
) -> dict[str, Any]:
    R, psi_r = order_parameter(th)
    dtheta = om + (K * R) * np.sin(psi_r - th - alpha)
    if zeta != 0.0:
        dtheta += zeta * np.sin(Psi - th)

    th1 = th + dt * dtheta
    if wrap:
        th1 = wrap_phase(th1)

    return {
        "theta1": th1,
        "dtheta": dtheta,
        "R": R,
        "Psi_r": psi_r,
        "Psi": Psi,
    }


def _kuramoto_reference_step(
    theta: FloatArray,
    omega: FloatArray,
    *,
    dt: float,
    K: float,
    alpha: float,
    zeta: float,
    psi_driver: float | None,
    psi_mode: str,
    wrap: bool,
) -> dict[str, Any]:
    Psi = GlobalPsiDriver(mode=psi_mode).resolve(theta, psi_driver)
    return _kuramoto_sakaguchi_step_numpy(
        theta,
        omega,
        dt=dt,
        K=K,
        alpha=alpha,
        zeta=zeta,
        Psi=Psi,
        wrap=wrap,
    )


def wrap_phase(x: FloatArray) -> FloatArray:
    """Map phases to (-π, π]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def order_parameter(
    theta: FloatArray,
    weights: FloatArray | None = None,
) -> tuple[float, float]:
    """Kuramoto order parameter R·exp(i·ψ_r) = <w·exp(i·θ)> / W.

    Returns (R, ψ_r).
    """
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim != 1:
        raise ValueError("theta must be a 1D phase vector")
    if not np.isfinite(th).all():
        raise ValueError("theta must contain only finite values")

    if weights is None:
        z = np.mean(np.exp(1j * th))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != th.shape:
            raise ValueError(f"weights shape {w.shape} != theta shape {th.shape}")
        if not np.isfinite(w).all():
            raise ValueError("weights must contain only finite values")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        W = float(np.sum(w))
        if W <= 0.0:
            raise ValueError("weights must have positive total mass")
        z = np.sum(w * np.exp(1j * th)) / W

    return float(np.abs(z)), float(np.angle(z))


@dataclass(frozen=True)
class GlobalPsiDriver:
    """Resolve the global field phase Ψ.

    mode="external"    : Ψ supplied by caller (intention/carrier, no dotΨ).
    mode="mean_field"  : Ψ = arg(<exp(iθ)>) from the oscillator population.
    """

    mode: str = "external"

    def resolve(self, theta: FloatArray, psi_external: float | None) -> float:
        if self.mode == "external":
            if psi_external is None:
                raise ValueError("psi_external required when mode='external'")
            if not np.isfinite(psi_external):
                raise ValueError("psi_external must be finite")
            return float(psi_external)
        if self.mode == "mean_field":
            _, psi = order_parameter(theta)
            return psi
        raise ValueError(f"Unknown mode: {self.mode}")


def lyapunov_v(theta: FloatArray, psi: float) -> float:
    """Lyapunov candidate V(t) = (1/N) Σ (1 − cos(θ_i − Ψ)).

    V=0 at perfect sync (all θ_i = Ψ), V=2 at maximal desync.
    Range: [0, 2].  Mirror of control-math/kuramoto.rs::lyapunov_v.
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.size == 0:
        return 0.0
    return float(np.mean(1.0 - np.cos(th - psi)))


def lyapunov_exponent(v_hist: Sequence[float], dt: float) -> float:
    """λ = (1/T) · ln(V_final / V_initial).  λ < 0 ⟹ stable."""
    if len(v_hist) < 2:
        return 0.0
    v0 = max(v_hist[0], 1e-15)
    vf = max(v_hist[-1], 1e-15)
    T = len(v_hist) * dt
    return float(np.log(vf / v0) / T)


def kuramoto_sakaguchi_step(
    theta: FloatArray,
    omega: FloatArray,
    *,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi_driver: float | None = None,
    psi_mode: str = "external",
    wrap: bool = True,
) -> dict[str, Any]:
    """Single Euler step of mean-field Kuramoto-Sakaguchi + global driver.

    dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)

    Uses Rust backend when available (rayon-parallelised, sub-ms for N>1000).
    """
    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    if th.ndim != 1:
        raise ValueError("theta must be a 1D phase vector")
    if om.ndim != 1:
        raise ValueError("omega must be a 1D frequency vector")
    if om.shape != th.shape:
        raise ValueError(f"omega shape {om.shape} != theta shape {th.shape}")
    if not np.isfinite(th).all():
        raise ValueError("theta must contain only finite values")
    if not np.isfinite(om).all():
        raise ValueError("omega must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(K) or K < 0.0:
        raise ValueError("K must be finite and non-negative")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(zeta):
        raise ValueError("zeta must be finite")

    # Resolve Ψ before dispatching (Rust kernel needs resolved value)
    Psi = GlobalPsiDriver(mode=psi_mode).resolve(th, psi_driver)

    if RUST_KURAMOTO and wrap:
        th1, R, psi_r, psi_g = _normalise_rust_step_result(
            _rust_step(
                th,
                om,
                dt,
                K,
                alpha,
                zeta,
                Psi,
            ),
        )
        return {
            "theta1": th1,
            "dtheta": th1 - th,  # approximate (post-wrap)
            "R": R,
            "Psi_r": psi_r,
            "Psi": psi_g,
        }

    return _kuramoto_sakaguchi_step_numpy(
        th,
        om,
        dt=dt,
        K=K,
        alpha=alpha,
        zeta=zeta,
        Psi=Psi,
        wrap=wrap,
    )


def _validate_kuramoto_runtime_payload(
    payload: Mapping[str, Any],
    *,
    require_deployment_claim: bool,
) -> KuramotoRuntimeEvidence:
    if payload.get("schema_version") != KURAMOTO_RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Kuramoto runtime evidence schema_version is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not _is_sha256(declared_digest):
        raise ValueError("Kuramoto runtime evidence payload_sha256 must be a SHA-256 hex digest")
    if declared_digest != _payload_sha256(payload):
        raise ValueError("Kuramoto runtime evidence payload_sha256 does not match payload")
    generated_utc = payload.get("generated_utc")
    if not isinstance(generated_utc, str) or not generated_utc.endswith("Z"):
        raise ValueError("Kuramoto runtime evidence generated_utc must be a UTC timestamp ending in Z")
    target_id = payload.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be non-empty")
    oscillator_count = _positive_int("oscillator_count", payload.get("oscillator_count"))
    deployment_target_oscillators = _positive_int(
        "deployment_target_oscillators",
        payload.get("deployment_target_oscillators"),
    )
    dt_s = _positive_float("dt_s", payload.get("dt_s"))
    coupling = _nonnegative_float("K", payload.get("K"))
    alpha = _finite_float("alpha", payload.get("alpha"))
    zeta = _finite_float("zeta", payload.get("zeta"))
    psi_mode = payload.get("psi_mode")
    if psi_mode not in {"external", "mean_field"}:
        raise ValueError("psi_mode is unsupported")
    bool_fields = (
        "wrap",
        "rust_available",
        "rust_parity_checked",
        "parity_passed",
        "timestep_refinement_checked",
        "timestep_refinement_passed",
        "deployment_claim_allowed",
    )
    for name in bool_fields:
        if not isinstance(payload.get(name), bool):
            raise ValueError(f"{name} must be boolean")
    input_sha256 = payload.get("input_sha256")
    python_reference_sha256 = payload.get("python_reference_sha256")
    if not _is_sha256(input_sha256):
        raise ValueError("input_sha256 must be a SHA-256 hex digest")
    if not _is_sha256(python_reference_sha256):
        raise ValueError("python_reference_sha256 must be a SHA-256 hex digest")
    rust_theta_max_abs_error_rad = _optional_nonnegative_float(
        "rust_theta_max_abs_error_rad",
        payload.get("rust_theta_max_abs_error_rad"),
    )
    rust_order_parameter_abs_error = _optional_nonnegative_float(
        "rust_order_parameter_abs_error",
        payload.get("rust_order_parameter_abs_error"),
    )
    parity_tolerance = _nonnegative_float("parity_tolerance", payload.get("parity_tolerance"))
    timestep_refinement_error_rad = _nonnegative_float(
        "timestep_refinement_error_rad",
        payload.get("timestep_refinement_error_rad"),
    )
    timestep_refinement_tolerance = _nonnegative_float(
        "timestep_refinement_tolerance",
        payload.get("timestep_refinement_tolerance"),
    )
    deployment_claim_allowed = bool(payload["deployment_claim_allowed"])
    expected_status = (
        KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED if deployment_claim_allowed else KURAMOTO_RUNTIME_EVIDENCE_BOUNDED
    )
    if payload.get("claim_status") != expected_status:
        raise ValueError("Kuramoto runtime evidence claim_status does not match deployment claim state")
    if bool(payload["rust_parity_checked"]):
        if rust_theta_max_abs_error_rad is None or rust_order_parameter_abs_error is None:
            raise ValueError("Rust parity evidence requires numerical error metrics")
    if require_deployment_claim or deployment_claim_allowed:
        if not deployment_claim_allowed:
            raise ValueError("Kuramoto runtime evidence is bounded-only and cannot support deployment claims")
        if oscillator_count < deployment_target_oscillators:
            raise ValueError("Kuramoto runtime evidence is below the declared deployment oscillator count")
        if not bool(payload["rust_available"]) or not bool(payload["rust_parity_checked"]):
            raise ValueError("Kuramoto runtime evidence requires Rust parity for deployment claims")
        if not bool(payload["parity_passed"]):
            raise ValueError("Kuramoto runtime evidence Rust parity did not pass")
        if not bool(payload["timestep_refinement_checked"]) or not bool(payload["timestep_refinement_passed"]):
            raise ValueError("Kuramoto runtime evidence timestep refinement did not pass")
        if rust_theta_max_abs_error_rad is None or rust_theta_max_abs_error_rad > parity_tolerance:
            raise ValueError("Kuramoto runtime evidence Rust phase parity exceeds tolerance")
        if rust_order_parameter_abs_error is None or rust_order_parameter_abs_error > parity_tolerance:
            raise ValueError("Kuramoto runtime evidence Rust order-parameter parity exceeds tolerance")
        if timestep_refinement_error_rad > timestep_refinement_tolerance:
            raise ValueError("Kuramoto runtime evidence timestep refinement exceeds tolerance")
    return KuramotoRuntimeEvidence(
        schema_version=str(payload["schema_version"]),
        generated_utc=generated_utc,
        target_id=target_id,
        oscillator_count=oscillator_count,
        deployment_target_oscillators=deployment_target_oscillators,
        dt_s=dt_s,
        K=coupling,
        alpha=alpha,
        zeta=zeta,
        psi_mode=str(psi_mode),
        wrap=bool(payload["wrap"]),
        input_sha256=str(input_sha256),
        python_reference_sha256=str(python_reference_sha256),
        rust_available=bool(payload["rust_available"]),
        rust_parity_checked=bool(payload["rust_parity_checked"]),
        rust_theta_max_abs_error_rad=rust_theta_max_abs_error_rad,
        rust_order_parameter_abs_error=rust_order_parameter_abs_error,
        parity_tolerance=parity_tolerance,
        parity_passed=bool(payload["parity_passed"]),
        timestep_refinement_checked=bool(payload["timestep_refinement_checked"]),
        timestep_refinement_error_rad=timestep_refinement_error_rad,
        timestep_refinement_tolerance=timestep_refinement_tolerance,
        timestep_refinement_passed=bool(payload["timestep_refinement_passed"]),
        deployment_claim_allowed=deployment_claim_allowed,
        claim_status=str(payload["claim_status"]),
        payload_sha256=str(declared_digest),
    )


def kuramoto_runtime_evidence(
    theta: FloatArray,
    omega: FloatArray,
    *,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi_driver: float | None = None,
    psi_mode: str = "external",
    wrap: bool = True,
    target_id: str = "local-python-runtime",
    deployment_target_oscillators: int | None = None,
    parity_tolerance: float = 1.0e-10,
    timestep_refinement_tolerance: float = 5.0e-3,
    generated_utc: str | None = None,
    deployment_claim_allowed: bool = False,
) -> KuramotoRuntimeEvidence:
    """Build bounded phase-runtime parity and timestep-refinement evidence."""

    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    if th.ndim != 1 or th.size == 0:
        raise ValueError("theta must be a non-empty 1D phase vector")
    if om.ndim != 1 or om.shape != th.shape:
        raise ValueError("omega must be a 1D frequency vector matching theta")
    if not np.isfinite(th).all():
        raise ValueError("theta must contain only finite values")
    if not np.isfinite(om).all():
        raise ValueError("omega must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(K) or K < 0.0:
        raise ValueError("K must be finite and non-negative")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(zeta):
        raise ValueError("zeta must be finite")
    reference = _kuramoto_reference_step(
        th,
        om,
        dt=dt,
        K=K,
        alpha=alpha,
        zeta=zeta,
        psi_driver=psi_driver,
        psi_mode=psi_mode,
        wrap=wrap,
    )
    half = _kuramoto_reference_step(
        th,
        om,
        dt=0.5 * dt,
        K=K,
        alpha=alpha,
        zeta=zeta,
        psi_driver=psi_driver,
        psi_mode=psi_mode,
        wrap=wrap,
    )
    refined = _kuramoto_reference_step(
        half["theta1"],
        om,
        dt=0.5 * dt,
        K=K,
        alpha=alpha,
        zeta=zeta,
        psi_driver=psi_driver,
        psi_mode=psi_mode,
        wrap=wrap,
    )
    parity_tolerance_value = _nonnegative_float("parity_tolerance", parity_tolerance)
    timestep_tolerance_value = _nonnegative_float(
        "timestep_refinement_tolerance",
        timestep_refinement_tolerance,
    )
    rust_available = bool(RUST_KURAMOTO and wrap)
    rust_parity_checked = False
    rust_theta_error: float | None = None
    rust_r_error: float | None = None
    if rust_available:
        Psi = GlobalPsiDriver(mode=psi_mode).resolve(th, psi_driver)
        rust_theta, rust_r, rust_psi_r, rust_psi = _normalise_rust_step_result(
            _rust_step(th, om, dt, K, alpha, zeta, Psi),
        )
        rust_parity_checked = True
        rust_theta_error = _phase_max_abs_difference(rust_theta, reference["theta1"])
        rust_r_error = max(
            abs(float(rust_r) - float(reference["R"])),
            _phase_max_abs_difference(np.array([rust_psi_r]), np.array([reference["Psi_r"]])),
            _phase_max_abs_difference(np.array([rust_psi]), np.array([reference["Psi"]])),
        )
    timestep_error = _phase_max_abs_difference(reference["theta1"], refined["theta1"])
    payload: dict[str, Any] = {
        "schema_version": KURAMOTO_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": generated_utc or _utc_now(),
        "target_id": str(target_id).strip(),
        "oscillator_count": int(th.size),
        "deployment_target_oscillators": int(deployment_target_oscillators or th.size),
        "dt_s": float(dt),
        "K": float(K),
        "alpha": float(alpha),
        "zeta": float(zeta),
        "psi_mode": str(psi_mode),
        "wrap": bool(wrap),
        "input_sha256": hashlib.sha256(
            _canonical_json(
                {
                    "theta": {"shape": th.shape, "values": th.tolist()},
                    "omega": {"shape": om.shape, "values": om.tolist()},
                    "psi_driver": psi_driver,
                },
            ).encode("utf-8"),
        ).hexdigest(),
        "python_reference_sha256": _array_sha256(reference["theta1"]),
        "rust_available": rust_available,
        "rust_parity_checked": rust_parity_checked,
        "rust_theta_max_abs_error_rad": rust_theta_error,
        "rust_order_parameter_abs_error": rust_r_error,
        "parity_tolerance": parity_tolerance_value,
        "parity_passed": bool(
            rust_parity_checked
            and rust_theta_error is not None
            and rust_r_error is not None
            and rust_theta_error <= parity_tolerance_value
            and rust_r_error <= parity_tolerance_value
        ),
        "timestep_refinement_checked": True,
        "timestep_refinement_error_rad": timestep_error,
        "timestep_refinement_tolerance": timestep_tolerance_value,
        "timestep_refinement_passed": bool(timestep_error <= timestep_tolerance_value),
        "deployment_claim_allowed": bool(deployment_claim_allowed),
        "claim_status": (
            KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED if deployment_claim_allowed else KURAMOTO_RUNTIME_EVIDENCE_BOUNDED
        ),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _validate_kuramoto_runtime_payload(payload, require_deployment_claim=bool(deployment_claim_allowed))


def assert_kuramoto_runtime_claim_admissible(
    evidence: KuramotoRuntimeEvidence,
) -> KuramotoRuntimeEvidence:
    """Fail closed unless phase-runtime evidence supports deployment claims."""

    if not isinstance(evidence, KuramotoRuntimeEvidence):
        raise ValueError("evidence must be KuramotoRuntimeEvidence")
    return _validate_kuramoto_runtime_payload(asdict(evidence), require_deployment_claim=True)


def save_kuramoto_runtime_evidence(evidence: KuramotoRuntimeEvidence, output_path: str | Path) -> None:
    """Persist Kuramoto runtime evidence as sorted JSON."""

    if not isinstance(evidence, KuramotoRuntimeEvidence):
        raise ValueError("evidence must be KuramotoRuntimeEvidence")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_kuramoto_runtime_evidence(
    path: str | Path,
    *,
    require_deployment_claim: bool = False,
) -> KuramotoRuntimeEvidence:
    """Load Kuramoto runtime evidence with duplicate-key and digest admission."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("Kuramoto runtime evidence must be a JSON object")
    return _validate_kuramoto_runtime_payload(payload, require_deployment_claim=require_deployment_claim)
