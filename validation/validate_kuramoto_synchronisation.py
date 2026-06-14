#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto-Sakaguchi published-benchmark synchronisation validation
"""Validate the phase runtime against analytically exact Kuramoto results.

The repository phase runtime (``src/scpn_control/phase/kuramoto.py``) implements
the mean-field Kuramoto-Sakaguchi step. This module checks that the runtime
reproduces the analytically exact mean-field results for a Lorentzian (Cauchy)
natural-frequency distribution ``g(ω) = (γ/π) / (ω² + γ²)``, ``g(0) = 1/(πγ)``.

Validated published references:

- **Critical coupling.** Synchronisation onsets at
  ``Kc(α) = 2 / (π · g(0) · cos α) = 2γ / cos α``. The ``α = 0`` form
  ``Kc = 2γ`` is Kuramoto's original result; the ``1/cos α`` shift is the
  Sakaguchi-Kuramoto phase-lag generalisation.
- **Partially synchronised branch (α = 0, exact for the Lorentzian).** Above
  threshold the steady mean-field order parameter follows
  ``R∞(K) = sqrt(1 − Kc/K)`` for ``K ≥ Kc`` and ``R∞ = 0`` below.

References:
  Kuramoto Y. (1975) in *International Symposium on Mathematical Problems in
  Theoretical Physics*, Lect. Notes Phys. 39, 420.
  Sakaguchi H., Kuramoto Y. (1986) *Prog. Theor. Phys.* 76, 576.
  Strogatz S. H. (2000) "From Kuramoto to Crawford", *Physica D* 143, 1.
  Acebrón J. A. et al. (2005) *Rev. Mod. Phys.* 77, 137.

The simulation is a finite-size, finite-time estimate of the thermodynamic
limit, so the order parameter is compared within a declared tolerance rather
than bit-exactly; sub-threshold incoherence is checked against a finite-size
ceiling that scales as ``O(1/sqrt(N))``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step

FloatArray = NDArray[np.float64]

KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION = "scpn-control.kuramoto-synchronisation-validation.v1"


def critical_coupling(gamma: float, alpha: float = 0.0) -> float:
    """Mean-field critical coupling ``Kc(α) = 2γ / cos α`` for a Lorentzian.

    ``g(0) = 1/(πγ)`` gives ``Kc = 2/(π g(0) cos α) = 2γ/cos α``. The phase lag
    must satisfy ``|α| < π/2`` for a finite positive threshold.
    """
    gamma = _positive_float("gamma", gamma)
    alpha = _finite_float("alpha", alpha)
    if abs(alpha) >= math.pi / 2.0:
        raise ValueError("alpha must satisfy |alpha| < pi/2 for a finite critical coupling")
    return 2.0 * gamma / math.cos(alpha)


def synchronised_order_parameter(coupling: float, gamma: float, alpha: float = 0.0) -> float:
    """Exact Lorentzian partially synchronised branch ``R∞ = sqrt(1 − Kc/K)``.

    Exact for ``α = 0``; for ``α ≠ 0`` it is the leading-order branch about the
    rotated locked cluster and is used only as an onset reference, so callers
    validating the order-parameter magnitude must pass ``α = 0``.
    """
    coupling = _positive_float("coupling", coupling)
    kc = critical_coupling(gamma, alpha)
    if coupling <= kc:
        return 0.0
    return math.sqrt(1.0 - kc / coupling)


@dataclass(frozen=True)
class SynchronisationCase:
    """One simulated operating point for the synchronisation validation."""

    n_oscillators: int
    gamma: float
    coupling: float
    alpha: float
    dt_s: float
    n_steps: int
    n_average: int
    seed: int

    def __post_init__(self) -> None:
        _positive_int("n_oscillators", self.n_oscillators)
        _positive_float("gamma", self.gamma)
        _nonnegative_float("coupling", self.coupling)
        _finite_float("alpha", self.alpha)
        _positive_float("dt_s", self.dt_s)
        _positive_int("n_steps", self.n_steps)
        _positive_int("n_average", self.n_average)
        if self.n_average > self.n_steps:
            raise ValueError("n_average must not exceed n_steps")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


def lorentzian_frequencies(n_oscillators: int, gamma: float, rng: np.random.Generator) -> FloatArray:
    """Draw ``n`` Cauchy/Lorentzian natural frequencies with half-width ``γ``.

    Inverse-transform sampling: ``ω = γ · tan(π (u − 1/2))`` for ``u ~ U(0, 1)``.
    """
    quantiles = rng.random(n_oscillators)
    return np.asarray(gamma * np.tan(np.pi * (quantiles - 0.5)), dtype=np.float64)


def simulate_steady_order_parameter(case: SynchronisationCase) -> float:
    """Run the runtime to steady state and return the time-averaged ``R``.

    Natural frequencies and initial phases are drawn from a single seeded
    generator so the measurement is deterministic. The mean-field order
    parameter is averaged over the final ``n_average`` steps.
    """
    rng = np.random.default_rng(case.seed)
    omega = lorentzian_frequencies(case.n_oscillators, case.gamma, rng)
    theta = np.asarray(rng.uniform(-np.pi, np.pi, case.n_oscillators), dtype=np.float64)

    order_history: list[float] = []
    tail_start = case.n_steps - case.n_average
    for step in range(case.n_steps):
        result = kuramoto_sakaguchi_step(
            theta,
            omega,
            dt=case.dt_s,
            K=case.coupling,
            alpha=case.alpha,
            zeta=0.0,
            psi_driver=None,
            psi_mode="mean_field",
            wrap=True,
        )
        theta = np.asarray(result["theta1"], dtype=np.float64)
        if step >= tail_start:
            order_history.append(float(result["R"]))
    return float(np.mean(order_history))


@dataclass(frozen=True)
class OrderParameterRecord:
    """Per-coupling comparison of measured ``R`` against the exact branch."""

    coupling: float
    coupling_over_critical: float
    order_parameter_measured: float
    order_parameter_reference: float
    abs_error: float
    within_tolerance: bool


@dataclass(frozen=True)
class SynchronisationValidationResult:
    """Outcome of the published-benchmark synchronisation validation."""

    gamma: float
    alpha: float
    critical_coupling: float
    n_oscillators: int
    branch_tolerance: float
    incoherent_ceiling: float
    branch_records: tuple[OrderParameterRecord, ...]
    branch_max_abs_error: float
    branch_passed: bool
    subthreshold_order_parameter: float
    subthreshold_passed: bool
    onset_below_order_parameter: float
    onset_above_order_parameter: float
    onset_shift_passed: bool
    passed: bool


def validate_synchronisation(
    *,
    gamma: float = 1.0,
    n_oscillators: int = 4000,
    dt_s: float = 0.05,
    n_steps: int = 1200,
    n_average: int = 300,
    seed: int = 20260614,
    branch_couplings_over_critical: Sequence[float] = (2.0, 3.0, 4.0),
    branch_tolerance: float = 0.03,
    sakaguchi_alpha: float = math.pi / 4.0,
) -> SynchronisationValidationResult:
    """Validate the runtime against the exact Lorentzian Kuramoto results.

    Three published facts are checked:

    1. **Order-parameter branch (α = 0).** For each ``K = ratio · Kc`` with
       ``ratio > 1`` the measured steady ``R`` matches ``sqrt(1 − Kc/K)`` within
       ``branch_tolerance``.
    2. **Sub-threshold incoherence.** At ``K = 0.5 · Kc`` the measured ``R`` sits
       below a finite-size ceiling ``4/sqrt(N)``.
    3. **Sakaguchi onset shift.** With phase lag ``α`` the threshold moves to
       ``Kc(α) = 2γ/cos α``: ``R`` is incoherent just below ``Kc(α)`` and
       coherent above it.
    """
    branch_tolerance = _positive_float("branch_tolerance", branch_tolerance)
    kc = critical_coupling(gamma, 0.0)
    incoherent_ceiling = 4.0 / math.sqrt(_positive_int("n_oscillators", n_oscillators))

    def _case(coupling: float, alpha: float) -> SynchronisationCase:
        return SynchronisationCase(
            n_oscillators=n_oscillators,
            gamma=gamma,
            coupling=coupling,
            alpha=alpha,
            dt_s=dt_s,
            n_steps=n_steps,
            n_average=n_average,
            seed=seed,
        )

    records: list[OrderParameterRecord] = []
    for ratio in branch_couplings_over_critical:
        if ratio <= 1.0:
            raise ValueError("branch couplings must exceed the critical coupling (ratio > 1)")
        coupling = ratio * kc
        measured = simulate_steady_order_parameter(_case(coupling, 0.0))
        reference = synchronised_order_parameter(coupling, gamma, 0.0)
        abs_error = abs(measured - reference)
        records.append(
            OrderParameterRecord(
                coupling=coupling,
                coupling_over_critical=coupling / kc,
                order_parameter_measured=measured,
                order_parameter_reference=reference,
                abs_error=abs_error,
                within_tolerance=abs_error <= branch_tolerance,
            )
        )
    branch_max_abs_error = max((record.abs_error for record in records), default=0.0)
    branch_passed = all(record.within_tolerance for record in records)

    subthreshold_order_parameter = simulate_steady_order_parameter(_case(0.5 * kc, 0.0))
    subthreshold_passed = subthreshold_order_parameter < incoherent_ceiling

    kc_alpha = critical_coupling(gamma, sakaguchi_alpha)
    onset_below = simulate_steady_order_parameter(_case(0.85 * kc_alpha, sakaguchi_alpha))
    onset_above = simulate_steady_order_parameter(_case(1.6 * kc_alpha, sakaguchi_alpha))
    onset_shift_passed = onset_below < 0.25 and onset_above > onset_below + 0.3

    passed = branch_passed and subthreshold_passed and onset_shift_passed
    return SynchronisationValidationResult(
        gamma=gamma,
        alpha=0.0,
        critical_coupling=kc,
        n_oscillators=n_oscillators,
        branch_tolerance=branch_tolerance,
        incoherent_ceiling=incoherent_ceiling,
        branch_records=tuple(records),
        branch_max_abs_error=branch_max_abs_error,
        branch_passed=branch_passed,
        subthreshold_order_parameter=subthreshold_order_parameter,
        subthreshold_passed=subthreshold_passed,
        onset_below_order_parameter=onset_below,
        onset_above_order_parameter=onset_above,
        onset_shift_passed=onset_shift_passed,
        passed=passed,
    )


def build_evidence(result: SynchronisationValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "gamma": result.gamma,
        "critical_coupling": result.critical_coupling,
        "n_oscillators": result.n_oscillators,
        "branch_tolerance": result.branch_tolerance,
        "incoherent_ceiling": result.incoherent_ceiling,
        "branch_records": [
            {
                "coupling": record.coupling,
                "coupling_over_critical": record.coupling_over_critical,
                "order_parameter_measured": record.order_parameter_measured,
                "order_parameter_reference": record.order_parameter_reference,
                "abs_error": record.abs_error,
                "within_tolerance": record.within_tolerance,
            }
            for record in result.branch_records
        ],
        "branch_max_abs_error": result.branch_max_abs_error,
        "branch_passed": result.branch_passed,
        "subthreshold_order_parameter": result.subthreshold_order_parameter,
        "subthreshold_passed": result.subthreshold_passed,
        "onset_below_order_parameter": result.onset_below_order_parameter,
        "onset_above_order_parameter": result.onset_above_order_parameter,
        "onset_shift_passed": result.onset_shift_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION:
        raise ValueError("unsupported kuramoto synchronisation evidence schema_version")
    declared = payload.get("payload_sha256")
    if not _is_sha256(declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return bool(payload.get("passed"))


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


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(description="Validate the Kuramoto-Sakaguchi runtime against published results")
    parser.add_argument("--n-oscillators", type=int, default=4000)
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--target-id", type=str, default="local-kuramoto-synchronisation")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    args = parser.parse_args(argv)

    result = validate_synchronisation(n_oscillators=args.n_oscillators, n_steps=args.n_steps)
    evidence = build_evidence(result, target_id=args.target_id)
    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        kc = result.critical_coupling
        print(f"Kuramoto-Sakaguchi synchronisation validation (gamma={result.gamma}, Kc={kc:.4f})")
        for record in result.branch_records:
            print(
                f"  K/Kc={record.coupling_over_critical:4.2f} "
                f"R={record.order_parameter_measured:6.4f} "
                f"R_ref={record.order_parameter_reference:6.4f} "
                f"|err|={record.abs_error:6.4f} "
                f"{'ok' if record.within_tolerance else 'FAIL'}"
            )
        print(
            f"  branch_passed={result.branch_passed} "
            f"subthreshold_passed={result.subthreshold_passed} "
            f"onset_shift_passed={result.onset_shift_passed}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
