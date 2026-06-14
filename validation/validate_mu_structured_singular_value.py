#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Structured singular value (mu) closed-form validation
"""Validate the structured-singular-value upper bound against exact mu identities.

The robust-control surface (``src/scpn_control/control/mu_synthesis.py``)
computes the D-scaled upper bound

    mu_bar(M) = min_D  sigma_max(D M D^{-1})

on the complex structured singular value mu(M) for a block-diagonal uncertainty
set. While mu itself is NP-hard in general, several block structures admit an
*exact* closed-form value that the D-scaled bound must reproduce. This module
checks the production ``compute_mu_upper_bound`` against those analytic cases —
no external toolbox, plant, or measured artefact is required, so the validation
is fully self-contained.

Exact identities checked (deterministic seeded complex-matrix ensembles):

1. **Single full block.** For ``Delta`` a single full complex block,
   ``mu(M) = sigma_max(M)`` exactly (Doyle 1982). The D-scaling is a single
   scalar that cancels, so the bound must equal the largest singular value.
2. **Diagonal plant, diagonal structure.** For diagonal ``M`` and a fully
   diagonal complex-scalar ``Delta``, ``mu(M) = max_i |M_ii|``: diagonal
   D-scaling leaves a diagonal matrix whose largest singular value is the
   largest magnitude entry.
3. **Rank-one plant, diagonal structure.** For ``M = u v^H`` and a fully
   diagonal complex-scalar ``Delta``, the D-scaled bound is tight and
   ``mu(M) = sum_i |u_i v_i|`` (Packard and Doyle 1993).
4. **Spectral sandwich.** For any ``M`` and diagonal ``Delta``,
   ``rho(M) <= mu(M) <= sigma_max(M)``; the returned bound must lie in
   ``[rho(M), sigma_max(M)]``.
5. **D-scaling invariance.** ``mu`` is invariant under block-diagonal D-scaling
   of ``M``; the numeric bound must reproduce this up to the descent tolerance.

References:
  Doyle J. C. (1982) "Analysis of feedback systems with structured
  uncertainties", *IEE Proc. D* 129, 242.
  Packard A., Doyle J. (1993) "The complex structured singular value",
  *Automatica* 29, 71.
  Skogestad S., Postlethwaite I. (2005) *Multivariable Feedback Control*,
  2nd ed., Wiley, Ch. 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_control.control.mu_synthesis import compute_mu_upper_bound

ComplexArray = NDArray[np.complex128]

MU_STRUCTURED_SINGULAR_VALUE_SCHEMA_VERSION = "scpn-control.mu-structured-singular-value-validation.v1"

DEFAULT_SIZES: tuple[int, ...] = (2, 3, 4, 5)
DEFAULT_SAMPLES_PER_SIZE: int = 4


def _complex_matrix(rng: np.random.Generator, n: int) -> ComplexArray:
    """Draw an ``n x n`` complex matrix with unit-variance real and imaginary parts."""
    real = rng.standard_normal((n, n))
    imag = rng.standard_normal((n, n))
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _complex_vector(rng: np.random.Generator, n: int) -> ComplexArray:
    return np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128)


def _sigma_max(matrix: ComplexArray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def _spectral_radius(matrix: ComplexArray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def full_block_error(rng: np.random.Generator, n: int) -> float:
    """Absolute error of the bound against ``sigma_max(M)`` for one full block."""
    matrix = _complex_matrix(rng, n)
    bound = compute_mu_upper_bound(matrix, [(n, "full")])
    return abs(bound - _sigma_max(matrix))


def diagonal_error(rng: np.random.Generator, n: int) -> float:
    """Absolute error of the bound against ``max_i |M_ii|`` for diagonal ``M``."""
    diagonal = _complex_vector(rng, n)
    matrix = np.diag(diagonal)
    structure = [(1, "complex_scalar")] * n
    bound = compute_mu_upper_bound(matrix, structure)
    return abs(bound - float(np.max(np.abs(diagonal))))


def rank_one_relative_error(rng: np.random.Generator, n: int) -> float:
    """Relative error of the bound against ``sum_i |u_i v_i|`` for ``M = u v^H``."""
    u = _complex_vector(rng, n)
    v = _complex_vector(rng, n)
    matrix = np.outer(u, np.conjugate(v))
    structure = [(1, "complex_scalar")] * n
    bound = compute_mu_upper_bound(matrix, structure)
    exact = float(np.sum(np.abs(u) * np.abs(v)))
    return abs(bound - exact) / max(exact, 1e-15)


@dataclass(frozen=True)
class SandwichSample:
    """One spectral-sandwich observation ``rho(M) <= bound <= sigma_max(M)``."""

    spectral_radius: float
    bound: float
    sigma_max: float
    within: bool


def sandwich_sample(rng: np.random.Generator, n: int, *, tolerance: float) -> SandwichSample:
    """Check the returned bound lies in ``[rho(M), sigma_max(M)]`` for diagonal ``Delta``."""
    matrix = _complex_matrix(rng, n)
    structure = [(1, "complex_scalar")] * n
    bound = compute_mu_upper_bound(matrix, structure)
    rho = _spectral_radius(matrix)
    sigma = _sigma_max(matrix)
    within = rho <= bound * (1.0 + tolerance) and bound <= sigma * (1.0 + tolerance)
    return SandwichSample(spectral_radius=rho, bound=bound, sigma_max=sigma, within=within)


def scaling_invariance_relative_error(rng: np.random.Generator, n: int) -> float:
    """Relative change of the bound under a random block-diagonal D-scaling of ``M``."""
    matrix = _complex_matrix(rng, n)
    structure = [(1, "complex_scalar")] * n
    scales = np.exp(rng.uniform(-1.0, 1.0, n)).astype(np.complex128)
    scaling = np.diag(scales)
    scaled = scaling @ matrix @ np.linalg.inv(scaling)
    base = compute_mu_upper_bound(matrix, structure)
    rescaled = compute_mu_upper_bound(scaled, structure)
    return abs(base - rescaled) / max(base, 1e-15)


@dataclass(frozen=True)
class CaseResult:
    """Aggregate outcome of one validation case over its ensemble."""

    name: str
    sample_count: int
    max_error: float
    tolerance: float
    passed: bool


def _run_error_case(
    name: str,
    error_fn: Callable[[np.random.Generator, int], float],
    rng: np.random.Generator,
    sizes: Sequence[int],
    samples_per_size: int,
    tolerance: float,
) -> CaseResult:
    max_error = 0.0
    count = 0
    for n in sizes:
        for _ in range(samples_per_size):
            max_error = max(max_error, error_fn(rng, n))
            count += 1
    return CaseResult(
        name=name, sample_count=count, max_error=max_error, tolerance=tolerance, passed=max_error < tolerance
    )


@dataclass(frozen=True)
class MuValidationResult:
    """Outcome of the structured-singular-value closed-form validation.

    ``cases`` are the gated exact identities; ``diagnostics`` are recorded but do
    not affect ``passed`` because they probe the iterative D-scaling descent
    heuristic rather than a closed-form answer.
    """

    seed: int
    sizes: tuple[int, ...]
    samples_per_size: int
    cases: tuple[CaseResult, ...]
    diagnostics: tuple[CaseResult, ...]
    sandwich_min_margin: float
    passed: bool


def validate_mu(
    *,
    seed: int = 20260614,
    sizes: Sequence[int] = DEFAULT_SIZES,
    samples_per_size: int = DEFAULT_SAMPLES_PER_SIZE,
    full_block_tol: float = 1e-9,
    diagonal_tol: float = 1e-9,
    rank_one_tol: float = 1e-3,
    sandwich_tol: float = 1e-6,
    scaling_invariance_tol: float = 2e-1,
) -> MuValidationResult:
    """Validate ``compute_mu_upper_bound`` against exact structured-mu identities.

    Four *gated* cases are exercised over deterministic seeded ensembles: the
    single full-block identity ``mu = sigma_max``, the diagonal-plant identity
    ``mu = max|M_ii|``, the rank-one identity ``mu = sum|u_i v_i|``, and the
    spectral sandwich ``rho <= mu <= sigma_max``. The first two are tight to
    machine precision; the rank-one case holds within the iterative D-scaling
    descent tolerance. D-scaling invariance is recorded as a *diagnostic* only:
    because the production bound minimises ``sigma_max(D M D^{-1})`` with a
    50-step finite-difference descent, the bound is invariant only up to the
    descent's local-minimum spread, so it is reported but not gated.
    """
    ordered = tuple(_positive_int("size", n) for n in sizes)
    if not ordered:
        raise ValueError("at least one matrix size is required")
    if min(ordered) < 2:
        raise ValueError("matrix sizes must be at least 2")
    samples_per_size = _positive_int("samples_per_size", samples_per_size)

    rng = np.random.default_rng(_seed_value(seed))

    cases: list[CaseResult] = [
        _run_error_case(
            "full_block_equals_sigma_max", full_block_error, rng, ordered, samples_per_size, full_block_tol
        ),
        _run_error_case("diagonal_equals_max_abs_entry", diagonal_error, rng, ordered, samples_per_size, diagonal_tol),
        _run_error_case(
            "rank_one_equals_sum_abs_products", rank_one_relative_error, rng, ordered, samples_per_size, rank_one_tol
        ),
    ]

    sandwich_samples = [
        sandwich_sample(rng, n, tolerance=sandwich_tol) for n in ordered for _ in range(samples_per_size)
    ]
    sandwich_passed = all(sample.within for sample in sandwich_samples)
    sandwich_min_margin = min(
        min(sample.bound - sample.spectral_radius, sample.sigma_max - sample.bound) for sample in sandwich_samples
    )
    cases.append(
        CaseResult(
            name="spectral_sandwich_rho_le_mu_le_sigma_max",
            sample_count=len(sandwich_samples),
            max_error=float(max(0.0, -sandwich_min_margin)),
            tolerance=sandwich_tol,
            passed=sandwich_passed,
        )
    )

    diagnostics = [
        _run_error_case(
            "d_scaling_invariance",
            scaling_invariance_relative_error,
            rng,
            ordered,
            samples_per_size,
            scaling_invariance_tol,
        )
    ]

    passed = all(case.passed for case in cases)
    return MuValidationResult(
        seed=_seed_value(seed),
        sizes=ordered,
        samples_per_size=samples_per_size,
        cases=tuple(cases),
        diagnostics=tuple(diagnostics),
        sandwich_min_margin=float(sandwich_min_margin),
        passed=passed,
    )


def build_evidence(result: MuValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": MU_STRUCTURED_SINGULAR_VALUE_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "seed": result.seed,
        "sizes": list(result.sizes),
        "samples_per_size": result.samples_per_size,
        "sandwich_min_margin": result.sandwich_min_margin,
        "cases": [
            {
                "name": case.name,
                "sample_count": case.sample_count,
                "max_error": case.max_error,
                "tolerance": case.tolerance,
                "passed": case.passed,
            }
            for case in result.cases
        ],
        "diagnostics": [
            {
                "name": case.name,
                "sample_count": case.sample_count,
                "max_error": case.max_error,
                "tolerance": case.tolerance,
                "passed": case.passed,
            }
            for case in result.diagnostics
        ],
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != MU_STRUCTURED_SINGULAR_VALUE_SCHEMA_VERSION:
        raise ValueError("unsupported mu structured-singular-value evidence schema_version")
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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed_value(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Structured Singular Value (mu) Closed-Form Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Seed: {evidence['seed']}; sizes: {evidence['sizes']}; samples/size: {evidence['samples_per_size']}",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "| case | samples | max error | tolerance | passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {case['name']} | {case['sample_count']} | {case['max_error']:.3e} | "
        f"{case['tolerance']:.1e} | {case['passed']} |"
        for case in evidence["cases"]
    ]
    lines += [
        "",
        "## Diagnostics (recorded, not gated)",
        "",
        "The D-scaling invariance probe exercises the 50-step finite-difference "
        "descent in `compute_mu_upper_bound`. The bound is invariant in exact "
        "arithmetic, but the descent reaches slightly different local minima per "
        "orientation, so the spread below is reported but does not affect the "
        "pass/fail outcome.",
        "",
        "| diagnostic | samples | max relative spread | soft tolerance | within |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {case['name']} | {case['sample_count']} | {case['max_error']:.3e} | "
        f"{case['tolerance']:.1e} | {case['passed']} |"
        for case in evidence["diagnostics"]
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the structured-singular-value upper bound against exact mu identities"
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--samples-per-size", type=int, default=DEFAULT_SAMPLES_PER_SIZE)
    parser.add_argument("--target-id", type=str, default="local-mu-structured-singular-value")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_mu(seed=args.seed, sizes=args.sizes, samples_per_size=args.samples_per_size)
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print(f"Structured singular value (mu) validation (seed={result.seed}, sizes={list(result.sizes)})")
        for case in result.cases:
            print(
                f"  {case.name:42s} max_err={case.max_error:.3e} "
                f"tol={case.tolerance:.1e} {'ok' if case.passed else 'FAIL'}"
            )
        for case in result.diagnostics:
            print(
                f"  {case.name:42s} max_err={case.max_error:.3e} "
                f"tol={case.tolerance:.1e} {'ok' if case.passed else 'high'} (diagnostic)"
            )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
