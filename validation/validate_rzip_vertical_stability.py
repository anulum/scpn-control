#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP rigid vertical stability analytic validation
"""Validate the RZIP rigid-plasma vertical stability model against exact results.

The rigid-plasma vertical response model (``src/scpn_control/control/rzip_model.py``)
assembles a linearised state-space ``A`` for ``x = [Z, dZ/dt, I_1, ...]`` and
returns the vertical growth rate as the largest real eigenvalue of ``A``. The
destabilising term is the curvature spring ``K = n mu_0 Ip^2 / (4 pi R_0)`` with
field decay index ``n``; the ``A[1, 0] = -K / M_eff`` entry drives the rigid
vertical mode. In the no-wall limit (no conducting vessel elements or coils) the
state collapses to the ``2 x 2`` block ``[[0, 1], [-K/M_eff, 0]]``, whose
eigenvalues are exactly ``+/- sqrt(-K / M_eff)``. This gives closed-form
references with no measured or external code required, so the validation is fully
self-contained.

Exact references checked against the production ``RZIPModel``:

1. **Unstable no-wall growth rate** (``n < 0``): the largest real eigenvalue is
   ``gamma = sqrt(-n mu_0 Ip^2 / (4 pi R_0 M_eff))``.
2. **Stable no-wall oscillation** (``n > 0``): the rigid mode is marginally
   stable (zero real part) and oscillates at ``omega = sqrt(n mu_0 Ip^2 /
   (4 pi R_0 M_eff))``, recovered as the largest eigenvalue imaginary part.
3. **Marginal index** (``n = 0``): the spring vanishes and the growth rate is
   zero.
4. **Exact scaling laws**: ``gamma`` scales linearly with ``Ip``, as
   ``sqrt(-n)``, and as ``1/sqrt(M_eff)``.
5. **Resistive-wall stabilisation**: adding a passive conducting wall reduces the
   growth rate below the no-wall value while keeping it finite, confirming the
   eddy-current circuit coupling is stabilising.

References
----------
  Lazarus E. A. et al. (1990) *Nucl. Fusion* 30, 111 (rigid vertical model).
  Wesson J. (2011) *Tokamaks*, 4th ed., Oxford University Press, Ch. 3.10
  (vertical stability and field index).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scpn_control.control.rzip_model import RZIPModel
from scpn_control.core.vessel_model import VesselElement, VesselModel
from validation import rzip_vertical_evidence as _evidence
from validation.rzip_vertical_models import (
    RzipValidationResult as RzipValidationResult,
)
from validation.rzip_vertical_models import (
    ScalingCheck as ScalingCheck,
)
from validation.rzip_vertical_models import (
    VerticalConfig as VerticalConfig,
)
from validation.rzip_vertical_models import (
    WallStabilisation as WallStabilisation,
)
from validation.rzip_vertical_models import (
    _negative_float,
    _positive_float,
)

RZIP_VERTICAL_STABILITY_SCHEMA_VERSION = _evidence.SCHEMA_VERSION


def default_config() -> VerticalConfig:
    """Return the declared R0=1.7 m, a=0.5 m, Ip=1 MA, inertia=2 kg reference.

    The frozen geometry is a local rigid-model case, without a calibrated
    facility identity or a reconstructed separatrix.
    """
    return VerticalConfig(r0=1.7, a=0.5, kappa=1.8, ip_ma=1.0, b0=2.0, m_eff_kg=2.0)


def analytic_no_wall_growth_rate(config: VerticalConfig, n_index: float) -> float:
    """Return the exact no-wall growth rate ``sqrt(-K/M_eff)`` in s^-1.

    ``config`` carries metre/MA/tesla/kg geometry and inertia. ``n_index`` must
    be a finite negative number; booleans, text, overflow and other signs raise
    ValueError. This reference reads configuration without mutating model state.
    """
    n_index = _negative_float("destabilising index", n_index)
    return math.sqrt(-config.curvature_spring(n_index) / config.m_eff_kg)


def analytic_no_wall_frequency(config: VerticalConfig, n_index: float) -> float:
    """Return the exact no-wall angular frequency ``sqrt(K/M_eff)`` in rad/s.

    ``config`` carries metre/MA/tesla/kg geometry and inertia. ``n_index`` must
    be a finite positive number; booleans, text, overflow and other signs raise
    ValueError. This reference reads configuration without mutating model state.
    """
    n_index = _positive_float("stabilising index", n_index)
    return math.sqrt(config.curvature_spring(n_index) / config.m_eff_kg)


def _build_no_wall(config: VerticalConfig, n_index: float) -> RZIPModel:
    """Construct the declared rigid plant with no wall or active coils."""
    return RZIPModel(
        config.r0,
        config.a,
        config.kappa,
        config.ip_ma,
        config.b0,
        n_index,
        VesselModel([]),
        vertical_inertia_kg=config.m_eff_kg,
    )


def no_wall_growth_rel_error(config: VerticalConfig, n_index: float) -> float:
    """Relative error of the model growth rate against the exact no-wall value."""
    measured = _build_no_wall(config, n_index).vertical_growth_rate()
    analytic = analytic_no_wall_growth_rate(config, n_index)
    return abs(measured - analytic) / analytic


def no_wall_frequency_rel_error(config: VerticalConfig, n_index: float) -> float:
    """Relative error of the largest eigenvalue imaginary part against ``sqrt(K/M)``."""
    state_matrix, _, _, _ = _build_no_wall(config, n_index).build_state_space()
    eigenvalues = np.linalg.eigvals(state_matrix)
    measured = float(np.max(np.abs(np.imag(eigenvalues))))
    analytic = analytic_no_wall_frequency(config, n_index)
    return abs(measured - analytic) / analytic


def no_wall_growth_time_consistency(config: VerticalConfig, n_index: float) -> float:
    """Return relative mismatch of growth-time times growth-rate against 1000 ms/s.

    ``n_index`` must be finite and negative; other domains raise ValueError.
    Stable or marginal modes have no finite exponential growth-time reference.
    The geometry is read without mutation and the result is dimensionless.
    """
    n_index = _negative_float("destabilising index", n_index)
    model = _build_no_wall(config, n_index)
    gamma = model.vertical_growth_rate()
    growth_time_ms = model.vertical_growth_time()
    return abs(growth_time_ms * gamma - 1000.0) / 1000.0


def scaling_checks(config: VerticalConfig) -> tuple[ScalingCheck, ...]:
    """Verify ``gamma`` scales as ``Ip``, ``sqrt(-n)``, and ``1/sqrt(M_eff)``."""
    base = _build_no_wall(config, -1.0).vertical_growth_rate()

    double_ip = _build_no_wall(
        VerticalConfig(config.r0, config.a, config.kappa, 2.0 * config.ip_ma, config.b0, config.m_eff_kg), -1.0
    ).vertical_growth_rate()
    quad_index = _build_no_wall(config, -4.0).vertical_growth_rate()
    quad_mass = _build_no_wall(
        VerticalConfig(config.r0, config.a, config.kappa, config.ip_ma, config.b0, 4.0 * config.m_eff_kg), -1.0
    ).vertical_growth_rate()

    specs = (
        ("current_linear", double_ip / base, 2.0),
        ("index_sqrt", quad_index / base, 2.0),
        ("inertia_inverse_sqrt", quad_mass / base, 0.5),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


def wall_stabilisation(config: VerticalConfig, n_index: float = -1.0) -> WallStabilisation:
    """Compare growth in s^-1 with and without two passive conducting wall loops.

    ``n_index`` must be finite and negative; other domains raise ValueError.
    Both models use the supplied metre/MA/tesla/kg configuration. The frozen
    comparison carries finite/slower-growth declarations without modifying it.
    This local circuit comparison establishes no facility-control admission.
    """
    n_index = _negative_float("destabilising index", n_index)
    no_wall = _build_no_wall(config, n_index).vertical_growth_rate()
    elements = [
        VesselElement(R=config.r0, Z=1.4 * config.a, resistance=1e-4, cross_section=0.01, inductance=2e-6),
        VesselElement(R=config.r0, Z=-1.4 * config.a, resistance=1e-4, cross_section=0.01, inductance=2e-6),
    ]
    walled = RZIPModel(
        config.r0,
        config.a,
        config.kappa,
        config.ip_ma,
        config.b0,
        n_index,
        VesselModel(elements),
        vertical_inertia_kg=config.m_eff_kg,
    ).vertical_growth_rate()
    return WallStabilisation(
        no_wall_growth_rate=no_wall,
        with_wall_growth_rate=walled,
        wall_slows_growth=bool(walled < no_wall),
        with_wall_finite=bool(math.isfinite(walled)),
    )


def validate_rzip_vertical_stability(
    *,
    config: VerticalConfig | None = None,
    unstable_indices: Sequence[float] = (-2.5, -1.2, -0.6),
    stable_indices: Sequence[float] = (0.8, 1.5),
    exact_tol: float = 1e-9,
    marginal_tol: float = 1e-6,
) -> RzipValidationResult:
    """Validate the production RZIP model against the exact no-wall references.

    The unstable growth rate, stable oscillation frequency, growth-time identity,
    and exact scaling laws must hold to ``exact_tol``; the marginal index must give
    a growth rate below ``marginal_tol``; and a passive wall must reduce the growth
    rate below the no-wall value.

    ``config=None`` selects the default rigid model. Index sequences must be
    nonempty with finite negative unstable and positive stable values. Tolerances
    are finite positive numbers: ``exact_tol`` bounds relative errors and
    ``marginal_tol`` bounds growth in s^-1. Invalid configuration, scalar domains
    or empty sequences raise ValueError before evaluating the model. The returned
    immutable result can report failure; this call writes no files and does not
    authenticate measurements or establish facility-control admission.
    """
    config = default_config() if config is None else config
    if not isinstance(config, VerticalConfig):
        raise ValueError("config must be VerticalConfig")
    exact_tol = _positive_float("exact_tol", exact_tol)
    marginal_tol = _positive_float("marginal_tol", marginal_tol)
    unstable = tuple(_negative_float("unstable index", n) for n in unstable_indices)
    stable = tuple(_positive_float("stable index", n) for n in stable_indices)
    if not unstable or not stable:
        raise ValueError("at least one unstable and one stable index are required")

    max_growth_err = max(no_wall_growth_rel_error(config, n) for n in unstable)
    max_growth_time_err = max(no_wall_growth_time_consistency(config, n) for n in unstable)
    max_freq_err = max(no_wall_frequency_rel_error(config, n) for n in stable)
    marginal = _build_no_wall(config, 0.0).vertical_growth_rate()
    scaling = scaling_checks(config)
    max_scaling_err = max(check.rel_error for check in scaling)
    wall = wall_stabilisation(config)

    growth_passed = max_growth_err < exact_tol
    frequency_passed = max_freq_err < exact_tol
    growth_time_passed = max_growth_time_err < exact_tol
    marginal_passed = abs(marginal) < marginal_tol
    scaling_passed = max_scaling_err < exact_tol
    wall_passed = wall.wall_slows_growth and wall.with_wall_finite

    passed = (
        growth_passed and frequency_passed and growth_time_passed and marginal_passed and scaling_passed and wall_passed
    )
    return RzipValidationResult(
        config=config,
        unstable_indices=unstable,
        stable_indices=stable,
        max_growth_rel_error=max_growth_err,
        max_frequency_rel_error=max_freq_err,
        max_growth_time_rel_error=max_growth_time_err,
        marginal_growth_rate=float(marginal),
        scaling=scaling,
        max_scaling_rel_error=max_scaling_err,
        wall=wall,
        exact_tol=exact_tol,
        marginal_tol=marginal_tol,
        growth_passed=growth_passed,
        frequency_passed=frequency_passed,
        growth_time_passed=growth_time_passed,
        marginal_passed=marginal_passed,
        scaling_passed=scaling_passed,
        wall_passed=wall_passed,
        passed=passed,
    )


def build_evidence(result: RzipValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a detached, checked v1 report from a bounded RZIP result.

    ``target_id`` is a nonempty caller label, not a trusted producer identity.
    The returned mapping records geometry in metres, current in MA, inertia in
    kg, growth rates in s^-1 and dimensionless relative errors. Both coherent
    passing and failing results are sealed with a UTC receipt. Inconsistent or
    malformed result fields raise ValueError; no files are written.
    """
    return _evidence.build_evidence(result, target_id=target_id)


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Check a complete v1 seal, finite domains and metric/verdict agreement.

    Returns the literal passing flag after consistency checks; a coherent
    failing report returns False. Invalid structure, names, domains, timestamp,
    seal or verdicts raise ValueError. A matching self-seal establishes neither
    source authentication nor facility calibration or freshness.
    """
    return _evidence.validate_evidence_payload(payload)


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Write checked JSON/Markdown directly; the parent must already exist.

    Existing destinations are overwritten. IO failure may leave a partial
    pair; no rollback or facility publication guarantee is provided.
    """
    _evidence.write_report(evidence, json_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded rigid-model validation through the operator CLI.

    ``argv=None`` reads process arguments. ``--exact-tol`` overrides the API's
    relative-error threshold and ``--marginal-tol`` overrides growth in s^-1;
    absent options retain the API defaults. ``--json-out`` selects sealed JSON
    stdout instead of text. ``--report`` writes JSON then same-stem Markdown at
    a cwd-relative or absolute path whose parent must already exist. Default
    execution writes no files. ``--target-id`` is a local case label.

    Return 0 for a passing model result or 1 for a coherent failed result.
    Argparse raises SystemExit(2) for malformed arguments or invalid tolerances
    and SystemExit(0) for help. Invalid labels raise ValueError; report IO raises
    OSError and can leave a JSON-only pair. No producer authentication, reference
    measurement, facility admission or source-freshness proof is performed.
    """
    parser = argparse.ArgumentParser(
        description="Validate the RZIP rigid vertical stability model against exact references"
    )
    parser.add_argument("--target-id", type=str, default="local-rzip-vertical-stability")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    parser.add_argument("--exact-tol", type=float, default=None, help="finite positive relative-error threshold")
    parser.add_argument("--marginal-tol", type=float, default=None, help="finite positive growth threshold in s^-1")
    args = parser.parse_args(argv)

    tolerances = {
        name: value
        for name, value in (("exact_tol", args.exact_tol), ("marginal_tol", args.marginal_tol))
        if value is not None
    }
    try:
        result = validate_rzip_vertical_stability(**tolerances)
    except ValueError as exc:
        parser.error(str(exc))
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("RZIP rigid vertical stability validation")
        print(
            f"  no-wall growth (n<0): max rel err={result.max_growth_rel_error:.3e} "
            f"{'ok' if result.growth_passed else 'FAIL'}"
        )
        print(
            f"  oscillation (n>0):    max rel err={result.max_frequency_rel_error:.3e} "
            f"{'ok' if result.frequency_passed else 'FAIL'}"
        )
        print(
            f"  scaling laws:         max rel err={result.max_scaling_rel_error:.3e} "
            f"{'ok' if result.scaling_passed else 'FAIL'}"
        )
        print(
            f"  wall stabilisation:   {result.wall.no_wall_growth_rate:.3e} -> "
            f"{result.wall.with_wall_growth_rate:.3e} s^-1 "
            f"{'ok' if result.wall_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
