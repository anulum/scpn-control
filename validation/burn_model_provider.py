# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Source-bound FUSION burn diagnostic adapter.
"""Execute the pinned FUSION 0D model without design-target calibration.

Uniform line-density use, equal-temperature heat capacity and the provider's
closures are declared approximations. Consistency is not empirical accuracy.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, cast

from validation.burn_model_checks import NUMERICAL_LIMITS, BurnPoint, check_history, evidence_failures

SOURCE_COMMIT = "2fe16dfbe939658e16dcd3302546ebfbf1115d75"
SOURCE_HASHES = {
    "dynamic_burn_model.py": "2525b616d0e556816dccb49e809d87a7aaf2facaa715c500abf7f7aa748c9e6d",
    "uncertainty.py": "935ddd27f634974a783cb58ba17d9a79d55b1d63f2727b261270e727934a3bd5",
    "scaling_laws.py": "996a99d365a876c7c6cdf5b212f6b6450d5cfaddcb7512208a5d567da25d23e3",
}


class BurnModel(Protocol):
    """Required provider surface; constructor discovery remains optional."""

    simulate: Callable[..., dict[str, object]]


def _point(reference: Mapping[str, object]) -> BurnPoint:
    """Read positive physical inputs without coercing booleans or using targets."""
    values = []
    for name in ("R_m", "a_m", "B_t_T", "I_p_MA", "kappa", "n_e_1e19", "P_aux_MW"):
        value = reference[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        values.append(float(value))
    values[5] /= 10
    return BurnPoint(*values)


def _provider() -> tuple[Callable[..., BurnModel], dict[str, Any]]:
    """Bind package version and actual installed numerical source bytes to the pin."""
    module = importlib.import_module("scpn_fusion.core.dynamic_burn_model")
    package_version = version("scpn-fusion")
    if package_version != "4.0.0" or module.__file__ is None:
        raise ValueError("unsupported FUSION burn provider version/source")
    directory = Path(module.__file__).parent
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    if hashes != SOURCE_HASHES:
        raise ValueError("FUSION burn numerical source does not match the immutable pin")
    return cast(Callable[..., BurnModel], module.DynamicBurnModel), {
        "package": "scpn-fusion",
        "version": package_version,
        "source_commit_pin": SOURCE_COMMIT,
        "source_files_sha256": hashes,
        "numpy_version": version("numpy"),
    }


def evaluate_burn_model(reference: Mapping[str, object]) -> tuple[float, dict[str, Any]]:
    """Run real coarse/refined 200-second diagnostics and retain every original trace.

    Time steps 0.01 and 0.005 seconds test first-order integration sensitivity;
    fixed relative numerical budgets are 1e-3. Total energy and pre-step power
    samples retain separate times. Q is reported both capped and as the actual
    fusion/auxiliary power ratio. No steady-state or facility claim is inferred.

    Raises
    ------
    RuntimeError
        The actual provider cannot produce finite source-bound histories.
    ValueError, KeyError
        Physical design inputs are missing or invalid.
    """
    point = _point(reference)
    try:
        factory, provenance = _provider()
        runs = []
        checks = []
        for dt in (0.01, 0.005):
            model = factory(
                R0=point.r_m,
                a=point.a_m,
                B_t=point.b_t,
                I_p=point.ip_ma,
                kappa=point.kappa,
                n_e20=point.ne20,
                M_eff=point.mass_amu,
                Z_eff=point.z_eff,
            )
            run = model.simulate(
                P_aux_mw=point.p_aux_mw,
                T_initial_keV=5.0,
                duration_s=200.0,
                dt_s=dt,
                f_he_initial=0.02,
                tau_he_factor=5.0,
                pumping_efficiency=0.8,
            )
            checks.append(check_history(run, point, 200.0, dt))
            runs.append(run)
        coarse_w = cast(list[float], runs[0]["W_MJ"])[-1]
        refined_w = cast(list[float], runs[1]["W_MJ"])[-1]
        coarse_power = cast(list[float], runs[0]["P_fus_MW"])[-1]
        refined_power = cast(list[float], runs[1]["P_fus_MW"])[-1]
        refinement = max(
            abs(coarse_w - refined_w) / max(abs(coarse_w), abs(refined_w), 1e-12),
            abs(coarse_power - refined_power) / max(abs(coarse_power), abs(refined_power), 1e-12),
        )
        combined = {
            key: max(float(check[key]) for check in checks)
            for key in NUMERICAL_LIMITS
            if key != "timestep_refinement_relative_error"
        }
        for key in (
            "temperature_cap_events",
            "temperature_lower_clip_events",
            "q_cap_events",
            "ash_clip_events",
            "energy_floor_events",
        ):
            combined[key] = max(check[key] for check in checks)
        combined["timestep_refinement_relative_error"] = refinement
        original_histories = [{key: value for key, value in run.items() if isinstance(value, list)} for run in runs]
        return point.beta_n(refined_w), {
            "W_MJ": float(refined_w),
            "P_fusion_MW": float(refined_power),
            "Q": float(cast(float, runs[1]["Q_final"])),
            "Q_unclipped": float(refined_power / point.p_aux_mw),
            "model_provenance": provenance,
            "model_inputs": asdict(point),
            "initial_state": {
                "T_initial_keV": 5.0,
                "f_he_initial": 0.02,
                "tau_he_factor": 5.0,
                "pumping_efficiency": 0.8,
            },
            "checks": combined,
            "check_failures": evidence_failures(combined),
            "run_checks": checks,
            "native_histories": original_histories,
            "numerical_limits": dict(NUMERICAL_LIMITS),
            "reference_kind": "design",
            "physical_validation_claim": False,
            "limitations": [
                "Uniform line-density approximation with Te=Ti and fixed 3neT heat capacity despite ash dilution.",
                "No H98 input; reference H98 and power/Q values need not describe this operating point.",
                "W is post-step; temperature, ash and powers are pre-step. Terminal stationarity is checked, not assumed.",
                "Clipped or floor-affected diagnostics cannot pass the numerical fidelity guard.",
                "No matched measured validation data or independently qualified species/profile model.",
            ],
        }
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"burn model unavailable or invalid: {exc}; reference values cannot replace model predictions"
        ) from exc
