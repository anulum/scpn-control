#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Fokker-Planck collision cross-validation
"""Cross-validate production collision coefficients against a Fokker-Planck reference.

The production test-particle coefficients in
:func:`scpn_control.core.gk_species.collision_frequencies` are compared against
the structurally independent Fokker-Planck reference in
:mod:`gk_collision_independent_reference` across cases spanning density,
temperature, effective charge, field temperature, and species mass.

The evidence has two honest parts, matching the module's bounded ("order-of-
magnitude for the linearised operator") self-description:

* **Scaling (exact).** The ratio of the production deflection rate to each
  independent reference rate is required to be *constant* across every case.
  A ratio that stays fixed while density, temperature, effective charge, and
  species mass vary proves the production functional form (``~ n Z_eff q^4
  m^{-1/2} T^{-3/2} lnLambda``) matches the Fokker-Planck reference to machine
  precision.
* **Prefactor (bounded).** The constant value of that ratio quantifies the
  O(1) prefactor by which the production coefficient differs from each
  canonical convention; it is required to sit inside a documented O(1) band,
  not to equal unity. This keeps quantitative collisional-damping claims
  blocked while recording the bound.

The energy-relaxation channel is validated structurally: the production
``nu_E / nu_D`` ratio must equal the independent elastic energy-transfer
efficiency times the field-temperature factor, confirming the mass-ratio
suppression is the correct kinematic factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scpn_control.core.gk_species import GKSpecies, collision_frequencies  # noqa: E402
from validation.gk_collision_independent_reference import independent_collision_rates  # noqa: E402

_REPORT_SCHEMA = "scpn-control.gk-collision-independent-crosscheck.v1"
# Constancy of the production/reference ratio across physically distinct cases
# is the scaling proof; the tolerance is loose enough to absorb only quadrature
# round-off in the reference Maxwellian average.
_SCALING_RTOL = 1.0e-9
# The energy-channel structural identity is an exact algebraic relation.
_ENERGY_STRUCTURE_RTOL = 1.0e-9
# Documented O(1) band spanned by standard collision-frequency conventions
# (deflection frequency, Braginskii 1/tau, NRL rate differ by O(1) prefactors).
_PREFACTOR_BAND = (0.1, 3.0)

_LN_LAMBDA = 17.0

# Cases vary density, temperature, effective charge, field temperature, and
# species mass so a constant production/reference ratio can only arise from a
# matching functional form.
_CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "deuterium_reference",
        "species": {"mass_amu": 2.0, "charge_e": 1.0, "temperature_keV": 8.0, "density_19": 10.0},
        "field": {"n_e_19": 10.0, "T_e_keV": 8.0, "z_eff": 1.0},
    },
    {
        "case": "deuterium_hot",
        "species": {"mass_amu": 2.0, "charge_e": 1.0, "temperature_keV": 20.0, "density_19": 10.0},
        "field": {"n_e_19": 10.0, "T_e_keV": 20.0, "z_eff": 1.0},
    },
    {
        "case": "deuterium_dense",
        "species": {"mass_amu": 2.0, "charge_e": 1.0, "temperature_keV": 8.0, "density_19": 30.0},
        "field": {"n_e_19": 30.0, "T_e_keV": 8.0, "z_eff": 1.0},
    },
    {
        "case": "deuterium_impure",
        "species": {"mass_amu": 2.0, "charge_e": 1.0, "temperature_keV": 8.0, "density_19": 10.0},
        "field": {"n_e_19": 10.0, "T_e_keV": 5.0, "z_eff": 2.5},
    },
    {
        "case": "electron",
        "species": {
            "mass_amu": 9.1093837015e-31 / 1.67262192369e-27,
            "charge_e": -1.0,
            "temperature_keV": 8.0,
            "density_19": 10.0,
        },
        "field": {"n_e_19": 10.0, "T_e_keV": 8.0, "z_eff": 1.0},
    },
)


def _case_rates(case: dict[str, Any]) -> dict[str, float]:
    species = GKSpecies(R_L_T=0.0, R_L_n=0.0, **case["species"])
    field = case["field"]
    nu_d_prod, nu_e_prod = collision_frequencies(
        species, field["n_e_19"], field["T_e_keV"], Z_eff=field["z_eff"], ln_lambda=_LN_LAMBDA
    )
    reference = independent_collision_rates(
        mass_amu=species.mass_amu,
        charge_e=species.charge_e,
        temperature_keV=species.temperature_keV,
        n_e_19=field["n_e_19"],
        T_e_keV=field["T_e_keV"],
        z_eff=field["z_eff"],
        ln_lambda=_LN_LAMBDA,
    )
    energy_structure = nu_d_prod * reference.elastic_energy_transfer_efficiency * reference.field_temperature_factor
    return {
        "nu_deflection_production": nu_d_prod,
        "nu_energy_production": nu_e_prod,
        "reference_deflection_rate": reference.thermal_deflection_rate,
        "reference_braginskii_rate": reference.braginskii_rate,
        "deflection_ratio": nu_d_prod / reference.thermal_deflection_rate,
        "braginskii_ratio": nu_d_prod / reference.braginskii_rate,
        "energy_structure_ratio": nu_e_prod / energy_structure,
    }


def _spread(values: list[float]) -> float:
    """Relative spread ``(max - min) / |mean|`` of a ratio across cases."""
    lo = min(values)
    hi = max(values)
    mean = sum(values) / len(values)
    if mean == 0.0:
        return float("inf")
    return abs(hi - lo) / abs(mean)


def validate_gk_collision_independent() -> dict[str, Any]:
    """Compare production collision coefficients against the Fokker-Planck reference."""
    report = _new_report()
    entries: list[dict[str, Any]] = report["entries"]
    findings: list[str] = report["findings"]

    deflection_ratios: list[float] = []
    braginskii_ratios: list[float] = []
    for case in _CASES:
        name = str(case["case"])
        rates = _case_rates(case)
        deflection_ratios.append(rates["deflection_ratio"])
        braginskii_ratios.append(rates["braginskii_ratio"])
        energy_ok = abs(rates["energy_structure_ratio"] - 1.0) <= _ENERGY_STRUCTURE_RTOL
        if not energy_ok:
            findings.append(
                f"{name}: energy-relaxation channel deviates from the elastic energy-transfer "
                f"identity (nu_E/(nu_D delta sqrt(T_s/T_e)) = {rates['energy_structure_ratio']:.6e})"
            )
            report["status"] = "fail"
        entries.append(
            {
                "case": name,
                "deflection_ratio": rates["deflection_ratio"],
                "braginskii_ratio": rates["braginskii_ratio"],
                "energy_structure_ratio": rates["energy_structure_ratio"],
                "energy_structure_exact": energy_ok,
                "case_sha256": _json_sha256({"case": name, "species": case["species"], "field": case["field"]}),
            }
        )

    deflection_spread = _spread(deflection_ratios)
    braginskii_spread = _spread(braginskii_ratios)
    deflection_prefactor = sum(deflection_ratios) / len(deflection_ratios)
    braginskii_prefactor = sum(braginskii_ratios) / len(braginskii_ratios)

    scaling_exact = deflection_spread <= _SCALING_RTOL and braginskii_spread <= _SCALING_RTOL
    prefactor_bounded = _PREFACTOR_BAND[0] <= deflection_prefactor <= _PREFACTOR_BAND[1]

    if not scaling_exact:
        findings.append(
            f"production/reference ratio is not constant across cases "
            f"(deflection spread={deflection_spread:.3e}, braginskii spread={braginskii_spread:.3e}) "
            f"— functional scaling diverges from the Fokker-Planck reference"
        )
        report["status"] = "fail"
    if not prefactor_bounded:
        findings.append(
            f"bounded deflection prefactor {deflection_prefactor:.4f} lies outside the documented "
            f"O(1) band {_PREFACTOR_BAND}"
        )
        report["status"] = "fail"

    report["cases"] = len(entries)
    report["aggregate"] = {
        "deflection_ratio_spread": deflection_spread,
        "braginskii_ratio_spread": braginskii_spread,
        "deflection_prefactor": deflection_prefactor,
        "braginskii_prefactor": braginskii_prefactor,
        "scaling_exact": scaling_exact,
        "prefactor_bounded": prefactor_bounded,
    }
    return _finalise_report(report)


def _new_report() -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "reference_implementation": "validation/gk_collision_independent_reference.py",
        "method": (
            "velocity-dependent Fokker-Planck deflection frequency (Chandrasekhar function) "
            "Maxwellian-averaged by Gauss-Legendre quadrature, with an independent Braginskii/NRL "
            "closed-form anchor and an elastic energy-transfer efficiency identity"
        ),
        "model_reference": "Helander & Sigmar 2002 §3; Braginskii 1965; NRL Plasma Formulary 2019",
        "tolerances": {
            "scaling_ratio_spread": _SCALING_RTOL,
            "energy_structure": _ENERGY_STRUCTURE_RTOL,
            "prefactor_band": list(_PREFACTOR_BAND),
        },
        "units": {
            "nu_deflection_production": "s-1",
            "nu_energy_production": "s-1",
            "reference_deflection_rate": "s-1",
            "reference_braginskii_rate": "s-1",
            "deflection_ratio": "dimensionless",
            "braginskii_ratio": "dimensionless",
            "energy_structure_ratio": "dimensionless",
        },
        "public_claims": {
            "bounded_collision_scaling_independently_verified": False,
            "quantitative_collisional_damping": False,
            "quantitative_collisional_damping_blocked_reason": (
                "Bounded test-particle coefficients only: scaling matches the Fokker-Planck reference "
                "but the prefactor is an O(1)-bounded convention choice, and field-particle "
                "momentum-conservation plus external multispecies validation remain outstanding."
            ),
        },
        "cases": 0,
        "aggregate": {},
        "entries": [],
        "findings": [],
        "payload_sha256": None,
    }


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    report["public_claims"]["bounded_collision_scaling_independently_verified"] = report["status"] == "pass"
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the cross-check and optionally persist the evidence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report to stdout")
    args = parser.parse_args(argv)

    report = validate_gk_collision_independent()
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK collision independent cross-check: {report['status']} cases={report['cases']}")
        for finding in report["findings"]:
            print(f"FINDING {finding}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
