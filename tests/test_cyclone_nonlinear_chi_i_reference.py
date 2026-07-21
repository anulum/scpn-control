# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Cyclone Base Case nonlinear chi_i published-reference tests
"""Integrity + physics-consistency tests for the obtained Dimits (2000) CBC reference.

The reference file transcribes the published cross-code nonlinear ion heat
diffusivity values (Dimits et al., Phys. Plasmas 7 (2000) 969) used to back the
`gk_nonlinear` saturated-CBC validation gap. These tests verify the file's
self-digest, that the recorded offset-linear fit independently reproduces the
recorded base-case value, that the Dimits-shift ordering is physical, and that
the provenance is honestly published-not-measured with the claim blocked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "validation" / "reference_data" / "cyclone_base" / "cyclone_nonlinear_chi_i_reference.json"


def _load() -> dict[str, object]:
    return json.loads(_REF.read_text(encoding="utf-8"))


def test_payload_digest_self_consistent() -> None:
    data = _load()
    stored = data["payload_sha256"]
    assert isinstance(stored, str) and len(stored) == 64
    payload = dict(data)
    payload["payload_sha256"] = None
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == stored


def test_offset_linear_fit_reproduces_base_case_value() -> None:
    # Independent evaluation of Dimits Eq. (1): chi_i L_n/(rho_i^2 v_ti) = A(1 - B L_T/R).
    fit = _load()["offset_linear_fit"]
    a = float(fit["A"])
    b = float(fit["B"])
    assert a == pytest.approx(15.4)
    assert b == pytest.approx(6.0)
    recomputed_6p9 = a * (1.0 - b / 6.9)
    assert recomputed_6p9 == pytest.approx(float(fit["chi_i_at_R_L_T_6p9"]), abs=1e-3)
    # Physics pin: the canonical Dimits flux-tube base-case value is ~2.0 (gyro-Bohm).
    assert recomputed_6p9 == pytest.approx(2.009, abs=5e-3)
    assert a * (1.0 - b / 6.92) == pytest.approx(float(fit["chi_i_at_R_L_T_6p92"]), abs=1e-3)


def test_dimits_shift_ordering_is_physical() -> None:
    shift = _load()["dimits_shift"]
    linear = float(shift["linear_critical_R_L_T"])
    effective = float(shift["nonlinear_effective_R_L_T"])
    # The Dimits shift: nonlinear effective critical exceeds the linear critical.
    assert effective > linear
    assert linear == pytest.approx(4.0)
    assert effective == pytest.approx(6.0)
    assert float(_load()["offset_linear_fit"]["valid_above_R_L_T_eff"]) == pytest.approx(effective)


def test_cbc_parameters_are_canonical() -> None:
    params = _load()["cbc_parameters"]
    assert float(params["R_over_L_T"]) == pytest.approx(6.9)
    assert float(params["R_over_L_n"]) == pytest.approx(2.2)
    assert float(params["q"]) == pytest.approx(1.4)
    assert float(params["s_hat"]) == pytest.approx(0.78)
    assert float(params["Ti_over_Te"]) == pytest.approx(1.0)
    assert params["electrons"] == "adiabatic"


def test_cross_model_factors_match_paper() -> None:
    factors = _load()["cross_model_factors_at_base_case"]
    # Global gyrokinetic is 2.4x LOWER than flux-tube (stored as the <1 ratio).
    assert float(factors["global_gyrokinetic_over_fluxtube"]) == pytest.approx(1.0 / 2.4, abs=1e-3)
    assert float(factors["ifs_pppl_over_gyrokinetic_fluxtube"]) == pytest.approx(2.7)
    assert float(factors["gyrofluid_1994_over_gyrokinetic_fluxtube"]) == pytest.approx(3.3)


def test_provenance_is_honest_published_not_measured_and_claim_blocked() -> None:
    data = _load()
    prov = data["provenance"]
    assert prov["synthetic"] is False
    assert prov["measured_by_us"] is False
    assert prov["re_run_here"] is False
    assert prov["source_kind"] == "published_literature"
    assert "Dimits" in str(data["reference"])
    # Exact-hostname match (not a substring check) so the provenance URL is the
    # authoritative PPPL host and CodeQL's URL-substring rule stays satisfied.
    assert urlparse(str(data["source_url"])).hostname == "w3.pppl.gov"
    boundary = str(data["claim_boundary"]).lower()
    assert "blocked" in boundary
    assert "not re-run" in boundary or "not a measured" in boundary
    # The unit-conversion caveat must be recorded (module uses a different gyro-Bohm norm).
    assert "conversion" in str(data["units_contract"]["caveat"]).lower()


def test_module_normalisation_reconciliation_is_derived_and_consistent() -> None:
    data = _load()
    rec = data["module_normalisation_reconciliation"]
    # Independently reproduce the conversion factor a/L_n = (a/R)(R/L_n) for the CBC.
    geom = rec["cbc_geometry"]
    a_over_r = float(geom["a_over_R"])
    r_over_ln = float(geom["R_over_L_n"])
    factor = a_over_r * r_over_ln
    assert factor == pytest.approx(float(rec["conversion_factor_a_over_Ln"]), abs=1e-4)
    assert factor == pytest.approx(0.79137, abs=1e-4)
    # Module-equivalent chi_i_gB = chi_hat_paper * (a/L_n); reproduce from the fit.
    fit = data["offset_linear_fit"]
    chi_hat_6p9 = float(fit["A"]) * (1.0 - float(fit["B"]) / 6.9)
    assert chi_hat_6p9 * factor == pytest.approx(float(rec["module_equivalent_chi_i_gB_at_R_L_T_6p9"]), abs=1e-3)
    assert float(rec["module_equivalent_chi_i_gB_at_R_L_T_6p9"]) == pytest.approx(1.59, abs=1e-2)
    # The converted target sits inside the module's diagnostic band (1.0, 5.0).
    assert 1.0 <= float(rec["module_equivalent_chi_i_gB_at_R_L_T_6p9"]) <= 5.0
    assert "rho_s^2 * c_s / a" in str(rec["module_chi_gB_definition"])
