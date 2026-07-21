# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FreeGS DIII-D free-boundary reference tests
"""Regression + contract test for the committed FreeGS DIII-D equilibrium reference.

The reference is an external-code (FreeGS 0.8.2) free-boundary Grad-Shafranov fixture
used to exercise the equilibrium psi-provider → `compute_equilibrium_shape` consumer
contract end-to-end. The test verifies the manifest (schema + artifact SHA-256), the
honest provenance, and that the documented preprocessing (transpose to ``[NR, NZ]`` +
LCFS shift) reproduces the recorded shape metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_control.core.equilibrium_shape import compute_equilibrium_shape
from scpn_control.core.real_data_manifest import load_real_data_manifest

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "validation" / "reference_data" / "diiid" / "freegs"
_MANIFEST = _REF / "manifests" / "diiid_freegs_1p5MA.manifest.json"


def test_manifest_validates_and_artifacts_match_checksums() -> None:
    manifest = load_real_data_manifest(_MANIFEST, verify_artifact=True)
    assert manifest.kind == "synthetic"
    assert manifest.dataset_id == "diiid-freegs-1p5ma-equilibrium"
    assert manifest.machine == "DIII-D"
    assert len(manifest.artifacts) == 2


def test_manifest_provenance_is_honest_external_code_not_measured() -> None:
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["source"]["kind"] == "synthetic"
    assert "FreeGS" in payload["synthetic_generator"]
    boundary = payload["claim_boundary"]
    assert "not measured DIII-D" in boundary.lower() or "not measured diii-d" in boundary.lower()
    assert "external-code" in boundary.lower()


def test_coilset_has_the_diiid_pf_set() -> None:
    coil = json.loads((_REF / "diiid_freegs_1p5MA_coilset.json").read_text(encoding="utf-8"))
    assert len(coil["coils"]) == 18
    assert coil["meta"]["Ip_A"] == 1.5e6
    for entry in coil["coils"]:
        assert set(entry) == {"name", "R_m", "Z_m", "current_A"}
        assert entry["R_m"] > 0.0


def test_shape_contract_reproduces_recorded_reference() -> None:
    coil = json.loads((_REF / "diiid_freegs_1p5MA_coilset.json").read_text(encoding="utf-8"))
    ref = coil["control_shape_reference"]
    data = np.load(_REF / "diiid_freegs_1p5MA.npz")
    psi_bndry = float(coil["meta"]["psi_bndry_Wb"])
    vac = float(coil["meta"]["fvac_Tm"])
    # Documented preprocessing: transpose [NZ,NR] -> [NR,NZ] and LCFS-shift.
    shape = compute_equilibrium_shape(
        data["psi_NZ_NR_Wb"].T - psi_bndry,
        data["R_m"],
        data["Z_m"],
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        ip=float(coil["meta"]["Ip_A"]),
        vacuum_rb_phi=vac,
    )
    assert shape is not None
    # Tolerances absorb minor cross-platform contourpy/numpy numerical differences
    # in the sub-grid contour extraction while still catching gross contract breaks
    # (a missing LCFS shift or a wrong transpose move these by >5% / several-fold).
    np.testing.assert_allclose(shape.R0, ref["R0_m"], rtol=2e-2)
    np.testing.assert_allclose(shape.a, ref["a_m"], rtol=2e-2)
    np.testing.assert_allclose(shape.kappa, ref["kappa"], rtol=3e-2)
    np.testing.assert_allclose(shape.delta_upper, ref["delta_upper"], atol=3e-2)
    np.testing.assert_allclose(shape.delta_lower, ref["delta_lower"], atol=3e-2)


def test_shape_matches_freegs_geometry_within_percent() -> None:
    # R0/a agree with FreeGS's own geometry to <1.5% (the cross-code check).
    coil = json.loads((_REF / "diiid_freegs_1p5MA_coilset.json").read_text(encoding="utf-8"))
    fg = coil["freegs_metrics"]
    ref = coil["control_shape_reference"]
    assert abs(ref["R0_m"] - fg["R_geo_m"]) / fg["R_geo_m"] < 0.015
    assert abs(ref["a_m"] - fg["a_minor_m"]) / fg["a_minor_m"] < 0.015
