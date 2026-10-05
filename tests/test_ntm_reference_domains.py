# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Persisted NTM numeric declaration tests

"""Exercise source-derived NTM numeric declarations through the real persisted public reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_ntm_reference_validation import _valid_ntm_reference_artifact

from validation.validate_ntm_reference import canonical_artifact_sha256, validate_ntm_reference


def _inspect(tmp_path: Path, field: str, value: object) -> dict[str, object]:
    """Reseal one changed declared field and inspect the actual JSON file without physics execution."""
    payload = _valid_ntm_reference_artifact()
    payload[field] = value
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_ntm_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [0],
        [0, True],
        [0, "1"],
        [0, 10**400],
        [0, float("nan")],
        [-0.1, 1],
        [0, 1.1],
        [0, 0],
        [1, 0],
        [0, 0.25, 0.5, 0.75, 1],
    ],
)
def test_grid_domains(tmp_path: Path, value: object) -> None:
    """Grid declaration requires finite, nonnegative, increasing values through inclusive unit endpoints."""
    report = _inspect(tmp_path, "rho_grid", value)
    accepted = value == [0, 0.25, 0.5, 0.75, 1]
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "rho_grid" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [1],
        [1, 2],
        [1, 2, 3, 4, True],
        [1, 2, 3, 4, 0],
        [1, 2, 3, 4, "5"],
        [1, 2, 3, 4, 10**400],
        [1, 2, 3, 4, float("nan")],
        [1, 2, 3, 4, 5],
    ],
)
def test_q_profile_domains(tmp_path: Path, value: object) -> None:
    """Q values must be positive, representable and length-matched, without interpolation or monotonicity demand."""
    report = _inspect(tmp_path, "q_profile", value)
    accepted = value == [1, 2, 3, 4, 5]
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "q_profile" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    ("field", "value"), [("m", True), ("n", True), ("m", 2.0), ("n", "1"), ("m", 0), ("n", 0), ("m", -1), ("n", -1)]
)
def test_integer_mode_domains(tmp_path: Path, field: str, value: object) -> None:
    """Declared modes require positive nonboolean integers for both indices."""
    payload = _valid_ntm_reference_artifact()
    surface = cast(dict[str, object], payload["rational_surface"])
    surface[field] = value
    report = _inspect(tmp_path, "rational_surface", surface)
    assert report["status"] == "fail"
    assert any(error["field"] == "rational_surface" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("field", ["r_s_m", "q", "a_m", "R0_m", "shear", "rho"])
@pytest.mark.parametrize("value", [None, True, "1", 10**400, float("inf"), float("nan")])
def test_surface_numeric_representation(tmp_path: Path, field: str, value: object) -> None:
    """Every numeric surface field refuses invalid, nonfinite and overflowing values without interpreter errors."""
    payload = _valid_ntm_reference_artifact()
    surface = cast(dict[str, object], payload["rational_surface"])
    surface[field] = value
    report = _inspect(tmp_path, "rational_surface", surface)
    assert report["status"] == "fail"
    assert any(error["field"] == "rational_surface" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("r_s_m", 0),
        ("q", 0),
        ("a_m", 0),
        ("R0_m", 0),
        ("rho", 0),
        ("rho", 1),
        ("a_m", 6.2),
        ("a_m", 7),
        ("r_s_m", 2),
        ("r_s_m", 3),
    ],
)
def test_surface_ordering(tmp_path: Path, field: str, value: object) -> None:
    """Surface positivity/open rho interval and strict radius ordering reject equality at excluded boundaries."""
    payload = _valid_ntm_reference_artifact()
    surface = cast(dict[str, object], payload["rational_surface"])
    surface[field] = value
    report = _inspect(tmp_path, "rational_surface", surface)
    assert report["status"] == "fail"
    assert any(error["field"] == "rational_surface" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("value", [None, {}, []])
def test_surface_structure(tmp_path: Path, value: object) -> None:
    """Absent or malformed named surface metadata fails as an authored field finding."""
    report = _inspect(tmp_path, "rational_surface", value)
    assert report["status"] == "fail"
    assert any(error["field"] == "rational_surface" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "value", [None, [], [0.1], [True, 0.2], [0.1, 10**400], [0, 0.2], [-1, 0.2], [0.2, 0.1], [0.1, 0.1], [0.1, 0.2]]
)
def test_seed_width_domains(tmp_path: Path, value: object) -> None:
    """Positive ordered seed endpoints preserve equality, while refusing malformed and unrepresentable values."""
    report = _inspect(tmp_path, "seed_island_width_range_m", value)
    accepted = value in [[0.1, 0.1], [0.1, 0.2]]
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(
            error["field"] == "seed_island_width_range_m" for error in cast(list[dict[str, object]], report["errors"])
        )


@pytest.mark.parametrize("field", ["power_W", "current_A", "deposition_width_m", "alignment_error_m"])
@pytest.mark.parametrize("value", [None, True, "1", -1, 10**400, float("inf"), float("nan"), 0, 1])
def test_eccd_numeric_domains(tmp_path: Path, field: str, value: object) -> None:
    """ECCD power/current/error admit zero; deposition width is positive, all values finite and representable."""
    payload = _valid_ntm_reference_artifact()
    alignment = cast(dict[str, object], payload["eccd_alignment"])
    alignment[field] = value
    report = _inspect(tmp_path, "eccd_alignment", alignment)
    accepted = (value == 1 and not isinstance(value, bool)) or (value == 0 and field != "deposition_width_m")
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "eccd_alignment" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("value", [None, {}, []])
def test_eccd_structure(tmp_path: Path, value: object) -> None:
    """ECCD block and all required scalars must be present in a dictionary."""
    report = _inspect(tmp_path, "eccd_alignment", value)
    assert report["status"] == "fail"
    assert any(error["field"] == "eccd_alignment" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, True, "1", -1, 10**400, float("inf"), float("nan")])
def test_error_numeric_domains(tmp_path: Path, block: str, value: object) -> None:
    """Declared errors and bounds refuse malformed, negative, nonfinite and overflowing values."""
    payload = _valid_ntm_reference_artifact()
    mapping = cast(dict[str, object], payload[block])
    mapping["rational_surface_rho_error"] = value
    report = _inspect(tmp_path, block, mapping)
    assert report["status"] == "fail"
    assert any(
        error["field"] == "rational_surface_rho_error" for error in cast(list[dict[str, object]], report["errors"])
    )


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
def test_error_blocks_require_objects(tmp_path: Path, block: str) -> None:
    """Both declared error and limit blocks require objects before numeric use."""
    report = _inspect(tmp_path, block, [])
    assert report["status"] == "fail"
    assert any(error["field"] == block for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "field",
    [
        "rational_surface_rho_error",
        "island_growth_relative_error",
        "saturated_width_relative_error",
        "suppression_time_relative_error",
        "eccd_alignment_error_m",
    ],
)
@pytest.mark.parametrize("mode", ["equal", "zero-error", "zero-bound", "excess"])
def test_each_error_boundary(tmp_path: Path, field: str, mode: str) -> None:
    """All five metrics honor their own positive bounds, admitting equality and zero errors."""
    payload = _valid_ntm_reference_artifact()
    errors = cast(dict[str, object], payload["metrics"])
    bounds = cast(dict[str, object], payload["tolerances"])
    errors[field] = bounds[field]
    if mode == "zero-error":
        errors[field] = 0
    elif mode == "zero-bound":
        bounds[field] = 0
    elif mode == "excess":
        errors[field] = 1
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_ntm_reference(path)
    assert report["status"] == ("pass" if mode in {"equal", "zero-error"} else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == field for error in report["errors"])


def test_admitted_declarations_do_not_authenticate_physics(tmp_path: Path) -> None:
    """Retain uncapped integer modes, signed finite shear and unordered q without inventing interpolation/equality tests."""
    payload = _valid_ntm_reference_artifact()
    surface = cast(dict[str, object], payload["rational_surface"])
    surface.update(m=10**400, n=10**400, q=8, shear=-1.25, rho=0.3)
    payload["q_profile"] = [3, 2, 4, 1, 5]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_ntm_reference(path)["status"] == "pass"
