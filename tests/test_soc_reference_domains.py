# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOC reference validation tests

"""Exercise original lattice, learning and error metadata through persisted public validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_soc_reference_validation import _valid_soc_reference_artifact

from validation.validate_soc_reference import validate_soc_reference

ERROR_FIELDS = (
    "mean_turbulence_relative_error",
    "flow_mean_abs_error",
    "policy_action_accuracy_error",
    "reward_relative_error",
    "core_temperature_relative_error",
)


def declaration() -> dict[str, Any]:
    """Supply original engineering SOC metadata with zero errors and unit declared bounds, without a simulation."""
    payload = cast(dict[str, Any], _valid_soc_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    return payload


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a caller declaration and invoke the public SOC reader."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_soc_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Five errors remain nonnegative and within positive finite bounds, including equality."""
    payload = declaration()
    payload[block][field] = value
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("field", ["size", "time_steps", "seed", "max_sub_steps"])
@pytest.mark.parametrize("value", [None, [], "8", True, -1, 0, 1, 7, 8, 10**400, 8.0])
def test_lattice_integer_domains(tmp_path: Path, field: str, value: object) -> None:
    """Size at least eight, positive step counts and nonnegative seed retain uncapped nonboolean integers."""
    payload = declaration()
    payload["lattice_metadata"][field] = value
    minimum = 8 if field == "size" else (0 if field == "seed" else 1)
    accepted = type(value) is int and value >= minimum
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "lattice_metadata" for error in report["errors"])


@pytest.mark.parametrize("field", ["z_crit_base", "flow_generation", "shear_efficiency"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1, 5e-324])
def test_nonnegative_lattice_couplings(tmp_path: Path, field: str, value: object) -> None:
    """Original declared lattice couplings include zero even where the separate runtime has a stricter critical slope."""
    payload = declaration()
    payload["lattice_metadata"][field] = value
    accepted = type(value) in {int, float} and value in [0, 1, 5e-324]
    assert inspect(tmp_path, payload)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("value", [None, [], "0", True, -1, 1, float("nan"), float("inf"), 10**400, 0, 0.999, 5e-324])
def test_half_open_flow_damping(tmp_path: Path, value: object) -> None:
    """Damping admits zero and subunit values, refusing one and unrepresentable/nonfinite declarations."""
    payload = declaration()
    payload["lattice_metadata"]["flow_damping"] = value
    accepted = type(value) in {int, float} and value in [0, 0.999, 5e-324]
    assert inspect(tmp_path, payload)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", ["alpha", "gamma", "epsilon"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, 1.001, float("nan"), float("inf"), 10**400, 0, 1, 5e-324])
def test_inclusive_learning_fractions(tmp_path: Path, field: str, value: object) -> None:
    """Learning fraction declarations retain inclusive zero and one without running Q-learning."""
    payload = declaration()
    payload["learning_metadata"][field] = value
    accepted = type(value) in {int, float} and value in [0, 1, 5e-324]
    assert inspect(tmp_path, payload)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", ["n_states_turb", "n_states_flow", "n_actions"])
@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_uncapped_learning_counts(tmp_path: Path, field: str, value: object) -> None:
    """Positive uncapped nonboolean integer state/action declarations perform no allocation."""
    payload = declaration()
    payload["learning_metadata"][field] = value
    accepted = type(value) is int and value in [1, 10**400]
    assert inspect(tmp_path, payload)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("value", [None, [], "1", True, 0, -1, 1.0, 1, 10**400])
def test_uncapped_reference_case_count(tmp_path: Path, value: object) -> None:
    """Reference case count retains positive uncapped nonboolean integer metadata."""
    payload = declaration()
    payload["reference_case_count"] = value
    accepted = type(value) is int and value in [1, 10**400]
    report = inspect(tmp_path, payload)
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == value


@pytest.mark.parametrize(
    ("block", "field"),
    [
        ("lattice_metadata", name)
        for name in (
            "size",
            "time_steps",
            "seed",
            "z_crit_base",
            "flow_generation",
            "flow_damping",
            "shear_efficiency",
            "max_sub_steps",
        )
    ]
    + [
        ("learning_metadata", name)
        for name in ("alpha", "gamma", "epsilon", "n_states_turb", "n_states_flow", "n_actions", "reward_definition")
    ],
)
def test_required_metadata_keys(tmp_path: Path, block: str, field: str) -> None:
    """Every original lattice and learning field remains required independently."""
    payload = declaration()
    payload[block].pop(field)
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == block for error in report["errors"])


@pytest.mark.parametrize("value", [None, [], True, "", " ", "nonblank"])
def test_reward_definition_presence(tmp_path: Path, value: object) -> None:
    """Reward text is a nonblank declaration, without a parsed or executed reward function."""
    payload = declaration()
    payload["learning_metadata"]["reward_definition"] = value
    assert inspect(tmp_path, payload)["status"] == ("pass" if value == "nonblank" else "fail")


def test_inclusive_error_bounds(tmp_path: Path) -> None:
    """Equality and subnormal positive bounds retain their original behavior; actual excess produces findings."""
    payload = declaration()
    for field in ERROR_FIELDS:
        payload["metrics"][field] = 1.0
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["metrics"][ERROR_FIELDS[0]] = 1.01
    assert inspect(tmp_path, payload)["errors"][0]["field"] == ERROR_FIELDS[0]
    payload["metrics"][ERROR_FIELDS[0]] = 0
    payload["tolerances"][ERROR_FIELDS[0]] = 5e-324
    assert inspect(tmp_path, payload)["status"] == "pass"
