# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual local transport and declared comparison-reader tests.

"""Exercise real transport and probe declared profile/report contracts.

Reference dictionaries in reader tests are explicitly derived declarations
from actual local observations; they never stand in for an executed TORAX.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation.code_to_code_benchmark import (
    ITER_SCENARIO,
    build_comparison_report,
    compare_transport_profiles,
    run_local_transport,
    verify_payload_digest,
    write_torax_config,
)
from validation.code_to_code_scenario import initial_profiles, validate_scenario


def _run_in(directory: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    """Invoke the real public solver API from a test-owned working directory."""
    previous = Path.cwd()
    os.chdir(directory)
    try:
        return run_local_transport(scenario)
    finally:
        os.chdir(previous)


@pytest.fixture(scope="module")
def local_observation(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Measure an actual two-step 17-point local solver run once for reader probes."""
    scenario = dict(ITER_SCENARIO, n_rho=17, n_steps=2, t_final=0.02, T_e0=17.0, T_i0=11.0, n_e0=8.0)
    return _run_in(tmp_path_factory.mktemp("code-to-code-real-local"), scenario)


def _declared_reference(local: dict[str, Any]) -> dict[str, Any]:
    """Copy actual local vectors as a declared reference; no provider is invoked."""
    return {
        "code": "torax",
        "scenario": local["scenario"],
        "status": "done",
        "rho": list(local["rho"]),
        "Te_final": list(local["Te_final"]),
        "Ti_final": list(local["Ti_final"]),
        "tau_E_s": 1.1,
    }


def test_actual_initial_profiles_species_grid_and_unique_configuration(tmp_path: Path) -> None:
    """Observe exact declared initialization while preserving the old fixed-name config."""
    sentinel = tmp_path / "validation/reports/_tmp_c2c_config.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"owner-supplied config")
    scenario = dict(ITER_SCENARIO, n_rho=13, n_steps=0, t_final=0.0, T_e0=20.0, T_i0=8.0, n_e0=12.0)
    result = _run_in(tmp_path, scenario)
    rho = np.linspace(0, 1, 13)
    np.testing.assert_array_equal(result["rho"], rho)
    np.testing.assert_allclose(result["Te_initial"], 20.0 - 19.0 * rho, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result["Ti_initial"], 8.0 - 7.6 * rho, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result["ne_initial"], 12.0 - 10.8 * rho, rtol=0, atol=2e-15)
    for channel in ("Te", "Ti", "ne"):
        np.testing.assert_array_equal(result[channel + "_initial"], result[channel + "_final"])
    np.testing.assert_array_equal(result["n_D_initial"], np.asarray(result["ne_initial"]) * 0.5)
    np.testing.assert_array_equal(result["n_T_initial"], result["n_D_initial"])
    np.testing.assert_array_equal(result["n_He_initial"], np.zeros(13))
    assert sentinel.read_bytes() == b"owner-supplied config"
    assert not list(tmp_path.glob("scpn-c2c-*"))
    assert result["t_final"] == result["n_steps"] == 0
    assert result["transport_model"] == "gyro_bohm"
    assert result["unmapped_scenario_fields"] == ["B0", "delta"]


def test_actual_evolution_and_declared_input_digest(local_observation: dict[str, Any]) -> None:
    """Observe evolved finite state and independently bind its normalized inputs."""
    result = local_observation
    assert result["n_steps"] == 2 and result["dt"] == 0.01 and result["t_final"] == 0.02
    for channel in ("Te", "Ti", "ne"):
        assert len(result[channel + "_final"]) == 17
        assert np.isfinite(result[channel + "_final"]).all()
    assert result["Te_initial"] != result["Te_final"]
    assert result["Ti_initial"] != result["Ti_final"]
    assert result["wall_time_s"] >= 0
    scenario = dict(ITER_SCENARIO, n_rho=17, n_steps=2, t_final=0.02, T_e0=17.0, T_i0=11.0, n_e0=8.0)
    expected = hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert result["scenario_sha256"] == expected


@pytest.mark.parametrize("power", [0.0, 1.0])
def test_public_integer_inputs_and_initial_profile_sampling(tmp_path: Path, power: float) -> None:
    """Use accepted NumPy integer counts and positive edge sampling through public APIs."""
    scenario = dict(ITER_SCENARIO, n_rho=np.int64(7), n_steps=np.int64(0), t_final=0.0, P_aux=power, T_e0=0.2)
    normalized = validate_scenario(scenario)
    assert type(normalized["n_rho"]) is int and type(normalized["n_steps"]) is int
    actual = _run_in(tmp_path, scenario)
    sampled = initial_profiles(scenario, np.asarray(actual["rho"], dtype=np.float64))
    np.testing.assert_array_equal(sampled["Te_initial"], actual["Te_initial"])
    assert actual["Te_initial"][-1] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", ""),
        ("name", 7),
        ("T_e0", None),
        ("T_e0", True),
        ("T_e0", np.bool_(False)),
        ("T_e0", "10"),
        ("T_e0", float("nan")),
        ("T_e0", float("inf")),
        ("T_e0", 10**400),
        ("T_e0", 0),
        ("dt", 0),
        ("P_aux", -1),
        ("t_final", -1),
        ("delta", 1),
        ("n_rho", True),
        ("n_rho", 50.0),
        ("n_rho", 2),
        ("n_steps", False),
        ("n_steps", -1),
        ("n_steps", 1.0),
        ("R0", 1),
        ("t_final", 0.5),
    ],
)
def test_invalid_scenario_refuses_before_configuration(tmp_path: Path, key: str, value: Any) -> None:
    """Refuse malformed domains before the real API creates a configuration."""
    with pytest.raises(ValueError):
        _run_in(tmp_path, dict(ITER_SCENARIO, **{key: value}))
    assert list(tmp_path.iterdir()) == []


def test_missing_scenario_field_refuses_without_files(tmp_path: Path) -> None:
    """A missing required control is refused before computation or writes."""
    scenario = dict(ITER_SCENARIO)
    del scenario["I_p"]
    with pytest.raises(ValueError):
        _run_in(tmp_path, scenario)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rho", [[0.0], [0.0, 0.0], [0.1, 1.1], [False, 1.0], [1.0, 0.0]])
def test_initial_profile_grid_contract(rho: list[float]) -> None:
    """Reject degenerate, unordered, out-of-domain and boolean radial coordinates."""
    with pytest.raises(ValueError):
        initial_profiles(ITER_SCENARIO, np.asarray(rho, dtype=object))


def test_equal_count_different_coordinates_use_interpolation(local_observation: dict[str, Any]) -> None:
    """Compare measured local profiles with declared shifted-grid samples by rho."""
    reference = _declared_reference(local_observation)
    reference["rho"] = np.asarray(reference["rho"]) ** 2
    result = compare_transport_profiles(local_observation, reference)
    expected = np.interp(local_observation["rho"], reference["rho"], reference["Te_final"])
    differences = np.asarray(local_observation["Te_final"]) - expected
    assert result["comparison"]["Te_rmse_keV"] == pytest.approx(np.sqrt(np.mean(differences**2)))
    assert result["comparison"]["Te_rmse_keV"] > 0
    assert result["comparison"]["Te_max_diff_keV"] == pytest.approx(np.max(np.abs(differences)))
    assert result["comparison_findings"] == []


def test_different_count_optional_ion_and_identical_profiles(local_observation: dict[str, Any]) -> None:
    """Exercise actual reference coordinates, optional ion metrics and exact zero RMSE."""
    reference = _declared_reference(local_observation)
    same = compare_transport_profiles(local_observation, reference)
    assert same["comparison"]["Te_rmse_keV"] == same["comparison"]["Ti_rmse_keV"] == 0
    reference["rho"] = reference["rho"][::2]
    reference["Te_final"] = reference["Te_final"][::2]
    del reference["Ti_final"]
    comparison = compare_transport_profiles(local_observation, reference)
    assert "Te_rmse_keV" in comparison["comparison"]
    assert "Ti_rmse_keV" not in comparison["comparison"]
    assert comparison["comparison"]["torax_tau_E_s"] == 1.1
    del reference["tau_E_s"]
    assert "torax_tau_E_s" not in compare_transport_profiles(local_observation, reference)["comparison"]
    assert compare_transport_profiles(local_observation, None)["comparison"] == {}


@pytest.mark.parametrize("bad", [None, [], [[1.0]], ["1", "2"], [True, 2], [0.0, 10**400], [0.0, float("nan")]])
def test_declared_profile_numeric_refusals(local_observation: dict[str, Any], bad: Any) -> None:
    """Refuse malformed declared samples; no external simulation is substituted."""
    reference = _declared_reference(local_observation)
    reference["Te_final"] = bad
    result = compare_transport_profiles(local_observation, reference)
    assert result["comparison"] == {}
    assert result["comparison_findings"] == ["torax_profile_coordinates_or_values"]


def test_excessive_array_depth_is_an_authored_refusal(local_observation: dict[str, Any]) -> None:
    """A genuinely unrepresentable nested array is rejected without conversion text."""
    value: Any = [1.0]
    for _ in range(40):
        value = [value]
    reference = _declared_reference(local_observation)
    reference["rho"] = value
    result = compare_transport_profiles(local_observation, reference)
    assert result["comparison"] == {} and result["comparison_findings"]


def test_coordinate_shape_order_and_domain_refusals(local_observation: dict[str, Any]) -> None:
    """Reject short, reversed, repeated, uncovered and mismatched coordinate arrays."""
    for rho in ([0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 1.0]):
        reference = _declared_reference(local_observation)
        reference["rho"] = rho
        assert compare_transport_profiles(local_observation, reference)["comparison"] == {}
    local = dict(local_observation, Te_final=[1.0])
    result = compare_transport_profiles(local, _declared_reference(local_observation))
    assert result["comparison_findings"] == ["scpn_profile_coordinates_or_values"]
    reference = _declared_reference(local_observation)
    reference["rho"] = np.linspace(0.1, 0.9, 17).tolist()
    assert compare_transport_profiles(local_observation, reference)["comparison_findings"] == [
        "reference_grid_does_not_cover_local_grid"
    ]


@pytest.mark.parametrize("tau", [True, "1", float("inf"), 10**400])
def test_optional_scalar_cannot_bypass_numeric_refusal(local_observation: dict[str, Any], tau: Any) -> None:
    """Invalid declared confinement scalars clear otherwise valid diagnostic metrics."""
    reference = _declared_reference(local_observation)
    reference["tau_E_s"] = tau
    result = compare_transport_profiles(local_observation, reference)
    assert result["comparison"] == {}
    assert result["comparison_findings"] == ["torax_tau_E_not_finite"]


def test_difference_overflow_is_refused(local_observation: dict[str, Any]) -> None:
    """Finite declared inputs with unrepresentable subtraction cannot yield metrics."""
    local = dict(local_observation, Te_final=[1e308] * 17)
    reference = _declared_reference(local_observation)
    reference["Te_final"] = [-1e308] * 17
    result = compare_transport_profiles(local, reference)
    assert result["comparison"] == {}
    assert result["comparison_findings"] == ["temperature_difference_not_representable"]


def test_declared_report_consistency_does_not_admit_physics(local_observation: dict[str, Any]) -> None:
    """Bind declared metrics while retaining all source-backed configured-model gaps."""
    comparison = compare_transport_profiles(local_observation, _declared_reference(local_observation))
    report = build_comparison_report(comparison, ITER_SCENARIO, requested_torax=True)
    external = report["external_reference"]
    assert external["admitted"] is False and external["status"] == "blocked"
    assert external["diagnostic_comparison_available"] is True
    assert external["blocked_reasons"] == report["model_contract"]["limitations"]
    assert verify_payload_digest(report)
    changed = copy.deepcopy(report)
    changed["benchmark"]["scpn_control"]["Te_final"][0] += 1
    assert not verify_payload_digest(changed)
    for digest in (None, 1, "", "a" * 63):
        assert not verify_payload_digest(dict(report, payload_sha256=digest))
    assert not verify_payload_digest({"payload_sha256": "a" * 64, "unencodable": object()})
    assert not verify_payload_digest({"payload_sha256": "a" * 64, "value": float("nan")})


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("not_requested", "torax_not_requested"),
        ("missing", "torax_not_available_or_failed"),
        ("wrong_type", "torax_payload_identity"),
        ("wrong_code", "torax_payload_identity"),
        ("bad_profile", "torax_numeric_payload"),
        ("missing_metrics", "comparison_metrics_missing"),
        ("wrong_metrics_type", "comparison_metrics_missing"),
        ("boolean_metric", "comparison_metrics_non_finite"),
        ("local_identity", "scpn_payload_identity"),
        ("local_type", "scpn_payload_identity"),
        ("local_numeric", "scpn_numeric_payload"),
        ("boolean_scalar", "torax_numeric_payload"),
    ],
)
def test_report_status_refusals_are_declarations(local_observation: dict[str, Any], change: str, reason: str) -> None:
    """Public report construction classifies malformed declarations without provider claims."""
    reference = _declared_reference(local_observation)
    comparison = compare_transport_profiles(local_observation, reference)
    requested = True
    if change == "not_requested":
        requested = False
    elif change == "missing":
        comparison["torax"] = None
    elif change == "wrong_type":
        comparison["torax"] = []
    elif change == "wrong_code":
        reference["code"] = "scpn-control"
    elif change == "bad_profile":
        reference["Ti_final"] = [True] * 17
    elif change == "missing_metrics":
        comparison["comparison"] = {}
    elif change == "wrong_metrics_type":
        comparison["comparison"] = []
    elif change == "boolean_metric":
        comparison["comparison"]["Te_rmse_keV"] = True
    elif change == "local_identity":
        comparison["scpn_control"] = dict(local_observation, code="other")
    elif change == "local_type":
        comparison["scpn_control"] = []
    elif change == "local_numeric":
        comparison["scpn_control"] = dict(local_observation, rho=[True] * 17)
    elif change == "boolean_scalar":
        reference["wall_time_s"] = True
    report = build_comparison_report(comparison, ITER_SCENARIO, requested_torax=requested)
    assert reason in report["external_reference"]["blocked_reasons"]
    assert report["external_reference"]["admitted"] is False


def test_exported_provider_controls_match_actual_local_initialization(tmp_path: Path) -> None:
    """Prepare real Python configuration bytes and match units against actual local profiles."""
    scenario = dict(ITER_SCENARIO, n_rho=13, n_steps=0, t_final=0.0, T_e0=20.0, T_i0=8.0, n_e0=12.0)
    local = _run_in(tmp_path, scenario)
    path = tmp_path / "torax/config.py"
    write_torax_config(path, scenario)
    original = path.read_bytes()
    spec = importlib.util.spec_from_file_location("actual_exported_c2c_config", path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.CONFIG
    assert config["numerics"]["fixed_dt"] == local["dt"]
    assert config["numerics"]["t_final"] == local["t_final"]
    assert config["geometry"]["n_rho"] == len(local["rho"])
    for name, channel in (("T_e", "Te"), ("T_i", "Ti")):
        assert config["profile_conditions"][name][0.0][0.0] == local[channel + "_initial"][0]
        assert config["profile_conditions"][name][0.0][1.0] == pytest.approx(local[channel + "_initial"][-1])
        assert config["profile_conditions"][name + "_right_bc"] == pytest.approx(local[channel + "_initial"][-1])
    density = config["profile_conditions"]["n_e"][0.0]
    assert density[0.0] == local["ne_initial"][0] * 1e19
    assert density[1.0] == pytest.approx(local["ne_initial"][-1] * 1e19)
    with pytest.raises(FileExistsError):
        write_torax_config(path, scenario)
    assert path.read_bytes() == original


def test_array_shape_and_scalar_not_vector(local_observation: dict[str, Any]) -> None:
    """NumPy matrices and scalar-like objects cannot be profile vectors."""
    for value in (np.ones((17, 1)), object(), 1.0, np.array([])):
        reference = _declared_reference(local_observation)
        reference["Te_final"] = value
        assert compare_transport_profiles(local_observation, reference)["comparison"] == {}


@pytest.mark.parametrize(
    ("key", "value"),
    [("n_steps", 10**400), ("n_rho", 10**400), ("extra", object()), ("extra", {"value": float("nan")})],
)
def test_counter_overflow_and_invalid_extra_metadata_refuse_before_files(tmp_path: Path, key: str, value: Any) -> None:
    """Observed overflow and encoding errors become domain refusals before config creation."""
    scenario = dict(ITER_SCENARIO, n_steps=0, t_final=0.0)
    scenario[key] = value
    with pytest.raises(ValueError):
        _run_in(tmp_path, scenario)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("requested", [1, "true", None, np.bool_(True)])
def test_provider_request_flag_is_a_boolean_declaration(local_observation: dict[str, Any], requested: Any) -> None:
    """Reject coercible non-boolean provider flags before declaring an external request."""
    with pytest.raises(ValueError):
        build_comparison_report(
            compare_transport_profiles(local_observation, None), ITER_SCENARIO, requested_torax=requested
        )
