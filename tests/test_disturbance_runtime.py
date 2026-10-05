# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual disturbance public surface tests.

"""Exercise actual disturbance trajectories, controller integration and reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from scpn_control.control.neuro_cybernetic_controller import SC_NEUROCORE_AVAILABLE
from validation import benchmark_disturbance_rejection as owner


def _cfg(name: str = "VDE", duration: float = 0.005) -> dict[str, object]:
    """Copy a real declared forcing scenario with a bounded actual run horizon."""
    result = dict(owner.SCENARIOS[name])
    result["duration_s"] = duration
    return result


def test_actual_PID_terminal_trace_and_metrics() -> None:
    """Real Euler states include the terminal time and independent quadrature agrees."""
    cfg = _cfg()
    m, trace = owner.run_scenario("PID", owner.PIDController(), "VDE", cfg)
    assert m.stable and m.completed_steps == m.requested_steps == 50
    assert trace.times[-1] == m.completed_duration_s == m.requested_duration_s == 0.005
    assert len(trace.times) == len(trace.positions) == len(trace.errors) == len(trace.controls) == 51
    np.testing.assert_array_equal(trace.errors, -trace.positions)
    independent_ise = sum(0.5 * (trace.errors[k] ** 2 + trace.errors[k + 1] ** 2) * 1e-4 for k in range(50))
    assert m.ise == pytest.approx(independent_ise)
    assert m.control_effort == pytest.approx(float(sum(abs(trace.controls[:-1]))) * 1e-4)
    assert trace.controls[-1] == trace.controls[-2]
    assert m.peak_overshoot == float(np.max(np.abs(trace.errors)))
    assert m.to_dict()["completed_steps"] == 50


def test_actual_uncontrolled_instability_has_no_fabricated_tail() -> None:
    """Zero-gain real PID stops after the first actual bound crossing."""
    m, t = owner.run_scenario("PID", owner.PIDController(kp=0, ki=0, kd=0), "VDE", _cfg(duration=0.2))
    assert not m.stable and not m.settled and m.termination == "position_bound_exceeded"
    assert 0 < m.completed_steps < m.requested_steps == 2000
    assert len(t.times) == m.completed_steps + 1 and t.times[-1] == pytest.approx(m.completed_steps * 1e-4)
    assert abs(t.positions[-1]) > 10.0 and abs(t.positions[-2]) <= 10.0
    np.testing.assert_array_equal(t.errors, -t.positions)
    assert m.completed_duration_s < m.requested_duration_s


def test_initial_bound_failure_records_only_the_initial_state() -> None:
    """An already unbounded real plant requires no fabricated control/trajectory."""
    cfg = _cfg()
    cfg["x0"] = np.array([11.0, 0.0])
    m, t = owner.run_scenario("PID", owner.PIDController(), "VDE", cfg)
    assert not m.stable and m.completed_steps == 0 and m.termination == "initial_bound_exceeded"
    assert t.times.tolist() == [0.0] and t.controls.tolist() == [0.0]
    assert m.ise == m.control_effort == 0.0


def test_actual_settling_distinguishes_terminal_band_and_unsettled_runs() -> None:
    """Actual PID reaches the VDE band; one-step density starts within its finite band."""
    m, _ = owner.run_scenario("PID", owner.PIDController(), "VDE", _cfg(duration=0.2))
    assert m.stable and m.settled and 0.0 < m.settling_time_s < m.completed_duration_s
    short, _ = owner.run_scenario("PID", owner.PIDController(), "VDE", _cfg())
    assert short.stable and not short.settled and short.settling_time_s == short.completed_duration_s
    zero, _ = owner.run_scenario("PID", owner.PIDController(), "Density ramp", _cfg("Density ramp", 1e-4))
    assert zero.settled and zero.settling_time_s == 0.0


def test_actual_controller_factory_and_every_real_scenario() -> None:
    """Available real PID/DGKF/MPC/SC-NeuroCore run all three unchanged forcings."""
    controllers = owner.build_controllers()
    expected = ["PID", "H-infinity", "MPC"] + (["SNN"] if SC_NEUROCORE_AVAILABLE else [])
    assert list(controllers) == expected
    for label, controller in controllers.items():
        for name in owner.SCENARIOS:
            m, t = owner.run_scenario(label, controller, name, _cfg(name, 0.002))
            assert m.stable and len(t.times) == 21
            assert np.all(np.isfinite(t.positions)) and np.all(np.isfinite(t.controls))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True])
def test_positive_controller_and_plant_domains(value: float) -> None:
    """Actual saturation/plant/prediction domains require finite positive values."""
    operations: list[Callable[[], object]] = [
        lambda: owner.PIDController(u_max=value),
        lambda: owner.LinearPlant(value),
        lambda: owner.MPCController(r_weight=value),
        lambda: owner.MPCController(learning_rate=value),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("state", [[0], [0, 1, 2], ["0", "1"], [True, False], [0, float("nan")]])
def test_public_plant_reset_refuses_bad_shapes_and_values(state: object) -> None:
    """Malformed real public reset inputs cannot replace valid plant state."""
    plant = owner.LinearPlant()
    plant.reset(np.array([0.2, 0.3]))
    before = plant.x.copy()
    with pytest.raises(ValueError):
        plant.reset(cast(npt.ArrayLike, state))
    np.testing.assert_array_equal(plant.x, before)


def test_real_plant_step_reset_and_arithmetic_state_preservation() -> None:
    """Actual Euler equations and failed finite arithmetic preserve the old state."""
    p = owner.LinearPlant()
    p.reset(np.array([0.01, 2.0]))
    assert p.step(3, 4, 0.001) == 0.012
    assert p.dz == pytest.approx(2.085)
    previous = p.x.copy()
    with pytest.raises(ValueError):
        p.step(float("nan"), 0, 0.001)
    np.testing.assert_array_equal(previous, p.x)
    p.reset(np.array([1e308, 0]))
    with pytest.raises(ValueError):
        p.step(0, 0, 1)
    assert p.z == 1e308
    p.reset()
    assert p.z == p.dz == 0.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("duration_s", 0),
        ("dt_s", 0),
        ("duration_s", 0.00015),
        ("duration_s", 0.00001),
        ("target_z", float("nan")),
        ("settling_threshold", 0),
        ("settling_threshold", 1.1),
        ("x0", [0]),
        ("disturbance", 7),
        ("dt_s", 1e308),
    ],
)
def test_actual_run_rejects_invalid_configuration(key: str, value: object) -> None:
    """Actual scenario validation refuses malformed declared inputs before controller execution."""
    cfg = _cfg()
    cfg[key] = value
    with pytest.raises(ValueError):
        owner.run_scenario("PID", owner.PIDController(), "VDE", cfg)


def test_actual_missing_configuration_and_metric_overflow_refuse() -> None:
    """Missing required configuration or nonrepresentable observed metrics cannot report stable."""
    cfg = _cfg()
    del cfg["x0"]
    with pytest.raises(ValueError):
        owner.run_scenario("PID", owner.PIDController(), "VDE", cfg)
    cfg = _cfg()
    cfg["x0"] = np.array([1e308, 0.0])
    with pytest.raises(ValueError):
        owner.run_scenario("PID", owner.PIDController(), "VDE", cfg)


def test_actual_public_reports_and_plots_use_observed_custom_horizon(tmp_path: Path) -> None:
    """Real public renderers retain custom target/time/band and require actual trace keys."""
    cfg = _cfg("Density ramp", 0.005)
    cfg["target_z"] = 0.1
    m, t = owner.run_scenario("PID", owner.PIDController(), "custom/target", cfg)
    report = owner.generate_json_results([m])
    assert report["complete_controller_cohort"] is False
    assert report["missing_controllers"] == ["H-infinity", "MPC", "SNN"]
    model, rows = report["model_contract"], report["results"]
    assert isinstance(model, dict) and model["physical_reference_admitted"] is False
    assert isinstance(rows, list) and rows[0]["requested_duration_s"] == 0.005
    markdown = owner.generate_markdown_report([m])
    assert "0.005" in markdown and "ITER" not in markdown
    assert "m² s" in owner.generate_stdout_table([m])
    plots = owner.save_overlay_plots({("PID", "custom/target"): t}, [m], tmp_path)
    assert len(plots) == 1 and Path(plots[0]).name == "benchmark_custom_target.png"
    assert Path(plots[0]).read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        owner.save_overlay_plots({}, [m], tmp_path)
    with pytest.raises(ValueError):
        owner.generate_json_results([replace(m, ise=float("nan"))])
    empty = owner.generate_json_results([])
    assert empty["all_runs_completed_bounded"] is False


def test_actual_public_main_rejects_nonboolean_strict_mode(tmp_path: Path) -> None:
    """The compatible public main refuses a nonboolean strict declaration before writes."""
    out = tmp_path / "invalid-strict"
    with pytest.raises(ValueError):
        owner.main(str(out), duration_scale=0.001, require_complete=cast(bool, 1))
    assert not out.exists()
