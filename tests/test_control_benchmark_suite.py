# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Control Benchmark Suite.

"""Exercise scalar benchmark contracts through real PI runs and exported records.

Analytic two/three-step recurrences establish metric values independently of
implementation helpers. Protocol-adversarial controllers test the actual plugin
boundary; they do not provide evidence for numerical or facility fidelity.
"""

from __future__ import annotations

import dataclasses
import doctest
import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import pytest

from validation.control_benchmark_suite import (
    REPO_ROOT,
    BenchmarkRunner,
    BenchmarkScenario,
    ControllerWrapper,
    PIDWrapper,
    setpoint_tracking,
)


def _constant(t: float) -> float:
    """Hold unit reference for analytical scalar-plant experiments."""
    return 1.0


def _scenario(reference: Callable[[float], float] = _constant, /, **changes: Any) -> BenchmarkScenario:
    """Construct a two-second, one-second-grid experiment with explicit overrides."""
    return dataclasses.replace(BenchmarkScenario("scalar", {"dt": 1.0}, reference, 2.0, n_episodes=1), **changes)


def test_untracked_constant_reference_is_not_settled() -> None:
    """Zero PI input leaves unit error; right-endpoint IAE/ISE/ITAE are 2/2/3."""
    result = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario()]).run()[0]
    assert (result.iae, result.ise, result.itae, result.control_effort) == (2, 2, 3, 0)
    assert result.settling_time_s is None
    assert result.settled_episodes == 0
    assert result.max_overshoot_pct == 0
    assert result.schema == "synthetic-control-benchmark/v2"
    assert result.synthetic and not result.facility_validation and not result.realtime_guarantee


def test_pi_recurrence_integrals_and_applied_effort() -> None:
    """Kp=1 gives x=[1,.5], u=[1,0], so IAE=.5, ISE=.25, ITAE=1."""
    result = BenchmarkRunner([PIDWrapper(1, 0)], [_scenario()]).run()[0]
    assert (result.iae, result.ise, result.itae, result.control_effort) == (0.5, 0.25, 1, 1)
    assert result.settling_time_s is None
    assert result.violations == 0
    assert result.computation_time_us >= 0


@pytest.mark.parametrize("target", [1.0, -1.0])
def test_signed_overshoot_and_sampled_settling(target: float) -> None:
    """Kp=2 overshoots either signed unit target by100% on the first sample."""
    scenario = _scenario(lambda t: target, duration_s=1.0)
    result = BenchmarkRunner([PIDWrapper(2, 0)], [scenario]).run()[0]
    assert result.max_overshoot_pct == 100
    assert result.settling_time_s is None


def test_settling_is_first_in_band_sample_not_last_outside() -> None:
    """Constant derivative forcing balances decay after x reaches1 at t=1."""
    scenario = _scenario(disturbances=[(0, "step", 1), (1, "step", -0.5)])
    result = BenchmarkRunner([PIDWrapper(0, 0)], [scenario]).run()[0]
    assert result.settling_time_s == 1
    assert result.settled_episodes == 1
    assert result.max_overshoot_pct == 0
    assert result.iae == 0


def test_final_downstep_to_zero_has_finite_metrics() -> None:
    """A2->0 final reference uses signed amplitude -2, never divides by final0."""
    scenario = _scenario(lambda t: 2.0 if t < 1 else 0.0, disturbances=[(0, "step", 2), (1, "step", -3)])
    result = BenchmarkRunner([PIDWrapper(0, 0)], [scenario]).run()[0]
    assert result.iae == 0
    assert result.max_overshoot_pct == 0
    assert result.settling_time_s == 1


def test_only_final_reference_segment_defines_overshoot() -> None:
    """Earlier excursion20 must not count against the later1->2 step response."""
    scenario = _scenario(lambda t: 1.0 if t < 1 else 2.0, disturbances=[(0, "step", 20), (1, "step", -28)])
    result = BenchmarkRunner([PIDWrapper(0, 0)], [scenario]).run()[0]
    assert result.max_overshoot_pct == 0
    assert result.settling_time_s == 1
    assert result.iae == 19


def test_zero_amplitude_has_undefined_percentage() -> None:
    """Zero reference/state settles at the first sample; a percentage has no scale."""
    result = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario(lambda t: 0.0)]).run()[0]
    assert result.max_overshoot_pct is None
    assert result.settling_time_s == 1


def test_violation_samples_are_totalled_across_episodes() -> None:
    """A persistent derivative10 exceeds3.5 twice per episode, totalling6."""
    scenario = _scenario(n_episodes=3, disturbances=[(0, "step", 10)])
    result = BenchmarkRunner([PIDWrapper(0, 0)], [scenario]).run()[0]
    assert result.violations == 6
    assert result.iae == 23
    assert result.n_episodes == 3


def test_repeat_run_replaces_results_and_keeps_returned_snapshot() -> None:
    """A second PI campaign resets state and cannot append into an earlier result list."""
    runner = BenchmarkRunner([PIDWrapper()], [setpoint_tracking()])
    first = runner.run()
    second = runner.run()
    assert len(first) == len(second) == len(runner.results) == 1
    assert first is not second and second is not runner.results
    assert first[0].iae == second[0].iae
    second.clear()
    assert len(runner.results) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        first[0].__setattr__("iae", -1)


def test_failure_after_an_executed_pair_preserves_previous_results() -> None:
    """A real PI overflow in pair2 must not publish the already-completed pair1."""
    runner = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario()])
    prior = runner.run()
    runner.controllers.append(PIDWrapper(1e308, 1e308))
    with pytest.raises(ValueError, match="PI output"):
        runner.run()
    assert runner.results == prior


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"env_config": {"unknown": 1.0}},
        {"env_config": {"dt": 0.0}},
        {"env_config": {"dt": -1.0}},
        {"env_config": {"dt": float("nan")}},
        {"env_config": {"dt": True}},
        {"duration_s": 0.0},
        {"duration_s": -1.0},
        {"duration_s": 0.5},
        {"duration_s": 1.5},
        {"duration_s": float("inf")},
        {"env_config": {"dt": 1e-308}, "duration_s": 1e308},
        {"n_episodes": 0},
        {"n_episodes": -1},
        {"n_episodes": 1.5},
        {"n_episodes": True},
        {"seed": -1},
        {"seed": True},
        {"disturbances": [(-1, "step", 1)]},
        {"disturbances": [(0, "pulse", 1)]},
        {"disturbances": [(0, "step", float("nan"))]},
        {"disturbances": [(0, "step", 1e308), (0, "step", 1e308)]},
    ],
)
def test_invalid_scenario_refused_before_controller_reset(changes: dict[str, object]) -> None:
    """Reject malformed metadata, fractional grids and invalid forcing before state mutation."""
    controller = PIDWrapper()
    controller.step(np.zeros(1), np.ones(1), 1)
    integral = controller.integral
    with pytest.raises(ValueError):
        BenchmarkRunner([controller], [_scenario(**changes)]).run()
    assert controller.integral == integral


def test_decimal_grid_is_not_silently_truncated() -> None:
    """Floating-point0.3/0.1 represents three intervals despite binary roundoff."""
    result = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario(env_config={"dt": 0.1}, duration_s=0.3)]).run()[0]
    assert result.n_steps == 3
    assert result.iae == pytest.approx(0.3)


def test_default_dt_is_explicit_in_result() -> None:
    """An empty environment configuration selects0.05s, not an unspecified simulator."""
    result = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario(env_config={}, duration_s=0.1)]).run()[0]
    assert result.dt_s == 0.05 and result.n_steps == 2


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "1"])
def test_invalid_reference_refused(invalid: object) -> None:
    """Reject nonfinite or non-real callback results before simulation begins."""
    with pytest.raises(ValueError, match="reference"):
        BenchmarkRunner([PIDWrapper()], [_scenario(lambda t: cast(float, invalid))]).run()


@pytest.mark.parametrize("controllers,scenarios", [([], [_scenario()]), ([PIDWrapper()], [])])
def test_empty_comparison_refused(controllers: list[ControllerWrapper], scenarios: list[BenchmarkScenario]) -> None:
    """An empty campaign cannot masquerade as a successful comparison."""
    with pytest.raises(ValueError, match="at least one"):
        BenchmarkRunner(controllers, scenarios).run()


def test_seeded_noise_replays_numerics_without_touching_global_rng() -> None:
    """Each controller/run gets equal observation noise, with independent global NumPy state."""
    before = np.random.get_state()
    assert isinstance(before, tuple)
    scenario = _scenario(name="noise_robustness", n_episodes=3, seed=42)
    runner = BenchmarkRunner([PIDWrapper(), PIDWrapper()], [scenario])
    first, other = runner.run()
    repeated = runner.run()[0]
    assert first.iae == other.iae == repeated.iae
    after = np.random.get_state()
    assert isinstance(after, tuple)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    changed = BenchmarkRunner([PIDWrapper()], [dataclasses.replace(scenario, seed=43)]).run()[0]
    assert changed.iae != first.iae


def test_saturation_metrics_use_applied_not_requested_input() -> None:
    """A real PI command100 clips to20 and yields effort400 in one second."""
    scenario = _scenario(lambda t: 100.0, name="actuator_saturation", duration_s=1)
    result = BenchmarkRunner([PIDWrapper(1, 0)], [scenario]).run()[0]
    assert result.control_effort == 400 and result.iae == 80


class MutatingPI(PIDWrapper):
    """Valid PI plugin that mutates input buffers after computing its output."""

    def step(self, obs: npt.NDArray[np.float64], ref: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """Exercise plugin ownership using private input snapshots."""
        output = super().step(obs, ref, dt)
        obs[:] = 1e6
        ref[:] = -1e6
        return output


def test_controller_cannot_mutate_plant_or_reference_history() -> None:
    """Input mutation by a controller leaves the actual PI recurrence unchanged."""
    normal, mutating = BenchmarkRunner([PIDWrapper(1, 0), MutatingPI(1, 0)], [_scenario()]).run()
    assert normal.iae == mutating.iae == 0.5
    assert normal.itae == mutating.itae == 1


class InvalidOutputPI(PIDWrapper):
    """External plugin violating output shape after executing its real PI update."""

    def step(self, obs: npt.NDArray[np.float64], ref: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """Return two channels so the public runner must refuse broadcasting."""
        output = super().step(obs, ref, dt)
        return np.repeat(output, 2)


def test_invalid_plugin_output_is_not_broadcast_into_plant() -> None:
    """A two-channel plugin cannot silently expand the one-channel plant."""
    with pytest.raises(ValueError, match="controller output"):
        BenchmarkRunner([InvalidOutputPI()], [_scenario()]).run()


class UnnamedPI(PIDWrapper):
    """PI plugin lacking a usable report identity."""

    def name(self) -> str:
        """Return whitespace to exercise report identity admission."""
        return " "


def test_controller_identity_is_validated() -> None:
    """A blank plugin name cannot enter evidence results."""
    with pytest.raises(ValueError, match="controller name"):
        BenchmarkRunner([UnnamedPI()], [_scenario()]).run()


@pytest.mark.parametrize(
    "value",
    [np.array([np.nan]), np.array([1.0, 2.0]), np.array(1.0), np.array([True]), np.array(["1"]), np.array([1j])],
)
def test_direct_pi_rejects_invalid_vectors_without_mutation(value: npt.NDArray[np.float64]) -> None:
    """PI direct callers receive the same shape/finite/real boundary as runner plugins."""
    controller = PIDWrapper()
    controller.integral = 2
    with pytest.raises(ValueError):
        controller.step(value, np.ones(1), 1)
    assert controller.integral == 2


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_direct_pi_refuses_invalid_dt(dt: float) -> None:
    """Nonpositive/nonfinite time increments cannot change the PI integral."""
    controller = PIDWrapper()
    with pytest.raises(ValueError):
        controller.step(np.zeros(1), np.ones(1), dt)
    assert controller.integral == 0


def test_pi_gain_overflow_and_integral_overflow_refused() -> None:
    """Unrepresentable gains and error integration fail explicitly without state publication."""
    with pytest.raises(ValueError, match="Kp"):
        PIDWrapper(10**400, 0)
    controller = PIDWrapper(0, 0)
    with pytest.raises(ValueError, match="integral"):
        controller.step(np.array([-1e308]), np.array([1e308]), 1)
    assert controller.integral == 0


def test_nonfinite_plant_and_metric_arithmetic_refused() -> None:
    """Finite inputs whose plant or squared metric overflows cannot become JSON evidence."""
    with pytest.raises(ValueError, match="plant state"):
        BenchmarkRunner(
            [PIDWrapper(0, 0)], [_scenario(env_config={"dt": 2.0}, disturbances=[(0, "step", 1e308)])]
        ).run()
    with pytest.raises(ValueError, match="benchmark metrics"):
        BenchmarkRunner([PIDWrapper(0, 0)], [_scenario(disturbances=[(0, "step", 1e200)])]).run()


def test_json_and_markdown_preserve_scope_and_null_metrics(tmp_path: Path) -> None:
    """Actual exports distinguish unavailable metrics and carry fixed synthetic authority fields."""
    runner = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario(name="case|name\nsecond")])
    runner.run()
    path = tmp_path / "report.json"
    runner.save_json(path)
    data = json.loads(path.read_text())
    assert data[0]["settling_time_s"] is None
    assert data[0]["schema"] == "synthetic-control-benchmark/v2"
    assert data[0]["facility_validation"] is False
    runner.save_markdown(tmp_path / "report.md")
    markdown = (tmp_path / "report.md").read_text()
    assert "not observed" in markdown and "case\\|name second" in markdown
    assert "no facility validation" in markdown
    zero = BenchmarkRunner([PIDWrapper(0, 0)], [_scenario(lambda t: 0.0)])
    zero.run()
    zero.save_markdown(tmp_path / "zero.md")
    assert "undefined" in (tmp_path / "zero.md").read_text()


def test_export_refuses_unrecorded_persistent_destination() -> None:
    """Real custody guard rejects repository evidence writes before file creation."""
    runner = BenchmarkRunner([PIDWrapper()], [_scenario()])
    for method in (runner.save_json, runner.save_markdown):
        with pytest.raises(RuntimeError, match="recorded_benchmark"):
            method(REPO_ROOT / "validation/reports/unrecorded-scalar-test.json")


def test_reference_is_sampled_once_for_all_controllers() -> None:
    """Public callback evaluation count is grid size, independent of controller/episode count."""
    calls: list[float] = []

    def reference(t: float) -> float:
        """Record sampled times and return the same unit trajectory."""
        calls.append(t)
        return 1.0

    BenchmarkRunner([PIDWrapper(), PIDWrapper()], [_scenario(reference, n_episodes=3)]).run()
    assert calls == [0.0, 1.0]


def test_rendered_module_example_executes() -> None:
    """Execute the exact native example rendered in the benchmark API reference."""
    from validation import control_benchmark_suite

    result = doctest.testmod(control_benchmark_suite, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 4


@pytest.mark.parametrize("reference,dt", [(1e200, 1e-200), (1e-200, 1e200)])
@pytest.mark.parametrize("gain", [0.0, 1.0])
def test_representable_weighted_integrals_survive_intermediate_square_extremes(
    reference: float, dt: float, gain: float
) -> None:
    """Balanced scalar forcing preserves representable error and PI-effort integrals at reciprocal scales."""
    result = BenchmarkRunner(
        [PIDWrapper(gain, 0)],
        [
            _scenario(
                lambda t: reference, env_config={"dt": dt}, duration_s=dt, disturbances=[(0, "step", -gain * reference)]
            )
        ],
    ).run()[0]
    assert result.iae == pytest.approx(1.0, rel=1e-14, abs=0)
    assert result.ise == pytest.approx(reference, rel=1e-14, abs=0)
    assert result.itae == pytest.approx(dt, rel=1e-14, abs=0)
    assert result.control_effort == pytest.approx(gain * reference, rel=1e-14, abs=0)
