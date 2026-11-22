# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Stress Campaign.
"""Repeated PID and H-infinity episodes through the real flight simulator.

The episodes run the actual ``IsoFluxController`` shot. Its kernel factory is
the simulator's documented seam: a small deterministic equilibrium stand-in
whose flux peak follows the radial and vertical coil currents, so the
controllers act on a moving interior axis. A second stand-in puts the flux
maximum on the box edge, which the simulator's checked axis reader refuses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import validation.stress_test_campaign as campaign
from scpn_control.control.h_infinity_controller import get_flight_sim_controller


class _MovingAxisKernel:
    """Deterministic equilibrium stand-in whose flux peak follows the coil currents."""

    def __init__(self, _config_file: str) -> None:
        """Start from a centred peak on a 13 x 13 grid around R = 6.1 m, Z = 0."""
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.8, 6.4, 13)
        self.Z = np.linspace(-0.3, 0.3, 13)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Move the single flux peak with the radial coil and the top/bottom difference."""
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(self.cfg["coils"][0]["current"])
        ir = int(np.argmin(np.abs(self.R - (6.1 + 0.05 * np.tanh(radial_drive / 10.0)))))
        iz = int(np.argmin(np.abs(self.Z - 0.04 * np.tanh(vertical_drive / 10.0))))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(self.cfg["physics"]["plasma_current_target"])

    def find_x_point(self, _psi: npt.NDArray[np.float64]) -> tuple[tuple[float, float], float]:
        """Return a fixed lower-outboard X-point."""
        return (float(self.R[-2]), float(self.Z[1])), 0.0


class _EdgeMaximumKernel(_MovingAxisKernel):
    """Stand-in whose flux maximum lies on the box edge, so no interior axis exists."""

    def solve_equilibrium(self) -> None:
        """Put the flux maximum in the last radial column."""
        self.Psi.fill(-1.0)
        self.Psi[6, -1] = 1.0


def test_campaign_aggregates_every_registered_controller_from_real_shots() -> None:
    """Each controller runs the requested shots; the deterministic shot gives zero reward spread."""
    results = campaign.run_campaign(n_episodes=3, shot_duration=8, kernel_factory=_MovingAxisKernel)
    assert list(results) == ["PID", "H-infinity"]
    for name, metrics in results.items():
        assert metrics.name == name and metrics.n_episodes == 3 and len(metrics.episodes) == 3
        assert metrics.std_reward == 0.0
        assert len({episode.reward for episode in metrics.episodes}) == 1
        first = metrics.episodes[0]
        assert first.reward == -(first.mean_abs_r_error + first.mean_abs_z_error)
        assert metrics.mean_reward == first.reward and metrics.mean_r_error == first.mean_abs_r_error
        assert 0.0 < metrics.mean_actuator_tracking <= 1.0
        assert 0.0 < metrics.p50_latency_us <= metrics.p95_latency_us <= metrics.p99_latency_us
        assert metrics.error_limit_rate == 0.0 and not first.exceeded_error_limit


def test_h_infinity_episode_steps_one_flight_simulator_controller_per_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The H-infinity shot synthesises the radial and vertical controllers and differs from the PID shot."""
    calls: list[tuple[float, float]] = []

    def recording_factory(response_gain: float = 0.05, actuator_tau: float = 0.06) -> Any:
        """Record the synthesis parameters and return the real controller."""
        calls.append((response_gain, actuator_tau))
        return get_flight_sim_controller(response_gain=response_gain, actuator_tau=actuator_tau)

    monkeypatch.setattr(campaign, "get_flight_sim_controller", recording_factory)
    hinf = campaign._run_hinf_episode("unused.json", 8, _MovingAxisKernel)
    pid = campaign._run_pid_episode("unused.json", 8, _MovingAxisKernel)
    assert calls == [(0.05, 0.06), (0.02, 0.06)]
    assert np.isfinite(hinf.reward) and hinf.reward != pid.reward


def test_episode_limit_flag_follows_the_mean_axis_error() -> None:
    """An episode exceeds the limit when either mean axis error is above it, and only then."""
    summary = {"steps": 4, "mean_abs_r_error": 0.1, "mean_abs_z_error": 0.2, "mean_abs_radial_actuator_lag": 1.0}
    inside = campaign._episode_result(summary, 8_000)
    outside = campaign._episode_result({**summary, "mean_abs_z_error": campaign.MEAN_AXIS_ERROR_LIMIT_M + 0.01}, 8_000)
    assert not inside.exceeded_error_limit and outside.exceeded_error_limit
    assert inside.latency_us == 2.0 and inside.actuator_tracking == 0.5
    assert inside.reward == pytest.approx(-0.3)


def test_an_episode_without_an_interior_axis_stops_the_campaign() -> None:
    """The refused equilibrium stops the campaign instead of producing zero-filled metrics."""
    from scpn_control.core.magnetic_axis import MagneticAxisError

    with pytest.raises(campaign.EpisodeFailure, match=r"^PID episode 0 failed: MagneticAxisError: ") as caught:
        campaign.run_campaign(n_episodes=2, shot_duration=4, kernel_factory=_EdgeMaximumKernel)
    assert isinstance(caught.value.__cause__, MagneticAxisError)


@pytest.mark.parametrize("episodes", [0, -1])
def test_campaign_needs_at_least_one_episode(episodes: int) -> None:
    """A campaign without episodes is refused before any shot."""
    with pytest.raises(ValueError, match="at least 1"):
        campaign.run_campaign(n_episodes=episodes, kernel_factory=_EdgeMaximumKernel)


def test_command_prints_the_table_and_writes_the_aggregates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command runs the quick campaign, prints both rows and writes the JSON aggregates."""
    output = tmp_path / "out" / "stress.json"
    assert campaign.main(["--quick", "--output", str(output)], kernel_factory=_MovingAxisKernel) == 0
    printed = capsys.readouterr().out
    assert "| PID" in printed and "| H-infinity" in printed and f"Results saved to {output}" in printed
    data = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(data) == ["H-infinity", "PID"]
    assert data["PID"]["n_episodes"] == 10 and set(data["PID"]) == {
        "n_episodes",
        "mean_reward",
        "std_reward",
        "mean_r_error",
        "p50_latency_us",
        "p95_latency_us",
        "p99_latency_us",
        "error_limit_rate",
        "mean_actuator_tracking",
    }


def test_command_refuses_and_writes_nothing_when_an_episode_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed episode makes the command exit 1 with the refusal on standard error and no report."""
    output = tmp_path / "stress.json"
    assert campaign.main(["--episodes", "1", "--output", str(output)], kernel_factory=_EdgeMaximumKernel) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Stress campaign refused: PID episode 0 failed: MagneticAxisError")
    assert not output.exists()


def test_command_without_an_output_path_only_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --output the command prints the table and writes no file, not even under its working directory."""
    monkeypatch.chdir(tmp_path)
    assert campaign.main(["--episodes", "1", "--duration", "4"], kernel_factory=_MovingAxisKernel) == 0
    printed = capsys.readouterr().out
    assert "| PID" in printed and "Results saved" not in printed
    assert list(tmp_path.iterdir()) == []


def test_summary_table_has_a_header_separator_and_one_row_per_controller() -> None:
    """The Markdown table lists every controller with its error-limit rate as a percentage."""
    results = campaign.run_campaign(n_episodes=1, shot_duration=4, kernel_factory=_MovingAxisKernel)
    lines = campaign.generate_summary_table(results).splitlines()
    assert len(lines) == 4 and lines[0].startswith("| Controller |") and set(lines[1]) <= {"|", "-"}
    assert lines[2].startswith("| PID") and lines[3].startswith("| H-infinity") and "0.00%" in lines[2]


def test_controllers_registry_has_pid_and_hinf() -> None:
    """CONTROLLERS registry should always have PID and H-infinity."""
    assert list(campaign.CONTROLLERS) == ["PID", "H-infinity"]


def test_flight_sim_controller_factory() -> None:
    """get_flight_sim_controller synthesizes a valid controller."""
    ctrl = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
    assert ctrl.is_stable
    assert ctrl.gamma > 1.0
    assert np.all(np.isfinite(ctrl.F))
    assert np.all(np.isfinite(ctrl.L_gain))


def test_flight_sim_controller_closed_loop_converges() -> None:
    """H-inf controller for flight-sim plant should drive error to zero."""
    ctrl = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
    dt = 0.05
    g = 0.05
    tau = 0.06

    x1, x2 = 0.3, 0.0
    errors = [x1]
    for _ in range(200):
        u = float(np.asarray(ctrl.step(x1, dt)).item())
        dx1 = -g * x2
        dx2 = (u - x2) / tau
        x1 += dx1 * dt
        x2 += dx2 * dt
        errors.append(x1)

    assert abs(errors[-1]) < 0.05 * abs(errors[0])


def test_flight_sim_controller_both_channels() -> None:
    """Radial and vertical controllers should both converge independently."""
    ctrl_R = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
    ctrl_Z = get_flight_sim_controller(response_gain=0.02, actuator_tau=0.06)
    dt = 0.05

    for ctrl, g, e0 in [(ctrl_R, 0.05, 0.2), (ctrl_Z, 0.02, -0.15)]:
        x1, x2 = e0, 0.0
        for _ in range(300):
            u = float(np.asarray(ctrl.step(x1, dt)).item())
            dx1 = -g * x2
            dx2 = (u - x2) / 0.06
            x1 += dx1 * dt
            x2 += dx2 * dt
        assert abs(x1) < 0.1 * abs(e0)


def test_flight_sim_controller_rejects_invalid_params() -> None:
    """Factory should reject non-physical parameter values."""
    with pytest.raises(ValueError):
        get_flight_sim_controller(response_gain=-1.0)
    with pytest.raises(ValueError):
        get_flight_sim_controller(actuator_tau=0.0)
