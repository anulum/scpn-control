# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller runtime safety admission tests

"""Tests for runtime safety certificate admission in the live controller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from scpn_control.scpn.artifact import Artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController, _artifact_topology_digest, _matrix_entries
from scpn_control.scpn.deadline_monitor import DeadlineMonitor
from scpn_control.scpn.formal_safety_certificate import generate_safety_certificate
from scpn_control.scpn.formal_verification import CTLFormula, EventuallyFires, LTLFormula
from scpn_control.scpn.runtime_safety_certificate import (
    CertificateReplayResult,
    ControllerRuntimeBinding,
    RuntimeTarget,
    TimingEnvelope,
    compute_petri_topology_digest,
    issue_runtime_safety_certificate,
)
from scpn_control.scpn.structure import StochasticPetriNet


def _certified_net(*, threshold: float = 1.0, delay_ticks: int = 0) -> StochasticPetriNet:
    """Return a tiny compiled controller net for admission tests."""
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_transition("disarm", threshold=threshold, delay_ticks=delay_ticks)
    net.add_arc("armed", "disarm", weight=1.0)
    net.add_arc("disarm", "safe", weight=1.0)
    net.compile()
    return net


def _artifact_from_net(net: StochasticPetriNet, *, injection_config: list[dict[str, object]] | None = None) -> Artifact:
    """Compile a test artifact from a controller net."""
    compiled = FusionCompiler(bitstream_length=64, seed=0).compile(net)
    return cast(
        Artifact,
        compiled.export_artifact(
            name="runtime-safety-controller",
            readout_config={
                "actions": [{"name": "disarm_bias", "pos_place": 1, "neg_place": 0}],
                "gains": [1.0],
                "abs_max": [1.0],
                "slew_per_s": [10.0],
            },
            injection_config=[] if injection_config is None else injection_config,
        ),
    )


def _formal_certificate(net: StochasticPetriNet, tmp_path: Path) -> dict[str, Any]:
    """Generate a bounded formal certificate for the test net."""
    return generate_safety_certificate(
        net,
        max_depth=4,
        marking_bounds={"armed": (0.0, 1.0), "safe": (0.0, 1.0)},
        json_path=str(tmp_path / "formal.json"),
        markdown_path=str(tmp_path / "formal.md"),
        temporal_specs=[EventuallyFires("disarm_fires", "disarm")],
        ctl_specs=[
            CTLFormula(
                name="armed_bounded",
                operator="AG",
                target="marking_bounds",
                params={"bounds": {"armed": (0.0, 1.0), "safe": (0.0, 1.0)}},
            )
        ],
        ltl_specs=[LTLFormula("safe_bounded", "G", "marking_bounds", {"bounds": {"safe": (0.0, 1.0)}})],
    )


def _runtime_target() -> RuntimeTarget:
    """Return the declared runtime target for controller admission tests."""
    return RuntimeTarget(name="controller-ci", architecture="x86_64", runtime="numpy", toolchain="pytest")


def _timing_envelope() -> TimingEnvelope:
    """Return a schedulable timing envelope for controller admission tests."""
    return TimingEnvelope(
        control_period_us=1000.0,
        worst_case_response_us=100.0,
        deadline_us=500.0,
        proof_firing_depth=4,
    )


def _binding(net: StochasticPetriNet) -> ControllerRuntimeBinding:
    """Build the runtime binding used by the controller admission tests."""
    return ControllerRuntimeBinding(
        controller_id="runtime-safety-controller",
        controller_config={"runtime_backend": "numpy", "sc_n_passes": 1},
        petri_topology_sha256=compute_petri_topology_digest(net),
        snn_parameters={"bitstream_length": 64, "seed": 0},
        solver_mode="scpn-numpy",
        runtime_target=_runtime_target(),
        timing_envelope=_timing_envelope(),
    )


def _issued_certificate(net: StochasticPetriNet, tmp_path: Path) -> tuple[dict[str, Any], ControllerRuntimeBinding]:
    """Issue a runtime certificate and matching binding for the test net."""
    binding = _binding(net)
    certificate = issue_runtime_safety_certificate(net, binding, formal_certificate=_formal_certificate(net, tmp_path))
    return certificate, binding


def _controller_kwargs(artifact: Artifact) -> dict[str, Any]:
    """Return common controller constructor arguments."""
    return {
        "artifact": artifact,
        "seed_base": 42,
        "targets": ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        "scales": ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        "sc_n_passes": 1,
        "runtime_backend": "numpy",
    }


def _control_relevant_state(controller: NeuroSymbolicController) -> dict[str, object]:
    """Return the state that a rejected strict cycle must not advance."""
    return {
        "marking": controller._marking.tolist(),
        "oracle_pending": controller._oracle_pending.tolist(),
        "sc_pending": controller._sc_pending.tolist(),
        "oracle_cursor": controller._oracle_cursor,
        "sc_cursor": controller._sc_cursor,
        "previous_actions": controller._prev_actions.tolist(),
        "last_oracle_marking": list(controller.last_oracle_marking),
        "last_oracle_firing": list(controller.last_oracle_firing),
        "last_sc_marking": list(controller.last_sc_marking),
        "last_sc_firing": list(controller.last_sc_firing),
    }


def test_controller_admits_matching_runtime_safety_certificate(tmp_path: Path) -> None:
    """Controller admits a certificate that matches the loaded artifact."""
    net = _certified_net()
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)

    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
    )

    assert controller.runtime_safety_admitted is True
    assert controller.runtime_safety_certificate_sha256 == certificate["payload_sha256"]
    actions = cast(Mapping[str, float], controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0))
    assert actions["disarm_bias"] <= 1.0


def test_controller_rejects_partial_runtime_safety_inputs(tmp_path: Path) -> None:
    """Controller fails closed when only part of admission evidence is supplied."""
    net = _certified_net()
    certificate, _binding = _issued_certificate(net, tmp_path)

    with pytest.raises(ValueError, match="requires certificate, binding, target, and replay"):
        NeuroSymbolicController(**_controller_kwargs(_artifact_from_net(net)), runtime_safety_certificate=certificate)


def test_controller_rejects_certificate_for_different_artifact(tmp_path: Path) -> None:
    """Controller compares the runtime binding topology to the loaded artifact."""
    certified = _certified_net()
    drifted = _certified_net(threshold=0.5)
    certificate, binding = _issued_certificate(certified, tmp_path)
    assert _artifact_topology_digest(_artifact_from_net(drifted)) != binding.petri_topology_sha256

    with pytest.raises(ValueError, match="topology does not match"):
        NeuroSymbolicController(
            **_controller_kwargs(_artifact_from_net(drifted)),
            runtime_safety_certificate=certificate,
            runtime_safety_binding=binding,
            runtime_safety_target=_runtime_target(),
            runtime_safety_replay=CertificateReplayResult(True, True, True, True),
        )


def test_controller_rejects_failed_runtime_safety_replay(tmp_path: Path) -> None:
    """Controller delegates replay admission to the runtime certificate gate."""
    net = _certified_net()
    certificate, binding = _issued_certificate(net, tmp_path)

    with pytest.raises(ValueError, match="proof replay"):
        NeuroSymbolicController(
            **_controller_kwargs(_artifact_from_net(net)),
            runtime_safety_certificate=certificate,
            runtime_safety_binding=binding,
            runtime_safety_target=_runtime_target(),
            runtime_safety_replay=CertificateReplayResult(
                topology_matches=True,
                holds_matches=False,
                checked_specs_match=True,
                formal_digest_matches=True,
                detail=["replayed proof does not hold"],
            ),
        )


def test_artifact_matrix_entries_rejects_invalid_shape() -> None:
    """Artifact topology digest guards malformed dense matrix shapes."""
    with pytest.raises(ValueError, match="two dimensions"):
        _matrix_entries([1.0], [1, 1, 1])


def test_artifact_matrix_entries_rejects_invalid_data_length() -> None:
    """Artifact topology digest guards dense matrix shape/data drift."""
    with pytest.raises(ValueError, match="data length"):
        _matrix_entries([1.0], [1, 2])


def test_step_traceable_logs_jsonl_record(tmp_path: Path) -> None:
    """Traceable controller steps write JSONL diagnostics when requested."""
    controller = NeuroSymbolicController(**_controller_kwargs(_artifact_from_net(_certified_net())))
    out = controller.step_traceable([6.2, 0.0], 0, log_path="trace.jsonl", log_root=tmp_path)

    assert out.shape == (1,)
    payload = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert '"actions":{"disarm_bias":' in payload


def test_step_traceable_rejects_passthrough_injections(tmp_path: Path) -> None:
    """Traceable mode fails closed when artifact injections need map lookups."""
    artifact = _artifact_from_net(
        _certified_net(),
        injection_config=[
            {"place_id": 0, "source": "external_density", "scale": 1.0, "offset": 0.0, "clamp_0_1": True}
        ],
    )
    controller = NeuroSymbolicController(**_controller_kwargs(artifact))

    with pytest.raises(RuntimeError, match="axis-only injections"):
        controller.step_traceable([6.2, 0.0], 0)


def test_admitted_controller_wires_and_logs_deadline_monitor(tmp_path: Path) -> None:
    """A controller with an admitted certificate monitors each cycle deadline (SS-14)."""
    net = _certified_net()
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)

    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
    )

    assert isinstance(controller.deadline_monitor, DeadlineMonitor)
    assert controller.deadline_monitor.deadline_us == 500.0
    assert controller.deadline_monitor.strict is False

    controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0, log_path="cycle.jsonl", log_root=tmp_path)
    assert controller.deadline_monitor.cycles == 1

    record = json.loads((tmp_path / "cycle.jsonl").read_text(encoding="utf-8").strip())
    assert record["deadline_us"] == 500.0
    assert isinstance(record["within_deadline"], bool)
    assert record["deadline_overruns"] == controller.deadline_monitor.overruns


def test_strict_deadline_overrun_discards_the_action_and_raises(tmp_path: Path) -> None:
    """In strict mode an overrun raises before return, so the cycle's action is dropped (SS-14)."""
    from scpn_control.scpn.deadline_monitor import DeadlineMonitor, DeadlineOverrunError

    net = _certified_net()
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)
    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
    )
    # Install a strict monitor whose deadline no real cycle can meet, so the very
    # next step deterministically overruns; strict mode must raise (dropping the
    # computed action) rather than return it.
    controller.deadline_monitor = DeadlineMonitor(deadline_us=1.0e-9, strict=True)
    with pytest.raises(DeadlineOverrunError, match="exceeded deadline"):
        controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0)
    assert controller.deadline_monitor.overruns == 1


def test_strict_deadline_overrun_rolls_back_controller_state(tmp_path: Path) -> None:
    """A rejected strict cycle commits no control-relevant state (SS-14 a).

    The real delayed-transition and slew-limited paths mutate queues, cursors, and
    previous actions while calculating a candidate. The strict transaction must
    restore all of them as well as the marking and diagnostic snapshots.
    """
    from scpn_control.scpn.deadline_monitor import DeadlineMonitor, DeadlineOverrunError

    net = _certified_net(delay_ticks=2)
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)
    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
    )

    # One admitted (fail-soft) cycle advances the marking; snapshot the committed state.
    controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0)
    state_after_commit = _control_relevant_state(controller)

    # Now force a deterministic strict overrun on the next cycle.
    controller.deadline_monitor = DeadlineMonitor(deadline_us=1.0e-9, strict=True)
    with pytest.raises(DeadlineOverrunError, match="exceeded deadline"):
        controller.step({"R_axis_m": 3.0, "Z_axis_m": 1.5}, 1)

    # The rejected cycle rolled back every control-relevant mutation. The deadline
    # monitor intentionally retains the overrun as safety telemetry.
    assert _control_relevant_state(controller) == state_after_commit
    assert controller.deadline_monitor.overruns == 1


def test_strict_traceable_overrun_rolls_back_controller_state(tmp_path: Path) -> None:
    """The fixed-order traceable path uses the same strict transaction (SS-14 a)."""
    from scpn_control.scpn.deadline_monitor import DeadlineMonitor, DeadlineOverrunError

    net = _certified_net(delay_ticks=2)
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)
    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
    )

    controller.step_traceable([6.2, 0.0], 0)
    state_after_commit = _control_relevant_state(controller)

    controller.deadline_monitor = DeadlineMonitor(deadline_us=1.0e-9, strict=True)
    with pytest.raises(DeadlineOverrunError, match="exceeded deadline"):
        controller.step_traceable([3.0, 1.5], 1)

    assert _control_relevant_state(controller) == state_after_commit
    assert controller.deadline_monitor.overruns == 1


def test_strict_deadline_monitor_forbids_synchronous_logging(tmp_path: Path) -> None:
    """Strict mode rejects synchronous JSONL logging in the real-time path (SS-14 b).

    Unbounded disk I/O cannot coexist with a hard-real-time deadline, so a
    ``log_path`` write under a strict monitor fails closed before any work and
    leaves no file.
    """
    net = _certified_net()
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)
    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
        deadline_monitor_strict=True,
    )

    with pytest.raises(ValueError, match="not permitted with a strict"):
        controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0, log_path="strict.jsonl", log_root=tmp_path)
    assert not (tmp_path / "strict.jsonl").exists()
    assert controller.deadline_monitor is not None
    assert controller.deadline_monitor.cycles == 0


def test_failsoft_logging_exposes_trace_write_cost(tmp_path: Path) -> None:
    """Fail-soft logging exposes the synchronous trace-write latency (SS-14 b).

    The write happens outside the measured interval; ``last_trace_write_us`` makes
    that otherwise-hidden cost observable so the cycle's true wall-clock is not
    silently under-reported.
    """
    net = _certified_net()
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)
    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
    )

    assert controller.last_trace_write_us is None
    controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0, log_path="cycle.jsonl", log_root=tmp_path)
    assert isinstance(controller.last_trace_write_us, float)
    assert controller.last_trace_write_us >= 0.0


def test_admitted_controller_forwards_strict_deadline_flag(tmp_path: Path) -> None:
    """The deadline_monitor_strict flag reaches the wired monitor."""
    net = _certified_net()
    artifact = _artifact_from_net(net)
    certificate, binding = _issued_certificate(net, tmp_path)

    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=_runtime_target(),
        runtime_safety_replay=CertificateReplayResult(True, True, True, True),
        deadline_monitor_strict=True,
    )

    assert isinstance(controller.deadline_monitor, DeadlineMonitor)
    assert controller.deadline_monitor.strict is True


def test_controller_without_certificate_has_no_deadline_monitor(tmp_path: Path) -> None:
    """A controller with no admitted certificate carries no deadline monitor."""
    controller = NeuroSymbolicController(**_controller_kwargs(_artifact_from_net(_certified_net())))
    assert controller.deadline_monitor is None
    # The monitor-absent branch still steps and logs cleanly.
    controller.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0, log_path="plain.jsonl", log_root=tmp_path)
    record = json.loads((tmp_path / "plain.jsonl").read_text(encoding="utf-8").strip())
    assert "within_deadline" not in record
