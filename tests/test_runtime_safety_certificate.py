# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime-bound formal safety certificate tests

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from scpn_control.scpn.formal_safety_certificate import generate_safety_certificate
from scpn_control.scpn.formal_verification import (
    CTLFormula,
    EventuallyFires,
    LTLFormula,
)
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.runtime_safety_certificate import (
    RUNTIME_SAFETY_CERTIFICATE_CLAIM_BOUNDARY,
    RUNTIME_SAFETY_CERTIFICATE_SCHEMA_VERSION,
    RUNTIME_SAFETY_CERTIFICATE_SCOPE,
    CertificateReplayResult,
    ControllerRuntimeBinding,
    RuntimeTarget,
    TimingEnvelope,
    _digest,
    assert_runtime_certificate_admissible,
    compute_petri_topology_digest,
    issue_runtime_safety_certificate,
    replay_runtime_safety_certificate,
    validate_runtime_safety_certificate_payload,
)


def _disarm_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_transition("disarm", threshold=1.0)
    net.add_arc("armed", "disarm", weight=1.0)
    net.add_arc("disarm", "safe", weight=1.0)
    net.compile()
    return net


def _make_formal(net: StochasticPetriNet) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        return generate_safety_certificate(
            net,
            max_depth=4,
            marking_bounds={"armed": (0.0, 1.0), "safe": (0.0, 1.0)},
            json_path=os.path.join(tmp, "c.json"),
            markdown_path=os.path.join(tmp, "c.md"),
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


def _target() -> RuntimeTarget:
    return RuntimeTarget(name="rpi4-rt", architecture="aarch64", runtime="PREEMPT_RT", toolchain="gcc-13")


def _envelope() -> TimingEnvelope:
    return TimingEnvelope(
        control_period_us=1000.0, worst_case_response_us=180.0, deadline_us=500.0, proof_firing_depth=4
    )


def _binding(net: StochasticPetriNet, **overrides: Any) -> ControllerRuntimeBinding:
    base: dict[str, Any] = {
        "controller_id": "burn-ctl-1",
        "controller_config": {"kp": 1.2, "ki": 0.3},
        "petri_topology_sha256": compute_petri_topology_digest(net),
        "snn_parameters": {"layers": 3, "weights_sha": "abc"},
        "solver_mode": "acados-rti",
        "runtime_target": _target(),
        "timing_envelope": _envelope(),
    }
    base.update(overrides)
    return ControllerRuntimeBinding(**base)


def _issued(net: StochasticPetriNet) -> tuple[dict[str, Any], ControllerRuntimeBinding]:
    binding = _binding(net)
    cert = issue_runtime_safety_certificate(net, binding, formal_certificate=_make_formal(net))
    return cert, binding


# ── topology digest ───────────────────────────────────────────────────


def test_topology_digest_is_deterministic() -> None:
    assert compute_petri_topology_digest(_disarm_net()) == compute_petri_topology_digest(_disarm_net())


def test_topology_digest_ignores_initial_tokens() -> None:
    base = _disarm_net()
    other = StochasticPetriNet()
    other.add_place("armed", initial_tokens=0.25)  # different tokens, same topology
    other.add_place("safe", initial_tokens=0.75)
    other.add_transition("disarm", threshold=1.0)
    other.add_arc("armed", "disarm", weight=1.0)
    other.add_arc("disarm", "safe", weight=1.0)
    other.compile()
    assert compute_petri_topology_digest(base) == compute_petri_topology_digest(other)


def test_topology_digest_changes_with_threshold() -> None:
    base = _disarm_net()
    changed = StochasticPetriNet()
    changed.add_place("armed", initial_tokens=1.0)
    changed.add_place("safe", initial_tokens=0.0)
    changed.add_transition("disarm", threshold=0.5)  # different threshold
    changed.add_arc("armed", "disarm", weight=1.0)
    changed.add_arc("disarm", "safe", weight=1.0)
    changed.compile()
    assert compute_petri_topology_digest(base) != compute_petri_topology_digest(changed)


def test_topology_digest_changes_with_arc_weight() -> None:
    base = _disarm_net()
    changed = StochasticPetriNet()
    changed.add_place("armed", initial_tokens=1.0)
    changed.add_place("safe", initial_tokens=0.0)
    changed.add_transition("disarm", threshold=1.0)
    changed.add_arc("armed", "disarm", weight=0.5)  # different arc weight
    changed.add_arc("disarm", "safe", weight=1.0)
    changed.compile()
    assert compute_petri_topology_digest(base) != compute_petri_topology_digest(changed)


def test_topology_digest_requires_compiled_net() -> None:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_transition("disarm", threshold=1.0)
    net.add_arc("armed", "disarm", weight=1.0)
    net.add_arc("disarm", "safe", weight=1.0)
    with pytest.raises(ValueError, match="compiled net"):
        compute_petri_topology_digest(net)


# ── RuntimeTarget ─────────────────────────────────────────────────────


def test_runtime_target_rejects_empty_field() -> None:
    with pytest.raises(ValueError, match="architecture"):
        RuntimeTarget(name="rpi4", architecture="", runtime="PREEMPT_RT", toolchain="gcc")


def test_runtime_target_digest_changes_with_field() -> None:
    assert _target().digest() != RuntimeTarget("jetson", "aarch64", "PREEMPT_RT", "gcc-13").digest()


# ── TimingEnvelope ────────────────────────────────────────────────────


def test_timing_envelope_accepts_schedulable_envelope() -> None:
    assert _envelope().schedulable is True


@pytest.mark.parametrize(
    "period,wcrt,deadline",
    [
        (1000.0, 600.0, 500.0),  # wcrt > deadline
        (400.0, 180.0, 500.0),  # deadline > period
    ],
)
def test_timing_envelope_rejects_unschedulable(period: float, wcrt: float, deadline: float) -> None:
    with pytest.raises(ValueError, match="schedulable"):
        TimingEnvelope(
            control_period_us=period, worst_case_response_us=wcrt, deadline_us=deadline, proof_firing_depth=4
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_timing_envelope_rejects_non_positive_times(bad: float) -> None:
    with pytest.raises(ValueError, match="positive, finite"):
        TimingEnvelope(control_period_us=bad, worst_case_response_us=180.0, deadline_us=500.0, proof_firing_depth=4)


def test_timing_envelope_requires_positive_proof_depth() -> None:
    with pytest.raises(ValueError, match="proof_firing_depth"):
        TimingEnvelope(control_period_us=1000.0, worst_case_response_us=180.0, deadline_us=500.0, proof_firing_depth=0)


# ── ControllerRuntimeBinding ──────────────────────────────────────────


def test_binding_digest_is_deterministic() -> None:
    net = _disarm_net()
    assert _binding(net).digest() == _binding(net).digest()


@pytest.mark.parametrize(
    "overrides",
    [
        {"controller_config": {"kp": 9.9}},
        {"snn_parameters": {"layers": 5}},
        {"solver_mode": "osqp"},
        {"controller_id": "other-ctl"},
    ],
)
def test_binding_digest_changes_with_each_field(overrides: dict[str, Any]) -> None:
    net = _disarm_net()
    assert _binding(net).digest() != _binding(net, **overrides).digest()


def test_binding_digest_changes_with_runtime_target() -> None:
    net = _disarm_net()
    other_target = RuntimeTarget("jetson", "aarch64", "PREEMPT_RT", "gcc-13")
    assert _binding(net).digest() != _binding(net, runtime_target=other_target).digest()


def test_binding_rejects_non_sha256_topology() -> None:
    with pytest.raises(ValueError, match="petri_topology_sha256"):
        _binding(_disarm_net(), petri_topology_sha256="not-a-digest")


def test_binding_rejects_empty_controller_config() -> None:
    with pytest.raises(ValueError, match="controller_config"):
        _binding(_disarm_net(), controller_config={})


# ── issuance ──────────────────────────────────────────────────────────


def test_issue_produces_a_holding_runtime_certificate() -> None:
    net = _disarm_net()
    cert, binding = _issued(net)
    assert cert["schema_version"] == RUNTIME_SAFETY_CERTIFICATE_SCHEMA_VERSION
    assert cert["holds"] is True
    assert cert["binding_sha256"] == binding.digest()
    assert cert["runtime_target_sha256"] == binding.runtime_target.digest()
    assert cert["timing_envelope_sha256"] == binding.timing_envelope.digest()
    assert "CTL:armed_bounded:AG" in cert["checked_specs"]
    assert "LTL:safe_bounded:G" in cert["checked_specs"]


def test_issue_fails_closed_on_topology_mismatch() -> None:
    net = _disarm_net()
    binding = _binding(net)
    other = StochasticPetriNet()
    other.add_place("a", initial_tokens=1.0)
    other.add_place("b", initial_tokens=0.0)
    other.add_transition("t", threshold=1.0)
    other.add_arc("a", "t", weight=1.0)
    other.add_arc("t", "b", weight=1.0)
    other.compile()
    with pytest.raises(ValueError, match="does not match the compiled net"):
        issue_runtime_safety_certificate(other, binding, formal_certificate=_make_formal(net))


def test_issue_fails_closed_on_non_holding_formal_certificate() -> None:
    net = _disarm_net()
    formal = _make_formal(net)
    formal["holds"] = False  # tamper: pretend the proof failed
    with pytest.raises(ValueError):
        issue_runtime_safety_certificate(net, _binding(net), formal_certificate=formal)


# ── validation ────────────────────────────────────────────────────────


def test_validate_accepts_issued_certificate() -> None:
    cert, _ = _issued(_disarm_net())
    assert validate_runtime_safety_certificate_payload(cert) == cert


def test_validate_rejects_wrong_schema() -> None:
    cert, _ = _issued(_disarm_net())
    cert["schema_version"] = "other"
    with pytest.raises(ValueError, match="schema_version"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_tampered_payload_digest() -> None:
    cert, _ = _issued(_disarm_net())
    cert["binding"]["controller_id"] = "swapped"  # mutate without re-stamping
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_formal_digest_mismatch() -> None:
    cert, _ = _issued(_disarm_net())
    cert["formal_certificate_sha256"] = "0" * 64
    cert["payload_sha256"] = _digest({k: v for k, v in cert.items() if k != "payload_sha256"})
    with pytest.raises(ValueError, match="formal_certificate_sha256"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_constants_tampering() -> None:
    cert, _ = _issued(_disarm_net())
    assert cert["scope"] == RUNTIME_SAFETY_CERTIFICATE_SCOPE
    assert cert["claim_boundary"] == RUNTIME_SAFETY_CERTIFICATE_CLAIM_BOUNDARY
    cert["scope"] = "broader than proven"
    with pytest.raises(ValueError, match="scope"):
        validate_runtime_safety_certificate_payload(cert)


# ── replay ────────────────────────────────────────────────────────────


def test_replay_passes_for_unchanged_net_and_proof() -> None:
    net = _disarm_net()
    cert, _ = _issued(net)
    replay = replay_runtime_safety_certificate(net, cert, reverify=lambda: _make_formal(net))
    assert replay.passed is True
    assert replay.detail == []


def test_replay_fails_on_topology_change() -> None:
    net = _disarm_net()
    cert, _ = _issued(net)
    moved = StochasticPetriNet()
    moved.add_place("armed", initial_tokens=1.0)
    moved.add_place("safe", initial_tokens=0.0)
    moved.add_transition("disarm", threshold=0.5)  # changed threshold
    moved.add_arc("armed", "disarm", weight=1.0)
    moved.add_arc("disarm", "safe", weight=1.0)
    moved.compile()
    replay = replay_runtime_safety_certificate(moved, cert, reverify=lambda: _make_formal(moved))
    assert replay.passed is False
    assert replay.topology_matches is False


def test_replay_fails_when_reverify_does_not_hold() -> None:
    net = _disarm_net()
    cert, _ = _issued(net)

    def failing_reverify() -> dict[str, Any]:
        # A genuinely failing proof: the "armed" place drains to 0, so a lower
        # bound of 0.5 is violated -> a valid failing formal certificate.
        with tempfile.TemporaryDirectory() as tmp:
            return generate_safety_certificate(
                net,
                max_depth=4,
                marking_bounds={"armed": (0.5, 1.0), "safe": (0.0, 1.0)},
                json_path=os.path.join(tmp, "c.json"),
                markdown_path=os.path.join(tmp, "c.md"),
            )

    replay = replay_runtime_safety_certificate(net, cert, reverify=failing_reverify)
    assert replay.passed is False
    assert replay.holds_matches is False


def test_replay_fails_when_reverify_covers_different_specs() -> None:
    net = _disarm_net()
    cert, _ = _issued(net)

    def reverify_fewer_specs() -> dict[str, Any]:
        # A holding proof that nonetheless covers fewer obligations than the
        # certificate: the replay must not accept a weaker re-proof.
        with tempfile.TemporaryDirectory() as tmp:
            return generate_safety_certificate(
                net,
                max_depth=4,
                marking_bounds={"armed": (0.0, 1.0), "safe": (0.0, 1.0)},
                json_path=os.path.join(tmp, "c.json"),
                markdown_path=os.path.join(tmp, "c.md"),
                temporal_specs=[EventuallyFires("disarm_fires", "disarm")],
            )

    replay = replay_runtime_safety_certificate(net, cert, reverify=reverify_fewer_specs)
    assert replay.passed is False
    assert replay.checked_specs_match is False


# ── admission ─────────────────────────────────────────────────────────


def _passing_replay() -> CertificateReplayResult:
    return CertificateReplayResult(True, True, True, True)


def test_admission_succeeds_for_matching_live_state() -> None:
    net = _disarm_net()
    cert, binding = _issued(net)
    admitted = assert_runtime_certificate_admissible(
        cert, live_binding=binding, live_runtime_target=_target(), replay=_passing_replay()
    )
    assert admitted["payload_sha256"] == cert["payload_sha256"]


def test_admission_fails_on_binding_mismatch() -> None:
    net = _disarm_net()
    cert, _ = _issued(net)
    drifted = _binding(net, controller_config={"kp": 9.9})
    with pytest.raises(ValueError, match="binding does not match"):
        assert_runtime_certificate_admissible(
            cert, live_binding=drifted, live_runtime_target=_target(), replay=_passing_replay()
        )


def test_admission_fails_on_runtime_target_mismatch() -> None:
    net = _disarm_net()
    cert, binding = _issued(net)
    other_target = RuntimeTarget("jetson", "aarch64", "PREEMPT_RT", "gcc-13")
    with pytest.raises(ValueError, match="runtime target does not match"):
        assert_runtime_certificate_admissible(
            cert, live_binding=binding, live_runtime_target=other_target, replay=_passing_replay()
        )


def test_admission_fails_when_replay_did_not_pass() -> None:
    net = _disarm_net()
    cert, binding = _issued(net)
    failed_replay = CertificateReplayResult(True, False, True, True, detail=["replayed proof does not hold"])
    with pytest.raises(ValueError, match="proof replay"):
        assert_runtime_certificate_admissible(
            cert, live_binding=binding, live_runtime_target=_target(), replay=failed_replay
        )


def test_replay_result_passed_requires_all_four_checks() -> None:
    assert CertificateReplayResult(True, True, True, True).passed is True
    assert CertificateReplayResult(True, True, True, False).passed is False
    assert CertificateReplayResult(False, True, True, True).passed is False


# ── error branches (exhaustive guard coverage) ────────────────────────


def test_timing_envelope_rejects_non_numeric_time() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        TimingEnvelope(
            control_period_us="fast",  # type: ignore[arg-type]
            worst_case_response_us=180.0,
            deadline_us=500.0,
            proof_firing_depth=4,
        )


def test_timing_envelope_rejects_non_integer_proof_depth() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        TimingEnvelope(
            control_period_us=1000.0,
            worst_case_response_us=180.0,
            deadline_us=500.0,
            proof_firing_depth=4.5,  # type: ignore[arg-type]
        )


def test_binding_rejects_non_mapping_snn_parameters() -> None:
    with pytest.raises(ValueError, match="snn_parameters"):
        _binding(_disarm_net(), snn_parameters=[1, 2, 3])


def test_binding_rejects_non_runtime_target() -> None:
    with pytest.raises(ValueError, match="runtime_target"):
        _binding(_disarm_net(), runtime_target="rpi4")  # type: ignore[arg-type]


def test_binding_rejects_non_timing_envelope() -> None:
    with pytest.raises(ValueError, match="timing_envelope"):
        _binding(_disarm_net(), timing_envelope="fast")  # type: ignore[arg-type]


def test_issue_rejects_valid_but_failing_formal_certificate() -> None:
    net = _disarm_net()
    # A genuinely failing proof (violated lower bound) is a *valid* failing
    # formal certificate; issuance must still reject it at its own holds gate.
    with tempfile.TemporaryDirectory() as tmp:
        failing = generate_safety_certificate(
            net,
            max_depth=4,
            marking_bounds={"armed": (0.5, 1.0), "safe": (0.0, 1.0)},
            json_path=os.path.join(tmp, "c.json"),
            markdown_path=os.path.join(tmp, "c.md"),
        )
    assert failing["holds"] is False
    with pytest.raises(ValueError, match="does not hold"):
        issue_runtime_safety_certificate(net, _binding(net), formal_certificate=failing)


def test_validate_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        validate_runtime_safety_certificate_payload("not-a-certificate")  # type: ignore[arg-type]


def test_validate_rejects_tampered_claim_boundary() -> None:
    cert, _ = _issued(_disarm_net())
    cert["claim_boundary"] = "broader than proven"
    with pytest.raises(ValueError, match="claim_boundary"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_non_object_binding() -> None:
    cert, _ = _issued(_disarm_net())
    cert["binding"] = "not-an-object"
    with pytest.raises(ValueError, match="binding must be an object"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_non_sha256_digest_field() -> None:
    cert, _ = _issued(_disarm_net())
    cert["runtime_target_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="runtime_target_sha256"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_non_holding_wrapper() -> None:
    cert, _ = _issued(_disarm_net())
    cert["holds"] = False  # caught before the payload-digest check
    with pytest.raises(ValueError, match="passing, holding proof"):
        validate_runtime_safety_certificate_payload(cert)


def test_validate_rejects_non_sha256_binding_topology() -> None:
    cert, _ = _issued(_disarm_net())
    cert["binding"]["petri_topology_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="petri_topology_sha256"):
        validate_runtime_safety_certificate_payload(cert)


def test_admission_fails_on_timing_envelope_mismatch() -> None:
    net = _disarm_net()
    cert, _ = _issued(net)
    # Same controller, but a different (still schedulable) timing envelope.
    drifted = _binding(
        net,
        timing_envelope=TimingEnvelope(
            control_period_us=2000.0, worst_case_response_us=200.0, deadline_us=600.0, proof_firing_depth=4
        ),
    )
    with pytest.raises(ValueError, match="timing envelope does not match"):
        assert_runtime_certificate_admissible(
            cert, live_binding=drifted, live_runtime_target=_target(), replay=_passing_replay()
        )
