# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN Formal Verification Tests

"""Behavioural tests for SCPN formal reachability and safety proofs."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalViolation,
    FormalPetriNetVerifier,
    LTLFormula,
    NeverCoMarked,
    PlaceInvariant,
    SafetyCertificatePolicy,
    SafetyCertificateBundlePolicy,
    admit_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_payload,
    build_safety_certificate_payload,
    generate_safety_certificate,
    validate_safety_certificate_bundle_payload,
    validate_safety_certificate_payload,
    validate_safety_certificate_bundle_artifact,
    verify_formal_contracts,
    write_safety_certificate_bundle,
    write_safety_certificate,
)
from scpn_control.scpn.structure import StochasticPetriNet

from scpn_control.scpn.z3_model_checking import (  # noqa: E402
    Z3BoundedModelChecker,
    Z3FormalVerificationReport,
    Z3ModelCheckingReport,
    SYMBIOSYS_SYMBOLIC_CONTRACT_VERSION,
    build_blocked_z3_formal_report_payload,
    build_z3_formal_report_payload,
    load_z3_formal_report,
    validate_z3_formal_report_payload,
    verify_z3_formal_contracts,
    write_z3_formal_report,
)

requires_z3 = pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3-solver optional dependency absent")


def _transfer_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _reseal_z3_report_payload(payload: dict[str, object]) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def _weighted_transfer_net(*, source_tokens: float) -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=float(source_tokens))
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=0.25)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _latency_chain_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("staging", initial_tokens=0.0)
    net.add_place("ready", initial_tokens=0.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_transition("trigger", threshold=1.0)
    net.add_transition("stage", threshold=1.0)
    net.add_transition("response", threshold=1.0)
    net.add_arc("armed", "trigger", weight=1.0)
    net.add_arc("trigger", "staging", weight=1.0)
    net.add_arc("staging", "stage", weight=1.0)
    net.add_arc("stage", "ready", weight=1.0)
    net.add_arc("ready", "response", weight=1.0)
    net.add_arc("response", "safe", weight=1.0)
    net.compile()
    return net


def test_reachability_finds_named_marking_without_random_trials() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).analyze_reachability(max_depth=2)

    assert report.holds is True
    assert report.reachable_count == 2
    assert {state.marking["sink"] for state in report.reachable_states} == {0.0, 1.0}
    assert report.fired_transitions == {"move"}


def test_marking_bounds_report_counterexample_path() -> None:
    bounds = {"sink": (0.0, 0.5)}
    report = FormalPetriNetVerifier(_transfer_net()).prove_marking_bounds(bounds, max_depth=2)

    assert report.holds is False
    assert report.violations[0].property_name == "marking_bounds"
    assert report.violations[0].place == "sink"
    assert report.violations[0].path == ["move"]
    assert report.violations[0].marking["sink"] == 1.0


def test_transition_liveness_distinguishes_dead_transition() -> None:
    net = StochasticPetriNet()
    net.add_place("p0", initial_tokens=0.0)
    net.add_place("p1", initial_tokens=0.0)
    net.add_transition("needs_token", threshold=1.0)
    net.add_arc("p0", "needs_token", weight=1.0)
    net.add_arc("needs_token", "p1", weight=1.0)
    net.compile()

    report = FormalPetriNetVerifier(net).prove_transition_liveness(max_depth=3)

    assert report.holds is False
    assert report.dead_transitions == {"needs_token"}
    assert report.violations[0].property_name == "transition_liveness"


def test_place_invariant_proves_token_conservation_structurally() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).prove_place_invariants(
        [PlaceInvariant("total_token_conserved", {"source": 1.0, "sink": 1.0})],
        max_depth=3,
    )

    assert report.holds is True
    assert report.checked_specs == ["total_token_conserved"]


def test_place_invariant_reports_transition_that_breaks_conservation() -> None:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("duplicate", threshold=1.0)
    net.add_arc("source", "duplicate", weight=1.0)
    net.add_arc("duplicate", "sink", weight=2.0)
    net.compile()

    report = FormalPetriNetVerifier(net).prove_place_invariants(
        [PlaceInvariant("total_token_conserved", {"source": 1.0, "sink": 1.0})],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "total_token_conserved"
    assert report.violations[0].transition == "duplicate"
    assert "changes invariant" in report.violations[0].message


def test_temporal_specs_cover_always_eventually_and_never() -> None:
    specs = [
        AlwaysBounded("all_markings_safe", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
        EventuallyFires("move_eventually_fires", "move"),
        NeverCoMarked("exclusive_source_sink", "source", "sink", threshold=0.5),
        AlwaysEventuallyMarked("sink_recoverable", "sink", threshold=0.5),
        FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0),
    ]

    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(specs, max_depth=2)

    assert report.holds is True
    assert report.checked_specs == [
        "all_markings_safe",
        "move_eventually_fires",
        "exclusive_source_sink",
        "sink_recoverable",
        "move_marks_sink",
    ]


def test_temporal_response_checks_all_bounded_firing_paths() -> None:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_place("unsafe", initial_tokens=0.0)
    net.add_transition("actuate", threshold=1.0)
    net.add_transition("drop", threshold=1.0)
    net.add_arc("armed", "actuate", weight=1.0)
    net.add_arc("actuate", "safe", weight=1.0)
    net.add_arc("armed", "drop", weight=1.0)
    net.add_arc("drop", "unsafe", weight=1.0)
    net.compile()

    report = FormalPetriNetVerifier(net).verify_temporal_specs(
        [FireLeadsToMarking("drop_must_mark_safe", "drop", "safe", threshold=0.5, within=0)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].transition == "drop"
    assert report.violations[0].place == "safe"
    assert report.violations[0].path == ["drop"]


def test_temporal_recurrence_reports_nonrecoverable_marking() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(
        [AlwaysEventuallyMarked("source_recovers", "source", threshold=0.5)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "source_recovers"
    assert report.violations[0].path == ["move"]


def test_temporal_specs_return_actionable_counterexample() -> None:
    specs = [NeverCoMarked("sink_never_marked", "sink", "sink", threshold=0.5)]

    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(specs, max_depth=2)

    assert report.holds is False
    assert report.violations[0].property_name == "sink_never_marked"
    assert report.violations[0].path == ["move"]


def test_verify_formal_contracts_combines_safety_liveness_and_temporal_specs() -> None:
    report = verify_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )

    assert report.holds is True
    assert report.reachability.reachable_count == 2
    assert report.safety.holds is True
    assert report.liveness.holds is True
    assert report.temporal.holds is True


def test_ctl_specs_compile_to_bounded_petri_net_obligations() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net())
    report = verifier.verify_ctl_specs(
        [
            CTLFormula.ag_bounded("AG_safe_markings", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            CTLFormula.ef_fires("EF_move_fires", "move"),
            CTLFormula.ag_not_comarked("AG_exclusive_transfer", "source", "sink", threshold=0.5),
            CTLFormula.ag_ef_marked("AG_EF_sink_marked", "sink", threshold=0.5),
        ],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == [
        "CTL:AG_safe_markings:AG",
        "CTL:EF_move_fires:EF",
        "CTL:AG_exclusive_transfer:AG",
        "CTL:AG_EF_sink_marked:AG_EF",
    ]


def test_ctl_specs_return_actionable_counterexample() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).verify_ctl_specs(
        [CTLFormula.ag_bounded("AG_sink_upper_bound", {"sink": (0.0, 0.5)})],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "CTL:AG_sink_upper_bound:AG"
    assert report.violations[0].path == ["move"]
    assert report.violations[0].place == "sink"


def test_ltl_specs_compile_to_bounded_path_obligations() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net())
    report = verifier.verify_ltl_specs(
        [
            LTLFormula.globally_bounded("G_safe_markings", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            LTLFormula.eventually_fires("F_move_fires", "move"),
            LTLFormula.globally_fire_leads_to_marking("G_move_implies_sink", "move", "sink", threshold=0.5, within=0),
        ],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == [
        "LTL:G_safe_markings:G",
        "LTL:F_move_fires:F",
        "LTL:G_move_implies_sink:G_implies_F",
    ]


def test_ltl_specs_reject_unsupported_operator_domains() -> None:
    with pytest.raises(ValueError, match="unsupported LTL operator"):
        LTLFormula("bad_until", "U", "transition", {"transition": "move"})


def test_safety_certificate_payload_is_tamper_evident() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
    report = verify_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
        backend="explicit-state",
    )
    ctl_report = verifier.verify_ctl_specs([CTLFormula.ef_fires("EF_move_fires", "move")], max_depth=2)
    ltl_report = verifier.verify_ltl_specs([LTLFormula.eventually_fires("F_move_fires", "move")], max_depth=2)

    payload = build_safety_certificate_payload(
        report,
        ctl_report=ctl_report,
        ltl_report=ltl_report,
        artifact_sha256="a" * 64,
        issuer="release-safety-gate",
    )

    assert payload["schema_version"] == "scpn-control.safety-certificate.v1"
    assert payload["status"] == "pass"
    assert payload["holds"] is True
    assert payload["artifact_sha256"] == "a" * 64
    assert payload["checked_specs"] == [
        "marking_bounds",
        "transition_liveness",
        "move_eventually_fires",
        "CTL:EF_move_fires:EF",
        "LTL:F_move_fires:F",
    ]
    assert validate_safety_certificate_payload(payload) == payload

    tampered = dict(payload)
    tampered["issuer"] = "modified"
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_safety_certificate_payload(tampered)


def test_safety_certificate_writer_publishes_json_and_markdown(tmp_path: Path) -> None:
    verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
    report = verify_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
        backend="explicit-state",
    )
    ctl_report = verifier.verify_ctl_specs([CTLFormula.ef_fires("EF_move_fires", "move")], max_depth=2)
    ltl_report = verifier.verify_ltl_specs([LTLFormula.eventually_fires("F_move_fires", "move")], max_depth=2)
    json_path = tmp_path / "certificate.json"
    markdown_path = tmp_path / "certificate.md"

    payload = write_safety_certificate(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        ctl_report=ctl_report,
        ltl_report=ltl_report,
        artifact_sha256="b" * 64,
        issuer="release-safety-gate",
    )

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert persisted == payload
    assert validate_safety_certificate_payload(persisted) == payload
    assert "# SCPN Formal Safety Certificate" in markdown
    assert "scpn-control.safety-certificate.v1" in markdown
    assert payload["payload_sha256"] in markdown
    assert "CTL:EF_move_fires:EF" in markdown
    assert "bounded formal safety certificate" in markdown


def test_generate_safety_certificate_runs_full_workflow_and_binds_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "controller.scpnctl"
    artifact_bytes = b"compiled-controller-artifact"
    artifact_path.write_bytes(artifact_bytes)
    json_path = tmp_path / "certificate.json"
    markdown_path = tmp_path / "certificate.md"

    payload = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
        ctl_specs=[CTLFormula.ef_fires("EF_move_fires", "move")],
        ltl_specs=[LTLFormula.eventually_fires("F_move_fires", "move")],
        artifact_path=artifact_path,
        json_path=json_path,
        markdown_path=markdown_path,
        issuer="release-safety-gate",
        backend="explicit-state",
    )

    assert payload["status"] == "pass"
    assert payload["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert payload["checked_specs"] == [
        "marking_bounds",
        "transition_liveness",
        "move_eventually_fires",
        "CTL:EF_move_fires:EF",
        "LTL:F_move_fires:F",
    ]
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert payload["payload_sha256"] in markdown_path.read_text(encoding="utf-8")


def test_generate_safety_certificate_rejects_artifact_digest_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "controller.scpnctl"
    artifact_path.write_bytes(b"compiled-controller-artifact")

    with pytest.raises(ValueError, match="artifact_sha256"):
        generate_safety_certificate(
            _transfer_net(),
            max_depth=2,
            marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
            artifact_path=artifact_path,
            artifact_sha256="0" * 64,
            json_path=tmp_path / "certificate.json",
            markdown_path=tmp_path / "certificate.md",
            backend="explicit-state",
        )


def test_generate_safety_certificate_enforces_certification_policy(tmp_path: Path) -> None:
    artifact_path = tmp_path / "controller.scpnctl"
    artifact_path.write_bytes(b"compiled-controller-artifact")
    policy = SafetyCertificatePolicy(
        name="bounded-ctl-ltl-release-gate",
        min_depth=2,
        require_artifact_binding=True,
        require_ctl=True,
        require_ltl=True,
        required_checked_specs=("move_eventually_fires", "CTL:EF_move_fires:EF", "LTL:F_move_fires:F"),
    )

    payload = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
        ctl_specs=[CTLFormula.ef_fires("EF_move_fires", "move")],
        ltl_specs=[LTLFormula.eventually_fires("F_move_fires", "move")],
        artifact_path=artifact_path,
        json_path=tmp_path / "certificate.json",
        markdown_path=tmp_path / "certificate.md",
        policy=policy,
        backend="explicit-state",
    )

    assert payload["policy"]["name"] == "bounded-ctl-ltl-release-gate"
    assert payload["policy"]["require_ctl"] is True
    assert payload["policy"]["required_checked_specs"] == [
        "move_eventually_fires",
        "CTL:EF_move_fires:EF",
        "LTL:F_move_fires:F",
    ]
    assert validate_safety_certificate_payload(payload) == payload

    with pytest.raises(ValueError, match="requires LTL"):
        generate_safety_certificate(
            _transfer_net(),
            max_depth=2,
            marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
            temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
            ctl_specs=[CTLFormula.ef_fires("EF_move_fires", "move")],
            artifact_path=artifact_path,
            json_path=tmp_path / "missing-ltl.json",
            markdown_path=tmp_path / "missing-ltl.md",
            policy=policy,
            backend="explicit-state",
        )


def test_safety_certificate_policy_rejects_semantic_policy_tampering(tmp_path: Path) -> None:
    artifact_path = tmp_path / "controller.scpnctl"
    artifact_path.write_bytes(b"compiled-controller-artifact")
    payload = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        ctl_specs=[CTLFormula.ef_fires("EF_move_fires", "move")],
        ltl_specs=[LTLFormula.eventually_fires("F_move_fires", "move")],
        artifact_path=artifact_path,
        json_path=tmp_path / "certificate.json",
        markdown_path=tmp_path / "certificate.md",
        backend="explicit-state",
    )
    tampered = dict(payload)
    tampered["policy"] = {
        "name": "requires-temporal",
        "min_depth": 0,
        "require_artifact_binding": False,
        "require_ctl": False,
        "require_ltl": False,
        "required_checked_specs": ["move_eventually_fires"],
    }
    _refresh_certificate_payload_digest(tampered)

    with pytest.raises(ValueError, match="required checked spec"):
        validate_safety_certificate_payload(tampered)


def test_safety_certificate_bundle_admits_consistent_independent_certificates(tmp_path: Path) -> None:
    artifact_path = tmp_path / "controller.scpnctl"
    artifact_path.write_bytes(b"compiled-controller-artifact")
    policy = SafetyCertificatePolicy.certification_gate(
        name="bounded-ctl-ltl-release-gate",
        min_depth=2,
        required_checked_specs=("CTL:EF_move_fires:EF", "LTL:F_move_fires:F"),
    )
    cert_a = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        ctl_specs=[CTLFormula.ef_fires("EF_move_fires", "move")],
        ltl_specs=[LTLFormula.eventually_fires("F_move_fires", "move")],
        artifact_path=artifact_path,
        json_path=tmp_path / "certificate-a.json",
        markdown_path=tmp_path / "certificate-a.md",
        policy=policy,
        issuer="release-safety-gate-a",
        backend="explicit-state",
    )
    cert_b = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        ctl_specs=[CTLFormula.ef_fires("EF_move_fires", "move")],
        ltl_specs=[LTLFormula.eventually_fires("F_move_fires", "move")],
        artifact_path=artifact_path,
        json_path=tmp_path / "certificate-b.json",
        markdown_path=tmp_path / "certificate-b.md",
        policy=policy,
        issuer="release-safety-gate-b",
        backend="explicit-state",
    )
    bundle_policy = SafetyCertificateBundlePolicy(
        name="two-reviewer-certificate-bundle",
        min_certificates=2,
        required_policy_name="bounded-ctl-ltl-release-gate",
        require_same_artifact=True,
        require_same_backend=True,
    )

    bundle = write_safety_certificate_bundle(
        [cert_a, cert_b],
        json_path=tmp_path / "bundle.json",
        markdown_path=tmp_path / "bundle.md",
        policy=bundle_policy,
    )

    assert bundle["schema_version"] == "scpn-control.safety-certificate-bundle.v1"
    assert bundle["status"] == "pass"
    assert bundle["certificate_count"] == 2
    assert bundle["artifact_sha256"] == hashlib.sha256(b"compiled-controller-artifact").hexdigest()
    assert bundle["policy"]["name"] == "two-reviewer-certificate-bundle"
    assert len({entry["payload_sha256"] for entry in bundle["certificates"]}) == 2
    assert validate_safety_certificate_bundle_payload(bundle) == bundle
    persisted = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "bundle.md").read_text(encoding="utf-8")
    assert persisted == bundle
    assert "# SCPN Formal Safety Certificate Bundle" in markdown
    assert bundle["payload_sha256"] in markdown


def test_safety_certificate_bundle_rejects_mismatched_artifact_binding(tmp_path: Path) -> None:
    artifact_a = tmp_path / "controller-a.scpnctl"
    artifact_b = tmp_path / "controller-b.scpnctl"
    artifact_a.write_bytes(b"compiled-controller-artifact-a")
    artifact_b.write_bytes(b"compiled-controller-artifact-b")
    cert_a = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        artifact_path=artifact_a,
        json_path=tmp_path / "certificate-a.json",
        markdown_path=tmp_path / "certificate-a.md",
        backend="explicit-state",
    )
    cert_b = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        artifact_path=artifact_b,
        json_path=tmp_path / "certificate-b.json",
        markdown_path=tmp_path / "certificate-b.md",
        backend="explicit-state",
    )

    with pytest.raises(ValueError, match="artifact"):
        build_safety_certificate_bundle_payload(
            [cert_a, cert_b],
            policy=SafetyCertificateBundlePolicy(name="same-artifact", min_certificates=2, require_same_artifact=True),
        )


def test_safety_certificate_bundle_artifact_admits_hash_verified_relative_uri(tmp_path: Path) -> None:
    artifact_path = tmp_path / "controller.scpnctl"
    artifact_path.write_bytes(b"compiled-controller-artifact")
    certificate = generate_safety_certificate(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        artifact_path=artifact_path,
        json_path=tmp_path / "certificate.json",
        markdown_path=tmp_path / "certificate.md",
        backend="explicit-state",
    )
    bundle_path = tmp_path / "evidence" / "bundle.json"
    bundle = write_safety_certificate_bundle(
        [certificate],
        json_path=bundle_path,
        markdown_path=tmp_path / "evidence" / "bundle.md",
    )
    artifact = build_safety_certificate_bundle_artifact(
        bundle_uri="evidence/bundle.json",
        bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        producer="release-safety-gate",
        created_at="2026-05-31T00:00:00Z",
    )

    admitted = admit_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)

    assert admitted == bundle
    assert validate_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path) == artifact


def test_safety_certificate_bundle_artifact_rejects_traversal_and_digest_mismatch(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="relative"):
        build_safety_certificate_bundle_artifact(
            bundle_uri="../bundle.json",
            bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            producer="release-safety-gate",
            created_at="2026-05-31T00:00:00Z",
        )

    artifact = build_safety_certificate_bundle_artifact(
        bundle_uri="bundle.json",
        bundle_sha256="0" * 64,
        producer="release-safety-gate",
        created_at="2026-05-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="bundle_sha256"):
        admit_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)


def test_safety_certificate_bundle_artifact_rejects_manifest_tampering_and_future_time() -> None:
    artifact = build_safety_certificate_bundle_artifact(
        bundle_uri="bundle.json",
        bundle_sha256="1" * 64,
        producer="release-safety-gate",
        created_at="2026-05-31T00:00:00Z",
    )

    assert "artifact_sha256" in artifact

    tampered = dict(artifact)
    tampered["producer"] = "untrusted-gate"
    with pytest.raises(ValueError, match="artifact_sha256"):
        validate_safety_certificate_bundle_artifact(tampered)

    with pytest.raises(ValueError, match="created_at"):
        build_safety_certificate_bundle_artifact(
            bundle_uri="bundle.json",
            bundle_sha256="1" * 64,
            producer="release-safety-gate",
            created_at="2999-01-01T00:00:00Z",
        )


def test_safety_certificate_validator_rejects_semantic_section_tampering() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
    report = verify_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
        backend="explicit-state",
    )
    payload = build_safety_certificate_payload(
        report,
        ctl_report=verifier.verify_ctl_specs([CTLFormula.ef_fires("EF_move_fires", "move")], max_depth=2),
        ltl_report=verifier.verify_ltl_specs([LTLFormula.eventually_fires("F_move_fires", "move")], max_depth=2),
    )
    tampered = dict(payload)
    tampered["sections"] = {**payload["sections"], "ctl": {**payload["sections"]["ctl"], "holds": False}}
    _refresh_certificate_payload_digest(tampered)

    with pytest.raises(ValueError, match="section holds"):
        validate_safety_certificate_payload(tampered)


def test_formal_verifier_rejects_non_integer_depth_and_nonfinite_bounds() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net())

    try:
        verifier.analyze_reachability(max_depth=True)
    except ValueError as exc:
        assert "max_depth" in str(exc)
    else:
        raise AssertionError("boolean max_depth must be rejected")

    try:
        verifier.prove_marking_bounds({"sink": (0.0, float("inf"))}, max_depth=2)
    except ValueError as exc:
        assert "numeric values" in str(exc)
    else:
        raise AssertionError("non-finite marking bounds must be rejected")


def test_z3_checker_rejects_uncompiled_net_before_solver_use() -> None:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)

    with pytest.raises(RuntimeError, match="compiled"):
        Z3BoundedModelChecker(net)


def test_z3_checker_rejects_invalid_safety_domains_before_solver_use() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    with pytest.raises(ValueError, match="max_depth"):
        checker.prove_marking_bounds({"source": (0.0, 1.0)}, max_depth=True)
    with pytest.raises(ValueError, match="must not be empty"):
        checker.prove_marking_bounds({}, max_depth=1)
    with pytest.raises(ValueError, match="unknown place"):
        checker.prove_marking_bounds({"missing": (0.0, 1.0)}, max_depth=1)
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        checker.prove_marking_bounds({"source": (1.0, 0.0)}, max_depth=1)


def test_z3_checker_rejects_invalid_temporal_domains_before_solver_use() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    with pytest.raises(ValueError, match="unknown transition"):
        checker.verify_temporal_specs([EventuallyFires("missing_fires", "missing")], max_depth=1)
    with pytest.raises(ValueError, match="unknown place"):
        checker.verify_temporal_specs([AlwaysEventuallyMarked("missing_marked", "missing")], max_depth=1)
    with pytest.raises(ValueError, match="within"):
        checker.verify_temporal_specs([FireLeadsToMarking("bad_window", "move", "sink", within=-1)], max_depth=1)


@requires_z3
def test_z3_model_checker_proves_safe_marking_bounds() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).prove_marking_bounds(
        {"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        max_depth=2,
    )

    assert report.holds is True
    assert report.backend == "z3"
    assert report.solver_status == "unsat"
    assert report.violations == []


@requires_z3
def test_z3_model_checker_returns_marking_bound_counterexample() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).prove_marking_bounds({"sink": (0.0, 0.5)}, max_depth=2)

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].property_name == "marking_bounds"
    assert report.violations[0].place == "sink"
    assert report.violations[0].path == ["move"]
    assert report.violations[0].marking["sink"] == pytest.approx(1.0)


@requires_z3
def test_z3_model_checker_distinguishes_dead_transition_liveness() -> None:
    net = StochasticPetriNet()
    net.add_place("empty", initial_tokens=0.0)
    net.add_place("marked", initial_tokens=0.0)
    net.add_transition("needs_token", threshold=1.0)
    net.add_arc("empty", "needs_token", weight=1.0)
    net.add_arc("needs_token", "marked", weight=1.0)
    net.compile()

    report = Z3BoundedModelChecker(net).verify_temporal_specs(
        [EventuallyFires("dead_transition_detected", "needs_token")],
        max_depth=2,
    )

    assert report.holds is False
    assert report.solver_status == "unsat"
    assert report.violations[0].transition == "needs_token"
    assert "cannot fire" in report.violations[0].message


@requires_z3
def test_z3_temporal_specs_find_exclusivity_counterexample() -> None:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("a", initial_tokens=0.0)
    net.add_place("b", initial_tokens=0.0)
    net.add_transition("split", threshold=1.0)
    net.add_arc("armed", "split", weight=1.0)
    net.add_arc("split", "a", weight=1.0)
    net.add_arc("split", "b", weight=1.0)
    net.compile()

    report = Z3BoundedModelChecker(net).verify_temporal_specs(
        [NeverCoMarked("a_b_exclusive", "a", "b", threshold=0.5)],
        max_depth=1,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "a_b_exclusive"
    assert report.violations[0].path == ["split"]


@requires_z3
def test_z3_temporal_specs_prove_exclusivity_for_token_transfer() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [NeverCoMarked("source_sink_exclusive", "source", "sink", threshold=0.5)],
        max_depth=2,
    )

    assert report.holds is True
    assert report.solver_status == "unsat"
    assert report.checked_specs == ["source_sink_exclusive"]


@requires_z3
def test_z3_temporal_specs_prove_response_contract() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0)],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == ["move_marks_sink"]


@requires_z3
def test_z3_temporal_specs_report_response_deadline_counterexample() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [FireLeadsToMarking("move_must_restore_source", "move", "source", threshold=0.5, within=0)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].transition == "move"
    assert report.violations[0].place == "source"
    assert report.violations[0].path == ["move"]


@requires_z3
def test_z3_temporal_specs_report_nonrecoverable_marking_path() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [AlwaysEventuallyMarked("sink_is_universally_marked", "sink", threshold=0.5)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].place == "sink"
    assert report.violations[0].path == []


@requires_z3
def test_z3_temporal_specs_prove_all_supported_contracts_together() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [
            AlwaysBounded("bounded_transfer", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            EventuallyFires("move_fires", "move"),
            NeverCoMarked("exclusive_transfer", "source", "sink", threshold=0.5),
            AlwaysEventuallyMarked("source_initially_marked", "source", threshold=0.5),
            FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0),
        ],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == [
        "bounded_transfer",
        "move_fires",
        "exclusive_transfer",
        "source_initially_marked",
        "move_marks_sink",
    ]


@requires_z3
def test_z3_checker_enforces_parametric_weight_bounds_for_marking_safety() -> None:
    checker = Z3BoundedModelChecker(_weighted_transfer_net(source_tokens=1.0))

    bounded = checker.prove_marking_bounds(
        {"sink": (0.0, 4.0)},
        max_depth=1,
        weight_bounds={"move": (0.0, 1.0)},
    )
    assert bounded.holds is True
    assert bounded.solver_status == "unsat"

    unbounded = checker.prove_marking_bounds(
        {"sink": (0.0, 1.5)},
        max_depth=1,
        weight_bounds={"move": (2.0, 4.0)},
    )
    assert unbounded.holds is False
    assert unbounded.solver_status == "sat"
    assert unbounded.violations[0].property_name == "marking_bounds"


@requires_z3
def test_z3_checker_rejects_weight_bound_domain_mismatch_and_invalid_range() -> None:
    checker = Z3BoundedModelChecker(_weighted_transfer_net(source_tokens=1.0))

    with pytest.raises(ValueError, match="unknown transition"):
        checker.prove_marking_bounds({"sink": (0.0, 1.0)}, max_depth=1, weight_bounds={"unknown": (0.0, 1.0)})

    with pytest.raises(ValueError, match="lower exceeds"):
        checker.prove_marking_bounds(
            {"sink": (0.0, 1.0)},
            max_depth=1,
            weight_bounds={"move": (2.0, 1.0)},
        )


def test_z3_checker_builds_symbiyosys_contract_with_parametric_weight_and_latency_contracts() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    contract = checker.build_symbiyosys_contract(
        max_depth=3,
        weight_bounds={"trigger": (0.5, 2.0), "stage": (0.25, 0.75), "response": (0.1, 0.9)},
        trigger_transition="trigger",
        response_transition="response",
        max_latency_ns=30.0,
        tick_period_ns=10.0,
        no_stall_window_ns=20.0,
    )
    payload = json.loads(contract["metadata"])
    assert payload["schema_version"] == SYMBIOSYS_SYMBOLIC_CONTRACT_VERSION
    assert payload["max_depth"] == 3
    assert payload["trigger_transition"] == "trigger"
    assert payload["response_transition"] == "response"
    assert payload["weight_bounds"]["trigger"] == [0.5, 2.0]
    assert payload["weight_bounds"]["response"] == [0.1, 0.9]
    for step in range(3):
        for transition in ("trigger", "stage", "response"):
            assert f"(assume (>= weight_{step}_{transition}" in contract["smt2"]
            assert f"(assume (<= weight_{step}_{transition}" in contract["smt2"]
    assert "(assert (=> fire_0_trigger (or fire_1_response fire_2_response)))" in contract["smt2"]
    assert "(assert (not fire_2_trigger))" not in contract["smt2"]
    assert "(assert (not (and idle_0 idle_1)))" in contract["smt2"]
    assert "(set-logic QF_NRA)" in contract["smt2"]


def test_z3_checker_enforces_50ns_symbiyosys_contract_budget_limits() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())

    with pytest.raises(ValueError, match="must not exceed 50.0 ns contract budget"):
        checker.build_symbiyosys_contract(
            max_depth=10,
            max_latency_ns=60.0,
            no_stall_window_ns=20.0,
            trigger_transition="trigger",
            response_transition="response",
        )

    with pytest.raises(ValueError, match="must not exceed 50.0 ns contract budget"):
        checker.build_symbiyosys_contract(
            max_depth=10,
            max_latency_ns=20.0,
            no_stall_window_ns=60.0,
            trigger_transition="trigger",
            response_transition="response",
        )


def test_z3_checker_rejects_symbiyosys_contract_depth_contract_mismatch() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())

    with pytest.raises(ValueError, match="max_depth must be >= ceil"):
        checker.build_symbiyosys_contract(
            max_depth=1, max_latency_ns=30.0, tick_period_ns=10.0, no_stall_window_ns=20.0
        )

    with pytest.raises(ValueError, match="max_depth must be >= ceil"):
        checker.build_symbiyosys_contract(
            max_depth=1, max_latency_ns=10.0, tick_period_ns=10.0, no_stall_window_ns=30.0
        )


@requires_z3
def test_z3_checker_proves_trigger_to_response_latency_bound() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    report = checker.verify_trigger_response_latency(
        trigger_transition="trigger",
        response_transition="response",
        max_latency_ns=20.0,
        tick_period_ns=10.0,
        max_depth=3,
    )

    assert report.holds is True
    assert report.solver_status == "unsat"
    assert report.checked_specs == ["trigger_response_latency"]


@requires_z3
def test_z3_checker_reports_trigger_to_response_latency_violation() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    report = checker.verify_trigger_response_latency(
        trigger_transition="trigger",
        response_transition="response",
        max_latency_ns=10.0,
        tick_period_ns=10.0,
        max_depth=3,
    )

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].property_name == "trigger_response_latency"
    assert "can fire without" in report.violations[0].message


def test_z3_checker_enforces_trigger_latency_contract_budget_before_solver_invoke() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())

    with pytest.raises(ValueError, match="must not exceed 50.0 ns contract budget"):
        checker.verify_trigger_response_latency(
            trigger_transition="trigger",
            response_transition="response",
            max_latency_ns=60.0,
            tick_period_ns=10.0,
            max_depth=7,
        )


@requires_z3
def test_z3_checker_accepts_ctl_and_ltl_formula_facades() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    ctl = checker.verify_ctl_specs(
        [
            CTLFormula.ag_bounded("AG_safe_markings", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            CTLFormula.ef_fires("EF_move_fires", "move"),
        ],
        max_depth=2,
    )
    ltl = checker.verify_ltl_specs(
        [
            LTLFormula.globally_bounded("G_safe_markings", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            LTLFormula.eventually_fires("F_move_fires", "move"),
        ],
        max_depth=2,
    )

    assert ctl.holds is True
    assert ctl.checked_specs == ["CTL:AG_safe_markings:AG", "CTL:EF_move_fires:EF"]
    assert ltl.holds is True
    assert ltl.checked_specs == ["LTL:G_safe_markings:G", "LTL:F_move_fires:F"]


@requires_z3
def test_z3_formal_report_writer_publishes_json_and_markdown(tmp_path: Path) -> None:
    report = verify_z3_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )
    json_path = tmp_path / "formal_z3.json"
    markdown_path = tmp_path / "formal_z3.md"

    write_z3_formal_report(report, json_path=json_path, markdown_path=markdown_path)

    assert report.holds is True
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scpn-control.z3-formal-report.v1"
    assert payload["status"] == "pass"
    assert payload["checked_specs"] == ["marking_bounds", "move_eventually_fires"]
    assert validate_z3_formal_report_payload(payload) == payload
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# SCPN Z3 Formal Verification Report" in markdown
    assert "bounded SMT evidence" in markdown
    assert payload["payload_sha256"] in markdown


def test_load_z3_formal_report_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-z3-report.json"
    path.write_text('{"status": "pass", "status": "fail"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key: status"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown-z3-report-field.json"
    payload = build_blocked_z3_formal_report_payload("z3 unavailable in fixture")
    payload["foreign_attestation"] = "unrelated-proof-engine"
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_unknown_proof_section_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown-z3-section-field.json"
    report = Z3FormalVerificationReport(
        holds=True,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["marking_bounds"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["move_eventually_fires"]),
    )
    payload = build_z3_formal_report_payload(report)
    payload["safety"]["foreign_bound"] = "unsafe-padding"
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="safety.*unknown fields"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_unknown_violation_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown-z3-violation-field.json"
    violation = FormalViolation(
        property_name="unsafe_bound",
        message="sink exceeds admitted control envelope",
        marking={"sink": 1.0},
        path=["move"],
        place="sink",
    )
    report = Z3FormalVerificationReport(
        holds=False,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(False, "z3", 2, "sat", [violation], ["unsafe_bound"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )
    payload = build_z3_formal_report_payload(report)
    payload["safety"]["violations"][0]["foreign_counterexample"] = "padded-proof"
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="violation.*unknown fields"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_sat_section_that_claims_hold(tmp_path: Path) -> None:
    path = tmp_path / "sat-section-holds-z3-report.json"
    report = Z3FormalVerificationReport(
        holds=True,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["marking_bounds"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )
    payload = build_z3_formal_report_payload(report)
    payload["safety"]["solver_status"] = "sat"
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sat.*must not hold"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_unsat_section_with_violations(tmp_path: Path) -> None:
    path = tmp_path / "unsat-section-violation-z3-report.json"
    report = Z3FormalVerificationReport(
        holds=True,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["marking_bounds"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )
    payload = build_z3_formal_report_payload(report)
    payload["safety"]["violations"] = [
        {
            "marking": {"sink": 1.0},
            "message": "unsat section cannot carry counterexamples",
            "path": ["move"],
            "place": "sink",
            "property_name": "unsafe_bound",
            "transition": None,
        }
    ]
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsat.*must not carry violations"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_blocked_report_with_proof_depth(tmp_path: Path) -> None:
    path = tmp_path / "blocked-proof-depth-z3-report.json"
    payload = build_blocked_z3_formal_report_payload("z3 unavailable in fixture")
    payload["max_depth"] = 2
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="blocked.*max_depth"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_blocked_report_with_proof_specs(tmp_path: Path) -> None:
    path = tmp_path / "blocked-proof-specs-z3-report.json"
    payload = build_blocked_z3_formal_report_payload("z3 unavailable in fixture")
    payload["checked_specs"] = ["marking_bounds"]
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="blocked.*checked_specs"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_blocked_report_with_live_solver_label(tmp_path: Path) -> None:
    path = tmp_path / "blocked-live-solver-z3-report.json"
    payload = build_blocked_z3_formal_report_payload("z3 unavailable in fixture")
    payload["solver"] = "z3-solver 4.16.0"
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="blocked.*solver"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_duplicate_section_checked_specs(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-section-spec-z3-report.json"
    report = Z3FormalVerificationReport(
        holds=True,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["marking_bounds"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )
    payload = build_z3_formal_report_payload(report)
    payload["temporal"]["checked_specs"] = ["response_ok", "response_ok"]
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="temporal.*checked_specs.*unique"):
        load_z3_formal_report(path)


def test_load_z3_formal_report_rejects_empty_section_checked_spec(tmp_path: Path) -> None:
    path = tmp_path / "empty-section-spec-z3-report.json"
    report = Z3FormalVerificationReport(
        holds=True,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["marking_bounds"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )
    payload = build_z3_formal_report_payload(report)
    payload["temporal"]["checked_specs"] = ["response_ok", ""]
    payload["checked_specs"] = ["marking_bounds", "response_ok", ""]
    _reseal_z3_report_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checked_specs.*non-empty"):
        load_z3_formal_report(path)


def test_z3_formal_payload_records_fail_closed_counterexample_evidence() -> None:
    violation = FormalViolation(
        property_name="unsafe_bound",
        message="sink exceeds admitted control envelope",
        marking={"sink": 1.0},
        path=["move"],
        place="sink",
    )
    report = Z3FormalVerificationReport(
        holds=False,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(False, "z3", 2, "sat", [violation], ["unsafe_bound"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )

    payload = build_z3_formal_report_payload(report)

    assert payload["status"] == "fail"
    assert payload["holds"] is False
    assert payload["checked_specs"] == ["marking_bounds", "unsafe_bound", "response_ok"]
    assert payload["safety"]["violations"][0]["path"] == ["move"]
    assert validate_z3_formal_report_payload(payload) == payload


@requires_z3
def test_z3_formal_report_validator_rejects_tampered_checked_specs(tmp_path: Path) -> None:
    report = verify_z3_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )
    json_path = tmp_path / "formal_z3.json"
    markdown_path = tmp_path / "formal_z3.md"
    write_z3_formal_report(report, json_path=json_path, markdown_path=markdown_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["checked_specs"] = ["marking_bounds"]

    with pytest.raises(ValueError, match="payload_sha256"):
        validate_z3_formal_report_payload(payload)


def test_z3_formal_report_validator_rejects_inconsistent_safety_case_payloads() -> None:
    valid = build_blocked_z3_formal_report_payload("z3-solver unavailable")

    tampered = dict(valid)
    tampered["status"] = "pass"
    tampered["holds"] = False
    tampered["safety"] = {}
    tampered["temporal"] = {}
    tampered.pop("reason")
    tampered["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_z3_formal_report_payload(tampered)

    blocked_with_sections = dict(valid)
    blocked_with_sections["safety"] = {"holds": False}
    blocked_with_sections["payload_sha256"] = build_blocked_z3_formal_report_payload("z3-solver unavailable")[
        "payload_sha256"
    ]
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_z3_formal_report_payload(blocked_with_sections)


def _refresh_z3_payload_digest(payload: dict[str, object]) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def _refresh_certificate_payload_digest(payload: dict[str, object]) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def test_z3_formal_report_validator_rejects_semantically_invalid_blocked_reports() -> None:
    payload = build_blocked_z3_formal_report_payload("z3-solver unavailable")

    holds_payload = _refresh_z3_payload_digest({**payload, "holds": True})
    with pytest.raises(ValueError, match="must not hold"):
        validate_z3_formal_report_payload(holds_payload)

    missing_reason = dict(payload)
    missing_reason.pop("reason")
    _refresh_z3_payload_digest(missing_reason)
    with pytest.raises(ValueError, match="include a reason"):
        validate_z3_formal_report_payload(missing_reason)

    section_payload = _refresh_z3_payload_digest({**payload, "safety": {"holds": False}})
    with pytest.raises(ValueError, match="must not carry proof sections"):
        validate_z3_formal_report_payload(section_payload)


def test_z3_formal_report_validator_rejects_semantically_invalid_pass_fail_reports() -> None:
    valid = build_z3_formal_report_payload(
        Z3FormalVerificationReport(
            holds=True,
            backend="z3",
            max_depth=1,
            safety=Z3ModelCheckingReport(True, "z3", 1, "unsat", [], ["safety_ok"]),
            temporal=Z3ModelCheckingReport(True, "z3", 1, "unsat", [], ["temporal_ok"]),
        )
    )

    pass_without_hold = _refresh_z3_payload_digest({**valid, "holds": False})
    with pytest.raises(ValueError, match="passing"):
        validate_z3_formal_report_payload(pass_without_hold)

    fail_with_hold = _refresh_z3_payload_digest({**valid, "status": "fail", "holds": True})
    with pytest.raises(ValueError, match="failed"):
        validate_z3_formal_report_payload(fail_with_hold)

    depth_mismatch = dict(valid)
    depth_mismatch["safety"] = {**valid["safety"], "max_depth": 2}
    _refresh_z3_payload_digest(depth_mismatch)
    with pytest.raises(ValueError, match="depth"):
        validate_z3_formal_report_payload(depth_mismatch)

    bad_solver_status = dict(valid)
    bad_solver_status["temporal"] = {**valid["temporal"], "solver_status": "unchecked"}
    _refresh_z3_payload_digest(bad_solver_status)
    with pytest.raises(ValueError, match="solver_status"):
        validate_z3_formal_report_payload(bad_solver_status)

    checked_spec_mismatch = _refresh_z3_payload_digest({**valid, "checked_specs": ["marking_bounds"]})
    with pytest.raises(ValueError, match="checked_specs"):
        validate_z3_formal_report_payload(checked_spec_mismatch)


def test_blocked_z3_formal_report_is_schema_versioned_and_fail_closed() -> None:
    payload = build_blocked_z3_formal_report_payload("z3-solver unavailable")

    assert payload["schema_version"] == "scpn-control.z3-formal-report.v1"
    assert payload["status"] == "blocked"
    assert payload["holds"] is False
    assert payload["safety"] is None
    assert payload["temporal"] is None
    assert validate_z3_formal_report_payload(payload) == payload


@requires_z3
def test_z3_checker_rejects_unknown_domains_and_bad_bound_order() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    with pytest.raises(ValueError, match="unknown place"):
        checker.prove_marking_bounds({"missing": (0.0, 1.0)}, max_depth=1)

    with pytest.raises(ValueError, match="lower bound"):
        checker.prove_marking_bounds({"source": (1.0, 0.0)}, max_depth=1)

    with pytest.raises(ValueError, match="unknown transition"):
        checker.verify_temporal_specs([EventuallyFires("missing_transition_fires", "missing_transition")], max_depth=1)

    with pytest.raises(ValueError, match="unknown place"):
        checker.verify_temporal_specs(
            [AlwaysEventuallyMarked("missing_place_recurs", "missing_place", threshold=0.5)], max_depth=1
        )


@requires_z3
def test_z3_fire_leads_to_marking_rejects_invalid_deadline_domain() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    with pytest.raises(ValueError, match="within"):
        checker.verify_temporal_specs([FireLeadsToMarking("bad_bool_window", "move", "sink", within=True)], max_depth=2)

    with pytest.raises(ValueError, match="within"):
        checker.verify_temporal_specs(
            [FireLeadsToMarking("bad_negative_window", "move", "sink", within=-1)], max_depth=2
        )


@requires_z3
def test_z3_combined_report_records_failed_safety_before_temporal_success() -> None:
    report = verify_z3_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"sink": (0.0, 0.5)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )
    payload = build_z3_formal_report_payload(report)

    assert report.holds is False
    assert payload["status"] == "fail"
    assert payload["safety"]["holds"] is False
    assert payload["temporal"]["holds"] is True
    assert "marking_bounds" in payload["checked_specs"]
    assert "move_eventually_fires" in payload["checked_specs"]


def test_z3_report_validator_rejects_blocked_payload_semantic_tampering() -> None:
    payload = build_blocked_z3_formal_report_payload("z3 unavailable in deployment image")

    blocked_with_sections = dict(payload)
    blocked_with_sections["safety"] = {"holds": False}
    _refresh_z3_payload_digest(blocked_with_sections)
    with pytest.raises(ValueError, match="must not carry proof sections"):
        validate_z3_formal_report_payload(blocked_with_sections)

    blocked_without_reason = dict(payload)
    blocked_without_reason["reason"] = ""
    _refresh_z3_payload_digest(blocked_without_reason)
    with pytest.raises(ValueError, match="include a reason"):
        validate_z3_formal_report_payload(blocked_without_reason)

    blocked_claiming_hold = dict(payload)
    blocked_claiming_hold["holds"] = True
    _refresh_z3_payload_digest(blocked_claiming_hold)
    with pytest.raises(ValueError, match="must not hold"):
        validate_z3_formal_report_payload(blocked_claiming_hold)


def test_z3_report_validator_rejects_report_level_type_tampering() -> None:
    valid = build_z3_formal_report_payload(
        Z3FormalVerificationReport(
            holds=True,
            backend="z3",
            max_depth=1,
            safety=Z3ModelCheckingReport(True, "z3", 1, "unsat", [], ["safety_ok"]),
            temporal=Z3ModelCheckingReport(True, "z3", 1, "unsat", [], ["temporal_ok"]),
        )
    )

    for key, value, match in (
        ("backend", "not-z3", "backend"),
        ("status", "unchecked", "status"),
        ("solver", "", "solver"),
        ("holds", "true", "holds"),
        ("max_depth", True, "max_depth"),
        ("checked_specs", [], "checked_specs"),
        ("scope", "unbounded claim", "scope"),
        ("claim_boundary", "certified", "claim_boundary"),
    ):
        tampered = dict(valid)
        tampered[key] = value
        _refresh_z3_payload_digest(tampered)
        with pytest.raises(ValueError, match=match):
            validate_z3_formal_report_payload(tampered)
