# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal Safety Certificate Tests

"""Behavioural tests for the bounded formal safety certificate I/O surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scpn_control.scpn.formal_safety_certificate import (
    SafetyCertificateBundlePolicy,
    SafetyCertificatePolicy,
    admit_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_payload,
    build_safety_certificate_payload,
    generate_safety_certificate,
    validate_safety_certificate_bundle_artifact,
    validate_safety_certificate_bundle_payload,
    validate_safety_certificate_payload,
    write_safety_certificate,
    write_safety_certificate_bundle,
)
from scpn_control.scpn.formal_verification import (
    CTLFormula,
    EventuallyFires,
    FormalPetriNetVerifier,
    LTLFormula,
    verify_formal_contracts,
)
from scpn_control.scpn.structure import StochasticPetriNet


def _transfer_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _refresh_certificate_payload_digest(payload: dict[str, object]) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


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
