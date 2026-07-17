# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal Safety Certificate Tests

"""Tests for the bounded formal safety certificate I/O surface.

Covers the payload, bundle, and bundle-artifact build/write/generate/validate/
admit behaviour together with the policy dataclass guards, the schema-validator
fail-closed branches, and the private helper guards. Schema validators run
before the trailing digest check, so a single mutated field of an otherwise
valid payload reaches exactly its intended guard without re-sealing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import scpn_control.scpn.formal_safety_certificate as fsc
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


def _valid_certificate(*, issuer: str = "scpn-control", passing: bool = True) -> dict[str, Any]:
    net = _transfer_net()
    bounds = {"sink": (0.0, 1.0)} if passing else {"sink": (0.0, 0.0)}
    report = verify_formal_contracts(net, max_depth=2, marking_bounds=bounds, backend="explicit-state")
    verifier = FormalPetriNetVerifier(net, backend="explicit-state")
    ctl = verifier.verify_ctl_specs([CTLFormula.ag_bounded("safe", {"sink": (0.0, 1.0)})], max_depth=2)
    ltl = verifier.verify_ltl_specs([LTLFormula.globally_bounded("g", {"sink": (0.0, 1.0)})], max_depth=2)
    return build_safety_certificate_payload(
        report, ctl_report=ctl, ltl_report=ltl, artifact_sha256="a" * 64, issuer=issuer
    )


def _valid_bundle() -> dict[str, Any]:
    return build_safety_certificate_bundle_payload(
        [_valid_certificate(issuer="scpn-control"), _valid_certificate(issuer="scpn-control-2")]
    )


def _valid_artifact() -> dict[str, Any]:
    return build_safety_certificate_bundle_artifact(
        bundle_uri="bundle.json",
        bundle_sha256="b" * 64,
        producer="scpn-control",
        created_at="2026-01-01T00:00:00Z",
    )


def _mutated(base: dict[str, Any], mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    clone = copy.deepcopy(base)
    mutator(clone)
    return clone


class TestPolicyDataclassGuards:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": ""}, "name must be a non-empty string"),
            ({"name": "p", "min_depth": -1}, "min_depth must be an integer >= 0"),
            ({"name": "p", "require_artifact_binding": "no"}, "require_artifact_binding must be boolean"),
            ({"name": "p", "require_ctl": "no"}, "require_ctl must be boolean"),
            ({"name": "p", "require_ltl": "no"}, "require_ltl must be boolean"),
            ({"name": "p", "required_checked_specs": ["x"]}, "required_checked_specs must be a tuple"),
            ({"name": "p", "required_checked_specs": ("",)}, "must contain non-empty strings"),
            ({"name": "p", "required_checked_specs": ("a", "a")}, "must be unique"),
        ],
    )
    def test_rejects_invalid_policy(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SafetyCertificatePolicy(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": ""}, "name must be a non-empty string"),
            ({"name": "b", "min_certificates": 0}, "min_certificates must be an integer >= 1"),
            ({"name": "b", "required_policy_name": ""}, "required_policy_name must be non-empty or None"),
            ({"name": "b", "require_same_artifact": "no"}, "require_same_artifact must be boolean"),
            ({"name": "b", "require_same_backend": "no"}, "require_same_backend must be boolean"),
        ],
    )
    def test_rejects_invalid_bundle_policy(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SafetyCertificateBundlePolicy(**kwargs)


class TestBuildSafetyCertificateGuards:
    def test_rejects_invalid_artifact_digest(self):
        net = _transfer_net()
        report = verify_formal_contracts(
            net, max_depth=2, marking_bounds={"sink": (0.0, 1.0)}, backend="explicit-state"
        )
        with pytest.raises(ValueError, match="artifact_sha256 must be a SHA-256"):
            build_safety_certificate_payload(report, artifact_sha256="bad")

    def test_rejects_empty_issuer(self):
        net = _transfer_net()
        report = verify_formal_contracts(
            net, max_depth=2, marking_bounds={"sink": (0.0, 1.0)}, backend="explicit-state"
        )
        with pytest.raises(ValueError, match="issuer must be a non-empty string"):
            build_safety_certificate_payload(report, issuer="")


def _set_section(field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutator(payload: dict[str, Any]) -> None:
        payload["sections"]["safety"][field] = value

    return mutator


class TestValidateSafetyCertificate:
    @pytest.mark.parametrize(
        ("mutator", "match"),
        [
            (lambda p: p.__setitem__("schema_version", "v0"), "schema_version is unsupported"),
            (lambda p: p.__setitem__("status", "maybe"), "status must be pass or fail"),
            (lambda p: p.__setitem__("holds", "yes"), "holds must be a boolean"),
            (lambda p: p.__setitem__("holds", False), "passing safety certificate must hold"),
            (
                lambda p: (p.__setitem__("status", "fail"), p.__setitem__("holds", True)),
                "failed safety certificate must not hold",
            ),
            (lambda p: p.__setitem__("backend", "smt"), "backend is unsupported"),
            (lambda p: p.__setitem__("max_depth", "two"), "max_depth must be an integer"),
            (lambda p: p.__setitem__("max_depth", -1), "max_depth must be non-negative"),
            (lambda p: p.__setitem__("issuer", ""), "issuer must be a non-empty string"),
            (lambda p: p.__setitem__("artifact_sha256", "zz"), "artifact_sha256 must be a SHA-256"),
            (lambda p: p.__setitem__("checked_specs", []), "checked_specs must be a non-empty list"),
            (lambda p: p.__setitem__("checked_specs", [""]), "checked_specs must contain non-empty strings"),
            (lambda p: p.__setitem__("checked_specs", ["dup", "dup"]), "checked_specs must be unique"),
            (lambda p: p.__setitem__("backend", "z3"), "safety certificate backend is unsupported"),
            (lambda p: p.__setitem__("scope", "other"), "scope is unsupported"),
            (lambda p: p.__setitem__("claim_boundary", "other"), "claim_boundary is unsupported"),
            (lambda p: p.__setitem__("sections", "flat"), "sections must be an object"),
            (lambda p: p["sections"].__setitem__("reachability", None), "reachability section must be an object"),
            (lambda p: p["sections"].__setitem__("safety", "flat"), "safety section must be an object"),
            (_set_section("holds", "yes"), "section holds must be a boolean"),
            (_set_section("backend", "z3"), "section backend must match certificate backend"),
            (_set_section("max_depth", 99), "section depth must match certificate depth"),
            (
                lambda p: p["sections"]["temporal"].__setitem__("checked_specs", "x"),
                "section checked_specs must be a list",
            ),
            (
                lambda p: p["sections"]["temporal"].__setitem__("checked_specs", [""]),
                "section checked_specs must contain non-empty strings",
            ),
            (
                lambda p: p.__setitem__("checked_specs", [*p["checked_specs"], "ghost_spec"]),
                "checked_specs must match certificate sections",
            ),
            (lambda p: p.__setitem__("payload_sha256", "short"), "payload_sha256 must be a SHA-256"),
        ],
    )
    def test_rejects_mutated_certificate(self, mutator, match):
        with pytest.raises(ValueError, match=match):
            validate_safety_certificate_payload(_mutated(_valid_certificate(), mutator))


# ── safety certificate bundle validation ──────────────────────────────


class TestValidateSafetyCertificateBundle:
    @pytest.mark.parametrize(
        ("mutator", "match"),
        [
            (lambda p: p.__setitem__("schema_version", "v0"), "bundle schema_version is unsupported"),
            (lambda p: p.__setitem__("status", "maybe"), "bundle status must be pass or fail"),
            (lambda p: p.__setitem__("holds", "yes"), "bundle holds must be a boolean"),
            (lambda p: p.__setitem__("certificate_count", "two"), "certificate_count must be an integer"),
            (lambda p: p.__setitem__("certificate_count", 0), "certificate_count must be positive"),
            (lambda p: p.__setitem__("scope", "other"), "bundle scope is unsupported"),
            (lambda p: p.__setitem__("claim_boundary", "other"), "bundle claim_boundary is unsupported"),
            (lambda p: p.__setitem__("backend", "smt"), "bundle backend is unsupported"),
            (lambda p: p.__setitem__("artifact_sha256", "zz"), "bundle artifact_sha256 must be a SHA-256"),
            (lambda p: p.__setitem__("certificate_count", 99), "certificates must match certificate_count"),
            (lambda p: p.__setitem__("holds", False), "bundle holds must match certificate holds"),
            (lambda p: p.__setitem__("artifact_sha256", "f" * 64), "must match certificate artifact bindings"),
            (lambda p: p.__setitem__("backend", "mixed"), "backend must match certificate backends"),
            (lambda p: p.__setitem__("payload_sha256", "short"), "payload_sha256 must be a SHA-256"),
            (lambda p: p.__setitem__("payload_sha256", "c" * 64), "payload_sha256 does not match payload"),
        ],
    )
    def test_rejects_mutated_bundle(self, mutator, match):
        with pytest.raises(ValueError, match=match):
            validate_safety_certificate_bundle_payload(_mutated(_valid_bundle(), mutator))

    def test_rejects_pass_status_on_failing_bundle(self):
        bundle = build_safety_certificate_bundle_payload([_valid_certificate(passing=False)])

        def mutator(payload: dict[str, Any]) -> None:
            payload["status"] = "pass"

        with pytest.raises(ValueError, match="passing safety certificate bundle must hold"):
            validate_safety_certificate_bundle_payload(_mutated(bundle, mutator))

    def test_rejects_fail_status_on_passing_bundle(self):
        def mutator(payload: dict[str, Any]) -> None:
            payload["status"] = "fail"

        with pytest.raises(ValueError, match="failed safety certificate bundle must not hold"):
            validate_safety_certificate_bundle_payload(_mutated(_valid_bundle(), mutator))


# ── bundle artifact build and validation ──────────────────────────────


class TestBundleArtifact:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"bundle_uri": "b.json", "bundle_sha256": "bad", "producer": "x", "created_at": "2026-01-01T00:00:00Z"},
                "bundle_sha256 must be a SHA-256",
            ),
            (
                {
                    "bundle_uri": "b.json",
                    "bundle_sha256": "b" * 64,
                    "producer": "",
                    "created_at": "2026-01-01T00:00:00Z",
                },
                "producer must be a non-empty string",
            ),
        ],
    )
    def test_build_rejects_invalid_fields(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            build_safety_certificate_bundle_artifact(**kwargs)

    @pytest.mark.parametrize(
        ("mutator", "match"),
        [
            (lambda a: a.__setitem__("schema_version", "v0"), "artifact schema_version is unsupported"),
            (lambda a: a.__setitem__("kind", "other"), "artifact kind is unsupported"),
            (lambda a: a.__setitem__("bundle_uri", "/abs"), "must be a safe relative path"),
            (lambda a: a.__setitem__("bundle_sha256", "bad"), "bundle_sha256 must be a SHA-256"),
            (lambda a: a.__setitem__("producer", ""), "producer must be a non-empty string"),
            (lambda a: a.__setitem__("scope", "other"), "artifact scope is unsupported"),
            (lambda a: a.__setitem__("claim_boundary", "other"), "artifact claim_boundary is unsupported"),
            (lambda a: a.__setitem__("artifact_sha256", "bad"), "artifact_sha256 must be a SHA-256"),
            (lambda a: a.__setitem__("artifact_sha256", "d" * 64), "does not match artifact metadata"),
        ],
    )
    def test_validate_rejects_mutated_artifact(self, mutator, match):
        with pytest.raises(ValueError, match=match):
            validate_safety_certificate_bundle_artifact(_mutated(_valid_artifact(), mutator))

    @pytest.mark.parametrize(
        ("created_at", "match"),
        [
            ("", "must be a non-empty UTC timestamp"),
            ("not-a-date", "must be an ISO-8601 UTC timestamp"),
            ("2026-01-01T00:00:00", "must include a UTC offset"),
        ],
    )
    def test_build_rejects_invalid_timestamp(self, created_at, match):
        with pytest.raises(ValueError, match=match):
            build_safety_certificate_bundle_artifact(
                bundle_uri="b.json", bundle_sha256="b" * 64, producer="x", created_at=created_at
            )

    def test_validate_rejects_non_object_artifact(self):
        with pytest.raises(ValueError, match="artifact must be an object"):
            validate_safety_certificate_bundle_artifact(["not", "an", "object"])  # type: ignore[arg-type]

    def test_validate_rejects_missing_bundle_file(self, tmp_path: Path):
        artifact = _valid_artifact()
        with pytest.raises(ValueError, match="target must be a file"):
            validate_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)

    def test_validate_rejects_non_object_bundle_file(self, tmp_path: Path):
        bundle = _valid_bundle()
        bundle_bytes = json.dumps(bundle, sort_keys=True).encode("utf-8")
        digest = fsc.hashlib.sha256(bundle_bytes).hexdigest()
        artifact = build_safety_certificate_bundle_artifact(
            bundle_uri="bundle.json", bundle_sha256=digest, producer="x", created_at="2026-01-01T00:00:00Z"
        )
        (tmp_path / "bundle.json").write_bytes(json.dumps([1, 2, 3]).encode("utf-8"))
        with pytest.raises(ValueError, match="bundle_sha256 does not match bundle bytes"):
            validate_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)

    def test_validate_accepts_bound_bundle_file(self, tmp_path: Path):
        bundle = _valid_bundle()
        bundle_bytes = (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (tmp_path / "bundle.json").write_bytes(bundle_bytes)
        digest = fsc.hashlib.sha256(bundle_bytes).hexdigest()
        artifact = build_safety_certificate_bundle_artifact(
            bundle_uri="bundle.json", bundle_sha256=digest, producer="x", created_at="2026-01-01T00:00:00Z"
        )
        validated = validate_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)
        assert validated["bundle_sha256"] == digest

    def test_validate_rejects_non_object_bundle_after_digest(self, tmp_path: Path):
        raw_bytes = json.dumps([1, 2, 3]).encode("utf-8")
        (tmp_path / "bundle.json").write_bytes(raw_bytes)
        digest = fsc.hashlib.sha256(raw_bytes).hexdigest()
        artifact = build_safety_certificate_bundle_artifact(
            bundle_uri="bundle.json", bundle_sha256=digest, producer="x", created_at="2026-01-01T00:00:00Z"
        )
        with pytest.raises(ValueError, match="target must contain a bundle object"):
            validate_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)

    def test_admit_rejects_non_object_target(self, tmp_path: Path, monkeypatch):
        artifact = _valid_artifact()
        monkeypatch.setattr(
            fsc, "validate_safety_certificate_bundle_artifact", lambda art, **kw: {"bundle_uri": "bundle.json"}
        )
        (tmp_path / "bundle.json").write_bytes(json.dumps([1, 2, 3]).encode("utf-8"))
        with pytest.raises(ValueError, match="target must contain a bundle object"):
            fsc.admit_safety_certificate_bundle_artifact(artifact, artifact_root=tmp_path)


# ── helper functions and policy enforcement ───────────────────────────


class TestHelperBranches:
    def test_checked_specs_from_sections_skips_non_dict(self):
        assert fsc._certificate_checked_specs_from_sections({"temporal": "x"}) == [
            "marking_bounds",
            "transition_liveness",
        ]

    def test_policy_from_payload_rejects_non_object(self):
        with pytest.raises(ValueError, match="policy must be an object or null"):
            fsc._policy_from_payload("flat")

    def test_policy_from_payload_rejects_non_list_specs(self):
        with pytest.raises(ValueError, match="required_checked_specs must be a list"):
            fsc._policy_from_payload({"name": "p", "required_checked_specs": "x"})

    def test_bundle_policy_from_payload_rejects_non_object(self):
        with pytest.raises(ValueError, match="bundle policy must be an object or null"):
            fsc._bundle_policy_from_payload("flat")

    def test_common_certificate_value_empty(self):
        assert fsc._common_certificate_value([], "backend") is None

    def test_jsonable_converts_tuple(self):
        assert fsc._jsonable((1, 2, 3)) == [1, 2, 3]

    def test_resolve_artifact_sha256_rejects_invalid_digest(self):
        with pytest.raises(ValueError, match="artifact_sha256 must be a SHA-256"):
            fsc._resolve_artifact_sha256(None, "bad")

    def test_resolve_artifact_sha256_passes_through_valid_digest(self):
        assert fsc._resolve_artifact_sha256(None, "a" * 64) == "a" * 64

    def test_safe_relative_uri_rejects_empty_value(self):
        with pytest.raises(ValueError, match="must be a non-empty relative path"):
            fsc._safe_relative_uri("bundle_uri", "")

    def test_file_sha256_rejects_missing_file(self, tmp_path: Path):
        with pytest.raises(ValueError, match="must be an existing file"):
            fsc._file_sha256(tmp_path / "absent.bin")

    def test_safe_relative_uri_rejects_scheme(self):
        with pytest.raises(ValueError, match="must be a safe relative path"):
            fsc._safe_relative_uri("bundle_uri", "https://evil/x")

    def test_resolve_under_root_rejects_symlink_escape(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation is not permitted on this platform")
        with pytest.raises(ValueError, match="path escapes artifact_root"):
            fsc._resolve_under_root(root, "link/file.json")


class TestPolicyEnforcement:
    def _payload(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "max_depth": 2,
            "artifact_sha256": "a" * 64,
            "sections": {"ctl": {}, "ltl": {}},
            "checked_specs": ["marking_bounds"],
        }
        base.update(overrides)
        return base

    def test_requires_min_depth(self):
        policy = SafetyCertificatePolicy(name="p", min_depth=5)
        with pytest.raises(ValueError, match="requires max_depth to meet min_depth"):
            fsc._enforce_safety_certificate_policy(self._payload(max_depth=1), policy)

    def test_requires_artifact_binding(self):
        policy = SafetyCertificatePolicy(name="p", require_artifact_binding=True)
        with pytest.raises(ValueError, match="requires artifact binding"):
            fsc._enforce_safety_certificate_policy(self._payload(artifact_sha256=None), policy)

    def test_requires_sections_object(self):
        policy = SafetyCertificatePolicy(name="p")
        with pytest.raises(ValueError, match="requires certificate sections"):
            fsc._enforce_safety_certificate_policy(self._payload(sections="flat"), policy)

    def test_requires_ctl_evidence(self):
        policy = SafetyCertificatePolicy(name="p", require_ctl=True)
        with pytest.raises(ValueError, match="requires CTL evidence"):
            fsc._enforce_safety_certificate_policy(self._payload(sections={"ctl": None, "ltl": {}}), policy)

    def test_requires_checked_specs_list(self):
        policy = SafetyCertificatePolicy(name="p")
        with pytest.raises(ValueError, match="requires checked specs"):
            fsc._enforce_safety_certificate_policy(self._payload(checked_specs="x"), policy)

    def test_bundle_requires_certificates_list(self):
        policy = SafetyCertificateBundlePolicy(name="b")
        with pytest.raises(ValueError, match="requires certificates"):
            fsc._enforce_safety_certificate_bundle_policy({"certificates": "x"}, policy)

    def test_bundle_requires_minimum_count(self):
        policy = SafetyCertificateBundlePolicy(name="b", min_certificates=3)
        with pytest.raises(ValueError, match="requires more certificates"):
            fsc._enforce_safety_certificate_bundle_policy({"certificates": [{}]}, policy)

    def test_bundle_requires_shared_backend(self):
        policy = SafetyCertificateBundlePolicy(name="b", require_same_backend=True)
        certificates = [{"backend": "explicit-state"}, {"backend": "other-backend"}]
        with pytest.raises(ValueError, match="requires a shared backend"):
            fsc._enforce_safety_certificate_bundle_policy({"certificates": certificates}, policy)

    def test_bundle_requires_matching_certificate_policy(self):
        policy = SafetyCertificateBundlePolicy(name="b", required_policy_name="gate")
        with pytest.raises(ValueError, match="requires matching certificate policy"):
            fsc._enforce_safety_certificate_bundle_policy({"certificates": [{"policy": {"name": "other"}}]}, policy)

    def test_unique_digests_requires_certificates(self):
        with pytest.raises(ValueError, match="must include at least one certificate"):
            fsc._enforce_unique_certificate_digests([])

    def test_unique_digests_requires_payload_digests(self):
        with pytest.raises(ValueError, match="must carry payload digests"):
            fsc._enforce_unique_certificate_digests([{"payload_sha256": "bad"}])

    def test_unique_digests_rejects_duplicates(self):
        with pytest.raises(ValueError, match="certificates must be unique"):
            fsc._enforce_unique_certificate_digests([{"payload_sha256": "a" * 64}, {"payload_sha256": "a" * 64}])


class TestCertificateMarkdownCounterexamples:
    def test_failing_certificate_markdown_lists_counterexamples(self, tmp_path: Path):
        net = _transfer_net()
        report = verify_formal_contracts(
            net, max_depth=2, marking_bounds={"sink": (0.0, 0.0)}, backend="explicit-state"
        )
        payload = write_safety_certificate(
            report,
            json_path=tmp_path / "cert.json",
            markdown_path=tmp_path / "cert.md",
        )
        assert payload["status"] == "fail"
        markdown = (tmp_path / "cert.md").read_text(encoding="utf-8")
        assert "## Counterexamples" in markdown


def test_certificate_checked_specs_skips_already_seeded_spec() -> None:
    """A section spec already present in the seed list is not appended twice (branch 530->529)."""
    report = SimpleNamespace(temporal=SimpleNamespace(checked_specs=["marking_bounds", "custom_spec"]))
    specs = fsc._certificate_checked_specs(report, None, None)
    assert specs == ["marking_bounds", "transition_liveness", "custom_spec"]


def test_certificate_checked_specs_from_sections_skips_duplicate() -> None:
    """A section spec already present in the seed list is recorded once (branch 544->543)."""
    sections = {"temporal": {"checked_specs": ["marking_bounds", "extra"]}}
    specs = fsc._certificate_checked_specs_from_sections(sections)
    assert specs == ["marking_bounds", "transition_liveness", "extra"]


def test_enforce_bundle_policy_without_required_policy_name() -> None:
    """A bundle policy with no required policy name returns after the earlier checks (branch 634->exit)."""
    policy = SafetyCertificateBundlePolicy(name="minimal", min_certificates=1)
    # required_policy_name defaults to None and the other requirements are disabled, so this must not raise.
    fsc._enforce_safety_certificate_bundle_policy({"certificates": [{}]}, policy)
