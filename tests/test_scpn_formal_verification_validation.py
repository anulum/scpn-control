# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal verification validator and fail-closed paths
"""Validator and fail-closed branches of the formal Petri-net verifier.

Exercises the policy/formula dataclass guards, the backend resolver, the
verifier construction and spec guards, the inhibitor and defensive firing
branches, and every field validator on the safety certificate, certificate
bundle, and bundle-artifact schemas. Schema validators run before the trailing
digest check, so a single mutated field of an otherwise valid payload reaches
exactly its intended guard without re-sealing.
"""

from __future__ import annotations

import builtins
import copy
import json
import types
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pytest

import scpn_control.scpn.formal_safety_certificate as fsc
import scpn_control.scpn.formal_verification as fv
from scpn_control.scpn.formal_safety_certificate import (
    SafetyCertificateBundlePolicy,
    SafetyCertificatePolicy,
    build_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_payload,
    build_safety_certificate_payload,
    validate_safety_certificate_bundle_artifact,
    validate_safety_certificate_bundle_payload,
    validate_safety_certificate_payload,
    write_safety_certificate,
)
from scpn_control.scpn.formal_verification import (
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalPetriNetVerifier,
    LTLFormula,
    PlaceInvariant,
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


def _inhibitor_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("guard", initial_tokens=1.0)
    net.add_place("token", initial_tokens=1.0)
    net.add_place("out", initial_tokens=0.0)
    net.add_transition("fire", threshold=1.0)
    net.add_arc("token", "fire", weight=1.0)
    net.add_arc("guard", "fire", weight=1.0, inhibitor=True)
    net.add_arc("fire", "out", weight=1.0)
    net.compile(allow_inhibitor=True)
    return net


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


# ── policy and formula dataclass guards ───────────────────────────────


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


class TestFormulaDataclassGuards:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "", "operator": "AG", "target": "marking_bounds", "params": {"a": 1}}, "name must be"),
            (
                {"name": "f", "operator": "XX", "target": "marking_bounds", "params": {"a": 1}},
                "unsupported CTL operator",
            ),
            ({"name": "f", "operator": "AG", "target": "xx", "params": {"a": 1}}, "unsupported CTL target"),
            ({"name": "f", "operator": "AG", "target": "marking_bounds", "params": {}}, "non-empty mapping"),
        ],
    )
    def test_rejects_invalid_ctl(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CTLFormula(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "", "operator": "G", "target": "marking_bounds", "params": {"a": 1}}, "name must be"),
            (
                {"name": "f", "operator": "XX", "target": "marking_bounds", "params": {"a": 1}},
                "unsupported LTL operator",
            ),
            ({"name": "f", "operator": "G", "target": "marking_bounds", "params": {}}, "non-empty mapping"),
        ],
    )
    def test_rejects_invalid_ltl(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            LTLFormula(**kwargs)

    def test_rejects_unsupported_ltl_target(self):
        with pytest.raises(ValueError, match="unsupported LTL target"):
            LTLFormula(name="f", operator="G", target="xx", params={"a": 1})


# ── backend resolution ────────────────────────────────────────────────


class TestBackendResolution:
    def test_explicit_z3_resolves_when_present(self):
        assert fv._resolve_backend("z3") == "z3"

    def test_rejects_unsupported_backend(self):
        with pytest.raises(ValueError, match="unsupported formal verification backend"):
            fv._resolve_backend("smt-magic")

    def test_explicit_z3_requires_solver(self, monkeypatch):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "z3":
                raise ModuleNotFoundError("no z3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(RuntimeError, match="z3-solver"):
            fv._resolve_backend("z3")

    def test_auto_falls_back_to_explicit_state(self, monkeypatch):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "z3":
                raise ModuleNotFoundError("no z3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        assert fv._resolve_backend("auto") == "explicit-state"


# ── verifier construction and spec guards ─────────────────────────────


class TestVerifierConstruction:
    def test_rejects_uncompiled_net(self):
        net = StochasticPetriNet()
        net.add_place("p", initial_tokens=1.0)
        net.add_transition("t", threshold=1.0)
        net.add_arc("p", "t", weight=1.0)
        with pytest.raises(RuntimeError, match="compiled before formal verification"):
            FormalPetriNetVerifier(net)

    def test_rejects_compiled_net_without_weight_matrices(self):
        net = _transfer_net()
        net.W_in = None
        with pytest.raises(RuntimeError, match="compiled before formal verification"):
            FormalPetriNetVerifier(net)


class TestSpecAndBoundGuards:
    def test_rejects_empty_marking_bounds(self):
        with pytest.raises(ValueError, match="marking bounds must not be empty"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_marking_bounds({}, max_depth=1)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_marking_bounds(
                {"sink": (1.0, 0.0)}, max_depth=1
            )

    def test_rejects_unknown_place_in_bounds(self):
        with pytest.raises(ValueError, match="unknown place"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_marking_bounds(
                {"ghost": (0.0, 1.0)}, max_depth=1
            )

    def test_rejects_empty_invariant_list(self):
        with pytest.raises(ValueError, match="place invariants must not be empty"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_place_invariants([], max_depth=1)

    @pytest.mark.parametrize(
        ("invariant", "match"),
        [
            (PlaceInvariant("", {"source": 1.0}), "place invariant name must not be empty"),
            (PlaceInvariant("inv", {}), "must include at least one place weight"),
            (PlaceInvariant("inv", {"source": 0.0, "sink": 0.0}), "must not be identically zero"),
        ],
    )
    def test_rejects_degenerate_invariants(self, invariant, match):
        with pytest.raises(ValueError, match=match):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_place_invariants(
                [invariant], max_depth=1
            )

    def test_rejects_unknown_transition_in_eventually_fires(self):
        with pytest.raises(ValueError, match="unknown transition"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").verify_temporal_specs(
                [EventuallyFires("live", "ghost")], max_depth=2
            )

    def test_eventually_fires_reports_unfired_transition(self):
        net = StochasticPetriNet()
        net.add_place("locked", initial_tokens=0.0)
        net.add_place("done", initial_tokens=0.0)
        net.add_transition("needs_token", threshold=1.0)
        net.add_arc("locked", "needs_token", weight=1.0)
        net.add_arc("needs_token", "done", weight=1.0)
        net.compile()
        report = FormalPetriNetVerifier(net, backend="explicit-state").verify_temporal_specs(
            [EventuallyFires("live", "needs_token")], max_depth=2
        )
        assert report.holds is False
        assert report.violations[0].transition == "needs_token"

    def test_rejects_negative_response_window(self):
        with pytest.raises(ValueError, match="within must be an integer >= 0"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").verify_temporal_specs(
                [FireLeadsToMarking("resp", "move", "sink", within=-1)], max_depth=2
            )

    def test_rejects_unsupported_temporal_spec_type(self):
        bogus = types.SimpleNamespace(name="bogus")
        with pytest.raises(TypeError, match="unsupported temporal specification"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").verify_temporal_specs(
                [bogus],  # type: ignore[list-item]
                max_depth=1,
            )


class TestCTLLTLCompilation:
    def _verifier(self) -> FormalPetriNetVerifier:
        return FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")

    @pytest.mark.parametrize(
        ("formula", "match"),
        [
            (CTLFormula("f", "AG", "marking_bounds", {"bounds": "x"}), "requires marking bounds"),
            (CTLFormula("f", "EF", "transition_fires", {"transition": ""}), "requires a transition"),
            (CTLFormula("f", "AG", "not_comarked", {"place_a": "a", "place_b": 1}), "requires two places"),
            (CTLFormula("f", "AG_EF", "marked", {"place": 1}), "requires a place"),
            (CTLFormula("f", "EF", "marking_bounds", {"bounds": {}}), "unsupported CTL formula combination"),
        ],
    )
    def test_rejects_invalid_ctl_compilation(self, formula, match):
        with pytest.raises(ValueError, match=match):
            self._verifier().verify_ctl_specs([formula], max_depth=1)

    @pytest.mark.parametrize(
        ("formula", "match"),
        [
            (LTLFormula("f", "G", "marking_bounds", {"bounds": "x"}), "requires marking bounds"),
            (LTLFormula("f", "F", "transition_fires", {"transition": ""}), "requires a transition"),
            (
                LTLFormula(
                    "f", "G_implies_F", "fire_leads_to_marking", {"trigger_transition": "t", "target_place": ""}
                ),
                "requires trigger transition and target place",
            ),
            (LTLFormula("f", "F", "marking_bounds", {"bounds": {}}), "unsupported LTL formula combination"),
        ],
    )
    def test_rejects_invalid_ltl_compilation(self, formula, match):
        with pytest.raises(ValueError, match=match):
            self._verifier().verify_ltl_specs([formula], max_depth=1)


# ── firing relation: inhibitor and defensive negative guard ───────────


class TestFiringRelation:
    def test_inhibitor_arc_blocks_transition(self):
        report = FormalPetriNetVerifier(_inhibitor_net(), backend="explicit-state").analyze_reachability(max_depth=2)
        assert "fire" not in report.fired_transitions

    def test_defensive_negative_marking_is_rejected(self):
        verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
        t_idx = verifier._transition_index["move"]
        verifier._outputs[t_idx] = {verifier._place_index["sink"]: Fraction(-5)}
        assert verifier._fire_if_enabled(verifier._initial, "move") is None

    def test_defensive_reachable_invariant_drift_is_reported(self, monkeypatch):
        verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
        monkeypatch.setattr(verifier, "_transition_invariant_delta", lambda transition, weights: Fraction(0))
        report = verifier.prove_place_invariants([PlaceInvariant("sink_only", {"sink": 1.0})], max_depth=2)
        assert report.holds is False
        assert "changes invariant" in report.violations[0].message


# ── safety certificate payload validation ─────────────────────────────


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
            (lambda p: p.__setitem__("backend", "z3"), "backend must match certificate backends"),
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
        certificates = [{"backend": "explicit-state"}, {"backend": "z3"}]
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
