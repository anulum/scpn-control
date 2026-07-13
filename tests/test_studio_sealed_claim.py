# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sealed safety-claim artefact tests
"""Sealed-claim artefact contract: real-certificate build, JCS float guard,
deterministic rendering, digest-stable writing, and drift degradation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from scpn_control.scpn.formal_safety_certificate import generate_safety_certificate
from scpn_control.scpn.formal_verification import (
    CTLFormula,
    EventuallyFires,
    LTLFormula,
)
from scpn_control.scpn.runtime_safety_certificate import (
    ControllerRuntimeBinding,
    RuntimeTarget,
    TimingEnvelope,
    compute_petri_topology_digest,
    issue_runtime_safety_certificate,
)
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.studio.sealed_claim import (
    SEALED_SAFETY_CLAIM_SCHEMA,
    assert_jcs_safe,
    build_safety_certificate_sealed_claim,
    render_sealed_claim_json,
    write_sealed_claim,
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


def _issued_certificate(net: StochasticPetriNet) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        formal = generate_safety_certificate(
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
    binding = ControllerRuntimeBinding(
        controller_id="burn-ctl-1",
        controller_config={"kp": 1.2, "ki": 0.3},
        petri_topology_sha256=compute_petri_topology_digest(net),
        snn_parameters={"layers": 3, "weights_sha": "abc"},
        solver_mode="acados-rti",
        runtime_target=RuntimeTarget(name="rpi4-rt", architecture="aarch64", runtime="PREEMPT_RT", toolchain="gcc-13"),
        timing_envelope=TimingEnvelope(
            control_period_us=1000.0, worst_case_response_us=180.0, deadline_us=500.0, proof_firing_depth=4
        ),
    )
    return issue_runtime_safety_certificate(net, binding, formal_certificate=formal)


def _build(certificate: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "live_topology_sha256": certificate["binding"]["petri_topology_sha256"],
        "checker": "z3",
        "checker_version": "4.13.0",
        "studio_version": "0.22.1",
        "claim_id": "control.safety-certificate.disarm-net",
        "issued_utc": "2026-07-04T00:00:00Z",
    }
    kwargs.update(overrides)
    return build_safety_certificate_sealed_claim(certificate, **kwargs)


@pytest.fixture(scope="module")
def certificate() -> dict[str, Any]:
    """One real z3-backed runtime certificate shared across the module."""
    return _issued_certificate(_disarm_net())


# ── build: real-certificate path ──────────────────────────────────────


def test_build_matching_topology_is_admitted(certificate: dict[str, Any]) -> None:
    payload = _build(certificate)
    assert payload["schema"] == SEALED_SAFETY_CLAIM_SCHEMA
    assert payload["studio"] == "scpn-control"
    claim = payload["claim"]
    assert claim["topology_matches"] is True
    assert claim["claim_status"] == "reference-validated"
    assert claim["admission"] == "admitted"
    assert claim["checker"] == "z3"
    assert claim["checked_specs"], "real certificate must carry proven specs"
    assert claim["theorem_id"] == " ; ".join(claim["checked_specs"])


def test_build_cites_certificate_digests(certificate: dict[str, Any]) -> None:
    payload = _build(certificate)
    prov = payload["provenance"]
    assert prov["certificate_sha256"] == certificate["payload_sha256"]
    assert prov["proof_sha256"] == certificate["formal_certificate_sha256"]
    assert prov["subject_topology_sha256"] == certificate["binding"]["petri_topology_sha256"]


def test_build_topology_drift_degrades_to_rejected(certificate: dict[str, Any]) -> None:
    payload = _build(certificate, live_topology_sha256="0" * 64)
    claim = payload["claim"]
    assert claim["topology_matches"] is False
    assert claim["claim_status"] == "validation-gap"
    assert claim["admission"] == "rejected"
    assert payload["provenance"]["live_topology_sha256"] == "0" * 64


@pytest.mark.parametrize(
    "field",
    ["live_topology_sha256", "checker", "checker_version", "studio_version", "claim_id", "issued_utc"],
)
def test_build_rejects_blank_identity_fields(certificate: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        _build(certificate, **{field: "  "})


def test_build_falls_back_to_scope_without_specs(certificate: dict[str, Any]) -> None:
    stripped = dict(certificate)
    stripped["checked_specs"] = []
    payload = _build(stripped)
    assert payload["claim"]["theorem_id"] == str(certificate["scope"])


def test_build_output_is_jcs_safe(certificate: dict[str, Any]) -> None:
    assert_jcs_safe(_build(certificate))


# ── assert_jcs_safe: the float-free gate ──────────────────────────────


def test_jcs_guard_rejects_float_scalars() -> None:
    with pytest.raises(ValueError, match=r"\$\.x: JSON floats"):
        assert_jcs_safe({"x": 2.014656})


def test_jcs_guard_rejects_nested_float_with_path() -> None:
    with pytest.raises(ValueError, match=r"\$\.a\[1\]\.b: JSON floats"):
        assert_jcs_safe({"a": [{"b": 1}, {"b": 0.5}]})


def test_jcs_guard_rejects_unsafe_integers() -> None:
    with pytest.raises(ValueError, match="2\\*\\*53-1"):
        assert_jcs_safe({"n": 2**53})
    assert_jcs_safe({"n": 2**53 - 1})
    with pytest.raises(ValueError, match="2\\*\\*53-1"):
        assert_jcs_safe([-(2**53)])


def test_jcs_guard_rejects_non_string_keys() -> None:
    with pytest.raises(ValueError, match="object keys must be strings"):
        assert_jcs_safe({1: "x"})


def test_jcs_guard_rejects_non_json_types() -> None:
    with pytest.raises(ValueError, match="not JSON-sealable"):
        assert_jcs_safe({"x": {1, 2}})
    with pytest.raises(ValueError, match="not JSON-sealable"):
        assert_jcs_safe(b"bytes")


def test_jcs_guard_accepts_exact_decimal_strings_and_scalars() -> None:
    assert_jcs_safe({"value": "2.014656", "count": 3, "flag": True, "none": None, "items": ["a", 1]})


# ── render: deterministic bytes ───────────────────────────────────────


def test_render_is_deterministic_and_newline_terminated(certificate: dict[str, Any]) -> None:
    payload = _build(certificate)
    first = render_sealed_claim_json(payload)
    second = render_sealed_claim_json(payload)
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first) == payload


def test_render_sorts_keys_compactly() -> None:
    rendered = render_sealed_claim_json({"b": 1, "a": {"d": None, "c": True}})
    assert rendered == '{"a":{"c":true,"d":null},"b":1}\n'


def test_render_fails_closed_on_float_injection(certificate: dict[str, Any]) -> None:
    payload = _build(certificate)
    payload["claim"]["latency_us"] = 59.2
    with pytest.raises(ValueError, match="JSON floats"):
        render_sealed_claim_json(payload)


# ── write: digest-stable artefact ─────────────────────────────────────


def test_write_returns_sha256_of_written_bytes(certificate: dict[str, Any], tmp_path: Path) -> None:
    payload = _build(certificate)
    target = tmp_path / "nested" / "claim.json"
    digest = write_sealed_claim(payload, target)
    data = target.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    assert json.loads(data.decode("utf-8")) == payload


def test_write_is_reproducible(certificate: dict[str, Any], tmp_path: Path) -> None:
    payload = _build(certificate)
    d1 = write_sealed_claim(payload, tmp_path / "one.json")
    d2 = write_sealed_claim(payload, tmp_path / "two.json")
    assert d1 == d2
