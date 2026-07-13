# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime-bound formal safety certificate

"""Bind a bounded formal safety certificate to a declared controller runtime.

`scpn_control.scpn.formal_verification` proves bounded CTL/LTL safety and
liveness obligations over a compiled `StochasticPetriNet` and emits a
schema-versioned `scpn-control.safety-certificate.v1` payload bound only to an
opaque artefact digest. That proves the *control logic* is safe, but it does not
tie the proof to the *thing that will run*: the exact controller configuration,
the Petri-net topology the proof actually ran on, the SNN / neuro-symbolic
parameters, the solver mode, the firmware / runtime target, and the declared
timing envelope.

This module closes that gap. It computes a canonical digest of a compiled net's
topology, models the structured controller runtime identity
(`ControllerRuntimeBinding`), and issues a `scpn-control.runtime-safety-certificate.v1`
wrapper *only after* the binding's declared topology matches the net the proof
ran on and the embedded formal certificate holds. Admission of the wrapper for
facility-facing use fails closed unless the certificate holds, its binding digest
matches the live controller binding, its declared runtime target matches the live
target, its timing envelope is schedulable, and a fresh proof replay reproduces
the certified obligations. It is not a facility safety approval; it is the
bounded, replayable evidence such approval would build on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from scpn_control.scpn.formal_safety_certificate import validate_safety_certificate_payload
from scpn_control.scpn.structure import StochasticPetriNet

RUNTIME_SAFETY_CERTIFICATE_SCHEMA_VERSION = "scpn-control.runtime-safety-certificate.v1"
RUNTIME_SAFETY_CERTIFICATE_SCOPE = (
    "bounded formal safety certificate bound to a declared controller runtime identity, "
    "Petri-net topology, and timing envelope"
)
RUNTIME_SAFETY_CERTIFICATE_CLAIM_BOUNDARY = (
    "not a facility safety approval or hardware timing certificate; facility admission requires "
    "proof replay and target-runtime match on the declared hardware and software stack"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")) or out <= 0.0:
        raise ValueError(f"{name} must be a positive, finite number")
    return out


def compute_petri_topology_digest(net: StochasticPetriNet) -> str:
    """Return a canonical SHA-256 of a compiled net's topology (not its tokens).

    The signature covers the place ordering, each transition's name, firing
    threshold and delay, and the compiled input/output incidence (which encodes
    arc weights, including inhibitor arcs as negative input weights). Initial
    token densities are deliberately excluded: the certificate binds the control
    *structure*, not a transient marking.
    """
    if not net.is_compiled or net.W_in is None or net.W_out is None:
        raise ValueError("Petri-net topology digest requires a compiled net")
    thresholds = net.get_thresholds().tolist()
    delays = net.get_delay_ticks().tolist()
    w_in = net.W_in.tocoo()
    w_out = net.W_out.tocoo()
    signature = {
        "places": net.place_names,
        "transitions": [
            {"name": name, "threshold": float(thresholds[i]), "delay_ticks": int(delays[i])}
            for i, name in enumerate(net.transition_names)
        ],
        "w_in": sorted([int(r), int(c), float(v)] for r, c, v in zip(w_in.row, w_in.col, w_in.data)),
        "w_out": sorted([int(r), int(c), float(v)] for r, c, v in zip(w_out.row, w_out.col, w_out.data)),
    }
    return _digest(signature)


@dataclass(frozen=True)
class RuntimeTarget:
    """The firmware / runtime target a certified controller is admitted onto."""

    name: str
    architecture: str
    runtime: str
    toolchain: str

    def __post_init__(self) -> None:
        _require_nonempty_str("runtime target name", self.name)
        _require_nonempty_str("runtime target architecture", self.architecture)
        _require_nonempty_str("runtime target runtime", self.runtime)
        _require_nonempty_str("runtime target toolchain", self.toolchain)

    def as_dict(self) -> dict[str, str]:
        """Return the runtime target as a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "runtime": self.runtime,
            "toolchain": self.toolchain,
        }

    def digest(self) -> str:
        """Return the SHA-256 digest of the runtime target."""
        return _digest(self.as_dict())


@dataclass(frozen=True)
class TimingEnvelope:
    """Declared bounded-runtime assumptions the certificate is valid under.

    `proof_firing_depth` is the bounded transition depth the formal proof
    covered; the timing fields assert each control tick completes inside its
    deadline within the control period. The envelope is schedulable only when
    ``0 < worst_case_response_us <= deadline_us <= control_period_us``.
    """

    control_period_us: float
    worst_case_response_us: float
    deadline_us: float
    proof_firing_depth: int

    def __post_init__(self) -> None:
        period = _require_positive_finite("control_period_us", self.control_period_us)
        wcrt = _require_positive_finite("worst_case_response_us", self.worst_case_response_us)
        deadline = _require_positive_finite("deadline_us", self.deadline_us)
        if isinstance(self.proof_firing_depth, bool) or not isinstance(self.proof_firing_depth, int):
            raise ValueError("proof_firing_depth must be an integer")
        if self.proof_firing_depth < 1:
            raise ValueError("proof_firing_depth must be >= 1")
        if not (wcrt <= deadline <= period):
            raise ValueError(
                "timing envelope is not schedulable: require worst_case_response_us <= deadline_us <= control_period_us"
            )

    @property
    def schedulable(self) -> bool:
        """Whether ``worst_case_response_us <= deadline_us <= control_period_us``."""
        return self.worst_case_response_us <= self.deadline_us <= self.control_period_us

    def as_dict(self) -> dict[str, float | int]:
        """Return the timing envelope as a JSON-serialisable mapping."""
        return {
            "control_period_us": self.control_period_us,
            "worst_case_response_us": self.worst_case_response_us,
            "deadline_us": self.deadline_us,
            "proof_firing_depth": self.proof_firing_depth,
        }

    def digest(self) -> str:
        """Return the SHA-256 digest of the timing envelope."""
        return _digest(self.as_dict())


@dataclass(frozen=True)
class ControllerRuntimeBinding:
    """The exact deployable controller identity a certificate is matched to."""

    controller_id: str
    controller_config: dict[str, Any]
    petri_topology_sha256: str
    snn_parameters: dict[str, Any]
    solver_mode: str
    runtime_target: RuntimeTarget
    timing_envelope: TimingEnvelope

    def __post_init__(self) -> None:
        _require_nonempty_str("controller_id", self.controller_id)
        if not isinstance(self.controller_config, dict) or not self.controller_config:
            raise ValueError("controller_config must be a non-empty mapping")
        if not _is_sha256(self.petri_topology_sha256):
            raise ValueError("petri_topology_sha256 must be a SHA-256 hex digest")
        if not isinstance(self.snn_parameters, dict):
            raise ValueError("snn_parameters must be a mapping")
        _require_nonempty_str("solver_mode", self.solver_mode)
        if not isinstance(self.runtime_target, RuntimeTarget):
            raise ValueError("runtime_target must be a RuntimeTarget")
        if not isinstance(self.timing_envelope, TimingEnvelope):
            raise ValueError("timing_envelope must be a TimingEnvelope")

    def canonical(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable view used for the binding digest."""
        return {
            "controller_id": self.controller_id,
            "controller_config": self.controller_config,
            "petri_topology_sha256": self.petri_topology_sha256,
            "snn_parameters": self.snn_parameters,
            "solver_mode": self.solver_mode,
            "runtime_target": self.runtime_target.as_dict(),
            "timing_envelope": self.timing_envelope.as_dict(),
        }

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical controller binding."""
        return _digest(self.canonical())


@dataclass(frozen=True)
class CertificateReplayResult:
    """Outcome of re-proving the obligations a certificate claims."""

    topology_matches: bool
    holds_matches: bool
    checked_specs_match: bool
    formal_digest_matches: bool
    detail: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the topology, holds, checked specs, and formal digest all matched."""
        return self.topology_matches and self.holds_matches and self.checked_specs_match and self.formal_digest_matches


def issue_runtime_safety_certificate(
    net: StochasticPetriNet,
    binding: ControllerRuntimeBinding,
    *,
    formal_certificate: dict[str, Any],
    issuer: str = "scpn-control",
) -> dict[str, Any]:
    """Issue a runtime-bound certificate only when proof and binding agree.

    Fails closed when the binding's declared topology does not match the net the
    proof ran on, when the embedded formal certificate is malformed, or when it
    does not hold.
    """
    _require_nonempty_str("issuer", issuer)
    formal = validate_safety_certificate_payload(formal_certificate)
    if not formal.get("holds", False):
        raise ValueError("cannot issue a runtime certificate for a formal certificate that does not hold")

    actual_topology = compute_petri_topology_digest(net)
    if actual_topology != binding.petri_topology_sha256:
        raise ValueError("binding petri_topology_sha256 does not match the compiled net the proof ran on")

    payload: dict[str, Any] = {
        "schema_version": RUNTIME_SAFETY_CERTIFICATE_SCHEMA_VERSION,
        "scope": RUNTIME_SAFETY_CERTIFICATE_SCOPE,
        "claim_boundary": RUNTIME_SAFETY_CERTIFICATE_CLAIM_BOUNDARY,
        "status": "pass",
        "holds": True,
        "issuer": issuer,
        "binding": binding.canonical(),
        "binding_sha256": binding.digest(),
        "runtime_target_sha256": binding.runtime_target.digest(),
        "timing_envelope_sha256": binding.timing_envelope.digest(),
        "formal_certificate": formal,
        "formal_certificate_sha256": formal["payload_sha256"],
        "checked_specs": list(formal.get("checked_specs", [])),
    }
    payload["payload_sha256"] = _digest({k: v for k, v in payload.items()})
    return validate_runtime_safety_certificate_payload(payload)


def validate_runtime_safety_certificate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a runtime safety certificate's structure, digests, and bindings."""
    if not isinstance(payload, dict):
        raise ValueError("runtime safety certificate must be an object")
    if payload.get("schema_version") != RUNTIME_SAFETY_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("runtime safety certificate schema_version is unsupported")
    if payload.get("scope") != RUNTIME_SAFETY_CERTIFICATE_SCOPE:
        raise ValueError("runtime safety certificate scope is unsupported")
    if payload.get("claim_boundary") != RUNTIME_SAFETY_CERTIFICATE_CLAIM_BOUNDARY:
        raise ValueError("runtime safety certificate claim_boundary is unsupported")
    for key in ("binding", "formal_certificate"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"runtime safety certificate {key} must be an object")
    for key in ("binding_sha256", "runtime_target_sha256", "timing_envelope_sha256", "formal_certificate_sha256"):
        if not _is_sha256(payload.get(key)):
            raise ValueError(f"runtime safety certificate {key} must be a SHA-256 hex digest")

    formal = validate_safety_certificate_payload(payload["formal_certificate"])
    if payload["formal_certificate_sha256"] != formal["payload_sha256"]:
        raise ValueError("formal_certificate_sha256 does not match the embedded formal certificate")
    if payload.get("holds") is not True or payload.get("status") != "pass":
        raise ValueError("runtime safety certificate must record a passing, holding proof")

    binding = payload["binding"]
    if not _is_sha256(binding.get("petri_topology_sha256")):
        raise ValueError("binding petri_topology_sha256 must be a SHA-256 hex digest")

    stamped = payload.get("payload_sha256")
    recomputed = _digest({k: v for k, v in payload.items() if k != "payload_sha256"})
    if stamped != recomputed:
        raise ValueError("runtime safety certificate payload_sha256 does not match its contents")
    return payload


def replay_runtime_safety_certificate(
    net: StochasticPetriNet,
    certificate: dict[str, Any],
    *,
    reverify: Callable[[], dict[str, Any]],
) -> CertificateReplayResult:
    """Re-prove the certified obligations and compare them to the certificate.

    `reverify` must re-run the formal proof on the same net and binding and
    return a fresh `scpn-control.safety-certificate.v1` payload. The replay
    passes only when the live net topology still matches the certificate's
    binding, the fresh proof still holds, and it covers the same checked specs
    and produces the same formal digest.
    """
    certificate = validate_runtime_safety_certificate_payload(certificate)
    detail: list[str] = []

    live_topology = compute_petri_topology_digest(net)
    topology_matches = live_topology == certificate["binding"]["petri_topology_sha256"]
    if not topology_matches:
        detail.append("live net topology digest does not match the certificate binding")

    fresh = validate_safety_certificate_payload(reverify())
    holds_matches = bool(fresh.get("holds", False)) and bool(certificate["formal_certificate"].get("holds", False))
    if not holds_matches:
        detail.append("replayed proof does not hold")

    certified_specs = sorted(certificate.get("checked_specs", []))
    replayed_specs = sorted(fresh.get("checked_specs", []))
    checked_specs_match = certified_specs == replayed_specs
    if not checked_specs_match:
        detail.append("replayed checked specs differ from the certificate")

    formal_digest_matches = fresh.get("payload_sha256") == certificate["formal_certificate_sha256"]
    if not formal_digest_matches:
        detail.append("replayed formal certificate digest differs from the certificate")

    return CertificateReplayResult(
        topology_matches=topology_matches,
        holds_matches=holds_matches,
        checked_specs_match=checked_specs_match,
        formal_digest_matches=formal_digest_matches,
        detail=detail,
    )


def assert_runtime_certificate_admissible(
    certificate: dict[str, Any],
    *,
    live_binding: ControllerRuntimeBinding,
    live_runtime_target: RuntimeTarget,
    replay: CertificateReplayResult,
) -> dict[str, Any]:
    """Fail closed unless a certificate may back a facility-facing safety claim.

    Admission requires, all on the declared stack: the certificate's binding
    digest matches the live controller binding, its declared runtime target
    matches the live target, its timing envelope matches the live binding, and a
    fresh proof replay reproduced the certified obligations. Validation (called
    first) already guarantees the certificate holds, and a `TimingEnvelope`
    cannot be constructed unschedulable, so those guarantees are upstream rather
    than re-checked here. Any failure raises; on success the validated
    certificate is returned.
    """
    certificate = validate_runtime_safety_certificate_payload(certificate)
    failures: list[str] = []

    if certificate["binding_sha256"] != live_binding.digest():
        failures.append("certificate binding does not match the live controller binding")
    if certificate["runtime_target_sha256"] != live_runtime_target.digest():
        failures.append("certificate runtime target does not match the live target")
    if certificate["timing_envelope_sha256"] != live_binding.timing_envelope.digest():
        failures.append("certificate timing envelope does not match the live binding")
    if not replay.passed:
        failures.append("proof replay did not reproduce the certified obligations: " + "; ".join(replay.detail))

    if failures:
        raise ValueError("runtime safety certificate is not admissible: " + "; ".join(failures))
    return certificate
