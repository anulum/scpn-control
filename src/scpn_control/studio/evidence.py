# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio evidence bundles
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Emit ``studio.*.v1`` evidence bundles for CONTROL results (schema B).

This module is the crux of CONTROL's vertical: it maps CONTROL's existing,
provenance-graded result surfaces onto the locked platform
:class:`scpn_studio_platform.evidence.EvidenceBundle`. Three mappers exercise the
contract's honesty invariant from three different sides, which is exactly why
CONTROL is the contract's second-domain test:

- An **EFIT reconstruction** is a real least-squares inverse but only
  closure-validated against synthetic diagnostics, so its claim is
  ``bounded-model`` and does **not** render as validated — the Hub shows the
  boundary verbatim (the honesty lever CONTROL contributed).
- A **runtime safety certificate** is ``formally-proven``: it carries a
  :class:`scpn_studio_platform.evidence.FormalCertificate` whose ``subject_digest``
  binds the proof to the exact Petri-net topology it ran on, so the Hub voids the
  proof the moment the live controller's topology drifts.
- A **controller-latency** result is a ``measured`` local benchmark, not a
  hardware-in-the-loop guarantee, so its claim is ``bounded-model`` too.

None of these outcomes is a CONTROL-private rule; each falls out of the shared
:class:`scpn_studio_platform.evidence.ClaimBoundary` lattice and the orthogonal
:class:`scpn_studio_platform.evidence.EvidenceKind` axis.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    FormalCertificate,
    NumericProvenance,
    PhysicalContract,
    ProvActivity,
    ProvAgent,
    ProvEntity,
)

from .verbs import (
    CONTROLLER_LATENCY_SCHEMA,
    EFIT_RECONSTRUCTION_SCHEMA,
    SAFETY_CERTIFICATE_SCHEMA,
    STUDIO_ID,
)


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex of a canonical-JSON payload.

    Parameters
    ----------
    payload
        A JSON-serialisable mapping.

    Returns
    -------
    str
        The lowercase hex SHA-256 over the canonical (sorted-key, tight-separator,
        NaN-rejecting) JSON encoding — the same convention the rest of the ecosystem
        uses so a result's digest matches across the studio boundary.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ── EFIT reconstruction (bounded-model, measured) ──────────────────────
@dataclass(frozen=True, slots=True)
class EfitReconstructionResult:
    """The path-free result of a real-time EFIT-lite reconstruction.

    Mirrors the public shape of ``RealtimeEFIT.reconstruct`` reduced to the fields
    the evidence bundle needs. The reconstruction is closure-validated against
    synthetic diagnostics only, so its claim stays bounded.

    Parameters
    ----------
    ip_reconstructed_a
        Plasma current recovered from the fitted flux, in amperes.
    chi_squared
        The measured weighted least-squares residual (not forced).
    n_iterations
        The number of outer Picard iterations actually run.
    nr, nz
        Radial and vertical grid sizes of the R-Z reconstruction grid.
    input_digest
        SHA-256 of the diagnostic/configuration inputs.
    result_digest
        SHA-256 of the reconstructed flux and profile coefficients.

    Raises
    ------
    ValueError
        If a count is non-positive or a digest is empty.
    """

    ip_reconstructed_a: float
    chi_squared: float
    n_iterations: int
    nr: int
    nz: int
    input_digest: str
    result_digest: str

    def __post_init__(self) -> None:
        """Validate counts and digests."""
        if self.n_iterations <= 0:
            raise ValueError("EfitReconstructionResult.n_iterations must be > 0")
        if self.nr <= 0 or self.nz <= 0:
            raise ValueError("EfitReconstructionResult grid sizes must be > 0")
        if not self.input_digest.strip() or not self.result_digest.strip():
            raise ValueError("EfitReconstructionResult digests must be non-empty")


def efit_reconstruction_evidence(
    result: EfitReconstructionResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
) -> EvidenceBundle:
    """Build the ``studio.efit-reconstruction.v1`` bundle.

    The inverse is real (measured chi-squared, recovered Ip) but only
    closure-validated against synthetic diagnostics, so the claim is
    ``bounded-model`` and the runtime admission is ``rejected`` for facility use —
    the bundle does not render as validated.

    Parameters
    ----------
    result
        The path-free reconstruction result.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio that produced the result.
    started, ended
        ISO-8601 start/end timestamps (passed in; no hidden clock).
    host
        Optional host descriptor the reconstruction ran on.

    Returns
    -------
    EvidenceBundle
        A schema-B bundle whose ``renders_as_validated`` is ``False``.
    """
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/efit/{result.result_digest}", digest=result.result_digest)
    activity = ProvActivity(verb="reconstruct", studio=STUDIO_ID, started=started, ended=ended, host=host)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(
        units={"psi": "Wb", "Ip": "A", "chi_squared": "dimensionless"},
        grid={"nz": result.nz, "nr": result.nr, "coords": "R-Z"},
    )
    numeric = NumericProvenance(active_backend="python", reference_backend="synthetic-closure")
    claim = ClaimBoundary(status=ClaimStatus.BOUNDED_MODEL, admission=AdmissionDecision.REJECTED)
    return EvidenceBundle(
        schema=EFIT_RECONSTRUCTION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=claim,
        numeric_provenance=numeric,
        physical_contract=physical,
    )


# ── runtime safety certificate (formally-proven) ───────────────────────
@dataclass(frozen=True, slots=True)
class SafetyCertificateResult:
    """The path-free result of a runtime safety certificate replay.

    Mirrors ``scpn_control.scpn.runtime_safety_certificate``: a machine-checked
    CTL/LTL proof over a compiled ``StochasticPetriNet`` bound to the exact
    Petri-net topology digest the proof ran on.

    Parameters
    ----------
    theorem_id
        Identifier of the proven safety/liveness obligation.
    checker, checker_version
        The proof engine and its exact version (e.g. ``"z3"`` / ``"4.13.0"``).
    proof_digest
        SHA-256 of the proof artifact itself.
    petri_topology_sha256
        SHA-256 of the Petri-net topology the proof was established against — the
        certificate's subject.
    live_topology_sha256
        SHA-256 of the live controller's topology at replay time.
    result_digest
        SHA-256 of the issued certificate record.
    non_vacuous
        Whether the obligation was checked to be non-vacuous.

    Raises
    ------
    ValueError
        If any digest or identity field is empty.
    """

    theorem_id: str
    checker: str
    checker_version: str
    proof_digest: str
    petri_topology_sha256: str
    live_topology_sha256: str
    result_digest: str
    non_vacuous: bool = False

    def __post_init__(self) -> None:
        """Validate that the identifying fields are non-empty."""
        for name in (
            "theorem_id",
            "checker",
            "checker_version",
            "proof_digest",
            "petri_topology_sha256",
            "live_topology_sha256",
            "result_digest",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"SafetyCertificateResult.{name} must be non-empty")

    @property
    def topology_matches(self) -> bool:
        """Whether the live controller topology matches the proven one."""
        return self.live_topology_sha256 == self.petri_topology_sha256


def safety_certificate_evidence(
    result: SafetyCertificateResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.safety-certificate.v1`` bundle (formally-proven).

    The bundle carries a :class:`FormalCertificate` whose ``subject_digest`` is the
    proven Petri-net topology. The claim renders as validated **only** when the
    proof holds for the live topology (no drift); a drifted topology degrades the
    claim to ``validation-gap`` / ``rejected`` so the Hub never presents a voided
    proof as established.

    Parameters
    ----------
    result
        The path-free certificate replay result.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``formally-proven`` schema-B bundle; admissible iff the proof covers the
        live subject.
    """
    certificate = FormalCertificate(
        checker=result.checker,
        checker_version=result.checker_version,
        theorem_id=result.theorem_id,
        proof_digest=result.proof_digest,
        subject_digest=result.petri_topology_sha256,
        non_vacuous=result.non_vacuous,
    )
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/safety-certificate/{result.result_digest}", digest=result.result_digest)
    activity = ProvActivity(verb="certify", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    proof_holds = certificate.covers(result.live_topology_sha256)
    if proof_holds:
        claim = ClaimBoundary(status=ClaimStatus.REFERENCE_VALIDATED, admission=AdmissionDecision.ADMITTED)
    else:
        claim = ClaimBoundary(status=ClaimStatus.VALIDATION_GAP, admission=AdmissionDecision.REJECTED)
    return EvidenceBundle(
        schema=SAFETY_CERTIFICATE_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.FORMALLY_PROVEN,
        claim_boundary=claim,
        formal_certificates=(certificate,),
    )


# ── controller latency (measured local benchmark) ──────────────────────
@dataclass(frozen=True, slots=True)
class ControllerLatencyResult:
    """The path-free result of a control-cycle latency benchmark.

    Mirrors the native-handoff / controller-latency reports: a measured local
    benchmark with explicit percentiles and a reference backend, but not a
    hardware-in-the-loop real-time guarantee.

    Parameters
    ----------
    controller
        Name of the controller benchmarked.
    active_backend
        The backend timed (e.g. ``"rust"``).
    reference_backend
        The reference backend the active one is compared against (e.g. ``"python"``).
    p50_us, p95_us, p99_us
        Latency percentiles in microseconds.
    sample_count
        The number of timed iterations.
    input_digest, result_digest
        SHA-256 of the benchmark configuration and the recorded percentiles.

    Raises
    ------
    ValueError
        If a percentile is non-positive, the sample count is too small, or a digest
        is empty.
    """

    controller: str
    active_backend: str
    reference_backend: str
    p50_us: float
    p95_us: float
    p99_us: float
    sample_count: int
    input_digest: str
    result_digest: str

    def __post_init__(self) -> None:
        """Validate percentiles, sample count, and digests."""
        if not (self.p50_us > 0 and self.p95_us > 0 and self.p99_us > 0):
            raise ValueError("ControllerLatencyResult percentiles must be positive")
        if self.sample_count < 1:
            raise ValueError("ControllerLatencyResult.sample_count must be >= 1")
        if not self.input_digest.strip() or not self.result_digest.strip():
            raise ValueError("ControllerLatencyResult digests must be non-empty")


def controller_latency_evidence(
    result: ControllerLatencyResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
) -> EvidenceBundle:
    """Build the ``studio.controller-latency.v1`` bundle (measured benchmark).

    A measured local benchmark is real but bounded — it is not a hardware-in-the-loop
    real-time guarantee — so the claim is ``bounded-model`` / ``rejected`` and does
    not render as validated.

    Parameters
    ----------
    result
        The path-free latency result.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.
    host
        Optional host descriptor the benchmark ran on.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle whose ``renders_as_validated`` is ``False``.
    """
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/controller-latency/{result.controller}/{result.result_digest}",
        digest=result.result_digest,
    )
    activity = ProvActivity(verb="benchmark", studio=STUDIO_ID, started=started, ended=ended, host=host)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    # A latency benchmark records which backend was timed against which reference;
    # it is NOT a numerical-parity claim, so it carries no ParityCheck (those assert
    # bit/tolerance agreement of values, not timing).
    numeric = NumericProvenance(
        active_backend=result.active_backend,
        reference_backend=result.reference_backend,
    )
    physical = PhysicalContract(units={"latency": "us"}, grid={"percentiles": ["p50", "p95", "p99"]})
    claim = ClaimBoundary(status=ClaimStatus.BOUNDED_MODEL, admission=AdmissionDecision.REJECTED)
    return EvidenceBundle(
        schema=CONTROLLER_LATENCY_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=claim,
        numeric_provenance=numeric,
        physical_contract=physical,
    )
