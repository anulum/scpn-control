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
    BlockedOn,
    ClaimBoundary,
    ClaimStatus,
    Convergence,
    ConvergenceStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Exactness,
    FormalCertificate,
    Freshness,
    NumericProvenance,
    ParityCheck,
    PhysicalContract,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    ValidityDomain,
)

from .verbs import (
    CONTROLLER_LATENCY_SCHEMA,
    CONTROLLER_RUN_SCHEMA,
    DISRUPTION_MITIGATION_SCHEMA,
    DISRUPTION_PREDICTION_SCHEMA,
    EFIT_RECONSTRUCTION_SCHEMA,
    EQUILIBRIUM_ANALYSIS_SCHEMA,
    GEOMETRY_NEUTRAL_REPLAY_SCHEMA,
    PARITY_REFUTATION_SCHEMA,
    PHASE_SYNC_MONITOR_SCHEMA,
    PHYSICS_VALIDATION_SCHEMA,
    SAFETY_CERTIFICATE_SCHEMA,
    SCENARIO_SIMULATION_SCHEMA,
    STUDIO_ID,
)

#: Maps a physics_traceability ``fidelity_status`` onto the platform claim lattice.
_FIDELITY_TO_STATUS: dict[str, ClaimStatus] = {
    "reference_validated": ClaimStatus.REFERENCE_VALIDATED,
    "bounded_model": ClaimStatus.BOUNDED_MODEL,
    "validation_gap": ClaimStatus.VALIDATION_GAP,
    "external_dependency_blocked": ClaimStatus.EXTERNAL_DEPENDENCY_BLOCKED,
}


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
    # The validity_domain (SDK 0.2.0, CONTROL's first v1.x additive) carries the
    # qualitative scope as prose — CONTROL's traceability validity_domains are
    # operating-regime/provenance caveats, not fabricated numeric ranges.
    validity = ValidityDomain(
        note=(
            "Fixed-boundary and free-boundary EFIT-lite reconstruction, closure-validated "
            "against synthetic diagnostics only; not facility-grade EFIT/P-EFIT validation "
            "unless matched reference evidence passes the strict admission gate."
        )
    )
    claim = ClaimBoundary(
        status=ClaimStatus.BOUNDED_MODEL,
        admission=AdmissionDecision.REJECTED,
        validity_domain=validity,
    )
    return EvidenceBundle(
        schema=EFIT_RECONSTRUCTION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=claim,
        freshness=Freshness.TRACEABLE_UNCHECKED,
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
        freshness=Freshness.VERIFIED_AT_SOURCE,
        formal_certificates=(certificate,),
    )


# ── cross-backend parity refutation (falsified / refuted) ──────────────
@dataclass(frozen=True, slots=True)
class ParityRefutationResult:
    """The path-free result of a cross-backend solver-parity check.

    Mirrors ``tests/test_rust_python_parity.py``: the Python and Rust ``FusionKernel``
    run the same equilibrium solver on the same grid and their final Psi fields are
    compared. For the SOR solver the hypothesis "the backends agree to ``target_rtol``"
    is TESTED and REFUTED with raw counts (PARITY-1) — both backends reach the same
    equilibrium *structure* (high ``pearson_r``, identical extrema) but SOR's update-norm
    stopping criterion plateaus at ``gs_residual_plateau`` far from the true GS solution,
    so the interior fields halt ``interior_l2_rel`` apart and exact-value parity is not
    reachable via SOR.

    Parameters
    ----------
    solver_method
        The solver whose cross-backend parity was checked (e.g. ``"sor"``).
    grid
        Opaque description of the test grid/case (e.g. the Solov'ev parameters).
    pearson_r
        Structural correlation of the two backends' final fields (~0.999 for SOR).
    interior_l2_rel
        Relative interior L2 gap at halt (the refutation magnitude; ~0.06 for SOR).
    gs_residual_plateau
        The Grad-Shafranov residual the SOR update-norm criterion plateaus at (~4).
    target_rtol
        The exact-value parity tolerance that is *not* reached (e.g. ``1e-3``).
    result_digest
        SHA-256 of the canonical result payload.
    raw_reference
        Pointer to the originating test / tracked finding (e.g. ``"PARITY-1"``).
    """

    solver_method: str
    grid: str
    pearson_r: float
    interior_l2_rel: float
    gs_residual_plateau: float
    target_rtol: float
    result_digest: str
    raw_reference: str


def parity_refutation_evidence(
    result: ParityRefutationResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.parity-refutation.v1`` bundle (falsified / refuted).

    A cross-backend numerical-parity hypothesis that was TESTED and REFUTED with raw
    counts — distinct from an untested ``validation-gap``. It is carried as
    ``evidence_kind=falsified`` + ``claim_status=refuted`` (admission ``rejected``) so
    the ledger records a promoted negative finding: a refuted parity can never render as
    a validated parity claim, and the raw ``ParityCheck``/``Convergence`` counts travel
    with it so a consumer can see exactly *how* it failed.

    Returns
    -------
    EvidenceBundle
        A ``falsified`` schema-B bundle whose ``renders_as_validated`` is ``False``.
    """
    parity = ParityCheck(
        reference="python",
        exactness=Exactness.TOLERANCE_AWARE,
        max_error=result.interior_l2_rel,
        passed=False,
        tolerance=result.target_rtol,
    )
    convergence = Convergence(
        status=ConvergenceStatus.MAX_ITERS,
        iterations=0,
        residual=result.gs_residual_plateau,
        tolerance=result.target_rtol,
    )
    numeric = NumericProvenance(
        active_backend="rust",
        reference_backend="python",
        parity=(parity,),
        convergence=convergence,
    )
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/parity-refutation/{result.result_digest}",
        digest=result.result_digest,
    )
    activity = ProvActivity(verb="verify", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    claim = ClaimBoundary(status=ClaimStatus.REFUTED, admission=AdmissionDecision.REJECTED)
    return EvidenceBundle(
        schema=PARITY_REFUTATION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.FALSIFIED,
        claim_boundary=claim,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=numeric,
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
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=numeric,
        physical_contract=physical,
    )


# ── physics-traceability claim (bounded, curated) ──────────────────────
@dataclass(frozen=True, slots=True)
class TraceabilityClaim:
    """A path-free physics-traceability registry claim.

    Mirrors one entry of ``validation/physics_traceability.json`` reduced to the
    fields the bundle needs: the component, its module, the fidelity status, whether
    its public claim is admitted, and the qualitative validity-domain prose.

    Parameters
    ----------
    component
        Human-readable name of the claim.
    module_path
        Source module the claim covers.
    fidelity_status
        One of ``reference_validated`` / ``bounded_model`` / ``validation_gap`` /
        ``external_dependency_blocked``.
    public_claim_allowed
        Whether the public (facility) claim is admitted.
    validity_domain
        The qualitative scope prose (no fabricated numeric ranges).
    blocking_dependency
        The missing dependency, required when ``fidelity_status`` is
        ``external_dependency_blocked``.

    Raises
    ------
    ValueError
        If a required string is empty, the status is unknown, or a blocked status
        carries no blocking dependency.
    """

    component: str
    module_path: str
    fidelity_status: str
    public_claim_allowed: bool
    validity_domain: str
    blocking_dependency: str | None = None

    def __post_init__(self) -> None:
        """Validate the fields and the status/blocking-dependency invariant."""
        for name in ("component", "module_path", "validity_domain"):
            if not getattr(self, name).strip():
                raise ValueError(f"TraceabilityClaim.{name} must be non-empty")
        if self.fidelity_status not in _FIDELITY_TO_STATUS:
            raise ValueError(f"unknown fidelity_status {self.fidelity_status!r}")
        if self.fidelity_status == "external_dependency_blocked" and not (self.blocking_dependency or "").strip():
            raise ValueError("external_dependency_blocked claim requires a blocking_dependency")


def physics_validation_evidence(
    claim: TraceabilityClaim,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build a ``studio.physics-validation.v1`` bundle for a traceability claim.

    Carries the claim's place on the seven-state lattice, its runtime admission
    (from ``public_claim_allowed``), and its qualitative validity-domain prose as a
    :class:`ValidityDomain` note — the honest scope without fabricated numeric
    ranges. Only the single reference-validated, admitted claim renders as
    validated; the rest render their boundary verbatim.

    Parameters
    ----------
    claim
        The path-free traceability claim.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``curated`` schema-B bundle whose admissibility reflects the claim status.
    """
    status = _FIDELITY_TO_STATUS[claim.fidelity_status]
    admission = AdmissionDecision.ADMITTED if claim.public_claim_allowed else AdmissionDecision.REJECTED
    blocked: tuple[BlockedOn, ...] = ()
    if status is ClaimStatus.EXTERNAL_DEPENDENCY_BLOCKED:
        blocked = (BlockedOn(dependency=str(claim.blocking_dependency), kind="external-dependency"),)
    boundary = ClaimBoundary(
        status=status,
        admission=admission,
        validity_domain=ValidityDomain(note=claim.validity_domain),
        blocked_on=blocked,
    )
    digest = canonical_digest(
        {
            "component": claim.component,
            "module_path": claim.module_path,
            "fidelity_status": claim.fidelity_status,
            "validity_domain": claim.validity_domain,
        }
    )
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/physics-validation/{claim.module_path}", digest=digest)
    activity = ProvActivity(verb="validate", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    level = (
        EvidenceLevel.ENGINEERING_VERIFIED
        if status is ClaimStatus.REFERENCE_VALIDATED
        else EvidenceLevel.SCIENTIFICALLY_CURATED
    )
    return EvidenceBundle(
        schema=PHYSICS_VALIDATION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=level,
        evidence_kind=EvidenceKind.CURATED,
        claim_boundary=boundary,
        freshness=Freshness.TRACEABLE_UNCHECKED,
    )


# ── geometry-neutral replay (bounded, measured) ────────────────────────
@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Path-free summary of a geometry-neutral replay.

    Mirrors the fields of ``GeometryNeutralReplayEvidence`` the bundle needs.

    Parameters
    ----------
    scenario_digest, trace_digest
        SHA-256 digests of the replayed scenario and trace.
    result_digest
        SHA-256 of the replay evidence payload.
    max_abs_current_a
        Peak absolute coil current over the replay, in amperes.
    p95_latency_us
        95th-percentile control-cycle latency over the replay, in microseconds.
    device_claim_allowed
        Whether the device (facility) claim is admitted.

    Raises
    ------
    ValueError
        If a digest is empty or a magnitude is negative.
    """

    scenario_digest: str
    trace_digest: str
    result_digest: str
    max_abs_current_a: float
    p95_latency_us: float
    device_claim_allowed: bool

    def __post_init__(self) -> None:
        """Validate the digests and magnitudes."""
        for name in ("scenario_digest", "trace_digest", "result_digest"):
            if not getattr(self, name).strip():
                raise ValueError(f"ReplaySummary.{name} must be non-empty")
        if self.max_abs_current_a < 0 or self.p95_latency_us < 0:
            raise ValueError("ReplaySummary magnitudes must be non-negative")


def geometry_neutral_replay_evidence_bundle(
    summary: ReplaySummary,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.geometry-neutral-replay.v1`` bundle.

    A replay is a recomputable measured run, but bounded synthetic regression
    evidence — its device claim is not admitted unless matched-reference evidence
    passes — so it does not render as validated.

    Parameters
    ----------
    summary
        The path-free replay summary.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle whose admissibility reflects the device claim.
    """
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/replay/{summary.result_digest}", digest=summary.result_digest)
    activity = ProvActivity(verb="replay", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(units={"coil_current": "A", "latency": "us"}, grid={})
    status = ClaimStatus.REFERENCE_VALIDATED if summary.device_claim_allowed else ClaimStatus.BOUNDED_MODEL
    admission = AdmissionDecision.ADMITTED if summary.device_claim_allowed else AdmissionDecision.REJECTED
    return EvidenceBundle(
        schema=GEOMETRY_NEUTRAL_REPLAY_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(status=status, admission=admission),
        freshness=Freshness.TRACEABLE_UNCHECKED,
        physical_contract=physical,
    )


# ── phase-sync monitor (bounded, measured telemetry) ───────────────────
@dataclass(frozen=True, slots=True)
class MonitorSnapshot:
    """Path-free phase-sync monitor snapshot.

    Mirrors the dashboard snapshot returned by ``RealtimeMonitor.tick``.

    Parameters
    ----------
    tick
        Monotonic tick index of the snapshot.
    r_global
        Global Kuramoto coherence in ``[0, 1]``.
    lambda_exp
        Estimated largest Lyapunov exponent.
    guard_approved
        Whether the Lyapunov guard approved the step.
    latency_us
        Tick wall time in microseconds.

    Raises
    ------
    ValueError
        If the tick or latency is negative, or coherence is outside ``[0, 1]``.
    """

    tick: int
    r_global: float
    lambda_exp: float
    guard_approved: bool
    latency_us: float

    def __post_init__(self) -> None:
        """Validate the snapshot ranges."""
        if self.tick < 0:
            raise ValueError("MonitorSnapshot.tick must be >= 0")
        if not (0.0 <= self.r_global <= 1.0):
            raise ValueError("MonitorSnapshot.r_global must be in [0, 1]")
        if self.latency_us < 0:
            raise ValueError("MonitorSnapshot.latency_us must be >= 0")


def phase_sync_monitor_evidence(
    snapshot: MonitorSnapshot,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.phase-sync-monitor.v1`` bundle.

    A live coherence/guard snapshot is measured telemetry, not a validated facility
    claim, so it is bounded-model and does not render as validated.

    Parameters
    ----------
    snapshot
        The path-free monitor snapshot.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle (bounded-model).
    """
    digest = canonical_digest({"tick": snapshot.tick, "r_global": snapshot.r_global, "lambda_exp": snapshot.lambda_exp})
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/phase-sync/{snapshot.tick}", digest=digest)
    activity = ProvActivity(verb="monitor", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(units={"R_global": "dimensionless", "lambda_exp": "1/s", "latency": "us"}, grid={})
    return EvidenceBundle(
        schema=PHASE_SYNC_MONITOR_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(status=ClaimStatus.BOUNDED_MODEL, admission=AdmissionDecision.REJECTED),
        freshness=Freshness.TRACEABLE_UNCHECKED,
        physical_contract=physical,
    )


# ── disruption prediction (validation-gap fixed-weight baseline) ──────────
@dataclass(frozen=True, slots=True)
class DisruptionPrediction:
    """Path-free disruption-risk prediction.

    Mirrors ``predict_disruption_risk`` output: a risk in ``[0, 1]`` from a
    hand-tuned fixed-weight baseline (U-015), not a model trained on a
    real disruption database — so its claim is a validation gap.

    Parameters
    ----------
    risk
        The predicted disruption risk in ``[0, 1]``.
    observable_count
        Number of toroidal-asymmetry observables the prediction used.
    result_digest
        SHA-256 of the prediction inputs/output.

    Raises
    ------
    ValueError
        If the risk is outside ``[0, 1]``, the count is negative, or the digest is
        empty.
    """

    risk: float
    observable_count: int
    result_digest: str

    def __post_init__(self) -> None:
        """Validate the risk range, count, and digest."""
        if not (0.0 <= self.risk <= 1.0):
            raise ValueError("DisruptionPrediction.risk must be in [0, 1]")
        if self.observable_count < 0:
            raise ValueError("DisruptionPrediction.observable_count must be >= 0")
        if not self.result_digest.strip():
            raise ValueError("DisruptionPrediction.result_digest must be non-empty")


def disruption_prediction_evidence(
    prediction: DisruptionPrediction,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.disruption-prediction.v1`` bundle.

    The predictor is a hand-tuned fixed-weight baseline with no ROC against a real
    disruption database, so the claim is a validation gap and is not admitted.

    Parameters
    ----------
    prediction
        The path-free prediction.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle on the validation-gap boundary.
    """
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/disruption-prediction/{prediction.result_digest}", digest=prediction.result_digest
    )
    activity = ProvActivity(verb="predict", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    validity = ValidityDomain(
        note=(
            "Hand-tuned fixed-weight sigmoid over toroidal-asymmetry observables; a "
            "fixed-weight baseline, not a model trained or ROC-validated on a real "
            "disruption database."
        )
    )
    return EvidenceBundle(
        schema=DISRUPTION_PREDICTION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.TAXONOMY,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.VALIDATION_GAP,
            admission=AdmissionDecision.REJECTED,
            validity_domain=validity,
        ),
        freshness=Freshness.TRACEABLE_UNCHECKED,
    )


# ── equilibrium analysis (bounded, measured) ───────────────────────────
@dataclass(frozen=True, slots=True)
class EquilibriumAnalysis:
    """Path-free macroscopic equilibrium-shape analysis.

    Mirrors the ``ShapeParams`` derived from a reconstruction.

    Parameters
    ----------
    r0, a, kappa
        Major radius, minor radius, elongation.
    q95, beta_pol, li
        Safety factor at 95% flux, poloidal beta, internal inductance.
    result_digest
        SHA-256 of the analysis output.

    Raises
    ------
    ValueError
        If a positive-definite geometry value is non-positive or the digest is empty.
    """

    r0: float
    a: float
    kappa: float
    q95: float
    beta_pol: float
    li: float
    result_digest: str

    def __post_init__(self) -> None:
        """Validate geometry positivity and the digest."""
        if self.r0 <= 0 or self.a <= 0 or self.kappa <= 0:
            raise ValueError("EquilibriumAnalysis R0/a/kappa must be positive")
        if not self.result_digest.strip():
            raise ValueError("EquilibriumAnalysis.result_digest must be non-empty")


def equilibrium_analysis_evidence(
    analysis: EquilibriumAnalysis,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.equilibrium-analysis.v1`` bundle.

    Macroscopic shape parameters derived from an EFIT-lite reconstruction inherit
    its bounded status, so the analysis does not render as validated.

    Parameters
    ----------
    analysis
        The path-free shape analysis.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle (bounded-model).
    """
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/equilibrium-analysis/{analysis.result_digest}", digest=analysis.result_digest
    )
    activity = ProvActivity(verb="analyse", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(
        units={"R0": "m", "a": "m", "kappa": "dimensionless", "q95": "dimensionless"},
        grid={},
    )
    return EvidenceBundle(
        schema=EQUILIBRIUM_ANALYSIS_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(status=ClaimStatus.BOUNDED_MODEL, admission=AdmissionDecision.REJECTED),
        freshness=Freshness.TRACEABLE_UNCHECKED,
        physical_contract=physical,
    )


# ── controller run (bounded, measured closed-loop run) ─────────────────
@dataclass(frozen=True, slots=True)
class ControllerRunResult:
    """Path-free summary of a closed-loop controller run against a plant.

    Mirrors the run summary of a closed-loop shape/current controller (for example
    ``run_neural_mpc_simulation``): the controller, the number of control steps, the
    realised mean tracking error, and the peak coil action/current the run drove.
    The run is a closed loop on a surrogate low-order plant, not a facility-certified
    live-hardware control outcome.

    Parameters
    ----------
    controller
        Name of the controller run.
    n_steps
        Number of closed-loop control steps executed.
    mean_tracking_error
        Mean state-tracking error over the run, in metres (the controlled state is
        the magnetic-axis / X-point position).
    max_abs_action
        Peak absolute per-step coil action over the run, in kA-turn.
    max_abs_coil_current
        Peak absolute coil current over the run, in kA-turn.
    result_digest
        SHA-256 of the run summary.

    Raises
    ------
    ValueError
        If the controller name is empty, the step count is below one, a magnitude is
        negative, or the digest is empty.
    """

    controller: str
    n_steps: int
    mean_tracking_error: float
    max_abs_action: float
    max_abs_coil_current: float
    result_digest: str

    def __post_init__(self) -> None:
        """Validate the controller name, step count, magnitudes, and digest."""
        if not self.controller.strip():
            raise ValueError("ControllerRunResult.controller must be non-empty")
        if self.n_steps < 1:
            raise ValueError("ControllerRunResult.n_steps must be >= 1")
        if self.mean_tracking_error < 0 or self.max_abs_action < 0 or self.max_abs_coil_current < 0:
            raise ValueError("ControllerRunResult magnitudes must be non-negative")
        if not self.result_digest.strip():
            raise ValueError("ControllerRunResult.result_digest must be non-empty")


def controller_run_evidence(
    result: ControllerRunResult,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
) -> EvidenceBundle:
    """Build the ``studio.controller-run.v1`` bundle (measured closed-loop run).

    ``regulate`` is the ecosystem's only realtime, live-hardware verb, but a
    *closed-loop run against a surrogate low-order plant* is measured-but-bounded
    evidence: it is not a facility-certified live-hardware control outcome (that is
    the ``certify`` verb's machine-checked safety certificate). So the claim is
    ``bounded-model`` / ``rejected`` and does not render as validated.

    Parameters
    ----------
    result
        The path-free controller-run summary.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.
    host
        Optional host descriptor the run executed on.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle whose ``renders_as_validated`` is ``False``.
    """
    entity = ProvEntity(
        entity_id=f"{STUDIO_ID}/controller-run/{result.controller}/{result.result_digest}",
        digest=result.result_digest,
    )
    activity = ProvActivity(verb="regulate", studio=STUDIO_ID, started=started, ended=ended, host=host)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(
        units={"tracking_error": "m", "action": "kA-turn", "coil_current": "kA-turn"},
        grid={},
    )
    validity = ValidityDomain(
        note=(
            "Closed-loop shape/current control run against a surrogate low-order plant "
            "model; not a facility-certified live-hardware control outcome (certification "
            "is the certify verb's machine-checked safety certificate)."
        )
    )
    return EvidenceBundle(
        schema=CONTROLLER_RUN_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.REJECTED,
            validity_domain=validity,
        ),
        freshness=Freshness.TRACEABLE_UNCHECKED,
        physical_contract=physical,
    )


# ── disruption mitigation (bounded, measured SPI run) ──────────────────
@dataclass(frozen=True, slots=True)
class MitigationRun:
    """Path-free summary of a shattered-pellet-injection mitigation run.

    Mirrors the deterministic summary of ``run_spi_mitigation``: the injected
    noble-gas dose per species, the post-injection effective charge, the plasma
    current at the end of the current quench, and the simulated sample count.

    Parameters
    ----------
    neon_quantity_mol, argon_quantity_mol, xenon_quantity_mol
        Injected noble-gas dose per species, in moles.
    z_eff
        Post-injection effective charge (>= 1).
    final_current_ma
        Plasma current at the end of the current quench, in MA.
    sample_count
        Number of simulated time samples.
    result_digest
        SHA-256 of the mitigation run summary.

    Raises
    ------
    ValueError
        If a dose is negative, ``z_eff`` is below one, the sample count is below one,
        or the digest is empty.
    """

    neon_quantity_mol: float
    argon_quantity_mol: float
    xenon_quantity_mol: float
    z_eff: float
    final_current_ma: float
    sample_count: int
    result_digest: str

    def __post_init__(self) -> None:
        """Validate the doses, effective charge, sample count, and digest."""
        if self.neon_quantity_mol < 0 or self.argon_quantity_mol < 0 or self.xenon_quantity_mol < 0:
            raise ValueError("MitigationRun doses must be non-negative")
        if self.z_eff < 1.0:
            raise ValueError("MitigationRun.z_eff must be >= 1")
        if self.sample_count < 1:
            raise ValueError("MitigationRun.sample_count must be >= 1")
        if not self.result_digest.strip():
            raise ValueError("MitigationRun.result_digest must be non-empty")


def disruption_mitigation_evidence(
    run: MitigationRun,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.disruption-mitigation.v1`` bundle (measured SPI run).

    The mitigation run is a low-order shattered-pellet-injection quench model
    (neutral-gas-shielding ablation, noble-gas radiative ``Z_eff``); it is real and
    deterministic but not a facility-validated disruption-mitigation outcome, so the
    claim is ``bounded-model`` / ``rejected``.

    Parameters
    ----------
    run
        The path-free mitigation run summary.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle (bounded-model).
    """
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/disruption-mitigation/{run.result_digest}", digest=run.result_digest)
    activity = ProvActivity(verb="mitigate", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(
        units={"dose": "mol", "Z_eff": "dimensionless", "current": "MA"},
        grid={},
    )
    validity = ValidityDomain(
        note=(
            "Low-order shattered-pellet-injection quench model (neutral-gas-shielding "
            "ablation, noble-gas radiative Z_eff); not a facility-validated "
            "disruption-mitigation outcome."
        )
    )
    return EvidenceBundle(
        schema=DISRUPTION_MITIGATION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.REJECTED,
            validity_domain=validity,
        ),
        freshness=Freshness.TRACEABLE_UNCHECKED,
        physical_contract=physical,
    )


# ── scenario simulation (bounded, measured replay rollout) ─────────────
@dataclass(frozen=True, slots=True)
class ScenarioSimulationRun:
    """Path-free summary of an integrated-scenario simulation rollout.

    Mirrors a ``ScenarioCouplingMetadata`` together with its audit pass flag (from
    ``audit_scenario_coupling``): the scenario, the digest of the exact config used,
    the step count and time window, the participating-module count, and whether the
    bounded replay-coupling audit passed.

    Parameters
    ----------
    scenario_name
        Name of the scenario.
    config_digest
        SHA-256 of the exact ``ScenarioConfig`` used.
    n_steps
        Number of simulated time steps.
    t_start_s, t_end_s
        Scenario time window in seconds (``t_end_s`` must exceed ``t_start_s``).
    module_count
        Number of physics/control modules in the coupling exchange.
    audit_passed
        Whether the bounded replay-coupling audit passed.

    Raises
    ------
    ValueError
        If a required string is empty, the step count is below one, the time window
        is empty, or the module count is below one.
    """

    scenario_name: str
    config_digest: str
    n_steps: int
    t_start_s: float
    t_end_s: float
    module_count: int
    audit_passed: bool

    def __post_init__(self) -> None:
        """Validate the identity, step count, time window, and module count."""
        if not self.scenario_name.strip() or not self.config_digest.strip():
            raise ValueError("ScenarioSimulationRun scenario_name and config_digest must be non-empty")
        if self.n_steps < 1:
            raise ValueError("ScenarioSimulationRun.n_steps must be >= 1")
        if self.t_end_s <= self.t_start_s:
            raise ValueError("ScenarioSimulationRun.t_end_s must exceed t_start_s")
        if self.module_count < 1:
            raise ValueError("ScenarioSimulationRun.module_count must be >= 1")


def scenario_simulation_evidence(
    run: ScenarioSimulationRun,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Build the ``studio.scenario-simulation.v1`` bundle (measured rollout).

    A scenario rollout is a closed-loop run on a low-order plant model checked by a
    bounded replay-coupling audit (finite, monotonic, conservation-bounded). A
    passing audit proves replay consistency, **not** facility validation — so even an
    audited, passing rollout stays ``bounded-model`` / ``rejected`` and does not
    render as validated; external trajectory validation is required for a facility
    claim.

    Parameters
    ----------
    run
        The path-free scenario rollout summary.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``measured`` schema-B bundle (bounded-model).
    """
    digest = canonical_digest(
        {
            "scenario_name": run.scenario_name,
            "config_digest": run.config_digest,
            "n_steps": run.n_steps,
            "audit_passed": run.audit_passed,
        }
    )
    entity = ProvEntity(entity_id=f"{STUDIO_ID}/scenario-simulation/{run.config_digest}", digest=digest)
    activity = ProvActivity(verb="simulate", studio=STUDIO_ID, started=started, ended=ended)
    agent = ProvAgent(studio_version=studio_version, operator=operator)
    physical = PhysicalContract(units={"time": "s"}, grid={"n_steps": run.n_steps, "modules": run.module_count})
    validity = ValidityDomain(
        note=(
            "Closed-loop scenario rollout on a low-order plant model with a bounded "
            "replay-coupling audit (finite, monotonic, conservation-bounded); not an "
            "external integrated-modelling or measured-discharge validation."
        )
    )
    return EvidenceBundle(
        schema=SCENARIO_SIMULATION_SCHEMA,
        entity=entity,
        activity=activity,
        agent=agent,
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.REJECTED,
            validity_domain=validity,
        ),
        freshness=Freshness.TRACEABLE_UNCHECKED,
        physical_contract=physical,
    )
