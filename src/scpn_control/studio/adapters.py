# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio live-emitter adapters
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Bridge CONTROL's live emitter outputs onto the studio EvidenceBundle mappers.

The mappers in :mod:`scpn_control.studio.evidence` take path-free result shapes so
they stay testable in isolation. These adapters connect the *real* CONTROL emitters
to them: a ``RealtimeEFIT.reconstruct`` result, an issued runtime safety
certificate, and a controller-latency measurement become schema-B EvidenceBundles
with no re-implementation of the honesty rules.

Where an emitter does not stamp a value the bundle needs — the proof engine and its
version are not recorded on a runtime safety certificate — the adapter takes it as
an explicit argument from the caller that ran the proof, rather than inventing one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .evidence import (
    ControllerLatencyResult,
    DisruptionPrediction,
    EfitReconstructionResult,
    EquilibriumAnalysis,
    MonitorSnapshot,
    ReplaySummary,
    SafetyCertificateResult,
    TraceabilityClaim,
    canonical_digest,
    controller_latency_evidence,
    disruption_prediction_evidence,
    efit_reconstruction_evidence,
    equilibrium_analysis_evidence,
    geometry_neutral_replay_evidence_bundle,
    phase_sync_monitor_evidence,
    physics_validation_evidence,
    safety_certificate_evidence,
)

if TYPE_CHECKING:
    from scpn_studio_platform.evidence import EvidenceBundle

    from scpn_control.control.realtime_efit import ReconstructionResult, ShapeParams
    from scpn_control.scpn.geometry_neutral_replay import GeometryNeutralReplayEvidence


def efit_evidence_from_reconstruction(
    result: ReconstructionResult,
    *,
    measurements: Mapping[str, Any],
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
) -> EvidenceBundle:
    """Map a live ``RealtimeEFIT.reconstruct`` result onto an evidence bundle.

    Parameters
    ----------
    result
        The reconstruction output (its ``shape.Ip_reconstructed``, ``chi_squared``,
        ``n_iterations`` and ``psi`` grid shape feed the bundle).
    measurements
        The diagnostic/configuration inputs the reconstruction ran on; hashed for
        the input provenance digest.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.
    host
        Optional host descriptor.

    Returns
    -------
    EvidenceBundle
        A ``studio.efit-reconstruction.v1`` bundle (bounded-model, not validated).
    """
    nz = int(result.psi.shape[0])
    nr = int(result.psi.shape[1])
    result_summary = {
        "ip_reconstructed_a": float(result.shape.Ip_reconstructed),
        "chi_squared": float(result.chi_squared),
        "n_iterations": int(result.n_iterations),
        "q95": float(result.shape.q95),
        "beta_pol": float(result.shape.beta_pol),
        "li": float(result.shape.li),
        "grid": [nz, nr],
    }
    src = EfitReconstructionResult(
        ip_reconstructed_a=float(result.shape.Ip_reconstructed),
        chi_squared=float(result.chi_squared),
        n_iterations=int(result.n_iterations),
        nr=nr,
        nz=nz,
        input_digest=canonical_digest(dict(measurements)),
        result_digest=canonical_digest(result_summary),
    )
    return efit_reconstruction_evidence(
        src,
        operator=operator,
        studio_version=studio_version,
        started=started,
        ended=ended,
        host=host,
    )


def safety_certificate_evidence_from_certificate(
    certificate: Mapping[str, Any],
    *,
    live_topology_sha256: str,
    checker: str,
    checker_version: str,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Map an issued runtime safety certificate onto a formally-proven bundle.

    Parameters
    ----------
    certificate
        A certificate from ``issue_runtime_safety_certificate`` — its
        ``binding.petri_topology_sha256`` is the proven subject, its
        ``formal_certificate_sha256`` the proof artifact, and ``checked_specs`` the
        proven obligations.
    live_topology_sha256
        SHA-256 of the live controller's Petri-net topology (e.g. from
        ``compute_petri_topology_digest`` on the deployed net). A mismatch with the
        proven topology degrades the claim and voids the proof.
    checker, checker_version
        The proof engine and its exact version. A runtime safety certificate does
        not stamp these, so the caller that ran the proof supplies them rather than
        the adapter inventing a value.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``studio.safety-certificate.v1`` bundle (formally-proven); admissible only
        when the live topology still matches the proven one.

    Raises
    ------
    KeyError
        If the certificate is missing the binding, topology, proof, or specs fields.
    """
    binding = certificate["binding"]
    checked_specs = list(certificate["checked_specs"])
    theorem_id = " ; ".join(checked_specs) if checked_specs else str(certificate["scope"])
    formal = certificate["formal_certificate"]
    non_vacuous = bool(formal.get("non_vacuous", False)) if isinstance(formal, Mapping) else False
    src = SafetyCertificateResult(
        theorem_id=theorem_id,
        checker=checker,
        checker_version=checker_version,
        proof_digest=str(certificate["formal_certificate_sha256"]),
        petri_topology_sha256=str(binding["petri_topology_sha256"]),
        live_topology_sha256=live_topology_sha256,
        result_digest=str(certificate["payload_sha256"]),
        non_vacuous=non_vacuous,
    )
    return safety_certificate_evidence(
        src,
        operator=operator,
        studio_version=studio_version,
        started=started,
        ended=ended,
    )


def controller_latency_evidence_from_measurement(
    measurement: Mapping[str, Any],
    *,
    controller: str,
    active_backend: str,
    reference_backend: str,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
    host: str | None = None,
) -> EvidenceBundle:
    """Map a controller-latency measurement onto a measured benchmark bundle.

    Parameters
    ----------
    measurement
        A latency record with ``p50_us``, ``p95_us``, ``p99_us`` and ``n`` (the
        sample count), as produced by the per-controller latency benchmark.
    controller
        Name of the controller benchmarked.
    active_backend
        The backend timed (e.g. ``"rust"``).
    reference_backend
        The reference backend it is compared against.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.
    host
        Optional host descriptor.

    Returns
    -------
    EvidenceBundle
        A ``studio.controller-latency.v1`` bundle (measured, bounded-model).

    Raises
    ------
    KeyError
        If the measurement is missing a percentile or the sample count.
    """
    p50 = float(measurement["p50_us"])
    p95 = float(measurement["p95_us"])
    p99 = float(measurement["p99_us"])
    sample_count = int(measurement["n"])
    src = ControllerLatencyResult(
        controller=controller,
        active_backend=active_backend,
        reference_backend=reference_backend,
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        sample_count=sample_count,
        input_digest=canonical_digest({"controller": controller, "active_backend": active_backend}),
        result_digest=canonical_digest({"p50_us": p50, "p95_us": p95, "p99_us": p99, "n": sample_count}),
    )
    return controller_latency_evidence(
        src,
        operator=operator,
        studio_version=studio_version,
        started=started,
        ended=ended,
        host=host,
    )


def physics_validation_evidences_from_registry(
    entries: Iterable[Mapping[str, Any]],
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> tuple[EvidenceBundle, ...]:
    """Map physics_traceability registry entries onto physics-validation bundles.

    Each entry of ``validation/physics_traceability.json`` becomes a
    ``studio.physics-validation.v1`` bundle carrying its fidelity status on the
    claim lattice, its admission, and its qualitative ``validity_domain`` as a prose
    note. For an ``external_dependency_blocked`` entry the blocking dependency is
    taken from the first ``required_actions`` item (the lattice requires one).

    Parameters
    ----------
    entries
        The registry entries (each a mapping with ``component``, ``module_path``,
        ``fidelity_status``, ``public_claim_allowed``, ``validity_domain`` and, for
        blocked entries, ``required_actions``).
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    tuple[EvidenceBundle, ...]
        One bundle per entry, in input order.

    Raises
    ------
    KeyError
        If an entry is missing a required field.
    """
    bundles: list[EvidenceBundle] = []
    for entry in entries:
        required_actions = list(entry.get("required_actions") or [])
        blocking = required_actions[0] if required_actions else None
        claim = TraceabilityClaim(
            component=str(entry["component"]),
            module_path=str(entry["module_path"]),
            fidelity_status=str(entry["fidelity_status"]),
            public_claim_allowed=bool(entry["public_claim_allowed"]),
            validity_domain=str(entry["validity_domain"]),
            blocking_dependency=blocking,
        )
        bundles.append(
            physics_validation_evidence(
                claim,
                operator=operator,
                studio_version=studio_version,
                started=started,
                ended=ended,
            )
        )
    return tuple(bundles)


def replay_evidence_from_geometry_neutral(
    evidence: GeometryNeutralReplayEvidence,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Map a ``GeometryNeutralReplayEvidence`` onto a replay bundle.

    Parameters
    ----------
    evidence
        The geometry-neutral replay evidence object.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``studio.geometry-neutral-replay.v1`` bundle.
    """
    summary = ReplaySummary(
        scenario_digest=evidence.scenario_digest,
        trace_digest=evidence.trace_digest,
        result_digest=evidence.payload_sha256,
        max_abs_current_a=float(evidence.max_abs_current_A),
        p95_latency_us=float(evidence.p95_latency_us),
        device_claim_allowed=bool(evidence.device_claim_allowed),
    )
    return geometry_neutral_replay_evidence_bundle(
        summary, operator=operator, studio_version=studio_version, started=started, ended=ended
    )


def phase_sync_monitor_evidence_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Map a ``RealtimeMonitor.tick`` snapshot onto a monitor bundle.

    Parameters
    ----------
    snapshot
        The dashboard snapshot dict (``tick``, ``R_global``, ``lambda_exp``,
        ``guard_approved``, ``latency_us``).
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``studio.phase-sync-monitor.v1`` bundle.

    Raises
    ------
    KeyError
        If a required snapshot field is missing.
    """
    snap = MonitorSnapshot(
        tick=int(snapshot["tick"]),
        r_global=float(snapshot["R_global"]),
        lambda_exp=float(snapshot["lambda_exp"]),
        guard_approved=bool(snapshot["guard_approved"]),
        latency_us=float(snapshot["latency_us"]),
    )
    return phase_sync_monitor_evidence(
        snap, operator=operator, studio_version=studio_version, started=started, ended=ended
    )


def disruption_prediction_evidence_from_risk(
    risk: float,
    *,
    observables: Mapping[str, float],
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Map a ``predict_disruption_risk`` output onto a prediction bundle.

    Parameters
    ----------
    risk
        The predicted disruption risk in ``[0, 1]``.
    observables
        The toroidal-asymmetry observables the prediction used.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``studio.disruption-prediction.v1`` bundle (validation-gap, fixed-weight).
    """
    prediction = DisruptionPrediction(
        risk=float(risk),
        observable_count=len(observables),
        result_digest=canonical_digest({"risk": float(risk), "observables": dict(observables)}),
    )
    return disruption_prediction_evidence(
        prediction, operator=operator, studio_version=studio_version, started=started, ended=ended
    )


def equilibrium_analysis_evidence_from_shape(
    shape: ShapeParams,
    *,
    operator: str,
    studio_version: str,
    started: str,
    ended: str,
) -> EvidenceBundle:
    """Map a ``ShapeParams`` onto an equilibrium-analysis bundle.

    Parameters
    ----------
    shape
        The macroscopic shape parameters derived from a reconstruction.
    operator
        Opaque identity of the operator/tenant.
    studio_version
        Version of the CONTROL studio.
    started, ended
        ISO-8601 start/end timestamps.

    Returns
    -------
    EvidenceBundle
        A ``studio.equilibrium-analysis.v1`` bundle (bounded-model).
    """
    analysis = EquilibriumAnalysis(
        r0=float(shape.R0),
        a=float(shape.a),
        kappa=float(shape.kappa),
        q95=float(shape.q95),
        beta_pol=float(shape.beta_pol),
        li=float(shape.li),
        result_digest=canonical_digest(
            {"R0": float(shape.R0), "a": float(shape.a), "kappa": float(shape.kappa), "q95": float(shape.q95)}
        ),
    )
    return equilibrium_analysis_evidence(
        analysis, operator=operator, studio_version=studio_version, started=started, ended=ended
    )
