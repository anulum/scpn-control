# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio panel feed
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Serialise the CONTROL studio surface into a wire feed the panel reads live.

The federated panel must not hard-code CONTROL's verbs and claims — a second copy
drifts from the Python contract. This module emits a JSON-safe *studio feed*: the
manifest's verbs reduced to the panel's rendering fields, plus a set of live
:class:`scpn_studio_platform.evidence.EvidenceBundle` claims reduced to their
schema, claim-boundary status, admission and modality. The panel reads the same
honesty grading the Python vertical emits, with no second source of truth.

The serialisers (:func:`verb_summary`, :func:`claim_summary`, :func:`studio_feed`)
take whatever bundles the serving runtime has produced. :func:`representative_feed`
builds one bundle per evidence schema from the real mappers, so the feed is complete
and self-describing even before a live run has emitted anything; a serving runtime
substitutes live bundles for the representative set.

The wire format is snake_case (matching the rest of the platform JSON); the panel's
loader narrows it to its own camelCase domain types at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, cast

from scpn_studio_platform.evidence import EvidenceBundle
from scpn_studio_platform.verbs import Verb

from .evidence import (
    ControllerLatencyResult,
    ControllerRunResult,
    DisruptionPrediction,
    EfitReconstructionResult,
    EquilibriumAnalysis,
    MitigationRun,
    MonitorSnapshot,
    ReplaySummary,
    SafetyCertificateResult,
    ScenarioSimulationRun,
    TraceabilityClaim,
    canonical_digest,
    controller_latency_evidence,
    controller_run_evidence,
    disruption_mitigation_evidence,
    disruption_prediction_evidence,
    efit_reconstruction_evidence,
    equilibrium_analysis_evidence,
    geometry_neutral_replay_evidence_bundle,
    phase_sync_monitor_evidence,
    physics_validation_evidence,
    safety_certificate_evidence,
    scenario_simulation_evidence,
)
from .manifest import STUDIO_VERSION, build_manifest
from .verbs import CONTROL_VERBS, STUDIO_ID

#: Wire schema of the studio feed document the panel reads.
FEED_SCHEMA = "studio.control-feed.v1"

#: Representative ISO-8601 window stamped on the representative feed (no hidden
#: clock — the serving runtime stamps live bundles with real timestamps).
_REPRESENTATIVE_STARTED = "2026-06-23T00:00:00Z"
_REPRESENTATIVE_ENDED = "2026-06-23T00:00:01Z"
_REPRESENTATIVE_OPERATOR = "opaque:representative"


def verb_summary(verb: Verb) -> dict[str, object]:
    """Reduce a platform :class:`Verb` to the panel's verb-rendering fields.

    Parameters
    ----------
    verb
        A CONTROL verb.

    Returns
    -------
    dict[str, object]
        The verb's ``name``, ``safety_tier``, ``side_effect`` and ``timing_class``
        (all wire strings from :meth:`Verb.to_dict`), ``domain_distinctive`` (true
        when the verb is not in the core spine), and ``deadline_us`` only when the
        verb declares a realtime deadline.
    """
    raw = verb.to_dict()
    timing = cast("dict[str, Any]", raw["timing"])
    summary: dict[str, object] = {
        "name": raw["verb"],
        "safety_tier": raw["safety_tier"],
        "side_effect": raw["side_effect"],
        "timing_class": timing["class"],
        "domain_distinctive": not verb.is_core,
    }
    deadline = timing.get("deadline_us")
    if deadline is not None:
        summary["deadline_us"] = deadline
    return summary


def claim_summary(bundle: EvidenceBundle) -> dict[str, str]:
    """Reduce an :class:`EvidenceBundle` to the panel's claim-rendering fields.

    Parameters
    ----------
    bundle
        A schema-B evidence bundle.

    Returns
    -------
    dict[str, str]
        The bundle's ``schema`` and its claim ``status`` / ``admission`` / ``kind``
        as wire strings — exactly what the panel needs to apply the honesty rule
        (validated only when reference-validated and admitted).
    """
    boundary = bundle.claim_boundary
    return {
        "schema": bundle.schema,
        "status": boundary.status.value,
        "admission": boundary.admission.value,
        "kind": bundle.evidence_kind.value,
    }


def studio_feed(
    bundles: Iterable[EvidenceBundle],
    *,
    studio_version: str = STUDIO_VERSION,
    verbs: Iterable[Verb] = CONTROL_VERBS,
    content_digest: str | None = None,
) -> dict[str, object]:
    """Assemble the studio feed document from verbs and live claim bundles.

    Parameters
    ----------
    bundles
        The evidence bundles the serving runtime has produced.
    studio_version
        The studio version to stamp; defaults to the installed package version.
    verbs
        The verbs to advertise; defaults to every CONTROL verb in manifest order.
    content_digest
        The manifest content digest to stamp; computed from the manifest when not
        supplied so the feed and manifest always agree.

    Returns
    -------
    dict[str, object]
        A JSON-safe feed: ``feed_schema``, ``studio``, ``studio_version``,
        ``content_digest``, ``verbs`` and ``claims``.
    """
    digest = (
        content_digest if content_digest is not None else build_manifest(studio_version=studio_version).content_digest
    )
    return {
        "feed_schema": FEED_SCHEMA,
        "studio": STUDIO_ID,
        "studio_version": studio_version,
        "content_digest": digest,
        "verbs": [verb_summary(verb) for verb in verbs],
        "claims": [claim_summary(bundle) for bundle in bundles],
    }


def representative_bundles(
    *,
    operator: str = _REPRESENTATIVE_OPERATOR,
    studio_version: str = STUDIO_VERSION,
    started: str = _REPRESENTATIVE_STARTED,
    ended: str = _REPRESENTATIVE_ENDED,
) -> tuple[EvidenceBundle, ...]:
    """Build one representative evidence bundle per CONTROL evidence schema.

    Each bundle is produced by the *real* evidence mapper on a representative
    path-free input, so the feed shows the same honesty grading the vertical emits: a
    held safety certificate renders validated, the fixed-weight disruption predictor
    lands on validation-gap, and every other claim is bounded. The set is
    representative, not a live-hardware capture — a serving runtime substitutes the
    bundles it actually produced.

    Parameters
    ----------
    operator
        Opaque identity to stamp on the representative bundles.
    studio_version
        The studio version to stamp.
    started, ended
        ISO-8601 window to stamp (representative; no hidden clock).

    Returns
    -------
    tuple[EvidenceBundle, ...]
        One bundle per evidence schema, in manifest order.
    """
    who = {"operator": operator, "studio_version": studio_version, "started": started, "ended": ended}
    efit = efit_reconstruction_evidence(
        EfitReconstructionResult(
            ip_reconstructed_a=1.5e7,
            chi_squared=4.38e-9,
            n_iterations=7,
            nr=33,
            nz=33,
            input_digest=canonical_digest({"measurements": "iter-baseline"}),
            result_digest=canonical_digest({"efit": "iter-baseline"}),
        ),
        **who,
    )
    certificate = safety_certificate_evidence(
        SafetyCertificateResult(
            theorem_id="AG(no_overflow) ; AF(safe_shutdown)",
            checker="z3",
            checker_version="4.13.0",
            proof_digest=canonical_digest({"proof": "runtime-safety"}),
            petri_topology_sha256=canonical_digest({"topology": "deployed"}),
            live_topology_sha256=canonical_digest({"topology": "deployed"}),
            result_digest=canonical_digest({"certificate": "runtime-safety"}),
            non_vacuous=True,
        ),
        **who,
    )
    latency = controller_latency_evidence(
        ControllerLatencyResult(
            controller="h_infinity",
            active_backend="rust",
            reference_backend="python",
            p50_us=5.05,
            p95_us=6.11,
            p99_us=6.40,
            sample_count=200,
            input_digest=canonical_digest({"benchmark": "h_infinity"}),
            result_digest=canonical_digest({"latency": "h_infinity"}),
        ),
        **who,
    )
    validation = physics_validation_evidence(
        TraceabilityClaim(
            component="real-time EFIT-lite reconstruction",
            module_path="scpn_control/control/realtime_efit.py",
            fidelity_status="bounded_model",
            public_claim_allowed=False,
            validity_domain=(
                "Closure-validated against synthetic diagnostics only; not facility-grade EFIT/P-EFIT validation."
            ),
        ),
        **who,
    )
    replay = geometry_neutral_replay_evidence_bundle(
        ReplaySummary(
            scenario_digest=canonical_digest({"scenario": "geometry-neutral"}),
            trace_digest=canonical_digest({"trace": "geometry-neutral"}),
            result_digest=canonical_digest({"replay": "geometry-neutral"}),
            max_abs_current_a=1.2e4,
            p95_latency_us=6.1,
            device_claim_allowed=False,
        ),
        **who,
    )
    monitor = phase_sync_monitor_evidence(
        MonitorSnapshot(tick=10, r_global=0.92, lambda_exp=-0.01, guard_approved=True, latency_us=4.2),
        **who,
    )
    prediction = disruption_prediction_evidence(
        DisruptionPrediction(
            risk=0.30,
            observable_count=4,
            result_digest=canonical_digest({"prediction": "toroidal-asymmetry"}),
        ),
        **who,
    )
    equilibrium = equilibrium_analysis_evidence(
        EquilibriumAnalysis(
            r0=1.7,
            a=0.5,
            kappa=1.8,
            q95=3.5,
            beta_pol=0.9,
            li=0.8,
            result_digest=canonical_digest({"equilibrium": "iter-baseline"}),
        ),
        **who,
    )
    controller_run = controller_run_evidence(
        ControllerRunResult(
            controller="surrogate_mpc",
            n_steps=60,
            mean_tracking_error=0.12,
            max_abs_action=1.8,
            max_abs_coil_current=28.0,
            result_digest=canonical_digest({"controller_run": "surrogate_mpc"}),
        ),
        **who,
    )
    mitigation = disruption_mitigation_evidence(
        MitigationRun(
            neon_quantity_mol=0.1,
            argon_quantity_mol=0.0,
            xenon_quantity_mol=0.0,
            z_eff=2.4,
            final_current_ma=3.1,
            sample_count=5000,
            result_digest=canonical_digest({"mitigation": "spi"}),
        ),
        **who,
    )
    scenario = scenario_simulation_evidence(
        ScenarioSimulationRun(
            scenario_name="iter_baseline",
            config_digest=canonical_digest({"scenario_config": "iter_baseline"}),
            n_steps=100,
            t_start_s=0.0,
            t_end_s=100.0,
            module_count=3,
            audit_passed=True,
        ),
        **who,
    )
    return (
        efit,
        certificate,
        latency,
        validation,
        replay,
        monitor,
        prediction,
        equilibrium,
        controller_run,
        mitigation,
        scenario,
    )


def representative_feed(
    *,
    operator: str = _REPRESENTATIVE_OPERATOR,
    studio_version: str = STUDIO_VERSION,
    started: str = _REPRESENTATIVE_STARTED,
    ended: str = _REPRESENTATIVE_ENDED,
) -> dict[str, object]:
    """Build the complete representative studio feed.

    Parameters
    ----------
    operator
        Opaque identity to stamp on the representative bundles.
    studio_version
        The studio version to stamp.
    started, ended
        ISO-8601 window to stamp (representative).

    Returns
    -------
    dict[str, object]
        The feed document with every verb and one claim per evidence schema, its
        ``content_digest`` taken from the manifest so feed and manifest agree.
    """
    bundles = representative_bundles(operator=operator, studio_version=studio_version, started=started, ended=ended)
    digest = build_manifest(studio_version=studio_version).content_digest
    return studio_feed(bundles, studio_version=studio_version, content_digest=digest)


def render_feed_json() -> str:
    """Render the representative feed as deterministic, indented JSON.

    Returns
    -------
    str
        The representative feed as sorted-key, two-space-indented JSON with a
        trailing newline — the on-disk artefact the standalone panel fetches.
    """
    return json.dumps(representative_feed(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    import sys

    sys.stdout.write(render_feed_json())
