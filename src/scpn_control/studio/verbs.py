# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio verbs
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""The SCPN-CONTROL studio's verbs, expressed on the locked platform contract.

CONTROL is the control-grade integration laboratory of the SCPN ecosystem, so its
verbs are control actions and evidence/admission surfaces rather than solver math.
Six are drawn from the shared :data:`scpn_studio_platform.verbs.CORE_VERBS` spine
(``reconstruct``, ``simulate``, ``analyse``, ``validate``, ``benchmark``,
``replay``); five are domain-distinctive to a control studio (``regulate``,
``certify``, ``predict``, ``mitigate``, ``monitor``).

Each is declared as a :class:`scpn_studio_platform.verbs.Verb` carrying the
attribute contract the Hub federates and gates against. Two declarations exercise
CONTROL's distinctive contributions to the v1 contract directly:

- ``regulate`` is the only verb in the ecosystem so far that is both ``realtime``
  (a hard ``deadline_us`` from the Rust control cycle) and ``live-hardware`` (it
  drives real actuators through HIL/CODAC/EPICS), so the Hub must hard-gate it per
  tenant.
- ``certify`` is a ``certified``-tier verb whose evidence is machine-checked
  (``formally-proven`` Petri/Z3/Lean certificates on the bundle), not a measurement.
"""

from __future__ import annotations

from scpn_studio_platform.verbs import (
    Fidelity,
    SafetyTier,
    Schedulability,
    SideEffect,
    Timing,
    TimingClass,
    Verb,
)

STUDIO_ID = "scpn-control"
"""The studio identifier this vertical implements (also the federation name)."""

#: Hard control-cycle deadline advertised by ``regulate``, in microseconds — the
#: published native control-cycle P50 (CI EPYC 7763). The realised envelope is
#: re-measured per run and recorded on each controller-run evidence bundle.
CONTROL_CYCLE_DEADLINE_US = 5.0

# ── evidence schema names this studio emits (studio.*.v1) ──────────────
EFIT_RECONSTRUCTION_SCHEMA = "studio.efit-reconstruction.v1"
CONTROLLER_RUN_SCHEMA = "studio.controller-run.v1"
SAFETY_CERTIFICATE_SCHEMA = "studio.safety-certificate.v1"
CONTROLLER_LATENCY_SCHEMA = "studio.controller-latency.v1"
PHYSICS_VALIDATION_SCHEMA = "studio.physics-validation.v1"
GEOMETRY_NEUTRAL_REPLAY_SCHEMA = "studio.geometry-neutral-replay.v1"
DISRUPTION_PREDICTION_SCHEMA = "studio.disruption-prediction.v1"
DISRUPTION_MITIGATION_SCHEMA = "studio.disruption-mitigation.v1"
PHASE_SYNC_MONITOR_SCHEMA = "studio.phase-sync-monitor.v1"
SCENARIO_SIMULATION_SCHEMA = "studio.scenario-simulation.v1"
EQUILIBRIUM_ANALYSIS_SCHEMA = "studio.equilibrium-analysis.v1"


# ── core-spine verbs ───────────────────────────────────────────────────
RECONSTRUCT = Verb(
    name="reconstruct",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.INTERACTIVE),
    fidelity=Fidelity.REDUCED_ORDER,
    produces=(EFIT_RECONSTRUCTION_SCHEMA,),
    backends=("python", "rust"),
)
"""Reconstruct an equilibrium from magnetic diagnostics (real-time EFIT-lite)."""

SIMULATE = Verb(
    name="simulate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.REDUCED_ORDER,
    produces=(SCENARIO_SIMULATION_SCHEMA,),
    backends=("python", "rust"),
)
"""Roll out a closed-loop control scenario on a low-order plant model."""

ANALYSE = Verb(
    name="analyse",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(EQUILIBRIUM_ANALYSIS_SCHEMA,),
    backends=("python",),
)
"""Extract macroscopic shape and stability descriptors from an equilibrium."""

VALIDATE = Verb(
    name="validate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(PHYSICS_VALIDATION_SCHEMA,),
    backends=("python",),
)
"""Check a bounded physics claim against its traceability reference."""

BENCHMARK = Verb(
    name="benchmark",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    produces=(CONTROLLER_LATENCY_SCHEMA,),
    backends=("python", "rust"),
)
"""Measure per-controller and native-handoff control-cycle latency."""

REPLAY = Verb(
    name="replay",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(GEOMETRY_NEUTRAL_REPLAY_SCHEMA,),
    backends=("python",),
)
"""Replay a recorded geometry-neutral control run with provenance."""


# ── domain-distinctive verbs ───────────────────────────────────────────
REGULATE = Verb(
    name="regulate",
    safety_tier=SafetyTier.CERTIFIED,
    side_effect=SideEffect.LIVE_HARDWARE,
    timing=Timing(
        TimingClass.REALTIME,
        deadline_us=CONTROL_CYCLE_DEADLINE_US,
        schedulability=Schedulability.DECLARED,
    ),
    fidelity=Fidelity.REDUCED_ORDER,
    produces=(CONTROLLER_RUN_SCHEMA,),
    backends=("python", "rust"),
)
"""Run a controller loop against a plant under explicit constraints (control/regulate)."""

CERTIFY = Verb(
    name="certify",
    safety_tier=SafetyTier.CERTIFIED,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(SAFETY_CERTIFICATE_SCHEMA,),
    backends=("python",),
)
"""Issue a machine-checked runtime safety certificate (admit/certify)."""

PREDICT = Verb(
    name="predict",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.INTERACTIVE),
    fidelity=Fidelity.REDUCED_ORDER,
    produces=(DISRUPTION_PREDICTION_SCHEMA,),
    backends=("python",),
)
"""Predict disruption risk from toroidal-asymmetry observables."""

MITIGATE = Verb(
    name="mitigate",
    safety_tier=SafetyTier.CERTIFIED,
    side_effect=SideEffect.LIVE_HARDWARE,
    timing=Timing(TimingClass.INTERACTIVE),
    produces=(DISRUPTION_MITIGATION_SCHEMA,),
    backends=("python",),
)
"""Actuate shattered-pellet / disruption mitigation on the coil set."""

MONITOR = Verb(
    name="monitor",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.INTERACTIVE),
    produces=(PHASE_SYNC_MONITOR_SCHEMA,),
    backends=("python",),
)
"""Stream live phase-sync coherence and guard-halt telemetry."""


CONTROL_VERBS: tuple[Verb, ...] = (
    RECONSTRUCT,
    SIMULATE,
    ANALYSE,
    VALIDATE,
    BENCHMARK,
    REPLAY,
    REGULATE,
    CERTIFY,
    PREDICT,
    MITIGATE,
    MONITOR,
)
"""Every verb the CONTROL studio advertises, in manifest order."""


def core_verbs() -> tuple[Verb, ...]:
    """Return the CONTROL verbs drawn from the shared core spine.

    Returns
    -------
    tuple[Verb, ...]
        The subset of :data:`CONTROL_VERBS` whose names are in the platform
        :data:`scpn_studio_platform.verbs.CORE_VERBS`.
    """
    return tuple(v for v in CONTROL_VERBS if v.is_core)


def domain_verbs() -> tuple[Verb, ...]:
    """Return the CONTROL verbs distinctive to a control studio.

    Returns
    -------
    tuple[Verb, ...]
        The subset of :data:`CONTROL_VERBS` not present in the platform core spine
        (``regulate``, ``certify``, ``predict``, ``mitigate``, ``monitor``).
    """
    return tuple(v for v in CONTROL_VERBS if not v.is_core)


def evidence_schemas() -> tuple[str, ...]:
    """Return the distinct evidence schemas every CONTROL verb emits.

    Returns
    -------
    tuple[str, ...]
        The sorted, de-duplicated union of each verb's ``produces`` schemas.
    """
    schemas = {schema for verb in CONTROL_VERBS for schema in verb.produces}
    return tuple(sorted(schemas))
