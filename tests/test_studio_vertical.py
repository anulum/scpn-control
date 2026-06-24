# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio vertical tests
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Tests for CONTROL's studio vertical built on the platform SDK.

The suite checks three things: the verb taxonomy maps onto the platform core spine
plus CONTROL's domain-distinctive verbs with the right gating attributes; the
capability manifest is content-addressed and deterministic; and — the crux — each
evidence mapper produces the honest rendering the shared lattice dictates (a
bounded reconstruction never renders as validated, a held formal proof does and is
voided by subject drift, a measured benchmark stays bounded).

Skips cleanly when the optional ``scpn-studio-platform`` SDK is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import EvidenceKind  # noqa: E402
from scpn_studio_platform.verbs import SafetyTier, SideEffect, TimingClass  # noqa: E402

from scpn_control.studio import (  # noqa: E402
    CONTROL_VERBS,
    STUDIO_ID,
    ControllerLatencyResult,
    EfitReconstructionResult,
    SafetyCertificateResult,
    build_manifest,
    canonical_digest,
    controller_latency_evidence,
    core_verbs,
    declared_surface,
    domain_verbs,
    efit_reconstruction_evidence,
    evidence_schemas,
    safety_certificate_evidence,
)

_TS = {"started": "2026-06-23T00:00:00Z", "ended": "2026-06-23T00:00:01Z"}
_WHO = {"operator": "opaque:tenant-1", "studio_version": "0.test"}


def _efit() -> EfitReconstructionResult:
    return EfitReconstructionResult(
        ip_reconstructed_a=1.5e7,
        chi_squared=4.38e-9,
        n_iterations=7,
        nr=33,
        nz=33,
        input_digest="a" * 64,
        result_digest="b" * 64,
    )


def _cert(*, live: str = "d" * 64) -> SafetyCertificateResult:
    return SafetyCertificateResult(
        theorem_id="ctl.safety.bounded-halt",
        checker="z3",
        checker_version="4.13.0",
        proof_digest="c" * 64,
        petri_topology_sha256="d" * 64,
        live_topology_sha256=live,
        result_digest="e" * 64,
        non_vacuous=True,
    )


def _latency() -> ControllerLatencyResult:
    return ControllerLatencyResult(
        controller="h_infinity",
        active_backend="rust",
        reference_backend="python",
        p50_us=5.05,
        p95_us=6.11,
        p99_us=6.11,
        sample_count=200,
        input_digest="g" * 64,
        result_digest="h" * 64,
    )


# ── verbs ──────────────────────────────────────────────────────────────
def test_control_advertises_twelve_verbs() -> None:
    assert len(CONTROL_VERBS) == 12
    assert STUDIO_ID == "scpn-control"


def test_core_and_domain_verb_split() -> None:
    assert {v.name for v in core_verbs()} == {
        "reconstruct",
        "simulate",
        "analyse",
        "validate",
        "benchmark",
        "replay",
    }
    assert {v.name for v in domain_verbs()} == {
        "regulate",
        "certify",
        "predict",
        "mitigate",
        "monitor",
        "verify",
    }
    assert len(core_verbs()) + len(domain_verbs()) == len(CONTROL_VERBS)


def test_regulate_is_realtime_live_hardware_gated() -> None:
    regulate = next(v for v in CONTROL_VERBS if v.name == "regulate")
    assert regulate.timing.timing_class is TimingClass.REALTIME
    assert regulate.timing.deadline_us == pytest.approx(5.0)
    assert regulate.side_effect is SideEffect.LIVE_HARDWARE
    assert regulate.requires_live_hardware_gate is True
    assert regulate.safety_tier is SafetyTier.CERTIFIED


def test_mitigate_is_live_hardware_and_certify_is_certified_read_only() -> None:
    mitigate = next(v for v in CONTROL_VERBS if v.name == "mitigate")
    certify = next(v for v in CONTROL_VERBS if v.name == "certify")
    assert mitigate.requires_live_hardware_gate is True
    assert certify.safety_tier is SafetyTier.CERTIFIED
    assert certify.side_effect is SideEffect.READ_ONLY
    assert certify.requires_live_hardware_gate is False


def test_every_verb_produces_exactly_one_schema() -> None:
    for verb in CONTROL_VERBS:
        assert len(verb.produces) == 1
        assert verb.produces[0].startswith("studio.")
        assert verb.produces[0].endswith(".v1")


def test_evidence_schemas_are_sorted_unique_and_complete() -> None:
    schemas = evidence_schemas()
    assert list(schemas) == sorted(set(schemas))
    assert len(schemas) == len(CONTROL_VERBS)


# ── manifest ───────────────────────────────────────────────────────────
def test_manifest_identity_and_surface() -> None:
    manifest = build_manifest()
    assert manifest.studio == "scpn-control"
    assert len(manifest.verbs) == 12
    assert len(manifest.evidence_types) == 12
    assert manifest.content_digest.startswith("sha256:")
    assert manifest.ui_module is not None


def test_manifest_content_digest_is_deterministic() -> None:
    assert build_manifest().content_digest == build_manifest().content_digest


def test_manifest_version_override_changes_stamp_not_digest() -> None:
    pinned = build_manifest(studio_version="9.9.9")
    assert pinned.studio_version == "9.9.9"
    # The content digest covers verbs + schemas, not the studio version stamp.
    assert pinned.content_digest == build_manifest().content_digest


def test_declared_surface_covers_every_verb_and_schema_list() -> None:
    surface = declared_surface()
    for verb in CONTROL_VERBS:
        assert f"verb/{verb.name}" in surface
    assert "evidence/schemas" in surface


# ── EFIT reconstruction evidence (bounded-model) ───────────────────────
def test_efit_evidence_does_not_render_as_validated() -> None:
    bundle = efit_reconstruction_evidence(_efit(), **_WHO, **_TS)
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.physical_contract is not None
    assert bundle.physical_contract.units["psi"] == "Wb"
    assert bundle.schema == "studio.efit-reconstruction.v1"


def test_efit_evidence_carries_prose_validity_domain() -> None:
    bundle = efit_reconstruction_evidence(_efit(), **_WHO, **_TS)
    validity = bundle.claim_boundary.validity_domain
    assert validity is not None
    assert validity.note is not None
    # CONTROL's validity_domain is qualitative prose, not fabricated numeric ranges.
    assert "closure-validated" in validity.note
    assert validity.ranges == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_iterations": 0},
        {"nr": 0},
        {"nz": 0},
        {"input_digest": "  "},
        {"result_digest": ""},
    ],
)
def test_efit_result_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    base = {
        "ip_reconstructed_a": 1.5e7,
        "chi_squared": 1e-9,
        "n_iterations": 7,
        "nr": 33,
        "nz": 33,
        "input_digest": "a" * 64,
        "result_digest": "b" * 64,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        EfitReconstructionResult(**base)  # type: ignore[arg-type]


# ── safety certificate evidence (formally-proven) ──────────────────────
def test_held_certificate_renders_as_validated_and_is_proven() -> None:
    bundle = safety_certificate_evidence(_cert(), **_WHO, **_TS)
    assert bundle.renders_as_validated is True
    assert bundle.evidence_kind is EvidenceKind.FORMALLY_PROVEN
    assert len(bundle.formal_certificates) == 1
    cert = bundle.formal_certificates[0]
    assert cert.checker == "z3"
    assert cert.covers("d" * 64) is True
    assert bundle.proof_voided_by("f" * 64) is True


def test_drifted_certificate_is_not_admissible() -> None:
    bundle = safety_certificate_evidence(_cert(live="f" * 64), **_WHO, **_TS)
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.FORMALLY_PROVEN


def test_certificate_topology_matches_property() -> None:
    assert _cert().topology_matches is True
    assert _cert(live="f" * 64).topology_matches is False


@pytest.mark.parametrize(
    "field",
    [
        "theorem_id",
        "checker",
        "checker_version",
        "proof_digest",
        "petri_topology_sha256",
        "live_topology_sha256",
        "result_digest",
    ],
)
def test_certificate_result_rejects_empty_fields(field: str) -> None:
    base = {
        "theorem_id": "t",
        "checker": "z3",
        "checker_version": "4.13.0",
        "proof_digest": "c" * 64,
        "petri_topology_sha256": "d" * 64,
        "live_topology_sha256": "d" * 64,
        "result_digest": "e" * 64,
    }
    base[field] = "  "
    with pytest.raises(ValueError):
        SafetyCertificateResult(**base)  # type: ignore[arg-type]


# ── controller latency evidence (measured benchmark) ───────────────────
def test_latency_evidence_is_measured_and_bounded() -> None:
    bundle = controller_latency_evidence(_latency(), **_WHO, **_TS)
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.numeric_provenance is not None
    assert bundle.numeric_provenance.active_backend == "rust"
    assert bundle.numeric_provenance.parity == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p50_us": 0.0},
        {"p95_us": -1.0},
        {"p99_us": 0.0},
        {"sample_count": 0},
        {"input_digest": ""},
        {"result_digest": "  "},
    ],
)
def test_latency_result_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    base = {
        "controller": "h_infinity",
        "active_backend": "rust",
        "reference_backend": "python",
        "p50_us": 5.05,
        "p95_us": 6.11,
        "p99_us": 6.11,
        "sample_count": 200,
        "input_digest": "g" * 64,
        "result_digest": "h" * 64,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        ControllerLatencyResult(**base)  # type: ignore[arg-type]


# ── canonical digest ───────────────────────────────────────────────────
def test_canonical_digest_is_order_independent_and_stable() -> None:
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    assert len(canonical_digest({"x": 1})) == 64


def test_canonical_digest_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_digest({"x": float("nan")})


def test_parity_refutation_evidence_is_a_promoted_negative_finding() -> None:
    from scpn_studio_platform.evidence import AdmissionDecision, ClaimStatus

    from scpn_control.studio.evidence import ParityRefutationResult, parity_refutation_evidence

    bundle = parity_refutation_evidence(
        ParityRefutationResult(
            solver_method="sor",
            grid="65x65 Solov'ev R0=1.7 a=0.5 B0=2.0 Ip=1.0MA",
            pearson_r=0.999,
            interior_l2_rel=0.06,
            gs_residual_plateau=4.0,
            target_rtol=1e-3,
            result_digest="a" * 64,
            raw_reference="PARITY-1",
        ),
        operator="op",
        studio_version="test",
        started="2026-06-25T00:00:00Z",
        ended="2026-06-25T00:00:01Z",
    )
    # The honest pair: a TESTED-and-refuted parity, not an untested validation gap.
    assert bundle.evidence_kind is EvidenceKind.FALSIFIED
    assert bundle.claim_boundary.status is ClaimStatus.REFUTED
    assert bundle.claim_boundary.admission is AdmissionDecision.REJECTED
    # A refuted parity can NEVER render as a validated parity claim (LOCK-4).
    assert bundle.renders_as_validated is False
    # The raw counts travel with the refutation so a consumer sees HOW it failed.
    parity = bundle.numeric_provenance.parity[0]
    assert parity.passed is False
    assert parity.max_error == 0.06
    assert bundle.numeric_provenance.convergence.residual == 4.0
