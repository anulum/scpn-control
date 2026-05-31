# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Debug Assistant Tests

"""Behavioural tests for local-first physics debugging assistance."""

from __future__ import annotations

import json

import pytest

from scpn_control.physics_debug import (
    HTTPChatProvider,
    PhysicsDebugAssistant,
    PhysicsDebugEvidence,
    PhysicsDebugGap,
    ProviderPolicy,
    validate_physics_debug_report,
)


def _cbc_evidence() -> PhysicsDebugEvidence:
    return PhysicsDebugEvidence(
        evidence_id="cbc-linear-dispersion",
        evidence_type="validation_gap",
        source="validation/reports/gk_cbc_linear.json",
        summary=(
            "local-dispersion path overpredicts the GENE CBC reference; "
            "SCPN_API_KEY=secret-token Authorization: Bearer hidden-token"
        ),
        sha256="a" * 64,
    )


def _cbc_gap() -> PhysicsDebugGap:
    return PhysicsDebugGap(
        gap_id="gk-cbc-linear-dispersion",
        severity="high",
        description="Local gyrokinetic dispersion closure exceeds the CBC reference band.",
        evidence_ids=("cbc-linear-dispersion",),
    )


def test_local_first_physics_debug_report_redacts_secrets_and_builds_falsifiable_campaign() -> None:
    def transport(payload: dict[str, object]) -> dict[str, object]:
        encoded = json.dumps(payload)
        assert "secret-token" not in encoded
        assert "hidden-token" not in encoded
        return {
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": "The local closure is using an inconsistent trapped-particle normalization.",
                    "falsification_test": "Run the same CBC case with trapped-particle weight set to zero and require the growth-rate error to contract below tolerance.",
                    "required_evidence_ids": ["cbc-linear-dispersion"],
                    "confidence": 0.62,
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": "campaign-cbc-normalization",
                    "linked_hypothesis_ids": ["h1"],
                    "objective": "Isolate whether trapped-particle normalization drives the CBC overprediction.",
                    "measurements": ["linear growth rate", "mode frequency", "normalization constants"],
                    "stop_conditions": ["growth-rate error remains outside tolerance after normalization sweep"],
                    "risk_controls": ["synthetic-only advisory run", "no controller parameter promotion"],
                }
            ],
        }

    assistant = PhysicsDebugAssistant()
    provider = HTTPChatProvider(
        provider_name="onsite-vllm",
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="local-physics-debugger",
        transport=transport,
    )

    report = assistant.analyze(evidence=[_cbc_evidence()], gaps=[_cbc_gap()], provider=provider)

    report_text = json.dumps(report, sort_keys=True)
    assert "secret-token" not in report_text
    assert "hidden-token" not in report_text
    assert report["provider"]["local_onsite"] is True
    assert report["hypotheses"][0]["falsification_test"].startswith("Run the same CBC case")
    assert report["campaign_suggestions"][0]["risk_controls"] == [
        "synthetic-only advisory run",
        "no controller parameter promotion",
    ]
    assert validate_physics_debug_report(report) == report


def test_physics_debug_rejects_remote_endpoint_without_explicit_policy() -> None:
    provider = HTTPChatProvider(
        provider_name="remote-compatible",
        endpoint="https://inference.example.invalid/v1/chat/completions",
        model="remote-model",
        transport=lambda payload: {},
    )

    with pytest.raises(ValueError, match="endpoint is not allowed"):
        PhysicsDebugAssistant().analyze(evidence=[_cbc_evidence()], gaps=[_cbc_gap()], provider=provider)


def test_physics_debug_rejects_uncited_or_non_falsifiable_hypothesis() -> None:
    provider = HTTPChatProvider(
        provider_name="onsite-vllm",
        endpoint="http://localhost:8000/v1/chat/completions",
        model="local-physics-debugger",
        transport=lambda payload: {
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": "Maybe the closure is wrong.",
                    "falsification_test": "",
                    "required_evidence_ids": ["unknown-evidence"],
                    "confidence": 0.7,
                }
            ],
            "campaign_suggestions": [],
        },
    )

    with pytest.raises(ValueError, match="falsification_test"):
        PhysicsDebugAssistant().analyze(evidence=[_cbc_evidence()], gaps=[_cbc_gap()], provider=provider)


def test_physics_debug_report_rejects_tampering() -> None:
    provider = HTTPChatProvider(
        provider_name="onsite-vllm",
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="local-physics-debugger",
        transport=lambda payload: {
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": "The drift-frequency normalization is inconsistent.",
                    "falsification_test": "Sweep drift-frequency normalization and require monotonic contraction in reference error.",
                    "required_evidence_ids": ["cbc-linear-dispersion"],
                    "confidence": 0.55,
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": "campaign-drift-frequency",
                    "linked_hypothesis_ids": ["h1"],
                    "objective": "Check whether drift-frequency normalization controls the discrepancy.",
                    "measurements": ["growth-rate error", "mode frequency"],
                    "stop_conditions": ["no monotonic error contraction"],
                    "risk_controls": ["offline advisory only"],
                }
            ],
        },
    )
    report = PhysicsDebugAssistant().analyze(evidence=[_cbc_evidence()], gaps=[_cbc_gap()], provider=provider)

    tampered = dict(report)
    tampered["hypotheses"] = [dict(report["hypotheses"][0], statement="Promote this immediately.")]

    with pytest.raises(ValueError, match="payload_sha256"):
        validate_physics_debug_report(tampered)


def test_physics_debug_remote_endpoint_requires_explicit_allowlist() -> None:
    provider = HTTPChatProvider(
        provider_name="site-gateway",
        endpoint="https://facility-gateway.example.invalid/v1/chat/completions",
        model="site-approved-model",
        transport=lambda payload: {
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": "The reference comparison may use inconsistent normalization.",
                    "falsification_test": "Replay the CBC comparison with declared normalization metadata fixed.",
                    "required_evidence_ids": ["cbc-linear-dispersion"],
                    "confidence": 0.5,
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": "campaign-reference-normalization",
                    "linked_hypothesis_ids": ["h1"],
                    "objective": "Replay the comparison with explicit normalization metadata.",
                    "measurements": ["growth-rate error"],
                    "stop_conditions": ["normalization replay does not change the error"],
                    "risk_controls": ["allowlisted facility gateway", "advisory output only"],
                }
            ],
        },
    )
    policy = ProviderPolicy(
        allow_remote_providers=True,
        allowed_endpoint_prefixes=("https://facility-gateway.example.invalid/",),
    )

    report = PhysicsDebugAssistant(policy=policy).analyze(
        evidence=[_cbc_evidence()], gaps=[_cbc_gap()], provider=provider
    )

    assert report["provider"]["local_onsite"] is False
    assert report["provider"]["endpoint"] == "https://facility-gateway.example.invalid/v1/chat/completions"
