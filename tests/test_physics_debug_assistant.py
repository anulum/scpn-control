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
    build_local_provider,
    run_provider_quorum,
    validate_physics_debug_report,
    validate_physics_debug_quorum_report,
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


def test_physics_debug_builds_local_provider_profiles_for_common_onsite_gateways() -> None:
    chat_provider = build_local_provider(
        family="chat-completions",
        model="onsite-chat-model",
        provider_name="site-chat-gateway",
    )
    ollama_provider = build_local_provider(
        family="ollama-chat",
        model="onsite-ollama-model",
        provider_name="site-ollama-gateway",
    )
    generation_provider = build_local_provider(
        family="text-generation",
        model="onsite-generation-model",
        provider_name="site-generation-gateway",
    )

    assert chat_provider.endpoint == "http://127.0.0.1:8000/v1/chat/completions"
    assert ollama_provider.endpoint == "http://127.0.0.1:11434/api/chat"
    assert generation_provider.endpoint == "http://127.0.0.1:8080/generate"
    assert {chat_provider.protocol, ollama_provider.protocol, generation_provider.protocol} == {
        "chat-completions",
        "ollama-chat",
        "text-generation",
    }
    assert all(provider.local_onsite for provider in (chat_provider, ollama_provider, generation_provider))


def test_physics_debug_local_provider_factory_rejects_non_loopback_host() -> None:
    with pytest.raises(ValueError, match="local provider host"):
        build_local_provider(
            family="chat-completions",
            model="onsite-chat-model",
            provider_name="bad-local-gateway",
            host="192.0.2.10",
        )


@pytest.mark.parametrize(
    ("protocol", "raw_response"),
    [
        (
            "chat-completions",
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "hypotheses": [
                                        {
                                            "hypothesis_id": "h1",
                                            "gap_id": "gk-cbc-linear-dispersion",
                                            "statement": "The local closure may use inconsistent species normalization.",
                                            "falsification_test": "Replay the CBC case with fixed species normalization metadata.",
                                            "required_evidence_ids": ["cbc-linear-dispersion"],
                                            "confidence": 0.5,
                                        }
                                    ],
                                    "campaign_suggestions": [
                                        {
                                            "campaign_id": "campaign-species-normalization",
                                            "linked_hypothesis_ids": ["h1"],
                                            "objective": "Replay with fixed species normalization metadata.",
                                            "measurements": ["growth-rate error"],
                                            "stop_conditions": ["normalization replay does not change the error"],
                                            "risk_controls": ["offline advisory only"],
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        ),
        (
            "ollama-chat",
            {
                "message": {
                    "content": json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "hypothesis_id": "h1",
                                    "gap_id": "gk-cbc-linear-dispersion",
                                    "statement": "The collisionless limit may be configured inconsistently.",
                                    "falsification_test": "Replay the CBC case with the collisionless limit toggled.",
                                    "required_evidence_ids": ["cbc-linear-dispersion"],
                                    "confidence": 0.52,
                                }
                            ],
                            "campaign_suggestions": [
                                {
                                    "campaign_id": "campaign-collisionless-limit",
                                    "linked_hypothesis_ids": ["h1"],
                                    "objective": "Replay with collisionless limit metadata fixed.",
                                    "measurements": ["growth-rate error"],
                                    "stop_conditions": ["limit toggle does not change the error"],
                                    "risk_controls": ["offline advisory only"],
                                }
                            ],
                        }
                    )
                }
            },
        ),
        (
            "text-generation",
            [
                {
                    "generated_text": json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "hypothesis_id": "h1",
                                    "gap_id": "gk-cbc-linear-dispersion",
                                    "statement": "The finite-Larmor-radius normalization may be inconsistent.",
                                    "falsification_test": "Replay the CBC case with finite-Larmor-radius terms disabled.",
                                    "required_evidence_ids": ["cbc-linear-dispersion"],
                                    "confidence": 0.51,
                                }
                            ],
                            "campaign_suggestions": [
                                {
                                    "campaign_id": "campaign-radius-normalization",
                                    "linked_hypothesis_ids": ["h1"],
                                    "objective": "Replay with finite-Larmor-radius terms isolated.",
                                    "measurements": ["growth-rate error"],
                                    "stop_conditions": ["isolated replay does not change the error"],
                                    "risk_controls": ["offline advisory only"],
                                }
                            ],
                        }
                    )
                }
            ],
        ),
    ],
)
def test_physics_debug_normalizes_common_onsite_provider_responses(
    protocol: str,
    raw_response: object,
) -> None:
    provider = build_local_provider(
        family=protocol,
        model="onsite-model",
        provider_name=f"site-{protocol}",
        transport=lambda payload: raw_response,
    )

    report = PhysicsDebugAssistant().analyze(evidence=[_cbc_evidence()], gaps=[_cbc_gap()], provider=provider)

    assert report["provider"]["protocol"] == protocol
    assert report["hypotheses"][0]["required_evidence_ids"] == ["cbc-linear-dispersion"]
    assert report["campaign_suggestions"][0]["risk_controls"] == ["offline advisory only"]


def _quorum_provider(provider_name: str, confidence: float = 0.6) -> HTTPChatProvider:
    return build_local_provider(
        family="direct-json",
        model=f"{provider_name}-model",
        provider_name=provider_name,
        transport=lambda payload: {
            "hypotheses": [
                {
                    "hypothesis_id": f"{provider_name}-h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": f"{provider_name} suspects normalization drift.",
                    "falsification_test": "Replay the CBC case with normalization metadata fixed.",
                    "required_evidence_ids": ["cbc-linear-dispersion"],
                    "confidence": confidence,
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": f"{provider_name}-campaign",
                    "linked_hypothesis_ids": [f"{provider_name}-h1"],
                    "objective": "Replay the comparison with declared normalization metadata.",
                    "measurements": ["growth-rate error", "mode frequency"],
                    "stop_conditions": ["normalization replay does not change the error"],
                    "risk_controls": ["offline advisory only", "quorum review before promotion"],
                }
            ],
        },
    )


def test_physics_debug_provider_quorum_prefers_local_reports_and_records_digests() -> None:
    providers = [_quorum_provider("onsite-a", 0.6), _quorum_provider("onsite-b", 0.72)]

    quorum = run_provider_quorum(
        evidence=[_cbc_evidence()],
        gaps=[_cbc_gap()],
        providers=providers,
        min_providers=2,
    )

    assert quorum["status"] == "advisory-quorum"
    assert quorum["local_provider_count"] == 2
    assert quorum["remote_provider_count"] == 0
    assert quorum["consensus_hypotheses"][0]["gap_id"] == "gk-cbc-linear-dispersion"
    assert quorum["consensus_hypotheses"][0]["provider_count"] == 2
    assert quorum["consensus_hypotheses"][0]["required_evidence_ids"] == ["cbc-linear-dispersion"]
    assert len(quorum["provider_reports"]) == 2
    assert all("payload_sha256" in item for item in quorum["provider_reports"])
    assert validate_physics_debug_quorum_report(quorum) == quorum


def test_physics_debug_provider_quorum_rejects_insufficient_local_coverage() -> None:
    remote_provider = HTTPChatProvider(
        provider_name="facility-gateway",
        endpoint="https://facility-gateway.example.invalid/v1/chat/completions",
        model="facility-approved-model",
        protocol="direct-json",
        transport=lambda payload: {
            "hypotheses": [
                {
                    "hypothesis_id": "remote-h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": "Remote advisory check.",
                    "falsification_test": "Replay the CBC comparison with fixed metadata.",
                    "required_evidence_ids": ["cbc-linear-dispersion"],
                    "confidence": 0.5,
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": "remote-campaign",
                    "linked_hypothesis_ids": ["remote-h1"],
                    "objective": "Replay the comparison.",
                    "measurements": ["growth-rate error"],
                    "stop_conditions": ["no error change"],
                    "risk_controls": ["facility allowlisted gateway", "advisory output only"],
                }
            ],
        },
    )
    policy = ProviderPolicy(
        allow_remote_providers=True,
        allowed_endpoint_prefixes=("https://facility-gateway.example.invalid/",),
    )

    with pytest.raises(ValueError, match="local provider"):
        run_provider_quorum(
            evidence=[_cbc_evidence()],
            gaps=[_cbc_gap()],
            providers=[remote_provider],
            policy=policy,
            min_providers=1,
            min_local_providers=1,
        )


def test_physics_debug_provider_quorum_rejects_uncorroborated_evidence() -> None:
    provider_a = _quorum_provider("onsite-a")
    provider_b = build_local_provider(
        family="direct-json",
        model="onsite-b-model",
        provider_name="onsite-b",
        transport=lambda payload: {
            "hypotheses": [
                {
                    "hypothesis_id": "onsite-b-h1",
                    "gap_id": "gk-cbc-linear-dispersion",
                    "statement": "Different uncited evidence drives the mismatch.",
                    "falsification_test": "Replay the CBC case with alternate evidence.",
                    "required_evidence_ids": ["cbc-linear-dispersion", "extra-evidence"],
                    "confidence": 0.6,
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": "onsite-b-campaign",
                    "linked_hypothesis_ids": ["onsite-b-h1"],
                    "objective": "Replay with alternate evidence.",
                    "measurements": ["growth-rate error"],
                    "stop_conditions": ["no error change"],
                    "risk_controls": ["offline advisory only"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="required_evidence_ids"):
        run_provider_quorum(
            evidence=[_cbc_evidence()],
            gaps=[_cbc_gap()],
            providers=[provider_a, provider_b],
            min_providers=2,
        )
