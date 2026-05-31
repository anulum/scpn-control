# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Local-First Physics Debug Assistance

"""Local-first physics debugging assistance for validation campaigns.

The assistant is an evidence triage layer. It can ask a local or allowlisted
language-model gateway for gap analyses, hypotheses, and campaign suggestions,
but the output is always schema-validated advisory evidence. It never promotes
physics constants, controller parameters, or facility claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, cast

PHYSICS_DEBUG_REPORT_SCHEMA_VERSION = "scpn-control.physics-debug-report.v1"
PHYSICS_DEBUG_QUORUM_REPORT_SCHEMA_VERSION = "scpn-control.physics-debug-quorum-report.v1"
PHYSICS_DEBUG_SCOPE = "advisory local-first physics gap analysis and campaign suggestion"
PHYSICS_DEBUG_QUORUM_SCOPE = "advisory local-first physics debug provider quorum"
PHYSICS_DEBUG_CLAIM_BOUNDARY = (
    "not validated physics truth, facility safety approval, controller-parameter promotion, or experimental evidence"
)
LOCAL_PROVIDER_PROFILES: dict[str, tuple[int, str]] = {
    "chat-completions": (8000, "/v1/chat/completions"),
    "ollama-chat": (11434, "/api/chat"),
    "text-generation": (8080, "/generate"),
    "direct-json": (8000, "/v1/chat/completions"),
}
GUARDRAIL_PROVIDER_PROFILES: dict[str, tuple[int, str]] = {
    "director-ai": (8765, "/v1/physics-debug/guardrail"),
    "direct-json": (8000, "/v1/guardrails/physics-debug"),
}
LOCAL_PROVIDER_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_FORBIDDEN_ACTION_PHRASES = (
    "promote controller",
    "controller promotion",
    "actuate controller",
    "change controller",
    "deploy to control",
    "bypass review",
    "facility safety approval",
    "validated physics truth",
)
PROMPT_INJECTION_REDACTION = "<redacted-prompt-injection>"
PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"),
    ),
    (
        "controller_promotion_instruction",
        re.compile(r"(?i)\bpromote\s+controller\b[^\.;\n]*"),
    ),
    (
        "review_bypass_instruction",
        re.compile(r"(?i)\bbypass\s+(?:human\s+)?review\b[^\.;\n]*"),
    ),
    (
        "approval_claim_instruction",
        re.compile(r"(?i)\bfacility\s+safety\s+approval\b[^\.;\n]*"),
    ),
)

ProviderTransport = Callable[[dict[str, Any]], Mapping[str, Any] | str | bytes]


@dataclass(frozen=True)
class ProviderPolicy:
    """Security policy for model-provider access."""

    allowed_endpoint_prefixes: tuple[str, ...] = (
        "http://127.0.0.1",
        "http://localhost",
        "https://127.0.0.1",
        "https://localhost",
    )
    allow_remote_providers: bool = False
    max_prompt_chars: int = 20_000
    max_response_chars: int = 50_000

    def __post_init__(self) -> None:
        if not self.allowed_endpoint_prefixes:
            raise ValueError("provider policy requires at least one endpoint prefix")
        if any(not isinstance(prefix, str) or not prefix for prefix in self.allowed_endpoint_prefixes):
            raise ValueError("provider policy endpoint prefixes must be non-empty strings")
        if self.max_prompt_chars <= 0:
            raise ValueError("provider policy max_prompt_chars must be positive")
        if self.max_response_chars <= 0:
            raise ValueError("provider policy max_response_chars must be positive")


@dataclass(frozen=True)
class PhysicsDebugGuardrailPolicy:
    """Admission policy for optional physics-debug hallucination guardrails."""

    required: bool = False
    block_actions: tuple[str, ...] = ("block",)
    allowed_decisions: tuple[str, ...] = ("allow", "block")
    blocking_severities: tuple[str, ...] = ("high", "critical")
    minimum_risk_controls: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.required, bool):
            raise ValueError("physics debug guardrail policy required must be a bool")
        if not self.block_actions:
            raise ValueError("physics debug guardrail policy block_actions must be non-empty")
        if not self.allowed_decisions:
            raise ValueError("physics debug guardrail policy allowed_decisions must be non-empty")
        if any(not isinstance(item, str) or not item for item in self.block_actions):
            raise ValueError("physics debug guardrail policy block_actions must contain non-empty strings")
        if any(not isinstance(item, str) or not item for item in self.allowed_decisions):
            raise ValueError("physics debug guardrail policy allowed_decisions must contain non-empty strings")
        if not set(self.block_actions).issubset(set(self.allowed_decisions)):
            raise ValueError("physics debug guardrail policy block_actions must be allowed decisions")
        severities = {"info", "low", "medium", "high", "critical"}
        if any(not isinstance(item, str) or item not in severities for item in self.blocking_severities):
            raise ValueError("physics debug guardrail policy blocking_severities are unsupported")
        if not isinstance(self.minimum_risk_controls, int) or self.minimum_risk_controls < 0:
            raise ValueError("physics debug guardrail policy minimum_risk_controls must be non-negative")

    def payload(self) -> dict[str, Any]:
        """Return a JSON-safe guardrail policy payload."""

        return {
            "required": self.required,
            "block_actions": list(self.block_actions),
            "allowed_decisions": list(self.allowed_decisions),
            "blocking_severities": list(self.blocking_severities),
            "minimum_risk_controls": self.minimum_risk_controls,
        }


@dataclass(frozen=True)
class PhysicsDebugSafetyPolicy:
    """Fail-closed admission policy for advisory physics-debug output."""

    human_review_required: bool = True
    max_advisory_confidence: float = 0.95
    forbidden_action_phrases: tuple[str, ...] = DEFAULT_FORBIDDEN_ACTION_PHRASES

    def __post_init__(self) -> None:
        if not isinstance(self.human_review_required, bool):
            raise ValueError("physics debug safety policy human_review_required must be a bool")
        if not isinstance(self.max_advisory_confidence, int | float) or not math.isfinite(
            float(self.max_advisory_confidence)
        ):
            raise ValueError("physics debug safety policy max_advisory_confidence must be finite")
        if self.max_advisory_confidence <= 0.0 or self.max_advisory_confidence > 1.0:
            raise ValueError("physics debug safety policy max_advisory_confidence must be in (0, 1]")
        if not self.forbidden_action_phrases:
            raise ValueError("physics debug safety policy forbidden_action_phrases must be non-empty")
        if any(not isinstance(item, str) or not item.strip() for item in self.forbidden_action_phrases):
            raise ValueError("physics debug safety policy forbidden_action_phrases must be non-empty strings")

    def payload(self) -> dict[str, Any]:
        """Return a JSON-safe policy payload for report binding."""

        return {
            "human_review_required": self.human_review_required,
            "max_advisory_confidence": float(self.max_advisory_confidence),
            "forbidden_action_phrases": list(self.forbidden_action_phrases),
        }


@dataclass(frozen=True)
class PhysicsDebugEvidence:
    """Evidence item supplied to the physics debugging assistant."""

    evidence_id: str
    evidence_type: str
    source: str
    summary: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("evidence_id", self.evidence_id)
        _require_non_empty("evidence_type", self.evidence_type)
        _require_non_empty("source", self.source)
        _require_non_empty("summary", self.summary)
        if self.sha256 is not None and not _is_sha256(self.sha256):
            raise ValueError("physics debug evidence sha256 must be a SHA-256 hex digest")


@dataclass(frozen=True)
class PhysicsDebugGap:
    """Physics validation gap to analyse."""

    gap_id: str
    severity: str
    description: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty("gap_id", self.gap_id)
        if self.severity not in {"low", "medium", "high", "critical"}:
            raise ValueError("physics debug gap severity is unsupported")
        _require_non_empty("description", self.description)
        if not self.evidence_ids:
            raise ValueError("physics debug gap requires evidence_ids")
        if any(not isinstance(evidence_id, str) or not evidence_id for evidence_id in self.evidence_ids):
            raise ValueError("physics debug gap evidence_ids must be non-empty strings")


@dataclass(frozen=True)
class HTTPChatProvider:
    """Allowlisted HTTP chat gateway for local or facility-approved providers."""

    provider_name: str
    endpoint: str
    model: str
    protocol: str = "chat-completions"
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    transport: ProviderTransport | None = None

    def __post_init__(self) -> None:
        _require_non_empty("provider_name", self.provider_name)
        _require_non_empty("endpoint", self.endpoint)
        _require_non_empty("model", self.model)
        if self.protocol not in LOCAL_PROVIDER_PROFILES:
            raise ValueError("provider protocol is unsupported")
        if self.timeout_seconds <= 0.0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("provider timeout_seconds must be positive and finite")
        for key, value in self.headers.items():
            if not isinstance(key, str) or not key or "\n" in key or "\r" in key:
                raise ValueError("provider header names must be non-empty single-line strings")
            if not isinstance(value, str) or "\n" in value or "\r" in value:
                raise ValueError("provider header values must be single-line strings")

    @property
    def local_onsite(self) -> bool:
        """Return whether the endpoint is local or onsite-loopback by construction."""

        parsed = urllib.parse.urlparse(self.endpoint)
        hostname = parsed.hostname or ""
        return hostname in {"localhost", "127.0.0.1", "::1"}

    def metadata(self) -> dict[str, Any]:
        """Return non-secret provider metadata for reports."""

        return {
            "provider_name": self.provider_name,
            "endpoint": self.endpoint,
            "model": self.model,
            "protocol": self.protocol,
            "local_onsite": self.local_onsite,
        }

    def complete_json(self, prompt: str, policy: ProviderPolicy) -> dict[str, Any]:
        """Return a JSON object from the configured provider gateway."""

        _validate_provider_endpoint(self, policy)
        if len(prompt) > policy.max_prompt_chars:
            raise ValueError("physics debug prompt exceeds policy max_prompt_chars")
        request_payload = _provider_request_payload(self.protocol, self.model, prompt)
        if self.transport is not None:
            return _coerce_provider_response(self.transport(request_payload), policy)
        return _coerce_provider_response(self._post_json(request_payload, policy), policy)

    def _post_json(self, payload: dict[str, Any], policy: ProviderPolicy) -> bytes:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", **dict(self.headers)}
        request = urllib.request.Request(self.endpoint, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = cast(bytes, response.read(policy.max_response_chars + 1))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"physics debug provider request failed: {exc}") from exc
        if len(body) > policy.max_response_chars:
            raise ValueError("physics debug provider response exceeds policy max_response_chars")
        return body


@dataclass(frozen=True)
class PhysicsDebugGuardrailProvider:
    """Allowlisted guardrail gateway for advisory physics-debug reports."""

    provider_name: str
    endpoint: str
    model: str
    profile: str = "director-ai"
    protocol: str = "direct-json"
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    transport: ProviderTransport | None = None

    def __post_init__(self) -> None:
        _require_non_empty("guardrail provider_name", self.provider_name)
        _require_non_empty("guardrail endpoint", self.endpoint)
        _require_non_empty("guardrail model", self.model)
        if self.profile not in GUARDRAIL_PROVIDER_PROFILES:
            raise ValueError("guardrail provider profile is unsupported")
        if self.protocol != "direct-json":
            raise ValueError("guardrail provider protocol is unsupported")
        if self.timeout_seconds <= 0.0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("guardrail provider timeout_seconds must be positive and finite")
        for key, value in self.headers.items():
            if not isinstance(key, str) or not key or "\n" in key or "\r" in key:
                raise ValueError("guardrail provider header names must be non-empty single-line strings")
            if not isinstance(value, str) or "\n" in value or "\r" in value:
                raise ValueError("guardrail provider header values must be single-line strings")

    @property
    def local_onsite(self) -> bool:
        """Return whether the guardrail endpoint is local by construction."""

        parsed = urllib.parse.urlparse(self.endpoint)
        hostname = parsed.hostname or ""
        return hostname in LOCAL_PROVIDER_HOSTS

    def metadata(self) -> dict[str, Any]:
        """Return non-secret guardrail metadata for report binding."""

        return {
            "provider_name": self.provider_name,
            "endpoint": self.endpoint,
            "model": self.model,
            "profile": self.profile,
            "protocol": self.protocol,
            "local_onsite": self.local_onsite,
        }

    def review(
        self,
        *,
        prompt: str,
        evidence: list[dict[str, Any]],
        gaps: list[dict[str, Any]],
        provider_output: Mapping[str, Any],
        policy: ProviderPolicy,
        guardrail_policy: PhysicsDebugGuardrailPolicy,
    ) -> dict[str, Any]:
        """Return a validated guardrail review for a provider draft."""

        _validate_guardrail_endpoint(self, policy)
        payload = _guardrail_request_payload(
            prompt=prompt,
            evidence=evidence,
            gaps=gaps,
            provider_output=provider_output,
            guardrail_policy=guardrail_policy,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if len(encoded) > policy.max_prompt_chars:
            raise ValueError("physics debug guardrail prompt exceeds policy max_prompt_chars")
        if self.transport is not None:
            raw_review = self.transport(payload)
        else:
            raw_review = self._post_json(payload, policy)
        return _validate_guardrail_review(
            _coerce_provider_response(raw_review, policy),
            evidence=evidence,
            provider=self,
            guardrail_policy=guardrail_policy,
            request=payload["request"],
        )

    def _post_json(self, payload: dict[str, Any], policy: ProviderPolicy) -> bytes:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", **dict(self.headers)}
        request = urllib.request.Request(self.endpoint, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = cast(bytes, response.read(policy.max_response_chars + 1))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"physics debug guardrail request failed: {exc}") from exc
        if len(body) > policy.max_response_chars:
            raise ValueError("physics debug guardrail response exceeds policy max_response_chars")
        return body


@dataclass(frozen=True)
class PhysicsDebugAssistant:
    """Evidence-first assistant for physics gap analysis."""

    policy: ProviderPolicy = field(default_factory=ProviderPolicy)
    safety_policy: PhysicsDebugSafetyPolicy = field(default_factory=PhysicsDebugSafetyPolicy)
    guardrail_policy: PhysicsDebugGuardrailPolicy = field(default_factory=PhysicsDebugGuardrailPolicy)

    def analyze(
        self,
        *,
        evidence: list[PhysicsDebugEvidence],
        gaps: list[PhysicsDebugGap],
        provider: HTTPChatProvider,
        guardrail_provider: PhysicsDebugGuardrailProvider | None = None,
    ) -> dict[str, Any]:
        """Ask an allowlisted provider for advisory hypotheses and campaigns."""

        if not evidence:
            raise ValueError("physics debug analysis requires evidence")
        if not gaps:
            raise ValueError("physics debug analysis requires gaps")
        evidence_payload, redactions, prompt_guard_findings = _sanitized_evidence_payload(evidence)
        gap_payload = [asdict(gap) for gap in gaps]
        _validate_evidence_gap_linkage(evidence_payload, gap_payload)
        prompt = _build_physics_debug_prompt(evidence_payload, gap_payload)
        provider_output = provider.complete_json(prompt, self.policy)
        guardrail = _disabled_guardrail_review(self.guardrail_policy)
        if guardrail_provider is None and self.guardrail_policy.required:
            raise ValueError("physics debug guardrail policy requires a guardrail provider")
        if guardrail_provider is not None:
            guardrail = guardrail_provider.review(
                prompt=prompt,
                evidence=evidence_payload,
                gaps=gap_payload,
                provider_output=provider_output,
                policy=self.policy,
                guardrail_policy=self.guardrail_policy,
            )
        return build_physics_debug_report(
            provider=provider,
            evidence=evidence_payload,
            gaps=gap_payload,
            provider_output=provider_output,
            redactions=redactions,
            safety_policy=self.safety_policy,
            prompt_guard_findings=prompt_guard_findings,
            guardrail_policy=self.guardrail_policy,
            guardrail=guardrail,
        )


def build_physics_debug_report(
    *,
    provider: HTTPChatProvider,
    evidence: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    provider_output: Mapping[str, Any],
    redactions: list[str] | None = None,
    safety_policy: PhysicsDebugSafetyPolicy | None = None,
    prompt_guard_findings: list[str] | None = None,
    guardrail_policy: PhysicsDebugGuardrailPolicy | None = None,
    guardrail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-versioned advisory physics debugging report."""

    resolved_safety_policy = PhysicsDebugSafetyPolicy() if safety_policy is None else safety_policy
    resolved_guardrail_policy = PhysicsDebugGuardrailPolicy() if guardrail_policy is None else guardrail_policy
    guardrail_payload = _disabled_guardrail_review(resolved_guardrail_policy) if guardrail is None else dict(guardrail)
    hypotheses, campaigns = _validate_provider_output(provider_output, evidence, gaps, resolved_safety_policy)
    payload: dict[str, Any] = {
        "schema_version": PHYSICS_DEBUG_REPORT_SCHEMA_VERSION,
        "status": "advisory",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": PHYSICS_DEBUG_SCOPE,
        "claim_boundary": PHYSICS_DEBUG_CLAIM_BOUNDARY,
        "human_review_required": resolved_safety_policy.human_review_required,
        "safety_policy": resolved_safety_policy.payload(),
        "guardrail_policy": resolved_guardrail_policy.payload(),
        "provider": provider.metadata(),
        "guardrail": guardrail_payload,
        "evidence": evidence,
        "gaps": gaps,
        "hypotheses": hypotheses,
        "campaign_suggestions": campaigns,
        "redactions": sorted(set(redactions or [])),
        "prompt_guard_findings": sorted(set(prompt_guard_findings or [])),
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_physics_debug_report(payload)


def build_local_provider(
    *,
    family: str,
    model: str,
    provider_name: str,
    host: str = "127.0.0.1",
    port: int | None = None,
    path: str | None = None,
    transport: ProviderTransport | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> HTTPChatProvider:
    """Build a loopback-only provider profile for onsite model gateways."""

    if family not in LOCAL_PROVIDER_PROFILES:
        raise ValueError("local provider family is unsupported")
    if host not in LOCAL_PROVIDER_HOSTS:
        raise ValueError("local provider host must be loopback")
    default_port, default_path = LOCAL_PROVIDER_PROFILES[family]
    resolved_port = default_port if port is None else port
    if not isinstance(resolved_port, int) or resolved_port <= 0 or resolved_port > 65_535:
        raise ValueError("local provider port must be in 1..65535")
    resolved_path = default_path if path is None else path
    if not isinstance(resolved_path, str) or not resolved_path.startswith("/") or "\x00" in resolved_path:
        raise ValueError("local provider path must be an absolute URL path")
    host_part = f"[{host}]" if host == "::1" else host
    endpoint = f"http://{host_part}:{resolved_port}{resolved_path}"
    return HTTPChatProvider(
        provider_name=provider_name,
        endpoint=endpoint,
        model=model,
        protocol=family,
        headers={} if headers is None else headers,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def build_guardrail_provider(
    *,
    model: str,
    provider_name: str,
    profile: str = "director-ai",
    host: str = "127.0.0.1",
    port: int | None = None,
    path: str | None = None,
    transport: ProviderTransport | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> PhysicsDebugGuardrailProvider:
    """Build a loopback-only guardrail profile for physics debug reviews."""

    if profile not in GUARDRAIL_PROVIDER_PROFILES:
        raise ValueError("guardrail provider profile is unsupported")
    if host not in LOCAL_PROVIDER_HOSTS:
        raise ValueError("guardrail provider host must be loopback")
    default_port, default_path = GUARDRAIL_PROVIDER_PROFILES[profile]
    resolved_port = default_port if port is None else port
    if not isinstance(resolved_port, int) or resolved_port <= 0 or resolved_port > 65_535:
        raise ValueError("guardrail provider port must be in 1..65535")
    resolved_path = default_path if path is None else path
    if not isinstance(resolved_path, str) or not resolved_path.startswith("/") or "\x00" in resolved_path:
        raise ValueError("guardrail provider path must be an absolute URL path")
    host_part = f"[{host}]" if host == "::1" else host
    endpoint = f"http://{host_part}:{resolved_port}{resolved_path}"
    return PhysicsDebugGuardrailProvider(
        provider_name=provider_name,
        endpoint=endpoint,
        model=model,
        profile=profile,
        headers={} if headers is None else headers,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def validate_physics_debug_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tamper-evident advisory physics debugging report."""

    if not isinstance(payload, dict):
        raise ValueError("physics debug report must be an object")
    if payload.get("schema_version") != PHYSICS_DEBUG_REPORT_SCHEMA_VERSION:
        raise ValueError("physics debug report schema_version is unsupported")
    if payload.get("status") != "advisory":
        raise ValueError("physics debug report status must be advisory")
    if payload.get("scope") != PHYSICS_DEBUG_SCOPE:
        raise ValueError("physics debug report scope is unsupported")
    if payload.get("claim_boundary") != PHYSICS_DEBUG_CLAIM_BOUNDARY:
        raise ValueError("physics debug report claim_boundary is unsupported")
    if not isinstance(payload.get("human_review_required"), bool):
        raise ValueError("physics debug report human_review_required must be a bool")
    safety_policy = _safety_policy_from_payload(payload.get("safety_policy"))
    if payload["human_review_required"] != safety_policy.human_review_required:
        raise ValueError("physics debug report human_review_required must match safety_policy")
    guardrail_policy = _guardrail_policy_from_payload(payload.get("guardrail_policy"))
    if not isinstance(payload.get("created_at"), str) or not payload["created_at"]:
        raise ValueError("physics debug report created_at must be a non-empty string")
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("physics debug report provider must be an object")
    for key in ("provider_name", "endpoint", "model", "protocol"):
        if not isinstance(provider.get(key), str) or not provider[key]:
            raise ValueError(f"physics debug report provider {key} must be a non-empty string")
    if provider["protocol"] not in LOCAL_PROVIDER_PROFILES:
        raise ValueError("physics debug report provider protocol is unsupported")
    if not isinstance(provider.get("local_onsite"), bool):
        raise ValueError("physics debug report provider local_onsite must be a bool")
    evidence = payload.get("evidence")
    gaps = payload.get("gaps")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("physics debug report evidence must be a non-empty list")
    if not isinstance(gaps, list) or not gaps:
        raise ValueError("physics debug report gaps must be a non-empty list")
    _validate_evidence_gap_linkage(evidence, gaps)
    _validate_report_guardrail(payload.get("guardrail"), evidence, guardrail_policy)
    _validate_hypotheses(payload.get("hypotheses"), evidence, gaps, safety_policy)
    _validate_campaigns(payload.get("campaign_suggestions"), payload["hypotheses"], safety_policy)
    redactions = payload.get("redactions")
    if not isinstance(redactions, list) or any(not isinstance(item, str) for item in redactions):
        raise ValueError("physics debug report redactions must be a list of strings")
    prompt_guard_findings = payload.get("prompt_guard_findings")
    if not isinstance(prompt_guard_findings, list) or any(not isinstance(item, str) for item in prompt_guard_findings):
        raise ValueError("physics debug report prompt_guard_findings must be a list of strings")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("physics debug report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("physics debug report payload_sha256 does not match payload")
    return payload


def write_physics_debug_report(payload: dict[str, Any], json_path: str | Path) -> dict[str, Any]:
    """Persist a validated advisory physics debugging report as JSON."""

    validated = validate_physics_debug_report(payload)
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validated


def run_provider_quorum(
    *,
    evidence: list[PhysicsDebugEvidence],
    gaps: list[PhysicsDebugGap],
    providers: list[HTTPChatProvider],
    policy: ProviderPolicy | None = None,
    safety_policy: PhysicsDebugSafetyPolicy | None = None,
    guardrail_policy: PhysicsDebugGuardrailPolicy | None = None,
    guardrail_provider: PhysicsDebugGuardrailProvider | None = None,
    min_providers: int = 2,
    min_local_providers: int = 1,
) -> dict[str, Any]:
    """Run multiple providers and admit only corroborated advisory evidence."""

    if len(providers) < min_providers:
        raise ValueError("physics debug quorum requires at least min_providers providers")
    if min_providers <= 0:
        raise ValueError("physics debug quorum min_providers must be positive")
    if min_local_providers < 0:
        raise ValueError("physics debug quorum min_local_providers must be non-negative")
    resolved_policy = ProviderPolicy() if policy is None else policy
    resolved_safety_policy = PhysicsDebugSafetyPolicy() if safety_policy is None else safety_policy
    resolved_guardrail_policy = PhysicsDebugGuardrailPolicy() if guardrail_policy is None else guardrail_policy
    assistant = PhysicsDebugAssistant(
        policy=resolved_policy,
        safety_policy=resolved_safety_policy,
        guardrail_policy=resolved_guardrail_policy,
    )
    provider_reports = [
        assistant.analyze(evidence=evidence, gaps=gaps, provider=provider, guardrail_provider=guardrail_provider)
        for provider in sorted(providers, key=lambda item: (not item.local_onsite, item.provider_name))
    ]
    local_count = sum(1 for report in provider_reports if report["provider"]["local_onsite"])
    remote_count = len(provider_reports) - local_count
    if local_count < min_local_providers:
        raise ValueError("physics debug quorum requires more local provider coverage")
    consensus = _build_consensus_hypotheses(provider_reports, min_providers)
    payload: dict[str, Any] = {
        "schema_version": PHYSICS_DEBUG_QUORUM_REPORT_SCHEMA_VERSION,
        "status": "advisory-quorum",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": PHYSICS_DEBUG_QUORUM_SCOPE,
        "claim_boundary": PHYSICS_DEBUG_CLAIM_BOUNDARY,
        "human_review_required": resolved_safety_policy.human_review_required,
        "safety_policy": resolved_safety_policy.payload(),
        "guardrail_policy": resolved_guardrail_policy.payload(),
        "provider_count": len(provider_reports),
        "local_provider_count": local_count,
        "remote_provider_count": remote_count,
        "min_providers": min_providers,
        "min_local_providers": min_local_providers,
        "provider_reports": provider_reports,
        "consensus_hypotheses": consensus,
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_physics_debug_quorum_report(payload)


def validate_physics_debug_quorum_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tamper-evident provider-quorum physics debug report."""

    if not isinstance(payload, dict):
        raise ValueError("physics debug quorum report must be an object")
    if payload.get("schema_version") != PHYSICS_DEBUG_QUORUM_REPORT_SCHEMA_VERSION:
        raise ValueError("physics debug quorum report schema_version is unsupported")
    if payload.get("status") != "advisory-quorum":
        raise ValueError("physics debug quorum report status must be advisory-quorum")
    if payload.get("scope") != PHYSICS_DEBUG_QUORUM_SCOPE:
        raise ValueError("physics debug quorum report scope is unsupported")
    if payload.get("claim_boundary") != PHYSICS_DEBUG_CLAIM_BOUNDARY:
        raise ValueError("physics debug quorum report claim_boundary is unsupported")
    if not isinstance(payload.get("human_review_required"), bool):
        raise ValueError("physics debug quorum report human_review_required must be a bool")
    safety_policy = _safety_policy_from_payload(payload.get("safety_policy"))
    if payload["human_review_required"] != safety_policy.human_review_required:
        raise ValueError("physics debug quorum report human_review_required must match safety_policy")
    guardrail_policy = _guardrail_policy_from_payload(payload.get("guardrail_policy"))
    for key in (
        "provider_count",
        "local_provider_count",
        "remote_provider_count",
        "min_providers",
        "min_local_providers",
    ):
        if not isinstance(payload.get(key), int) or payload[key] < 0:
            raise ValueError(f"physics debug quorum report {key} must be a non-negative integer")
    if payload["provider_count"] < payload["min_providers"]:
        raise ValueError("physics debug quorum report provider_count must meet min_providers")
    if payload["local_provider_count"] < payload["min_local_providers"]:
        raise ValueError("physics debug quorum report local_provider_count must meet min_local_providers")
    if payload["local_provider_count"] + payload["remote_provider_count"] != payload["provider_count"]:
        raise ValueError("physics debug quorum report provider counts are inconsistent")
    provider_reports = payload.get("provider_reports")
    if not isinstance(provider_reports, list) or len(provider_reports) != payload["provider_count"]:
        raise ValueError("physics debug quorum report provider_reports count is inconsistent")
    for report in provider_reports:
        validated_report = validate_physics_debug_report(report)
        if validated_report["safety_policy"] != safety_policy.payload():
            raise ValueError("physics debug quorum report provider safety_policy mismatch")
        if validated_report["guardrail_policy"] != guardrail_policy.payload():
            raise ValueError("physics debug quorum report provider guardrail_policy mismatch")
    consensus = payload.get("consensus_hypotheses")
    if not isinstance(consensus, list) or not consensus:
        raise ValueError("physics debug quorum report consensus_hypotheses must be non-empty")
    for item in consensus:
        _validate_consensus_hypothesis(item, payload["min_providers"])
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("physics debug quorum report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("physics debug quorum report payload_sha256 does not match payload")
    return payload


def write_physics_debug_quorum_report(payload: dict[str, Any], json_path: str | Path) -> dict[str, Any]:
    """Persist a validated provider-quorum physics debug report as JSON."""

    validated = validate_physics_debug_quorum_report(payload)
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validated


def _build_physics_debug_prompt(evidence: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> str:
    prompt_payload = {
        "task": "physics_debug_gap_analysis",
        "claim_boundary": PHYSICS_DEBUG_CLAIM_BOUNDARY,
        "required_output_schema": {
            "hypotheses": [
                {
                    "hypothesis_id": "non-empty string",
                    "gap_id": "existing gap_id",
                    "statement": "non-empty physics hypothesis",
                    "falsification_test": "non-empty falsifiable test",
                    "required_evidence_ids": ["existing evidence_id"],
                    "confidence": "float in [0, 1]",
                }
            ],
            "campaign_suggestions": [
                {
                    "campaign_id": "non-empty string",
                    "linked_hypothesis_ids": ["existing hypothesis_id"],
                    "objective": "non-empty objective",
                    "measurements": ["non-empty measurement"],
                    "stop_conditions": ["non-empty stop condition"],
                    "risk_controls": ["non-empty risk control"],
                }
            ],
        },
        "evidence": evidence,
        "gaps": gaps,
    }
    return json.dumps(prompt_payload, sort_keys=True, separators=(",", ":"))


def _provider_request_payload(protocol: str, model: str, prompt: str) -> dict[str, Any]:
    system_prompt = (
        "Return strict JSON only. Produce falsifiable physics hypotheses and advisory "
        "experimental campaign suggestions. Do not promote controller parameters or facility claims."
    )
    if protocol in {"chat-completions", "direct-json"}:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
    if protocol == "ollama-chat":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
    if protocol == "text-generation":
        return {
            "inputs": f"{system_prompt}\n\n{prompt}",
            "parameters": {"temperature": 0.1, "return_full_text": False},
        }
    raise ValueError("provider protocol is unsupported")


def _build_consensus_hypotheses(provider_reports: list[dict[str, Any]], min_providers: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for report in provider_reports:
        provider_name = report["provider"]["provider_name"]
        for hypothesis in report["hypotheses"]:
            key = (hypothesis["gap_id"], tuple(hypothesis["required_evidence_ids"]))
            grouped.setdefault(key, []).append({**hypothesis, "provider_name": provider_name})
    consensus: list[dict[str, Any]] = []
    for (gap_id, required_evidence_ids), hypotheses in sorted(grouped.items()):
        if len(hypotheses) >= min_providers:
            consensus.append(
                {
                    "gap_id": gap_id,
                    "required_evidence_ids": list(required_evidence_ids),
                    "provider_count": len(hypotheses),
                    "provider_names": sorted(item["provider_name"] for item in hypotheses),
                    "mean_confidence": sum(item["confidence"] for item in hypotheses) / len(hypotheses),
                    "falsification_tests": sorted({item["falsification_test"] for item in hypotheses}),
                }
            )
    if not consensus:
        raise ValueError("physics debug quorum requires corroborated required_evidence_ids")
    return consensus


def _validate_consensus_hypothesis(item: object, min_providers: int) -> None:
    if not isinstance(item, dict):
        raise ValueError("physics debug quorum consensus hypothesis must be an object")
    _require_non_empty("consensus gap_id", item.get("gap_id"))
    required_evidence_ids = _require_non_empty_string_list(
        "consensus required_evidence_ids", item.get("required_evidence_ids")
    )
    if len(set(required_evidence_ids)) != len(required_evidence_ids):
        raise ValueError("physics debug quorum consensus required_evidence_ids must be unique")
    provider_count = item.get("provider_count")
    if not isinstance(provider_count, int) or provider_count < min_providers:
        raise ValueError("physics debug quorum consensus provider_count must meet min_providers")
    provider_names = _require_non_empty_string_list("consensus provider_names", item.get("provider_names"))
    if len(set(provider_names)) != provider_count:
        raise ValueError("physics debug quorum consensus provider_names must match provider_count")
    mean_confidence = item.get("mean_confidence")
    if not isinstance(mean_confidence, int | float) or not math.isfinite(float(mean_confidence)):
        raise ValueError("physics debug quorum consensus mean_confidence must be finite")
    if float(mean_confidence) < 0.0 or float(mean_confidence) > 1.0:
        raise ValueError("physics debug quorum consensus mean_confidence must be in [0, 1]")
    _require_non_empty_string_list("consensus falsification_tests", item.get("falsification_tests"))


def _sanitized_evidence_payload(
    evidence: list[PhysicsDebugEvidence],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    redactions: list[str] = []
    prompt_guard_findings: list[str] = []
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        if item.evidence_id in seen:
            raise ValueError("physics debug evidence IDs must be unique")
        seen.add(item.evidence_id)
        summary, summary_redactions = _redact_sensitive_text(item.summary)
        source, source_redactions = _redact_sensitive_text(item.source)
        summary, summary_guard_findings = _neutralize_prompt_injection_text(summary)
        source, source_guard_findings = _neutralize_prompt_injection_text(source)
        redactions.extend(summary_redactions)
        redactions.extend(source_redactions)
        prompt_guard_findings.extend(summary_guard_findings)
        prompt_guard_findings.extend(source_guard_findings)
        payload.append(
            {
                "evidence_id": item.evidence_id,
                "evidence_type": item.evidence_type,
                "source": source,
                "summary": summary,
                "sha256": item.sha256.lower() if item.sha256 is not None else None,
            }
        )
    return payload, redactions, sorted(set(prompt_guard_findings))


def _validate_provider_endpoint(provider: HTTPChatProvider, policy: ProviderPolicy) -> None:
    if not provider.endpoint.startswith(policy.allowed_endpoint_prefixes):
        raise ValueError("physics debug provider endpoint is not allowed by policy")
    if not provider.local_onsite and not policy.allow_remote_providers:
        raise ValueError("physics debug provider endpoint is not allowed unless remote providers are enabled")


def _validate_guardrail_endpoint(provider: PhysicsDebugGuardrailProvider, policy: ProviderPolicy) -> None:
    if not provider.endpoint.startswith(policy.allowed_endpoint_prefixes):
        raise ValueError("physics debug guardrail endpoint is not allowed by policy")
    if not provider.local_onsite and not policy.allow_remote_providers:
        raise ValueError("physics debug guardrail endpoint is not allowed unless remote providers are enabled")


def _guardrail_request_payload(
    *,
    prompt: str,
    evidence: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    provider_output: Mapping[str, Any],
    guardrail_policy: PhysicsDebugGuardrailPolicy,
) -> dict[str, Any]:
    request = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "evidence_sha256": _payload_digest({"evidence": evidence}),
        "gaps_sha256": _payload_digest({"gaps": gaps}),
        "provider_output_sha256": _payload_digest({"provider_output": dict(provider_output)}),
    }
    return {
        "task": "physics_debug_guardrail_review",
        "claim_boundary": PHYSICS_DEBUG_CLAIM_BOUNDARY,
        "guardrail_policy": guardrail_policy.payload(),
        "request": request,
        "required_output_schema": {
            "decision": "allow or block",
            "reviewed_output_sha256": request["provider_output_sha256"],
            "findings": [
                {
                    "finding_id": "non-empty string",
                    "severity": "info, low, medium, high, or critical",
                    "message": "non-empty guardrail finding",
                    "evidence_ids": ["existing evidence_id"],
                    "action": "allow or block",
                }
            ],
            "risk_controls": ["non-empty risk control"],
        },
        "prompt": prompt,
        "evidence": evidence,
        "gaps": gaps,
        "provider_output": dict(provider_output),
    }


def _validate_guardrail_review(
    raw_review: Mapping[str, Any],
    *,
    evidence: list[dict[str, Any]],
    provider: PhysicsDebugGuardrailProvider,
    guardrail_policy: PhysicsDebugGuardrailPolicy,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload = _validate_guardrail_request_metadata(request)
    decision = _require_non_empty("guardrail decision", raw_review.get("decision"))
    if decision not in guardrail_policy.allowed_decisions:
        raise ValueError("physics debug guardrail decision is unsupported")
    reviewed_output_sha256 = raw_review.get("reviewed_output_sha256", request_payload["provider_output_sha256"])
    if not isinstance(reviewed_output_sha256, str) or not _is_sha256(reviewed_output_sha256):
        raise ValueError("physics debug guardrail reviewed_output_sha256 must be a SHA-256 hex digest")
    if reviewed_output_sha256.lower() != request_payload["provider_output_sha256"]:
        raise ValueError("physics debug guardrail reviewed_output_sha256 must match provider_output_sha256")
    findings = _validate_guardrail_findings(
        raw_review.get("findings"),
        evidence=evidence,
        guardrail_policy=guardrail_policy,
        require_findings=True,
    )
    risk_controls = _require_non_empty_string_list("guardrail risk_controls", raw_review.get("risk_controls"))
    if len(risk_controls) < guardrail_policy.minimum_risk_controls:
        raise ValueError("physics debug guardrail risk_controls do not meet policy minimum")
    blocked = decision in guardrail_policy.block_actions or any(
        finding["action"] in guardrail_policy.block_actions for finding in findings
    )
    review = {
        "enabled": True,
        "required": guardrail_policy.required,
        "provider": provider.metadata(),
        "request": request_payload,
        "reviewed_output_sha256": reviewed_output_sha256.lower(),
        "decision": decision,
        "blocked": blocked,
        "findings": findings,
        "risk_controls": risk_controls,
    }
    if blocked:
        raise ValueError("physics debug guardrail blocks advisory report")
    return review


def _disabled_guardrail_review(guardrail_policy: PhysicsDebugGuardrailPolicy) -> dict[str, Any]:
    return {
        "enabled": False,
        "required": guardrail_policy.required,
        "provider": None,
        "request": None,
        "reviewed_output_sha256": None,
        "decision": "not-configured",
        "blocked": False,
        "findings": [],
        "risk_controls": [],
    }


def _validate_report_guardrail(
    value: object,
    evidence: list[dict[str, Any]],
    guardrail_policy: PhysicsDebugGuardrailPolicy,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("physics debug report guardrail must be an object")
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("physics debug report guardrail enabled must be a bool")
    if value.get("required") != guardrail_policy.required:
        raise ValueError("physics debug report guardrail required must match guardrail_policy")
    blocked = value.get("blocked")
    if not isinstance(blocked, bool):
        raise ValueError("physics debug report guardrail blocked must be a bool")
    if blocked:
        raise ValueError("physics debug report guardrail cannot persist blocked reviews")
    decision = _require_non_empty("guardrail decision", value.get("decision"))
    if enabled:
        if decision not in guardrail_policy.allowed_decisions:
            raise ValueError("physics debug report guardrail decision is unsupported")
        _validate_guardrail_provider_metadata(value.get("provider"))
        request = _validate_guardrail_request_metadata(value.get("request"))
        reviewed_output_sha256 = value.get("reviewed_output_sha256")
        if not isinstance(reviewed_output_sha256, str) or not _is_sha256(reviewed_output_sha256):
            raise ValueError("physics debug report guardrail reviewed_output_sha256 must be a SHA-256 hex digest")
        if reviewed_output_sha256.lower() != request["provider_output_sha256"]:
            raise ValueError("physics debug report guardrail reviewed_output_sha256 must match request")
    else:
        if guardrail_policy.required:
            raise ValueError("physics debug report guardrail is required")
        if value.get("provider") is not None:
            raise ValueError("physics debug report disabled guardrail provider must be null")
        if value.get("request") is not None:
            raise ValueError("physics debug report disabled guardrail request must be null")
        if value.get("reviewed_output_sha256") is not None:
            raise ValueError("physics debug report disabled guardrail reviewed_output_sha256 must be null")
        if decision != "not-configured":
            raise ValueError("physics debug report disabled guardrail decision must be not-configured")
    findings = _validate_guardrail_findings(
        value.get("findings"),
        evidence=evidence,
        guardrail_policy=guardrail_policy,
        require_findings=enabled,
    )
    if any(finding["action"] in guardrail_policy.block_actions for finding in findings):
        raise ValueError("physics debug report guardrail cannot persist blocked findings")
    risk_controls = value.get("risk_controls")
    if not isinstance(risk_controls, list) or any(not isinstance(item, str) for item in risk_controls):
        raise ValueError("physics debug report guardrail risk_controls must be a list of strings")
    if enabled and len(risk_controls) < guardrail_policy.minimum_risk_controls:
        raise ValueError("physics debug report guardrail risk_controls do not meet policy minimum")


def _validate_guardrail_request_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("physics debug guardrail request must be an object")
    result: dict[str, str] = {}
    for key in ("prompt_sha256", "evidence_sha256", "gaps_sha256", "provider_output_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise ValueError(f"physics debug guardrail request {key} must be a SHA-256 hex digest")
        result[key] = digest.lower()
    return result


def _validate_guardrail_provider_metadata(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("physics debug report guardrail provider must be an object")
    for key in ("provider_name", "endpoint", "model", "profile", "protocol"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"physics debug report guardrail provider {key} must be a non-empty string")
    if value["profile"] not in GUARDRAIL_PROVIDER_PROFILES:
        raise ValueError("physics debug report guardrail provider profile is unsupported")
    if value["protocol"] != "direct-json":
        raise ValueError("physics debug report guardrail provider protocol is unsupported")
    if not isinstance(value.get("local_onsite"), bool):
        raise ValueError("physics debug report guardrail provider local_onsite must be a bool")


def _validate_guardrail_findings(
    value: object,
    *,
    evidence: list[dict[str, Any]],
    guardrail_policy: PhysicsDebugGuardrailPolicy,
    require_findings: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("physics debug guardrail findings must be a list")
    if require_findings and not value:
        raise ValueError("physics debug guardrail findings must be non-empty")
    evidence_ids = {item["evidence_id"] for item in evidence}
    severities = {"info", "low", "medium", "high", "critical"}
    seen: set[str] = set()
    findings: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("physics debug guardrail finding must be an object")
        finding_id = _require_non_empty("guardrail finding_id", raw.get("finding_id"))
        if finding_id in seen:
            raise ValueError("physics debug guardrail finding IDs must be unique")
        seen.add(finding_id)
        severity = _require_non_empty("guardrail severity", raw.get("severity"))
        if severity not in severities:
            raise ValueError("physics debug guardrail severity is unsupported")
        message = _require_non_empty("guardrail message", raw.get("message"))
        linked_evidence_ids = _require_non_empty_string_list("guardrail evidence_ids", raw.get("evidence_ids"))
        missing = [evidence_id for evidence_id in linked_evidence_ids if evidence_id not in evidence_ids]
        if missing:
            raise ValueError("physics debug guardrail evidence_ids must cite existing evidence")
        action = _require_non_empty("guardrail action", raw.get("action"))
        if action not in guardrail_policy.allowed_decisions:
            raise ValueError("physics debug guardrail action is unsupported")
        if severity in guardrail_policy.blocking_severities and action not in guardrail_policy.block_actions:
            raise ValueError("physics debug guardrail severity requires a block action")
        findings.append(
            {
                "finding_id": finding_id,
                "severity": severity,
                "message": message,
                "evidence_ids": linked_evidence_ids,
                "action": action,
            }
        )
    return findings


def _coerce_provider_response(raw: Mapping[str, Any] | str | bytes, policy: ProviderPolicy) -> dict[str, Any]:
    if isinstance(raw, bytes):
        if len(raw) > policy.max_response_chars:
            raise ValueError("physics debug provider response exceeds policy max_response_chars")
        text = raw.decode("utf-8")
        parsed = json.loads(text)
    elif isinstance(raw, str):
        if len(raw) > policy.max_response_chars:
            raise ValueError("physics debug provider response exceeds policy max_response_chars")
        parsed = json.loads(raw)
    else:
        parsed = raw if isinstance(raw, list) else dict(raw)
    parsed = _extract_provider_json(parsed, policy)
    if not isinstance(parsed, dict):
        raise ValueError("physics debug provider response must be a JSON object")
    return parsed


def _extract_provider_json(parsed: Any, policy: ProviderPolicy) -> Any:
    if isinstance(parsed, dict) and "hypotheses" in parsed:
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("choices"), list) and parsed["choices"]:
        first = parsed["choices"][0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return _load_provider_text(message["content"], policy)
            if isinstance(first.get("text"), str):
                return _load_provider_text(first["text"], policy)
    if isinstance(parsed, dict):
        message = parsed.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return _load_provider_text(message["content"], policy)
        for key in ("response", "generated_text", "content"):
            if isinstance(parsed.get(key), str):
                return _load_provider_text(parsed[key], policy)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        first = parsed[0]
        for key in ("generated_text", "text", "content"):
            if isinstance(first.get(key), str):
                return _load_provider_text(first[key], policy)
    return parsed


def _load_provider_text(text: str, policy: ProviderPolicy) -> dict[str, Any]:
    if len(text) > policy.max_response_chars:
        raise ValueError("physics debug provider response exceeds policy max_response_chars")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("physics debug provider text response must decode to an object")
    return parsed


def _validate_provider_output(
    provider_output: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    safety_policy: PhysicsDebugSafetyPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hypotheses = _validate_hypotheses(provider_output.get("hypotheses"), evidence, gaps, safety_policy)
    campaigns = _validate_campaigns(provider_output.get("campaign_suggestions"), hypotheses, safety_policy)
    return hypotheses, campaigns


def _validate_hypotheses(
    raw_hypotheses: object,
    evidence: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    safety_policy: PhysicsDebugSafetyPolicy,
) -> list[dict[str, Any]]:
    if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
        raise ValueError("physics debug hypotheses must be a non-empty list")
    evidence_ids = {item["evidence_id"] for item in evidence}
    gap_ids = {item["gap_id"] for item in gaps}
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in raw_hypotheses:
        if not isinstance(raw, dict):
            raise ValueError("physics debug hypothesis must be an object")
        hypothesis_id = _require_non_empty("hypothesis_id", raw.get("hypothesis_id"))
        if hypothesis_id in seen:
            raise ValueError("physics debug hypothesis IDs must be unique")
        seen.add(hypothesis_id)
        gap_id = _require_non_empty("gap_id", raw.get("gap_id"))
        if gap_id not in gap_ids:
            raise ValueError("physics debug hypothesis gap_id must cite an existing gap")
        statement = _require_non_empty("statement", raw.get("statement"))
        falsification_test = _require_non_empty("falsification_test", raw.get("falsification_test"))
        required_evidence_ids = _require_non_empty_string_list(
            "required_evidence_ids", raw.get("required_evidence_ids")
        )
        missing = [evidence_id for evidence_id in required_evidence_ids if evidence_id not in evidence_ids]
        if missing:
            raise ValueError("physics debug hypothesis required_evidence_ids must cite existing evidence")
        confidence = raw.get("confidence")
        if not isinstance(confidence, int | float) or not math.isfinite(float(confidence)):
            raise ValueError("physics debug hypothesis confidence must be finite")
        confidence_value = float(confidence)
        if confidence_value < 0.0 or confidence_value > 1.0:
            raise ValueError("physics debug hypothesis confidence must be in [0, 1]")
        if confidence_value > safety_policy.max_advisory_confidence:
            raise ValueError("physics debug hypothesis confidence exceeds safety policy max_advisory_confidence")
        _enforce_advisory_safety_policy(statement, safety_policy)
        _enforce_advisory_safety_policy(falsification_test, safety_policy)
        validated.append(
            {
                "hypothesis_id": hypothesis_id,
                "gap_id": gap_id,
                "statement": statement,
                "falsification_test": falsification_test,
                "required_evidence_ids": required_evidence_ids,
                "confidence": confidence_value,
            }
        )
    return validated


def _validate_campaigns(
    raw_campaigns: object,
    hypotheses: list[dict[str, Any]],
    safety_policy: PhysicsDebugSafetyPolicy,
) -> list[dict[str, Any]]:
    if not isinstance(raw_campaigns, list) or not raw_campaigns:
        raise ValueError("physics debug campaign_suggestions must be a non-empty list")
    hypothesis_ids = {item["hypothesis_id"] for item in hypotheses}
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in raw_campaigns:
        if not isinstance(raw, dict):
            raise ValueError("physics debug campaign suggestion must be an object")
        campaign_id = _require_non_empty("campaign_id", raw.get("campaign_id"))
        if campaign_id in seen:
            raise ValueError("physics debug campaign IDs must be unique")
        seen.add(campaign_id)
        linked_hypothesis_ids = _require_non_empty_string_list(
            "linked_hypothesis_ids", raw.get("linked_hypothesis_ids")
        )
        missing = [hypothesis_id for hypothesis_id in linked_hypothesis_ids if hypothesis_id not in hypothesis_ids]
        if missing:
            raise ValueError("physics debug campaign linked_hypothesis_ids must cite existing hypotheses")
        objective = _require_non_empty("objective", raw.get("objective"))
        measurements = _require_non_empty_string_list("measurements", raw.get("measurements"))
        stop_conditions = _require_non_empty_string_list("stop_conditions", raw.get("stop_conditions"))
        risk_controls = _require_non_empty_string_list("risk_controls", raw.get("risk_controls"))
        _enforce_advisory_safety_policy(objective, safety_policy)
        for value in [*measurements, *stop_conditions, *risk_controls]:
            _enforce_advisory_safety_policy(value, safety_policy)
        validated.append(
            {
                "campaign_id": campaign_id,
                "linked_hypothesis_ids": linked_hypothesis_ids,
                "objective": objective,
                "measurements": measurements,
                "stop_conditions": stop_conditions,
                "risk_controls": risk_controls,
            }
        )
    return validated


def _safety_policy_from_payload(value: object) -> PhysicsDebugSafetyPolicy:
    if not isinstance(value, dict):
        raise ValueError("physics debug safety_policy must be an object")
    forbidden = value.get("forbidden_action_phrases")
    if not isinstance(forbidden, list | tuple):
        raise ValueError("physics debug safety_policy forbidden_action_phrases must be a list")
    return PhysicsDebugSafetyPolicy(
        human_review_required=value.get("human_review_required", True),
        max_advisory_confidence=value.get("max_advisory_confidence", 0.95),
        forbidden_action_phrases=tuple(forbidden),
    )


def _guardrail_policy_from_payload(value: object) -> PhysicsDebugGuardrailPolicy:
    if not isinstance(value, dict):
        raise ValueError("physics debug guardrail_policy must be an object")
    block_actions = value.get("block_actions")
    allowed_decisions = value.get("allowed_decisions")
    blocking_severities = value.get("blocking_severities", ("high", "critical"))
    if not isinstance(block_actions, list | tuple):
        raise ValueError("physics debug guardrail_policy block_actions must be a list")
    if not isinstance(allowed_decisions, list | tuple):
        raise ValueError("physics debug guardrail_policy allowed_decisions must be a list")
    if not isinstance(blocking_severities, list | tuple):
        raise ValueError("physics debug guardrail_policy blocking_severities must be a list")
    return PhysicsDebugGuardrailPolicy(
        required=value.get("required", False),
        block_actions=tuple(block_actions),
        allowed_decisions=tuple(allowed_decisions),
        blocking_severities=tuple(blocking_severities),
        minimum_risk_controls=value.get("minimum_risk_controls", 1),
    )


def _enforce_advisory_safety_policy(value: str, safety_policy: PhysicsDebugSafetyPolicy) -> None:
    lower_value = value.casefold()
    for phrase in safety_policy.forbidden_action_phrases:
        if phrase.casefold() in lower_value:
            raise ValueError("physics debug safety policy rejects provider action language")


def _validate_evidence_gap_linkage(evidence: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> None:
    evidence_ids: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("physics debug evidence items must be objects")
        evidence_id = _require_non_empty("evidence_id", item.get("evidence_id"))
        if evidence_id in evidence_ids:
            raise ValueError("physics debug evidence IDs must be unique")
        evidence_ids.add(evidence_id)
        _require_non_empty("evidence_type", item.get("evidence_type"))
        _require_non_empty("source", item.get("source"))
        _require_non_empty("summary", item.get("summary"))
        sha256 = item.get("sha256")
        if sha256 is not None and (not isinstance(sha256, str) or not _is_sha256(sha256)):
            raise ValueError("physics debug evidence sha256 must be a SHA-256 hex digest")
    gap_ids: set[str] = set()
    for item in gaps:
        if not isinstance(item, dict):
            raise ValueError("physics debug gap items must be objects")
        gap_id = _require_non_empty("gap_id", item.get("gap_id"))
        if gap_id in gap_ids:
            raise ValueError("physics debug gap IDs must be unique")
        gap_ids.add(gap_id)
        if item.get("severity") not in {"low", "medium", "high", "critical"}:
            raise ValueError("physics debug gap severity is unsupported")
        _require_non_empty("description", item.get("description"))
        linked = _require_non_empty_string_list("evidence_ids", item.get("evidence_ids"))
        missing = [evidence_id for evidence_id in linked if evidence_id not in evidence_ids]
        if missing:
            raise ValueError("physics debug gap evidence_ids must cite existing evidence")


def _redact_sensitive_text(value: str) -> tuple[str, list[str]]:
    redactions: list[str] = []
    text = value
    patterns = (
        (
            "secret_assignment",
            re.compile(r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD))\s*[:=]\s*([^\s,;]+)"),
            lambda match: f"{match.group(1)}=<redacted>",
        ),
        (
            "bearer_token",
            re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
            lambda match: "Bearer <redacted>",
        ),
    )
    for label, pattern, replacement in patterns:
        updated, count = pattern.subn(replacement, text)
        if count:
            redactions.append(label)
        text = updated
    return text, redactions


def _neutralize_prompt_injection_text(value: str) -> tuple[str, list[str]]:
    findings: list[str] = []
    text = value
    for label, pattern in PROMPT_INJECTION_PATTERNS:
        text, count = pattern.subn(PROMPT_INJECTION_REDACTION, text)
        if count:
            findings.append(label)
    return text, findings


def _require_non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"physics debug {name} must be a non-empty string")
    return value


def _require_non_empty_string_list(name: str, value: object) -> list[str]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"physics debug {name} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"physics debug {name} must contain non-empty strings")
        result.append(item)
    return result


def _payload_digest(payload: Mapping[str, Any]) -> str:
    digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(_jsonable(digest_payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)
