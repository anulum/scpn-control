# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Published SPO device diagnostic review admission.

"""Fail-closed admission of one exact published SPO diagnostic review."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from scpn_phase_orchestrator.reactor_semantics import DeviceDiagnosticPlanReview
else:
    DeviceDiagnosticPlanReview = Any

from .device_diagnostic_review_decision import (
    DeviceDiagnosticReviewAdmissionDecision,
    DeviceDiagnosticReviewAdmissionStatus,
)

SPO_DISTRIBUTION: Final = "scpn-phase-orchestrator"
SPO_DISTRIBUTION_VERSION: Final = "1.4.3"
SPO_RELEASE_WHEEL_SHA256: Final = "5da94500760f9394a637f7edec044a844c12230d8d16c25a573b6e67a1ddb409"
SPO_CONTRACT_MODULE_SHA256: Final = "8e8c9cd3f253da1bf52983127fdb7d643822bbf14ec27e7081068e42642d3a33"
SPO_REVIEW_SCHEMA: Final = "scpn-phase-orchestrator.device-diagnostic-plan-review.v1"
SPO_REVIEW_SCHEMA_VERSION: Final = "1.1.0"

TOKAMAK_REVIEW_SHA256: Final = "de5573351115ae6d28787aadfec1106f037e5cf8c68206ea5074255055f5e795"
TOKAMAK_REVIEW_ID: Final = "55cec93310c72602abefde47f200c0f3fb32c834fa57cd4c5ff70923c48ce53c"
TOKAMAK_SOURCE_PROJECT: Final = "SCPN-TOKAMAK-CORE"
TOKAMAK_SOURCE_REVISION: Final = "7402191c43e8fe57cffda1dd5b3cf4319d6d398d"
TOKAMAK_SOURCE_ARTIFACT_SHA256: Final = "a0c6ccbf8c398d80ed65f03a82e7a313761d09dea81acb5ab8ad565997cb2720"
TOKAMAK_SOURCE_MANIFEST_SHA256: Final = "ed4dd4f86eb7a62bf9674c0bfffa341f3afed42b7754ddcd809bc0b1804a19ab"
TOKAMAK_SOURCE_ENVELOPE_SHA256: Final = "5a5aff510df9bdfdf35f76edfc9f147dae79e95f4edd98c3a610a60d8ea9e28a"
TOKAMAK_SOURCE_PLAN_SHA256: Final = "6a015adfaa2cda7ec1bf04fc685d00d6b7209ca9b78d761fff47ab0919eeec94"
TOKAMAK_CLOCK_CUSTODY_SHA256: Final = "29eb15ecd40efb911b4150004fd7b0252ab2ee890c971e5aaab3d9abd3263478"
TOKAMAK_CONFIGURATIONS: Final = ("conventional_tokamak", "spherical_tokamak")

_HEX_40: Final = re.compile(r"^[0-9a-f]{40}$")
_HEX_64: Final = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DeviceDiagnosticReviewAdmissionPolicy:
    """Exact caller-owned binding for one sealed device diagnostic review.

    Parameters
    ----------
    expected_review_sha256, expected_review_id : str
        Exact canonical review envelope digest and upstream review identity.
    expected_source_project, expected_source_revision : str
        Exact device owner and producer Git revision.
    expected_source_artifact_sha256 : str
        Producer wheel digest supplied to the SPO review.
    expected_source_manifest_sha256, expected_source_envelope_sha256,
    expected_source_plan_sha256 : str
        Complete source-document custody digests exposed by the sealed review.
    expected_configurations : tuple[str, ...]
        Sorted exact device configurations admitted by this consumer.
    expected_clock_custody_sha256 : str
        Digest over every ordered SPO clock-review field.

    """

    expected_review_sha256: str
    expected_review_id: str
    expected_source_project: str
    expected_source_revision: str
    expected_source_artifact_sha256: str
    expected_source_manifest_sha256: str
    expected_source_envelope_sha256: str
    expected_source_plan_sha256: str
    expected_configurations: tuple[str, ...]
    expected_clock_custody_sha256: str

    def __post_init__(self) -> None:
        """Validate all expected identities before accepting review bytes."""
        for name in (
            "expected_review_sha256",
            "expected_review_id",
            "expected_source_artifact_sha256",
            "expected_source_manifest_sha256",
            "expected_source_envelope_sha256",
            "expected_source_plan_sha256",
            "expected_clock_custody_sha256",
        ):
            _digest(getattr(self, name), name)
        if _HEX_40.fullmatch(self.expected_source_revision) is None:
            raise ValueError("expected_source_revision must be a lowercase Git commit")
        if not isinstance(self.expected_source_project, str) or not self.expected_source_project:
            raise ValueError("expected_source_project must be non-empty text")
        if not isinstance(self.expected_configurations, tuple) or any(
            not isinstance(value, str) or not value for value in self.expected_configurations
        ):
            raise ValueError("expected_configurations must be a tuple of non-empty strings")
        if tuple(sorted(set(self.expected_configurations))) != self.expected_configurations:
            raise ValueError("expected_configurations must be sorted and unique")


def tokamak_device_diagnostic_review_policy() -> DeviceDiagnosticReviewAdmissionPolicy:
    """Return the immutable published Tokamak review binding for SPO 1.4.3."""
    return DeviceDiagnosticReviewAdmissionPolicy(
        expected_review_sha256=TOKAMAK_REVIEW_SHA256,
        expected_review_id=TOKAMAK_REVIEW_ID,
        expected_source_project=TOKAMAK_SOURCE_PROJECT,
        expected_source_revision=TOKAMAK_SOURCE_REVISION,
        expected_source_artifact_sha256=TOKAMAK_SOURCE_ARTIFACT_SHA256,
        expected_source_manifest_sha256=TOKAMAK_SOURCE_MANIFEST_SHA256,
        expected_source_envelope_sha256=TOKAMAK_SOURCE_ENVELOPE_SHA256,
        expected_source_plan_sha256=TOKAMAK_SOURCE_PLAN_SHA256,
        expected_configurations=TOKAMAK_CONFIGURATIONS,
        expected_clock_custody_sha256=TOKAMAK_CLOCK_CUSTODY_SHA256,
    )


def admit_device_diagnostic_plan_review(
    payload: bytes,
    *,
    policy: DeviceDiagnosticReviewAdmissionPolicy,
) -> DeviceDiagnosticReviewAdmissionDecision:
    """Decode and bind exact public SPO review bytes without granting authority.

    The SPO public decoder is the sole review ingress. CONTROL never parses the
    embedded producer manifest, diagnostic envelope, or plan. Decoder,
    installation, identity, custody, clock, and authority drift all return a
    sealed rejection with every operational-authority field fixed false.
    """
    raw_digest = hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else None
    try:
        decoder, module_digest = _load_spo_decoder()
    except _SPOContractError as exc:  # pragma: no cover - SPO installation guard
        return _decision(
            review=None,
            review_sha256=raw_digest,
            module_digest=exc.module_digest,
            refusal_codes={exc.code},
        )
    try:
        review = decoder(payload)
    except (TypeError, ValueError):
        return _decision(
            review=None,
            review_sha256=raw_digest,
            module_digest=module_digest,
            refusal_codes={"review_decode_failed"},
        )
    refusal_codes = _evaluate_review(review, review_sha256=raw_digest, policy=policy)
    return _decision(
        review=review,
        review_sha256=raw_digest,
        module_digest=module_digest,
        refusal_codes=refusal_codes,
    )


def device_diagnostic_review_clock_custody_digest(
    review: DeviceDiagnosticPlanReview,
) -> str:
    """Digest every ordered field of all public SPO clock-review records."""
    records = [
        {
            "compatibility": item.compatibility.value,
            "epoch": item.epoch,
            "mapping_evidence_claimed": item.mapping_evidence_claimed,
            "plan_clock_identifier": item.plan_clock_identifier,
            "plan_clock_kind": item.plan_clock_kind,
            "resolution_s": item.resolution_s,
            "spo_clock_kind_candidate": (
                None if item.spo_clock_kind_candidate is None else item.spo_clock_kind_candidate.value
            ),
            "uncertainty_s": item.uncertainty_s,
        }
        for item in review.clock_reviews
    ]
    canonical = json.dumps(records, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


class _SPOContractError(RuntimeError):
    def __init__(  # pragma: no cover - SPO installation guard
        self, code: str, *, module_digest: str | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.module_digest = module_digest


def _load_spo_decoder() -> tuple[Callable[[bytes], DeviceDiagnosticPlanReview], str]:
    try:
        installed_version = metadata.version(SPO_DISTRIBUTION)
        if installed_version != SPO_DISTRIBUTION_VERSION:  # pragma: no cover - SPO installation guard
            raise _SPOContractError("spo_distribution_version_mismatch")
        facade = import_module("scpn_phase_orchestrator.reactor_semantics")
        contract_module = import_module("scpn_phase_orchestrator.reactor_semantics.diagnostic_plan_review")
    except (  # pragma: no cover - SPO installation guard
        ImportError,
        metadata.PackageNotFoundError,
    ) as exc:
        raise _SPOContractError("spo_contract_unavailable") from exc
    source_path = getattr(contract_module, "__file__", None)
    if not isinstance(source_path, str):  # pragma: no cover - SPO installation guard
        raise _SPOContractError("spo_contract_unavailable")
    try:
        module_digest = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - SPO installation guard
        raise _SPOContractError("spo_contract_unavailable") from exc
    if module_digest != SPO_CONTRACT_MODULE_SHA256:  # pragma: no cover - SPO installation guard
        raise _SPOContractError("spo_contract_module_digest_mismatch", module_digest=module_digest)
    try:
        decoder = facade.device_diagnostic_plan_review_from_bytes
        contract_decoder = contract_module.device_diagnostic_plan_review_from_bytes
        schema = facade.DEVICE_DIAGNOSTIC_PLAN_REVIEW_SCHEMA
        schema_version = facade.DEVICE_DIAGNOSTIC_PLAN_REVIEW_VERSION
    except AttributeError as exc:  # pragma: no cover - SPO installation guard
        raise _SPOContractError("spo_contract_unavailable", module_digest=module_digest) from exc
    if (  # pragma: no cover - SPO installation guard
        decoder is not contract_decoder or schema != SPO_REVIEW_SCHEMA or schema_version != SPO_REVIEW_SCHEMA_VERSION
    ):
        raise _SPOContractError("spo_contract_unavailable", module_digest=module_digest)
    return cast("Callable[[bytes], DeviceDiagnosticPlanReview]", decoder), module_digest


def _evaluate_review(
    review: DeviceDiagnosticPlanReview,
    *,
    review_sha256: str | None,
    policy: DeviceDiagnosticReviewAdmissionPolicy,
) -> set[str]:
    refusals: set[str] = set()
    if review_sha256 != policy.expected_review_sha256:
        refusals.add("review_digest_mismatch")
    if review.review_id != policy.expected_review_id:
        refusals.add("review_identity_mismatch")
    if review.source_project != policy.expected_source_project:
        refusals.add("source_project_mismatch")
    if review.source_revision != policy.expected_source_revision:
        refusals.add("source_revision_mismatch")
    if review.source_artifact_sha256 != policy.expected_source_artifact_sha256:
        refusals.add("source_artifact_mismatch")
    if review.source_manifest_sha256 != policy.expected_source_manifest_sha256:
        refusals.add("source_manifest_digest_mismatch")
    if review.source_envelope_sha256 != policy.expected_source_envelope_sha256:
        refusals.add("source_envelope_digest_mismatch")
    if review.source_plan_sha256 != policy.expected_source_plan_sha256:
        refusals.add("source_plan_digest_mismatch")
    if review.configurations != policy.expected_configurations:
        refusals.add("configuration_mismatch")
    if device_diagnostic_review_clock_custody_digest(review) != (policy.expected_clock_custody_sha256):
        refusals.add("clock_custody_mismatch")
    if (  # pragma: no branch - pinned SPO decoder already enforces these invariants
        review.accepted_as_design_declaration is not True
        or review.evidence_claimed is not False
        or review.observation_claimed is not False
        or review.measurement_claimed is not False
        or review.facility_binding_claimed is not False
        or review.classification_performed is not False
        or review.semantic_ingress_declared is not False
        or review.control_intent_created is not False
        or review.authority != "review_only"
        or review.actionable is not False
        or any(item.mapping_evidence_claimed is not False for item in review.clock_reviews)
        or any(
            item.synthetic is not True or item.evidence_claimed is not False or item.observation_claimed is not False
            for item in review.signal_reviews
        )
    ):
        refusals.add("authority_escalation")  # pragma: no cover - decoder invariant
    return refusals


def _decision(
    *,
    review: DeviceDiagnosticPlanReview | None,
    review_sha256: str | None,
    module_digest: str | None,
    refusal_codes: set[str],
) -> DeviceDiagnosticReviewAdmissionDecision:
    accepted = not refusal_codes
    return DeviceDiagnosticReviewAdmissionDecision(
        decision=(
            DeviceDiagnosticReviewAdmissionStatus.ACCEPTED_FOR_REVIEW
            if accepted
            else DeviceDiagnosticReviewAdmissionStatus.REJECTED
        ),
        accepted_for_review=accepted,
        review_sha256=review_sha256,
        review_id=None if review is None else review.review_id,
        review_schema=SPO_REVIEW_SCHEMA,
        review_schema_version=SPO_REVIEW_SCHEMA_VERSION,
        source_project=None if review is None else review.source_project,
        source_revision=None if review is None else review.source_revision,
        source_artifact_sha256=(None if review is None else review.source_artifact_sha256),
        source_manifest_sha256=(None if review is None else review.source_manifest_sha256),
        source_envelope_sha256=(None if review is None else review.source_envelope_sha256),
        source_plan_sha256=None if review is None else review.source_plan_sha256,
        configurations=() if review is None else review.configurations,
        clock_custody_sha256=(None if review is None else device_diagnostic_review_clock_custody_digest(review)),
        spo_distribution_version=SPO_DISTRIBUTION_VERSION,
        spo_release_wheel_sha256=SPO_RELEASE_WHEEL_SHA256,
        spo_contract_module_sha256=module_digest,
        refusal_codes=tuple(sorted(refusal_codes)),
    )


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "SPO_CONTRACT_MODULE_SHA256",
    "SPO_DISTRIBUTION",
    "SPO_DISTRIBUTION_VERSION",
    "SPO_RELEASE_WHEEL_SHA256",
    "SPO_REVIEW_SCHEMA",
    "SPO_REVIEW_SCHEMA_VERSION",
    "TOKAMAK_CLOCK_CUSTODY_SHA256",
    "TOKAMAK_CONFIGURATIONS",
    "TOKAMAK_REVIEW_ID",
    "TOKAMAK_REVIEW_SHA256",
    "TOKAMAK_SOURCE_ARTIFACT_SHA256",
    "TOKAMAK_SOURCE_ENVELOPE_SHA256",
    "TOKAMAK_SOURCE_MANIFEST_SHA256",
    "TOKAMAK_SOURCE_PLAN_SHA256",
    "TOKAMAK_SOURCE_PROJECT",
    "TOKAMAK_SOURCE_REVISION",
    "DeviceDiagnosticReviewAdmissionPolicy",
    "admit_device_diagnostic_plan_review",
    "device_diagnostic_review_clock_custody_digest",
    "tokamak_device_diagnostic_review_policy",
]
