#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Versioned MAST disruption shot-label contract
"""Represent disruption outcomes, event times, and label authority separately.

The contract prevents a deterministic Ip-current threshold from masquerading as
facility ground truth. It records label authority, source and algorithm digests,
programme class, nullable event times, review identities, and an explicit
confidence method. Ip-derived records are always ``ip_proxy`` and uncalibrated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from validation.mast_source_object_manifest import array_value_sha256, canonical_json_sha256

SHOT_LABEL_RECORD_SCHEMA = "scpn-control.mast-shot-label-record.v1.0.0"
IP_QUENCH_PROXY_ALGORITHM_SCHEMA = "scpn-control.ip-quench-proxy-algorithm.v1.0.0"
IP_PROXY_SOURCE_CHANNELS = ("Ip_MA", "time_s")
FLAT_TOP_FRACTION = 0.8

Outcome = Literal["disruption", "non_disruption", "aborted_no_plasma", "ambiguous"]
ProgrammeClass = Literal["spontaneous", "forced_vde", "control", "other", "unknown"]
LabelAuthority = Literal[
    "facility",
    "defuse",
    "independent_manual_review",
    "programme_metadata",
    "ip_proxy",
]
ConfidenceKind = Literal["calibrated_probability", "bounded_score", "unquantified"]

OUTCOMES = frozenset({"disruption", "non_disruption", "aborted_no_plasma", "ambiguous"})
PROGRAMME_CLASSES = frozenset({"spontaneous", "forced_vde", "control", "other", "unknown"})
LABEL_AUTHORITIES = frozenset({"facility", "defuse", "independent_manual_review", "programme_metadata", "ip_proxy"})
CONFIDENCE_KINDS = frozenset({"calibrated_probability", "bounded_score", "unquantified"})
_SHA256_LENGTH = 64


class ShotLabelRecordError(ValueError):
    """Raised when a shot-label record violates the versioned contract."""


@dataclass(frozen=True)
class LabelConfidence:
    """Confidence value and the method that gives the value meaning."""

    kind: ConfidenceKind
    value: float | None
    method: str

    def __post_init__(self) -> None:
        """Reject numeric confidence without bounded semantics."""
        if self.kind not in CONFIDENCE_KINDS:
            raise ShotLabelRecordError(f"unsupported confidence kind {self.kind!r}")
        if not self.method.strip():
            raise ShotLabelRecordError("confidence method must be non-empty")
        if self.kind == "unquantified":
            if self.value is not None:
                raise ShotLabelRecordError("unquantified confidence cannot carry a numeric value")
            return
        if self.value is None or not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise ShotLabelRecordError("quantified confidence must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-ready confidence representation."""
        return {"kind": self.kind, "value": self.value, "method": self.method}


@dataclass(frozen=True)
class ShotLabelRecord:
    """One provenance-bound disruption label for one machine shot."""

    schema_version: str
    machine: str
    shot_id: int
    outcome: Outcome
    programme_class: ProgrammeClass
    thermal_quench_time_s: float | None
    current_quench_time_s: float | None
    termination_time_s: float | None
    authority: LabelAuthority
    label_detail: str
    confidence: LabelConfidence
    reviewers: tuple[str, ...]
    source_channels: tuple[str, ...]
    source_digest: str
    algorithm_digest: str

    def __post_init__(self) -> None:
        """Validate taxonomy, timing, review, and provenance invariants."""
        if self.schema_version != SHOT_LABEL_RECORD_SCHEMA:
            raise ShotLabelRecordError(f"schema_version must equal {SHOT_LABEL_RECORD_SCHEMA!r}")
        if not self.machine.strip() or self.shot_id <= 0:
            raise ShotLabelRecordError("machine must be non-empty and shot_id must be positive")
        if self.outcome not in OUTCOMES:
            raise ShotLabelRecordError(f"unsupported outcome {self.outcome!r}")
        if self.programme_class not in PROGRAMME_CLASSES:
            raise ShotLabelRecordError(f"unsupported programme_class {self.programme_class!r}")
        if self.authority not in LABEL_AUTHORITIES:
            raise ShotLabelRecordError(f"unsupported authority {self.authority!r}")
        if not self.label_detail.strip():
            raise ShotLabelRecordError("label_detail must be non-empty")
        self._validate_times()
        self._validate_reviewers()
        if not self.source_channels or tuple(sorted(set(self.source_channels))) != self.source_channels:
            raise ShotLabelRecordError("source_channels must be non-empty, unique, and sorted")
        for field, digest in (("source_digest", self.source_digest), ("algorithm_digest", self.algorithm_digest)):
            if len(digest) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in digest):
                raise ShotLabelRecordError(f"{field} must be a lowercase SHA-256 digest")
        if self.authority == "ip_proxy":
            if self.source_channels != IP_PROXY_SOURCE_CHANNELS:
                raise ShotLabelRecordError("ip_proxy records must identify only Ip_MA and time_s as sources")
            if self.confidence.kind != "unquantified":
                raise ShotLabelRecordError("ip_proxy confidence must remain unquantified")
        if self.authority == "programme_metadata" and (
            self.thermal_quench_time_s is not None or self.current_quench_time_s is not None
        ):
            raise ShotLabelRecordError("programme_metadata authority cannot assert TQ or CQ event times")

    def _validate_times(self) -> None:
        times = (
            ("thermal_quench_time_s", self.thermal_quench_time_s),
            ("current_quench_time_s", self.current_quench_time_s),
            ("termination_time_s", self.termination_time_s),
        )
        for field, value in times:
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ShotLabelRecordError(f"{field} must be finite and non-negative when present")
        if self.outcome != "disruption" and (
            self.thermal_quench_time_s is not None or self.current_quench_time_s is not None
        ):
            raise ShotLabelRecordError("only disruption outcomes may carry TQ or CQ event times")
        if (
            self.thermal_quench_time_s is not None
            and self.current_quench_time_s is not None
            and self.thermal_quench_time_s > self.current_quench_time_s
        ):
            raise ShotLabelRecordError("thermal_quench_time_s cannot follow current_quench_time_s")
        event_times = [value for value in (self.thermal_quench_time_s, self.current_quench_time_s) if value is not None]
        if self.termination_time_s is not None and any(value > self.termination_time_s for value in event_times):
            raise ShotLabelRecordError("quench event times cannot follow termination_time_s")

    def _validate_reviewers(self) -> None:
        if tuple(sorted(set(self.reviewers))) != self.reviewers or any(
            not reviewer.strip() for reviewer in self.reviewers
        ):
            raise ShotLabelRecordError("reviewers must be non-empty, unique, and sorted")
        if self.authority == "independent_manual_review":
            if len(self.reviewers) < 2:
                raise ShotLabelRecordError("independent_manual_review requires at least two reviewers")
        elif self.reviewers:
            raise ShotLabelRecordError("reviewers are accepted only for independent_manual_review authority")

    @property
    def independent_of_ip_features(self) -> bool:
        """Return whether the label authority is independent of Ip-derived features."""
        return self.authority in {"facility", "defuse", "independent_manual_review"}

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, self-digested JSON-ready record."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "machine": self.machine,
            "shot_id": self.shot_id,
            "outcome": self.outcome,
            "programme_class": self.programme_class,
            "thermal_quench_time_s": self.thermal_quench_time_s,
            "current_quench_time_s": self.current_quench_time_s,
            "termination_time_s": self.termination_time_s,
            "authority": self.authority,
            "label_detail": self.label_detail,
            "confidence": self.confidence.to_dict(),
            "reviewers": list(self.reviewers),
            "source_channels": list(self.source_channels),
            "source_digest": self.source_digest,
            "algorithm_digest": self.algorithm_digest,
            "independent_of_ip_features": self.independent_of_ip_features,
            "payload_sha256": None,
        }
        payload["payload_sha256"] = canonical_json_sha256(payload)
        return payload


@dataclass(frozen=True)
class IpQuenchProxyResult:
    """Ip threshold result plus its authoritative shot-label record."""

    is_disruption: bool
    onset_index: int
    termination_index: int
    classification: str
    record: ShotLabelRecord


def ip_quench_proxy_algorithm(*, drop_fraction: float, quench_window_ms: float) -> dict[str, object]:
    """Return the self-digested deterministic Ip-proxy algorithm contract."""
    if not math.isfinite(drop_fraction) or not 0.0 < drop_fraction < 1.0:
        raise ShotLabelRecordError("drop_fraction must be finite and within (0, 1)")
    if not math.isfinite(quench_window_ms) or quench_window_ms <= 0.0:
        raise ShotLabelRecordError("quench_window_ms must be finite and positive")
    payload: dict[str, object] = {
        "schema_version": IP_QUENCH_PROXY_ALGORITHM_SCHEMA,
        "authority": "ip_proxy",
        "inputs": list(IP_PROXY_SOURCE_CHANNELS),
        "flat_top_fraction": FLAT_TOP_FRACTION,
        "drop_fraction": drop_fraction,
        "quench_window_ms": quench_window_ms,
        "onset_definition": "last sample at or above flat_top_fraction times maximum absolute Ip",
        "termination_definition": "first subsequent sample below one minus drop_fraction times maximum absolute Ip",
        "missing_ground_truth": "does not infer thermal-quench time or independent facility outcome",
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def derive_ip_quench_proxy(
    ip_ma: NDArray[np.float64],
    time_s: NDArray[np.float64],
    *,
    shot_id: int,
    programme_class: ProgrammeClass = "unknown",
    drop_fraction: float = 0.8,
    quench_window_ms: float = 5.0,
) -> IpQuenchProxyResult:
    """Derive an explicit, uncalibrated Ip-proxy record from one shot trace."""
    ip_values = np.asarray(ip_ma, dtype=np.float64)
    time_values = np.asarray(time_s, dtype=np.float64)
    _validate_proxy_arrays(ip_values, time_values)
    algorithm = ip_quench_proxy_algorithm(
        drop_fraction=drop_fraction,
        quench_window_ms=quench_window_ms,
    )
    source_digest = canonical_json_sha256(
        {
            "Ip_MA": array_value_sha256(ip_values),
            "time_s": array_value_sha256(time_values),
        }
    )
    ip_abs = np.abs(ip_values)
    ip_max = float(ip_abs.max())
    if ip_max <= 0.0:
        return _proxy_result(
            shot_id=shot_id,
            programme_class=programme_class,
            outcome="ambiguous",
            classification="no_current",
            onset_index=-1,
            termination_index=-1,
            time_s=time_values,
            source_digest=source_digest,
            algorithm_digest=cast(str, algorithm["payload_sha256"]),
        )
    onset = int(np.nonzero(ip_abs >= FLAT_TOP_FRACTION * ip_max)[0][-1])
    low_threshold = (1.0 - drop_fraction) * ip_max
    collapsed = np.nonzero(ip_abs[onset:] < low_threshold)[0]
    if collapsed.size == 0:
        return _proxy_result(
            shot_id=shot_id,
            programme_class=programme_class,
            outcome="non_disruption",
            classification="no_quench",
            onset_index=-1,
            termination_index=-1,
            time_s=time_values,
            source_digest=source_digest,
            algorithm_digest=cast(str, algorithm["payload_sha256"]),
        )
    termination = onset + int(collapsed[0])
    quench_ms = float((time_values[termination] - time_values[onset]) * 1000.0)
    if quench_ms <= quench_window_ms:
        return _proxy_result(
            shot_id=shot_id,
            programme_class=programme_class,
            outcome="disruption",
            classification="current_quench",
            onset_index=onset,
            termination_index=termination,
            time_s=time_values,
            source_digest=source_digest,
            algorithm_digest=cast(str, algorithm["payload_sha256"]),
        )
    return _proxy_result(
        shot_id=shot_id,
        programme_class=programme_class,
        outcome="non_disruption",
        classification="slow_rampdown",
        onset_index=-1,
        termination_index=termination,
        time_s=time_values,
        source_digest=source_digest,
        algorithm_digest=cast(str, algorithm["payload_sha256"]),
    )


def _proxy_result(
    *,
    shot_id: int,
    programme_class: ProgrammeClass,
    outcome: Outcome,
    classification: str,
    onset_index: int,
    termination_index: int,
    time_s: NDArray[np.float64],
    source_digest: str,
    algorithm_digest: str,
) -> IpQuenchProxyResult:
    current_quench_time = float(time_s[onset_index]) if outcome == "disruption" else None
    termination_time = float(time_s[termination_index]) if termination_index >= 0 else None
    record = ShotLabelRecord(
        schema_version=SHOT_LABEL_RECORD_SCHEMA,
        machine="MAST",
        shot_id=shot_id,
        outcome=outcome,
        programme_class=programme_class,
        thermal_quench_time_s=None,
        current_quench_time_s=current_quench_time,
        termination_time_s=termination_time,
        authority="ip_proxy",
        label_detail=f"ip_current_quench:{classification}",
        confidence=LabelConfidence(
            kind="unquantified",
            value=None,
            method="deterministic threshold proxy; no calibrated probability",
        ),
        reviewers=(),
        source_channels=IP_PROXY_SOURCE_CHANNELS,
        source_digest=source_digest,
        algorithm_digest=algorithm_digest,
    )
    return IpQuenchProxyResult(
        is_disruption=outcome == "disruption",
        onset_index=onset_index,
        termination_index=termination_index,
        classification=classification,
        record=record,
    )


def _validate_proxy_arrays(ip_ma: NDArray[np.float64], time_s: NDArray[np.float64]) -> None:
    if ip_ma.ndim != 1 or time_s.ndim != 1 or ip_ma.shape != time_s.shape or ip_ma.size == 0:
        raise ShotLabelRecordError("Ip_MA and time_s must be non-empty 1-D arrays with identical shape")
    if not bool(np.all(np.isfinite(ip_ma))) or not bool(np.all(np.isfinite(time_s))):
        raise ShotLabelRecordError("Ip_MA and time_s must be finite")
    if not bool(np.all(np.diff(time_s) > 0.0)):
        raise ShotLabelRecordError("time_s must be strictly increasing")


def shot_label_record_from_dict(payload: Mapping[str, object]) -> ShotLabelRecord:
    """Validate a self-digested JSON mapping and return its immutable record."""
    expected = {
        "schema_version",
        "machine",
        "shot_id",
        "outcome",
        "programme_class",
        "thermal_quench_time_s",
        "current_quench_time_s",
        "termination_time_s",
        "authority",
        "label_detail",
        "confidence",
        "reviewers",
        "source_channels",
        "source_digest",
        "algorithm_digest",
        "independent_of_ip_features",
        "payload_sha256",
    }
    if set(payload) != expected:
        raise ShotLabelRecordError("shot-label payload fields do not match the schema")
    digest = _required_string(payload, "payload_sha256")
    digest_input = dict(payload)
    digest_input["payload_sha256"] = None
    if canonical_json_sha256(digest_input) != digest:
        raise ShotLabelRecordError("shot-label payload_sha256 does not match the payload")
    confidence_payload = payload["confidence"]
    if not isinstance(confidence_payload, Mapping) or set(confidence_payload) != {"kind", "value", "method"}:
        raise ShotLabelRecordError("confidence must contain exactly kind, value, and method")
    confidence = LabelConfidence(
        kind=cast(ConfidenceKind, _required_string(confidence_payload, "kind")),
        value=_optional_float(confidence_payload, "value"),
        method=_required_string(confidence_payload, "method"),
    )
    record = ShotLabelRecord(
        schema_version=_required_string(payload, "schema_version"),
        machine=_required_string(payload, "machine"),
        shot_id=_required_int(payload, "shot_id"),
        outcome=cast(Outcome, _required_string(payload, "outcome")),
        programme_class=cast(ProgrammeClass, _required_string(payload, "programme_class")),
        thermal_quench_time_s=_optional_float(payload, "thermal_quench_time_s"),
        current_quench_time_s=_optional_float(payload, "current_quench_time_s"),
        termination_time_s=_optional_float(payload, "termination_time_s"),
        authority=cast(LabelAuthority, _required_string(payload, "authority")),
        label_detail=_required_string(payload, "label_detail"),
        confidence=confidence,
        reviewers=_string_tuple(payload, "reviewers"),
        source_channels=_string_tuple(payload, "source_channels"),
        source_digest=_required_string(payload, "source_digest"),
        algorithm_digest=_required_string(payload, "algorithm_digest"),
    )
    independent = payload["independent_of_ip_features"]
    if not isinstance(independent, bool) or independent != record.independent_of_ip_features:
        raise ShotLabelRecordError("independent_of_ip_features contradicts label authority")
    return record


def shot_label_record_from_json(encoded: str) -> ShotLabelRecord:
    """Validate a JSON-encoded shot-label record without accepting duplicate keys."""
    try:
        payload = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ShotLabelRecordError) as exc:
        raise ShotLabelRecordError(f"invalid shot-label JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ShotLabelRecordError("shot-label JSON root must be an object")
    return shot_label_record_from_dict(payload)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ShotLabelRecordError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ShotLabelRecordError(f"{field} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShotLabelRecordError(f"{field} must be an integer")
    return value


def _optional_float(payload: Mapping[str, object], field: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShotLabelRecordError(f"{field} must be numeric or null")
    return float(value)


def _string_tuple(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ShotLabelRecordError(f"{field} must be a string array")
    return tuple(cast(list[str], value))
