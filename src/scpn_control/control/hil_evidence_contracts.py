# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware-in-the-Loop Test Harness

"""HIL evidence schema and primitive validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np

HIL_REPLAY_EVIDENCE_SCHEMA_VERSION = "scpn-control.hil-replay-evidence.v2"
HIL_REPLAY_EVIDENCE_BOUNDARY = "local_hil_replay_or_qualified_target_hardware"
_LOCAL_CLAIM_STATUS = "bounded_local_hil_replay_only"
_TARGET_HARDWARE_CLAIM_STATUS = "qualified_target_hardware_deployment_evidence"
_PLACEHOLDER_HARDWARE_VALUES = {
    "",
    "ci",
    "dev",
    "generic",
    "generic-userspace",
    "local",
    "localhost",
    "local-unqualified-host",
    "local-userspace-replay",
    "n/a",
    "none",
    "test",
    "unknown",
}


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ValueError("HIL evidence contains a non-finite JSON number") from exc


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        out[key] = value
    return out


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _is_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_qualified_hardware(value: Any, name: str) -> str:
    text = _require_non_empty_text(value, name)
    normalised = text.casefold()
    if normalised in _PLACEHOLDER_HARDWARE_VALUES or normalised.startswith("local"):
        raise ValueError(f"{name} must identify qualified target hardware")
    if "unknown" in normalised or "placeholder" in normalised:
        raise ValueError(f"{name} must not be a placeholder")
    return text
