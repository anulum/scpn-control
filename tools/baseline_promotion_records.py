# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Immutable promotion source inspection
"""Read exact source bytes once and validate declared run/report bindings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

from scpn_control.benchmark_records import RUN_SCHEMA
from tools.baseline_promotion_payloads import REPORT_SCHEMA, PromotionInputError, canonical_digest, json_bytes


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse repeated JSON names at every depth without reflecting raw names."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise PromotionInputError("Promotion JSON object member names must be unique")
        result[name] = value
    return result


def _nonfinite(value: str) -> object:
    """Refuse the decoder's nonstandard NaN and infinity extensions."""
    raise PromotionInputError("Promotion JSON numbers must be finite")


def _finite_float(value: str) -> float:
    """Refuse decimal exponent overflow rather than retaining an infinite float."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PromotionInputError("Promotion JSON numbers must be finite")
    return parsed


def _object(raw: bytes) -> dict[str, object]:
    """Parse one exact captured input buffer as unambiguous finite JSON."""
    value: object = json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite, parse_float=_finite_float
    )
    if not isinstance(value, dict):
        raise PromotionInputError("Promotion input must contain a JSON object")
    return cast(dict[str, object], value)


def _repository_path(path: Path, label: str, repository_root: Path) -> Path:
    """Resolve an input path inside the selected repository boundary."""
    resolved = path if path.is_absolute() else repository_root / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise PromotionInputError(f"{label} must remain inside the repository")
    return resolved


def _artifact_binding(
    manifest_path: Path,
    role: str,
    repository_root: Path,
) -> tuple[dict[str, object], Path, bytes]:
    """Capture and bind a successful run envelope and one immutable file once."""
    if "runs" not in manifest_path.parts or manifest_path.name != "manifest.json":
        raise PromotionInputError("source manifest is not inside an immutable runs directory")
    manifest = _object(manifest_path.read_bytes())
    if manifest.get("schema_version") != RUN_SCHEMA or manifest.get("status") != "succeeded":
        raise PromotionInputError("source manifest is not a successful benchmark run")
    unsigned = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if manifest.get("payload_sha256") != hashlib.sha256(json_bytes(unsigned)).hexdigest():
        raise PromotionInputError("source manifest payload digest is invalid")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise PromotionInputError("source manifest artifacts must be an object list")
    matches = [cast(dict[str, object], entry) for entry in entries if entry.get("role") == role]
    if len(matches) != 1:
        raise PromotionInputError("source manifest must contain exactly one selected artifact")
    entry = matches[0]
    if entry.get("kind") != "file":
        raise PromotionInputError("baseline source artifact must be a file")
    raw_path = entry.get("immutable_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PromotionInputError("source artifact requires a non-empty immutable path")
    artifact_path = Path(raw_path)
    artifact_path = artifact_path if artifact_path.is_absolute() else repository_root / artifact_path
    artifact_path = artifact_path.resolve()
    if (
        not artifact_path.is_relative_to((manifest_path.parent / "artifacts").resolve())
        or not artifact_path.is_relative_to(manifest_path.parent)
        or not artifact_path.is_relative_to(repository_root.resolve())
    ):
        raise PromotionInputError("source artifact escapes its immutable run directory")
    raw = artifact_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
        raise PromotionInputError("source artifact digest does not match the run manifest")
    return manifest, artifact_path, raw


def _report_from_bytes(raw: bytes) -> dict[str, object]:
    """Validate an existing report envelope from the same digest-checked buffer."""
    report = _object(raw)
    if report.get("schema_version") != REPORT_SCHEMA:
        raise PromotionInputError("source report schema must be scpn-control.benchmark-regression.v1")
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise PromotionInputError("source report has no benchmark metrics")
    unsigned = {key: value for key, value in report.items() if key != "payload_sha256"}
    if report.get("payload_sha256") != canonical_digest(unsigned):
        raise PromotionInputError("source report payload digest is invalid")
    return report


def load_promotion_source(
    manifest_path: Path,
    artifact_role: str,
    expected_source_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, object], Path, str, dict[str, object]]:
    """Return digest-bound declarations captured from one manifest/artifact read.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Repository-local immutable successful run manifest.
    artifact_role : str
        Selected role requiring exactly one immutable file declaration.
    expected_source_sha256 : str
        Caller-selected exact artifact-byte digest.
    repository_root : pathlib.Path
        Boundary for repository-relative artifact paths.

    Returns
    -------
    tuple of (dict, pathlib.Path, str, dict)
        Manifest, immutable artifact path, byte digest and checked report.

    Raises
    ------
    PromotionInputError
        Ambiguous/malformed JSON, invalid envelopes, aliases or digest drift.
    OSError, UnicodeError, json.JSONDecodeError
        File bytes cannot be read or decoded; CLI callers use fixed text.

    Notes
    -----
    Byte binding and declared status do not authenticate a producer, authority,
    Git object or hardware. Source namespaces require cooperating callers.
    """
    manifest_path = _repository_path(manifest_path, "source manifest", repository_root)
    manifest, artifact, raw = _artifact_binding(manifest_path, artifact_role, repository_root)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_source_sha256:
        raise PromotionInputError("expected source digest does not match the immutable artifact")
    return manifest, artifact, digest, _report_from_bytes(raw)
