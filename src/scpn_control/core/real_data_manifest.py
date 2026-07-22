# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real Data Manifest Validation

"""Validation contracts for real-shot and synthetic-shot data manifests.

The manifest is intentionally strict: synthetic fixtures are useful for CI, but
they must never be counted as experimental validation evidence by accident.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

ManifestKind = Literal["real", "synthetic"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REAL_SOURCE_KINDS = {"mdsplus", "imas", "omas", "geqdsk", "omfit", "local_archive"}
_SYNTHETIC_SOURCE_KINDS = {"synthetic", "mock"}
_BAD_REAL_UNITS = {"", "arb", "a.u.", "arbitrary", "unknown", "none"}


class RealDataManifestError(ValueError):
    """Raised when a data manifest cannot support its claimed validation role."""


@dataclass(frozen=True)
class SignalManifest:
    """Single measured or generated signal entry in a manifest."""

    name: str
    path: str
    units: str
    timebase: str


@dataclass(frozen=True)
class DataSourceManifest:
    """Acquisition source and provenance for a manifest."""

    kind: str
    uri: str
    access: str


@dataclass(frozen=True)
class ArtifactManifest:
    """Local artefact covered by a manifest checksum."""

    uri: str
    checksum_sha256: str


@dataclass(frozen=True)
class RealDataManifest:
    """Validated manifest separating real-shot evidence from synthetic fixtures."""

    schema_version: str
    dataset_id: str
    machine: str
    shot: str
    synthetic: bool
    source: DataSourceManifest
    signals: tuple[SignalManifest, ...]
    retrieved_at: str | None = None
    checksum_sha256: str | None = None
    licence: str | None = None
    synthetic_generator: str | None = None
    synthetic_seed: int | None = None
    artifacts: tuple[ArtifactManifest, ...] = ()
    licence_url: str | None = None
    citation: str | None = None
    citations: tuple[str, ...] = ()
    source_policy_url: str | None = None

    @property
    def kind(self) -> ManifestKind:
        """Return the validation role claimed by this manifest."""
        return "synthetic" if self.synthetic else "real"


def load_real_data_manifest(path: str | Path, *, verify_artifact: bool = False) -> RealDataManifest:
    """Load and validate a JSON real-data manifest."""
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_manifest_keys)
    if not isinstance(payload, dict):
        raise RealDataManifestError("manifest root must be a JSON object")
    manifest = validate_real_data_manifest(payload)
    if verify_artifact:
        verify_manifest_artifact(manifest, manifest_path=manifest_path)
    return manifest


def _reject_duplicate_manifest_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate provenance keys."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RealDataManifestError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def verify_manifest_artifact(manifest: RealDataManifest, *, manifest_path: str | Path) -> Path | None:
    """Verify local artefact checksums for manifests that reference local files.

    Remote acquisition sources such as MDSplus URIs are provenance records rather
    than local files, so they return ``None`` here.
    """
    if manifest.artifacts:
        for artifact in manifest.artifacts:
            artifact_path = _resolve_local_artifact(artifact.uri, Path(manifest_path))
            digest = _sha256_file(artifact_path)
            if digest != artifact.checksum_sha256:
                raise RealDataManifestError(
                    f"artifact checksum mismatch for {artifact_path}: expected {artifact.checksum_sha256}, got {digest}"
                )
        return None
    if manifest.synthetic and manifest.checksum_sha256 is None:
        return None
    if not manifest.synthetic and manifest.source.kind not in {"geqdsk", "local_archive"}:
        return None
    if manifest.checksum_sha256 is None:
        raise RealDataManifestError("artifact verification requires checksum_sha256")
    artifact_path = _resolve_local_artifact(manifest.source.uri, Path(manifest_path))
    digest = _sha256_file(artifact_path)
    if digest != manifest.checksum_sha256:
        raise RealDataManifestError(
            f"artifact checksum mismatch for {artifact_path}: expected {manifest.checksum_sha256}, got {digest}"
        )
    return artifact_path


def validate_real_data_manifest(payload: dict[str, Any]) -> RealDataManifest:
    """Validate a real-shot or synthetic-shot manifest.

    Real manifests require stable provenance, licence, units, checksum, and a
    non-synthetic acquisition source. Synthetic manifests require generator
    metadata so CI fixtures cannot masquerade as experimental evidence.
    """
    _require_keys(payload, ("schema_version", "dataset_id", "machine", "shot", "synthetic", "source", "signals"))
    schema_version = _require_non_empty_str(payload, "schema_version")
    if schema_version != "1.0":
        raise RealDataManifestError(f"unsupported manifest schema_version: {schema_version!r}")

    synthetic = payload["synthetic"]
    if not isinstance(synthetic, bool):
        raise RealDataManifestError("synthetic must be a boolean")

    source_payload = payload["source"]
    if not isinstance(source_payload, dict):
        raise RealDataManifestError("source must be an object")
    source = _parse_source(source_payload)

    signals_payload = payload["signals"]
    if not isinstance(signals_payload, list) or not signals_payload:
        raise RealDataManifestError("signals must be a non-empty array")
    signals = tuple(_parse_signal(signal, index) for index, signal in enumerate(signals_payload))

    manifest = RealDataManifest(
        schema_version=schema_version,
        dataset_id=_require_non_empty_str(payload, "dataset_id"),
        machine=_require_non_empty_str(payload, "machine"),
        shot=str(payload["shot"]).strip(),
        synthetic=synthetic,
        source=source,
        signals=signals,
        retrieved_at=_optional_non_empty_str(payload, "retrieved_at"),
        checksum_sha256=_optional_non_empty_str(payload, "checksum_sha256"),
        licence=_optional_non_empty_str(payload, "licence"),
        licence_url=_optional_non_empty_str(payload, "licence_url"),
        citation=_optional_non_empty_str(payload, "citation"),
        citations=_optional_non_empty_str_tuple(payload, "citations"),
        source_policy_url=_optional_non_empty_str(payload, "source_policy_url"),
        synthetic_generator=_optional_non_empty_str(payload, "synthetic_generator"),
        synthetic_seed=_optional_int(payload, "synthetic_seed"),
        artifacts=_parse_artifacts(payload),
    )

    if not manifest.shot:
        raise RealDataManifestError("shot must not be empty")
    if synthetic:
        _validate_synthetic_manifest(manifest)
    else:
        _validate_real_manifest(manifest)
    return manifest


def _parse_source(payload: dict[str, Any]) -> DataSourceManifest:
    _require_keys(payload, ("kind", "uri", "access"))
    return DataSourceManifest(
        kind=_require_non_empty_str(payload, "kind").lower(),
        uri=_require_non_empty_str(payload, "uri"),
        access=_require_non_empty_str(payload, "access"),
    )


def _parse_signal(payload: object, index: int) -> SignalManifest:
    if not isinstance(payload, dict):
        raise RealDataManifestError(f"signals[{index}] must be an object")
    _require_keys(payload, ("name", "path", "units", "timebase"))
    return SignalManifest(
        name=_require_non_empty_str(payload, "name"),
        path=_require_non_empty_str(payload, "path"),
        units=_require_non_empty_str(payload, "units"),
        timebase=_require_non_empty_str(payload, "timebase"),
    )


def _parse_artifacts(payload: dict[str, Any]) -> tuple[ArtifactManifest, ...]:
    artifacts_payload = payload.get("artifacts", [])
    if not isinstance(artifacts_payload, list):
        raise RealDataManifestError("artifacts must be an array when present")
    artifacts: list[ArtifactManifest] = []
    for index, artifact_payload in enumerate(artifacts_payload):
        if not isinstance(artifact_payload, dict):
            raise RealDataManifestError(f"artifacts[{index}] must be an object")
        _require_keys(artifact_payload, ("uri", "checksum_sha256"))
        checksum = _require_non_empty_str(artifact_payload, "checksum_sha256")
        if not _SHA256_RE.fullmatch(checksum):
            raise RealDataManifestError(f"artifacts[{index}].checksum_sha256 must be lowercase 64-hex")
        artifacts.append(
            ArtifactManifest(
                uri=_require_non_empty_str(artifact_payload, "uri"),
                checksum_sha256=checksum,
            )
        )
    return tuple(artifacts)


def _validate_real_manifest(manifest: RealDataManifest) -> None:
    if manifest.source.kind in _SYNTHETIC_SOURCE_KINDS:
        raise RealDataManifestError("real manifest cannot use a synthetic or mock source kind")
    if manifest.source.kind not in _REAL_SOURCE_KINDS:
        allowed = ", ".join(sorted(_REAL_SOURCE_KINDS))
        raise RealDataManifestError(f"real manifest source.kind must be one of: {allowed}")
    if not manifest.retrieved_at:
        raise RealDataManifestError("real manifest requires retrieved_at")
    if not manifest.licence:
        raise RealDataManifestError("real manifest requires licence")
    if not manifest.artifacts and (
        manifest.checksum_sha256 is None or not _SHA256_RE.fullmatch(manifest.checksum_sha256)
    ):
        raise RealDataManifestError("real manifest requires a lowercase 64-hex checksum_sha256")
    for signal in manifest.signals:
        if signal.units.strip().lower() in _BAD_REAL_UNITS:
            raise RealDataManifestError(f"real signal {signal.name!r} requires physical units")


def _validate_synthetic_manifest(manifest: RealDataManifest) -> None:
    if manifest.source.kind not in _SYNTHETIC_SOURCE_KINDS:
        raise RealDataManifestError("synthetic manifest source.kind must be synthetic or mock")
    if not manifest.synthetic_generator:
        raise RealDataManifestError("synthetic manifest requires synthetic_generator")
    if manifest.synthetic_seed is None:
        raise RealDataManifestError("synthetic manifest requires synthetic_seed")


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise RealDataManifestError(f"manifest missing required key(s): {joined}")


def _require_non_empty_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RealDataManifestError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_non_empty_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RealDataManifestError(f"{key} must be a non-empty string when present")
    return value.strip()


def _optional_non_empty_str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise RealDataManifestError(f"{key} must be a non-empty string array when present")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RealDataManifestError(f"{key}[{index}] must be a non-empty string")
        out.append(item.strip())
    return tuple(out)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RealDataManifestError(f"{key} must be an integer when present")
    return value


def _resolve_local_artifact(uri: str, manifest_path: Path) -> Path:
    if "://" in uri:
        raise RealDataManifestError(f"artifact verification requires a local URI, got {uri!r}")
    candidate = Path(uri)
    posix_candidate = PurePosixPath(uri)
    windows_candidate = PureWindowsPath(uri)
    if (
        candidate.is_absolute()
        or posix_candidate.is_absolute()
        or windows_candidate.is_absolute()
        or bool(posix_candidate.root)
        or bool(windows_candidate.root)
        or bool(windows_candidate.drive)
    ):
        raise RealDataManifestError("artifact URI must be relative to the manifest evidence tree")
    if any(
        part == ".."
        for path_parts in (candidate.parts, posix_candidate.parts, windows_candidate.parts)
        for part in path_parts
    ):
        raise RealDataManifestError("artifact URI must not contain parent traversal")

    roots = _artifact_resolution_roots(manifest_path)
    for root in roots:
        root_resolved = root.resolve(strict=False)
        resolved = (root_resolved / candidate).resolve(strict=False)
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise RealDataManifestError(f"artifact file not found: {uri}")


def _artifact_resolution_roots(manifest_path: Path) -> tuple[Path, ...]:
    manifest_parent = manifest_path.resolve(strict=False).parent
    roots: list[Path] = [manifest_parent]
    if manifest_parent.name == "manifests":
        roots.append(manifest_parent.parent)
    for parent in manifest_parent.parents:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            roots.append(parent)
            break
    return tuple(dict.fromkeys(roots))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
