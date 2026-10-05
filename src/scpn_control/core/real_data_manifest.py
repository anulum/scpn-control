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

Records carry declarations, not facility authentication. JSON loading refuses
duplicate keys/nonfinite floats; mapping validation does not interpret unknown
metadata. Local resolution/hashing operates without numerical dependencies.
Run these examples from the repository root against the actual fixture corpus.

>>> reference = load_real_data_manifest("validation/reference_data/diiid/manifests/diiid_hmode_1p5MA.geqdsk.manifest.json", verify_artifact=True)
>>> (reference.kind, reference.source.kind)
('synthetic', 'synthetic')
>>> resolve_manifest_artifact("diiid_hmode_1p5MA.geqdsk", manifest_path="validation/reference_data/diiid/manifests/example.json").suffix
'.geqdsk'
"""

from __future__ import annotations

import json
import math
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
    """Frozen signal declaration; direct construction performs no validation.

    Parameters
    ----------
    name : str
        Dataset-unique signal identity.
    path : str
        Declared archive key or acquisition node, not opened here.
    units : str
        Declared physical/dimensionless units; real schemas refuse vague units.
    timebase : str
        Declared time key or node spelling, without array alignment validation.
    """

    name: str
    path: str
    units: str
    timebase: str


@dataclass(frozen=True)
class DataSourceManifest:
    """Frozen declared source provenance without network or access validation.

    Parameters
    ----------
    kind : str
        Source family, lowercased by schema validation.
    uri : str
        Remote provenance or local file spelling; not authenticated here.
    access : str
        Declared access policy, not facility authorization.
    """

    kind: str
    uri: str
    access: str


@dataclass(frozen=True)
class ArtifactManifest:
    """Frozen local artifact declaration, without construction-time byte checks.

    Parameters
    ----------
    uri : str
        Relative artifact spelling, resolved only by selected verification.
    checksum_sha256 : str
        Lowercase 64-hex digest enforced by schema parsing.
    """

    uri: str
    checksum_sha256: str


@dataclass(frozen=True)
class RealDataManifest:
    """Frozen schema record separating declared real and synthetic roles.

    Required fields are schema_version, dataset_id, machine, normalized shot,
    synthetic, source and signals. Optional retrieval/licence/digest/citation/
    source-policy strings carry provenance declarations. Synthetic generator
    and integer seed carry fixture reproducibility declarations; artifacts is
    an immutable tuple of local digest references. Optional strings default to
    None and citation/artifact tuples to empty. No numerical array or mutable
    nested mapping is retained by the schema loader.

    Direct construction bypasses parsing. Use validate_real_data_manifest or
    load_real_data_manifest for required strings, identity uniqueness, checksum
    shape, source-role and physical-unit checks. Frozen fields and digest
    equality neither authenticate acquisition nor establish physical truth.
    """

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
    """Load finite, unique-key UTF-8 JSON and optionally verify local bytes.

    Parameters
    ----------
    path : str or Path
        Manifest file; local artifact lookup starts in its evidence tree.
    verify_artifact : bool, default False
        Literal boolean selecting SHA-256 checks for local references.

    Returns
    -------
    RealDataManifest
        Validated declarations, without proof of acquisition or physical truth.

    Raises
    ------
    RealDataManifestError
        Invalid policy, JSON, metadata, path, read or selected checksum check.
        Duplicate keys and nonfinite floats are refused at every JSON depth.
    """
    if not isinstance(verify_artifact, bool):
        raise RealDataManifestError("verify_artifact must be a boolean")
    try:
        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_manifest_keys,
                parse_constant=_reject_nonfinite_json,
                parse_float=_finite_json_float,
            )
        if not isinstance(payload, dict):
            raise RealDataManifestError("manifest root must be a JSON object")
        manifest = validate_real_data_manifest(payload)
        if verify_artifact:
            verify_manifest_artifact(manifest, manifest_path=manifest_path)
        return manifest
    except RealDataManifestError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise RealDataManifestError(f"cannot load manifest: {exc}") from exc


def _reject_nonfinite_json(token: str) -> Any:
    """Refuse NaN and infinity constants anywhere in a manifest JSON document."""
    raise RealDataManifestError(f"nonfinite JSON value: {token}")


def _finite_json_float(token: str) -> float:
    """Parse a floating token while refusing finite-text overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise RealDataManifestError(f"nonfinite JSON value: {token}")
    return value


def _reject_duplicate_manifest_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate provenance keys."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RealDataManifestError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def verify_manifest_artifact(manifest: RealDataManifest, *, manifest_path: str | Path) -> Path | None:
    """Check local file bytes against the digests of a validated manifest.

    Parameters
    ----------
    manifest : RealDataManifest
        Validated declaration. Direct dataclass construction bypasses schema checks.
    manifest_path : str or Path
        File spelling defining the ordered contained evidence roots.

    Returns
    -------
    Path or None
        Resolved file for a single local source; None for a verified artifact list
        or a remote/unchecked source. None alone does not prove a checksum check.

    Raises
    ------
    RealDataManifestError
        Unsafe/missing local reference, missing required digest or byte mismatch.
    OSError, ValueError, RuntimeError
        Direct file/path failures. The loader translates these to manifest errors.

    Notes
    -----
    An artifact list takes precedence over the source URI/root checksum. No list
    means synthetic data without a digest and nonlocal real sources are unchecked.
    Hashing does not authenticate a facility, licence or measured signal contents.
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

    Parameters
    ----------
    payload : dict
        Schema 1.0 declaration; unknown metadata is not interpreted.

    Returns
    -------
    RealDataManifest
        Frozen record whose shot is a trimmed string or normalized integer.

    Raises
    ------
    RealDataManifestError
        Missing/ill-typed provenance, duplicate signal or artifact identities,
        malformed provided checksums, or incompatible real/synthetic role.

    Notes
    -----
    Source URIs, policies, licences and retrieval times are declarations. This
    function neither reads artifact bodies nor authenticates facility access.
    """
    if not isinstance(payload, dict):
        raise RealDataManifestError("manifest root must be a JSON object")
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
    if len({signal.name for signal in signals}) != len(signals):
        raise RealDataManifestError("duplicate signal name")

    shot = payload["shot"]
    if isinstance(shot, bool) or not isinstance(shot, (str, int)):
        raise RealDataManifestError("shot must be a string or integer")

    manifest = RealDataManifest(
        schema_version=schema_version,
        dataset_id=_require_non_empty_str(payload, "dataset_id"),
        machine=_require_non_empty_str(payload, "machine"),
        shot=str(shot).strip(),
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
    if manifest.checksum_sha256 is not None and not _SHA256_RE.fullmatch(manifest.checksum_sha256):
        raise RealDataManifestError("checksum_sha256 must be lowercase 64-hex")
    if synthetic:
        _validate_synthetic_manifest(manifest)
    else:
        _validate_real_manifest(manifest)
    return manifest


def _parse_source(payload: dict[str, Any]) -> DataSourceManifest:
    """Parse required kind/URI/access strings, lowercasing only the source kind."""
    _require_keys(payload, ("kind", "uri", "access"))
    return DataSourceManifest(
        kind=_require_non_empty_str(payload, "kind").lower(),
        uri=_require_non_empty_str(payload, "uri"),
        access=_require_non_empty_str(payload, "access"),
    )


def _parse_signal(payload: object, index: int) -> SignalManifest:
    """Parse a single signal object with required name/path/units/timebase strings."""
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
    """Parse optional local artifact declarations, validating digests and unique URI spellings."""
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
    if len({artifact.uri for artifact in artifacts}) != len(artifacts):
        raise RealDataManifestError("duplicate dataset-manifest artifact URI")
    return tuple(artifacts)


def _validate_real_manifest(manifest: RealDataManifest) -> None:
    """Require an approved source kind, retrieval/licence/checksum provenance and physical units."""
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
    """Require a synthetic/mock source, generator identity and integer seed."""
    if manifest.source.kind not in _SYNTHETIC_SOURCE_KINDS:
        raise RealDataManifestError("synthetic manifest source.kind must be synthetic or mock")
    if not manifest.synthetic_generator:
        raise RealDataManifestError("synthetic manifest requires synthetic_generator")
    if manifest.synthetic_seed is None:
        raise RealDataManifestError("synthetic manifest requires synthetic_seed")


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Refuse absent required keys, preserving their declared diagnostic order."""
    missing = [key for key in keys if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise RealDataManifestError(f"manifest missing required key(s): {joined}")


def _require_non_empty_str(payload: dict[str, Any], key: str) -> str:
    """Return a required trimmed string; direct indexing follows required-key validation."""
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RealDataManifestError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_non_empty_str(payload: dict[str, Any], key: str) -> str | None:
    """Treat absent/null optional metadata as unspecified; otherwise require a trimmed string."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RealDataManifestError(f"{key} must be a non-empty string when present")
    return value.strip()


def _optional_non_empty_str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    """Treat absent/null citations as empty; otherwise validate every array string."""
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
    """Accept absent/null or integer metadata, explicitly excluding boolean values."""
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RealDataManifestError(f"{key} must be an integer when present")
    return value


def resolve_manifest_artifact(uri: str, *, manifest_path: str | Path) -> Path:
    """Resolve a local URI through the same roots used for checksum verification.

    Parameters
    ----------
    uri : str
        Relative file spelling without URI scheme, drive or parent traversal.
    manifest_path : str or Path
        Manifest whose directory starts ordered lookup. A ``manifests`` parent
        adds its containing directory; the nearest repository marker adds its
        repository root. The process working directory is never a lookup root.

    Returns
    -------
    Path
        First existing regular file canonically contained in a candidate root.
        No checksum or scientific contents are checked by this function.

    Raises
    ------
    RealDataManifestError
        Unsafe, missing or unresolvable local reference. Escaping symlinks are
        skipped. Resolution is not an atomic filesystem snapshot.
    """
    try:
        return _resolve_local_artifact(uri, Path(manifest_path))
    except RealDataManifestError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise RealDataManifestError(f"cannot resolve artifact: {exc}") from exc


def _resolve_local_artifact(uri: str, manifest_path: Path) -> Path:
    """Apply relative-path policy and ordered contained evidence-root lookup."""
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
    """Return ordered unique manifest/evidence/nearest-repository roots, excluding cwd."""
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
    """Stream local file bytes in bounded chunks into a lowercase SHA-256 digest."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
