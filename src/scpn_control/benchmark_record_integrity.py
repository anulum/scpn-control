# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark record encoding and latest integrity.
"""Encode digest-bound benchmark records and verify their complete stored artifacts.

These integrity checks detect inconsistent local bytes; they do not authenticate
an issuer, attest the producer's execution or grant scientific admission.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scpn_control.benchmark_artifacts import (
    DIRECTORY_DIGEST_ALGORITHM,
    LEGACY_DIRECTORY_DIGEST_ALGORITHM,
)
from scpn_control.benchmark_artifacts import (
    path_size as _path_size,
)
from scpn_control.benchmark_artifacts import (
    sha256_path as _sha256_path,
)

RUN_SCHEMA = "scpn-control.benchmark-run.v1"
LATEST_SCHEMA = "scpn-control.benchmark-latest.v1"
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")


def _sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 binding an exact byte carrier."""
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode the canonical sorted UTF-8 record spelling with its final newline."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_verified_latest(records_root: Path, family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a latest-completed index, canonical manifest payload and every artifact.

    Parameters
    ----------
    records_root : Path
        Native campaign-storage root containing latest/ and runs/ directories.
    family : str
        Descriptive family identifier. Latest follows successful atomic index
        publication order, not invocation start time.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Verified index and successful immutable manifest. Files require SHA-256;
        new directory entries name the portable tree-v2 algorithm. Legacy paths
        locate their basename within the immutable run, and legacy directory
        entries retain native-order file-only v1 digests. Older directory records
        do not bind empty-directory structure or promise cross-platform ordering.

    Raises
    ------
    ValueError
        Invalid schema/family/status/path/type/size/role or inconsistent index,
        payload or actual artifact digests. No records are modified.
    OSError, UnicodeError
        A required carrier/artifact is absent or unreadable.

    Notes
    -----
    Readers should consume the returned immutable artifact set rather than
    mutable producer destinations. Local digest integrity is not issuer
    authentication, execution attestation, dirty-source provenance or physics
    validation. Cooperating producers never modify a sealed run.
    """
    if _IDENTIFIER.fullmatch(family) is None:
        raise ValueError("invalid benchmark family identifier")
    run_family = family
    root = records_root.resolve()
    latest_path = root / "latest" / f"{run_family}.json"
    latest: dict[str, Any] = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(latest, dict):
        raise ValueError("benchmark latest index must be an object")
    if latest.get("schema_version") != LATEST_SCHEMA or latest.get("benchmark_family") != run_family:
        raise ValueError("benchmark latest index schema or family mismatch")
    manifest_path = root / Path(str(latest.get("manifest_path", "")))
    resolved_manifest = manifest_path.resolve()
    runs_root = (root / "runs").resolve()
    if not resolved_manifest.is_relative_to(runs_root):
        raise ValueError("benchmark latest manifest escapes the immutable runs root")
    manifest_bytes = resolved_manifest.read_bytes()
    if _sha256_bytes(manifest_bytes) != latest.get("manifest_sha256"):
        raise ValueError("benchmark latest manifest digest mismatch")
    manifest: dict[str, Any] = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("benchmark manifest must be an object")
    if (
        manifest.get("status") != "succeeded"
        or manifest.get("campaign_id") != latest.get("campaign_id")
        or type(manifest.get("exit_code")) is not int
        or manifest["exit_code"] != 0
        or manifest.get("missing_output_roles") != []
        or manifest.get("empty_output_roles", []) != []
    ):
        raise ValueError("benchmark latest references an inadmissible run")
    if manifest.get("schema_version") != RUN_SCHEMA or manifest.get("benchmark_family") != run_family:
        raise ValueError("benchmark manifest schema or family mismatch")
    unsigned = dict(manifest)
    payload_digest = unsigned.pop("payload_sha256", None)
    if _sha256_bytes(_json_bytes(unsigned)) != payload_digest:
        raise ValueError("benchmark manifest payload digest mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("benchmark manifest must contain artifacts")
    verified: dict[str, str] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("invalid benchmark artifact entry")
        role = entry.get("role")
        if not isinstance(role, str) or _IDENTIFIER.fullmatch(role) is None or role in verified:
            raise ValueError("invalid or duplicate benchmark artifact role")
        relative = entry.get("immutable_path_in_run")
        if "immutable_path_in_run" not in entry:
            observed = entry.get("immutable_path")
            if not isinstance(observed, str):
                raise ValueError("benchmark artifact path is missing")
            relative = "artifacts/" + PurePosixPath(observed.replace("\\", "/")).name
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("invalid benchmark artifact path")
        spelling = PurePosixPath(relative)
        if len(spelling.parts) != 2 or spelling.parts[0] != "artifacts" or spelling.name in (".", ".."):
            raise ValueError("benchmark artifact path must remain in the run artifacts directory")
        artifact = resolved_manifest.parent / Path(relative)
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(resolved_manifest.parent / "artifacts"):
            raise ValueError("benchmark artifact escapes its immutable directory")
        kind = entry.get("kind")
        if kind not in ("file", "directory") or (kind == "file") != artifact.is_file():
            raise ValueError("benchmark artifact kind mismatch")
        algorithm = entry.get(
            "digest_algorithm", LEGACY_DIRECTORY_DIGEST_ALGORITHM if kind == "directory" else "sha256"
        )
        if kind == "file" and algorithm != "sha256":
            raise ValueError("unsupported benchmark file digest algorithm")
        if not isinstance(algorithm, str):
            raise ValueError("invalid benchmark artifact digest algorithm")
        digest = _sha256_path(
            artifact, directory_algorithm=algorithm if kind == "directory" else DIRECTORY_DIGEST_ALGORITHM
        )
        size = entry.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("benchmark artifact size must be a positive integer")
        if digest != entry.get("sha256") or _path_size(artifact) != size:
            raise ValueError("benchmark immutable artifact digest or size mismatch")
        verified[role] = digest
    if verified != latest.get("artifact_sha256"):
        raise ValueError("benchmark latest artifact digests mismatch")
    return latest, manifest
