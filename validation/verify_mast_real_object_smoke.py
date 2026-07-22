#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Provenance-bound FAIR-MAST real-object smoke gate
"""Verify one exact external FAIR-MAST NPZ without publishing raw data.

The gate consumes a legacy material manifest and its externally stored NPZ
objects, migrates the manifest to SourceObjectManifest v2 in memory, and opens
the requested shot through the production verified-artifact reader. Callers
must pin both the legacy manifest and selected artifact digests, preventing a
different external object from silently satisfying the smoke proof.

This establishes transport interoperability only. It does not establish signal
bindings, label authority, cohort validity, predictive performance, facility
validation, or control admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from validation.extract_mast_disruption_channels import assess_artifact_binding_readiness
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE
from validation.mast_source_artifact_reader import (
    SourceArtifactReaderError,
    VerifiedSourceArtifact,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import SourceObjectManifestError, canonical_json_sha256
from validation.migrate_mast_source_object_manifest import migrate_material_manifest_v1

REAL_OBJECT_SMOKE_SCHEMA = "scpn-control.mast-real-object-smoke.v1"


class RealObjectSmokeError(ValueError):
    """Raised when the external object does not satisfy the pinned smoke contract."""


@dataclass(frozen=True)
class PinnedRealObject:
    """One digest-pinned external object opened through the verified reader.

    Parameters
    ----------
    artifact:
        Immutable source artefact exposed by the production v2 reader.
    legacy_manifest_sha256:
        Exact byte digest of the input legacy material manifest.
    migrated_manifest_sha256:
        Self-digest of the in-memory SourceObjectManifest v2 reconstruction.
    legacy_declared_licence:
        Licence string retained only as migration provenance.
    licence_spdx:
        Authoritative source policy applied by the verified migration.
    """

    artifact: VerifiedSourceArtifact
    legacy_manifest_sha256: str
    migrated_manifest_sha256: str
    legacy_declared_licence: str | None
    licence_spdx: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RealObjectSmokeError(f"duplicate JSON key {key!r} in legacy material manifest")
        payload[key] = value
    return payload


def _load_legacy_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        manifest_bytes = path.read_bytes()
    except OSError as exc:
        raise RealObjectSmokeError(f"cannot read legacy material manifest: {exc}") from exc
    try:
        text = manifest_bytes.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except RealObjectSmokeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RealObjectSmokeError(f"cannot decode legacy material manifest: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RealObjectSmokeError("legacy material manifest root must be an object")
    return payload, hashlib.sha256(manifest_bytes).hexdigest()


def read_pinned_real_object(
    manifest_path: Path,
    *,
    artifact_root: Path,
    shot_id: int,
    expected_manifest_sha256: str,
    expected_artifact_sha256: str,
) -> PinnedRealObject:
    """Open one exact legacy FAIR-MAST object through the verified v2 reader.

    Parameters
    ----------
    manifest_path:
        Legacy material manifest stored beside the external NPZ objects.
    artifact_root:
        External root beneath which all manifest paths must resolve.
    shot_id:
        Positive shot identifier selected for verification.
    expected_manifest_sha256:
        Exact SHA-256 digest of the legacy manifest bytes.
    expected_artifact_sha256:
        Exact SHA-256 digest of the selected NPZ bytes.

    Returns
    -------
    PinnedRealObject
        Immutable verified artefact plus the migration-policy digests needed by
        downstream evidence surfaces.

    Raises
    ------
    RealObjectSmokeError
        If identity, manifest, source declaration, migration, or artefact
        integrity verification fails.
    """
    if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
        raise RealObjectSmokeError("shot_id must be a positive integer")
    legacy, actual_manifest_sha256 = _load_legacy_manifest(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RealObjectSmokeError("legacy material manifest SHA-256 does not match the pinned digest")
    if legacy.get("synthetic") is not False:
        raise RealObjectSmokeError("real-object smoke requires an explicit synthetic=false declaration")

    try:
        migrated = migrate_material_manifest_v1(legacy, artifact_root=artifact_root)
        artifact = read_verified_npz_artifact(migrated, artifact_root=artifact_root, shot_id=shot_id)
    except (OSError, SourceObjectManifestError, SourceArtifactReaderError) as exc:
        raise RealObjectSmokeError(f"external source-object verification failed: {exc}") from exc
    if artifact.artifact_sha256 != expected_artifact_sha256:
        raise RealObjectSmokeError("selected NPZ SHA-256 does not match the pinned digest")

    legacy_licence = migrated["migration"]["legacy_declared_licence"]
    return PinnedRealObject(
        artifact=artifact,
        legacy_manifest_sha256=actual_manifest_sha256,
        migrated_manifest_sha256=migrated["payload_sha256"],
        legacy_declared_licence=legacy_licence if isinstance(legacy_licence, str) else None,
        licence_spdx=migrated["licence_spdx"],
    )


def verify_real_object_smoke(
    manifest_path: Path,
    *,
    artifact_root: Path,
    shot_id: int,
    expected_manifest_sha256: str,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    """Verify one digest-pinned external NPZ through the production v2 boundary.

    Parameters
    ----------
    manifest_path:
        Legacy material manifest stored beside the external NPZ objects.
    artifact_root:
        External root beneath which all manifest paths must resolve.
    shot_id:
        Positive shot identifier selected for the smoke proof.
    expected_manifest_sha256:
        Exact SHA-256 digest of the legacy manifest bytes.
    expected_artifact_sha256:
        Exact SHA-256 digest of the selected NPZ bytes.

    Returns
    -------
    dict[str, Any]
        Digest-bound transport proof with every scientific claim gate closed.

    Raises
    ------
    RealObjectSmokeError
        If an expected digest, source declaration, or selected shot is invalid.
    """
    pinned = read_pinned_real_object(
        manifest_path,
        artifact_root=artifact_root,
        shot_id=shot_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    artifact = pinned.artifact

    readiness = assess_artifact_binding_readiness(artifact)
    report: dict[str, Any] = {
        "schema_version": REAL_OBJECT_SMOKE_SCHEMA,
        "status": "verified_transport_only",
        "transport_interoperability_verified": True,
        "channel_extraction_admissible": False,
        "scientific_validity_claim_admissible": False,
        "facility_claim_admissible": False,
        "control_admission_admissible": False,
        "shot_id": artifact.shot_id,
        "legacy_manifest_sha256": pinned.legacy_manifest_sha256,
        "migrated_manifest_sha256": pinned.migrated_manifest_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "parent_digest": artifact.parent_digest,
        "transform_digest": artifact.transform_digest,
        "source_uri": artifact.source_uri,
        "legacy_declared_licence": pinned.legacy_declared_licence,
        "licence_spdx": pinned.licence_spdx,
        "licence_corrected_in_memory": pinned.licence_spdx == FAIR_MAST_LICENCE,
        "archive_key_count": len(artifact.archive_keys),
        "binding_readiness_sha256": readiness["payload_sha256"],
        "binding_blockers": readiness["blocking_contracts"],
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Legacy material manifest path.")
    parser.add_argument("--artifact-root", type=Path, required=True, help="External NPZ artifact root.")
    parser.add_argument("--shot-id", type=int, required=True, help="Shot selected for the smoke proof.")
    parser.add_argument("--expected-manifest-sha256", required=True, help="Pinned legacy manifest SHA-256.")
    parser.add_argument("--expected-artifact-sha256", required=True, help="Pinned selected NPZ SHA-256.")
    parser.add_argument("--json-out", type=Path, required=True, help="Destination for the smoke report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pinned external-object smoke and write its bounded report."""
    args = _parse_args(argv)
    report = verify_real_object_smoke(
        args.manifest,
        artifact_root=args.artifact_root,
        shot_id=args.shot_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_artifact_sha256=args.expected_artifact_sha256,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "MAST real-object smoke: "
        f"shot {report['shot_id']} transport verified; extraction/facility/control claims remain blocked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
