#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST material-manifest migration
"""Migrate verified legacy FAIR-MAST NPZ material manifests to v2."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import (
    LEGACY_MATERIAL_MANIFEST_SCHEMA,
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    SourceObjectManifestError,
    build_derived_npz_artifact,
    file_sha256,
    finalise_source_object_manifest,
    validate_source_object_manifest,
)


def migrate_material_manifest_v1(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Migrate legacy v1 metadata by verifying and inspecting its real NPZ files.

    The migration reconstructs array dtype, shape, value digest, group, and name.
    It cannot recover source dimensions, units, attributes, Zarr chunks, or native
    object bytes that v1 discarded; those gaps remain explicit in every array.
    A legacy licence is recorded as replaced, never propagated into v2 policy.
    """
    if payload.get("schema_version") != LEGACY_MATERIAL_MANIFEST_SCHEMA:
        raise SourceObjectManifestError(f"expected legacy schema {LEGACY_MATERIAL_MANIFEST_SCHEMA!r}")
    legacy_shots = payload.get("shots")
    if not isinstance(legacy_shots, list):
        raise SourceObjectManifestError("legacy shots must be a list")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise SourceObjectManifestError("legacy source must be an object")
    path_template = source.get("path_template")
    if not isinstance(path_template, str) or "{shot_id}" not in path_template:
        raise SourceObjectManifestError("legacy source.path_template must contain {shot_id}")
    migrated_shots: list[dict[str, Any]] = []
    for index, legacy_shot in enumerate(legacy_shots):
        migrated_shots.append(
            _migrate_shot(
                legacy_shot,
                index=index,
                artifact_root=artifact_root,
                path_template=path_template,
            )
        )
    acquired = [shot for shot in migrated_shots if shot["status"] == "acquired"]
    migrated: dict[str, Any] = {
        "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
        "manifest_kind": "source_object_inventory",
        "machine": "MAST",
        "campaign": "FAIR-MAST level2 disruption material",
        "status": "empty" if not acquired else ("partial" if len(acquired) != len(migrated_shots) else "complete"),
        "synthetic": payload.get("synthetic", False),
        "consumers": payload.get("consumers", []),
        "source": dict(source),
        "licence_spdx": FAIR_MAST_LICENCE,
        **fair_mast_provenance(),
        "fidelity": {
            "sample_values": "legacy v1 declared native resolution with no downsampling",
            "source_hierarchy": "reconstructed from NPZ archive keys",
            "source_metadata": "not recoverable from v1",
            "native_zarr_bytes": "not preserved by v1",
        },
        "migration": {
            "from_schema": LEGACY_MATERIAL_MANIFEST_SCHEMA,
            "method": "verified_npz_reconstruction",
            "limitations": [
                "source dimensions, units, attributes, and chunks were absent in v1",
                "native Zarr object bytes and byte digest were absent in v1",
                "parent digest binds selected values and explicit missing-metadata markers",
            ],
            "legacy_declared_licence": payload.get("licence"),
            "source_policy_action": "replaced with authoritative FAIR-MAST policy",
        },
        "retrieved_at": payload.get("retrieved_at"),
        "generated_at": payload.get("generated_at"),
        "n_acquired": len(acquired),
        "n_requested": len(migrated_shots),
        "total_bytes": sum(int(shot["artifacts"][0]["bytes"]) for shot in acquired),
        "shots": migrated_shots,
    }
    finalised = finalise_source_object_manifest(migrated)
    validate_source_object_manifest(finalised, artifact_root=artifact_root)
    return finalised


def _migrate_shot(
    legacy_shot: Any,
    *,
    index: int,
    artifact_root: Path,
    path_template: str,
) -> dict[str, Any]:
    if not isinstance(legacy_shot, Mapping):
        raise SourceObjectManifestError(f"legacy shots[{index}] must be an object")
    shot_id = legacy_shot.get("shot_id")
    if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
        raise SourceObjectManifestError(f"legacy shots[{index}].shot_id must be a positive integer")
    if legacy_shot.get("status") == "failed":
        return {
            "shot_id": shot_id,
            "status": "failed",
            "programme_class": "unknown",
            "error": str(legacy_shot.get("error", "legacy acquisition failed without an error message")),
        }
    local_path = legacy_shot.get("npz")
    if not isinstance(local_path, str):
        raise SourceObjectManifestError(f"legacy shots[{index}].npz must be a string")
    artifact_path = _resolve_legacy_artifact(artifact_root, local_path, index=index)
    if file_sha256(artifact_path) != legacy_shot.get("checksum_sha256"):
        raise SourceObjectManifestError(f"legacy shots[{index}] NPZ checksum mismatch")
    if artifact_path.stat().st_size != legacy_shot.get("bytes"):
        raise SourceObjectManifestError(f"legacy shots[{index}] NPZ byte count mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    artifact = build_derived_npz_artifact(
        local_path=local_path,
        artifact_path=artifact_path,
        source_uri=path_template.format(shot_id=shot_id),
        arrays=arrays,
    )
    summary = {
        key: legacy_shot[key]
        for key in ("ip_max_ka", "saddle_channels", "saddle_samples", "variables")
        if key in legacy_shot
    }
    return {
        "shot_id": shot_id,
        "status": "acquired",
        "programme_class": "unknown",
        "artifacts": [artifact],
        "summary": summary,
    }


def _resolve_legacy_artifact(root: Path, local_path: str, *, index: int) -> Path:
    candidate = Path(local_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SourceObjectManifestError(f"legacy shots[{index}].npz must stay beneath the artefact root")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceObjectManifestError(f"legacy shots[{index}].npz escapes the artefact root") from exc
    if not resolved.is_file():
        raise SourceObjectManifestError(f"legacy shots[{index}].npz does not resolve to a file")
    return resolved
