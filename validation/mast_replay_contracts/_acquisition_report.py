# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST acquisition source-object report assembly
"""Assemble declared FAIR-MAST acquisition metadata without source I/O."""

from __future__ import annotations

from typing import Any

from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_replay_contracts._acquisition_arrays import GROUP_VARIABLES
from validation.mast_replay_contracts._acquisition_source import BUCKET, CACHE_GENERATION_SCHEMA, ENDPOINT_URL
from validation.mast_source_object_manifest import SOURCE_GENERATION_DIGEST_KIND, SOURCE_OBJECT_MANIFEST_SCHEMA


def acquisition_report(
    records: list[dict[str, Any]], *, requested_count: int, generated_at: str, retrieved_at: str
) -> dict[str, Any]:
    """Assemble an unsealed v2 source-object declaration from per-shot records.

    Parameters
    ----------
    records : list of dict
        Acquired/failed shot records with declared arrays, caches and file bytes.
    requested_count : int
        Count of the acquisition caller's validated selection.
    generated_at, retrieved_at : str
        Reproducibility labels supplied by the caller.

    Returns
    -------
    dict
        Unsealed report with complete/partial/empty status and aggregate counts.
        Root/array/file authentication is not performed; the acquisition owner
        subsequently seals and validates its structure and local artifact bytes.

    Raises
    ------
    KeyError, TypeError, ValueError
        Per-shot record structure or aggregate byte-count conversion fails.

    Notes
    -----
    Fixed synthetic=False/source/licence/label-policy fields are declarations.
    Records and nested values remain shared with the caller. No source read,
    manifest write, label derivation or scientific admission occurs here.
    """
    acquired = [r for r in records if r["status"] == "acquired"]
    failed = [r for r in records if r["status"] == "failed"]
    status = "empty" if not acquired else ("partial" if failed else "complete")
    manifest: dict[str, Any] = {
        "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
        "manifest_kind": "source_object_inventory",
        "machine": "MAST",
        "campaign": "FAIR-MAST level2 disruption material",
        "status": status,
        "synthetic": False,
        "consumers": ["SCPN-CONTROL", "SCPN-FUSION-CORE", "MIF-CORE"],
        "source": {
            "bucket": f"s3://{BUCKET}",
            "endpoint": ENDPOINT_URL,
            "access": "anonymous",
            "format": "zarr_v3_level2",
            "path_template": f"s3://{BUCKET}/level2/shots/{{shot_id}}.zarr",
        },
        "licence_spdx": FAIR_MAST_LICENCE,
        **fair_mast_provenance(),
        "fidelity": {
            "sample_values": "selected source-resolution values; no resampling",
            "native_source": "remote FAIR-MAST Zarr v3",
            "local_cache": "derived NPZ; not a native/raw object",
            "source_hierarchy": "preserved in manifest, flattened in NPZ archive keys",
            "source_metadata": "preserved in manifest when exposed by xarray",
            "source_chunking": "recorded when exposed; not preserved in NPZ",
            "source_generation": "exact root zarr.json bytes checked before and after acquisition",
        },
        "cache_policy": {
            "schema_version": CACHE_GENERATION_SCHEMA,
            "strategy": "unique empty namespace per shot and acquisition label",
            "persistent_cross_run_reuse": False,
            "generation_identity": SOURCE_GENERATION_DIGEST_KIND,
            "pre_and_post_generation_check": True,
        },
        "group_variables": {group: list(variables) for group, variables in GROUP_VARIABLES.items()},
        "label_policy": (
            "labels not assigned; DEFUSE HDF5 labels are HTTP 403 and its shot ids do "
            "not intersect the level2 range, so consumers derive the Ip current-quench label"
        ),
        "retrieved_at": retrieved_at,
        "n_acquired": len(acquired),
        "n_requested": requested_count,
        "total_bytes": sum(int(r["artifacts"][0]["bytes"]) for r in acquired),
        "shots": records,
        "generated_at": generated_at,
    }
    return manifest
