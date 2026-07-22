#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST saddle-probe modal source-authority gate
"""Admit MAST ``n1_amp`` and ``n2_amp`` only from joined probe authority.

The FAIR-MAST Level-2 snapshot contains a twelve-row saddle-field array and
three twelve-polygon toroidal geometry arrays. Matching row counts and regular
angles do not prove which geometry row belongs to which field channel, which
vertical saddle set the field represents, or how calibration uncertainty and
bad channels must be handled. This gate measures the source facts without
performing a modal reduction and fails closed until those joins are persisted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from validation.mast_source_artifact_reader import (
    VerifiedSourceArtifact,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256

SADDLE_MODAL_AUTHORITY_SCHEMA = "scpn-control.mast-saddle-modal-authority.v1.0.0"
SADDLE_MODAL_AUTHORITY_VERSION = "1.0.0"
FAIR_MAST_MAPPING_COMMIT = "862f08d7d91930b988d674e7ec67f3a03aacafac"
FAIR_MAST_MAPPING_SHA256 = "cb5420f8a9f78bbf417e3d2314e35bf58f402f221a6c9fb0eed5dfbfcefb9b76"
FAIR_MAST_MAPPING_URL = (
    f"https://github.com/ukaea/fair-mast-ingestion/blob/{FAIR_MAST_MAPPING_COMMIT}/mappings/level2/mast.yml#L3498-L3685"
)
FAIR_MAST_PAPER_URL = "https://doi.org/10.1016/j.softx.2024.101869"

FIELD_KEY = "magnetics.b_field_tor_probe_saddle_field"
TIMEBASE_KEY = "magnetics.time_saddle"
GEOMETRY_KEYS: tuple[str, ...] = (
    "magnetics.b_field_tor_probe_saddle_l_phi",
    "magnetics.b_field_tor_probe_saddle_m_phi",
    "magnetics.b_field_tor_probe_saddle_u_phi",
)
EXPECTED_FIELD_CHANNELS: tuple[str, ...] = tuple(f"ASM_SAD/M{index:02d}" for index in range(1, 13))
EXPECTED_CENTRES_DEG: tuple[float, ...] = tuple(15.0 + 30.0 * index for index in range(12))


class SaddleModalAuthorityError(ValueError):
    """Raised when a saddle-modal authority input or report is inconsistent."""


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_trusted_authority_url(value: object) -> bool:
    """Restrict declarations to primary FAIR-MAST and UKAEA surfaces."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.username is not None:
        return False
    if parsed.hostname in {"mastapp.site", "git.ccfe.ac.uk", "doi.org", "ukaea.uk", "www.ukaea.uk"}:
        return True
    return parsed.hostname == "github.com" and parsed.path.startswith("/ukaea/")


def mast_saddle_modal_authority_spec() -> dict[str, object]:
    """Return the deterministic, self-digested L2F-12c authority contract."""
    payload: dict[str, object] = {
        "schema_version": SADDLE_MODAL_AUTHORITY_SCHEMA,
        "version": SADDLE_MODAL_AUTHORITY_VERSION,
        "machine": "MAST",
        "canonical_channels": ["n1_amp", "n2_amp"],
        "canonical_units": "T",
        "source_field_key": FIELD_KEY,
        "source_timebase_key": TIMEBASE_KEY,
        "source_geometry_keys": list(GEOMETRY_KEYS),
        "expected_field_channels": list(EXPECTED_FIELD_CHANNELS),
        "expected_toroidal_centres_deg": list(EXPECTED_CENTRES_DEG),
        "candidate_transform": {
            "formula": "A_n(t) = (2/N) * abs(sum_k B_k(t) * exp(-i*n*phi_k)))",
            "orders": [1, 2],
            "angle_conversion": "phi_rad = phi_deg * pi / 180",
            "status": "inadmissible_until_source_rows_and_geometry_rows_are_authoritatively_joined",
        },
        "required_source_authority": [
            "released geometry revision tied to exact geometry-value digest",
            "ordered field-row identities for ASM_SAD/M01 through ASM_SAD/M12",
            "field-to-geometry row join and selected lower/middle/upper saddle set",
            "field calibration and one-standard-deviation uncertainty",
            "baseline, saturation, missing, and bad-channel policy",
            "content digests for the row join, selected geometry, and calibration authority",
            "allowlisted primary-source authority citation",
        ],
        "mapping_commit": FAIR_MAST_MAPPING_COMMIT,
        "mapping_file_sha256": FAIR_MAST_MAPPING_SHA256,
        "citations": [FAIR_MAST_MAPPING_URL, FAIR_MAST_PAPER_URL],
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _metadata_evidence(
    artifact: VerifiedSourceArtifact,
) -> tuple[list[str], dict[str, object]]:
    """Validate exact source members and the authority-bearing metadata fields."""
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    required = (FIELD_KEY, TIMEBASE_KEY, *GEOMETRY_KEYS)
    missing = [key for key in required if key not in artifact.arrays or key not in artifact.metadata]
    if missing:
        blockers.append("saddle_modal_source_members_absent")
        evidence["missing_source_keys"] = missing
        return blockers, evidence

    expected_dimensions = {
        FIELD_KEY: ("b_field_tor_probe_saddle_field_channel", "time_saddle"),
        TIMEBASE_KEY: ("time_saddle",),
        GEOMETRY_KEYS[0]: ("b_field_tor_probe_saddle_l_geometry_channel", "coordinate"),
        GEOMETRY_KEYS[1]: ("b_field_tor_probe_saddle_m_geometry_channel", "coordinate"),
        GEOMETRY_KEYS[2]: ("b_field_tor_probe_saddle_u_geometry_channel", "coordinate"),
    }
    expected_units = {FIELD_KEY: "T", TIMEBASE_KEY: "s", **dict.fromkeys(GEOMETRY_KEYS, "degrees")}
    mismatches: list[str] = []
    for key in required:
        metadata = artifact.metadata[key]
        if metadata.get("metadata_status") != "source_xarray":
            mismatches.append(f"{key}:metadata_status")
        if tuple(metadata.get("dimensions") or ()) != expected_dimensions[key]:
            mismatches.append(f"{key}:dimensions")
        if metadata.get("units") != expected_units[key]:
            mismatches.append(f"{key}:units")
    if mismatches:
        blockers.append("saddle_modal_source_metadata_mismatch")
        evidence["metadata_mismatches"] = mismatches

    field_attributes = artifact.metadata[FIELD_KEY].get("source_attributes")
    if not isinstance(field_attributes, Mapping):
        blockers.append("saddle_field_source_attributes_absent")
        field_attributes = {}
    evidence["field_metadata"] = {
        "name": field_attributes.get("name"),
        "uda_name": field_attributes.get("uda_name"),
        "label": field_attributes.get("label"),
        "units": field_attributes.get("units"),
        "description": field_attributes.get("description"),
    }
    if tuple(field_attributes.get("source_channels") or ()) != EXPECTED_FIELD_CHANNELS:
        blockers.append("saddle_field_row_identities_not_preserved")
    vertical_set = field_attributes.get("geometry_vertical_set")
    if vertical_set not in {"lower", "middle", "upper"}:
        blockers.append("saddle_field_geometry_vertical_set_not_attested")
    else:
        selected_key = dict(zip(("lower", "middle", "upper"), GEOMETRY_KEYS, strict=True))[vertical_set]
        selected_digest = artifact.metadata[selected_key].get("value_sha256")
        if not _is_sha256(selected_digest) or field_attributes.get("geometry_value_sha256") != selected_digest:
            blockers.append("saddle_field_geometry_value_identity_not_attested")
    if not _is_sha256(field_attributes.get("row_join_evidence_sha256")):
        blockers.append("saddle_field_row_join_evidence_absent")
    if not _is_sha256(field_attributes.get("calibration_evidence_sha256")):
        blockers.append("saddle_field_calibration_evidence_absent")
    if not _is_trusted_authority_url(field_attributes.get("authority_citation")):
        blockers.append("saddle_field_authority_citation_absent")
    uncertainty = field_attributes.get("standard_uncertainty_t")
    if not isinstance(uncertainty, (int, float)) or isinstance(uncertainty, bool) or not math.isfinite(uncertainty):
        blockers.append("saddle_field_standard_uncertainty_absent")
    elif uncertainty <= 0.0:
        blockers.append("saddle_field_standard_uncertainty_not_positive")
    for policy_key, blocker in (
        ("baseline_policy", "saddle_field_baseline_policy_absent"),
        ("bad_channel_policy", "saddle_field_bad_channel_policy_absent"),
    ):
        policy = field_attributes.get(policy_key)
        if not isinstance(policy, str) or not policy.strip():
            blockers.append(blocker)

    geometry_metadata: list[dict[str, object]] = []
    geometry_value_digests: list[str | None] = []
    for key in GEOMETRY_KEYS:
        metadata = artifact.metadata[key]
        attributes = metadata.get("source_attributes")
        if not isinstance(attributes, Mapping):
            blockers.append("saddle_geometry_source_attributes_absent")
            attributes = {}
        geometry_metadata.append(
            {
                "source_key": key,
                "status": attributes.get("status"),
                "revision": attributes.get("revision"),
                "creator_commit_id": attributes.get("creatorCommitId"),
                "signed_off_by": attributes.get("signedOffBy"),
                "source": attributes.get("source"),
                "observed_units": metadata.get("units"),
                "value_sha256": metadata.get("value_sha256"),
            }
        )
        geometry_value_digests.append(
            metadata.get("value_sha256") if isinstance(metadata.get("value_sha256"), str) else None
        )
        if attributes.get("status") not in {"released", "approved"}:
            blockers.append("saddle_geometry_not_released")
        creator_commit = attributes.get("creatorCommitId")
        if (
            not isinstance(creator_commit, str)
            or len(creator_commit) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in creator_commit)
        ):
            blockers.append("saddle_geometry_creator_commit_absent")
        if attributes.get("signedOffBy") in {None, ""}:
            blockers.append("saddle_geometry_signoff_absent")
    if any(not _is_sha256(digest) for digest in geometry_value_digests):
        blockers.append("saddle_geometry_value_digest_absent")
    evidence["geometry_vertical_sets_share_value_digest"] = len(set(geometry_value_digests)) == 1
    evidence["geometry_metadata"] = geometry_metadata
    return blockers, evidence


def _circular_centres_deg(polygons_deg: np.ndarray[Any, np.dtype[np.float64]]) -> tuple[float, ...] | None:
    """Return circular-mean polygon centres, or ``None`` for undefined rows."""
    angles = np.deg2rad(polygons_deg)
    phasors = np.mean(np.exp(1j * angles), axis=1)
    if bool(np.any(~np.isfinite(phasors))) or bool(np.any(np.abs(phasors) <= 1.0e-12)):
        return None
    return tuple(float(value) for value in np.mod(np.rad2deg(np.angle(phasors)), 360.0))


def _array_evidence(
    artifact: VerifiedSourceArtifact,
) -> tuple[list[str], dict[str, object]]:
    """Measure source shapes, finite coverage, time order, and polygon centres."""
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    required = (FIELD_KEY, TIMEBASE_KEY, *GEOMETRY_KEYS)
    if any(key not in artifact.arrays for key in required):
        return ["saddle_modal_source_members_absent"], evidence

    field = np.asarray(artifact.arrays[FIELD_KEY], dtype=np.float64)
    time = np.asarray(artifact.arrays[TIMEBASE_KEY], dtype=np.float64)
    geometry = [np.asarray(artifact.arrays[key], dtype=np.float64) for key in GEOMETRY_KEYS]
    evidence["source_shapes"] = {
        FIELD_KEY: list(field.shape),
        TIMEBASE_KEY: list(time.shape),
        **{key: list(values.shape) for key, values in zip(GEOMETRY_KEYS, geometry, strict=True)},
    }
    if field.ndim != 2 or field.shape[0] != 12 or time.ndim != 1 or field.shape[1] != time.size:
        blockers.append("saddle_field_or_timebase_shape_mismatch")
    if any(values.ndim != 2 or values.shape[0] != 12 or values.shape[1] < 3 for values in geometry):
        blockers.append("saddle_geometry_shape_mismatch")

    if time.ndim == 1:
        finite_time = time[np.isfinite(time)]
        evidence["finite_time_samples"] = int(finite_time.size)
        if finite_time.size == 0:
            blockers.append("saddle_timebase_has_no_finite_samples")
        elif finite_time.size != time.size or (finite_time.size > 1 and not bool(np.all(np.diff(finite_time) > 0.0))):
            blockers.append("saddle_timebase_not_finite_and_strictly_increasing")
    if field.ndim == 2 and field.shape[0] == 12:
        finite_counts = [int(np.count_nonzero(np.isfinite(field[index]))) for index in range(12)]
        evidence["field_finite_samples_per_row"] = finite_counts
        evidence["field_rows_without_finite_samples_one_based"] = [
            index + 1 for index, count in enumerate(finite_counts) if count == 0
        ]
        if any(count == 0 for count in finite_counts):
            blockers.append("saddle_field_row_has_no_finite_samples")

    if not any(blocker == "saddle_geometry_shape_mismatch" for blocker in blockers):
        centres = [_circular_centres_deg(values) for values in geometry]
        if any(value is None for value in centres):
            blockers.append("saddle_geometry_circular_centre_undefined")
        else:
            resolved = [value for value in centres if value is not None]
            evidence["geometry_centres_deg"] = {
                key: list(values) for key, values in zip(GEOMETRY_KEYS, resolved, strict=True)
            }
            if any(not np.allclose(values, EXPECTED_CENTRES_DEG, rtol=0.0, atol=1.0e-9) for values in resolved):
                blockers.append("saddle_geometry_centres_do_not_match_pinned_mapping_order")
            evidence["geometry_vertical_sets_have_identical_values"] = all(
                np.array_equal(geometry[0], values, equal_nan=True) for values in geometry[1:]
            )
    return blockers, evidence


def assess_saddle_modal_authority(artifact: VerifiedSourceArtifact) -> dict[str, object]:
    """Assess one verified source artefact without deriving modal amplitudes."""
    metadata_blockers, metadata_evidence = _metadata_evidence(artifact)
    array_blockers, array_evidence = _array_evidence(artifact)
    blockers = list(dict.fromkeys((*metadata_blockers, *array_blockers)))
    report: dict[str, object] = {
        "schema_version": SADDLE_MODAL_AUTHORITY_SCHEMA,
        "spec_sha256": mast_saddle_modal_authority_spec()["payload_sha256"],
        "shot_id": artifact.shot_id,
        "source_uri": artifact.source_uri,
        "artifact_sha256": artifact.artifact_sha256,
        "metadata_evidence": metadata_evidence,
        "array_evidence": array_evidence,
        "modal_reduction_executed": False,
        "canonical_modal_bindings_admissible": not blockers,
        "status": "authority_verified" if not blockers else "blocked",
        "blockers": blockers,
        "claim_boundary": {
            "scientific_validation": False,
            "training_admission": False,
            "facility_prediction": False,
            "control_admission": False,
        },
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def build_saddle_modal_authority_report(
    artifacts: Sequence[VerifiedSourceArtifact],
) -> dict[str, object]:
    """Build one deterministic multi-shot saddle-modal authority assessment."""
    if not artifacts:
        raise SaddleModalAuthorityError("at least one verified source artefact is required")
    shot_ids = [artifact.shot_id for artifact in artifacts]
    if shot_ids != sorted(set(shot_ids)):
        raise SaddleModalAuthorityError("source artefacts must have unique, ascending shot identifiers")
    assessments = [assess_saddle_modal_authority(artifact) for artifact in artifacts]
    admissible = all(bool(item["canonical_modal_bindings_admissible"]) for item in assessments)
    report: dict[str, object] = {
        "schema_version": SADDLE_MODAL_AUTHORITY_SCHEMA,
        "spec": mast_saddle_modal_authority_spec(),
        "status": "authority_verified" if admissible else "blocked",
        "canonical_modal_bindings_admissible": admissible,
        "shot_count": len(assessments),
        "shots": assessments,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Write a real-object L2F-12c saddle-modal authority assessment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--shot-id", type=int, action="append", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_verified_source_manifest(args.manifest, artifact_root=args.artifact_root)
    shot_ids = sorted(set(args.shot_id))
    if len(shot_ids) != len(args.shot_id):
        parser.error("--shot-id values must be unique")
    artifacts = [
        read_verified_npz_artifact(manifest, artifact_root=args.artifact_root, shot_id=shot_id) for shot_id in shot_ids
    ]
    report = build_saddle_modal_authority_report(artifacts)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.json_out.open("x", encoding="utf-8") as report_handle:
            json.dump(report, report_handle, indent=2, sort_keys=True)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
    except FileExistsError as exc:
        raise SaddleModalAuthorityError("refusing to overwrite an existing authority report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"{report['status']}: {report['shot_count']} shot(s) -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
