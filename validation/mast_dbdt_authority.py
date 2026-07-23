#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST dB/dt source-authority gate
"""Admit ``dBdt_gauss_per_s`` only from an attested Mirnov source contract.

The FAIR-MAST mapping stores the centre-column poloidal Mirnov array as tesla
with a 2e-6 scale, while its live label says tesla per second and the MAST
diagnostics paper describes the fast Bv signal as non-integrated. This gate
measures that conflict without differentiating any trace and fails closed until
the source lineage declares whether the array is B or dB/dt and binds the
corresponding transform, probe join, calibration, filtering, and uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeGuard, cast

import numpy as np

from validation.mast_saddle_modal_authority import FAIR_MAST_MAPPING_COMMIT, FAIR_MAST_MAPPING_SHA256
from validation.mast_source_artifact_reader import (
    VerifiedSourceArtifact,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256

DBDT_AUTHORITY_SCHEMA = "scpn-control.mast-dbdt-authority.v1.0.0"
DBDT_AUTHORITY_VERSION = "1.0.0"
FIELD_KEY = "magnetics.b_field_pol_probe_cc_field"
TIMEBASE_KEY = "magnetics.time_mirnov"
GEOMETRY_KEYS: tuple[str, ...] = (
    "magnetics.b_field_pol_probe_cc_phi",
    "magnetics.b_field_pol_probe_cc_r",
    "magnetics.b_field_pol_probe_cc_z",
)
EXPECTED_SOURCE_CHANNELS: tuple[str, ...] = tuple(f"xmc/CC/MV/{number}" for number in (201, 210, 220, 230, 240))
EXPECTED_GEOMETRY_CHANNELS: tuple[str, ...] = tuple(f"cc_mv_{number}" for number in range(201, 241))
EXPECTED_SOURCE_TO_GEOMETRY_INDICES: tuple[int, ...] = (0, 9, 19, 29, 39)
FAIR_MAST_DBDT_MAPPING_URL = (
    f"https://github.com/ukaea/fair-mast-ingestion/blob/{FAIR_MAST_MAPPING_COMMIT}/mappings/level2/mast.yml#L3105-L3123"
)
FAIR_MAST_DBDT_PROFILE_URL = (
    f"https://github.com/ukaea/fair-mast-ingestion/blob/{FAIR_MAST_MAPPING_COMMIT}/mappings/level2/mast.yml#L3934-L4033"
)
MAST_MAGNETICS_PAPER_URL = "https://doi.org/10.1063/1.1309009"
MAST_MAGNETICS_UKAEA_URL = "https://scientific-publications.ukaea.uk/wp-content/uploads/Published/RSIVOL72p421.pdf"
MAPPING_SCALE = 2.0e-6
TESLA_PER_SECOND_TO_GAUSS_PER_SECOND = 1.0e4


class DbdtAuthorityError(ValueError):
    """Raised when a dB/dt authority input or report is inconsistent."""


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_positive_finite_number(value: object) -> TypeGuard[int | float]:
    """Return whether ``value`` is a positive finite non-boolean number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def _nonempty_text(value: object) -> bool:
    """Return whether ``value`` is a non-empty text declaration."""
    return isinstance(value, str) and bool(value.strip())


def mast_dbdt_authority_spec() -> dict[str, object]:
    """Return the deterministic, self-digested L2F-12d dB/dt contract."""
    payload: dict[str, object] = {
        "schema_version": DBDT_AUTHORITY_SCHEMA,
        "version": DBDT_AUTHORITY_VERSION,
        "machine": "MAST",
        "canonical_channel": "dBdt_gauss_per_s",
        "canonical_units": "G/s",
        "source_field_key": FIELD_KEY,
        "source_timebase_key": TIMEBASE_KEY,
        "source_geometry_keys": list(GEOMETRY_KEYS),
        "expected_source_channels": list(EXPECTED_SOURCE_CHANNELS),
        "expected_geometry_channels": list(EXPECTED_GEOMETRY_CHANNELS),
        "expected_source_to_geometry_indices": list(EXPECTED_SOURCE_TO_GEOMETRY_INDICES),
        "mapping_commit": FAIR_MAST_MAPPING_COMMIT,
        "mapping_file_sha256": FAIR_MAST_MAPPING_SHA256,
        "mapping_scale": MAPPING_SCALE,
        "tesla_per_second_to_gauss_per_second": TESLA_PER_SECOND_TO_GAUSS_PER_SECOND,
        "source_conflict": {
            "mapping_units": "T",
            "live_label": "Tesla/sec",
            "paper_observation": "non-integrated signal from a mid-plane Bv coil",
        },
        "admissible_transform_branches": {
            "magnetic_field": "differentiate_once_then_multiply_by_1e4_G_per_T",
            "magnetic_field_time_derivative": "multiply_by_1e4_G_per_T_without_differentiation",
        },
        "legacy_candidate": {
            "row": 0,
            "missing_value_policy": "replace_nonfinite_with_zero",
            "transform": "np.gradient(candidate, time_mirnov) * 1e4",
            "grid_reduction": "per-bin peak magnitude",
            "status": "compatibility_only_not_source_authorised",
        },
        "required_source_authority": [
            "source quantity and units resolving the T versus Tesla/sec conflict",
            "meaning and dimensional effect of the mapping scale 2e-6",
            "ordered five-row source identities and exact five-to-forty geometry join",
            "measured component, probe orientation, and sign convention",
            "probe reduction, missing-value, and bad-channel policies",
            "single-derivative or identity transform selected from the source quantity",
            "physical-frequency filter, cutoff below Nyquist, and edge policy",
            "calibration evidence and positive one-standard-deviation uncertainty",
            "content-digested authority contract and primary-source citations",
        ],
        "citations": [
            FAIR_MAST_DBDT_MAPPING_URL,
            FAIR_MAST_DBDT_PROFILE_URL,
            MAST_MAGNETICS_PAPER_URL,
            MAST_MAGNETICS_UKAEA_URL,
        ],
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_authority_contract(
    contract_value: object,
    *,
    sample_rate_hz: float | None,
    observed_source_units: object,
) -> tuple[list[str], dict[str, object]]:
    """Validate a future source-supplied dB/dt authority contract."""
    if not isinstance(contract_value, Mapping):
        return ["dbdt_authority_contract_absent"], {"authority_contract": None}
    contract = dict(contract_value)
    blockers: list[str] = []
    if contract.get("schema_version") != DBDT_AUTHORITY_SCHEMA:
        blockers.append("dbdt_authority_contract_schema_mismatch")
    if (
        contract.get("mapping_commit") != FAIR_MAST_MAPPING_COMMIT
        or contract.get("mapping_file_sha256") != FAIR_MAST_MAPPING_SHA256
    ):
        blockers.append("dbdt_mapping_lineage_mismatch")
    source_quantity = contract.get("source_quantity")
    source_units = contract.get("source_units")
    transform = contract.get("transform")
    expected = {
        "magnetic_field": ("T", "differentiate_once_then_multiply_by_1e4_G_per_T"),
        "magnetic_field_time_derivative": ("T/s", "multiply_by_1e4_G_per_T_without_differentiation"),
    }
    if source_quantity not in expected:
        blockers.append("dbdt_source_quantity_not_attested")
    else:
        expected_units, expected_transform = expected[cast(str, source_quantity)]
        if source_units != expected_units:
            blockers.append("dbdt_source_units_do_not_match_quantity")
        if transform != expected_transform:
            blockers.append("dbdt_transform_does_not_match_source_quantity")
    if source_units != observed_source_units:
        blockers.append("dbdt_contract_units_do_not_match_source_metadata")
    if tuple(contract.get("source_channels") or ()) != EXPECTED_SOURCE_CHANNELS:
        blockers.append("dbdt_source_row_identities_not_attested")
    if tuple(contract.get("geometry_channels") or ()) != EXPECTED_GEOMETRY_CHANNELS:
        blockers.append("dbdt_geometry_row_identities_not_attested")
    if tuple(contract.get("source_to_geometry_indices") or ()) != EXPECTED_SOURCE_TO_GEOMETRY_INDICES:
        blockers.append("dbdt_field_geometry_join_indices_not_attested")
    for key, blocker in (
        ("field_to_geometry_join_sha256", "dbdt_field_geometry_join_evidence_absent"),
        ("mapping_scale_evidence_sha256", "dbdt_mapping_scale_evidence_absent"),
        ("calibration_evidence_sha256", "dbdt_calibration_evidence_absent"),
    ):
        if not _is_sha256(contract.get(key)):
            blockers.append(blocker)
    contract_digest = contract.get("authority_contract_sha256")
    digest_payload = dict(contract)
    digest_payload["authority_contract_sha256"] = None
    if not _is_sha256(contract_digest):
        blockers.append("dbdt_authority_contract_digest_absent")
    elif contract_digest != canonical_json_sha256(digest_payload):
        blockers.append("dbdt_authority_contract_digest_mismatch")
    if contract.get("mapping_scale") != MAPPING_SCALE or not _nonempty_text(contract.get("mapping_scale_units")):
        blockers.append("dbdt_mapping_scale_semantics_not_attested")
    for key, blocker in (
        ("measured_component", "dbdt_measured_component_not_attested"),
        ("probe_orientation_policy", "dbdt_probe_orientation_not_attested"),
        ("sign_convention", "dbdt_sign_convention_not_attested"),
        ("probe_reduction_policy", "dbdt_probe_reduction_policy_absent"),
        ("missing_data_policy", "dbdt_missing_data_policy_absent"),
        ("bad_channel_policy", "dbdt_bad_channel_policy_absent"),
        ("filter_policy", "dbdt_filter_policy_absent"),
        ("edge_policy", "dbdt_edge_policy_absent"),
    ):
        if not _nonempty_text(contract.get(key)):
            blockers.append(blocker)
    uncertainty = contract.get("standard_uncertainty_gauss_per_s")
    if not _is_positive_finite_number(uncertainty):
        blockers.append("dbdt_standard_uncertainty_absent")
    cutoff = contract.get("low_pass_cutoff_hz")
    if not _is_positive_finite_number(cutoff):
        blockers.append("dbdt_low_pass_cutoff_not_attested")
    elif sample_rate_hz is not None and cutoff >= 0.5 * sample_rate_hz:
        blockers.append("dbdt_low_pass_cutoff_not_below_nyquist")
    if tuple(contract.get("primary_source_citations") or ()) != (
        FAIR_MAST_DBDT_PROFILE_URL,
        MAST_MAGNETICS_PAPER_URL,
    ):
        blockers.append("dbdt_primary_source_citations_absent")
    return blockers, {"authority_contract": contract}


def _source_evidence(artifact: VerifiedSourceArtifact) -> tuple[list[str], dict[str, object], float | None]:
    """Measure exact members, metadata conflicts, shapes, finite rows, and rate."""
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    required = (FIELD_KEY, TIMEBASE_KEY, *GEOMETRY_KEYS)
    missing = [key for key in required if key not in artifact.arrays or key not in artifact.metadata]
    if missing:
        return ["dbdt_source_members_absent"], {"missing_source_keys": missing}, None

    field = np.asarray(artifact.arrays[FIELD_KEY], dtype=np.float64)
    time = np.asarray(artifact.arrays[TIMEBASE_KEY], dtype=np.float64)
    geometry = [np.asarray(artifact.arrays[key], dtype=np.float64) for key in GEOMETRY_KEYS]
    evidence["source_shapes"] = {
        FIELD_KEY: list(field.shape),
        TIMEBASE_KEY: list(time.shape),
        **{key: list(values.shape) for key, values in zip(GEOMETRY_KEYS, geometry, strict=True)},
    }
    if field.ndim != 2 or field.shape[0] != 5 or time.ndim != 1 or field.shape[1] != time.size:
        blockers.append("dbdt_field_or_timebase_shape_mismatch")
    if any(values.ndim != 1 or values.size != 40 for values in geometry):
        blockers.append("dbdt_geometry_shape_mismatch")
    geometry_finite_counts = [int(np.count_nonzero(np.isfinite(values))) for values in geometry]
    evidence["geometry_finite_samples"] = dict(zip(GEOMETRY_KEYS, geometry_finite_counts, strict=True))
    if any(count != values.size for count, values in zip(geometry_finite_counts, geometry, strict=True)):
        blockers.append("dbdt_geometry_has_nonfinite_values")

    field_metadata = artifact.metadata[FIELD_KEY]
    expected_dimensions = ("b_field_pol_probe_cc_channel", "time_mirnov")
    if tuple(field_metadata.get("dimensions") or ()) != expected_dimensions:
        blockers.append("dbdt_field_dimensions_mismatch")
    if field_metadata.get("units") not in {"T", "T/s"}:
        blockers.append("dbdt_source_units_unsupported")
    time_metadata = artifact.metadata[TIMEBASE_KEY]
    if tuple(time_metadata.get("dimensions") or ()) != ("time_mirnov",) or time_metadata.get("units") != "s":
        blockers.append("dbdt_timebase_metadata_mismatch")
    geometry_units = [artifact.metadata[key].get("units") for key in GEOMETRY_KEYS]
    evidence["geometry_units"] = dict(zip(GEOMETRY_KEYS, geometry_units, strict=True))
    if geometry_units != ["degrees", "m", "m"]:
        blockers.append("dbdt_geometry_units_not_component_specific")

    attributes_value = field_metadata.get("source_attributes")
    attributes: Mapping[str, object] = attributes_value if isinstance(attributes_value, Mapping) else {}
    evidence["field_metadata"] = {
        "name": attributes.get("name"),
        "uda_name": attributes.get("uda_name"),
        "label": attributes.get("label"),
        "units": attributes.get("units"),
        "description": attributes.get("description"),
    }
    if attributes.get("label") == "Tesla/sec" and field_metadata.get("units") == "T":
        blockers.append("dbdt_source_label_unit_conflict")
    if attributes.get("mapping_scale") != MAPPING_SCALE:
        blockers.append("dbdt_mapping_scale_not_preserved")

    finite_counts: list[int] = []
    nonfinite_positions: list[list[int]] = []
    if field.ndim == 2 and field.shape[0] == 5:
        for row in field:
            finite_counts.append(int(np.count_nonzero(np.isfinite(row))))
            nonfinite_positions.append([int(value) for value in np.flatnonzero(~np.isfinite(row))])
        evidence["field_finite_samples_per_row"] = finite_counts
        evidence["field_nonfinite_positions_per_row"] = nonfinite_positions
        if any(count != field.shape[1] for count in finite_counts):
            blockers.append("dbdt_source_row_has_nonfinite_samples")

    sample_rate: float | None = None
    if time.ndim == 1 and time.size > 1 and bool(np.all(np.isfinite(time))):
        differences = np.diff(time)
        if bool(np.all(differences > 0.0)):
            median_period = float(np.median(differences))
            sample_rate = 1.0 / median_period
            evidence["sample_period_s"] = median_period
            evidence["sample_rate_hz"] = sample_rate
            evidence["timebase_uniform"] = bool(np.allclose(differences, median_period, rtol=1.0e-9, atol=1.0e-12))
            if evidence["timebase_uniform"] is False:
                blockers.append("dbdt_timebase_not_uniform")
        else:
            blockers.append("dbdt_timebase_not_strictly_increasing")
    else:
        blockers.append("dbdt_timebase_not_finite")

    geometry_metadata: list[dict[str, object]] = []
    for key in GEOMETRY_KEYS:
        metadata = artifact.metadata[key]
        source_attributes = metadata.get("source_attributes")
        geometry_attributes: Mapping[str, object] = source_attributes if isinstance(source_attributes, Mapping) else {}
        geometry_metadata.append(
            {
                "source_key": key,
                "status": geometry_attributes.get("status"),
                "revision": geometry_attributes.get("revision"),
                "creator_commit_id": geometry_attributes.get("creatorCommitId"),
                "signed_off_by": geometry_attributes.get("signedOffBy"),
                "value_sha256": metadata.get("value_sha256"),
            }
        )
        if geometry_attributes.get("status") not in {"released", "approved"}:
            blockers.append("dbdt_geometry_not_released")
        creator_commit = geometry_attributes.get("creatorCommitId")
        if (
            not isinstance(creator_commit, str)
            or len(creator_commit) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in creator_commit)
        ):
            blockers.append("dbdt_geometry_creator_commit_absent")
        if geometry_attributes.get("signedOffBy") in {None, ""}:
            blockers.append("dbdt_geometry_signoff_absent")
        if not _is_sha256(metadata.get("value_sha256")):
            blockers.append("dbdt_geometry_value_digest_absent")
    evidence["geometry_metadata"] = geometry_metadata
    return blockers, evidence, sample_rate


def assess_dbdt_authority(artifact: VerifiedSourceArtifact) -> dict[str, object]:
    """Assess one verified source artefact without deriving dB/dt."""
    source_blockers, source_evidence, sample_rate = _source_evidence(artifact)
    field_metadata = artifact.metadata.get(FIELD_KEY)
    source_attributes = field_metadata.get("source_attributes") if isinstance(field_metadata, Mapping) else None
    contract_value = (
        source_attributes.get("dbdt_authority_contract") if isinstance(source_attributes, Mapping) else None
    )
    contract_blockers, contract_evidence = _validate_authority_contract(
        contract_value,
        sample_rate_hz=sample_rate,
        observed_source_units=field_metadata.get("units") if isinstance(field_metadata, Mapping) else None,
    )
    blockers = list(dict.fromkeys((*source_blockers, *contract_blockers)))
    report: dict[str, object] = {
        "schema_version": DBDT_AUTHORITY_SCHEMA,
        "spec_sha256": mast_dbdt_authority_spec()["payload_sha256"],
        "shot_id": artifact.shot_id,
        "source_uri": artifact.source_uri,
        "artifact_sha256": artifact.artifact_sha256,
        "source_evidence": source_evidence,
        "contract_evidence": contract_evidence,
        "dbdt_transform_executed": False,
        "canonical_dbdt_binding_admissible": not blockers,
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


def build_dbdt_authority_report(artifacts: Sequence[VerifiedSourceArtifact]) -> dict[str, object]:
    """Build one deterministic multi-shot dB/dt authority assessment."""
    if not artifacts:
        raise DbdtAuthorityError("at least one verified source artefact is required")
    shot_ids = [artifact.shot_id for artifact in artifacts]
    if shot_ids != sorted(set(shot_ids)):
        raise DbdtAuthorityError("source artefacts must have unique, ascending shot identifiers")
    assessments = [assess_dbdt_authority(artifact) for artifact in artifacts]
    admissible = all(bool(item["canonical_dbdt_binding_admissible"]) for item in assessments)
    report: dict[str, object] = {
        "schema_version": DBDT_AUTHORITY_SCHEMA,
        "spec": mast_dbdt_authority_spec(),
        "status": "authority_verified" if admissible else "blocked",
        "canonical_dbdt_binding_admissible": admissible,
        "shot_count": len(assessments),
        "shots": assessments,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Write a real-object L2F-12d dB/dt authority assessment."""
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
    report = build_dbdt_authority_report(artifacts)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.json_out.open("x", encoding="utf-8") as report_handle:
            json.dump(report, report_handle, indent=2, sort_keys=True)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
    except FileExistsError as exc:
        raise DbdtAuthorityError("refusing to overwrite an existing authority report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"{report['status']}: {report['shot_count']} shot(s) -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
