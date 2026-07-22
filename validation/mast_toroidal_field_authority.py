#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST toroidal-field source-authority gate
"""Admit canonical MAST ``BT_T`` only from complete physical authority.

FAIR-MAST exposes distinct total-field and vacuum-field arrays.  A shared unit,
timebase, or similar value range does not make those quantities interchangeable.
This module specifies the direct total-field route and the alternative TF-current
derivation, then assesses verified source artefacts without deriving or relabelling
any channel.  Missing sign or uncertainty authority remains an explicit blocker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import numpy as np

from validation.mast_source_artifact_reader import (
    VerifiedSourceArtifact,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256

TOROIDAL_FIELD_AUTHORITY_SCHEMA = "scpn-control.mast-toroidal-field-authority.v1.0.0"
TOROIDAL_FIELD_AUTHORITY_VERSION = "1.0.0"
FAIR_MAST_MAPPING_URL = (
    "https://github.com/ukaea/fair-mast-ingestion/blob/"
    "862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L224-L233"
)
FAIR_MAST_PAPER_URL = "https://doi.org/10.1016/j.softx.2024.101869"

DIRECT_FIELD_KEY = "equilibrium.bphi_rmag"
VACUUM_CANDIDATE_KEY = "equilibrium.bvac_rmag"
REFERENCE_RADIUS_KEY = "equilibrium.magnetic_axis_r"
TIMEBASE_KEY = "equilibrium.time"

UncertaintyKind = Literal["constant_standard_uncertainty", "per_sample_standard_uncertainty"]


class ToroidalFieldAuthorityError(ValueError):
    """Raised when a toroidal-field authority declaration is inconsistent."""


@dataclass(frozen=True)
class DirectFieldAuthority:
    """External authority needed in addition to FAIR-MAST array metadata.

    The source artefact proves values, units, dimensions, timebase, and reference
    radius.  This declaration must independently source the physical positive
    direction, total-versus-vacuum interpretation, and one-standard-deviation
    uncertainty.  Observed data signs and guessed machine constants are invalid
    substitutes.
    """

    positive_direction: str
    positive_direction_citation: str
    positive_direction_evidence_sha256: str
    quantity_citation: str
    quantity_evidence_sha256: str
    uncertainty_kind: UncertaintyKind
    uncertainty_citation: str
    uncertainty_evidence_sha256: str
    constant_standard_uncertainty_t: float | None = None
    uncertainty_key: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete, non-sourceable, or dimensionally ambiguous authority."""
        strings = (
            self.positive_direction,
            self.positive_direction_citation,
            self.quantity_citation,
            self.uncertainty_citation,
        )
        if any(not value.strip() for value in strings):
            raise ToroidalFieldAuthorityError("authority statements and citations must be non-empty")
        citations = (
            self.positive_direction_citation,
            self.quantity_citation,
            self.uncertainty_citation,
        )
        if any(not _is_trusted_authority_url(citation) for citation in citations):
            raise ToroidalFieldAuthorityError("authority citations must use a trusted primary-source HTTPS URL")
        evidence_digests = (
            self.positive_direction_evidence_sha256,
            self.quantity_evidence_sha256,
            self.uncertainty_evidence_sha256,
        )
        if any(not _is_sha256(digest) for digest in evidence_digests):
            raise ToroidalFieldAuthorityError("authority evidence digests must be lowercase SHA-256")
        if self.uncertainty_kind == "constant_standard_uncertainty":
            value = self.constant_standard_uncertainty_t
            if value is None or not math.isfinite(value) or value <= 0.0:
                raise ToroidalFieldAuthorityError("constant standard uncertainty must be finite and positive")
            if self.uncertainty_key is not None:
                raise ToroidalFieldAuthorityError("constant uncertainty cannot also name a per-sample key")
        elif self.uncertainty_kind == "per_sample_standard_uncertainty":
            if not self.uncertainty_key:
                raise ToroidalFieldAuthorityError("per-sample uncertainty requires an exact source key")
            if self.constant_standard_uncertainty_t is not None:
                raise ToroidalFieldAuthorityError("per-sample uncertainty cannot also provide a constant")
        else:
            raise ToroidalFieldAuthorityError(f"unsupported uncertainty kind {self.uncertainty_kind!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-ready representation."""
        return {
            "positive_direction": self.positive_direction,
            "positive_direction_citation": self.positive_direction_citation,
            "positive_direction_evidence_sha256": self.positive_direction_evidence_sha256,
            "quantity_citation": self.quantity_citation,
            "quantity_evidence_sha256": self.quantity_evidence_sha256,
            "uncertainty_kind": self.uncertainty_kind,
            "uncertainty_citation": self.uncertainty_citation,
            "uncertainty_evidence_sha256": self.uncertainty_evidence_sha256,
            "constant_standard_uncertainty_t": self.constant_standard_uncertainty_t,
            "uncertainty_key": self.uncertainty_key,
        }


def _is_sha256(value: str) -> bool:
    """Return whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_trusted_authority_url(value: str) -> bool:
    """Restrict declarations to primary FAIR-MAST or UKAEA publication surfaces."""
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
        return False
    host = parsed.hostname
    if host in {"mastapp.site", "ukaea.uk", "www.ukaea.uk", "doi.org", "ieeexplore.ieee.org"}:
        return True
    return host == "github.com" and parsed.path.startswith("/ukaea/")


def mast_toroidal_field_authority_spec() -> dict[str, object]:
    """Return the deterministic, self-digested L2F-12a authority specification."""
    payload: dict[str, object] = {
        "schema_version": TOROIDAL_FIELD_AUTHORITY_SCHEMA,
        "version": TOROIDAL_FIELD_AUTHORITY_VERSION,
        "machine": "MAST",
        "canonical_channel": "BT_T",
        "canonical_quantity": "signed total toroidal magnetic field at the magnetic-axis major radius",
        "canonical_units": "T",
        "direct_total_field_route": {
            "field_key": DIRECT_FIELD_KEY,
            "reference_radius_key": REFERENCE_RADIUS_KEY,
            "timebase_key": TIMEBASE_KEY,
            "transform": "identity_preserve_source_sign",
            "required_field_attributes": {"name": "bphi_rmag", "uda_name": "EFM_BPHI_RMAG"},
            "required_radius_attributes": {
                "name": "magnetic_axis_r",
                "uda_name": "EFM_MAGNETIC_AXIS_R",
            },
            "requires_external_authority": [
                "total_versus_vacuum_quantity",
                "physical_positive_direction",
                "one_standard_deviation_uncertainty",
            ],
        },
        "vacuum_candidate": {
            "field_key": VACUUM_CANDIDATE_KEY,
            "status": "inadmissible_as_total_field_without_sourced_equivalence_policy",
            "forbidden_transform": "identity_or_absolute_value_relabelling_as_BT_T",
        },
        "tf_current_derivation_route": {
            "formula": "B_phi(R,t) = polarity * K_TF * I_TF(t) / R(t)",
            "required_inputs": [
                "exact signed TF-current source and timebase",
                "sourced K_TF in T m A^-1 and its standard uncertainty",
                "positive-current and positive-field direction authority",
                "positive reference-radius source in m and its standard uncertainty",
                "input covariance declaration or an explicit independence authority",
            ],
            "status": "blocked_until_every_input_is_source-attested",
            "prohibited": ["ideal-solenoid constant guess", "implicit polarity", "zero uncertainty"],
        },
        "missing_data_rule": "emit a validity mask; never interpolate across gaps, zero-fill, or take absolute value",
        "citations": [FAIR_MAST_MAPPING_URL, FAIR_MAST_PAPER_URL],
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _metadata_blockers(artifact: VerifiedSourceArtifact) -> tuple[list[str], dict[str, object]]:
    """Verify direct-route members, metadata, shapes, and finite joint coverage."""
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    required = (DIRECT_FIELD_KEY, REFERENCE_RADIUS_KEY, TIMEBASE_KEY)
    missing = [key for key in required if key not in artifact.arrays]
    if missing:
        blockers.append("direct_route_source_members_absent")
        evidence["missing_source_keys"] = missing
        return blockers, evidence

    expected = {
        DIRECT_FIELD_KEY: (("time",), "T", "bphi_rmag", "EFM_BPHI_RMAG"),
        REFERENCE_RADIUS_KEY: (("time",), "m", "magnetic_axis_r", "EFM_MAGNETIC_AXIS_R"),
        TIMEBASE_KEY: (("time",), "s", None, None),
    }
    metadata_mismatches: list[str] = []
    for key, (dimensions, units, name, uda_name) in expected.items():
        metadata = artifact.metadata[key]
        attributes = metadata.get("source_attributes")
        if metadata.get("metadata_status") != "source_xarray":
            metadata_mismatches.append(f"{key}:metadata_status")
        if tuple(metadata.get("dimensions") or ()) != dimensions:
            metadata_mismatches.append(f"{key}:dimensions")
        if metadata.get("units") != units:
            metadata_mismatches.append(f"{key}:units")
        if name is not None and isinstance(attributes, Mapping):
            if attributes.get("name") != name:
                metadata_mismatches.append(f"{key}:name")
            if attributes.get("uda_name") != uda_name:
                metadata_mismatches.append(f"{key}:uda_name")
        elif name is not None:
            metadata_mismatches.append(f"{key}:source_attributes")
    if metadata_mismatches:
        blockers.append("direct_route_source_metadata_mismatch")
        evidence["metadata_mismatches"] = metadata_mismatches

    field = np.asarray(artifact.arrays[DIRECT_FIELD_KEY], dtype=np.float64)
    radius = np.asarray(artifact.arrays[REFERENCE_RADIUS_KEY], dtype=np.float64)
    time = np.asarray(artifact.arrays[TIMEBASE_KEY], dtype=np.float64)
    evidence["source_shapes"] = {
        DIRECT_FIELD_KEY: list(field.shape),
        REFERENCE_RADIUS_KEY: list(radius.shape),
        TIMEBASE_KEY: list(time.shape),
    }
    if field.ndim != 1 or radius.ndim != 1 or time.ndim != 1 or not (field.shape == radius.shape == time.shape):
        blockers.append("direct_route_source_shape_mismatch")
        return blockers, evidence

    finite = np.isfinite(field) & np.isfinite(radius) & np.isfinite(time) & (radius > 0.0)
    finite_count = int(np.count_nonzero(finite))
    evidence["joint_finite_positive_radius_samples"] = finite_count
    evidence["total_samples"] = int(field.size)
    if finite_count == 0:
        blockers.append("direct_route_has_no_joint_finite_positive_radius_samples")
        return blockers, evidence
    finite_time = time[finite]
    if finite_time.size > 1 and not bool(np.all(np.diff(finite_time) > 0.0)):
        blockers.append("direct_route_timebase_not_strictly_increasing")
    evidence["observed_field_sign_counts_not_authority"] = {
        "negative": int(np.count_nonzero(field[finite] < 0.0)),
        "zero": int(np.count_nonzero(field[finite] == 0.0)),
        "positive": int(np.count_nonzero(field[finite] > 0.0)),
    }

    if VACUUM_CANDIDATE_KEY in artifact.arrays:
        vacuum = np.asarray(artifact.arrays[VACUUM_CANDIDATE_KEY], dtype=np.float64)
        joint = finite & np.isfinite(vacuum) if vacuum.shape == field.shape else np.zeros(field.shape, dtype=np.bool_)
        evidence["vacuum_candidate_comparison"] = {
            "source_key": VACUUM_CANDIDATE_KEY,
            "same_shape": vacuum.shape == field.shape,
            "joint_finite_samples": int(np.count_nonzero(joint)),
            "byte_or_value_equal_with_nan": bool(np.array_equal(field, vacuum, equal_nan=True)),
            "maximum_absolute_difference_t": (
                float(np.max(np.abs(field[joint] - vacuum[joint]))) if bool(np.any(joint)) else None
            ),
        }
    return blockers, evidence


def _uncertainty_blockers(
    artifact: VerifiedSourceArtifact,
    authority: DirectFieldAuthority,
) -> tuple[list[str], dict[str, object]]:
    """Verify the declared one-standard-deviation uncertainty surface."""
    evidence: dict[str, object] = {"kind": authority.uncertainty_kind}
    if authority.uncertainty_kind == "constant_standard_uncertainty":
        evidence["constant_standard_uncertainty_t"] = authority.constant_standard_uncertainty_t
        return [], evidence
    key = authority.uncertainty_key
    if key is None:
        raise ToroidalFieldAuthorityError("per-sample authority lost its validated uncertainty key")
    if key not in artifact.arrays or key not in artifact.metadata:
        return ["declared_uncertainty_source_member_absent"], {**evidence, "source_key": key}
    metadata = artifact.metadata[key]
    uncertainty = np.asarray(artifact.arrays[key], dtype=np.float64)
    field = np.asarray(artifact.arrays[DIRECT_FIELD_KEY], dtype=np.float64)
    evidence.update({"source_key": key, "shape": list(uncertainty.shape), "units": metadata.get("units")})
    if metadata.get("metadata_status") != "source_xarray" or metadata.get("units") != "T":
        return ["declared_uncertainty_source_metadata_mismatch"], evidence
    if uncertainty.shape != field.shape or uncertainty.ndim != 1:
        return ["declared_uncertainty_source_shape_mismatch"], evidence
    finite_field = np.isfinite(field)
    if bool(np.any(~np.isfinite(uncertainty[finite_field]) | (uncertainty[finite_field] < 0.0))):
        return ["declared_uncertainty_is_missing_negative_or_non_finite"], evidence
    return [], evidence


def assess_toroidal_field_authority(
    artifact: VerifiedSourceArtifact,
    *,
    authority: DirectFieldAuthority | None = None,
) -> dict[str, object]:
    """Assess one verified source artefact without producing canonical ``BT_T``."""
    blockers, source_evidence = _metadata_blockers(artifact)
    authority_evidence: dict[str, object] | None = None
    uncertainty_evidence: dict[str, object] | None = None
    if authority is None:
        blockers.extend(
            (
                "total_versus_vacuum_quantity_authority_absent",
                "physical_positive_direction_authority_absent",
                "one_standard_deviation_uncertainty_authority_absent",
            )
        )
    else:
        authority_evidence = authority.to_dict()
        uncertainty_blockers, uncertainty_evidence = _uncertainty_blockers(artifact, authority)
        blockers.extend(uncertainty_blockers)
    blockers = list(dict.fromkeys(blockers))
    report: dict[str, object] = {
        "schema_version": TOROIDAL_FIELD_AUTHORITY_SCHEMA,
        "spec_sha256": mast_toroidal_field_authority_spec()["payload_sha256"],
        "shot_id": artifact.shot_id,
        "source_uri": artifact.source_uri,
        "artifact_sha256": artifact.artifact_sha256,
        "direct_total_field_route": source_evidence,
        "authority_declaration": authority_evidence,
        "uncertainty_evidence": uncertainty_evidence,
        "canonical_bt_binding_admissible": not blockers,
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


def build_toroidal_field_authority_report(
    artifacts: Sequence[VerifiedSourceArtifact],
) -> dict[str, object]:
    """Build a deterministic multi-shot assessment for the current no-authority state."""
    if not artifacts:
        raise ToroidalFieldAuthorityError("at least one verified source artefact is required")
    shot_ids = [artifact.shot_id for artifact in artifacts]
    if shot_ids != sorted(set(shot_ids)):
        raise ToroidalFieldAuthorityError("source artefacts must have unique, ascending shot identifiers")
    assessments = [assess_toroidal_field_authority(artifact) for artifact in artifacts]
    report: dict[str, object] = {
        "schema_version": TOROIDAL_FIELD_AUTHORITY_SCHEMA,
        "spec": mast_toroidal_field_authority_spec(),
        "status": "authority_verified"
        if all(item["canonical_bt_binding_admissible"] for item in assessments)
        else "blocked",
        "canonical_bt_binding_admissible": all(item["canonical_bt_binding_admissible"] for item in assessments),
        "shot_count": len(assessments),
        "shots": assessments,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Write a real-object L2F-12a authority assessment."""
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
    report = build_toroidal_field_authority_report(artifacts)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.json_out.open("x", encoding="utf-8") as report_handle:
            json.dump(report, report_handle, indent=2, sort_keys=True)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
    except FileExistsError as exc:
        raise ToroidalFieldAuthorityError("refusing to overwrite an existing authority report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"{report['status']}: {report['shot_count']} shot(s) -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
