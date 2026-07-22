#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST normalised-beta source-authority gate
"""Diagnose FAIR-MAST normalised-beta units without guessing a conversion.

The live ``EFM_BETAN`` array declares tesla while its FAIR-MAST description and
the IMAS Data Dictionary define a conventional normalised-beta quantity with
unit ``1``.  Treating this as a numerical T-to-1 conversion would be physically
wrong.  This gate records the metadata conflict, the missing formula-input
lineage, negative-value validity ambiguity, and absent uncertainty authority.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from validation.mast_source_artifact_reader import (
    VerifiedSourceArtifact,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256

NORMALISED_BETA_AUTHORITY_SCHEMA = "scpn-control.mast-normalised-beta-authority.v1.0.0"
NORMALISED_BETA_AUTHORITY_VERSION = "1.0.0"
FAIR_MAST_BETA_MAPPING_URL = (
    "https://github.com/ukaea/fair-mast-ingestion/blob/"
    "862f08d7d91930b988d674e7ec67f3a03aacafac/mappings/level2/mast.yml#L201-L207"
)
IMAS_BETA_DEFINITION_URL = (
    "https://github.com/iterorganization/IMAS-Data-Dictionary/blob/"
    "ba4fa8a57b0c5f35d3225fdeb5e0ea08f1c8d16f/schemas/equilibrium/dd_equilibrium.xsd#L751-L759"
)

BETA_N_KEY = "equilibrium.beta_tor_normal"
BETA_TOR_KEY = "equilibrium.beta_tor"
MINOR_RADIUS_KEY = "equilibrium.minor_radius"
VACUUM_FIELD_GEOMETRIC_AXIS_KEY = "equilibrium.bvac_rgeom"
FITTED_PLASMA_CURRENT_KEY = "equilibrium.plasma_current_x"
TIMEBASE_KEY = "equilibrium.time"

_SOURCE_DESCRIPTION = "Normalized toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]"
_SOURCE_IMAS_TARGET = "equilibrium.time_slice[:].global_quantities.beta_tor_normal"


class NormalisedBetaAuthorityError(ValueError):
    """Raised when a normalised-beta authority report cannot be constructed."""


def mast_normalised_beta_authority_spec() -> dict[str, object]:
    """Return the deterministic, self-digested L2F-12b authority specification."""
    payload: dict[str, object] = {
        "schema_version": NORMALISED_BETA_AUTHORITY_SCHEMA,
        "version": NORMALISED_BETA_AUTHORITY_VERSION,
        "machine": "MAST",
        "canonical_channel": "beta_N",
        "canonical_definition": "100 * beta_tor * a[m] * B0[T] / Ip[MA]",
        "canonical_units": "1",
        "direct_source": {
            "source_key": BETA_N_KEY,
            "uda_name": "EFM_BETAN",
            "observed_source_units": "T",
            "observed_imas_target": _SOURCE_IMAS_TARGET,
            "current_imas_leaf": "equilibrium.time_slice[:].global_quantities.beta_tor_norm",
            "candidate_transform": "identity_values_with_metadata_unit_repair_only",
            "numeric_unit_conversion": "forbidden",
        },
        "independent_reproduction_route": {
            "required_source_keys": [
                BETA_TOR_KEY,
                MINOR_RADIUS_KEY,
                VACUUM_FIELD_GEOMETRIC_AXIS_KEY,
                FITTED_PLASMA_CURRENT_KEY,
                TIMEBASE_KEY,
            ],
            "required_semantics": [
                "whether beta_tor source values are fractional or percent",
                "B0 must be the exact vacuum field used by EFM at the geometric axis",
                "Ip sign or magnitude convention",
                "common native timebase or a source-attested alignment",
                "equilibrium validity or reconstruction-quality rule",
                "one-standard-deviation uncertainty and covariance policy",
            ],
        },
        "missing_data_rule": "preserve a validity mask; never zero-fill, clamp negatives, or infer scale from range",
        "citations": [FAIR_MAST_BETA_MAPPING_URL, IMAS_BETA_DEFINITION_URL],
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _source_evidence(artifact: VerifiedSourceArtifact) -> tuple[list[str], dict[str, object]]:
    """Verify the direct array and record its exact source-semantic conflicts."""
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    required = (BETA_N_KEY, TIMEBASE_KEY)
    missing = [key for key in required if key not in artifact.arrays]
    if missing:
        return ["normalised_beta_source_members_absent"], {"missing_source_keys": missing}

    beta_metadata = artifact.metadata[BETA_N_KEY]
    time_metadata = artifact.metadata[TIMEBASE_KEY]
    attributes = beta_metadata.get("source_attributes")
    mismatches: list[str] = []
    if beta_metadata.get("metadata_status") != "source_xarray":
        mismatches.append(f"{BETA_N_KEY}:metadata_status")
    if tuple(beta_metadata.get("dimensions") or ()) != ("time",):
        mismatches.append(f"{BETA_N_KEY}:dimensions")
    if beta_metadata.get("units") != "T":
        mismatches.append(f"{BETA_N_KEY}:observed_units")
    if not isinstance(attributes, Mapping):
        mismatches.append(f"{BETA_N_KEY}:source_attributes")
    else:
        expected_attributes = {
            "name": "beta_tor_normal",
            "uda_name": "EFM_BETAN",
            "description": _SOURCE_DESCRIPTION,
            "imas": _SOURCE_IMAS_TARGET,
            "units": "T",
        }
        for name, expected in expected_attributes.items():
            if attributes.get(name) != expected:
                mismatches.append(f"{BETA_N_KEY}:{name}")
    if time_metadata.get("metadata_status") != "source_xarray":
        mismatches.append(f"{TIMEBASE_KEY}:metadata_status")
    if tuple(time_metadata.get("dimensions") or ()) != ("time",):
        mismatches.append(f"{TIMEBASE_KEY}:dimensions")
    if time_metadata.get("units") != "s":
        mismatches.append(f"{TIMEBASE_KEY}:units")
    if mismatches:
        blockers.append("normalised_beta_source_metadata_mismatch")
        evidence["metadata_mismatches"] = mismatches

    beta_n = np.asarray(artifact.arrays[BETA_N_KEY], dtype=np.float64)
    time = np.asarray(artifact.arrays[TIMEBASE_KEY], dtype=np.float64)
    evidence["source_shapes"] = {BETA_N_KEY: list(beta_n.shape), TIMEBASE_KEY: list(time.shape)}
    if beta_n.ndim != 1 or time.ndim != 1 or beta_n.shape != time.shape:
        blockers.append("normalised_beta_source_shape_mismatch")
        return blockers, evidence
    finite = np.isfinite(beta_n) & np.isfinite(time)
    count = int(np.count_nonzero(finite))
    evidence["finite_samples"] = count
    evidence["total_samples"] = int(beta_n.size)
    if count == 0:
        blockers.append("normalised_beta_has_no_finite_time_aligned_samples")
        return blockers, evidence
    finite_time = time[finite]
    if finite_time.size > 1 and not bool(np.all(np.diff(finite_time) > 0.0)):
        blockers.append("normalised_beta_timebase_not_strictly_increasing")
    values = beta_n[finite]
    negative_count = int(np.count_nonzero(values < 0.0))
    evidence["observed_value_summary_not_scale_authority"] = {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "negative": negative_count,
        "zero": int(np.count_nonzero(values == 0.0)),
        "positive": int(np.count_nonzero(values > 0.0)),
    }
    if negative_count:
        blockers.append("normalised_beta_negative_value_validity_policy_absent")
    return blockers, evidence


def _formula_input_evidence(artifact: VerifiedSourceArtifact) -> tuple[list[str], dict[str, object]]:
    """Record whether every independent-reproduction input is lineage-bound."""
    required = (
        BETA_TOR_KEY,
        MINOR_RADIUS_KEY,
        VACUUM_FIELD_GEOMETRIC_AXIS_KEY,
        FITTED_PLASMA_CURRENT_KEY,
        TIMEBASE_KEY,
    )
    present = [key for key in required if key in artifact.arrays]
    missing = [key for key in required if key not in artifact.arrays]
    evidence: dict[str, object] = {"present_source_keys": present, "missing_source_keys": missing}
    if missing:
        return ["normalised_beta_formula_inputs_not_lineage_bound"], evidence

    shapes = {key: list(np.asarray(artifact.arrays[key]).shape) for key in required}
    evidence["source_shapes"] = shapes
    expected_shape = np.asarray(artifact.arrays[TIMEBASE_KEY]).shape
    if any(
        np.asarray(artifact.arrays[key]).ndim != 1 or np.asarray(artifact.arrays[key]).shape != expected_shape
        for key in required
    ):
        return ["normalised_beta_formula_input_shape_mismatch"], evidence
    return [], evidence


def assess_normalised_beta_authority(artifact: VerifiedSourceArtifact) -> dict[str, object]:
    """Assess one source artefact without repairing metadata or values."""
    blockers, direct_evidence = _source_evidence(artifact)
    formula_blockers, formula_evidence = _formula_input_evidence(artifact)
    blockers.extend(formula_blockers)
    blockers.extend(
        (
            "fair_mast_beta_n_source_unit_conflicts_with_imas",
            "fair_mast_beta_n_imas_target_name_is_not_current",
            "normalised_beta_formula_sign_and_scale_authority_absent",
            "normalised_beta_reconstruction_quality_authority_absent",
            "one_standard_deviation_uncertainty_authority_absent",
        )
    )
    blockers = list(dict.fromkeys(blockers))
    report: dict[str, object] = {
        "schema_version": NORMALISED_BETA_AUTHORITY_SCHEMA,
        "spec_sha256": mast_normalised_beta_authority_spec()["payload_sha256"],
        "shot_id": artifact.shot_id,
        "source_uri": artifact.source_uri,
        "artifact_sha256": artifact.artifact_sha256,
        "direct_source_evidence": direct_evidence,
        "formula_input_evidence": formula_evidence,
        "metadata_repair_candidate": {
            "from_units": "T",
            "to_units": "1",
            "numeric_transform": None,
            "admissible": False,
        },
        "canonical_beta_n_binding_admissible": False,
        "status": "blocked",
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


def build_normalised_beta_authority_report(artifacts: Sequence[VerifiedSourceArtifact]) -> dict[str, object]:
    """Build a deterministic multi-shot L2F-12b blocker report."""
    if not artifacts:
        raise NormalisedBetaAuthorityError("at least one verified source artefact is required")
    shot_ids = [artifact.shot_id for artifact in artifacts]
    if shot_ids != sorted(set(shot_ids)):
        raise NormalisedBetaAuthorityError("source artefacts must have unique, ascending shot identifiers")
    assessments = [assess_normalised_beta_authority(artifact) for artifact in artifacts]
    report: dict[str, object] = {
        "schema_version": NORMALISED_BETA_AUTHORITY_SCHEMA,
        "spec": mast_normalised_beta_authority_spec(),
        "status": "blocked",
        "canonical_beta_n_binding_admissible": False,
        "shot_count": len(assessments),
        "shots": assessments,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Write an exclusive real-object L2F-12b authority assessment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--shot-id", type=int, action="append", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    shot_ids = sorted(set(args.shot_id))
    if len(shot_ids) != len(args.shot_id):
        parser.error("--shot-id values must be unique")
    manifest = load_verified_source_manifest(args.manifest, artifact_root=args.artifact_root)
    artifacts = [
        read_verified_npz_artifact(manifest, artifact_root=args.artifact_root, shot_id=shot_id) for shot_id in shot_ids
    ]
    report = build_normalised_beta_authority_report(artifacts)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.json_out.open("x", encoding="utf-8") as report_handle:
            json.dump(report, report_handle, indent=2, sort_keys=True)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
    except FileExistsError as exc:
        raise NormalisedBetaAuthorityError("refusing to overwrite an existing authority report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"blocked: {report['shot_count']} shot(s) -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
