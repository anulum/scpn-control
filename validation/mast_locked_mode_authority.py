#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST locked-mode source-authority gate
"""Admit ``locked_mode_amp`` only from an attested stationary n=1 estimator.

The MAST literature identifies a growing, non-rotating n=1 radial perturbation
on the outer-midplane saddle array. It does not authorise SCPN-CONTROL's legacy
201-sample complex boxcar as the canonical estimator. This gate records the
candidate's physical time scale without executing it and fails closed until the
source lineage binds the component, probe location, frame, filtering,
background/pickup/vessel corrections, calibration, and uncertainty.
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

from validation.mast_saddle_modal_authority import (
    FIELD_KEY,
    GEOMETRY_KEYS,
    TIMEBASE_KEY,
    assess_saddle_modal_authority,
    mast_saddle_modal_authority_spec,
)
from validation.mast_source_artifact_reader import (
    VerifiedSourceArtifact,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256

LOCKED_MODE_AUTHORITY_SCHEMA = "scpn-control.mast-locked-mode-authority.v1.0.0"
LOCKED_MODE_AUTHORITY_VERSION = "1.0.0"
LOCKED_MODE_PAPER_URL = "https://doi.org/10.1088/0741-3335/56/10/104003"
LOCKED_MODE_PAPER_UKAEA_URL = "https://scientific-publications.ukaea.uk/wp-content/uploads/Published/Miss90.pdf"
LEGACY_WINDOW_SAMPLES = 201


class LockedModeAuthorityError(ValueError):
    """Raised when a locked-mode authority input or report is inconsistent."""


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_positive_finite_number(value: object) -> TypeGuard[int | float]:
    """Return whether ``value`` is a positive finite non-boolean number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def _nonempty_text(value: object) -> bool:
    """Return whether ``value`` is a non-empty text declaration."""
    return isinstance(value, str) and bool(value.strip())


def mast_locked_mode_authority_spec() -> dict[str, object]:
    """Return the deterministic, self-digested L2F-12d locked-mode contract."""
    saddle_spec = mast_saddle_modal_authority_spec()
    payload: dict[str, object] = {
        "schema_version": LOCKED_MODE_AUTHORITY_SCHEMA,
        "version": LOCKED_MODE_AUTHORITY_VERSION,
        "machine": "MAST",
        "canonical_channel": "locked_mode_amp",
        "canonical_units": "T",
        "source_field_key": FIELD_KEY,
        "source_timebase_key": TIMEBASE_KEY,
        "source_geometry_keys": list(GEOMETRY_KEYS),
        "saddle_modal_authority_spec_sha256": saddle_spec["payload_sha256"],
        "primary_source_observation": {
            "measured_component": "radial",
            "probe_location": "outer_midplane",
            "toroidal_order": 1,
            "stationarity": "formed locked with no rotating m=2,n=1 signature",
            "observable": "growing n=1 perturbation on the saddle array",
        },
        "legacy_candidate": {
            "formula": "abs(boxcar_window(A_1_complex(t), 201 samples))",
            "window_samples": LEGACY_WINDOW_SAMPLES,
            "status": "compatibility_only_not_source_authorised",
            "warning": "sample-count windows change physical meaning when the source sample rate changes",
        },
        "required_source_authority": [
            "complete saddle field-row and selected-geometry authority",
            "radial measured-field component and outer-midplane probe location",
            "stationary n=1 definition in the machine frame",
            "physical-frequency filter policy and positive low-pass cutoff",
            "filter edge-handling policy and cutoff below source Nyquist",
            "background and poloidal-field pickup correction policies",
            "vacuum-vessel response policy",
            "positive one-standard-deviation locked-mode uncertainty",
            "content digest for the estimator authority",
            "primary-source locked-mode citation",
        ],
        "citations": [LOCKED_MODE_PAPER_URL, LOCKED_MODE_PAPER_UKAEA_URL],
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _locked_mode_metadata_evidence(artifact: VerifiedSourceArtifact) -> tuple[list[str], dict[str, object]]:
    """Validate locked-mode-specific declarations on the saddle field source."""
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    field_metadata = artifact.metadata.get(FIELD_KEY)
    attributes: Mapping[str, object] = {}
    if not isinstance(field_metadata, Mapping):
        blockers.append("locked_mode_field_metadata_absent")
    else:
        source_attributes = field_metadata.get("source_attributes")
        if not isinstance(source_attributes, Mapping):
            blockers.append("locked_mode_field_source_attributes_absent")
        else:
            attributes = source_attributes

    declarations = {
        "measured_field_component": attributes.get("measured_field_component"),
        "probe_location": attributes.get("probe_location"),
        "locked_mode_estimator": attributes.get("locked_mode_estimator"),
        "locked_mode_toroidal_order": attributes.get("locked_mode_toroidal_order"),
        "locked_mode_stationary_frame": attributes.get("locked_mode_stationary_frame"),
        "locked_mode_low_pass_cutoff_hz": attributes.get("locked_mode_low_pass_cutoff_hz"),
        "locked_mode_filter_policy": attributes.get("locked_mode_filter_policy"),
        "locked_mode_edge_policy": attributes.get("locked_mode_edge_policy"),
        "locked_mode_background_policy": attributes.get("locked_mode_background_policy"),
        "locked_mode_pickup_correction_policy": attributes.get("locked_mode_pickup_correction_policy"),
        "locked_mode_vessel_response_policy": attributes.get("locked_mode_vessel_response_policy"),
        "locked_mode_standard_uncertainty_t": attributes.get("locked_mode_standard_uncertainty_t"),
        "locked_mode_estimator_evidence_sha256": attributes.get("locked_mode_estimator_evidence_sha256"),
        "locked_mode_authority_citation": attributes.get("locked_mode_authority_citation"),
    }
    evidence["source_declarations"] = declarations
    expected_values = (
        ("measured_field_component", "radial", "locked_mode_measured_component_not_attested"),
        ("probe_location", "outer_midplane", "locked_mode_probe_location_not_attested"),
        (
            "locked_mode_estimator",
            "stationary_n1_radial_field_perturbation",
            "locked_mode_estimator_not_attested",
        ),
        ("locked_mode_toroidal_order", 1, "locked_mode_toroidal_order_not_attested"),
        ("locked_mode_stationary_frame", "machine", "locked_mode_stationary_frame_not_attested"),
    )
    for key, expected, blocker in expected_values:
        if declarations[key] != expected:
            blockers.append(blocker)
    if not _is_positive_finite_number(declarations["locked_mode_low_pass_cutoff_hz"]):
        blockers.append("locked_mode_low_pass_cutoff_not_attested")
    for key, blocker in (
        ("locked_mode_filter_policy", "locked_mode_filter_policy_absent"),
        ("locked_mode_edge_policy", "locked_mode_edge_policy_absent"),
        ("locked_mode_background_policy", "locked_mode_background_policy_absent"),
        ("locked_mode_pickup_correction_policy", "locked_mode_pickup_correction_policy_absent"),
        ("locked_mode_vessel_response_policy", "locked_mode_vessel_response_policy_absent"),
    ):
        if not _nonempty_text(declarations[key]):
            blockers.append(blocker)
    if not _is_positive_finite_number(declarations["locked_mode_standard_uncertainty_t"]):
        blockers.append("locked_mode_standard_uncertainty_absent")
    if not _is_sha256(declarations["locked_mode_estimator_evidence_sha256"]):
        blockers.append("locked_mode_estimator_evidence_absent")
    if declarations["locked_mode_authority_citation"] != LOCKED_MODE_PAPER_URL:
        blockers.append("locked_mode_primary_source_citation_absent")
    return blockers, evidence


def _legacy_candidate_evidence(artifact: VerifiedSourceArtifact) -> tuple[list[str], dict[str, object]]:
    """Measure the legacy sample window's time scale without filtering a signal."""
    time_values = artifact.arrays.get(TIMEBASE_KEY)
    if time_values is None:
        return ["locked_mode_timebase_absent"], {}
    time = np.asarray(time_values, dtype=np.float64)
    evidence: dict[str, object] = {
        "legacy_window_samples": LEGACY_WINDOW_SAMPLES,
        "legacy_estimator_executed": False,
    }
    blockers: list[str] = []
    if time.ndim != 1 or time.size < 2:
        return ["locked_mode_timebase_shape_mismatch"], evidence
    differences = np.diff(time)
    if not bool(np.all(np.isfinite(time))) or not bool(np.all(differences > 0.0)):
        return ["locked_mode_timebase_not_finite_and_strictly_increasing"], evidence
    median_period = float(np.median(differences))
    sample_rate = 1.0 / median_period
    timebase_uniform = bool(np.allclose(differences, median_period, rtol=1.0e-9, atol=1.0e-12))
    evidence.update(
        {
            "source_samples": int(time.size),
            "sample_period_s": median_period,
            "sample_rate_hz": sample_rate,
            "timebase_uniform": timebase_uniform,
        }
    )
    if time.size < LEGACY_WINDOW_SAMPLES:
        blockers.append("locked_mode_legacy_window_exceeds_source_samples")
    else:
        evidence["legacy_window_endpoint_span_s"] = float(time[LEGACY_WINDOW_SAMPLES - 1] - time[0])
        evidence["legacy_window_support_s"] = median_period * LEGACY_WINDOW_SAMPLES
        evidence["legacy_boxcar_first_null_hz"] = sample_rate / LEGACY_WINDOW_SAMPLES if timebase_uniform else None
    return blockers, evidence


def assess_locked_mode_authority(artifact: VerifiedSourceArtifact) -> dict[str, object]:
    """Assess one verified source artefact without deriving locked-mode amplitude."""
    saddle_report = assess_saddle_modal_authority(artifact)
    metadata_blockers, metadata_evidence = _locked_mode_metadata_evidence(artifact)
    candidate_blockers, candidate_evidence = _legacy_candidate_evidence(artifact)
    saddle_blockers = cast(list[str], saddle_report["blockers"])
    blockers = list(dict.fromkeys((*saddle_blockers, *metadata_blockers, *candidate_blockers)))
    declarations = cast(dict[str, object], metadata_evidence["source_declarations"])
    cutoff = declarations["locked_mode_low_pass_cutoff_hz"]
    sample_rate = candidate_evidence.get("sample_rate_hz")
    if _is_positive_finite_number(cutoff) and isinstance(sample_rate, float) and cutoff >= 0.5 * sample_rate:
        blockers.append("locked_mode_low_pass_cutoff_not_below_nyquist")
    report: dict[str, object] = {
        "schema_version": LOCKED_MODE_AUTHORITY_SCHEMA,
        "spec_sha256": mast_locked_mode_authority_spec()["payload_sha256"],
        "shot_id": artifact.shot_id,
        "source_uri": artifact.source_uri,
        "artifact_sha256": artifact.artifact_sha256,
        "saddle_modal_authority_payload_sha256": saddle_report["payload_sha256"],
        "saddle_modal_authority_admissible": saddle_report["canonical_modal_bindings_admissible"],
        "metadata_evidence": metadata_evidence,
        "legacy_candidate_evidence": candidate_evidence,
        "locked_mode_estimator_executed": False,
        "canonical_locked_mode_binding_admissible": not blockers,
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


def build_locked_mode_authority_report(artifacts: Sequence[VerifiedSourceArtifact]) -> dict[str, object]:
    """Build one deterministic multi-shot locked-mode authority assessment."""
    if not artifacts:
        raise LockedModeAuthorityError("at least one verified source artefact is required")
    shot_ids = [artifact.shot_id for artifact in artifacts]
    if shot_ids != sorted(set(shot_ids)):
        raise LockedModeAuthorityError("source artefacts must have unique, ascending shot identifiers")
    assessments = [assess_locked_mode_authority(artifact) for artifact in artifacts]
    admissible = all(bool(item["canonical_locked_mode_binding_admissible"]) for item in assessments)
    report: dict[str, object] = {
        "schema_version": LOCKED_MODE_AUTHORITY_SCHEMA,
        "spec": mast_locked_mode_authority_spec(),
        "status": "authority_verified" if admissible else "blocked",
        "canonical_locked_mode_binding_admissible": admissible,
        "shot_count": len(assessments),
        "shots": assessments,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Write a real-object L2F-12d locked-mode authority assessment."""
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
    report = build_locked_mode_authority_report(artifacts)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.json_out.open("x", encoding="utf-8") as report_handle:
            json.dump(report, report_handle, indent=2, sort_keys=True)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
    except FileExistsError as exc:
        raise LockedModeAuthorityError("refusing to overwrite an existing authority report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"{report['status']}: {report['shot_count']} shot(s) -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
