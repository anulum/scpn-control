#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digest-bound FAIR-MAST real-object alignment gate
"""Run mask-preserving alignment on one digest-pinned external MAST object.

The gate reuses the L2F-04 verified real-object loader, invokes the production
L2F-10/L2F-11 binding and alignment path, and emits only lineage digests,
counts, channel proof digests, and explicit blocker reasons. Raw arrays remain
outside the repository and are never serialised into the evidence report.

An ``alignment_blocked`` result is a successful fail-closed proof when the
legacy selected-array snapshot cannot attest the required source metadata. It
does not establish scientific validity, facility validation, prediction
quality, or control admission.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from validation.extract_mast_disruption_channels import assess_artifact_binding_readiness
from validation.mast_source_artifact_reader import (
    SourceArtifactReaderError,
    VerifiedSourceArtifact,
    load_pinned_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256
from validation.verify_mast_real_object_smoke import RealObjectSmokeError, read_pinned_real_object

REAL_OBJECT_ALIGNMENT_SCHEMA = "scpn-control.mast-real-object-alignment.v1.0.0"
ManifestFormat = Literal["legacy-material-v1", "source-object-v2"]


class RealObjectAlignmentError(ValueError):
    """Raised when a pinned external object cannot yield bounded alignment evidence."""


@dataclass(frozen=True)
class _AlignmentInput:
    artifact: VerifiedSourceArtifact
    input_manifest_sha256: str
    source_manifest_sha256: str


def _read_alignment_input(
    manifest_path: Path,
    *,
    manifest_format: ManifestFormat,
    artifact_root: Path,
    shot_id: int,
    expected_manifest_sha256: str,
    expected_artifact_sha256: str,
) -> _AlignmentInput:
    if manifest_format == "legacy-material-v1":
        try:
            pinned = read_pinned_real_object(
                manifest_path,
                artifact_root=artifact_root,
                shot_id=shot_id,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_artifact_sha256=expected_artifact_sha256,
            )
        except RealObjectSmokeError as exc:
            raise RealObjectAlignmentError(f"pinned legacy real-object verification failed: {exc}") from exc
        return _AlignmentInput(
            artifact=pinned.artifact,
            input_manifest_sha256=pinned.legacy_manifest_sha256,
            source_manifest_sha256=pinned.migrated_manifest_sha256,
        )
    if manifest_format != "source-object-v2":
        raise RealObjectAlignmentError(f"unsupported manifest format {manifest_format!r}")
    try:
        manifest, input_sha256 = load_pinned_source_manifest(
            manifest_path,
            artifact_root=artifact_root,
            expected_sha256=expected_manifest_sha256,
        )
        artifact = read_verified_npz_artifact(manifest, artifact_root=artifact_root, shot_id=shot_id)
    except SourceArtifactReaderError as exc:
        raise RealObjectAlignmentError(f"pinned v2 real-object verification failed: {exc}") from exc
    if artifact.artifact_sha256 != expected_artifact_sha256:
        raise RealObjectAlignmentError("selected NPZ SHA-256 does not match the pinned digest")
    return _AlignmentInput(
        artifact=artifact,
        input_manifest_sha256=input_sha256,
        source_manifest_sha256=manifest["payload_sha256"],
    )


def _channel_proofs(alignment: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project alignment details into raw-data-free per-channel proofs."""
    channels = alignment.get("channels")
    if not isinstance(channels, list):
        return []
    proofs: list[dict[str, Any]] = []
    for raw in cast(list[Mapping[str, Any]], channels):
        proof: dict[str, Any] = {
            "channel": raw.get("channel"),
            "status": raw.get("status"),
        }
        for key in (
            "reason_code",
            "binding_assessment_status",
            "source_key",
            "timebase_key",
            "method",
            "finite_source_samples",
            "valid_target_samples",
            "total_target_samples",
            "values_sha256",
            "valid_mask_sha256",
        ):
            if key in raw:
                proof[key] = raw[key]
        proofs.append(proof)
    return proofs


def verify_real_object_alignment(
    manifest_path: Path,
    *,
    manifest_format: ManifestFormat = "legacy-material-v1",
    artifact_root: Path,
    shot_id: int,
    expected_manifest_sha256: str,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    """Verify and align one exact external MAST selected-array snapshot.

    Parameters
    ----------
    manifest_path:
        Legacy material manifest or SourceObjectManifest v2 stored beside the
        external NPZ objects.
    manifest_format:
        Explicit input contract; format guessing is forbidden.
    artifact_root:
        External root beneath which all manifest paths must resolve.
    shot_id:
        Positive shot identifier selected for the proof.
    expected_manifest_sha256:
        Exact SHA-256 digest of the legacy manifest bytes.
    expected_artifact_sha256:
        Exact SHA-256 digest of the selected NPZ bytes.

    Returns
    -------
    dict[str, Any]
        Self-digested bounded alignment report without raw arrays.

    Raises
    ------
    RealObjectAlignmentError
        If the external object fails identity, integrity, or report-shape
        verification.
    """
    pinned = _read_alignment_input(
        manifest_path,
        manifest_format=manifest_format,
        artifact_root=artifact_root,
        shot_id=shot_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    readiness = assess_artifact_binding_readiness(pinned.artifact)
    alignment = cast(Mapping[str, Any], readiness["time_alignment_assessment"])
    alignment_complete = alignment.get("bound_scalar_alignment_complete") is True
    full_extraction = alignment.get("full_canonical_extraction_admissible") is True
    report: dict[str, Any] = {
        "schema_version": REAL_OBJECT_ALIGNMENT_SCHEMA,
        "status": "bound_scalar_alignment_verified" if alignment_complete else "alignment_blocked",
        "transport_interoperability_verified": True,
        "bound_scalar_alignment_complete": alignment_complete,
        "full_canonical_extraction_admissible": full_extraction,
        "scientific_validity_claim_admissible": False,
        "facility_claim_admissible": False,
        "control_admission_admissible": False,
        "shot_id": pinned.artifact.shot_id,
        "input_manifest_format": manifest_format,
        "input_manifest_sha256": pinned.input_manifest_sha256,
        "source_manifest_sha256": pinned.source_manifest_sha256,
        "artifact_sha256": pinned.artifact.artifact_sha256,
        "parent_digest": pinned.artifact.parent_digest,
        "transform_digest": pinned.artifact.transform_digest,
        "binding_readiness_sha256": readiness["payload_sha256"],
        "alignment_report_sha256": alignment.get("payload_sha256"),
        "binding_spec_sha256": alignment.get("binding_spec_sha256"),
        "alignment_spec_sha256": alignment.get("alignment_spec_sha256"),
        "target_time_sha256": alignment.get("target_time_sha256"),
        "target_samples": alignment.get("target_samples"),
        "n_bound_channels_aligned": alignment.get("n_bound_channels_aligned", 0),
        "n_channels_not_aligned": alignment.get("n_channels_not_aligned", 11),
        "reason_code": alignment.get("reason_code"),
        "channels": _channel_proofs(alignment),
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest path.")
    parser.add_argument(
        "--manifest-format",
        choices=("legacy-material-v1", "source-object-v2"),
        default="legacy-material-v1",
        help="Explicit manifest contract; defaults to the recovered legacy campaign format.",
    )
    parser.add_argument("--artifact-root", type=Path, required=True, help="External NPZ artefact root.")
    parser.add_argument("--shot-id", type=int, required=True, help="Shot selected for alignment proof.")
    parser.add_argument("--expected-manifest-sha256", required=True, help="Pinned legacy manifest SHA-256.")
    parser.add_argument("--expected-artifact-sha256", required=True, help="Pinned selected NPZ SHA-256.")
    parser.add_argument("--json-out", type=Path, required=True, help="Destination for the alignment report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one bounded real-object alignment proof and write its report."""
    args = _parse_args(argv)
    report = verify_real_object_alignment(
        args.manifest,
        manifest_format=cast(ManifestFormat, args.manifest_format),
        artifact_root=args.artifact_root,
        shot_id=args.shot_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_artifact_sha256=args.expected_artifact_sha256,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "MAST real-object alignment: "
        f"shot {report['shot_id']} status={report['status']} "
        f"aligned={report['n_bound_channels_aligned']} "
        f"blocked={report['n_channels_not_aligned']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
