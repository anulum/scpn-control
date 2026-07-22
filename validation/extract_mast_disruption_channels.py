#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption binding-readiness extraction gate
"""Inspect verified FAIR-MAST artefacts before physical channel extraction.

This gate closes the SourceObjectManifest v2 transport boundary without
inventing physical bindings. It opens only provenance-verified derived NPZ
artefacts, reports exact group-aware source keys, and distinguishes transport
resolution from the unit, dimension, timebase, sign, and reduction decisions
owned by the later MAST signal-binding specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.mast_disruption_signal_binding import assess_artifact_signal_bindings
from validation.mast_source_artifact_reader import (
    VerifiedSourceArtifact,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import canonical_json_sha256

BINDING_READINESS_SCHEMA = "scpn-control.mast-disruption-binding-readiness.v1"

TRANSPORT_RESOLVABLE_KEYS: dict[str, str] = {
    "time_s": "summary.time",
    "ip_amperes": "summary.ip",
    "ne_per_m3": "summary.line_average_n_e",
    "beta_normal": "equilibrium.beta_tor_normal",
    "q95_direct": "equilibrium.q95",
    "saddle_tesla": "magnetics.b_field_tor_probe_saddle_field",
}

PHYSICAL_BINDING_BLOCKERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "axis_z_m": (
        "magnetic_axis_z_binding_unspecified",
        ("equilibrium.z", "equilibrium.x_point_z"),
    ),
    "bt_t_tesla": ("direct_bt_or_sourced_tf_current_absent", ()),
    "poloidal_probe_tesla": (
        "poloidal_probe_channel_reduction_unspecified",
        ("magnetics.b_field_pol_probe_cc_field",),
    ),
    "saddle_angles_rad": (
        "saddle_angle_units_and_probe_alignment_unspecified",
        (
            "magnetics.b_field_tor_probe_saddle_l_phi",
            "magnetics.b_field_tor_probe_saddle_m_phi",
            "magnetics.b_field_tor_probe_saddle_u_phi",
        ),
    ),
}


def assess_artifact_binding_readiness(artifact: VerifiedSourceArtifact) -> dict[str, Any]:
    """Return deterministic transport and physical-binding readiness for one shot.

    Parameters
    ----------
    artifact:
        Provenance-verified source artefact with exact group-aware archive keys.

    Returns
    -------
    dict[str, Any]
        Digest-bound report. ``channel_extraction_admissible`` remains false
        until the versioned physical signal-binding contract is available.
    """
    available = set(artifact.archive_keys)
    signal_binding_assessment = assess_artifact_signal_bindings(artifact)
    resolved: list[dict[str, str]] = []
    unresolved: list[dict[str, Any]] = []
    for semantic, archive_key in sorted(TRANSPORT_RESOLVABLE_KEYS.items()):
        if archive_key in available:
            resolved.append(
                {
                    "semantic": semantic,
                    "archive_key": archive_key,
                    "status": "transport_resolved",
                }
            )
        else:
            unresolved.append(
                {
                    "semantic": semantic,
                    "status": "source_key_absent",
                    "reason_code": "required_group_aware_source_key_absent",
                    "required_source_keys": [archive_key],
                    "available_source_keys": [],
                }
            )
    for semantic, (reason_code, candidates) in sorted(PHYSICAL_BINDING_BLOCKERS.items()):
        unresolved.append(
            {
                "semantic": semantic,
                "status": "physical_binding_unresolved",
                "reason_code": reason_code,
                "required_source_keys": list(candidates),
                "available_source_keys": sorted(available.intersection(candidates)),
            }
        )
    report: dict[str, Any] = {
        "schema_version": BINDING_READINESS_SCHEMA,
        "status": "blocked",
        "readiness_scope": "transport_only",
        "channel_extraction_admissible": False,
        "shot_id": artifact.shot_id,
        "artifact_kind": artifact.artifact_kind,
        "local_path": artifact.local_path,
        "source_uri": artifact.source_uri,
        "manifest_sha256": artifact.manifest_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "parent_digest": artifact.parent_digest,
        "transform_digest": artifact.transform_digest,
        "archive_keys": list(artifact.archive_keys),
        "resolved": resolved,
        "unresolved": unresolved,
        "signal_binding_assessment": signal_binding_assessment,
        "blocking_contracts": [
            {
                "contract": "cross_group_timebase_alignment",
                "reason_code": "resampling_and_validity_mask_contract_not_yet_bound",
            },
            {
                "contract": "mast_signal_binding_spec",
                "reason_code": "binding_spec_contains_explicit_source_semantic_or_metadata_blockers",
            },
        ],
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def inspect_manifest_binding_readiness(
    manifest_path: Path,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Verify every acquired manifest artefact and build a readiness report.

    Parameters
    ----------
    manifest_path:
        SourceObjectManifest v2 JSON path.
    artifact_root:
        Root containing the manifest-declared derived NPZ files.

    Returns
    -------
    dict[str, Any]
        Deterministic campaign-level binding-readiness report.
    """
    manifest = load_verified_source_manifest(manifest_path, artifact_root=artifact_root)
    records: list[dict[str, Any]] = []
    for shot in sorted(manifest["shots"], key=lambda item: item["shot_id"]):
        shot_id = shot["shot_id"]
        if shot["status"] == "failed":
            records.append(
                {
                    "shot_id": shot_id,
                    "status": "not_acquired",
                    "reason": shot["error"],
                }
            )
            continue
        artifact = read_verified_npz_artifact(manifest, artifact_root=artifact_root, shot_id=shot_id)
        records.append(assess_artifact_binding_readiness(artifact))
    acquired = [record for record in records if record["status"] != "not_acquired"]
    report: dict[str, Any] = {
        "schema_version": BINDING_READINESS_SCHEMA,
        "status": "blocked",
        "readiness_scope": "transport_only",
        "channel_extraction_admissible": False,
        "manifest_sha256": manifest["payload_sha256"],
        "n_requested": len(records),
        "n_transport_verified": len(acquired),
        "n_not_acquired": len(records) - len(acquired),
        "shots": records,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="SourceObjectManifest v2 JSON path.")
    parser.add_argument("--artifact-root", type=Path, required=True, help="Root holding declared NPZ artefacts.")
    parser.add_argument("--json-out", type=Path, required=True, help="Binding-readiness report output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Verify acquired artefacts and write the fail-closed readiness report."""
    args = _parse_args(argv)
    report = inspect_manifest_binding_readiness(args.manifest, artifact_root=args.artifact_root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "binding readiness: "
        f"{report['n_transport_verified']} verified, {report['n_not_acquired']} unavailable "
        f"(status={report['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
