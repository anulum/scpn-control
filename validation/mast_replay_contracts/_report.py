# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST candidate report and authority-boundary declarations.
"""Build local candidate metadata without authenticating source material."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from validation.mast_dbdt_authority import GEOMETRY_KEYS as DBDT_GEOMETRY_KEYS
from validation.mast_dbdt_authority import mast_dbdt_authority_spec
from validation.mast_disruption_shot_label import SHOT_LABEL_RECORD_SCHEMA
from validation.mast_locked_mode_authority import mast_locked_mode_authority_spec
from validation.mast_replay_contracts._inputs import MEASURED_CHANNELS
from validation.mast_saddle_modal_authority import GEOMETRY_KEYS, mast_saddle_modal_authority_spec

REPORT_SCHEMA = "scpn-control.mast-disruption-replay-channels.v2.0.0"
DATASET_SCHEMA = "scpn-control.mast-disruption-supervised-dataset.v2.0.0"


def _sha256_json(payload: dict[str, Any]) -> str:
    """Hash the historical sorted compact JSON body without authenticating its origin."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def replay_report(
    archive_binding: dict[str, Any],
    records: list[dict[str, Any]],
    shot_ids: list[int],
    *,
    material_name: str,
    archive_name: str,
    generated_at: str,
    locked_window: int,
) -> dict[str, Any]:
    """Assemble unchanged v2 metadata with every physical admission held false.

    Inputs are caller observations, not independently validated provenance.
    Hashing checks body consistency; authority specs declare required evidence,
    not its presence. This pure function writes nothing and returns fresh
    report containers (the supplied binding/records remain referenced).
    """
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "synthetic": False,
        "material_dir": material_name,
        "channels_npz": archive_name,
        "channels_archive": archive_binding,
        "channel_schema": list(MEASURED_CHANNELS),
        "channel_authority": {
            "BT_T": {
                "source_key": "equilibrium.bphi_rmag",
                "reference_radius_key": "equilibrium.magnetic_axis_r",
                "canonical_binding_admissible": False,
                "blocker": "toroidal_field_authority_incomplete",
            },
            "beta_N": {
                "source_key": "equilibrium.beta_tor_normal",
                "canonical_binding_admissible": False,
                "blocker": "normalised_beta_authority_incomplete",
            },
            "n1_amp": {
                "source_key": "magnetics.b_field_tor_probe_saddle_field",
                "geometry_keys": list(GEOMETRY_KEYS),
                "authority_spec_sha256": mast_saddle_modal_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "saddle_modal_authority_incomplete",
            },
            "n2_amp": {
                "source_key": "magnetics.b_field_tor_probe_saddle_field",
                "geometry_keys": list(GEOMETRY_KEYS),
                "authority_spec_sha256": mast_saddle_modal_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "saddle_modal_authority_incomplete",
            },
            "locked_mode_amp": {
                "source_key": "magnetics.b_field_tor_probe_saddle_field",
                "geometry_keys": list(GEOMETRY_KEYS),
                "authority_spec_sha256": mast_locked_mode_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "locked_mode_authority_incomplete",
            },
            "dBdt_gauss_per_s": {
                "source_key": "magnetics.b_field_pol_probe_cc_field",
                "geometry_keys": list(DBDT_GEOMETRY_KEYS),
                "authority_spec_sha256": mast_dbdt_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "dbdt_authority_incomplete",
            },
        },
        "claim_boundary": {
            "scientific_validation": False,
            "training_admission": False,
            "facility_prediction": False,
            "control_admission": False,
        },
        "locked_window": locked_window,
        "n_derived": len(shot_ids),
        "shots": records,
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def dataset_report(
    records: list[dict[str, Any]],
    *,
    dataset_id: str,
    manifest_name: str,
    dataset_fingerprint: str,
    label_algorithm: dict[str, object],
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the retained v2 proxy-label report with admission blocked.

    Records, fingerprints, labels and timestamps are supplied observations.
    No arrays, files or source identities are read here. The fresh report
    references the supplied records/algorithm; its self-digest only checks
    body consistency. All labels are input-derived, not independent outcomes.
    """
    report: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA,
        "status": "blocked",
        "admission_ready": False,
        "blocked_reason": (
            "all labels have ip_proxy authority derived from an input feature; "
            "they are uncalibrated and are not independent facility ground truth"
        ),
        "dataset_id": dataset_id,
        "synthetic": False,
        "manifest": manifest_name,
        "dataset_sha256": dataset_fingerprint,
        "n_shots": len(records),
        "n_disruptive": sum(r["label"] for r in records),
        "n_ambiguous": sum(r["label_record"]["outcome"] == "ambiguous" for r in records),
        "channel_schema": [*MEASURED_CHANNELS, "is_disruption", "disruption_time_idx", "disruption_type"],
        "metadata_schema": {"shot_label_record_json": SHOT_LABEL_RECORD_SCHEMA},
        "label_authority_counts": {"ip_proxy": len(records)},
        "independent_label_count": 0,
        "label_algorithm": label_algorithm,
        "shots": records,
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report
