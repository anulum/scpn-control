#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption feature-source and label-algorithm audit
"""Audit the FAIR-MAST signal provenance for a disruption dataset.

The disruption dataset builder must fill the ``run_real_shot_replay`` channel
schema from public FAIR-MAST level2 signals, and it must derive disruption labels
(the ``level2/defuse`` DEFUSE labels are listed but return HTTP 403). This module
is the committed, schema-versioned contract for both: for every NPZ channel it
records the source level2 signal group, units, transform and readiness status
(``source_ready`` / ``derived`` / ``lookup_needed``), and it documents the exact
Ip current-quench labelling algorithm and its parameters.

It is a policy audit, not a live Zarr probe: the heavy MAST stores are acquired
out-of-band, so the mapping and label algorithm are declared and self-checked
here, then implemented by the manifest and dataset builders. The overall
``status`` is ``blocked`` while any channel is still ``lookup_needed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "scpn-control.mast-disruption-feature-source-audit.v1"

# The ``run_real_shot_replay`` channel schema the dataset builder must fill.
NPZ_CHANNELS: tuple[str, ...] = (
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
    "is_disruption",
    "disruption_time_idx",
    "disruption_type",
)

# Readiness of each channel against the public FAIR-MAST level2 catalogue
# (docs/internal/fairmast_level2_inventory_2026-07-07.md, sample shot 11766):
#   source_ready  — a catalogued level2 signal maps directly (unit transform only)
#   derived       — recipe from catalogued signals is known but not yet implemented
#   lookup_needed — no catalogued signal located yet; acquisition must resolve it
FEATURE_SOURCE_POLICY: dict[str, dict[str, str]] = {
    "time_s": {
        "level2_source": "summary/time",
        "units": "s",
        "transform": "identity",
        "status": "source_ready",
    },
    "Ip_MA": {
        "level2_source": "magnetics/ip",
        "units": "MA",
        "transform": "amperes_to_megamperes",
        "status": "source_ready",
    },
    "BT_T": {
        "level2_source": "toroidal_field (not in the sampled level2 catalogue)",
        "units": "T",
        "transform": "vacuum_field_from_tf_current_or_direct_signal",
        "status": "lookup_needed",
    },
    "beta_N": {
        "level2_source": "equilibrium/beta_normal",
        "units": "dimensionless",
        "transform": "identity",
        "status": "source_ready",
    },
    "q95": {
        "level2_source": "equilibrium EFIT q-profile (cpf q95_min scalar available)",
        "units": "dimensionless",
        "transform": "q_at_psi_norm_0p95",
        "status": "derived",
    },
    "ne_1e19": {
        "level2_source": "summary/line_average_n_e",
        "units": "1e19 m^-3",
        "transform": "per_1e19",
        "status": "source_ready",
    },
    "n1_amp": {
        "level2_source": "magnetics/saddle (12-channel toroidal array)",
        "units": "T",
        "transform": "n1_toroidal_mode_decomposition",
        "status": "derived",
    },
    "n2_amp": {
        "level2_source": "magnetics/saddle (12-channel toroidal array)",
        "units": "T",
        "transform": "n2_toroidal_mode_decomposition",
        "status": "derived",
    },
    "locked_mode_amp": {
        "level2_source": "magnetics/saddle and poloidal probe arrays",
        "units": "T",
        "transform": "locked_mode_envelope",
        "status": "derived",
    },
    "dBdt_gauss_per_s": {
        "level2_source": "magnetics poloidal probe",
        "units": "G/s",
        "transform": "time_derivative_tesla_to_gauss",
        "status": "derived",
    },
    "vertical_position_m": {
        "level2_source": "equilibrium magnetic-axis / centroid Z (x_point_z available)",
        "units": "m",
        "transform": "identity",
        "status": "derived",
    },
    "is_disruption": {
        "level2_source": "derived from magnetics/ip",
        "units": "boolean",
        "transform": "ip_current_quench_label",
        "status": "derived",
    },
    "disruption_time_idx": {
        "level2_source": "derived from magnetics/ip",
        "units": "sample index",
        "transform": "ip_current_quench_onset",
        "status": "derived",
    },
    "disruption_type": {
        "level2_source": "derived from magnetics/ip and mode activity",
        "units": "category",
        "transform": "ip_current_quench_class",
        "status": "derived",
    },
}

# The documented Ip current-quench labelling algorithm (DEFUSE labels are 403).
LABEL_ALGORITHM: dict[str, Any] = {
    "method": "ip_current_quench",
    "signal": "magnetics/ip",
    "flat_top_reference": "pre-quench Ip flat-top maximum",
    "quench_criterion": (
        "terminal current quench where |Ip| falls below (1 - drop_fraction) of the "
        "flat-top maximum within quench_window_ms"
    ),
    "drop_fraction": 0.8,
    "quench_window_ms": 5.0,
    "disruption_time": "onset sample of the terminal current quench",
    "exclusions": [
        "cpf abort == 1",
        "cpf useful != 1",
        "cpf ip_max below a declared floor",
    ],
    "validation_requirement": (
        "the detector must be cross-checked on >= 50 manually inspected shots before the derived labels are trusted"
    ),
    "defuse_cross_check": (
        "level2/defuse HDF5 labels return HTTP 403; labels are derived only, with "
        "DEFUSE reserved as a future cross-check if access is granted"
    ),
}

_LABEL_REQUIRED_KEYS: tuple[str, ...] = (
    "method",
    "signal",
    "quench_criterion",
    "drop_fraction",
    "quench_window_ms",
    "disruption_time",
    "exclusions",
    "validation_requirement",
)


def _sha256_json(payload: dict[str, Any]) -> str:
    """Canonical SHA-256 of a JSON payload (sorted keys, no whitespace)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_feature_source_audit(*, ip_max_floor_ka: float = 100.0) -> dict[str, Any]:
    """Assemble the schema-versioned feature-source and label-algorithm audit.

    ``ip_max_floor_ka`` is the CPF ``ip_max`` exclusion floor recorded with the
    label algorithm. The overall ``status`` is ``blocked`` while any channel is
    still ``lookup_needed``.
    """
    missing_policy = [channel for channel in NPZ_CHANNELS if channel not in FEATURE_SOURCE_POLICY]
    if missing_policy:
        raise ValueError(f"feature-source policy is missing channels: {missing_policy}")
    missing_label = [key for key in _LABEL_REQUIRED_KEYS if key not in LABEL_ALGORITHM]
    if missing_label:
        raise ValueError(f"label algorithm is missing keys: {missing_label}")

    channels = {name: dict(FEATURE_SOURCE_POLICY[name]) for name in NPZ_CHANNELS}
    status_counts = {"source_ready": 0, "derived": 0, "lookup_needed": 0}
    for entry in channels.values():
        status_counts[entry["status"]] += 1
    overall_status = "blocked" if status_counts["lookup_needed"] > 0 else "source_ready"

    label_algorithm = dict(LABEL_ALGORITHM)
    label_algorithm["ip_max_floor_ka"] = ip_max_floor_ka

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": overall_status,
        "npz_channel_schema": list(NPZ_CHANNELS),
        "channel_status_counts": status_counts,
        "channels": channels,
        "label_algorithm": label_algorithm,
        "data_access": {
            "bucket": "s3://mast",
            "endpoint": "https://s3.echo.stfc.ac.uk",
            "access": "--no-sign-request",
            "signal_format": "zarr_v3_level2",
            "catalogue": "mast-level2-signals.parquet",
            "cpf_catalogue": "mast_cpf_data.parquet",
            "licence": "MIT",
            "citation": "Jackson et al., SoftwareX 27 (2024) 101869, DOI 10.1016/j.softx.2024.101869",
        },
        "blocked_reason": (
            "channels are still lookup_needed; acquisition must locate their level2 "
            "signals before the dataset builder can fill the schema"
            if overall_status == "blocked"
            else "all channels have a resolved level2 source or derivation recipe"
        ),
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary of the audit."""
    counts = report["channel_status_counts"]
    lines = [
        "# FAIR-MAST Disruption Feature-Source Audit",
        "",
        f"- **Status**: {report['status']}",
        f"- **Channels**: {len(report['npz_channel_schema'])} "
        f"(source_ready={counts['source_ready']}, derived={counts['derived']}, "
        f"lookup_needed={counts['lookup_needed']})",
        f"- **Label method**: {report['label_algorithm']['method']} "
        f"(drop_fraction={report['label_algorithm']['drop_fraction']}, "
        f"window={report['label_algorithm']['quench_window_ms']} ms)",
        "",
        "## Channel provenance",
        "",
        "| Channel | Level2 source | Units | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name, entry in report["channels"].items():
        lines.append(f"| {name} | {entry['level2_source']} | {entry['units']} | {entry['status']} |")
    lines.extend(
        [
            "",
            f"> {report['blocked_reason']}",
            "",
            f"DEFUSE note: {report['label_algorithm']['defuse_cross_check']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--report-out", type=Path, required=True, help="Report Markdown output path.")
    parser.add_argument(
        "--ip-max-floor-ka",
        type=float,
        default=100.0,
        help="CPF ip_max exclusion floor (kA) recorded with the label algorithm.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build and write the feature-source audit report."""
    args = _parse_args(argv)
    report = build_feature_source_audit(ip_max_floor_ka=args.ip_max_floor_ka)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"feature-source audit: status={report['status']} ({args.json_out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
