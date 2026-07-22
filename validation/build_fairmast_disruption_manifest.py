#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption acquisition-campaign builder
"""Select FAIR-MAST shots and emit a disruption acquisition campaign spec.

Given the public FAIR-MAST CPF summary rows (exported from
``mast_cpf_data.parquet``), this tool applies the documented quality filter —
keep ``useful == 1`` shots that were not aborted and whose ``ip_max`` clears a
floor — and emits a schema-versioned acquisition campaign spec: the selected
shots with their expected level2 S3 keys, the embedded Ip current-quench label
algorithm, the exclusions applied, and the S3 access parameters.

The spec is the out-of-band acquisition plan; it carries no checksums because the
disruptive/safe split and per-file digests are only known after the Ip signals
are acquired and the quench detector runs (the dataset builder does that). The
status stays ``"planned"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validation.audit_mast_disruption_feature_sources import LABEL_ALGORITHM
from validation.fair_mast_source_policy import fair_mast_provenance

CAMPAIGN_SCHEMA = "scpn-control.mast-disruption-acquisition-campaign.v1"
S3_BUCKET = "s3://mast"
S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
S3_ACCESS = "--no-sign-request"
S3_LEVEL2_PREFIX = "level2/shots"

_REQUIRED_CPF_KEYS: tuple[str, ...] = ("shot_id", "useful", "abort", "ip_max_ka")


def _sha256_json(payload: dict[str, Any]) -> str:
    """Canonical SHA-256 of a JSON payload (sorted keys, no whitespace)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_shots(cpf_rows: list[dict[str, Any]], *, ip_max_floor_ka: float) -> dict[str, list[dict[str, Any]]]:
    """Apply the CPF quality filter to CPF summary rows.

    A shot is selected when ``useful == 1``, ``abort != 1`` and ``ip_max_ka`` is at
    least ``ip_max_floor_ka``; otherwise it is excluded with its failing reasons.
    The disruptive/safe split is deferred to the dataset builder's Ip current-quench
    detector, so every selected shot is an acquisition candidate.
    """
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in cpf_rows:
        missing = [key for key in _REQUIRED_CPF_KEYS if key not in row]
        if missing:
            raise ValueError(f"CPF row is missing keys {missing}: {row}")
        shot_id = int(row["shot_id"])
        useful = int(row["useful"])
        abort = int(row["abort"])
        ip_max_ka = float(row["ip_max_ka"])
        reasons: list[str] = []
        if useful != 1:
            reasons.append("not_useful")
        if abort == 1:
            reasons.append("aborted")
        if ip_max_ka < ip_max_floor_ka:
            reasons.append("ip_max_below_floor")
        if reasons:
            excluded.append({"shot_id": shot_id, "reasons": reasons})
        else:
            selected.append(
                {
                    "shot_id": shot_id,
                    "ip_max_ka": ip_max_ka,
                    "s3_key": f"{S3_LEVEL2_PREFIX}/{shot_id}",
                }
            )
    return {"selected": selected, "excluded": excluded}


def build_campaign(
    cpf_rows: list[dict[str, Any]],
    *,
    dataset_id: str,
    ip_max_floor_ka: float,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the schema-versioned acquisition campaign spec."""
    split = select_shots(cpf_rows, ip_max_floor_ka=ip_max_floor_ka)
    label_algorithm = dict(LABEL_ALGORITHM)
    label_algorithm["ip_max_floor_ka"] = ip_max_floor_ka
    report: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "planned",
        "dataset_id": dataset_id,
        "machine": "MAST",
        "synthetic": False,
        "data_access": {
            "bucket": S3_BUCKET,
            "endpoint": S3_ENDPOINT,
            "access": S3_ACCESS,
            "level2_prefix": S3_LEVEL2_PREFIX,
            "signal_format": "zarr_v3",
            **fair_mast_provenance(),
        },
        "selection_filter": {
            "useful": 1,
            "abort": 0,
            "ip_max_floor_ka": ip_max_floor_ka,
        },
        "label_algorithm": label_algorithm,
        "n_candidates": len(cpf_rows),
        "n_selected": len(split["selected"]),
        "n_excluded": len(split["excluded"]),
        "selected_shots": split["selected"],
        "excluded_shots": split["excluded"],
        "acquisition_note": (
            "acquire each selected shot's level2 signals out-of-band with "
            f"'{S3_ACCESS} --endpoint-url {S3_ENDPOINT} {S3_BUCKET}/{S3_LEVEL2_PREFIX}/<shot>'; "
            "the dataset builder derives labels and per-file checksums post-acquisition."
        ),
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary of the campaign spec."""
    lines = [
        "# FAIR-MAST Disruption Acquisition Campaign",
        "",
        f"- **Status**: {report['status']} (synthetic={report['synthetic']})",
        f"- **Dataset**: {report['dataset_id']}",
        f"- **Selected**: {report['n_selected']} of {report['n_candidates']} candidates "
        f"({report['n_excluded']} excluded)",
        f"- **Filter**: useful=1, abort=0, ip_max ≥ {report['selection_filter']['ip_max_floor_ka']} kA",
        f"- **Label method**: {report['label_algorithm']['method']}",
        "",
        f"> {report['acquisition_note']}",
    ]
    return "\n".join(lines) + "\n"


def _load_cpf_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("CPF JSON must be a list of row objects.")
    return payload


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpf-json", type=Path, required=True, help="JSON list of CPF summary rows.")
    parser.add_argument("--dataset-id", type=str, required=True, help="Campaign dataset identifier.")
    parser.add_argument("--json-out", type=Path, required=True, help="Campaign JSON output path.")
    parser.add_argument("--report-out", type=Path, required=True, help="Campaign Markdown output path.")
    parser.add_argument("--ip-max-floor-ka", type=float, default=100.0, help="CPF ip_max floor (kA).")
    parser.add_argument("--generated-at", type=str, default="", help="Fixed UTC timestamp label.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: select shots and write the acquisition campaign spec."""
    args = _parse_args(argv)
    cpf_rows = _load_cpf_rows(args.cpf_json)
    report = build_campaign(
        cpf_rows,
        dataset_id=args.dataset_id,
        ip_max_floor_ka=args.ip_max_floor_ka,
        generated_at=args.generated_at,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"campaign: {report['n_selected']}/{report['n_candidates']} selected (status={report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
