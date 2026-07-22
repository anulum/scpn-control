#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption dataset builder (labelled NPZ + manifest)
"""Assemble a labelled FAIR-MAST disruption dataset from extracted channels.

Given per-shot channel arrays (the eleven measured ``run_real_shot_replay``
channels, extracted from acquired level2 signals out-of-band), this builder
derives the disruption labels with the documented Ip current-quench detector,
writes each shot as an ``.npz`` in the full replay schema, checksums every file,
and emits a ``synthetic:false`` :class:`RealDataManifest` plus a schema-versioned
dataset report.

The mapping from raw level2 Zarr variables to the extracted channels — including
the derived-channel recipes (n-mode decomposition, EFIT q95, toroidal field) — is
the out-of-band acquisition step documented by the feature-source audit; this
module owns the labelling, assembly, checksums and provenance. The dataset report
stays ``status:"blocked"`` (bounded labels, not facility-validated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.real_data_manifest import validate_real_data_manifest
from validation.audit_mast_disruption_feature_sources import LABEL_ALGORITHM
from validation.fair_mast_source_policy import fair_mast_provenance

DATASET_SCHEMA = "scpn-control.mast-disruption-supervised-dataset.v1"

# The eleven measured channels the acquisition must supply per shot; the three
# label channels (is_disruption, disruption_time_idx, disruption_type) are derived
# here from Ip and appended to the written NPZ.
MEASURED_CHANNELS: tuple[str, ...] = (
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
)
CHANNEL_UNITS: dict[str, str] = {
    "time_s": "s",
    "Ip_MA": "MA",
    "BT_T": "T",
    "beta_N": "dimensionless",
    "q95": "dimensionless",
    "ne_1e19": "1e19 m^-3",
    "n1_amp": "T",
    "n2_amp": "T",
    "locked_mode_amp": "T",
    "dBdt_gauss_per_s": "G/s",
    "vertical_position_m": "m",
}
# Near-flat-top fraction used to locate the quench onset before the collapse.
_FLAT_TOP_FRACTION = 0.8


def _sha256_json(payload: dict[str, Any]) -> str:
    """Canonical SHA-256 of a JSON payload (sorted keys, no whitespace)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_ip_quench_label(
    ip: NDArray[np.float64],
    time_s: NDArray[np.float64],
    *,
    drop_fraction: float = 0.8,
    quench_window_ms: float = 5.0,
) -> tuple[bool, int, str]:
    """Derive a disruption label from a plasma-current trace.

    A shot is disruptive when, after the last near-flat-top sample, the current
    terminally collapses below ``(1 - drop_fraction)`` of its flat-top maximum
    within ``quench_window_ms``. Anchoring on the last near-flat-top sample means
    the initial current ramp (also below the collapse threshold) is ignored.
    Returns ``(is_disruption, onset_index, disruption_type)`` with ``onset_index``
    ``-1`` for a non-disruptive shot, implementing the algorithm documented by the
    feature-source audit's ``LABEL_ALGORITHM``.
    """
    ip_abs = np.abs(np.asarray(ip, dtype=np.float64))
    ip_max = float(ip_abs.max()) if ip_abs.size else 0.0
    if ip_max <= 0.0:
        return (False, -1, "no_current")
    low_threshold = (1.0 - drop_fraction) * ip_max
    # The maximum sample always clears the flat-top band, so an onset exists.
    onset = int(np.nonzero(ip_abs >= _FLAT_TOP_FRACTION * ip_max)[0][-1])
    collapsed = np.nonzero(ip_abs[onset:] < low_threshold)[0]
    if collapsed.size == 0:
        return (False, -1, "no_quench")
    first_collapsed = onset + int(collapsed[0])
    quench_ms = float((time_s[first_collapsed] - time_s[onset]) * 1000.0)
    if quench_ms <= quench_window_ms:
        return (True, onset, "current_quench")
    return (False, -1, "slow_rampdown")


def _validate_channels(shot_id: int, channels: dict[str, NDArray[np.float64]]) -> int:
    missing = [c for c in MEASURED_CHANNELS if c not in channels]
    if missing:
        raise ValueError(f"shot {shot_id}: missing measured channels {missing}.")
    n_samples = int(np.asarray(channels["time_s"]).shape[0])
    for name in MEASURED_CHANNELS:
        array = np.asarray(channels[name], dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != n_samples:
            raise ValueError(f"shot {shot_id}: channel {name!r} must be 1-D with {n_samples} samples.")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"shot {shot_id}: channel {name!r} must be finite.")
    return n_samples


def build_shot_npz(
    shot_id: int,
    channels: dict[str, NDArray[np.float64]],
    *,
    out_dir: Path,
    drop_fraction: float,
    quench_window_ms: float,
) -> dict[str, Any]:
    """Label one shot, write its ``.npz``, and return a checksummed record."""
    _validate_channels(shot_id, channels)
    ip = np.asarray(channels["Ip_MA"], dtype=np.float64)
    time_s = np.asarray(channels["time_s"], dtype=np.float64)
    is_disruption, onset, disruption_type = derive_ip_quench_label(
        ip, time_s, drop_fraction=drop_fraction, quench_window_ms=quench_window_ms
    )
    payload = {name: np.asarray(channels[name], dtype=np.float64) for name in MEASURED_CHANNELS}
    payload["is_disruption"] = np.asarray(is_disruption)
    payload["disruption_time_idx"] = np.asarray(onset)
    payload["disruption_type"] = np.asarray(disruption_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"shot_{shot_id}.npz"
    np.savez(npz_path, **payload)  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool
    return {
        "shot_id": shot_id,
        "npz": npz_path.name,
        "checksum_sha256": _sha256_file(npz_path),
        "label": 1 if is_disruption else 0,
        "disruption_time_idx": onset,
        "disruption_type": disruption_type,
        "n_samples": int(time_s.shape[0]),
    }


def _build_manifest(
    records: list[dict[str, Any]],
    *,
    dataset_id: str,
    retrieved_at: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "machine": "MAST",
        "shot": f"campaign:{dataset_id}",
        "synthetic": False,
        "source": {
            "kind": "local_archive",
            "uri": "s3://mast/level2/shots",
            "access": "s3_no_sign_request",
        },
        "signals": [
            {"name": name, "path": name, "units": CHANNEL_UNITS[name], "timebase": "time_s"}
            for name in MEASURED_CHANNELS
        ],
        "retrieved_at": retrieved_at,
        "checksum_sha256": None,
        **fair_mast_provenance(),
        "synthetic_generator": None,
        "synthetic_seed": None,
        "artifacts": [{"uri": r["npz"], "checksum_sha256": r["checksum_sha256"]} for r in records],
    }
    # Fail closed: the manifest must satisfy the real-data provenance contract.
    validate_real_data_manifest(manifest)
    return manifest


def build_dataset(
    shots: list[dict[str, Any]],
    *,
    dataset_id: str,
    out_dir: Path,
    retrieved_at: str,
    generated_at: str,
    drop_fraction: float = 0.8,
    quench_window_ms: float = 5.0,
) -> dict[str, Any]:
    """Build the labelled dataset: write NPZ, checksum, manifest, and report.

    ``shots`` is a list of ``{"shot_id": int, "channels": {channel: array}}`` with
    the eleven measured channels. Writes each shot's ``.npz`` and the
    ``synthetic:false`` manifest into ``out_dir`` and returns the dataset report
    (``status:"blocked"``).
    """
    records = [
        build_shot_npz(
            int(shot["shot_id"]),
            shot["channels"],
            out_dir=out_dir,
            drop_fraction=drop_fraction,
            quench_window_ms=quench_window_ms,
        )
        for shot in shots
    ]
    manifest = _build_manifest(records, dataset_id=dataset_id, retrieved_at=retrieved_at)
    manifest_path = out_dir / f"{dataset_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset_fingerprint = hashlib.sha256(
        "".join(sorted(r["checksum_sha256"] for r in records)).encode("utf-8")
    ).hexdigest()
    label_algorithm = dict(LABEL_ALGORITHM)
    label_algorithm["drop_fraction"] = drop_fraction
    label_algorithm["quench_window_ms"] = quench_window_ms
    report: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA,
        "status": "blocked",
        "admission_ready": False,
        "blocked_reason": (
            "derived Ip current-quench labels are a bounded detector, not "
            "facility-validated; the detector must be cross-checked on manually "
            "inspected shots before promotion."
        ),
        "dataset_id": dataset_id,
        "synthetic": False,
        "manifest": manifest_path.name,
        "dataset_sha256": dataset_fingerprint,
        "n_shots": len(records),
        "n_disruptive": sum(r["label"] for r in records),
        "channel_schema": [*MEASURED_CHANNELS, "is_disruption", "disruption_time_idx", "disruption_type"],
        "label_algorithm": label_algorithm,
        "shots": records,
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def _load_shots(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        shot_ids = [int(s) for s in np.atleast_1d(data["shot_ids"])]
        shots: list[dict[str, Any]] = []
        for shot_id in shot_ids:
            channels = {name: np.asarray(data[f"{shot_id}:{name}"], dtype=np.float64) for name in MEASURED_CHANNELS}
            shots.append({"shot_id": shot_id, "channels": channels})
    return shots


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels-npz", type=Path, required=True, help="Extracted per-shot channel arrays.")
    parser.add_argument("--dataset-id", type=str, required=True, help="Dataset identifier.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for NPZ + manifest.")
    parser.add_argument("--json-out", type=Path, required=True, help="Dataset report JSON output path.")
    parser.add_argument("--retrieved-at", type=str, required=True, help="Acquisition timestamp (ISO 8601).")
    parser.add_argument("--generated-at", type=str, default="", help="Fixed UTC timestamp label.")
    parser.add_argument("--drop-fraction", type=float, default=0.8, help="Ip quench drop fraction.")
    parser.add_argument("--quench-window-ms", type=float, default=5.0, help="Ip quench window (ms).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build the labelled dataset and write the report."""
    args = _parse_args(argv)
    shots = _load_shots(args.channels_npz)
    report = build_dataset(
        shots,
        dataset_id=args.dataset_id,
        out_dir=args.out_dir,
        retrieved_at=args.retrieved_at,
        generated_at=args.generated_at,
        drop_fraction=args.drop_fraction,
        quench_window_ms=args.quench_window_ms,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"dataset: {report['n_disruptive']}/{report['n_shots']} disruptive (status={report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
