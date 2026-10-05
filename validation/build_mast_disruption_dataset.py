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
derives explicit Ip-proxy labels with the documented current-quench detector,
writes each shot as an ``.npz`` in the full replay schema with a self-digested
``ShotLabelRecord``, checksums every file, and emits a ``synthetic:false``
:class:`RealDataManifest` plus a schema-versioned dataset report.

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
import re
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from scpn_control._npz import save_npz_arrays
from scpn_control.core.real_data_manifest import validate_real_data_manifest
from validation.fair_mast_source_policy import fair_mast_provenance
from validation.mast_disruption_shot_label import SHOT_LABEL_RECORD_SCHEMA as SHOT_LABEL_RECORD_SCHEMA
from validation.mast_disruption_shot_label import (
    ProgrammeClass,
    derive_ip_quench_proxy,
    ip_quench_proxy_algorithm,
)
from validation.mast_replay_contracts._archive import read_replay_archive_bytes
from validation.mast_replay_contracts._inputs import MEASURED_CHANNELS as MEASURED_CHANNELS
from validation.mast_replay_contracts._inputs import channel_vectors, shot_identity
from validation.mast_replay_contracts._report import DATASET_SCHEMA as DATASET_SCHEMA
from validation.mast_replay_contracts._report import _sha256_json as _sha256_json
from validation.mast_replay_contracts._report import dataset_report
from validation.report_output_paths import checked_report_destination

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
    """Return the legacy tuple projection of an uncalibrated Ip-proxy label.

    A shot is disruptive when, after the last near-flat-top sample, the current
    terminally collapses below ``(1 - drop_fraction)`` of its flat-top maximum
    within ``quench_window_ms``. Anchoring on the last near-flat-top sample means
    the initial current ramp (also below the collapse threshold) is ignored.
    Returns ``(is_disruption, onset_index, disruption_type)`` with ``onset_index``
    ``-1`` for a non-disruptive shot, implementing the algorithm documented by the
    feature-source audit's ``LABEL_ALGORITHM``. ``ip`` uses MA and ``time_s``
    seconds; both must be finite nonempty aligned 1-D vectors, with time strictly
    increasing. The detector uses absolute current, so either current sign
    is supported. Defaults are ``drop_fraction=0.8`` and ``quench_window_ms=5.0``;
    the former must lie strictly within (0, 1), the latter be finite/positive.

    No current gives an ambiguous ``no_current`` classification, while no
    collapse or a slow rampdown gives a non-disruptive result. The tuple omits
    the full record's authority/ambiguity detail. Input arrays are unchanged;
    invalid domains raise ``ShotLabelRecordError`` (a ``ValueError`` subtype),
    and array conversion errors propagate. There is no I/O or independent
    facility outcome, calibrated probability or thermal-quench inference.
    """
    result = derive_ip_quench_proxy(
        np.asarray(ip, dtype=np.float64),
        np.asarray(time_s, dtype=np.float64),
        shot_id=1,
        drop_fraction=drop_fraction,
        quench_window_ms=quench_window_ms,
    )
    return (result.is_disruption, result.onset_index, result.classification)


def _validate_channels(shot_id: int, channels: dict[str, NDArray[np.float64]]) -> int:
    """Validate aligned finite candidate vectors and their chronological clock."""
    return channel_vectors(channels, shot_id=shot_id)


def build_shot_npz(
    shot_id: int,
    channels: dict[str, NDArray[np.float64]],
    *,
    out_dir: Path,
    drop_fraction: float,
    quench_window_ms: float,
    programme_class: ProgrammeClass = "unknown",
) -> dict[str, Any]:
    """Label one measured shot and write its uncompressed replay archive.

    Parameters
    ----------
    shot_id
        Positive Python integer excluding bool, within signed int64, embedded
        in the label record and output name.
    channels
        Eleven measured vectors named by ``MEASURED_CHANNELS``. They must be
        finite, one-dimensional and share the ``time_s`` sample count; values
        are normalised to float64 using the units in ``CHANNEL_UNITS``.
        The nonempty time vector must be strictly increasing. Extra mapping
        keys are ignored. Caller arrays remain unchanged.
    out_dir
        Directory created as needed for ``shot_<shot_id>.npz``. An existing
        archive at that path is overwritten.
    drop_fraction
        Fractional current decrease used by the Ip-proxy label detector.
    quench_window_ms
        Maximum current-quench interval in milliseconds.
    programme_class
        Programme classification carried by the label record, not inferred
        from the measured channels.

    Returns
    -------
    dict
        Shot identity, relative archive path, archive SHA-256, sample count and
        explicit proxy-label metadata. The archive stores measured vectors and
        scalar legacy labels plus the canonical label-record JSON string.

    Raises
    ------
    ValueError
        If a measured vector or the proxy-label inputs are invalid.
    OSError
        If directory creation or archive writing fails. Output publication is
        direct and may leave a partial file.
    """
    shot_id = shot_identity(shot_id)
    _validate_channels(shot_id, channels)
    ip = np.asarray(channels["Ip_MA"], dtype=np.float64)
    time_s = np.asarray(channels["time_s"], dtype=np.float64)
    proxy = derive_ip_quench_proxy(
        ip,
        time_s,
        shot_id=shot_id,
        programme_class=programme_class,
        drop_fraction=drop_fraction,
        quench_window_ms=quench_window_ms,
    )
    label_record = proxy.record.to_dict()
    payload = {name: np.asarray(channels[name], dtype=np.float64) for name in MEASURED_CHANNELS}
    payload["is_disruption"] = np.asarray(proxy.is_disruption)
    payload["disruption_time_idx"] = np.asarray(proxy.onset_index)
    payload["disruption_type"] = np.asarray(proxy.classification)
    payload["shot_label_record_json"] = np.asarray(
        json.dumps(label_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"shot_{shot_id}.npz"
    save_npz_arrays(npz_path, payload, allow_pickle=True)
    return {
        "shot_id": shot_id,
        "npz": npz_path.name,
        "checksum_sha256": _sha256_file(npz_path),
        "label": 1 if proxy.is_disruption else 0,
        "disruption_time_idx": proxy.onset_index,
        "disruption_type": proxy.classification,
        "label_record": label_record,
        "n_samples": int(time_s.shape[0]),
    }


def _build_manifest(
    records: list[dict[str, Any]],
    *,
    dataset_id: str,
    retrieved_at: str,
) -> dict[str, Any]:
    """Validate local candidate provenance declarations without authenticating sources."""
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
    """Write candidate shot archives/manifest and return a blocked v2 report.

    ``shots`` is a nonempty list of mappings with a distinct positive Python
    signed-int64 ``shot_id`` and eleven finite aligned channel vectors. An
    optional ``programme_class`` is passed through to the proxy record;
    ``unknown`` is the default. All IDs, vectors and proxy-label domains are
    checked before the first write. Caller arrays are unchanged.

    ``dataset_id`` is a single ASCII filename identifier beginning with an
    alphanumeric character and containing only letters/digits/dot/underscore/
    hyphen. ``retrieved_at`` must be nonempty; this label and ``generated_at``
    are copied without timestamp parsing. Label defaults and units match
    ``derive_ip_quench_label``. The returned dataset digest hashes the sorted
    shot-file checksums; the report payload has its own canonical body digest.

    Files are written directly into ``out_dir`` and existing shot/manifest
    files may be overwritten. Later I/O/manifest failures can leave partial
    output; batch preflight is not a filesystem transaction. This API does
    not enforce the CLI's selected-source/report path protections. Invalid
    domains raise authored ``ValueError`` or its proxy/manifest subtypes;
    missing keys and native I/O/conversion errors propagate.

    No report file is written here. Return status remains ``blocked``,
    ``admission_ready=False`` and ``independent_label_count=0``: all labels use
    input-feature-derived ``ip_proxy`` authority, never facility ground truth.
    """
    if not isinstance(dataset_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset_id) is None:
        raise ValueError("dataset_id must be a nonempty single filename identifier")
    if not shots:
        raise ValueError("dataset requires at least one shot")
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        raise ValueError("retrieved_at must be a nonempty acquisition label")
    identities = []
    for shot in shots:
        identity = shot_identity(shot["shot_id"])
        if identity in identities:
            raise ValueError("dataset shot identities must be unique")
        identities.append(identity)
        channel_vectors(shot["channels"], shot_id=identity)
        derive_ip_quench_proxy(
            np.asarray(shot["channels"]["Ip_MA"], dtype=np.float64),
            np.asarray(shot["channels"]["time_s"], dtype=np.float64),
            shot_id=identity,
            programme_class=cast(ProgrammeClass, shot.get("programme_class", "unknown")),
            drop_fraction=drop_fraction,
            quench_window_ms=quench_window_ms,
        )
    records = [
        build_shot_npz(
            int(shot["shot_id"]),
            shot["channels"],
            out_dir=out_dir,
            drop_fraction=drop_fraction,
            quench_window_ms=quench_window_ms,
            programme_class=cast(ProgrammeClass, shot.get("programme_class", "unknown")),
        )
        for shot in shots
    ]
    manifest = _build_manifest(records, dataset_id=dataset_id, retrieved_at=retrieved_at)
    manifest_path = out_dir / f"{dataset_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset_fingerprint = hashlib.sha256(
        "".join(sorted(r["checksum_sha256"] for r in records)).encode("utf-8")
    ).hexdigest()
    label_algorithm = ip_quench_proxy_algorithm(
        drop_fraction=drop_fraction,
        quench_window_ms=quench_window_ms,
    )
    return dataset_report(
        records,
        dataset_id=dataset_id,
        manifest_name=manifest_path.name,
        dataset_fingerprint=dataset_fingerprint,
        label_algorithm=label_algorithm,
        generated_at=generated_at,
    )


def _load_shots(path: Path) -> list[dict[str, Any]]:
    """Decode one local snapshot; retain exact-integral float-ID compatibility."""
    shots, _ = read_replay_archive_bytes(path.read_bytes(), path_name=path.name, integral_float_compatibility=True)
    return shots


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the maintained source-tree dataset command arguments."""
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
    """Run the local source-tree dataset command and publish its blocked report.

    ``argv=None`` reads process arguments; argparse help/parser exits are 0/2.
    The source NPZ snapshot is decoded without pickle. Exact unique members,
    sorted positive int64 IDs (including historical exact-integral floats),
    eleven finite floating vectors and increasing nonempty clocks are required.

    Before candidate writes, report/artifact paths are checked against the
    selected NPZ, this module and current contract helper sources using direct,
    resolved, symbolic and existing hardlink identity. Checks are sequential,
    not a race-proof snapshot or protection of every dependency. Outputs are
    direct writes, can overwrite unrelated existing destinations, and have
    no rollback after a later I/O failure.

    Return ``0`` when candidate files and JSON report are written, even though
    scientific admission remains blocked. Direct calls propagate input/path,
    decode and native I/O errors. The module command maps caught value/I/O/
    type/key errors to fixed stderr and exit ``2`` without internal exception text.
    """
    args = _parse_args(argv)
    shots = _load_shots(args.channels_npz)
    outputs = [
        args.out_dir / f"{args.dataset_id}.manifest.json",
        *(args.out_dir / f"shot_{shot['shot_id']}.npz" for shot in shots),
    ]
    sources = [args.channels_npz, Path(__file__), *Path(__file__).parent.joinpath("mast_replay_contracts").glob("*.py")]
    checked_report_destination(args.json_out, inputs=[*sources, *outputs])
    for destination in outputs:
        checked_report_destination(destination, inputs=sources)
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
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError, KeyError):
        print("Could not build the MAST candidate dataset.", file=sys.stderr)
        raise SystemExit(2) from None
