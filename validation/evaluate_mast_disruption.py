#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption-prediction evaluation harness
"""Evaluate the disruption predictor on a set of real-schema shot arrays.

Consumes ``.npz`` shots in the ``run_real_shot_replay`` channel schema and reports
a bounded, threshold-swept ROC/AUC plus DisruptionBench-style warning-time recall
for the existing fixed-weight disruption heuristic. The report is deliberately
fail-closed: ``status`` stays ``"blocked"`` and ``admission_ready`` is ``False``,
and it re-emits the frozen :func:`disruption_risk_claim_boundary` so the internal
number can never be mistaken for a promoted public claim.

Data provenance (real vs. synthetic) is taken from the supplied
:class:`RealDataManifest`, which is provenance-truthful by construction, so this
harness cannot mislabel a synthetic fixture campaign as real. It runs end-to-end
on the committed synthetic disruption fixtures to prove the path; the real-MAST
number is produced out-of-band once shots are acquired.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control.control.disruption_predictor import disruption_risk_claim_boundary
from scpn_control.control.disruption_roc import (
    ShotEvaluation,
    disruption_metrics,
    score_risk_series,
)
from scpn_control.core.real_data_manifest import load_real_data_manifest

REPORT_SCHEMA = "scpn-control.mast-disruption-evaluation.v1"
DEFAULT_WINDOW_SIZE = 128
DEFAULT_ALARM_THRESHOLD = 0.65
DEFAULT_WARNING_MS: tuple[int, ...] = (10, 20, 30, 50, 100)
_N_THRESHOLDS = 51

# Every ``run_real_shot_replay`` shot carries these channels; the heuristic only
# consumes dBdt + n=1/n=2, but the full schema is required so the dataset is
# well-formed and the same NPZ can feed the mitigation replay.
REQUIRED_ARRAY_CHANNELS: tuple[str, ...] = (
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
REQUIRED_SCALAR_CHANNELS: tuple[str, ...] = (
    "is_disruption",
    "disruption_time_idx",
    "disruption_type",
)


def _sha256_json(payload: dict[str, Any]) -> str:
    """Canonical SHA-256 of a JSON payload (sorted keys, no whitespace)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True)
class ShotRecord:
    """A loaded real-schema shot plus its identifier."""

    shot_id: str
    dbdt: NDArray[np.float64]
    n1_amp: NDArray[np.float64]
    n2_amp: NDArray[np.float64]
    time_s: NDArray[np.float64]
    label: int
    disruption_time_idx: int
    disruption_type: str


def load_shot(path: Path) -> ShotRecord:
    """Load and validate one ``.npz`` shot in the replay channel schema."""
    with np.load(path, allow_pickle=False) as data:
        present = set(data.files)
        missing = [c for c in (*REQUIRED_ARRAY_CHANNELS, *REQUIRED_SCALAR_CHANNELS) if c not in present]
        if missing:
            raise ValueError(f"{path.name}: missing shot channels {missing}.")
        arrays = {name: np.asarray(data[name], dtype=np.float64) for name in REQUIRED_ARRAY_CHANNELS}
        n_samples = int(arrays["time_s"].shape[0])
        for name, array in arrays.items():
            if array.ndim != 1 or array.shape[0] != n_samples:
                raise ValueError(f"{path.name}: channel {name!r} must be 1-D with {n_samples} samples.")
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"{path.name}: channel {name!r} must be finite.")
        label = 1 if bool(data["is_disruption"]) else 0
        disruption_time_idx = int(data["disruption_time_idx"])
        disruption_type = str(data["disruption_type"])
    return ShotRecord(
        shot_id=path.stem,
        dbdt=arrays["dBdt_gauss_per_s"],
        n1_amp=arrays["n1_amp"],
        n2_amp=arrays["n2_amp"],
        time_s=arrays["time_s"],
        label=label,
        disruption_time_idx=disruption_time_idx,
        disruption_type=disruption_type,
    )


def build_evaluation(shot: ShotRecord, *, window_size: int) -> ShotEvaluation:
    """Score a shot and wrap it as a :class:`ShotEvaluation`."""
    risk_series = score_risk_series(shot.dbdt, shot.n1_amp, shot.n2_amp, window_size=window_size)
    return ShotEvaluation(
        risk_series=risk_series,
        label=shot.label,
        disruption_time_idx=shot.disruption_time_idx,
        time_s=shot.time_s,
        window_size=window_size,
    )


def load_shots(shots_dir: Path) -> list[ShotRecord]:
    """Load every ``.npz`` shot in ``shots_dir`` sorted by filename."""
    paths = sorted(shots_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"no .npz shots found under {shots_dir}.")
    return [load_shot(path) for path in paths]


def build_report(
    shots: list[ShotRecord],
    *,
    manifest_path: Path,
    window_size: int,
    alarm_threshold: float,
    warning_ms: tuple[int, ...],
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the fail-closed evaluation report from loaded shots."""
    manifest = load_real_data_manifest(manifest_path)
    evaluations = [build_evaluation(shot, window_size=window_size) for shot in shots]
    thresholds = np.linspace(0.0, 1.0, _N_THRESHOLDS)
    metrics = disruption_metrics(
        evaluations,
        thresholds=[float(t) for t in thresholds],
        alarm_threshold=alarm_threshold,
        warning_ms=list(warning_ms),
    )
    boundary = dataclasses.asdict(disruption_risk_claim_boundary())
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "blocked",
        "admission_ready": False,
        "blocked_reason": (
            "Bounded internal ROC/AUC for the fixed-weight disruption heuristic; "
            "the disruption-risk claim boundary remains locked pending a real fit "
            "and a sealed reference artefact."
        ),
        "dataset_id": manifest.dataset_id,
        "data_provenance": {
            "synthetic": manifest.synthetic,
            "manifest_path": str(manifest_path),
            "source_kind": manifest.source.kind,
            "source_uri": manifest.source.uri,
            "licence": manifest.licence,
        },
        "predictor": {
            "score_source": "fixed_weight_logistic_heuristic",
            "feature_note": (
                "scores the dBdt window plus n=1/n=2 toroidal observables; does "
                "not consume the Rea-2019 q95/beta_N/li/Greenwald/P_rad features."
            ),
        },
        "window_size": window_size,
        "metrics": metrics,
        "shots": [
            {
                "shot_id": shot.shot_id,
                "label": shot.label,
                "disruption_type": shot.disruption_type,
                "disruption_time_idx": shot.disruption_time_idx,
                "n_samples": int(shot.time_s.shape[0]),
            }
            for shot in shots
        ],
        "claim_boundary": boundary,
        "generated_at_utc": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary of the report."""
    metrics = report["metrics"]
    provenance = report["data_provenance"]
    recall = metrics["recall_at_warning_ms"]
    lines = [
        "# FAIR-MAST Disruption Evaluation",
        "",
        f"- **Status**: {report['status']} (admission_ready={report['admission_ready']})",
        f"- **Dataset**: {report['dataset_id']} (synthetic={provenance['synthetic']})",
        f"- **Shots**: {metrics['n_shots']} ({metrics['n_disruptive']} disruptive)",
        f"- **Window**: {report['window_size']} samples · alarm threshold {metrics['alarm_threshold']:.2f}",
        f"- **AUC**: {metrics['auc']:.4f}",
        "",
        "## Warning-time recall (disruptive shots alarmed ≥ N ms before disruption)",
        "",
        "| Warning (ms) | Recall |",
        "| --- | --- |",
    ]
    lines.extend(f"| {ms} | {value:.4f} |" for ms, value in recall.items())
    lines.extend(
        [
            "",
            f"> {report['blocked_reason']}",
            "",
            f"Predictor: {report['predictor']['feature_note']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-dir", type=Path, required=True, help="Directory of .npz shots.")
    parser.add_argument("--manifest", type=Path, required=True, help="RealDataManifest for the shots.")
    parser.add_argument("--json-out", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--report-out", type=Path, required=True, help="Report Markdown output path.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--alarm-threshold", type=float, default=DEFAULT_ALARM_THRESHOLD)
    parser.add_argument(
        "--generated-at",
        type=str,
        default=None,
        help="Override the UTC timestamp (ISO 8601) for reproducible reports.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: evaluate shots and write the JSON + Markdown report."""
    args = _parse_args(argv)
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    shots = load_shots(args.shots_dir)
    report = build_report(
        shots,
        manifest_path=args.manifest,
        window_size=args.window_size,
        alarm_threshold=args.alarm_threshold,
        warning_ms=DEFAULT_WARNING_MS,
        generated_at=generated_at,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_out.write_text(render_markdown(report), encoding="utf-8")
    auc = report["metrics"]["auc"]
    print(f"AUC={auc:.4f} (status={report['status']}, synthetic={report['data_provenance']['synthetic']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
