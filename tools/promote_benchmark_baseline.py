#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Explicit benchmark baseline promotion
"""Promote one digest-verified immutable benchmark report to a baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from scpn_control.benchmark_records import RUN_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "scpn-control.benchmark-regression.v1"
BASELINE_SCHEMA = "scpn-control.benchmark-baseline.v1"
PROMOTION_SCHEMA = "scpn-control.benchmark-baseline-promotion.v1"
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _repository_path(path: Path, label: str, repository_root: Path) -> Path:
    resolved = path if path.is_absolute() else repository_root / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{label} must remain inside the repository")
    return resolved


def _artifact_from_manifest(manifest_path: Path, role: str, repository_root: Path) -> tuple[dict[str, Any], Path, str]:
    if "runs" not in manifest_path.parts or manifest_path.name != "manifest.json":
        raise ValueError("source manifest is not inside an immutable runs directory")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA or manifest.get("status") != "succeeded":
        raise ValueError("source manifest is not a successful benchmark run")
    claimed_manifest_digest = manifest.get("payload_sha256")
    unsigned_manifest = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if claimed_manifest_digest != _sha256_bytes(_json_bytes(unsigned_manifest)):
        raise ValueError("source manifest payload digest is invalid")
    matches = [entry for entry in manifest.get("artifacts", []) if entry.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"source manifest must contain exactly one {role!r} artifact")
    entry = matches[0]
    if entry.get("kind") != "file":
        raise ValueError("baseline source artifact must be a file")
    raw_path = Path(str(entry.get("immutable_path", "")))
    artifact_path = raw_path if raw_path.is_absolute() else repository_root / raw_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_relative_to((manifest_path.parent / "artifacts").resolve()):
        raise ValueError("source artifact escapes its immutable run directory")
    digest = _sha256_bytes(artifact_path.read_bytes())
    if digest != entry.get("sha256"):
        raise ValueError("source artifact digest does not match the run manifest")
    return manifest, artifact_path, digest


def _validated_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"source report schema must be {REPORT_SCHEMA}")
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("source report has no benchmark metrics")
    claimed_digest = report.get("payload_sha256")
    unsigned = {key: value for key, value in report.items() if key != "payload_sha256"}
    if claimed_digest != _canonical_digest(unsigned):
        raise ValueError("source report payload digest is invalid")
    return report


def build_baseline(
    report: Mapping[str, Any],
    *,
    suite: str,
    source_manifest: str,
    source_sha256: str,
    authority_ref: str,
    hardware_compatibility: str,
    promoted_utc: str,
) -> dict[str, Any]:
    """Build a baseline that carries its immutable promotion provenance."""
    benchmarks = report["benchmarks"]
    baseline = {
        "schema_version": BASELINE_SCHEMA,
        "suite": suite,
        "baseline_commit": report["provenance"]["commit"],
        "measured_utc": report["generated_utc"],
        "evidence_class": report["evidence_class"],
        "production_claim_allowed": False,
        "provenance": report["provenance"],
        "benchmarks": benchmarks,
        "promotion": {
            "source_manifest": source_manifest,
            "source_artifact_sha256": source_sha256,
            "authority_ref": authority_ref,
            "hardware_compatibility": hardware_compatibility,
            "promoted_utc": promoted_utc,
        },
    }
    baseline["baseline_sha256"] = _canonical_digest(benchmarks)
    return baseline


def promote(
    *,
    source_manifest: Path,
    artifact_role: str,
    expected_source_sha256: str,
    baseline_path: Path,
    suite: str,
    authority_ref: str,
    hardware_compatibility: str,
    promotion_id: str,
    repository_root: Path = REPO_ROOT,
) -> Path:
    """Verify and apply one explicit promotion, returning its receipt path."""
    root = repository_root.resolve()
    if _IDENTIFIER.fullmatch(suite) is None or _IDENTIFIER.fullmatch(promotion_id) is None:
        raise ValueError("suite and promotion identifier must be filesystem-safe identifiers")
    if not authority_ref.strip():
        raise ValueError("authority reference must not be empty")
    if hardware_compatibility not in {"matched", "initial-baseline", "reviewed-mismatch"}:
        raise ValueError("hardware compatibility decision is invalid")
    if _SHA256.fullmatch(expected_source_sha256) is None:
        raise ValueError("expected source digest must be a lowercase SHA-256 digest")
    manifest_path = _repository_path(source_manifest, "source manifest", root)
    destination = _repository_path(baseline_path, "baseline path", root)
    manifest, artifact_path, source_digest = _artifact_from_manifest(manifest_path, artifact_role, root)
    if source_digest != expected_source_sha256:
        raise ValueError("expected source digest does not match the immutable artifact")
    report = _validated_report(artifact_path)
    promoted_utc = _utc_now()
    source_reference = manifest_path.relative_to(root).as_posix()
    baseline = build_baseline(
        report,
        suite=suite,
        source_manifest=source_reference,
        source_sha256=source_digest,
        authority_ref=authority_ref,
        hardware_compatibility=hardware_compatibility,
        promoted_utc=promoted_utc,
    )
    baseline_bytes = _json_bytes(baseline)
    baseline_file_sha256 = _sha256_bytes(baseline_bytes)

    history_root = root / "benchmarks" / "baseline_history" / suite
    receipt_path = history_root / "promotions" / f"{promotion_id}.json"
    if receipt_path.exists():
        raise FileExistsError(f"promotion identifier already exists: {receipt_path}")
    previous: dict[str, Any] | None = None
    if destination.is_file():
        previous_bytes = destination.read_bytes()
        previous_digest = _sha256_bytes(previous_bytes)
        archive_path = history_root / "baselines" / f"{previous_digest}.json"
        if not archive_path.exists():
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with archive_path.open("xb") as handle:
                handle.write(previous_bytes)
        previous = {
            "path": archive_path.relative_to(root).as_posix(),
            "sha256": previous_digest,
        }

    receipt = {
        "schema_version": PROMOTION_SCHEMA,
        "promotion_id": promotion_id,
        "suite": suite,
        "campaign_id": manifest["campaign_id"],
        "source_manifest": source_reference,
        "source_artifact_role": artifact_role,
        "source_artifact_sha256": source_digest,
        "baseline_path": destination.relative_to(root).as_posix(),
        "baseline_file_sha256": baseline_file_sha256,
        "baseline_metrics_sha256": baseline["baseline_sha256"],
        "authority_ref": authority_ref,
        "hardware_compatibility": hardware_compatibility,
        "promoted_utc": promoted_utc,
        "previous_baseline": previous,
    }
    receipt["payload_sha256"] = _canonical_digest(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(destination, baseline_bytes)
    with receipt_path.open("xb") as handle:
        handle.write(_json_bytes(receipt))
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    """Validate promotion arguments, apply the baseline, and emit its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--artifact-role", default="report")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--authority-ref", required=True)
    parser.add_argument(
        "--hardware-compatibility",
        required=True,
        choices=("matched", "initial-baseline", "reviewed-mismatch"),
    )
    parser.add_argument("--promotion-id", default="")
    args = parser.parse_args(argv)
    promotion_id = args.promotion_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    try:
        receipt = promote(
            source_manifest=args.source_manifest,
            artifact_role=args.artifact_role,
            expected_source_sha256=args.expected_source_sha256,
            baseline_path=args.baseline,
            suite=args.suite,
            authority_ref=args.authority_ref,
            hardware_compatibility=args.hardware_compatibility,
            promotion_id=promotion_id,
            repository_root=args.repository_root,
        )
    except (FileExistsError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"baseline promotion FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"baseline promotion receipt: {receipt.relative_to(args.repository_root.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
