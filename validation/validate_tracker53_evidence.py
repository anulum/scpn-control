# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tracker #53 evidence gate
"""Aggregate tracker #53 hardware/runtime evidence and fail closed on overclaims."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.validate_native_formal_certificate_evidence import validate_native_formal_certificate_evidence
from validation.validate_runtime_admission_evidence import validate_runtime_admission_evidence

TRACKER_ISSUE = 53
TRACKER53_SCHEMA_VERSION = "scpn-control.tracker53-evidence-gate.v1"
DEFAULT_REGISTRY = ROOT / "validation" / "physics_traceability.json"
DEFAULT_RUNTIME_REPORT = ROOT / "validation" / "reports" / "runtime_admission_release_20260605T000000Z.json"
DEFAULT_NATIVE_FORMAL_REPORT = (
    ROOT / "validation" / "reports" / "native_formal_aot_certificate_admission_20260604T103219Z.json"
)
DEFAULT_Z3_REPORT = ROOT / "validation" / "reports" / "scpn_z3_formal.json"

TRACKER53_MODULE_ORDER = (
    "src/scpn_control/core/checkpoint.py",
    "src/scpn_control/phase/kuramoto.py",
    "src/scpn_control/scpn/formal_verification.py",
    "src/scpn_control/scpn/fpga_export.py",
    "src/scpn_control",
    "src/scpn_control/core/runtime_admission.py",
)
QUALIFIED_EVIDENCE_CLASSES = frozenset(
    {
        "checkpoint_replay_qualified",
        "deployment_target_parity",
        "formal_production_benchmark",
        "hardware_qualified",
        "qualified_hdl_synthesis",
        "runtime_production_benchmark",
    }
)


@dataclass(frozen=True)
class Tracker53EvidenceResult:
    """Admission result for the aggregate tracker #53 evidence gate."""

    status: str
    tracker_issue: int
    entries: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    production_claim_allowed: bool
    evidence_classes: dict[str, str]
    require_production_claim: bool


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _tracker53_registry_entries(registry: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    try:
        payload = _load_json(registry)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"tracker53.registry: {exc}")
        return {}

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("tracker53.registry.entries must be a list")
        return {}

    selected: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, dict) or raw.get("external_validation_tracker_issue") != TRACKER_ISSUE:
            continue
        module_path = raw.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            errors.append("tracker53.registry entry has invalid module_path")
            continue
        if module_path in selected:
            errors.append(f"tracker53.registry duplicate module_path: {module_path}")
            continue
        selected[module_path] = raw

    missing = [module for module in TRACKER53_MODULE_ORDER if module not in selected]
    if missing:
        errors.append(f"tracker53.registry missing modules: {', '.join(missing)}")
    return selected


def _z3_report_status(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"formal_verification.z3_report: {path} does not exist")
        return {"path": str(path), "status": "missing", "report_sha256": None}
    try:
        payload = _load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"formal_verification.z3_report: {exc}")
        return {"path": str(path), "status": "fail", "report_sha256": None}
    status = payload.get("status")
    return {
        "path": str(path.relative_to(ROOT)),
        "status": status if isinstance(status, str) else "unknown",
        "report_sha256": _sha256_file(path),
    }


def _base_entry(raw: dict[str, Any], evidence_class: str) -> dict[str, Any]:
    return {
        "module_path": raw["module_path"],
        "component": raw.get("component"),
        "fidelity_status": raw.get("fidelity_status"),
        "public_claim_allowed": raw.get("public_claim_allowed"),
        "evidence_class": evidence_class,
        "qualified_for_production_claim": evidence_class in QUALIFIED_EVIDENCE_CLASSES,
        "validity_domain": raw.get("validity_domain"),
        "evidence_paths": raw.get("evidence_paths", []),
    }


def _checkpoint_entry(raw: dict[str, Any]) -> dict[str, Any]:
    entry = _base_entry(raw, "local_bounded_checkpoint")
    entry["claim_boundary"] = "schema and round-trip replay checks only; not hardware-qualified restoration evidence"
    return entry


def _kuramoto_entry(raw: dict[str, Any]) -> dict[str, Any]:
    entry = _base_entry(raw, "runtime_local_regression")
    entry["claim_boundary"] = "exact-form synchronisation and local runtime parity; not deployment-class timing evidence"
    return entry


def _fpga_entry(raw: dict[str, Any]) -> dict[str, Any]:
    entry = _base_entry(raw, "generated_hdl")
    entry["claim_boundary"] = "generated HDL schema and fixed-point checks only; no synthesis timing or bitstream closure"
    return entry


def _aggregate_entry(raw: dict[str, Any]) -> dict[str, Any]:
    entry = _base_entry(raw, "aggregate_boundary")
    entry["claim_boundary"] = "package-level tracker guard; production unblock depends on all child module evidence classes"
    return entry


def _runtime_entry(raw: dict[str, Any], report: Path, errors: list[str]) -> dict[str, Any]:
    result = validate_runtime_admission_evidence(report)
    if result.errors:
        errors.extend(result.errors)
    evidence_class = (
        "runtime_production_benchmark"
        if result.benchmark_evidence_class == "production_benchmark" and result.production_claim_allowed is True
        else "runtime_local_regression"
    )
    entry = _base_entry(raw, evidence_class)
    entry.update(
        {
            "validator_status": result.status,
            "benchmark_evidence_class": result.benchmark_evidence_class,
            "production_claim_allowed_by_report": result.production_claim_allowed,
            "admission_status": result.admission_status,
            "admission_error_count": result.admission_error_count,
            "samples": result.samples,
            "report_path": _display_path(Path(report)),
            "report_sha256": result.report_sha256,
            "payload_sha256": result.payload_sha256,
            "claim_boundary": "local runtime-admission benchmark only unless production realtime-host evidence is supplied",
        }
    )
    return entry


def _formal_entry(raw: dict[str, Any], native_report: Path, z3_report: Path, errors: list[str]) -> dict[str, Any]:
    native_result = validate_native_formal_certificate_evidence(native_report)
    if native_result.errors:
        errors.extend(f"native_formal_certificate.{error}" for error in native_result.errors)
    evidence_class = (
        "formal_production_benchmark"
        if native_result.benchmark_evidence_class == "production_benchmark"
        and native_result.production_claim_allowed is True
        else "formal_local_regression"
    )
    entry = _base_entry(raw, evidence_class)
    entry.update(
        {
            "validator_status": native_result.status,
            "benchmark_evidence_class": native_result.benchmark_evidence_class,
            "production_claim_allowed_by_report": native_result.production_claim_allowed,
            "admitted_cases": list(native_result.admitted_cases),
            "certificate_assumption_sha256": native_result.certificate_assumption_sha256,
            "native_report_path": _display_path(Path(native_report)),
            "native_report_sha256": native_result.report_sha256,
            "z3_report": _z3_report_status(z3_report, errors),
            "claim_boundary": "local exact-form proof packaging; not an externally reviewed safety-case signoff",
        }
    )
    return entry


def build_tracker53_manifest(result: Tracker53EvidenceResult) -> dict[str, Any]:
    """Build a canonical, digest-bound JSON manifest from a validation result."""
    payload: dict[str, Any] = {
        "schema_version": TRACKER53_SCHEMA_VERSION,
        "tracker_issue": result.tracker_issue,
        "status": result.status,
        "require_production_claim": result.require_production_claim,
        "production_claim_allowed": result.production_claim_allowed,
        "errors": list(result.errors),
        "evidence_classes": result.evidence_classes,
        "entries": list(result.entries),
    }
    payload["manifest_sha256"] = _canonical_digest(payload)
    return payload


def validate_tracker53_evidence(
    *,
    registry: str | Path = DEFAULT_REGISTRY,
    runtime_report: str | Path = DEFAULT_RUNTIME_REPORT,
    native_formal_report: str | Path = DEFAULT_NATIVE_FORMAL_REPORT,
    z3_report: str | Path = DEFAULT_Z3_REPORT,
    require_production_claim: bool = False,
    output_json: str | Path | None = None,
) -> Tracker53EvidenceResult:
    """Validate the local tracker #53 evidence manifest and optional production claim."""
    errors: list[str] = []
    raw_entries = _tracker53_registry_entries(Path(registry), errors)
    entries_by_module: dict[str, dict[str, Any]] = {}

    builders = {
        "src/scpn_control/core/checkpoint.py": lambda raw: _checkpoint_entry(raw),
        "src/scpn_control/phase/kuramoto.py": lambda raw: _kuramoto_entry(raw),
        "src/scpn_control/scpn/formal_verification.py": lambda raw: _formal_entry(
            raw, Path(native_formal_report), Path(z3_report), errors
        ),
        "src/scpn_control/scpn/fpga_export.py": lambda raw: _fpga_entry(raw),
        "src/scpn_control": lambda raw: _aggregate_entry(raw),
        "src/scpn_control/core/runtime_admission.py": lambda raw: _runtime_entry(
            raw, Path(runtime_report), errors
        ),
    }

    for module in TRACKER53_MODULE_ORDER:
        raw = raw_entries.get(module)
        if raw is None:
            continue
        entries_by_module[module] = builders[module](raw)

    entries = tuple(entries_by_module[module] for module in TRACKER53_MODULE_ORDER if module in entries_by_module)
    evidence_classes = {entry["module_path"]: entry["evidence_class"] for entry in entries}

    if require_production_claim:
        unqualified = [
            module for module in TRACKER53_MODULE_ORDER if evidence_classes.get(module) not in QUALIFIED_EVIDENCE_CLASSES
        ]
        if unqualified:
            errors.append(
                "production tracker #53 claim requires qualified hardware evidence for all module surfaces: "
                + ", ".join(unqualified)
            )

    production_claim_allowed = require_production_claim and not errors and all(
        evidence_classes.get(module) in QUALIFIED_EVIDENCE_CLASSES for module in TRACKER53_MODULE_ORDER
    )
    result = Tracker53EvidenceResult(
        status="pass" if not errors else "fail",
        tracker_issue=TRACKER_ISSUE,
        entries=entries,
        errors=tuple(errors),
        production_claim_allowed=production_claim_allowed,
        evidence_classes=evidence_classes,
        require_production_claim=require_production_claim,
    )

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(build_tracker53_manifest(result), indent=2, sort_keys=True) + "\n")

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for tracker #53 evidence-gate validation."""
    parser = argparse.ArgumentParser(description="Validate aggregate tracker #53 hardware/runtime evidence")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Physics traceability JSON registry")
    parser.add_argument("--runtime-report", default=str(DEFAULT_RUNTIME_REPORT), help="Runtime-admission report JSON")
    parser.add_argument(
        "--native-formal-report",
        default=str(DEFAULT_NATIVE_FORMAL_REPORT),
        help="Native formal certificate report JSON",
    )
    parser.add_argument("--z3-report", default=str(DEFAULT_Z3_REPORT), help="Z3 formal report JSON")
    parser.add_argument("--require-production-claim", action="store_true")
    parser.add_argument("--output-json", help="Write digest-bound manifest JSON to this path")
    parser.add_argument("--json-out", action="store_true", help="Print the manifest JSON to stdout")
    args = parser.parse_args(argv)

    result = validate_tracker53_evidence(
        registry=args.registry,
        runtime_report=args.runtime_report,
        native_formal_report=args.native_formal_report,
        z3_report=args.z3_report,
        require_production_claim=args.require_production_claim,
        output_json=args.output_json,
    )
    manifest = build_tracker53_manifest(result)
    if args.json_out:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Tracker #53 evidence gate: {result.status}")
        print(f"production_claim_allowed={result.production_claim_allowed}")
        for error in result.errors:
            print(f"ERROR {error}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
