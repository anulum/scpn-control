#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# ----------------------------------------------------------------------
# SCPN Control - Data Manifest Validation Runner
# Copyright (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ----------------------------------------------------------------------
"""Validate repository data manifests and local artefact checksums."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

_REAL_DATA_MODULE_PATH = SRC / "scpn_control" / "core" / "real_data_manifest.py"
_REAL_DATA_MODULE_NAME = "_scpn_control_real_data_manifest_contract"


def _load_real_data_manifest_api() -> tuple[type[Exception], Any]:
    spec = importlib.util.spec_from_file_location(_REAL_DATA_MODULE_NAME, _REAL_DATA_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load real-data manifest contract from {_REAL_DATA_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(type[Exception], module.RealDataManifestError), module.load_real_data_manifest


RealDataManifestError, load_real_data_manifest = _load_real_data_manifest_api()


_DIIID_ARTIFACT_PATTERNS = ("*.geqdsk", "disruption_shots/*.npz")


@dataclass(frozen=True)
class AcquisitionSpec:
    """Stdlib-only acquisition request summary used by the CI manifest gate."""

    tree: str
    shot: int
    source_uri: str
    signals: int


def iter_manifest_paths(root: str | Path) -> list[Path]:
    """Return repository manifest paths under ``root`` in stable order."""
    root_path = Path(root)
    return sorted(root_path.glob("**/manifests/*.manifest.json"))


def iter_acquisition_spec_paths(root: str | Path) -> list[Path]:
    """Return repository acquisition specification paths under ``root``."""
    root_path = Path(root)
    return sorted(root_path.glob("**/acquisition_specs/*.json"))


def iter_diiid_artifact_paths(root: str | Path) -> list[Path]:
    """Return DIII-D GEQDSK and disruption-shot artefacts requiring manifests."""
    root_path = Path(root)
    diiid_root = root_path / "diiid" if (root_path / "diiid").is_dir() else root_path
    paths: list[Path] = []
    for pattern in _DIIID_ARTIFACT_PATTERNS:
        paths.extend(diiid_root.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def validate_manifest_directory(root: str | Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    """Validate all manifests below ``root`` and return a CI-friendly report."""
    root_path = Path(root)
    manifest_paths = iter_manifest_paths(root)
    acquisition_spec_paths = iter_acquisition_spec_paths(root)
    expected_artifacts = iter_diiid_artifact_paths(root)
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root_path),
        "total": len(manifest_paths),
        "real": 0,
        "synthetic": 0,
        "artifact_verification": bool(verify_artifacts),
        "artifact_coverage": {
            "expected": len(expected_artifacts),
            "covered": 0,
            "missing": [],
        },
        "acquisition_specs": {
            "total": len(acquisition_spec_paths),
            "mdsplus": 0,
            "specs": [],
        },
        "manifests": [],
        "errors": [],
    }

    if not manifest_paths:
        report["status"] = "fail"
        report["errors"].append({"path": str(Path(root)), "error": "no data manifests found"})
        return report

    manifests: list[dict[str, object]] = report["manifests"]
    acquisition_specs = cast(dict[str, Any], report["acquisition_specs"])
    spec_entries = cast(list[dict[str, object]], acquisition_specs["specs"])
    errors: list[dict[str, str]] = report["errors"]
    covered_artifacts: set[Path] = set()
    for spec_path in acquisition_spec_paths:
        try:
            spec = load_acquisition_spec(spec_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(spec_path), "error": str(exc)})
            continue
        acquisition_specs["mdsplus"] = cast(int, acquisition_specs["mdsplus"]) + 1
        spec_entries.append(
            {
                "path": str(spec_path),
                "kind": "mdsplus",
                "tree": spec.tree,
                "shot": spec.shot,
                "source_uri": spec.source_uri,
                "signals": spec.signals,
            }
        )

    for manifest_path in manifest_paths:
        try:
            manifest = load_real_data_manifest(manifest_path, verify_artifact=verify_artifacts)
        except (OSError, RealDataManifestError, json.JSONDecodeError) as exc:
            errors.append({"path": str(manifest_path), "error": str(exc)})
            continue

        if manifest.synthetic:
            report["synthetic"] += 1
        else:
            report["real"] += 1
        manifests.append(
            {
                "path": str(manifest_path),
                "dataset_id": manifest.dataset_id,
                "kind": manifest.kind,
                "machine": manifest.machine,
                "shot": manifest.shot,
                "source_kind": manifest.source.kind,
                "signals": len(manifest.signals),
            }
        )
        for uri in _covered_artifact_uris(manifest):
            resolved = _resolve_manifest_uri(uri, manifest_path, root_path)
            if resolved is not None:
                covered_artifacts.add(resolved)

    expected_set = {path.resolve() for path in expected_artifacts}
    missing = sorted(expected_set - covered_artifacts)
    coverage: dict[str, object] = report["artifact_coverage"]
    coverage["covered"] = len(expected_set & covered_artifacts)
    coverage["missing"] = [str(path) for path in missing]
    if missing:
        report["status"] = "fail"
        for path in missing:
            errors.append({"path": str(path), "error": "missing data manifest coverage"})

    if errors:
        report["status"] = "fail"
    return report


def load_acquisition_spec(path: str | Path) -> AcquisitionSpec:
    """Load an MDSplus acquisition request without importing runtime dependencies."""
    spec_path = Path(path)
    with spec_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("MDSplus acquisition request root must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("MDSplus acquisition request schema_version must be '1.0'")

    tree = _required_str(payload, "tree")
    source_uri = _required_str(payload, "source_uri")
    _required_str(payload, "access_policy")
    _required_str(payload, "licence")
    shot = payload.get("shot")
    if isinstance(shot, bool) or not isinstance(shot, int):
        raise ValueError("MDSplus acquisition request shot must be an integer")

    signals_payload = payload.get("signals")
    if not isinstance(signals_payload, list):
        raise ValueError("MDSplus acquisition request requires a signals array")
    if not signals_payload:
        raise ValueError("MDSplus acquisition requires at least one signal")
    _validate_signal_specs(signals_payload)
    return AcquisitionSpec(tree=tree, shot=shot, source_uri=source_uri, signals=len(signals_payload))


def _validate_signal_specs(signals: list[object]) -> None:
    seen: set[str] = set()
    for signal_payload in signals:
        if not isinstance(signal_payload, dict):
            raise ValueError("MDSplus signal specification must be a JSON object")
        name = _required_str(signal_payload, "name")
        node = _required_str(signal_payload, "node")
        units = _required_str(signal_payload, "units")
        timebase = _required_str(signal_payload, "timebase")
        if name in seen:
            raise ValueError(f"duplicate MDSplus signal name: {name}")
        seen.add(name)
        if not node.strip():
            raise ValueError(f"MDSplus signal {name!r} requires a node path")
        if not units.strip():
            raise ValueError(f"MDSplus signal {name!r} requires units")
        if not timebase.strip():
            raise ValueError(f"MDSplus signal {name!r} requires a timebase")


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MDSplus acquisition request requires non-empty {key}")
    return value.strip()


def _covered_artifact_uris(manifest: Any) -> list[str]:
    if manifest.synthetic:
        return []
    if manifest.artifacts:
        return [artifact.uri for artifact in manifest.artifacts]
    if manifest.source.kind in {"geqdsk", "local_archive"} and "://" not in manifest.source.uri:
        return [manifest.source.uri]
    return []


def _resolve_manifest_uri(uri: str, manifest_path: Path, root_path: Path) -> Path | None:
    candidate = Path(uri)
    if candidate.is_absolute():
        return candidate.resolve() if candidate.is_file() else None
    for parent in (Path.cwd(), root_path, *manifest_path.resolve().parents):
        resolved = parent / candidate
        if resolved.is_file():
            return resolved.resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for local and CI manifest validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(ROOT / "validation" / "reference_data"),
        help="Root directory to scan for **/manifests/*.manifest.json files",
    )
    parser.add_argument(
        "--no-verify-artifacts",
        action="store_true",
        help="Validate manifest metadata without local checksum verification",
    )
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    report = validate_manifest_directory(args.root, verify_artifacts=not args.no_verify_artifacts)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Data manifests: "
            f"{report['status']} "
            f"total={report['total']} "
            f"real={report['real']} "
            f"synthetic={report['synthetic']} "
            f"acquisition_specs={report['acquisition_specs']['total']}"
        )
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
