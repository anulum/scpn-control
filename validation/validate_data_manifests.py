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
import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_control.core.mdsplus_acquisition import load_mdsplus_acquisition_request
from scpn_control.core.real_data_manifest import RealDataManifestError, load_real_data_manifest


_DIIID_ARTIFACT_PATTERNS = ("*.geqdsk", "disruption_shots/*.npz")


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
            spec = load_mdsplus_acquisition_request(spec_path)
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
                "signals": len(spec.signals),
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
