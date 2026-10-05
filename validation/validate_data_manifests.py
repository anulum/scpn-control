#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Data Manifest Validation Runner

"""Validate repository data manifests and local artefact checksums."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.report_output_paths import checked_report_destination, manifest_report_inputs

SRC = ROOT / "src"

_REAL_DATA_MODULE_PATH = SRC / "scpn_control" / "core" / "real_data_manifest.py"
_REAL_DATA_MODULE_NAME = "_scpn_control_real_data_manifest_contract"


def _load_real_data_manifest_api() -> tuple[type[Exception], Any, Any]:
    """Load the defining stdlib contract without importing numerical packages."""
    spec = importlib.util.spec_from_file_location(_REAL_DATA_MODULE_NAME, _REAL_DATA_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load real-data manifest contract from {_REAL_DATA_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return (
        cast(type[Exception], module.RealDataManifestError),
        module.load_real_data_manifest,
        module.resolve_manifest_artifact,
    )


RealDataManifestError, load_real_data_manifest, resolve_manifest_artifact = _load_real_data_manifest_api()


_DIIID_ARTIFACT_PATTERNS = ("*.geqdsk", "disruption_shots/*.npz")


@dataclass(frozen=True)
class AcquisitionSignal:
    """Trimmed requested signal name, MDSplus node, units and timebase."""

    name: str
    node: str
    units: str
    timebase: str


@dataclass(frozen=True)
class AcquisitionSpec:
    """Stdlib-only acquisition request summary used by the CI manifest gate."""

    tree: str
    shot: int
    source_uri: str
    signals: int
    access_policy: str = ""
    licence: str = ""
    signal_specs: tuple[AcquisitionSignal, ...] = ()

    @property
    def expected_dataset_id(self) -> str:
        """Dataset id emitted by the MDSplus acquisition command for this spec."""
        return f"{self.tree.lower()}-{self.shot}-mdsplus"


def iter_manifest_paths(root: str | Path) -> list[Path]:
    """List sorted ``**/manifests/*.manifest.json`` declarations below a root.

    Parameters
    ----------
    root : str or Path
        Glob search directory. Missing roots produce an empty list.

    Returns
    -------
    list of Path
        Lexical discovered paths, without schema or containment validation.

    Raises
    ------
    OSError, ValueError, RuntimeError
        Supported filesystem/path failures; directory validation handles these.
    """
    root_path = Path(root)
    return sorted(root_path.glob("**/manifests/*.manifest.json"))


def iter_acquisition_spec_paths(root: str | Path) -> list[Path]:
    """List sorted ``**/acquisition_specs/*.json`` request paths.

    Parameters
    ----------
    root : str or Path
        Glob directory; missing roots produce no discoveries.

    Returns
    -------
    list of Path
        Discovered spellings without request validation or acquisition.

    Raises
    ------
    OSError, ValueError, RuntimeError
        Supported filesystem/path failures.
    """
    root_path = Path(root)
    return sorted(root_path.glob("**/acquisition_specs/*.json"))


def iter_diiid_artifact_paths(root: str | Path) -> list[Path]:
    """List sorted existing DIII-D GEQDSK and disruption-shot NPZ files.

    Parameters
    ----------
    root : str or Path
        Uses its ``diiid`` child when that directory exists, otherwise the root.

    Returns
    -------
    list of Path
        Files matching ``*.geqdsk`` and ``disruption_shots/*.npz`` only. No tracked
        Git state, other machines or decoded numerical contents are inspected.

    Raises
    ------
    OSError, ValueError, RuntimeError
        Supported filesystem/path failures.
    """
    root_path = Path(root)
    diiid_root = root_path / "diiid" if (root_path / "diiid").is_dir() else root_path
    paths: list[Path] = []
    for pattern in _DIIID_ARTIFACT_PATTERNS:
        paths.extend(diiid_root.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def validate_manifest_directory(
    root: str | Path,
    *,
    verify_artifacts: bool = True,
    require_real_acquisition: bool = False,
) -> dict[str, Any]:
    """Inspect a manifest tree and return declaration/custody findings.

    Parameters
    ----------
    root : str or Path
        Existing directory. Discovered manifest/spec/required-artifact symlinks
        must remain canonically contained in this root.
    verify_artifacts : bool, default True
        Literal boolean enabling the defining loader's local SHA-256 checks.
    require_real_acquisition : bool, default False
        Literal boolean refusing pending valid acquisition specs.

    Returns
    -------
    dict
        pass/fail status, discovered counts, admitted real/synthetic declarations,
        expected DIII-D artifact coverage, valid spec linkage and ordered errors.
        Scan/policy/read/shape findings return FAIL. Empty manifest discovery ends
        before specs are decoded; invalid specs are findings, not pending entries.

    Notes
    -----
    Duplicate dataset/spec identities fail rather than silently replacing entries.
    Realised requires an MDSplus real declaration matching tree/shot/URI/access/
    licence and every requested signal's name/node/units/timebase, plus an artifact
    list. Extra declared signals are allowed. With verification disabled, realised
    remains a metadata observation without byte verification. Coverage uses the
    same contained evidence-root resolver as hashing, never process cwd. Counts
    on FAIL remain diagnostic; no field authenticates acquisition or physical truth.
    There is no atomic filesystem snapshot or artifact-format/signal-array replay.

    Examples
    --------
    From the repository root, inspect the actual synthetic reference corpus:

    >>> report = validate_manifest_directory(ROOT / "validation/reference_data")
    >>> (report["status"], report["real"], report["artifact_coverage"]["covered"])
    ('pass', 0, 21)
    """
    manifest_paths: list[Path] = []
    acquisition_spec_paths: list[Path] = []
    expected_artifacts: list[Path] = []
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
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
            "realised": 0,
            "pending": 0,
            "require_real_acquisition": bool(require_real_acquisition),
            "specs": [],
        },
        "manifests": [],
        "errors": [],
    }

    errors: list[dict[str, str]] = report["errors"]
    if not isinstance(verify_artifacts, bool) or not isinstance(require_real_acquisition, bool):
        report["status"] = "fail"
        errors.append({"path": str(root), "error": "validation policies must be booleans"})
        return report
    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise ValueError("manifest root must be a directory")
        manifest_paths = iter_manifest_paths(root_path)
        acquisition_spec_paths = iter_acquisition_spec_paths(root_path)
        expected_artifacts = iter_diiid_artifact_paths(root_path)
        for path in [*manifest_paths, *acquisition_spec_paths, *expected_artifacts]:
            path.resolve().relative_to(root_path)
    except (OSError, ValueError, RuntimeError) as exc:
        report["status"] = "fail"
        errors.append({"path": str(root), "error": f"cannot scan manifest root: {exc}"})
        return report
    report["total"] = len(manifest_paths)
    report["acquisition_specs"]["total"] = len(acquisition_spec_paths)
    report["artifact_coverage"]["expected"] = len(expected_artifacts)
    if not manifest_paths:
        report["status"] = "fail"
        report["errors"].append({"path": str(Path(root)), "error": "no data manifests found"})
        return report

    manifests: list[dict[str, object]] = report["manifests"]
    acquisition_specs = cast(dict[str, Any], report["acquisition_specs"])
    spec_entries = cast(list[dict[str, object]], acquisition_specs["specs"])
    covered_artifacts: set[Path] = set()
    spec_records: list[tuple[Path, AcquisitionSpec]] = []
    seen_spec_ids: set[str] = set()
    seen_dataset_ids: set[str] = set()
    for spec_path in acquisition_spec_paths:
        try:
            spec = load_acquisition_spec(spec_path)
            if spec.expected_dataset_id in seen_spec_ids:
                raise ValueError("duplicate acquisition dataset id")
            seen_spec_ids.add(spec.expected_dataset_id)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"path": str(spec_path), "error": str(exc)})
            continue
        acquisition_specs["mdsplus"] = cast(int, acquisition_specs["mdsplus"]) + 1
        spec_records.append((spec_path, spec))
        spec_entries.append(
            {
                "path": str(spec_path),
                "kind": "mdsplus",
                "tree": spec.tree,
                "shot": spec.shot,
                "source_uri": spec.source_uri,
                "signals": spec.signals,
                "expected_dataset_id": spec.expected_dataset_id,
                "manifest_path": None,
            }
        )

    acquired_mdsplus_manifests: dict[str, tuple[Path, Any]] = {}
    for manifest_path in manifest_paths:
        try:
            manifest = load_real_data_manifest(manifest_path, verify_artifact=verify_artifacts)
            if manifest.dataset_id in seen_dataset_ids:
                raise ValueError("duplicate manifest dataset id")
            seen_dataset_ids.add(manifest.dataset_id)
        except (OSError, ValueError, RuntimeError) as exc:
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
        if not manifest.synthetic and manifest.source.kind == "mdsplus":
            acquired_mdsplus_manifests[manifest.dataset_id] = (manifest_path, manifest)
        for uri in _covered_artifact_uris(manifest):
            resolved = _resolve_manifest_uri(uri, manifest_path, root_path)
            if resolved is not None:
                covered_artifacts.add(resolved)

    spec_by_dataset = {spec.expected_dataset_id: index for index, (_, spec) in enumerate(spec_records)}
    for dataset_id, (manifest_path, manifest) in acquired_mdsplus_manifests.items():
        spec_index = spec_by_dataset.get(dataset_id)
        if spec_index is not None:
            if _matches_acquisition(spec_records[spec_index][1], manifest):
                spec_entries[spec_index]["manifest_path"] = str(manifest_path)
            else:
                errors.append(
                    {
                        "path": str(manifest_path),
                        "error": "acquired manifest does not match acquisition spec or lacks local artifacts",
                    }
                )

    realised = sum(1 for entry in spec_entries if entry.get("manifest_path") is not None)
    pending = len(spec_entries) - realised
    acquisition_specs["realised"] = realised
    acquisition_specs["pending"] = pending
    if require_real_acquisition:
        for spec_path, spec in spec_records:
            if spec_entries[spec_by_dataset[spec.expected_dataset_id]]["manifest_path"] is None:
                errors.append(
                    {
                        "path": str(spec_path),
                        "error": "missing acquired MDSplus manifest",
                    }
                )

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
    """Load a stdlib-only schema 1.0 request with finite, unique-key JSON.

    Parameters
    ----------
    path : str or Path
        UTF-8 JSON file; no acquisition or numerical dependency is imported.

    Returns
    -------
    AcquisitionSpec
        Trimmed tree/URI/access/licence, nonboolean integer shot and unique signal
        records. ``signals`` is the number of requested records.

    Raises
    ------
    ValueError, OSError, RuntimeError
        Invalid JSON/shape/nonfinite tokens, file/path/UTF-8 or decoding depth error.
        Directory validation translates these failures into report findings.
    """
    spec_path = Path(path)
    with spec_path.open(encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
            parse_float=_finite_json_float,
        )
    if not isinstance(payload, dict):
        raise ValueError("MDSplus acquisition request root must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("MDSplus acquisition request schema_version must be '1.0'")

    tree = _required_str(payload, "tree")
    source_uri = _required_str(payload, "source_uri")
    access_policy = _required_str(payload, "access_policy")
    licence = _required_str(payload, "licence")
    shot = payload.get("shot")
    if isinstance(shot, bool) or not isinstance(shot, int):
        raise ValueError("MDSplus acquisition request shot must be an integer")

    signals_payload = payload.get("signals")
    if not isinstance(signals_payload, list):
        raise ValueError("MDSplus acquisition request requires a signals array")
    if not signals_payload:
        raise ValueError("MDSplus acquisition requires at least one signal")
    signal_specs = _validate_signal_specs(signals_payload)
    return AcquisitionSpec(
        tree=tree,
        shot=shot,
        source_uri=source_uri,
        signals=len(signal_specs),
        access_policy=access_policy,
        licence=licence,
        signal_specs=signal_specs,
    )


def _reject_nonfinite_json(token: str) -> Any:
    """Refuse NaN and infinity constants anywhere in acquisition JSON."""
    raise ValueError(f"nonfinite JSON value: {token}")


def _finite_json_float(token: str) -> float:
    """Parse a float while refusing exponent overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON value: {token}")
    return value


def _matches_acquisition(spec: AcquisitionSpec, manifest: Any) -> bool:
    """Match exact provenance and requested signal subset, requiring local artifacts."""
    declared = {(s.name, s.path, s.units, s.timebase) for s in manifest.signals}
    requested = {(s.name, s.node, s.units, s.timebase) for s in spec.signal_specs}
    return bool(manifest.artifacts) and (
        manifest.machine == spec.tree
        and manifest.shot == str(spec.shot)
        and manifest.source.uri == spec.source_uri
        and manifest.source.access == spec.access_policy
        and manifest.licence == spec.licence
        and requested <= declared
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate acquisition-spec keys."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _validate_signal_specs(signals: list[object]) -> tuple[AcquisitionSignal, ...]:
    """Parse required signal strings and reject duplicate trimmed names."""
    seen: set[str] = set()
    records: list[AcquisitionSignal] = []
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
        records.append(AcquisitionSignal(name, node, units, timebase))
    return tuple(records)


def _required_str(payload: dict[str, Any], key: str) -> str:
    """Return a required trimmed string or report an acquisition shape error."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MDSplus acquisition request requires non-empty {key}")
    return value.strip()


def _covered_artifact_uris(manifest: Any) -> list[str]:
    """List local artifact references supported by declared checksum custody."""
    if manifest.artifacts:
        return [artifact.uri for artifact in manifest.artifacts]
    if manifest.synthetic and manifest.checksum_sha256 is not None and "://" not in manifest.source.uri:
        return [manifest.source.uri]
    if manifest.source.kind in {"geqdsk", "local_archive"} and "://" not in manifest.source.uri:
        return [manifest.source.uri]
    return []


def _resolve_manifest_uri(uri: str, manifest_path: Path, root_path: Path) -> Path | None:
    """Use the checksum resolver; unresolved references supply no coverage.

    ``root_path`` remains for call compatibility. The manifest evidence-tree
    policy owns lookup, never absolute paths or the process working directory.
    """
    try:
        return cast(Path, resolve_manifest_artifact(uri, manifest_path=manifest_path))
    except RealDataManifestError:
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the standalone manifest directory report CLI.

    Parameters
    ----------
    argv : list of str or None
        argparse tokens; None reads process arguments. Default root is the actual
        repository reference-data tree, independent of cwd.

    Returns
    -------
    int
        0 for PASS, 1 for validation or supported JSON-output failures. Selected
        manifests/specifications/local artifacts and the root cannot be output
        destinations, including existing hard links. JSON mode
        emits the report; text mode prints summary plus ordered stderr findings.

    Raises
    ------
    SystemExit
        argparse help or argument refusal. A failed write may leave partial output
        and does not establish report custody.
    """
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
    parser.add_argument(
        "--require-real-acquisition",
        action="store_true",
        help="Fail when an acquisition spec has no corresponding acquired MDSplus manifest",
    )
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    report = validate_manifest_directory(
        args.root,
        verify_artifacts=not args.no_verify_artifacts,
        require_real_acquisition=args.require_real_acquisition,
    )
    if args.output_json:
        try:
            output_path = checked_report_destination(args.output_json, inputs=manifest_report_inputs(args.root))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except (OSError, ValueError, RuntimeError) as exc:
            report["status"] = "fail"
            report["errors"].append({"path": args.output_json, "error": f"cannot write report: {exc}"})
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
