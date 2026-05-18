# SPDX-License-Identifier: AGPL-3.0-or-later
# ----------------------------------------------------------------------
# SCPN Control - MDSplus Acquisition
# Copyright (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ----------------------------------------------------------------------
"""Optional MDSplus shot acquisition with manifest provenance output."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

from scpn_control.core.real_data_manifest import load_real_data_manifest


@dataclass(frozen=True)
class MDSplusSignalSpec:
    """Single MDSplus signal acquisition request."""

    name: str
    node: str
    units: str
    timebase: str


@dataclass(frozen=True)
class MDSplusAcquisitionRequest:
    """Versioned MDSplus acquisition request loaded from JSON."""

    tree: str
    shot: int
    source_uri: str
    access_policy: str
    licence: str
    signals: list[MDSplusSignalSpec]


@dataclass(frozen=True)
class MDSplusAcquisitionResult:
    """Paths and manifest identity produced by an acquisition run."""

    output_npz: Path
    manifest_json: Path
    dataset_id: str
    checksum_sha256: str


def load_mdsplus_acquisition_request(path: str | Path) -> MDSplusAcquisitionRequest:
    """Load a versioned MDSplus acquisition request JSON file."""
    spec_path = Path(path)
    with spec_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("MDSplus acquisition request root must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("MDSplus acquisition request schema_version must be '1.0'")

    signals_payload = payload.get("signals")
    if not isinstance(signals_payload, list):
        raise ValueError("MDSplus acquisition request requires a signals array")
    if not signals_payload:
        raise ValueError("MDSplus acquisition requires at least one signal")
    signals = [_signal_spec_from_mapping(signal_payload) for signal_payload in signals_payload]
    _validate_signal_specs(signals)

    tree = _required_str(payload, "tree")
    shot = payload.get("shot")
    if isinstance(shot, bool) or not isinstance(shot, int):
        raise ValueError("MDSplus acquisition request shot must be an integer")
    return MDSplusAcquisitionRequest(
        tree=tree,
        shot=shot,
        source_uri=_required_str(payload, "source_uri"),
        access_policy=_required_str(payload, "access_policy"),
        licence=_required_str(payload, "licence"),
        signals=signals,
    )


def acquire_mdsplus_shot(
    *,
    tree: str,
    shot: int,
    signals: list[MDSplusSignalSpec],
    output_npz: str | Path,
    manifest_json: str | Path,
    source_uri: str,
    access_policy: str,
    licence: str,
    retrieved_at: str | None = None,
    mdsplus_module: Any | None = None,
) -> MDSplusAcquisitionResult:
    """Acquire selected signals from MDSplus, write NPZ, and emit a manifest."""
    if not signals:
        raise ValueError("MDSplus acquisition requires at least one signal")
    _validate_signal_specs(signals)

    module = mdsplus_module if mdsplus_module is not None else _import_mdsplus()
    tree_handle = module.Tree(tree, int(shot))

    arrays: dict[str, np.ndarray] = {}
    for signal in signals:
        raw = tree_handle.getNode(signal.node).data()
        array = np.asarray(raw)
        if array.size == 0:
            raise ValueError(f"MDSplus signal {signal.name!r} returned an empty array")
        if not np.all(np.isfinite(array.astype(np.float64, copy=False))):
            raise ValueError(f"MDSplus signal {signal.name!r} contains non-finite values")
        arrays[signal.name] = array

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **cast(Any, arrays))
    checksum = _sha256_file(output_path)

    manifest_path = Path(manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = f"{tree.lower()}-{int(shot)}-mdsplus"
    manifest = {
        "spdx_license_id": "AGPL-3.0-or-later",
        "copyright": "Copyright (C) 1998-2026 Miroslav Sotek. All rights reserved.",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "orcid": "https://orcid.org/0009-0009-3560-0851",
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "machine": tree,
        "shot": str(int(shot)),
        "synthetic": False,
        "source": {
            "kind": "mdsplus",
            "uri": source_uri,
            "access": access_policy,
        },
        "retrieved_at": retrieved_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "checksum_sha256": checksum,
        "licence": licence,
        "artifacts": [
            {
                "uri": str(output_path),
                "checksum_sha256": checksum,
            }
        ],
        "signals": [
            {
                "name": signal.name,
                "path": signal.node,
                "units": signal.units,
                "timebase": signal.timebase,
            }
            for signal in signals
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_real_data_manifest(manifest_path, verify_artifact=True)
    return MDSplusAcquisitionResult(
        output_npz=output_path,
        manifest_json=manifest_path,
        dataset_id=dataset_id,
        checksum_sha256=checksum,
    )


def _import_mdsplus() -> Any:
    try:
        return importlib.import_module("MDSplus")
    except ImportError as exc:
        raise RuntimeError("MDSplus is not installed; install the optional facility data client") from exc


def _validate_signal_specs(signals: list[MDSplusSignalSpec]) -> None:
    seen: set[str] = set()
    for signal in signals:
        if not signal.name.strip():
            raise ValueError("MDSplus signal name must not be empty")
        if signal.name in seen:
            raise ValueError(f"duplicate MDSplus signal name: {signal.name}")
        seen.add(signal.name)
        if not signal.node.strip():
            raise ValueError(f"MDSplus signal {signal.name!r} requires a node path")
        if not signal.units.strip():
            raise ValueError(f"MDSplus signal {signal.name!r} requires units")
        if not signal.timebase.strip():
            raise ValueError(f"MDSplus signal {signal.name!r} requires a timebase")


def _signal_spec_from_mapping(payload: object) -> MDSplusSignalSpec:
    if not isinstance(payload, dict):
        raise ValueError("MDSplus signal specification must be a JSON object")
    return MDSplusSignalSpec(
        name=_required_str(payload, "name"),
        node=_required_str(payload, "node"),
        units=_required_str(payload, "units"),
        timebase=_required_str(payload, "timebase"),
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MDSplus acquisition request requires non-empty {key}")
    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
