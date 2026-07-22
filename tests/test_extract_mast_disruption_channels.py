# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST disruption binding-readiness gate
"""Production-boundary tests for the FAIR-MAST binding-readiness gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.acquire_mast_disruption_shots import acquire
from validation.extract_mast_disruption_channels import (
    BINDING_READINESS_SCHEMA,
    PHYSICAL_BINDING_BLOCKERS,
    TRANSPORT_RESOLVABLE_KEYS,
    assess_artifact_binding_readiness,
    inspect_manifest_binding_readiness,
    main,
)
from validation.mast_source_artifact_reader import load_verified_source_manifest, read_verified_npz_artifact
from validation.mast_source_object_manifest import canonical_json_sha256

_SHOT_ID = 30421
_FIXED_TS = "2026-07-22T00:00:00Z"


class _SourceArray:
    def __init__(self, values: NDArray[np.float64], *, dimensions: tuple[str, ...], units: str | None) -> None:
        self.values = values
        self.dims = dimensions
        self.attrs = {} if units is None else {"units": units}
        self.chunks: None = None


class _SourceGroup:
    def __init__(self, variables: dict[str, _SourceArray]) -> None:
        self.variables = variables

    def __getitem__(self, key: str) -> _SourceArray:
        return self.variables[key]


def _source_group(group: str, *, omit_density: bool = False) -> _SourceGroup:
    summary = {
        "time": _SourceArray(np.linspace(0.0, 0.05, 32, dtype=np.float64), dimensions=("time",), units="s"),
        "ip": _SourceArray(np.linspace(5.0e5, 4.0e5, 32, dtype=np.float64), dimensions=("time",), units="A"),
        "line_average_n_e": _SourceArray(np.full(32, 3.0e19), dimensions=("time",), units="1 / m ** 3"),
    }
    if omit_density:
        del summary["line_average_n_e"]
    groups = {
        "summary": summary,
        "equilibrium": {
            "time": _SourceArray(np.linspace(0.0, 0.05, 16, dtype=np.float64), dimensions=("time",), units="s"),
            "q95": _SourceArray(np.full(16, 3.8), dimensions=("time",), units=""),
            "beta_tor_normal": _SourceArray(np.full(16, 1.5), dimensions=("time",), units="T"),
            "magnetic_axis_z": _SourceArray(np.zeros(16), dimensions=("time",), units="m"),
            "z": _SourceArray(np.zeros((16, 9)), dimensions=("time", "profile"), units="m"),
        },
        "interferometer": {},
        "magnetics": {
            "time_saddle": _SourceArray(
                np.linspace(0.0, 0.05, 64, dtype=np.float64), dimensions=("time_saddle",), units="s"
            ),
            "b_field_tor_probe_saddle_field": _SourceArray(
                np.ones((12, 64)), dimensions=("channel", "time_saddle"), units="T"
            ),
            "b_field_tor_probe_saddle_m_phi": _SourceArray(
                np.tile(np.arange(12, dtype=np.float64).reshape(-1, 1), (1, 64)),
                dimensions=("channel", "time_saddle"),
                units="degree",
            ),
            "b_field_pol_probe_cc_field": _SourceArray(
                np.ones((24, 64)), dimensions=("channel", "time_mirnov"), units="T"
            ),
        },
    }
    return _SourceGroup(groups[group])


def _acquire_to_disk(root: Path, *, with_failed_shot: bool = False, omit_density: bool = False) -> Path:
    def open_group(_fs: Any, shot_id: int, group: str) -> _SourceGroup:
        if with_failed_shot and shot_id == _SHOT_ID + 1:
            raise KeyError("shot unavailable")
        return _source_group(group, omit_density=omit_density)

    shot_ids = [_SHOT_ID, _SHOT_ID + 1] if with_failed_shot else [_SHOT_ID]
    material_root = root / "material"
    manifest = acquire(
        shot_ids,
        out_dir=material_root,
        cache_dir=root / "cache",
        generated_at=_FIXED_TS,
        retrieved_at=_FIXED_TS,
        make_fs=lambda _path: object(),
        open_group=open_group,
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_same_schema_acquisition_to_verified_npz_to_binding_gate(tmp_path: Path) -> None:
    """Acquisition output reaches the gate through its unchanged v2 schema."""
    manifest_path = _acquire_to_disk(tmp_path)
    material_root = tmp_path / "material"

    report = inspect_manifest_binding_readiness(manifest_path, artifact_root=material_root)

    assert report["schema_version"] == BINDING_READINESS_SCHEMA
    assert report["status"] == "blocked"
    assert report["channel_extraction_admissible"] is False
    assert report["n_transport_verified"] == 1
    shot = report["shots"][0]
    assert {item["semantic"] for item in shot["resolved"]} == set(TRANSPORT_RESOLVABLE_KEYS)
    assert {item["semantic"] for item in shot["unresolved"]} == set(PHYSICAL_BINDING_BLOCKERS)
    assert shot["unresolved"] == sorted(shot["unresolved"], key=lambda item: item["semantic"])
    assert "summary.ip" in shot["archive_keys"]
    assert "ip" not in shot["archive_keys"]
    binding = shot["signal_binding_assessment"]
    assert binding["binding_contract_complete"] is True
    assert binding["channel_extraction_admissible"] is False
    assert binding["n_source_metadata_verified"] == 5
    assert binding["n_blocked"] == 6
    assert report["shots"][0]["blocking_contracts"][1] == {
        "contract": "mast_signal_binding_spec",
        "reason_code": "binding_spec_contains_explicit_source_semantic_or_metadata_blockers",
    }
    assert report["payload_sha256"] == canonical_json_sha256({**report, "payload_sha256": None})


def test_readiness_reports_absent_transport_key_without_aliasing(tmp_path: Path) -> None:
    """A missing exact source key stays absent instead of receiving an alias."""
    manifest_path = _acquire_to_disk(tmp_path, omit_density=True)
    manifest = load_verified_source_manifest(manifest_path, artifact_root=tmp_path / "material")
    artifact = read_verified_npz_artifact(manifest, artifact_root=tmp_path / "material", shot_id=_SHOT_ID)

    report = assess_artifact_binding_readiness(artifact)

    density = next(item for item in report["unresolved"] if item["semantic"] == "ne_per_m3")
    assert density["status"] == "source_key_absent"
    assert density["required_source_keys"] == ["summary.line_average_n_e"]
    assert {item["semantic"] for item in report["resolved"]} == set(TRANSPORT_RESOLVABLE_KEYS) - {"ne_per_m3"}


def test_campaign_report_preserves_failed_shot_boundary(tmp_path: Path) -> None:
    """Unavailable acquisition records remain explicit campaign results."""
    manifest_path = _acquire_to_disk(tmp_path, with_failed_shot=True)

    report = inspect_manifest_binding_readiness(manifest_path, artifact_root=tmp_path / "material")

    assert report["n_requested"] == 2
    assert report["n_transport_verified"] == 1
    assert report["n_not_acquired"] == 1
    failed = next(item for item in report["shots"] if item["status"] == "not_acquired")
    assert failed["shot_id"] == _SHOT_ID + 1
    assert "shot unavailable" in failed["reason"]


def test_readiness_is_deterministic_and_lineage_bound(tmp_path: Path) -> None:
    """Repeated inspection preserves the manifest-bound artefact identity."""
    manifest_path = _acquire_to_disk(tmp_path)
    first = inspect_manifest_binding_readiness(manifest_path, artifact_root=tmp_path / "material")
    second = inspect_manifest_binding_readiness(manifest_path, artifact_root=tmp_path / "material")
    assert first == second
    assert (
        first["shots"][0]["artifact_sha256"]
        == json.loads(manifest_path.read_text())["shots"][0]["artifacts"][0]["sha256"]
    )


def test_cli_writes_blocked_readiness_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI persists a successful transport proof without opening admission."""
    manifest_path = _acquire_to_disk(tmp_path)
    json_out = tmp_path / "reports" / "readiness.json"

    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(tmp_path / "material"),
            "--json-out",
            str(json_out),
        ]
    )

    assert code == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["status"] == "blocked"
    assert "1 verified, 0 unavailable" in capsys.readouterr().out


def test_tampered_npz_fails_before_binding_assessment(tmp_path: Path) -> None:
    """Byte tampering is rejected before semantic readiness is assessed."""
    manifest_path = _acquire_to_disk(tmp_path)
    npz_path = tmp_path / "material" / f"shot_{_SHOT_ID}.npz"
    npz_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verification failed"):
        inspect_manifest_binding_readiness(manifest_path, artifact_root=tmp_path / "material")
