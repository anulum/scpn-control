# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM feature-provenance audit tests

"""Exercise actual public audit/file/CLI behavior with retained engineering NPZ formats, never authentic MAST measurements."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

from validation.audit_mast_efm_feature_provenance import (
    AUDIT_SCHEMA,
    build_audit,
    main,
    parse_args,
    validate_audit_report,
    write_report,
)
from validation.build_mast_efm_neural_equilibrium_dataset import DatasetInput, build_dataset
from validation.build_mast_efm_neural_equilibrium_dataset import write_report as write_dataset_report
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import CANDIDATE_SCHEMA


def _write_reference(path: Path, *, include_ip: bool = False, include_ffprime_rms: bool = False) -> None:
    """Preserve original numerical reference arrays while supplying required converter identity and grids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "shot_id": np.array([1]),
        "r_grid_m": np.array([0.4, 0.7, 1.0]),
        "z_grid_m": np.array([-0.2, 0.2]),
        "psi_axis_Wb_per_rad": np.array([0.1]),
        "psi_boundary_Wb_per_rad": np.array([1.1]),
        "time_s": np.array([0.1]),
        "psirz_Wb_per_rad": np.zeros((1, 2, 3)),
        "psirz_valid_mask": np.ones((1, 2, 3), dtype=bool),
        "pprime_Pa_per_Wb_rad": np.ones((1, 2)),
        "pprime_valid_mask": np.ones((1, 2), dtype=bool),
        "q_profile": np.ones((1, 2)),
        "q_profile_valid_mask": np.ones((1, 2), dtype=bool),
        "lcfs_r_m": np.ones((1, 4)),
        "lcfs_z_m": np.ones((1, 4)),
        "lcfs_valid_mask": np.ones((1, 4), dtype=bool),
        "magnetic_axis_r_m": np.array([0.7]),
        "magnetic_axis_z_m": np.array([0.0]),
    }
    if include_ip:
        payload["Ip_MA"] = np.array([1.0])
    if include_ffprime_rms:
        payload["ffprime_rms_T_rad"] = np.array([0.5])
    np.savez_compressed(path, **payload)


def _write_dataset_report(path: Path, references: list[str]) -> None:
    """Run the actual public dataset producer on three explicit engineering shot partitions.

    The historical first reference arrays stay unchanged. Two separate reference
    files supply validation/test partitions; this supplies required declarations,
    not independent physical measurements.
    """
    storage = path.parent / "storage"
    first = storage / references[0]
    with np.load(first, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    selected = [first]
    for shot_id in (2, 3):
        ref = storage / f"converted/shot_{shot_id}.npz"
        np.savez_compressed(ref, **{**data, "shot_id": np.array([shot_id])})
        selected.append(ref)
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "pass",
        "source": "documented_public_reference",
        "fixture_kind": "engineering_schema_contract",
        "reference_dataset_id": "mast-efm-test",
        "reference_equilibria_count": 3,
        "target_schema_status": "reference_only_no_prediction_metrics",
        "admission_ready": False,
        "errors": [],
        "shots": [
            {
                "shot_id": shot,
                "output_path": ref.as_posix(),
                "source_path": "engineering_schema_contract",
                "sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
                "selected_time_count": 1,
                "grid_shape": [2, 3],
                "lcfs_points": 4,
                "status": "reference_candidate",
            }
            for shot, ref in enumerate(selected, start=1)
        ],
    }
    candidate["payload_sha256"] = canonical_campaign_digest(candidate)
    cp = storage / "converted/candidate.json"
    cp.write_text(json.dumps(candidate))
    report = build_dataset(DatasetInput(cp, storage, storage / "processed/dataset.npz", (1,), (2,), (3,)))
    write_dataset_report(report, path, path.with_suffix(".md"))


def _fixture(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Path]:
    """Prepare actual producer JSON and selected local NPZ files for public source-audit tests."""
    storage = tmp_path / "storage"
    ref = storage / "converted/reference.npz"
    _write_reference(ref, include_ip=complete, include_ffprime_rms=complete)
    if complete:
        with np.load(ref, allow_pickle=False) as payload:
            data = {key: payload[key] for key in payload.files}
        np.savez_compressed(ref, **{**data, "Bt_T": np.array([0.5])})
    rp = tmp_path / "dataset.json"
    _write_dataset_report(rp, ["converted/reference.npz"])
    return rp, storage


def _replace_reference(report_path: Path, storage: Path, data: dict[str, Any], *, shot: int = 0) -> None:
    """Write real changed reference bytes and bind their declared SHA for independent admission checks."""
    report = json.loads(report_path.read_text())
    declaration = report["shots"][shot]
    ref = storage / declaration["reference_path"]
    np.savez_compressed(ref, **data)
    declaration["reference_sha256"] = hashlib.sha256(ref.read_bytes()).hexdigest()
    report["payload_sha256"] = canonical_campaign_digest(report)
    report_path.write_text(json.dumps(report))


def _reference_data(report_path: Path, storage: Path, *, shot: int = 0) -> dict[str, Any]:
    """Decode actual retained test input bytes without accessing any production-private helper."""
    report = json.loads(report_path.read_text())
    with np.load(storage / report["shots"][shot]["reference_path"], allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def test_feature_provenance_audit_blocks_unresolved_fallbacks(tmp_path: Path) -> None:
    """Missing canonical source channels remain blocked through the actual producer and auditor."""
    storage_root = tmp_path / "storage"
    reference = storage_root / "converted/reference.npz"
    _write_reference(reference)
    report = tmp_path / "dataset.json"
    _write_dataset_report(report, ["converted/reference.npz"])

    audit = build_audit(report, storage_root)

    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["status"] == "blocked"
    assert set(audit["blocked_features"]) == {"Ip_MA", "Bt_T", "ffprime_scale"}
    assert audit["feature_status"]["Ip_MA"]["present_keys"] == []


def test_feature_provenance_audit_records_resolved_direct_key(tmp_path: Path) -> None:
    """The original finite current vector resolves only when every selected shot carries it."""
    storage_root = tmp_path / "storage"
    reference = storage_root / "converted/reference.npz"
    _write_reference(reference, include_ip=True)
    report = tmp_path / "dataset.json"
    _write_dataset_report(report, ["converted/reference.npz"])

    audit = build_audit(report, storage_root)

    assert audit["feature_status"]["Ip_MA"]["status"] == "resolved"
    assert audit["feature_status"]["Ip_MA"]["present_keys"] == ["Ip_MA"]
    assert "Bt_T" in audit["blocked_features"]


def test_feature_provenance_audit_records_converted_ffprime_rms_key(tmp_path: Path) -> None:
    """The original positive RMS vector resolves across every selected reference."""
    storage_root = tmp_path / "storage"
    reference = storage_root / "converted/reference.npz"
    _write_reference(reference, include_ffprime_rms=True)
    report = tmp_path / "dataset.json"
    _write_dataset_report(report, ["converted/reference.npz"])

    audit = build_audit(report, storage_root)

    assert audit["feature_status"]["ffprime_scale"]["status"] == "resolved"
    assert audit["feature_status"]["ffprime_scale"]["present_keys"] == ["ffprime_rms_T_rad"]


def test_write_report_lists_available_keys_and_next_steps(tmp_path: Path) -> None:
    """Real validated audit declarations reach both JSON and Markdown public writer outputs."""
    report, storage = _fixture(tmp_path, complete=False)
    audit = build_audit(report, storage)

    write_report(audit, tmp_path / "audit.json", tmp_path / "audit.md")

    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "MAST EFM Feature-Provenance Audit" in markdown
    assert "`Ip_MA`" in markdown
    assert "inspect original metadata" in markdown


def _rebuild_dataset(report_path: Path, storage: Path) -> None:
    """Regenerate the actual producer output after deliberately changing engineering source files."""
    cp = storage / "converted/candidate.json"
    candidate = json.loads(cp.read_text())
    for shot in candidate["shots"]:
        shot["sha256"] = hashlib.sha256(Path(shot["output_path"]).read_bytes()).hexdigest()
    candidate["payload_sha256"] = canonical_campaign_digest(candidate)
    cp.write_text(json.dumps(candidate))
    report = build_dataset(DatasetInput(cp, storage, storage / "processed/dataset.npz", (1,), (2,), (3,)))
    write_dataset_report(report, report_path, report_path.with_suffix(".md"))


def test_complete_sources_and_changed_container_custody(tmp_path: Path) -> None:
    """All-shot canonical vectors pass; identical arrays in changed ZIP bytes still require a new declaration."""
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    assert audit["status"] == "pass" and audit["reference_count"] == 3
    assert audit["feature_status"]["Ip_MA"]["complete_reference_count"] == 3
    ref = storage / audit["shots"][0]["reference_path"]
    original = _reference_data(rp, storage)
    with ZipFile(ref, "a") as archive:
        archive.comment = b"real changed archive container with identical arrays"
    with np.load(ref, allow_pickle=False) as changed:
        assert all(np.array_equal(original[key], changed[key]) for key in original)
    with pytest.raises(ValueError, match="SHA-256"):
        build_audit(rp, storage)
    _rebuild_dataset(rp, storage)
    assert build_audit(rp, storage)["status"] == "pass"


@pytest.mark.parametrize("feature", ["Ip_MA", "Bt_T", "ffprime_scale"])
def test_partial_shot_and_alias_inventory_never_resolve_all_shots(tmp_path: Path, feature: str) -> None:
    """A real producer fallback on one shot remains blocked even while other shots or aliases expose keys."""
    from validation.neural_equilibrium_dataset_contracts import FEATURE_SOURCE_POLICY

    rp, storage = _fixture(tmp_path)
    data = _reference_data(rp, storage)
    source_key = FEATURE_SOURCE_POLICY[feature]["source_key"]
    value = data.pop(source_key)
    alias = {"Ip_MA": "current_A", "Bt_T": "bcentr", "ffprime_scale": "fpol"}[feature]
    data[alias] = value
    _replace_reference(rp, storage, data)
    _rebuild_dataset(rp, storage)
    report = json.loads(rp.read_text())
    audit = build_audit(rp, storage)
    assert report["fallback_features"] == [feature]
    assert audit["status"] == "blocked" and audit["blocked_features"] == [feature]
    assert audit["feature_status"][feature]["complete_reference_count"] == 2
    assert alias in audit["feature_status"][feature]["present_keys"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("Ip_MA", np.array([True])),
        ("Ip_MA", np.array(["1.0"])),
        ("Bt_T", np.array([1j])),
        ("Bt_T", np.array([1.0, 2.0])),
        ("Ip_MA", np.array([np.nan])),
        ("Bt_T", np.array([np.inf])),
        ("ffprime_rms_T_rad", np.array([0.0])),
        ("ffprime_rms_T_rad", np.array([-1.0])),
        ("shot_id", np.array([True])),
        ("shot_id", np.array([2])),
        ("shot_id", np.array([1, 1])),
        ("time_s", np.array([-0.1])),
        ("time_s", np.array([0.2])),
        ("time_s", np.array([np.nan])),
        ("r_grid_m", np.array([0.4, 0.4, 1.0])),
        ("r_grid_m", np.array([0.3, 0.7, 1.0])),
        ("r_grid_m", np.array([0.4, 0.8, 1.0])),
        ("r_grid_m", np.array([0.4, 1.0])),
    ],
)
def test_present_reference_channels_must_match_real_values_and_metadata(tmp_path: Path, key: str, value: Any) -> None:
    """Actual changed NPZ bytes with valid SHA cannot bypass source shape, dtype, numeric or identity contracts."""
    rp, storage = _fixture(tmp_path)
    data = _reference_data(rp, storage)
    data[key] = value
    _replace_reference(rp, storage, data)
    with pytest.raises(ValueError):
        build_audit(rp, storage)


def test_descending_source_grids_and_missing_required_identity(tmp_path: Path) -> None:
    """Descending coordinates retain the producer's accepted binding; missing shot identity refuses."""
    rp, storage = _fixture(tmp_path)
    data = _reference_data(rp, storage)
    data["r_grid_m"] = data["r_grid_m"][::-1]
    data["z_grid_m"] = data["z_grid_m"][::-1]
    _replace_reference(rp, storage, data)
    assert build_audit(rp, storage)["status"] == "pass"
    del data["shot_id"]
    _replace_reference(rp, storage, data)
    with pytest.raises(ValueError, match="shot_id"):
        build_audit(rp, storage)


@pytest.mark.parametrize(
    "contents",
    ["[]", "{}", '{"schema_version":1,"schema_version":2}', '{"value":NaN}', '{"value":1e9999}', "invalid", "\ufffd"],
)
def test_dataset_declaration_decode_and_contract_refusals(tmp_path: Path, contents: str) -> None:
    """Finite duplicate-free producer metadata is required before any reference inspection."""
    rp = tmp_path / "bad.json"
    rp.write_text(contents)
    with pytest.raises(ValueError):
        build_audit(rp, tmp_path / "storage")


def test_dataset_fallback_declaration_and_symlink_escape_refuse(tmp_path: Path) -> None:
    """Resealed metadata cannot disagree with actual sourced channels or escape selected storage."""
    rp, storage = _fixture(tmp_path)
    report = json.loads(rp.read_text())
    report["fallback_features"] = ["Ip_MA"]
    del report["feature_source_policy"]["Ip_MA"]
    report["payload_sha256"] = canonical_campaign_digest(report)
    rp.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fallback_features disagree"):
        build_audit(rp, storage)
    ref = storage / report["reference_paths"][0]
    outside = tmp_path / "outside.npz"
    outside.write_bytes(ref.read_bytes())
    ref.unlink()
    ref.symlink_to(outside)
    with pytest.raises(ValueError, match="within storage_root"):
        build_audit(rp, storage)


def test_stale_writer_and_output_aliases_preserve_selected_inputs(tmp_path: Path) -> None:
    """The real writer refuses stale content and path/hardlink aliases before changing outputs or inputs."""
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    stale = {**audit, "reference_dataset_id": "changed"}
    json_out, md_out = tmp_path / "audit.json", tmp_path / "audit.md"
    with pytest.raises(ValueError, match="payload_sha256"):
        write_report(stale, json_out, md_out)
    assert not json_out.exists() and not md_out.exists()
    ref = storage / audit["shots"][0]["reference_path"]
    before = hashlib.sha256(ref.read_bytes()).hexdigest()
    alias = tmp_path / "hardlink.json"
    os.link(ref, alias)
    for output in (rp, ref, alias, storage / audit["dataset_path"], storage / audit["candidate_report"]):
        with pytest.raises(ValueError, match="distinct"):
            write_report(audit, output, md_out)
    with pytest.raises(ValueError, match="distinct"):
        write_report(audit, json_out, json_out)
    assert hashlib.sha256(ref.read_bytes()).hexdigest() == before and not md_out.exists()


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "status",
        "reference_count",
        "reference_dataset_id",
        "dataset_report_sha256",
        "fallback_features",
        "feature_status",
        "blocked_features",
        "all_reference_keys",
        "next_processing_steps",
    ],
)
def test_resealed_writer_declarations_cannot_bypass_semantic_bindings(tmp_path: Path, field: str) -> None:
    """A matching consistency digest cannot admit malformed or promoted report declarations to persistence."""
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    audit[field] = None
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    with pytest.raises(ValueError):
        write_report(audit, tmp_path / "bad.json", tmp_path / "bad.md")
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "shot_id",
        "reference_path",
        "reference_sha256",
        "equilibria_count",
        "keys",
        "key_count",
        "shapes",
        "sourced_features",
    ],
)
def test_resealed_writer_shot_declarations_refuse(tmp_path: Path, field: str) -> None:
    """Per-shot provenance declarations are admitted before aggregate rendering and output persistence."""
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    audit["shots"][0][field] = None
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    with pytest.raises(ValueError):
        validate_audit_report(audit)


def test_writer_second_file_io_failure_is_authored_and_sequential(tmp_path: Path) -> None:
    """A real second-output filesystem failure raises ValueError while exposing the documented first-file residue."""
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    md = tmp_path / "blocked.md"
    md.mkdir()
    output = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="cannot write"):
        write_report(audit, output, md)
    assert json.loads(output.read_text()) == audit and md.is_dir()


@pytest.mark.parametrize(
    "kind",
    [
        "shots_type",
        "empty_shots",
        "shot_type",
        "unsafe_reference",
        "duplicate_id",
        "duplicate_reference",
        "unsorted_keys",
        "wrong_key_count",
        "bad_shape_type",
        "negative_shape",
        "boolean_shape",
        "wrong_source_shape",
        "omitted_source",
        "empty_steps",
        "duplicate_steps",
    ],
)
def test_public_writer_refuses_inconsistent_shot_inventory_before_io(tmp_path: Path, kind: str) -> None:
    """Real writer submissions require portable unique shot bindings and canonical source/inventory shapes."""
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    first = audit["shots"][0]
    if kind == "shots_type":
        audit["shots"] = {}
    elif kind == "empty_shots":
        audit["shots"] = []
    elif kind == "shot_type":
        audit["shots"][0] = []
    elif kind == "unsafe_reference":
        first["reference_path"] = "../outside.npz"
    elif kind == "duplicate_id":
        audit["shots"][1]["shot_id"] = first["shot_id"]
    elif kind == "duplicate_reference":
        audit["shots"][1]["reference_path"] = first["reference_path"]
    elif kind == "unsorted_keys":
        first["keys"] = first["keys"][::-1]
    elif kind == "wrong_key_count":
        first["key_count"] += 1
    elif kind == "bad_shape_type":
        first["shapes"]["Ip_MA"] = 1
    elif kind == "negative_shape":
        first["shapes"]["Ip_MA"] = [-1]
    elif kind == "boolean_shape":
        first["shapes"]["Ip_MA"] = [True]
    elif kind == "wrong_source_shape":
        first["shapes"]["Ip_MA"] = [2]
    elif kind == "omitted_source":
        first["sourced_features"] = []
    elif kind == "empty_steps":
        audit["next_processing_steps"] = []
    else:
        audit["next_processing_steps"] *= 2
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_report(audit, output, tmp_path / "invalid.md")
    assert not output.exists()


def test_public_validator_refuses_nonobject_and_nonfinite_payloads(tmp_path: Path) -> None:
    """Public declaration admission rejects a nonobject and nonfinite extension before digest or rendering."""
    bad: Any = []
    with pytest.raises(ValueError, match="JSON object"):
        validate_audit_report(bad)
    rp, storage = _fixture(tmp_path)
    audit = build_audit(rp, storage)
    audit["extension"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        write_report(audit, tmp_path / "invalid.json", tmp_path / "invalid.md")


def test_explicit_cli_and_foreign_cwd_domain_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Actual CLI success/blocked/help/usage and missing-source failure exercise authored exit behavior from foreign cwd."""
    rp, storage = _fixture(tmp_path, complete=False)
    args = [
        "--dataset-report",
        str(rp),
        "--storage-root",
        str(storage),
        "--json-out",
        str(tmp_path / "cli.json"),
        "--report-out",
        str(tmp_path / "cli.md"),
    ]
    assert main(args) == 0 and "blocked" in capsys.readouterr().out
    assert parse_args(args).storage_root == storage
    with pytest.raises(SystemExit) as help_exit:
        parse_args(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as usage_exit:
        parse_args(["--unknown"])
    assert usage_exit.value.code == 2
    args[3] = str(tmp_path / "absent-storage")
    assert main(args) == 1 and "FAIL:" in capsys.readouterr().err
    source = Path(__file__).resolve().parents[1] / "validation/audit_mast_efm_feature_provenance.py"
    for selected, code in ((args, 1), (["--help"], 0), (["--unknown"], 2)):
        proc = subprocess.run(
            [sys.executable, str(source), *selected],
            cwd=tmp_path,
            env={
                **{k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "MYPYPATH"}},
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == code and "Traceback" not in proc.stderr
