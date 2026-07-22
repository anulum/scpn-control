# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for producer-time FAIR-MAST dataset lineage
"""Real-file contract tests for producer-time dataset lineage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import validation.build_mast_lineage_bound_dataset as producer
from scpn_control.core.real_data_manifest import load_real_data_manifest
from validation.build_disruption_replay_channels import REPORT_SCHEMA, inspect_replay_archive
from validation.build_mast_disruption_dataset import DATASET_SCHEMA, MEASURED_CHANNELS
from validation.build_mast_lineage_bound_dataset import (
    BASE_BLOCKERS,
    CLAIM_FIELDS,
    LINEAGE_MODE,
    PRODUCER_LINEAGE_SCHEMA,
    DatasetProducerLineageError,
    build_lineage_bound_dataset,
    main,
    validate_producer_lineage_manifest,
    validate_transform_spec,
)
from validation.mast_source_object_manifest import (
    SOURCE_GENERATION_DIGEST_KIND,
    SOURCE_GENERATION_SCHEMA,
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    canonical_json_sha256,
    finalise_source_object_manifest,
)

_FIXED_TS = "2026-07-22T21:30:00Z"
_TRANSFORM_PATH = Path(__file__).parents[1] / "validation/mast_dataset_producer_transform_spec_v1.json"


def _channels(ip: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    n_samples = int(ip.shape[0])
    channels = {name: np.linspace(0.0, 1.0, n_samples, dtype=np.float64) for name in MEASURED_CHANNELS}
    channels["time_s"] = np.arange(n_samples, dtype=np.float64) * 1.0e-3
    channels["Ip_MA"] = ip
    return channels


def _disruptive_ip() -> NDArray[np.float64]:
    ip = np.ones(300, dtype=np.float64)
    ip[:100] = np.linspace(0.0, 1.0, 100)
    ip[250] = 0.5
    ip[251:] = 0.0
    return ip


def _source_generation(shot_id: int) -> dict[str, Any]:
    uri = f"s3://mast/level2/shots/{shot_id}.zarr"
    return {
        "schema_version": SOURCE_GENERATION_SCHEMA,
        "digest_kind": SOURCE_GENERATION_DIGEST_KIND,
        "source_uri": uri,
        "metadata_path": "zarr.json",
        "sha256": f"{shot_id:064x}"[-64:],
        "bytes": 42,
        "zarr_format": 3,
        "consolidated_metadata_kind": "inline",
        "etag": f'"etag-{shot_id}"',
        "last_modified": "Wed, 22 Jul 2026 20:00:00 GMT",
    }


def _write_source_manifest(root: Path, *, pin_generation: bool) -> Path:
    root.mkdir(parents=True)
    shots: list[dict[str, Any]] = []
    total_bytes = 0
    for shot_id, programme_class in ((101, "forced_vde"), (102, "unknown")):
        arrays = {"summary.time": np.arange(3, dtype=np.float64)}
        artifact_path = root / f"shot_{shot_id}.npz"
        np.savez_compressed(artifact_path, **arrays)  # type: ignore[arg-type]
        artifact = build_derived_npz_artifact(
            local_path=artifact_path.name,
            artifact_path=artifact_path,
            source_uri=f"s3://mast/level2/shots/{shot_id}.zarr",
            arrays=arrays,
            source_generation=_source_generation(shot_id) if pin_generation else None,
        )
        total_bytes += cast(int, artifact["bytes"])
        shots.append(
            {
                "shot_id": shot_id,
                "status": "acquired",
                "programme_class": programme_class,
                "artifacts": [artifact],
            }
        )
    manifest = finalise_source_object_manifest(
        {
            "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
            "machine": "MAST",
            "campaign": "lineage fixture",
            "retrieved_at": _FIXED_TS,
            "synthetic": False,
            "licence_spdx": "CC-BY-SA-4.0",
            "licence": "CC-BY-SA-4.0",
            "citation": "FAIR-MAST citation",
            "citations": ["FAIR-MAST citation"],
            "source_policy_url": "https://mastapp.site/",
            "status": "complete",
            "n_requested": 2,
            "n_acquired": 2,
            "total_bytes": total_bytes,
            "shots": shots,
        }
    )
    path = root / "source-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_replay(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    archive_path = root / "channels.npz"
    arrays: dict[str, NDArray[Any]] = {"shot_ids": np.asarray([101], dtype=np.int64)}
    for name, value in _channels(_disruptive_ip()).items():
        arrays[f"101:{name}"] = value
    np.savez(archive_path, **arrays)  # type: ignore[arg-type]
    binding = inspect_replay_archive(archive_path)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "synthetic": False,
        "material_dir": "source",
        "channels_npz": archive_path.name,
        "channels_archive": binding,
        "channel_schema": list(MEASURED_CHANNELS),
        "locked_window": 201,
        "n_derived": 1,
        "shots": [
            {"shot_id": 101, "status": "derived", "n_samples": 300},
            {"shot_id": 102, "status": "failed", "error": "density absent"},
        ],
        "generated_at": _FIXED_TS,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    report_path = root / "replay-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, archive_path


def _inputs(tmp_path: Path, *, pin_generation: bool = False) -> tuple[Path, Path, Path]:
    source = _write_source_manifest(tmp_path / "source", pin_generation=pin_generation)
    replay_report, archive = _write_replay(tmp_path / "replay")
    return source, replay_report, archive


def _build(tmp_path: Path, out_name: str = "out", *, pin_generation: bool = False) -> dict[str, Any]:
    source, replay_report, archive = _inputs(tmp_path, pin_generation=pin_generation)
    return build_lineage_bound_dataset(
        source_manifest_path=source,
        replay_report_path=replay_report,
        replay_archive_path=archive,
        transform_spec_path=_TRANSFORM_PATH,
        dataset_id="mast-lineage-fixture",
        out_dir=tmp_path / out_name,
        retrieved_at=_FIXED_TS,
        generated_at=_FIXED_TS,
    )


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None], *, reseal: bool = True) -> None:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    if reseal and "payload_sha256" in payload:
        payload["payload_sha256"] = None
        payload["payload_sha256"] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_build_lineage_bound_dataset_seals_every_stage(tmp_path: Path) -> None:
    """Produce real dataset files and exact per-shot producer bindings."""
    manifest = _build(tmp_path)
    out = tmp_path / "out"
    assert manifest["schema_version"] == PRODUCER_LINEAGE_SCHEMA
    assert manifest["lineage_mode"] == LINEAGE_MODE
    assert manifest["producer_time_lineage"] is True
    assert manifest["counts"] == {"acquired_source_shots": 2, "lineage_records": 1, "excluded_shots": 1}
    assert set(manifest["claim_boundary"].values()) == {False}
    assert set(BASE_BLOCKERS).issubset(manifest["blockers"])
    assert "source_generation_not_pinned_for_every_shot" in manifest["blockers"]
    record = manifest["lineage_records"][0]
    assert record["shot_id"] == 101
    assert record["producer_time_attested"] is True
    assert record["replay_member"]["producer_digest_bound"] is True
    assert record["source_parent"]["source_generation_sha256"] is None
    assert manifest["payload_sha256"] == canonical_json_sha256({**manifest, "payload_sha256": None})
    assert validate_producer_lineage_manifest(manifest) == manifest["payload_sha256"]
    assert validate_producer_lineage_manifest(manifest, artifact_root=out) == manifest["payload_sha256"]
    assert json.loads((out / "producer-lineage.json").read_text()) == manifest
    report = json.loads((out / "dataset-report.json").read_text())
    assert report["schema_version"] == DATASET_SCHEMA
    assert report["status"] == "blocked"
    assert report["label_authority_counts"] == {"ip_proxy": 1}
    assert load_real_data_manifest(out / "mast-lineage-fixture.manifest.json", verify_artifact=True).synthetic is False


def test_source_generation_pin_is_bound_without_claim_promotion(tmp_path: Path) -> None:
    """Record a verified generation pin while every label/admission claim stays false."""
    manifest = _build(tmp_path, pin_generation=True)
    assert manifest["source_generation_pinned_for_all_records"] is True
    assert "source_generation_not_pinned_for_every_shot" not in manifest["blockers"]
    assert manifest["lineage_records"][0]["source_parent"]["source_generation_sha256"] is not None
    assert set(manifest["claim_boundary"]) == set(CLAIM_FIELDS)
    assert set(manifest["claim_boundary"].values()) == {False}


def test_two_fresh_runs_are_byte_identical(tmp_path: Path) -> None:
    """Reproduce every generated byte from fixed snapshots and timestamps."""
    source, replay_report, archive = _inputs(tmp_path)
    manifests = []
    for name in ("first", "second"):
        manifests.append(
            build_lineage_bound_dataset(
                source_manifest_path=source,
                replay_report_path=replay_report,
                replay_archive_path=archive,
                transform_spec_path=_TRANSFORM_PATH,
                dataset_id="mast-lineage-fixture",
                out_dir=tmp_path / name,
                retrieved_at=_FIXED_TS,
                generated_at=_FIXED_TS,
            )
        )
    assert manifests[0] == manifests[1]
    first_files = {path.name: path.read_bytes() for path in (tmp_path / "first").iterdir()}
    second_files = {path.name: path.read_bytes() for path in (tmp_path / "second").iterdir()}
    assert first_files == second_files


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["lineage_records"][0]["source_parent"].update({"extra": True}), "source_parent fields"),
        (lambda p: p["source_manifest_binding"].update({"extra": True}), "source_manifest_binding fields"),
        (lambda p: p.update({"extra": True}), "producer lineage fields"),
        (lambda p: p.update({"schema_version": "wrong"}), "unsupported producer lineage schema"),
        (lambda p: p.update({"status": "ready"}), "identity/status"),
        (lambda p: p.update({"licence_spdx": "MIT"}), "provenance mismatch"),
        (lambda p: p["replay_archive_binding"].update({"extra": True}), "archive_binding fields"),
        (lambda p: p["replay_archive_binding"].update({"bytes": 0}), "bytes must be"),
        (lambda p: p["replay_archive_binding"].update({"shot_count": 2}), "shot_count mismatch"),
        (lambda p: p["lineage_records"][0].update({"producer_time_attested": False}), "attestation"),
        (lambda p: p["lineage_records"].append(deepcopy(p["lineage_records"][0])), "duplicate lineage"),
        (
            lambda p: p["lineage_records"][0]["replay_member"].update({"producer_digest_bound": False}),
            "not producer-bound",
        ),
        (lambda p: p["lineage_records"][0]["transform"].update({"extra": True}), "transform fields"),
        (
            lambda p: p["lineage_records"][0]["dataset_artifact"].update({"extra": True}),
            "dataset_artifact fields",
        ),
        (lambda p: p["exclusion_ledger"][0].update({"replay_member_present": True}), "presence flags"),
        (lambda p: p["exclusion_ledger"][0].update({"shot_id": 101}), "overlapping exclusion"),
        (lambda p: p["counts"].update({"lineage_records": 2}), "counts mismatch"),
        (lambda p: p.update({"source_generation_pinned_for_all_records": True}), "aggregate mismatch"),
        (lambda p: p.update({"blockers": []}), "blockers mismatch"),
        (lambda p: p["claim_boundary"].update({"training_admission": True}), "claim boundary"),
        (lambda p: p["dataset_report_binding"].update({"extra": True}), "dataset_report_binding fields"),
        (lambda p: p["dataset_manifest_binding"].update({"extra": True}), "dataset_manifest_binding fields"),
    ],
)
def test_producer_lineage_validator_rejects_tampering(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    """Reject structural, digest, inventory, attestation, and claim tampering."""
    manifest = _build(tmp_path)
    mutate(manifest)
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match=message):
        validate_producer_lineage_manifest(manifest)


def test_producer_lineage_validator_reopens_bound_output_bytes(tmp_path: Path) -> None:
    """Reject report, manifest, and dataset bytes that drift after production."""
    manifest = _build(tmp_path)
    out = tmp_path / "out"
    report_path = out / "dataset-report.json"
    report_path.write_text(report_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(DatasetProducerLineageError, match="report file_sha256 mismatch"):
        validate_producer_lineage_manifest(manifest, artifact_root=out)

    manifest = _build(tmp_path / "second")
    out = tmp_path / "second/out"
    dataset_manifest_path = out / "mast-lineage-fixture.manifest.json"
    dataset_manifest_path.write_text(dataset_manifest_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(DatasetProducerLineageError, match="manifest file_sha256 mismatch"):
        validate_producer_lineage_manifest(manifest, artifact_root=out)

    manifest = _build(tmp_path / "third")
    out = tmp_path / "third/out"
    artifact_path = out / "shot_101.npz"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"drift")
    dataset_manifest_path = out / "mast-lineage-fixture.manifest.json"
    dataset_manifest = cast(dict[str, Any], json.loads(dataset_manifest_path.read_text()))
    dataset_manifest["artifacts"][0]["checksum_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["dataset_manifest_binding"]["file_sha256"] = hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match="dataset_artifact.sha256 mismatch"):
        validate_producer_lineage_manifest(manifest, artifact_root=out)

    manifest = _build(tmp_path / "fourth")
    out = tmp_path / "fourth/out"
    report_path = out / "dataset-report.json"
    report = cast(dict[str, Any], json.loads(report_path.read_text()))
    report["generated_at"] = "changed"
    report["payload_sha256"] = None
    report["payload_sha256"] = canonical_json_sha256(report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["dataset_report_binding"]["file_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match="report payload_sha256 mismatch"):
        validate_producer_lineage_manifest(manifest, artifact_root=out)

    manifest = _build(tmp_path / "fifth")
    out = tmp_path / "fifth/out"
    dataset_manifest_path = out / "mast-lineage-fixture.manifest.json"
    dataset_manifest = cast(dict[str, Any], json.loads(dataset_manifest_path.read_text()))
    dataset_manifest["schema_version"] = "wrong"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["dataset_manifest_binding"]["file_sha256"] = hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match="manifest is invalid"):
        validate_producer_lineage_manifest(manifest, artifact_root=out)


@pytest.mark.parametrize("bad_path", ["../report.json", "/tmp/report.json", "C:\\report.json"])
def test_producer_lineage_validator_confines_bound_paths(tmp_path: Path, bad_path: str) -> None:
    """Reject traversal, absolute, drive-qualified, and backslash output paths."""
    manifest = _build(tmp_path)
    manifest["dataset_report_binding"]["path"] = bad_path
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match="confined POSIX relative path"):
        validate_producer_lineage_manifest(manifest, artifact_root=tmp_path / "out")


def test_producer_lineage_validator_rejects_missing_bound_file(tmp_path: Path) -> None:
    """Reject a syntactically safe binding that resolves to no output file."""
    manifest = _build(tmp_path)
    manifest["dataset_report_binding"]["path"] = "missing.json"
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match="does not resolve to a file"):
        validate_producer_lineage_manifest(manifest, artifact_root=tmp_path / "out")


def test_producer_lineage_validator_rejects_symlink_escape(tmp_path: Path) -> None:
    """Reject a relative output path whose resolved target leaves artifact_root."""
    manifest = _build(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "out/link.json").symlink_to(outside)
    manifest["dataset_report_binding"]["path"] = "link.json"
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    with pytest.raises(DatasetProducerLineageError, match="escapes artifact_root"):
        validate_producer_lineage_manifest(manifest, artifact_root=tmp_path / "out")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "schema"),
        ("lineage_mode", "wrong", "lineage_mode"),
        ("source_schema", "wrong", "source_schema"),
        ("input_schema", "wrong", "input_schema"),
        ("output_schema", "wrong", "output_schema"),
        ("operation_chain", [], "operation_chain"),
        ("label_authority", "independent", "producer-time ip_proxy"),
        ("producer_time_attested", False, "producer-time ip_proxy"),
        ("independent_label_authority", True, "claim fields"),
        ("training_admission", True, "claim fields"),
        ("transform_id", "", "non-empty"),
    ],
)
def test_transform_spec_rejects_contract_drift(field: str, value: Any, message: str) -> None:
    """Reject schema, operation, identity, and claim promotion drift."""
    payload = cast(dict[str, Any], json.loads(_TRANSFORM_PATH.read_text()))
    payload[field] = value
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(DatasetProducerLineageError, match=message):
        validate_transform_spec(payload)


def test_transform_spec_rejects_shape_and_digest_drift() -> None:
    """Reject an extra field and a stale self-digest."""
    payload = cast(dict[str, Any], json.loads(_TRANSFORM_PATH.read_text()))
    payload["extra"] = True
    with pytest.raises(DatasetProducerLineageError, match="fields"):
        validate_transform_spec(payload)
    payload.pop("extra")
    payload["payload_sha256"] = "0" * 64
    with pytest.raises(DatasetProducerLineageError, match="does not match"):
        validate_transform_spec(payload)
    payload["payload_sha256"] = "BAD"
    with pytest.raises(DatasetProducerLineageError, match="lowercase SHA-256"):
        validate_transform_spec(payload)
    payload["operation_chain"] = {}
    with pytest.raises(DatasetProducerLineageError, match="must be an array"):
        validate_transform_spec(payload)


def test_json_reader_and_source_inventory_fail_closed(tmp_path: Path) -> None:
    """Reject absent, ambiguous, non-object, and structurally unusable inputs."""
    with pytest.raises(DatasetProducerLineageError, match="cannot read fixture"):
        producer._read_json(tmp_path / "missing.json", label="fixture")
    path = tmp_path / "input.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(DatasetProducerLineageError, match="duplicate JSON key"):
        producer._read_json(path, label="fixture")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetProducerLineageError, match="root must be an object"):
        producer._read_json(path, label="fixture")

    failed = {"shots": [{"shot_id": 1, "status": "failed"}]}
    with pytest.raises(DatasetProducerLineageError, match="must contain acquired"):
        producer._source_inventory(failed)
    with pytest.raises(DatasetProducerLineageError, match="exactly one artifact"):
        producer._source_inventory(
            {
                "shots": [
                    {
                        "shot_id": 1,
                        "status": "acquired",
                        "programme_class": "unknown",
                        "artifacts": [{}, {}],
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report, binding: report.update({"schema_version": "wrong"}), "producer-bound v2"),
        (lambda report, binding: report.update({"synthetic": True}), "identity or channel"),
        (lambda report, binding: report.update({"channel_schema": []}), "identity or channel"),
        (lambda report, binding: report.update({"channels_archive": []}), "must be an object"),
        (lambda report, binding: report.update({"shots": {}}), "must be an array"),
        (
            lambda report, binding: report["shots"].append(report["shots"][0]),
            "duplicate replay shot_id",
        ),
        (lambda report, binding: report["shots"][0].update({"shot_id": True}), "positive integer"),
        (lambda report, binding: report["shots"][0].update({"status": "wrong"}), "status is unsupported"),
        (lambda report, binding: report.update({"n_derived": 2}), "n_derived mismatch"),
        (
            lambda report, binding: (
                report["shots"].append({"shot_id": 103, "status": "derived", "n_samples": 3}),
                report.update({"n_derived": 2}),
            ),
            "derived inventory differs",
        ),
        (
            lambda report, binding: (
                binding["shot_members"][0].update({"sha256": "BAD"}),
                report.update({"channels_archive": binding}),
            ),
            "lowercase SHA-256",
        ),
    ],
)
def test_replay_inventory_rejects_structural_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any]], Any],
    message: str,
) -> None:
    """Reject malformed replay identity, records, counts, and member digests."""
    report_path, archive_path = _write_replay(tmp_path / "replay")
    report = cast(dict[str, Any], json.loads(report_path.read_text()))
    binding = inspect_replay_archive(archive_path)
    mutate(report, binding)
    report["payload_sha256"] = None
    report["payload_sha256"] = canonical_json_sha256(report)
    with pytest.raises(DatasetProducerLineageError, match=message):
        producer._replay_inventory(report, binding)


def test_replay_binding_and_inventory_drift_fail_closed(tmp_path: Path) -> None:
    """Reject a report binding mismatch and source/replay set mismatch."""
    source, replay_report, archive = _inputs(tmp_path)
    _rewrite(replay_report, lambda payload: payload["channels_archive"].update({"bytes": 1}))
    with pytest.raises(DatasetProducerLineageError, match="binding does not match"):
        build_lineage_bound_dataset(
            source_manifest_path=source,
            replay_report_path=replay_report,
            replay_archive_path=archive,
            transform_spec_path=_TRANSFORM_PATH,
            dataset_id="fixture",
            out_dir=tmp_path / "out",
            retrieved_at=_FIXED_TS,
            generated_at=_FIXED_TS,
        )

    source, replay_report, archive = _inputs(tmp_path / "second")
    _rewrite(replay_report, lambda payload: payload["shots"].pop())
    with pytest.raises(DatasetProducerLineageError, match="source acquired shots"):
        build_lineage_bound_dataset(
            source_manifest_path=source,
            replay_report_path=replay_report,
            replay_archive_path=archive,
            transform_spec_path=_TRANSFORM_PATH,
            dataset_id="fixture",
            out_dir=tmp_path / "out2",
            retrieved_at=_FIXED_TS,
            generated_at=_FIXED_TS,
        )


def test_output_confinement_overwrite_and_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect immutable inputs, refuse overwrite, and remove handled-failure outputs."""
    source, replay_report, archive = _inputs(tmp_path)

    def _call(out_dir: Path) -> dict[str, Any]:
        return build_lineage_bound_dataset(
            source_manifest_path=source,
            replay_report_path=replay_report,
            replay_archive_path=archive,
            transform_spec_path=_TRANSFORM_PATH,
            dataset_id="fixture",
            out_dir=out_dir,
            retrieved_at=_FIXED_TS,
            generated_at=_FIXED_TS,
        )

    with pytest.raises(DatasetProducerLineageError, match="outside every immutable"):
        _call(source.parent / "out")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(DatasetProducerLineageError, match="refusing to overwrite"):
        _call(existing)
    with pytest.raises(DatasetProducerLineageError, match="parent"):
        _call(tmp_path / "missing/child")

    def _fail_build(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(producer, "build_dataset", _fail_build)
    failed_out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture failure"):
        _call(failed_out)
    assert not failed_out.exists()


def test_invalid_source_and_archive_errors_are_normalised(tmp_path: Path) -> None:
    """Normalise invalid source contracts and missing or malformed archive bytes."""
    source, replay_report, archive = _inputs(tmp_path)

    def _call(out_dir: Path) -> dict[str, Any]:
        return build_lineage_bound_dataset(
            source_manifest_path=source,
            replay_report_path=replay_report,
            replay_archive_path=archive,
            transform_spec_path=_TRANSFORM_PATH,
            dataset_id="fixture",
            out_dir=out_dir,
            retrieved_at=_FIXED_TS,
            generated_at=_FIXED_TS,
        )

    _rewrite(source, lambda payload: payload.update({"schema_version": "wrong"}))
    with pytest.raises(DatasetProducerLineageError, match="source manifest is invalid"):
        _call(tmp_path / "out")

    source = _write_source_manifest(tmp_path / "source2", pin_generation=False)
    archive.unlink()
    with pytest.raises(DatasetProducerLineageError, match="cannot read replay archive"):
        _call(tmp_path / "out2")

    archive.write_bytes(b"not-an-archive")
    with pytest.raises(DatasetProducerLineageError, match="replay archive is invalid"):
        _call(tmp_path / "out3")


def test_output_directory_creation_race_preserves_foreign_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not clean a destination when exclusive directory creation loses a race."""
    source, replay_report, archive = _inputs(tmp_path)
    out = tmp_path / "race"
    original_mkdir = Path.mkdir

    def _race(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == out:
            raise FileExistsError("fixture race")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _race)
    with pytest.raises(FileExistsError, match="fixture race"):
        build_lineage_bound_dataset(
            source_manifest_path=source,
            replay_report_path=replay_report,
            replay_archive_path=archive,
            transform_spec_path=_TRANSFORM_PATH,
            dataset_id="fixture",
            out_dir=out,
            retrieved_at=_FIXED_TS,
            generated_at=_FIXED_TS,
        )
    assert not out.exists()


def test_cli_builds_lineage_bound_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise the complete public CLI boundary."""
    source, replay_report, archive = _inputs(tmp_path)
    code = main(
        [
            "--source-manifest",
            str(source),
            "--replay-report",
            str(replay_report),
            "--replay-archive",
            str(archive),
            "--transform-spec",
            str(_TRANSFORM_PATH),
            "--dataset-id",
            "fixture",
            "--out-dir",
            str(tmp_path / "out"),
            "--retrieved-at",
            _FIXED_TS,
            "--generated-at",
            _FIXED_TS,
        ]
    )
    assert code == 0
    assert "producer lineage: 1 records" in capsys.readouterr().out
    assert (tmp_path / "out/producer-lineage.json").is_file()
