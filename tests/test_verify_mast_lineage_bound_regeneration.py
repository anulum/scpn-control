# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for reproducible lineage-bound MAST regeneration
"""Real-file tests for the two-run FAIR-MAST regeneration verifier."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import validation.verify_mast_lineage_bound_regeneration as verifier
from validation.build_disruption_replay_channels import REPORT_SCHEMA, inspect_replay_archive
from validation.build_mast_disruption_dataset import MEASURED_CHANNELS
from validation.build_mast_lineage_bound_dataset import CLAIM_FIELDS, build_lineage_bound_dataset
from validation.mast_source_object_manifest import (
    SOURCE_GENERATION_DIGEST_KIND,
    SOURCE_GENERATION_SCHEMA,
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    canonical_json_sha256,
    finalise_source_object_manifest,
)
from validation.verify_mast_lineage_bound_regeneration import (
    REGENERATION_SCHEMA,
    RegenerationVerificationError,
    build_regeneration_verification,
    main,
    validate_regeneration_verification,
)

_FIXED_TS = "2026-07-22T21:46:00Z"
_EVIDENCE_TS = "2026-07-22T21:50:00Z"
_TRANSFORM_PATH = Path(__file__).parents[1] / "validation/mast_dataset_producer_transform_spec_v1.json"


def _channels(ip: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    channels = {name: np.linspace(0.0, 1.0, ip.size, dtype=np.float64) for name in MEASURED_CHANNELS}
    channels["time_s"] = np.arange(ip.size, dtype=np.float64) * 1.0e-3
    channels["Ip_MA"] = ip
    return channels


def _disruptive_ip() -> NDArray[np.float64]:
    ip = np.ones(300, dtype=np.float64)
    ip[:100] = np.linspace(0.0, 1.0, 100)
    ip[250] = 0.5
    ip[251:] = 0.0
    return ip


def _source_generation(shot_id: int) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_GENERATION_SCHEMA,
        "digest_kind": SOURCE_GENERATION_DIGEST_KIND,
        "source_uri": f"s3://mast/level2/shots/{shot_id}.zarr",
        "metadata_path": "zarr.json",
        "sha256": f"{shot_id:064x}"[-64:],
        "bytes": 42,
        "zarr_format": 3,
        "consolidated_metadata_kind": "inline",
        "etag": f'"etag-{shot_id}"',
        "last_modified": "Wed, 22 Jul 2026 20:00:00 GMT",
    }


def _write_source(root: Path, *, pin_generation: bool = True) -> Path:
    root.mkdir(parents=True)
    shots: list[dict[str, Any]] = []
    total_bytes = 0
    for shot_id, programme_class in ((101, "forced_vde"), (102, "unknown")):
        arrays = {"summary.time": np.arange(3, dtype=np.float64)}
        artifact_path = root / f"shot_{shot_id}.npz"
        np.savez_compressed(artifact_path, **arrays)  # type: ignore[arg-type]  # NumPy kwargs stub limitation
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
            "campaign": "regeneration verifier fixture",
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
    archive = root / "channels.npz"
    arrays: dict[str, NDArray[Any]] = {"shot_ids": np.asarray([101], dtype=np.int64)}
    for name, value in _channels(_disruptive_ip()).items():
        arrays[f"101:{name}"] = value
    np.savez(archive, **arrays)  # type: ignore[arg-type]  # NumPy kwargs stub limitation
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "synthetic": False,
        "material_dir": "source",
        "channels_npz": archive.name,
        "channels_archive": inspect_replay_archive(archive),
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
    path = root / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, archive


def _build_fixture(tmp_path: Path, *, pin_generation: bool = True) -> tuple[Path, Path, Path, Path, Path]:
    source = _write_source(tmp_path / "source", pin_generation=pin_generation)
    replay, archive = _write_replay(tmp_path / "replay")
    for name in ("run-a", "run-b"):
        build_lineage_bound_dataset(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            transform_spec_path=_TRANSFORM_PATH,
            dataset_id="mast-regeneration-fixture",
            out_dir=tmp_path / name,
            retrieved_at=_FIXED_TS,
            generated_at=_FIXED_TS,
        )
    return source, replay, archive, tmp_path / "run-a", tmp_path / "run-b"


def _build_report(tmp_path: Path) -> tuple[dict[str, Any], tuple[Path, Path, Path, Path, Path]]:
    inputs = _build_fixture(tmp_path)
    source, replay, archive, run_a, run_b = inputs
    report = build_regeneration_verification(
        source_manifest_path=source,
        replay_report_path=replay,
        replay_archive_path=archive,
        run_a_root=run_a,
        run_b_root=run_b,
        generated_at=_EVIDENCE_TS,
    )
    return report, inputs


def _reseal(payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None], *, reseal: bool = True) -> None:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    if reseal:
        _reseal(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mutate_inventory_fields(payload: dict[str, Any]) -> None:
    for run_name in ("run_a", "run_b"):
        cast(dict[str, Any], payload[run_name]["file_inventory"][0])["extra"] = True
    payload["comparison"]["inventory_sha256"] = canonical_json_sha256({"files": payload["run_a"]["file_inventory"]})


def test_build_verification_reopens_complete_real_file_chain(tmp_path: Path) -> None:
    """Bind source, replay, two output trees, blockers, and false claims."""
    report, _ = _build_report(tmp_path)
    assert report["schema_version"] == REGENERATION_SCHEMA
    assert report["status"] == "reproducible_blocked"
    assert report["comparison"]["byte_identical"] is True
    assert report["comparison"]["file_count"] == 4
    assert report["run_a"]["file_inventory"] == report["run_b"]["file_inventory"]
    assert report["run_a"]["producer_lineage"] == report["run_b"]["producer_lineage"]
    assert report["source_generation_pinned_for_all_records"] is True
    assert set(report["claim_boundary"]) == set(CLAIM_FIELDS)
    assert set(report["claim_boundary"].values()) == {False}
    assert validate_regeneration_verification(report) == report["payload_sha256"]


def test_main_writes_once_and_removes_reserved_output_after_failure(tmp_path: Path) -> None:
    """Exercise the public CLI, exclusive output, and failure cleanup contract."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path)
    out = tmp_path / "evidence/report.json"
    argv = [
        "--source-manifest",
        str(source),
        "--replay-report",
        str(replay),
        "--replay-archive",
        str(archive),
        "--run-a",
        str(run_a),
        "--run-b",
        str(run_b),
        "--generated-at",
        _EVIDENCE_TS,
        "--json-out",
        str(out),
    ]
    assert main(argv) == 0
    assert validate_regeneration_verification(json.loads(out.read_text(encoding="utf-8")))
    with pytest.raises(RegenerationVerificationError, match="refusing to overwrite"):
        main(argv)
    out.unlink()
    (run_b / "shot_101.npz").write_bytes(b"drift")
    with pytest.raises(RegenerationVerificationError, match="producer lineage is invalid"):
        main(argv)
    assert not out.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"extra": True}), "fields do not match"),
        (lambda p: p.update({"schema_version": "wrong"}), "identity/status"),
        (lambda p: p.update({"status": "admitted"}), "identity/status"),
        (lambda p: p.update({"synthetic": True}), "identity/status"),
        (lambda p: p.update({"source_generation_pinned_for_all_records": False}), "identity/status"),
        (lambda p: p.update({"dataset_id": ""}), "dataset_id"),
        (lambda p: p.update({"generated_at": ""}), "generated_at"),
        (lambda p: p["comparison"].update({"byte_identical": False}), "byte-identical"),
        (lambda p: p.update({"comparison": {}}), "comparison fields"),
        (lambda p: p["comparison"].update({"file_count": 0}), "positive integer"),
        (lambda p: p["comparison"].update({"inventory_sha256": "bad"}), "SHA-256"),
        (lambda p: p["run_a"].update({"extra": True}), "run_a fields"),
        (lambda p: p["run_a"].update({"label": ""}), "run_a.label"),
        (lambda p: p["run_a"]["producer_lineage"].update({"extra": True}), "producer_lineage fields"),
        (_mutate_inventory_fields, "file_inventory.*fields"),
        (lambda p: p["run_a"]["file_inventory"][0].update({"bytes": 0}), "inventories are not exactly equal"),
        (lambda p: p["run_b"].update({"file_inventory": []}), "inventories are not exactly equal"),
        (lambda p: p["run_b"]["producer_lineage"].update({"file_sha256": "a" * 64}), "bindings differ"),
        (lambda p: p.update({"blockers": []}), "blockers"),
        (lambda p: p.update({"blockers": ["z", "a"]}), "blockers"),
        (lambda p: p["claim_boundary"].update({"training_admission": True}), "claim boundary"),
        (lambda p: p["source_manifest_binding"].update({"file_sha256": "bad"}), "SHA-256"),
        (lambda p: p["replay_archive_binding"].update({"file_sha256": "bad"}), "SHA-256"),
    ],
)
def test_report_validator_rejects_contract_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject identity, digest, inventory, blocker, and claim promotion drift."""
    report, _ = _build_report(tmp_path)
    mutate(report)
    _reseal(report)
    with pytest.raises(RegenerationVerificationError, match=message):
        validate_regeneration_verification(report)


def test_report_validator_rejects_self_digest_and_inventory_digest_drift(tmp_path: Path) -> None:
    """Reject an unsealed edit and a resealed but inconsistent inventory digest."""
    report, _ = _build_report(tmp_path)
    report["dataset_id"] = "drift"
    with pytest.raises(RegenerationVerificationError, match="payload_sha256"):
        validate_regeneration_verification(report)
    _reseal(report)
    report["comparison"]["inventory_sha256"] = "a" * 64
    _reseal(report)
    with pytest.raises(RegenerationVerificationError, match="inventory_sha256 mismatch"):
        validate_regeneration_verification(report)


def test_builder_rejects_unpinned_source_and_replay_drift(tmp_path: Path) -> None:
    """Fail before comparison when source generations or replay bindings are absent."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path, pin_generation=False)
    with pytest.raises(RegenerationVerificationError, match="source generation is not pinned"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path / "pinned")
    _rewrite(replay, lambda payload: payload.update({"synthetic": True}))
    with pytest.raises(RegenerationVerificationError, match="producer-bound v2"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )


def test_builder_rejects_partition_counts_and_distinctness(tmp_path: Path) -> None:
    """Require explicit time, distinct roots, exact partition, and replay count."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path)
    with pytest.raises(RegenerationVerificationError, match="generated_at"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at="",
        )
    with pytest.raises(RegenerationVerificationError, match="distinct"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_a,
            generated_at=_EVIDENCE_TS,
        )
    _rewrite(
        replay,
        lambda payload: payload["shots"].append({"shot_id": 103, "status": "failed", "error": "absent"}),
    )
    with pytest.raises(RegenerationVerificationError, match="derived/failed partition"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )
    _rewrite(replay, lambda payload: payload["shots"].pop())
    _rewrite(replay, lambda payload: payload.update({"n_derived": 2}))
    with pytest.raises(RegenerationVerificationError, match="n_derived"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )


def test_builder_rejects_archive_and_run_tree_drift(tmp_path: Path) -> None:
    """Reject archive changes, missing roots, symlinks, and unequal output bytes."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path)
    archive.write_bytes(b"not an archive")
    with pytest.raises(RegenerationVerificationError, match="replay archive is invalid"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path / "tree")
    (run_a / "escape").symlink_to(source)
    with pytest.raises(RegenerationVerificationError, match="symlink"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )
    (run_a / "escape").unlink()
    (run_b / "extra.bin").write_bytes(b"extra")
    with pytest.raises(RegenerationVerificationError, match="not byte-identical"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )


def test_json_reader_and_tree_inventory_normalise_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalise missing, malformed, duplicate-key, scalar, empty-tree, and bad-entry inputs."""
    with pytest.raises(RegenerationVerificationError, match="cannot read fixture"):
        verifier._read_json(tmp_path / "missing.json", label="fixture")
    path = tmp_path / "bad.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(RegenerationVerificationError, match="duplicate JSON key"):
        verifier._read_json(path, label="fixture")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RegenerationVerificationError, match="root must be an object"):
        verifier._read_json(path, label="fixture")
    with pytest.raises(RegenerationVerificationError, match="not a directory"):
        verifier._tree_inventory(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RegenerationVerificationError, match="must contain files"):
        verifier._tree_inventory(empty)
    nested = tmp_path / "nested"
    (nested / "child").mkdir(parents=True)
    (nested / "child/file.bin").write_bytes(b"bytes")
    assert verifier._tree_inventory(nested)[0]["path"] == "child/file.bin"
    non_regular_root = tmp_path / "non_regular"
    non_regular_root.mkdir()
    if hasattr(os, "mkfifo"):
        # POSIX: exercise a real non-regular filesystem entry.
        os.mkfifo(non_regular_root / "pipe")
    else:
        # Windows and other platforms without FIFO: discover a real path via
        # rglob, then force the production non-regular classification.
        odd = non_regular_root / "odd"
        odd.write_bytes(b"payload")
        original_is_file = Path.is_file
        original_is_dir = Path.is_dir
        original_is_symlink = Path.is_symlink

        def _is_target(candidate: Path) -> bool:
            try:
                return candidate.resolve() == odd.resolve()
            except OSError:
                return candidate == odd

        monkeypatch.setattr(
            Path,
            "is_file",
            lambda self: False if _is_target(self) else original_is_file(self),
        )
        monkeypatch.setattr(
            Path,
            "is_dir",
            lambda self: False if _is_target(self) else original_is_dir(self),
        )
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: False if _is_target(self) else original_is_symlink(self),
        )
    with pytest.raises(RegenerationVerificationError, match="non-regular"):
        verifier._tree_inventory(non_regular_root)


def test_helper_shape_guards_reject_non_collections_and_bad_shot_sets() -> None:
    """Reject non-object/list fields and ambiguous source or replay inventories."""
    with pytest.raises(RegenerationVerificationError, match="must be an object"):
        verifier._mapping([], field="fixture")
    with pytest.raises(RegenerationVerificationError, match="must be an array"):
        verifier._list({}, field="fixture")
    assert verifier._source_shots(
        {
            "shots": [
                {"shot_id": 1, "status": "failed"},
                {"shot_id": 2, "status": "acquired", "artifacts": [{"source_generation": {}}]},
            ]
        }
    ) == ([2], True)
    with pytest.raises(RegenerationVerificationError, match="must have one artifact"):
        verifier._source_shots({"shots": [{"shot_id": 1, "status": "acquired", "artifacts": []}]})
    with pytest.raises(RegenerationVerificationError, match="non-empty, unique, and sorted"):
        verifier._source_shots({"shots": []})
    with pytest.raises(RegenerationVerificationError, match="unsupported status"):
        verifier._replay_shots({"shots": [{"shot_id": 1, "status": "pending"}]})
    with pytest.raises(RegenerationVerificationError, match="contains duplicates"):
        verifier._replay_shots({"shots": [{"shot_id": 1, "status": "derived"}, {"shot_id": 1, "status": "failed"}]})


@pytest.mark.parametrize(
    ("binding_name", "message"),
    [
        ("source_manifest_binding", "source-manifest binding mismatch"),
        ("replay_report_binding", "replay-report binding mismatch"),
        ("replay_archive_binding", "replay-archive binding mismatch"),
    ],
)
def test_run_binding_rejects_each_external_binding_drift(
    tmp_path: Path,
    binding_name: str,
    message: str,
) -> None:
    """Reject a resealed lineage file that no longer names its exact external input."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path)
    lineage_path = run_b / "producer-lineage.json"

    def _mutate(payload: dict[str, Any]) -> None:
        binding = cast(dict[str, Any], payload[binding_name])
        binding["file_sha256"] = "a" * 64

    _rewrite(lineage_path, _mutate)
    with pytest.raises(RegenerationVerificationError, match=message):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )


def test_builder_rejects_invalid_source_and_resealed_replay_archive_binding(tmp_path: Path) -> None:
    """Normalise source validation failure and reject a self-consistent but wrong replay declaration."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path)
    (source.parent / "shot_101.npz").write_bytes(b"drift")
    with pytest.raises(RegenerationVerificationError, match="source manifest is invalid"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path / "replay-drift")
    _rewrite(
        replay,
        lambda payload: cast(dict[str, Any], payload["channels_archive"]).update({"file_sha256": "a" * 64}),
    )
    with pytest.raises(RegenerationVerificationError, match="does not bind the exact replay archive"):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_id", "different-dataset", "dataset identifiers differ"),
        ("generated_at", "2026-07-22T21:47:00Z", "generation timestamps differ"),
    ],
)
def test_builder_rejects_cross_run_identity_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    """Reject independently valid runs that do not describe the same production event."""
    source, replay, archive, run_a, run_b = _build_fixture(tmp_path)
    _rewrite(run_b / "producer-lineage.json", lambda payload: payload.update({field: value}))
    with pytest.raises(RegenerationVerificationError, match=message):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=run_a,
            run_b_root=run_b,
            generated_at=_EVIDENCE_TS,
        )


def test_run_binding_detects_tree_change_and_defensive_generation_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detect a changing tree and retain an independent aggregate-generation guard."""
    source, replay, archive, run_a, _ = _build_fixture(tmp_path)
    source_payload = cast(dict[str, Any], json.loads(source.read_text(encoding="utf-8")))
    replay_payload = cast(dict[str, Any], json.loads(replay.read_text(encoding="utf-8")))
    source_binding = verifier._file_binding(source.read_bytes(), source_payload)
    replay_binding = verifier._file_binding(replay.read_bytes(), replay_payload)
    archive_binding = inspect_replay_archive(archive)
    original_inventory = verifier._tree_inventory
    calls = 0

    def _changing_inventory(root: Path) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        inventory = original_inventory(root)
        if calls == 2:
            return [*inventory, {"path": "late", "bytes": 1, "sha256": "a" * 64}]
        return inventory

    monkeypatch.setattr(verifier, "_tree_inventory", _changing_inventory)
    with pytest.raises(RegenerationVerificationError, match="changed while"):
        verifier._run_binding(
            run_a,
            label="run-a",
            source_binding=source_binding,
            replay_binding=replay_binding,
            archive_binding=archive_binding,
        )
    monkeypatch.setattr(verifier, "_tree_inventory", original_inventory)
    lineage_path = run_a / "producer-lineage.json"
    _rewrite(lineage_path, lambda payload: payload.update({"source_generation_pinned_for_all_records": False}))
    monkeypatch.setattr(verifier, "validate_producer_lineage_manifest", lambda *_args, **_kwargs: "ok")
    with pytest.raises(RegenerationVerificationError, match="does not pin every source generation"):
        verifier._run_binding(
            run_a,
            label="run-a",
            source_binding=source_binding,
            replay_binding=replay_binding,
            archive_binding=archive_binding,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("blockers", "dataset blockers differ"),
        ("claims", "dataset claim boundaries differ"),
    ],
)
def test_builder_retains_cross_run_blocker_and_claim_equality_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    """Exercise defensive equality guards after independently valid run admission."""
    source, replay, archive, run_a_root, _ = _build_fixture(tmp_path)
    lineage = cast(dict[str, Any], json.loads((run_a_root / "producer-lineage.json").read_text(encoding="utf-8")))
    inventory = verifier._tree_inventory(run_a_root)
    calls = 0

    def _binding(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal calls
        calls += 1
        candidate = cast(dict[str, Any], json.loads(json.dumps(lineage)))
        if calls == 2 and drift == "blockers":
            candidate["blockers"] = [*candidate["blockers"], "defensive-drift"]
        if calls == 2 and drift == "claims":
            candidate["claim_boundary"]["training_admission"] = True
        return (
            {
                "label": f"run-{calls}",
                "producer_lineage": {"file_sha256": "a" * 64, "payload_sha256": "b" * 64},
                "file_inventory": inventory,
            },
            candidate,
        )

    monkeypatch.setattr(verifier, "_run_binding", _binding)
    with pytest.raises(RegenerationVerificationError, match=message):
        build_regeneration_verification(
            source_manifest_path=source,
            replay_report_path=replay,
            replay_archive_path=archive,
            run_a_root=tmp_path / "run-a",
            run_b_root=tmp_path / "run-b",
            generated_at=_EVIDENCE_TS,
        )
