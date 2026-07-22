# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for FAIR-MAST campaign reconciliation
"""Adversarial production-path tests for the L2F-90 reconciliation gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import validation.reconcile_mast_campaign_lineage as reconciliation
from validation.build_mast_disruption_dataset import MEASURED_CHANNELS
from validation.evaluate_mast_disruption import REPORT_SCHEMA as EVALUATION_SCHEMA
from validation.mast_source_object_manifest import SourceObjectManifestError, canonical_json_sha256
from validation.reconcile_mast_campaign_lineage import (
    INPUT_NAMES,
    LEGACY_DATASET_SCHEMA,
    RECONCILIATION_REPORT_SCHEMA,
    RECONCILIATION_SPEC_SCHEMA,
    REPLAY_SCHEMA,
    CampaignReconciliationError,
    main,
    reconcile_campaign,
)

_FIXED_TS = "2026-07-22T18:00:00Z"


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _material(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    for shot_id in (101, 102):
        path = root / f"shot_{shot_id}.npz"
        np.savez_compressed(
            path,
            **{  # type: ignore[arg-type]
                "summary.time": np.arange(3),
                "summary.ip": np.arange(3) + shot_id,
            },
        )
        size = path.stat().st_size
        total += size
        records.append(
            {
                "shot_id": shot_id,
                "status": "acquired",
                "npz": path.name,
                "checksum_sha256": _digest(path),
                "bytes": size,
            }
        )
    records.append({"shot_id": 103, "status": "failed", "error": "source array absent"})
    return _seal(
        {
            "schema_version": "scpn-control.mast-disruption-material.v1",
            "synthetic": False,
            "consumers": [],
            "source": {
                "path_template": "s3://mast/level2/shots/{shot_id}.zarr",
            },
            "licence": "MIT",
            "retrieved_at": _FIXED_TS,
            "generated_at": _FIXED_TS,
            "n_requested": 3,
            "n_acquired": 2,
            "total_bytes": total,
            "shots": records,
            "payload_sha256": None,
        }
    )


def _replay() -> dict[str, Any]:
    return _seal(
        {
            "schema_version": REPLAY_SCHEMA,
            "synthetic": False,
            "material_dir": "campaign01",
            "channels_npz": "channels.npz",
            "channel_schema": [
                "time_s",
                "Ip_MA",
                "BT_T",
                "beta_N",
                "q95",
                "ne_1e19",
                "n1_amp",
                "n2_amp",
                "locked_mode_amp",
                "dBdt_gauss_per_s",
                "vertical_position_m",
            ],
            "locked_window": 201,
            "n_derived": 1,
            "shots": [
                {"shot_id": 101, "status": "derived", "n_samples": 3},
                {"shot_id": 102, "status": "failed", "error": "summary.line_average_n_e absent"},
            ],
            "generated_at": _FIXED_TS,
            "payload_sha256": None,
        }
    )


def _dataset_manifest(checksum: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": "mast-disruption-campaign01",
        "machine": "MAST",
        "shot": "campaign:mast-disruption-campaign01",
        "synthetic": False,
        "source": {"kind": "local_archive", "uri": "s3://mast/level2/shots", "access": "anonymous"},
        "signals": [{"name": "Ip_MA", "path": "Ip_MA", "units": "MA", "timebase": "time_s"}],
        "retrieved_at": _FIXED_TS,
        "checksum_sha256": None,
        "licence": "MIT",
        "synthetic_generator": None,
        "synthetic_seed": None,
        "artifacts": [{"uri": "shot_101.npz", "checksum_sha256": checksum}],
    }


def _dataset_report(checksum: str) -> dict[str, Any]:
    fingerprint = sha256(checksum.encode()).hexdigest()
    return _seal(
        {
            "schema_version": LEGACY_DATASET_SCHEMA,
            "status": "blocked",
            "admission_ready": False,
            "blocked_reason": "proxy labels",
            "dataset_id": "mast-disruption-campaign01",
            "synthetic": False,
            "manifest": "mast-disruption-campaign01.manifest.json",
            "dataset_sha256": fingerprint,
            "n_shots": 1,
            "n_disruptive": 1,
            "channel_schema": [],
            "label_algorithm": {},
            "shots": [
                {
                    "shot_id": 101,
                    "npz": "shot_101.npz",
                    "checksum_sha256": checksum,
                    "label": 1,
                    "disruption_time_idx": 2,
                    "disruption_type": "current_quench",
                    "n_samples": 3,
                }
            ],
            "generated_at": _FIXED_TS,
            "payload_sha256": None,
        }
    )


def _evaluation(*, valid_digest: bool = False) -> dict[str, Any]:
    payload = _seal(
        {
            "schema_version": EVALUATION_SCHEMA,
            "status": "blocked",
            "admission_ready": False,
            "blocked_reason": "historical only",
            "dataset_id": "mast-disruption-campaign01",
            "data_provenance": {"licence": "MIT"},
            "predictor": {},
            "window_size": 128,
            "metrics": {},
            "shots": [{"shot_id": "shot_101"}],
            "claim_boundary": {"public_claim_allowed": False, "facility_roc_validated": False},
            "generated_at_utc": _FIXED_TS,
            "payload_sha256": None,
        }
    )
    if not valid_digest:
        payload["blocked_reason"] = "mutated historical report"
    return payload


def _write_channels(
    path: Path,
    *,
    shot_ids: tuple[int, ...] = (101,),
    mutate: Callable[[dict[str, np.ndarray[Any, Any]]], None] | None = None,
) -> None:
    arrays: dict[str, np.ndarray[Any, Any]] = {"shot_ids": np.asarray(shot_ids)}
    for shot_id in shot_ids:
        arrays.update({f"{shot_id}:{channel}": np.arange(3, dtype=np.float64) for channel in MEASURED_CHANNELS})
    if mutate is not None:
        mutate(arrays)
    np.savez(path, **arrays)  # type: ignore[arg-type]


def _drop_channel(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    del arrays["101:BT_T"]


def _drop_identifiers(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    del arrays["shot_ids"]


def _add_channel(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    arrays["101:extra"] = np.arange(3, dtype=np.float64)


def _non_finite_channel(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    arrays["101:BT_T"] = np.asarray([0.0, np.nan, 1.0])


def _short_channel(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    arrays["101:BT_T"] = np.arange(2, dtype=np.float64)


def _integer_channel(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    arrays["101:BT_T"] = np.arange(3, dtype=np.int64)


def _float_identifiers(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    arrays["shot_ids"] = np.asarray([101.0])


def _tree(root: Path, *, valid_evaluation_digest: bool = False) -> Path:
    root.mkdir()
    _write(root / "material.json", _material(root))
    (root / "derived").mkdir()
    _write_channels(root / "derived/channels.npz")
    _write(root / "derived/replay.json", _replay())
    dataset_npz = root / "dataset/shot_101.npz"
    dataset_npz.parent.mkdir()
    np.savez(dataset_npz, Ip_MA=np.asarray([1.0, 0.5, 0.0]))
    checksum = _digest(dataset_npz)
    _write(root / "dataset/manifest.json", _dataset_manifest(checksum))
    _write(root / "dataset/report.json", _dataset_report(checksum))
    _write(root / "evaluation/report.json", _evaluation(valid_digest=valid_evaluation_digest))
    input_paths = {
        "channels_archive": "derived/channels.npz",
        "dataset_manifest": "dataset/manifest.json",
        "dataset_report": "dataset/report.json",
        "evaluation_report": "evaluation/report.json",
        "material_manifest": "material.json",
        "replay_report": "derived/replay.json",
    }
    spec = _seal(
        {
            "schema_version": RECONCILIATION_SPEC_SCHEMA,
            "dataset_id": "mast-disruption-campaign01",
            "inputs": {
                name: {"path": relative, "file_sha256": _digest(root / relative)}
                for name, relative in input_paths.items()
            },
            "payload_sha256": None,
        }
    )
    return _write(root.parent / "spec.json", spec)


def _run(root: Path, spec: Path) -> dict[str, Any]:
    return reconcile_campaign(campaign_root=root, spec_path=spec, generated_at=_FIXED_TS)


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None], *, reseal: bool = True) -> None:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    if reseal and "payload_sha256" in payload:
        _seal(payload)
    _write(path, payload)


def _repin(spec_path: Path, root: Path, name: str) -> None:
    payload = cast(dict[str, Any], json.loads(spec_path.read_text(encoding="utf-8")))
    relative = payload["inputs"][name]["path"]
    payload["inputs"][name]["file_sha256"] = _digest(root / relative)
    _write(spec_path, _seal(payload))


def test_reconciliation_binds_bytes_and_remains_blocked(tmp_path: Path) -> None:
    """Bind every fixture byte while keeping every scientific claim false."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    report = _run(root, spec)

    assert report["schema_version"] == RECONCILIATION_REPORT_SCHEMA
    assert report["status"] == "blocked"
    assert set(cast(dict[str, bool], report["claim_boundary"]).values()) == {False}
    source = cast(dict[str, Any], report["source_inventory"])
    assert (source["requested_count"], source["acquired_count"], source["failed_count"]) == (3, 2, 1)
    assert source["native_zarr_bytes_preserved"] is False
    assert source["source_generation_pinned"] is False
    replay = cast(dict[str, Any], report["replay_inventory"])
    assert replay["derived_shot_ids"] == [101]
    assert replay["failed_shot_ids"] == [102]
    assert replay["channels_archive_member_count"] == len(MEASURED_CHANNELS)
    assert replay["channels_archive_total_samples"] == 3
    assert replay["channels_archive_shot_inventory_verified"] is True
    dataset = cast(dict[str, Any], report["dataset_inventory"])
    assert dataset["proxy_positive_count"] == 1
    assert dataset["proxy_negative_count"] == 0
    cross = cast(dict[str, Any], report["cross_inventory"])
    assert cross["dataset_equals_replay"] is True
    assert cross["evaluation_equals_dataset"] is True
    assert cross["acquired_without_dataset"] == [102]
    assert cross["replay_failures_without_acquired"] == []
    assert cross["exclusions"] == [
        {
            "shot_id": 102,
            "reason": "summary.line_average_n_e absent",
            "reason_evidence": "verified_self_digested_replay_report",
        }
    ]
    assert "evaluation_report_self_digest_invalid" in report["blockers"]
    assert report["payload_sha256"] == canonical_json_sha256({**report, "payload_sha256": None})
    assert str(tmp_path) not in json.dumps(report)


def test_valid_historical_digest_removes_only_that_blocker(tmp_path: Path) -> None:
    """Accept a valid historical digest without promoting admission state."""
    root = tmp_path / "campaign01"
    spec = _tree(root, valid_evaluation_digest=True)
    report = _run(root, spec)
    assert cast(dict[str, Any], report["evaluation_inventory"])["payload_sha256_valid"] is True
    assert "evaluation_report_self_digest_invalid" not in report["blockers"]
    assert report["status"] == "blocked"


def test_cli_writes_the_deterministic_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Write the same deterministic report through the command-line path."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    output = tmp_path / "evidence/report.json"
    assert (
        main(
            ["--campaign-root", str(root), "--spec", str(spec), "--generated-at", _FIXED_TS, "--json-out", str(output)]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == _run(root, spec)
    assert "requested=3 acquired=2 dataset=1" in capsys.readouterr().out


@pytest.mark.parametrize("relative_output", ["material.json", "evidence/report.json"])
def test_cli_refuses_output_inside_campaign_root(tmp_path: Path, relative_output: str) -> None:
    """Prevent the report writer from replacing or adding campaign evidence."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    material = root / "material.json"
    material_before = material.read_bytes()
    output = root / relative_output

    with pytest.raises(CampaignReconciliationError, match="json_out must be outside campaign_root"):
        main(
            [
                "--campaign-root",
                str(root),
                "--spec",
                str(spec),
                "--generated-at",
                _FIXED_TS,
                "--json-out",
                str(output),
            ]
        )

    assert material.read_bytes() == material_before
    if relative_output != "material.json":
        assert not output.exists()


@pytest.mark.parametrize(
    ("name", "relative"),
    [
        ("material_manifest", "material.json"),
        ("replay_report", "derived/replay.json"),
        ("dataset_report", "dataset/report.json"),
    ],
)
def test_self_digested_inputs_reject_tamper(tmp_path: Path, name: str, relative: str) -> None:
    """Reject pinned JSON whose embedded self-digest no longer matches."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(root / relative, lambda payload: payload.update({"generated_at": "tampered"}), reseal=False)
    _repin(spec, root, name)
    with pytest.raises(CampaignReconciliationError, match="payload_sha256 does not match"):
        _run(root, spec)


def test_spec_and_input_pins_reject_drift(tmp_path: Path) -> None:
    """Reject drift in either an input byte stream or the binding spec."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    with (root / "derived/channels.npz").open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(CampaignReconciliationError, match="file_sha256 does not match"):
        _run(root, spec)

    _rewrite(spec, lambda payload: payload.update({"dataset_id": "changed"}), reseal=False)
    with pytest.raises(CampaignReconciliationError, match="payload_sha256 does not match"):
        _run(root, spec)


@pytest.mark.parametrize(
    "bad_path", ["../material.json", "/tmp/material.json", "C:\\material.json", "derived\\replay.json"]
)
def test_spec_paths_are_confined(tmp_path: Path, bad_path: str) -> None:
    """Reject every input path that can escape the campaign root."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(spec, lambda payload: payload["inputs"]["material_manifest"].update({"path": bad_path}))
    with pytest.raises(CampaignReconciliationError, match="confined POSIX relative path"):
        _run(root, spec)


def test_duplicate_json_keys_and_non_object_roots_fail_closed(tmp_path: Path) -> None:
    """Reject ambiguous duplicate keys and non-object JSON roots."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    bad = root / "material.json"
    bad.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    _repin(spec, root, "material_manifest")
    with pytest.raises(CampaignReconciliationError, match="duplicate JSON key"):
        _run(root, spec)
    bad.write_text("[]", encoding="utf-8")
    _repin(spec, root, "material_manifest")
    with pytest.raises(CampaignReconciliationError, match="root must be an object"):
        _run(root, spec)


@pytest.mark.parametrize(
    ("relative", "name", "mutate", "message"),
    [
        ("material.json", "material_manifest", lambda p: p.update({"n_requested": 4}), "n_requested"),
        ("material.json", "material_manifest", lambda p: p.update({"n_acquired": 1}), "n_acquired"),
        ("material.json", "material_manifest", lambda p: p.update({"total_bytes": 1}), "total_bytes"),
        (
            "derived/replay.json",
            "replay_report",
            lambda p: p.update({"n_derived": 2}),
            "n_derived",
        ),
        ("dataset/report.json", "dataset_report", lambda p: p.update({"n_shots": 2}), "n_shots"),
        (
            "dataset/report.json",
            "dataset_report",
            lambda p: p.update({"n_disruptive": 0}),
            "n_disruptive",
        ),
        (
            "dataset/report.json",
            "dataset_report",
            lambda p: p.update({"dataset_sha256": "0" * 64}),
            "dataset_sha256",
        ),
    ],
)
def test_inventory_counters_and_fingerprints_are_recomputed(
    tmp_path: Path,
    relative: str,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Recompute declared counters and fingerprints from bound inventories."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(root / relative, mutate)
    _repin(spec, root, name)
    with pytest.raises(CampaignReconciliationError, match=message):
        _run(root, spec)


def test_dataset_file_and_manifest_inventory_are_verified(tmp_path: Path) -> None:
    """Verify dataset artifact bytes and agreement between both inventories."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    with (root / "dataset/shot_101.npz").open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(CampaignReconciliationError, match="checksum mismatch"):
        _run(root, spec)

    root = tmp_path / "second"
    spec = _tree(root)
    _rewrite(root / "dataset/manifest.json", lambda p: p["artifacts"][0].update({"checksum_sha256": "0" * 64}))
    _repin(spec, root, "dataset_manifest")
    with pytest.raises(CampaignReconciliationError, match="inventories differ"):
        _run(root, spec)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_identifiers, "unique members and shot_ids"),
        (_drop_channel, "shot/channel inventory"),
        (_add_channel, "shot/channel inventory"),
        (_non_finite_channel, "non-finite"),
        (_short_channel, "channel lengths differ"),
        (_integer_channel, "floating-point vector"),
        (_float_identifiers, "integer vector"),
    ],
)
def test_channels_archive_structure_is_verified(
    tmp_path: Path,
    mutate: Callable[[dict[str, np.ndarray[Any, Any]]], None],
    message: str,
) -> None:
    """Reject missing, extra, non-finite, incorrectly shaped, or incorrectly typed members."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _write_channels(root / "derived/channels.npz", mutate=mutate)
    _repin(spec, root, "channels_archive")
    with pytest.raises(CampaignReconciliationError, match=message):
        _run(root, spec)


def test_channels_archive_identity_and_container_are_verified(tmp_path: Path) -> None:
    """Reject duplicate/mismatched shot identities and malformed NPZ bytes."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _write_channels(root / "derived/channels.npz", shot_ids=(101, 101))
    _repin(spec, root, "channels_archive")
    with pytest.raises(CampaignReconciliationError, match="unique positive integers"):
        _run(root, spec)

    root = tmp_path / "second"
    spec = _tree(root)
    _write_channels(root / "derived/channels.npz", shot_ids=(102,))
    _repin(spec, root, "channels_archive")
    with pytest.raises(CampaignReconciliationError, match="differs from the replay report"):
        _run(root, spec)

    root = tmp_path / "third"
    spec = _tree(root)
    (root / "derived/channels.npz").write_bytes(b"not-an-npz")
    _repin(spec, root, "channels_archive")
    with pytest.raises(CampaignReconciliationError, match="cannot validate channels archive"):
        _run(root, spec)


def test_set_mismatches_are_reported_without_claim_promotion(tmp_path: Path) -> None:
    """Expose cross-surface shot mismatch while every claim remains false."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(
        root / "evaluation/report.json",
        lambda payload: payload.update({"shots": [{"shot_id": "shot_102"}]}),
    )
    _repin(spec, root, "evaluation_report")
    report = _run(root, spec)
    assert "evaluation_shot_set_differs_from_dataset" in report["blockers"]
    cross = cast(dict[str, Any], report["cross_inventory"])
    assert cross["evaluation_without_dataset"] == [102]
    assert cross["dataset_without_evaluation"] == [101]
    assert set(cast(dict[str, bool], report["claim_boundary"]).values()) == {False}


def test_empty_timestamp_and_missing_root_are_rejected(tmp_path: Path) -> None:
    """Reject missing provenance time and an absent campaign root."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    with pytest.raises(CampaignReconciliationError, match="generated_at"):
        reconcile_campaign(campaign_root=root, spec_path=spec, generated_at="")
    with pytest.raises(CampaignReconciliationError, match="campaign_root"):
        reconcile_campaign(campaign_root=tmp_path / "missing", spec_path=spec, generated_at=_FIXED_TS)


def test_spec_requires_exact_surface_set_and_unique_paths(tmp_path: Path) -> None:
    """Require the six named surfaces to resolve to six unique files."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(spec, lambda payload: payload["inputs"].pop(INPUT_NAMES[0]))
    with pytest.raises(CampaignReconciliationError, match="exact reconciliation surfaces"):
        _run(root, spec)

    spec = _tree(tmp_path / "second")
    _rewrite(
        spec,
        lambda payload: payload["inputs"]["dataset_report"].update(
            {"path": payload["inputs"]["dataset_manifest"]["path"]}
        ),
    )
    with pytest.raises(CampaignReconciliationError, match="paths must be unique"):
        _run(tmp_path / "second", spec)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"extra": True}), "fields do not match"),
        (lambda p: p.update({"schema_version": "wrong"}), "unsupported reconciliation spec schema"),
        (lambda p: p.update({"dataset_id": ""}), "dataset_id must be a non-empty string"),
        (lambda p: p.update({"inputs": []}), "inputs must be an object"),
        (
            lambda p: p["inputs"]["material_manifest"].update({"extra": True}),
            "inputs.material_manifest fields do not match",
        ),
        (
            lambda p: p["inputs"]["material_manifest"].update({"file_sha256": "BAD"}),
            "must be a lowercase SHA-256",
        ),
    ],
)
def test_spec_scalar_and_shape_contracts_fail_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject malformed spec scalars, containers, fields, and digests."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(spec, mutate)
    with pytest.raises(CampaignReconciliationError, match=message):
        _run(root, spec)


@pytest.mark.parametrize(
    ("relative", "name", "mutate", "message"),
    [
        ("material.json", "material_manifest", lambda p: p.update({"schema_version": "wrong"}), "material schema"),
        ("material.json", "material_manifest", lambda p: p.update({"shots": {}}), "shots must be an array"),
        ("material.json", "material_manifest", lambda p: p["shots"].append(p["shots"][0]), "duplicate material"),
        ("material.json", "material_manifest", lambda p: p["shots"][0].update({"shot_id": True}), "integer >= 1"),
        ("material.json", "material_manifest", lambda p: p["shots"][0].update({"status": "unknown"}), "unsupported"),
        (
            "derived/replay.json",
            "replay_report",
            lambda p: p.update({"schema_version": "wrong"}),
            "replay report schema",
        ),
        ("derived/replay.json", "replay_report", lambda p: p.update({"synthetic": True}), "synthetic:false"),
        ("derived/replay.json", "replay_report", lambda p: p.update({"channel_schema": []}), "eleven-channel"),
        (
            "derived/replay.json",
            "replay_report",
            lambda p: p["shots"].append(p["shots"][0]),
            "duplicate replay",
        ),
        (
            "derived/replay.json",
            "replay_report",
            lambda p: p["shots"][0].update({"status": "unknown"}),
            "unsupported",
        ),
        (
            "dataset/report.json",
            "dataset_report",
            lambda p: p.update({"schema_version": "wrong"}),
            "dataset report schema",
        ),
        ("dataset/report.json", "dataset_report", lambda p: p.update({"status": "ready"}), "blocked and non-synthetic"),
        ("dataset/report.json", "dataset_report", lambda p: p["shots"].append(p["shots"][0]), "duplicate dataset"),
        (
            "dataset/report.json",
            "dataset_report",
            lambda p: p["shots"][0].update({"npz": "shot_999.npz"}),
            "mismatched NPZ",
        ),
        ("dataset/report.json", "dataset_report", lambda p: p["shots"][0].update({"label": 2}), "zero or one"),
        (
            "evaluation/report.json",
            "evaluation_report",
            lambda p: p.update({"schema_version": "wrong"}),
            "evaluation report schema",
        ),
        ("evaluation/report.json", "evaluation_report", lambda p: p.update({"status": "ready"}), "identity/status"),
        (
            "evaluation/report.json",
            "evaluation_report",
            lambda p: p.update({"admission_ready": True}),
            "admission_ready:false",
        ),
        (
            "evaluation/report.json",
            "evaluation_report",
            lambda p: p["claim_boundary"].update({"public_claim_allowed": True}),
            "claims false",
        ),
        (
            "evaluation/report.json",
            "evaluation_report",
            lambda p: p.update({"shots": [{"shot_id": "bad"}]}),
            "shot_id is malformed",
        ),
        (
            "evaluation/report.json",
            "evaluation_report",
            lambda p: p["shots"].append(p["shots"][0]),
            "duplicate evaluation",
        ),
    ],
)
def test_surface_contract_mutations_fail_closed(
    tmp_path: Path,
    relative: str,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject malformed material, replay, dataset, and evaluation surfaces."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(root / relative, mutate)
    _repin(spec, root, name)
    with pytest.raises(CampaignReconciliationError, match=message):
        _run(root, spec)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"schema_version": "wrong"}), "dataset manifest is invalid"),
        (lambda p: p.update({"machine": "JET"}), "identity must be"),
        (lambda p: p["artifacts"][0].update({"uri": "other.npz"}), "shot_<id>"),
        (lambda p: p["artifacts"].append(p["artifacts"][0]), "duplicate dataset-manifest"),
        (
            lambda p: (p.update({"artifacts": []}), p.update({"checksum_sha256": "0" * 64})),
            "must enumerate dataset artefacts",
        ),
    ],
)
def test_dataset_manifest_contracts_fail_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject invalid, misidentified, unnamed, duplicate, and empty artifacts."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(root / "dataset/manifest.json", mutate)
    _repin(spec, root, "dataset_manifest")
    with pytest.raises(CampaignReconciliationError, match=message):
        _run(root, spec)


def test_missing_and_escaping_input_paths_fail_closed(tmp_path: Path) -> None:
    """Reject absent input files and symlinks that resolve outside the root."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    missing = root / "derived/channels.npz"
    missing.unlink()
    with pytest.raises(CampaignReconciliationError, match="does not resolve to a file"):
        _run(root, spec)

    root = tmp_path / "second"
    spec = _tree(root)
    outside = tmp_path / "outside.npz"
    outside.write_bytes((root / "derived/channels.npz").read_bytes())
    (root / "derived/channels.npz").unlink()
    (root / "derived/channels.npz").symlink_to(outside)
    with pytest.raises(CampaignReconciliationError, match="escapes the campaign root"):
        _run(root, spec)


def test_unreadable_spec_is_normalised_to_domain_error(tmp_path: Path) -> None:
    """Normalise filesystem read failures to the reconciliation error type."""
    with pytest.raises(CampaignReconciliationError, match="cannot read pinned input"):
        reconciliation._read_bytes(tmp_path / "absent.json")


def test_replay_dataset_identity_and_migration_failures_are_normalised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject cross-file identity drift and normalise migration failures."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(root / "derived/replay.json", lambda p: p.update({"channels_npz": "other.npz"}))
    _repin(spec, root, "replay_report")
    with pytest.raises(CampaignReconciliationError, match="pinned channels archive"):
        _run(root, spec)

    root = tmp_path / "second"
    spec = _tree(root)
    _rewrite(root / "dataset/report.json", lambda p: p.update({"dataset_id": "other"}))
    _repin(spec, root, "dataset_report")
    with pytest.raises(CampaignReconciliationError, match="dataset_id mismatch"):
        _run(root, spec)

    root = tmp_path / "third"
    spec = _tree(root)

    def _fail_migration(payload: dict[str, Any], *, artifact_root: Path) -> dict[str, Any]:
        raise SourceObjectManifestError("fixture failure")

    monkeypatch.setattr(reconciliation, "migrate_material_manifest_v1", _fail_migration)
    with pytest.raises(CampaignReconciliationError, match="legacy material migration failed"):
        _run(root, spec)

    root = tmp_path / "fourth"
    spec = _tree(root)

    def _wrong_migration(payload: dict[str, Any], *, artifact_root: Path) -> dict[str, Any]:
        return {"schema_version": "wrong"}

    monkeypatch.setattr(reconciliation, "migrate_material_manifest_v1", _wrong_migration)
    with pytest.raises(CampaignReconciliationError, match="source-object manifest v2"):
        _run(root, spec)


def test_fair_licences_remove_only_policy_drift_blockers(tmp_path: Path) -> None:
    """Remove legacy licence blockers while preserving lineage blockers."""
    root = tmp_path / "campaign01"
    spec = _tree(root, valid_evaluation_digest=True)
    for relative, name in (
        ("material.json", "material_manifest"),
        ("dataset/manifest.json", "dataset_manifest"),
        ("evaluation/report.json", "evaluation_report"),
    ):
        _rewrite(root / relative, lambda p: p.update({"licence": "CC-BY-SA-4.0"}))
        if name == "evaluation_report":
            _rewrite(
                root / relative,
                lambda p: p["data_provenance"].update({"licence": "CC-BY-SA-4.0"}),
            )
        _repin(spec, root, name)
    report = _run(root, spec)
    assert not any("licence" in blocker for blocker in report["blockers"])
    assert cast(dict[str, Any], report["licence_reconciliation"])["dataset_manifest_action"] == "none"


def test_cross_inventory_anomalies_are_explicit_and_never_promoted(tmp_path: Path) -> None:
    """Report source, replay, dataset, and exclusion mismatches as blockers."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(
        root / "derived/replay.json",
        lambda p: p.update(
            {
                "n_derived": 2,
                "shots": [
                    {"shot_id": 101, "status": "derived", "n_samples": 3},
                    {"shot_id": 104, "status": "derived", "n_samples": 3},
                ],
            }
        ),
    )
    _repin(spec, root, "replay_report")
    _write_channels(root / "derived/channels.npz", shot_ids=(101, 104))
    _repin(spec, root, "channels_archive")
    report = _run(root, spec)
    assert {
        "replay_contains_shots_absent_from_acquired_material",
        "dataset_shot_set_differs_from_replay_derived_set",
        "acquired_shots_missing_without_digest_bound_replay_failure",
    }.issubset(report["blockers"])
    cross = cast(dict[str, Any], report["cross_inventory"])
    assert cross["replay_without_acquired"] == [104]
    assert cross["exclusions"][0]["reason_evidence"] == "unverified"
    assert set(cast(dict[str, bool], report["claim_boundary"]).values()) == {False}


def test_replay_failure_records_outside_material_are_blocked(tmp_path: Path) -> None:
    """Expose replay-failure identities that never existed in acquired material."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(
        root / "derived/replay.json",
        lambda p: p.update(
            {
                "shots": [
                    {"shot_id": 101, "status": "derived", "n_samples": 3},
                    {"shot_id": 999, "status": "failed", "error": "not acquired"},
                ]
            }
        ),
    )
    _repin(spec, root, "replay_report")
    report = _run(root, spec)
    assert "replay_failure_records_absent_from_acquired_material" in report["blockers"]
    cross = cast(dict[str, Any], report["cross_inventory"])
    assert cross["replay_failures_without_acquired"] == [999]


def test_malformed_optional_evaluation_digest_stays_historical(tmp_path: Path) -> None:
    """Treat a malformed historical self-digest as another explicit blocker."""
    root = tmp_path / "campaign01"
    spec = _tree(root)
    _rewrite(
        root / "evaluation/report.json",
        lambda payload: payload.update({"payload_sha256": None}),
        reseal=False,
    )
    _repin(spec, root, "evaluation_report")
    report = _run(root, spec)
    assert "evaluation_report_self_digest_invalid" in report["blockers"]
