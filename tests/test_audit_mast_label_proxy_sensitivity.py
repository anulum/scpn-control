# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for MAST label/proxy sensitivity audit
"""Adversarial tests for the fail-closed L2F-22 sensitivity gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.audit_mast_label_proxy_sensitivity import (
    AUDIT_SCHEMA,
    PROGRAMME_LABEL_MANIFEST_SCHEMA,
    DetectorPoint,
    LabelProxyAuditError,
    _load_json,
    _parse_point,
    audit_label_proxy_sensitivity,
    main,
)
from validation.mast_source_object_manifest import canonical_json_sha256, file_sha256

_FIXED_TS = "2026-07-22T15:30:00Z"


def _ip_fast_quench() -> NDArray[np.float64]:
    values = np.ones(40, dtype=np.float64)
    values[:5] = np.linspace(0.0, 1.0, 5)
    values[20:24] = np.array([0.79, 0.55, 0.3, 0.1])
    values[24:] = 0.0
    return values


def _ip_no_quench() -> NDArray[np.float64]:
    return np.ones(40, dtype=np.float64)


def _time() -> NDArray[np.float64]:
    return np.arange(40, dtype=np.float64) * 1.0e-3


def _write_npz(path: Path, ip_ma: NDArray[np.float64], *, include_time: bool = True) -> None:
    payload: dict[str, NDArray[np.float64]] = {"Ip_MA": ip_ma}
    if include_time:
        payload["time_s"] = _time()
    np.savez(path, **payload)  # type: ignore[arg-type]  # NumPy stub treats the kwarg splat as allow_pickle.


def _write_dataset(
    root: Path,
    *,
    ips: tuple[NDArray[np.float64], ...] = (_ip_fast_quench(), _ip_no_quench()),
) -> tuple[Path, Path]:
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for shot_id, ip_ma in enumerate(ips, start=101):
        artifact = dataset_dir / f"shot_{shot_id}.npz"
        _write_npz(artifact, ip_ma)
        records.append(
            {
                "shot_id": shot_id,
                "npz": artifact.name,
                "checksum_sha256": file_sha256(artifact),
            }
        )
    report: dict[str, object] = {
        "schema_version": "scpn-control.mast-disruption-supervised-dataset.v1",
        "status": "blocked",
        "dataset_id": "test-campaign",
        "synthetic": False,
        "label_algorithm": {"drop_fraction": 0.8, "quench_window_ms": 5.0},
        "n_shots": len(records),
        "shots": records,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    report_path = root / "dataset.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return report_path, dataset_dir


def _rewrite_payload(path: Path, mutate: Any) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def _write_programme_manifest(root: Path, labels: list[dict[str, object]]) -> Path:
    payload: dict[str, object] = {
        "schema_version": PROGRAMME_LABEL_MANIFEST_SCHEMA,
        "machine": "MAST",
        "authority": "programme_metadata",
        "source_uri": "urn:test:programme-log",
        "labels": labels,
        "payload_sha256": None,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    path = root / "programme.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _audit(root: Path, **updates: Any) -> dict[str, object]:
    report_path, dataset_dir = _write_dataset(root)
    arguments: dict[str, object] = {
        "dataset_report_path": report_path,
        "dataset_dir": dataset_dir,
        "generated_at": _FIXED_TS,
        "parameter_grid": ((0.8, 2.5), (0.8, 5.0)),
        "reference_point": (0.8, 5.0),
    }
    arguments.update(updates)
    return audit_label_proxy_sensitivity(**arguments)  # type: ignore[arg-type]


def test_audit_verifies_sources_and_reports_proxy_instability(tmp_path: Path) -> None:
    """Quantify reference disagreement without inventing programme labels."""
    report = _audit(tmp_path)
    assert report["schema_version"] == AUDIT_SCHEMA
    assert report["status"] == "blocked"
    assert report["source_dataset_schema"] == "scpn-control.mast-disruption-supervised-dataset.v1"
    assert report["source_label_algorithm_sha256"] == canonical_json_sha256(
        {"drop_fraction": 0.8, "quench_window_ms": 5.0}
    )
    assert report["verified_shot_artifact_count"] == 2
    assert report["stable_shot_count"] == 1
    assert report["unstable_shot_count"] == 1
    assert report["unstable_shot_ids"] == [101]
    points = cast(list[dict[str, object]], report["sensitivity_points"])
    assert points[0]["disagreement_from_reference_count"] == 1
    assert points[1]["disagreement_from_reference_count"] == 0
    comparison = cast(dict[str, object], report["programme_comparison"])
    assert comparison["status"] == "not_computable"
    assert comparison["reason"] == "programme_label_manifest_not_supplied"
    assert comparison["disagreement_rate"] is None
    assert comparison["independent_validation_claim"] is False
    assert set(cast(dict[str, object], report["claim_boundary"]).values()) == {False}
    payload_sha = report["payload_sha256"]
    assert payload_sha == canonical_json_sha256({**report, "payload_sha256": None})
    encoded = json.dumps(report)
    assert "Ip_MA" not in encoded
    assert "time_s" not in encoded


def test_programme_comparison_is_descriptive_and_tracks_unmatched_labels(tmp_path: Path) -> None:
    """Compare only explicit expectations and retain the weak authority boundary."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    programme = _write_programme_manifest(
        tmp_path,
        [
            {"shot_id": 101, "programme_class": "forced_vde", "proxy_expectation": "disruption"},
            {"shot_id": 102, "programme_class": "control", "proxy_expectation": "disruption"},
            {"shot_id": 999, "programme_class": "other", "proxy_expectation": "no_expectation"},
        ],
    )
    report = audit_label_proxy_sensitivity(
        dataset_report_path=report_path,
        dataset_dir=dataset_dir,
        generated_at=_FIXED_TS,
        parameter_grid=((0.8, 5.0),),
        programme_labels_path=programme,
    )
    comparison = cast(dict[str, object], report["programme_comparison"])
    assert comparison["status"] == "descriptive_only"
    assert comparison["authority"] == "programme_metadata"
    assert comparison["independent_of_ip_features"] is False
    assert comparison["comparison_count"] == 2
    assert comparison["disagreement_count"] == 1
    assert comparison["disagreement_rate"] == 0.5
    assert comparison["disagreement_shot_ids"] == [102]
    assert comparison["unmatched_programme_shot_ids"] == [999]
    assert comparison["dataset_shots_without_programme_label"] == []
    assert comparison["manifest_file_sha256"] == file_sha256(programme)


def test_programme_manifest_without_comparable_expectation_stays_not_computable(tmp_path: Path) -> None:
    """Do not convert programme class alone into an outcome expectation."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    programme = _write_programme_manifest(
        tmp_path,
        [{"shot_id": 101, "programme_class": "forced_vde", "proxy_expectation": "no_expectation"}],
    )
    report = audit_label_proxy_sensitivity(
        dataset_report_path=report_path,
        dataset_dir=dataset_dir,
        generated_at=_FIXED_TS,
        parameter_grid=((0.8, 5.0),),
        programme_labels_path=programme,
    )
    comparison = cast(dict[str, object], report["programme_comparison"])
    assert comparison["status"] == "not_computable"
    assert comparison["reason"] == "no_matched_programme_labels_with_declared_proxy_expectation"
    assert comparison["comparison_count"] == 0


@pytest.mark.parametrize(
    ("points", "reference", "message"),
    [
        ((), (0.8, 5.0), "must not be empty"),
        (((0.8, 5.0), (0.8, 5.0)), (0.8, 5.0), "duplicate"),
        (((0.0, 5.0),), (0.0, 5.0), "drop_fraction"),
        (((float("nan"), 5.0),), (0.8, 5.0), "drop_fraction"),
        (((0.8, 0.0),), (0.8, 0.0), "quench_window_ms"),
        (((0.8, float("inf")),), (0.8, 5.0), "quench_window_ms"),
        (((0.8, 2.5),), (0.8, 5.0), "reference point"),
    ],
)
def test_parameter_grid_rejects_ambiguous_contracts(
    tmp_path: Path,
    points: tuple[tuple[float, float], ...],
    reference: tuple[float, float],
    message: str,
) -> None:
    """Reject invalid, duplicated, or unreferenced detector points."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    with pytest.raises(LabelProxyAuditError, match=message):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=points,
            reference_point=reference,
        )


def test_generated_timestamp_and_empty_dataset_are_required(tmp_path: Path) -> None:
    """Reject an unlabelled run time and a zero-shot report."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    with pytest.raises(LabelProxyAuditError, match="generated_at"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at="",
            parameter_grid=((0.8, 5.0),),
        )
    _rewrite_payload(report_path, lambda payload: payload.update({"n_shots": 0, "shots": []}))
    with pytest.raises(LabelProxyAuditError, match="contains no shots"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )


def test_json_loader_rejects_invalid_roots_duplicates_and_encoding(tmp_path: Path) -> None:
    """Fail closed before accepting ambiguous JSON bytes."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(LabelProxyAuditError, match="cannot read verified JSON"):
        _load_json(invalid)
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(LabelProxyAuditError, match="root must be an object"):
        _load_json(invalid)
    invalid.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(LabelProxyAuditError, match="duplicate JSON key"):
        _load_json(invalid)
    invalid.write_bytes(b"\xff")
    with pytest.raises(LabelProxyAuditError, match="cannot read verified JSON"):
        _load_json(invalid)
    with pytest.raises(LabelProxyAuditError, match="cannot read verified JSON"):
        _load_json(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"schema_version": "test.dataset.v0"}), "unsupported dataset report"),
        (lambda payload: payload.update({"synthetic": True}), "synthetic:false"),
        (lambda payload: payload.update({"status": "ready"}), "status:blocked"),
        (lambda payload: payload.update({"label_algorithm": []}), "label_algorithm must be an object"),
        (
            lambda payload: payload["label_algorithm"].update({"drop_fraction": True}),
            "drop_fraction must be numeric",
        ),
        (
            lambda payload: payload["label_algorithm"].update({"quench_window_ms": "five"}),
            "quench_window_ms must be numeric",
        ),
        (
            lambda payload: payload["label_algorithm"].update({"drop_fraction": 0.7}),
            "reference point does not match",
        ),
        (lambda payload: payload.pop("dataset_id"), "dataset_id"),
        (lambda payload: payload.update({"n_shots": True}), "n_shots must be an integer"),
        (lambda payload: payload.update({"n_shots": 3}), "shot count"),
        (lambda payload: payload.update({"shots": {}}), "shot count"),
        (lambda payload: payload["shots"].__setitem__(0, "bad"), "record must be an object"),
        (lambda payload: payload["shots"][0].update({"shot_id": True}), "shot_id must be an integer"),
        (lambda payload: payload["shots"][0].update({"shot_id": -1}), "invalid or duplicate"),
        (lambda payload: payload["shots"][1].update({"shot_id": 101}), "invalid or duplicate"),
        (lambda payload: payload["shots"][1].update({"npz": "shot_101.npz"}), "duplicate dataset artifact"),
        (lambda payload: payload["shots"][0].update({"npz": "../shot.npz"}), "unsafe shot artifact"),
        (lambda payload: payload["shots"][0].update({"checksum_sha256": "BAD"}), "lowercase SHA-256"),
    ],
)
def test_dataset_report_rejects_malformed_records(tmp_path: Path, mutate: Any, message: str) -> None:
    """Reject malformed, duplicate, and unsafe dataset declarations."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    _rewrite_payload(report_path, mutate)
    with pytest.raises(LabelProxyAuditError, match=message):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )


def test_dataset_report_and_artifact_integrity_are_mandatory(tmp_path: Path) -> None:
    """Reject report-payload and NPZ-byte substitution."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["dataset_id"] = "tampered"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelProxyAuditError, match="payload_sha256 does not match"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )
    report_path, dataset_dir = _write_dataset(tmp_path / "second")
    with (dataset_dir / "shot_101.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(LabelProxyAuditError, match="checksum mismatch"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )


def test_artifact_resolution_rejects_missing_and_symlink_escape(tmp_path: Path) -> None:
    """Keep every declared NPZ confined to the exact dataset directory."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    artifact = dataset_dir / "shot_101.npz"
    artifact.unlink()
    with pytest.raises(LabelProxyAuditError, match="cannot resolve shot artifact"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )

    report_path, dataset_dir = _write_dataset(tmp_path / "symlink")
    artifact = dataset_dir / "shot_101.npz"
    artifact.unlink()
    outside = tmp_path / "outside.npz"
    _write_npz(outside, _ip_fast_quench())
    artifact.symlink_to(outside)
    _rewrite_payload(
        report_path,
        lambda payload: payload["shots"][0].update({"checksum_sha256": file_sha256(outside)}),
    )
    with pytest.raises(LabelProxyAuditError, match="escapes dataset directory"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )


def test_missing_proxy_arrays_and_invalid_npz_fail_closed(tmp_path: Path) -> None:
    """Reject artefacts that cannot supply the exact production proxy inputs."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    artifact = dataset_dir / "shot_101.npz"
    _write_npz(artifact, _ip_fast_quench(), include_time=False)
    _rewrite_payload(
        report_path,
        lambda payload: payload["shots"][0].update({"checksum_sha256": file_sha256(artifact)}),
    )
    with pytest.raises(LabelProxyAuditError, match="lacks Ip_MA or time_s"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )
    artifact.write_bytes(b"not an npz")
    _rewrite_payload(
        report_path,
        lambda payload: payload["shots"][0].update({"checksum_sha256": file_sha256(artifact)}),
    )
    with pytest.raises(LabelProxyAuditError, match="cannot load shot"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("source_uri"), "fields do not match"),
        (lambda payload: payload.update({"schema_version": "v0"}), "unsupported.*schema"),
        (lambda payload: payload.update({"machine": "JET"}), "machine must be MAST"),
        (lambda payload: payload.update({"authority": "facility"}), "authority must be"),
        (lambda payload: payload.update({"source_uri": ""}), "source_uri"),
        (lambda payload: payload.update({"labels": {}}), "labels must be an array"),
        (lambda payload: payload["labels"].__setitem__(0, {}), "record fields"),
        (lambda payload: payload["labels"][0].update({"shot_id": 0}), "shot_id must be positive"),
        (lambda payload: payload["labels"][0].update({"programme_class": "guessed"}), "programme_class"),
        (lambda payload: payload["labels"][0].update({"proxy_expectation": "maybe"}), "proxy_expectation"),
        (lambda payload: payload["labels"].append(dict(payload["labels"][0])), "duplicate programme label"),
    ],
)
def test_programme_manifest_rejects_authority_and_schema_drift(tmp_path: Path, mutate: Any, message: str) -> None:
    """Accept programme comparisons only from an explicit self-digested contract."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    programme = _write_programme_manifest(
        tmp_path,
        [{"shot_id": 101, "programme_class": "forced_vde", "proxy_expectation": "disruption"}],
    )
    _rewrite_payload(programme, mutate)
    with pytest.raises(LabelProxyAuditError, match=message):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
            programme_labels_path=programme,
        )


def test_programme_manifest_digest_tamper_is_rejected(tmp_path: Path) -> None:
    """Reject a weak-label byte edit without its matching payload digest."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    programme = _write_programme_manifest(tmp_path, [])
    payload = json.loads(programme.read_text(encoding="utf-8"))
    payload["source_uri"] = "urn:tampered"
    programme.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelProxyAuditError, match="payload_sha256 does not match"):
        audit_label_proxy_sensitivity(
            dataset_report_path=report_path,
            dataset_dir=dataset_dir,
            generated_at=_FIXED_TS,
            parameter_grid=((0.8, 5.0),),
            programme_labels_path=programme,
        )


@pytest.mark.parametrize("value", ["0.8", "a,5"])
def test_point_parser_rejects_malformed_values(value: str) -> None:
    """Keep CLI parameter syntax unambiguous."""
    with pytest.raises(Exception, match="point"):
        _parse_point(value)
    assert _parse_point("0.8,5") == (0.8, 5.0)


def test_main_writes_a_deterministic_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise the complete CLI boundary and its explicit status line."""
    report_path, dataset_dir = _write_dataset(tmp_path)
    output = tmp_path / "evidence" / "audit.json"
    assert (
        main(
            [
                "--dataset-report",
                str(report_path),
                "--dataset-dir",
                str(dataset_dir),
                "--generated-at",
                _FIXED_TS,
                "--point",
                "0.8,2.5",
                "--point",
                "0.8,5.0",
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["payload_sha256"] == canonical_json_sha256({**written, "payload_sha256": None})
    assert "1/2 unstable" in capsys.readouterr().out


def test_detector_point_key_is_canonical() -> None:
    """Keep parameter keys stable across equivalent float spellings."""
    point = DetectorPoint(0.80, 5.000)
    assert point.key == "drop=0.8;window_ms=5"
    assert point.to_dict() == {"drop_fraction": 0.8, "quench_window_ms": 5.0}
