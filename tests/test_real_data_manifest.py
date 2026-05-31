# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real Data Manifest Tests

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from scpn_control.core.real_data_manifest import (
    DataSourceManifest,
    RealDataManifest,
    RealDataManifestError,
    load_real_data_manifest,
    validate_real_data_manifest,
    verify_manifest_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "validation" / "reference_data" / "diiid" / "manifests"


def _real_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_id": "diii-d-163303-control-replay",
        "machine": "DIII-D",
        "shot": "163303",
        "synthetic": False,
        "source": {
            "kind": "mdsplus",
            "uri": "mdsplus://DIII-D/163303",
            "access": "facility-approved",
        },
        "retrieved_at": "2026-05-18T01:20:00Z",
        "checksum_sha256": "a" * 64,
        "licence": "facility data policy",
        "signals": [
            {
                "name": "plasma_current",
                "path": "\\\\IP",
                "units": "A",
                "timebase": "s",
            },
            {
                "name": "normalised_beta",
                "path": "\\\\BETAN",
                "units": "1",
                "timebase": "s",
            },
        ],
    }


def _synthetic_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_id": "ci-mock-diiid-999999",
        "machine": "DIII-D",
        "shot": "999999",
        "synthetic": True,
        "source": {
            "kind": "synthetic",
            "uri": "tests/mock_diiid.py",
            "access": "repository fixture",
        },
        "synthetic_generator": "tests.mock_diiid.generate_mock_shot",
        "synthetic_seed": 7,
        "signals": [
            {
                "name": "normalised_beta",
                "path": "beta_N",
                "units": "1",
                "timebase": "s",
            },
        ],
    }


def test_real_manifest_accepts_physical_provenance() -> None:
    manifest = validate_real_data_manifest(_real_payload())

    assert manifest.kind == "real"
    assert manifest.source.kind == "mdsplus"
    assert manifest.signals[0].units == "A"


def test_load_real_data_manifest_from_json(tmp_path) -> None:
    path = tmp_path / "real_manifest.json"
    path.write_text(json.dumps(_real_payload()), encoding="utf-8")

    manifest = load_real_data_manifest(path)

    assert manifest.dataset_id == "diii-d-163303-control-replay"


def test_load_real_data_manifest_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "duplicate_manifest.json"
    path.write_text(
        '{"schema_version":"1.0","dataset_id":"first","dataset_id":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(RealDataManifestError, match="duplicate JSON key: dataset_id"):
        load_real_data_manifest(path)


def test_load_real_data_manifest_rejects_non_object_root(tmp_path) -> None:
    path = tmp_path / "array_manifest.json"
    path.write_text(json.dumps(["not", "a", "manifest"]), encoding="utf-8")

    with pytest.raises(RealDataManifestError, match="manifest root must be a JSON object"):
        load_real_data_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"schema_version": "2.0"}), "unsupported manifest schema_version"),
        (lambda payload: payload.update({"synthetic": "false"}), "synthetic must be a boolean"),
        (lambda payload: payload.update({"source": "mdsplus://DIII-D/163303"}), "source must be an object"),
        (lambda payload: payload.update({"signals": []}), "signals must be a non-empty array"),
        (lambda payload: payload.update({"shot": "  "}), "shot must not be empty"),
        (lambda payload: payload.update({"dataset_id": ""}), "dataset_id must be a non-empty string"),
        (lambda payload: payload.update({"retrieved_at": ""}), "retrieved_at must be a non-empty string"),
    ],
)
def test_manifest_rejects_malformed_top_level_provenance(mutation, message) -> None:
    payload = _real_payload()
    mutation(payload)

    with pytest.raises(RealDataManifestError, match=message):
        validate_real_data_manifest(payload)


def test_manifest_rejects_non_object_signal() -> None:
    payload = _real_payload()
    payload["signals"] = ["not an object"]

    with pytest.raises(RealDataManifestError, match=r"signals\[0\] must be an object"):
        validate_real_data_manifest(payload)


@pytest.mark.parametrize(
    ("artifacts", "message"),
    [
        ("shot.npz", "artifacts must be an array"),
        (["shot.npz"], r"artifacts\[0\] must be an object"),
        ([{"uri": "shot.npz", "checksum_sha256": "ABC"}], r"artifacts\[0\]\.checksum_sha256"),
    ],
)
def test_manifest_rejects_malformed_artifact_entries(artifacts, message) -> None:
    payload = _real_payload()
    payload["artifacts"] = artifacts
    payload.pop("checksum_sha256")

    with pytest.raises(RealDataManifestError, match=message):
        validate_real_data_manifest(payload)


def test_real_manifest_rejects_synthetic_source_kind() -> None:
    payload = _real_payload()
    payload["source"] = {
        "kind": "mock",
        "uri": "tests/mock_diiid.py",
        "access": "repository fixture",
    }

    with pytest.raises(RealDataManifestError, match="synthetic or mock source"):
        validate_real_data_manifest(payload)


def test_real_manifest_rejects_unknown_source_kind() -> None:
    payload = _real_payload()
    source = deepcopy(payload["source"])
    assert isinstance(source, dict)
    source["kind"] = "spreadsheet"
    payload["source"] = source

    with pytest.raises(RealDataManifestError, match="real manifest source.kind must be one of"):
        validate_real_data_manifest(payload)


@pytest.mark.parametrize(("key", "message"), [("retrieved_at", "retrieved_at"), ("licence", "licence")])
def test_real_manifest_requires_experimental_provenance_fields(key: str, message: str) -> None:
    payload = _real_payload()
    payload.pop(key)

    with pytest.raises(RealDataManifestError, match=message):
        validate_real_data_manifest(payload)


def test_real_manifest_requires_checksum_and_licence() -> None:
    payload = _real_payload()
    payload.pop("checksum_sha256")

    with pytest.raises(RealDataManifestError, match="checksum_sha256"):
        validate_real_data_manifest(payload)


def test_real_manifest_rejects_arbitrary_units() -> None:
    payload = _real_payload()
    signals = payload["signals"]
    assert isinstance(signals, list)
    signal = signals[0]
    assert isinstance(signal, dict)
    signal["units"] = "arb"

    with pytest.raises(RealDataManifestError, match="physical units"):
        validate_real_data_manifest(payload)


def test_synthetic_manifest_rejects_real_source_kind() -> None:
    payload = _synthetic_payload()
    source = deepcopy(payload["source"])
    assert isinstance(source, dict)
    source["kind"] = "mdsplus"
    payload["source"] = source

    with pytest.raises(RealDataManifestError, match="synthetic manifest source.kind"):
        validate_real_data_manifest(payload)


def test_synthetic_manifest_requires_integer_seed() -> None:
    payload = _synthetic_payload()
    payload["synthetic_seed"] = "7"

    with pytest.raises(RealDataManifestError, match="synthetic_seed must be an integer"):
        validate_real_data_manifest(payload)


def test_synthetic_manifest_requires_generator_metadata() -> None:
    payload = _synthetic_payload()
    payload.pop("synthetic_generator")

    with pytest.raises(RealDataManifestError, match="synthetic_generator"):
        validate_real_data_manifest(payload)


def test_synthetic_manifest_accepts_ci_fixture_metadata() -> None:
    manifest = validate_real_data_manifest(_synthetic_payload())

    assert manifest.kind == "synthetic"
    assert manifest.synthetic_seed == 7


@pytest.mark.parametrize(
    ("filename", "kind", "source_kind"),
    [
        ("diiid_hmode_1p5MA.geqdsk.manifest.json", "real", "geqdsk"),
        ("shot_163303_hmode.npz.manifest.json", "real", "local_archive"),
        ("mock_diiid_ci.manifest.json", "synthetic", "synthetic"),
    ],
)
def test_repository_reference_manifests_validate(filename: str, kind: str, source_kind: str) -> None:
    manifest = load_real_data_manifest(MANIFEST_DIR / filename)

    assert manifest.kind == kind
    assert manifest.source.kind == source_kind


def test_verify_manifest_artifact_ignores_remote_real_sources(tmp_path) -> None:
    manifest = validate_real_data_manifest(_real_payload())

    assert verify_manifest_artifact(manifest, manifest_path=tmp_path / "manifest.json") is None


def test_verify_manifest_artifact_requires_checksum_for_local_source(tmp_path) -> None:
    manifest = RealDataManifest(
        schema_version="1.0",
        dataset_id="local-shot",
        machine="DIII-D",
        shot="163303",
        synthetic=False,
        source=DataSourceManifest(kind="local_archive", uri="shot.npz", access="local archive"),
        signals=validate_real_data_manifest(_real_payload()).signals,
        retrieved_at="2026-05-18T01:20:00Z",
        licence="facility data policy",
    )

    with pytest.raises(RealDataManifestError, match="artifact verification requires checksum_sha256"):
        verify_manifest_artifact(manifest, manifest_path=tmp_path / "manifest.json")


def test_repository_manifest_verifies_local_artifact_checksum() -> None:
    manifest = load_real_data_manifest(
        MANIFEST_DIR / "diiid_hmode_1p5MA.geqdsk.manifest.json",
        verify_artifact=True,
    )

    assert manifest.dataset_id == "diiid-hmode-1p5ma-geqdsk"


def test_manifest_artifact_verification_rejects_checksum_mismatch(tmp_path) -> None:
    artefact = tmp_path / "shot.npz"
    artefact.write_bytes(b"not the recorded shot")
    manifest_path = tmp_path / "manifest.json"
    payload = _real_payload()
    payload["source"] = {
        "kind": "local_archive",
        "uri": artefact.name,
        "access": "temporary test archive",
    }
    payload["checksum_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RealDataManifestError, match="checksum mismatch"):
        load_real_data_manifest(manifest_path, verify_artifact=True)


def test_manifest_artifact_list_verifies_each_local_checksum(tmp_path) -> None:
    artefact = tmp_path / "shot.npz"
    content = b"measured archive payload"
    artefact.write_bytes(content)
    payload = _real_payload()
    payload["source"] = {
        "kind": "local_archive",
        "uri": "manifested-by-artifacts",
        "access": "temporary test archive",
    }
    payload.pop("checksum_sha256")
    payload["artifacts"] = [{"uri": artefact.name, "checksum_sha256": sha256(content).hexdigest()}]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_real_data_manifest(manifest_path, verify_artifact=True)

    assert manifest.artifacts[0].uri == artefact.name


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("mdsplus://DIII-D/163303", "artifact verification requires a local URI"),
        ("/definitely/missing/scpn-control-artifact.npz", "artifact URI must be relative"),
        ("../escape.npz", "artifact URI must not contain parent traversal"),
        ("missing-shot.npz", "artifact file not found"),
    ],
)
def test_manifest_artifact_verification_rejects_unresolvable_local_artifacts(tmp_path, uri: str, message: str) -> None:
    manifest = RealDataManifest(
        schema_version="1.0",
        dataset_id="local-shot",
        machine="DIII-D",
        shot="163303",
        synthetic=False,
        source=DataSourceManifest(kind="local_archive", uri=uri, access="local archive"),
        signals=validate_real_data_manifest(_real_payload()).signals,
        retrieved_at="2026-05-18T01:20:00Z",
        checksum_sha256="0" * 64,
        licence="facility data policy",
    )

    with pytest.raises(RealDataManifestError, match=message):
        verify_manifest_artifact(manifest, manifest_path=tmp_path / "manifest.json")


def test_manifest_artifact_list_rejects_checksum_mismatch(tmp_path) -> None:
    artefact = tmp_path / "shot.npz"
    artefact.write_bytes(b"measured archive payload")
    payload = _real_payload()
    payload["source"] = {
        "kind": "local_archive",
        "uri": "manifested-by-artifacts",
        "access": "temporary test archive",
    }
    payload.pop("checksum_sha256")
    payload["artifacts"] = [{"uri": artefact.name, "checksum_sha256": "0" * 64}]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RealDataManifestError, match="checksum mismatch"):
        load_real_data_manifest(manifest_path, verify_artifact=True)
