# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real Data Manifest Tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_control.core.real_data_manifest import (
    RealDataManifestError,
    load_real_data_manifest,
    validate_real_data_manifest,
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


def test_real_manifest_rejects_synthetic_source_kind() -> None:
    payload = _real_payload()
    payload["source"] = {
        "kind": "mock",
        "uri": "tests/mock_diiid.py",
        "access": "repository fixture",
    }

    with pytest.raises(RealDataManifestError, match="synthetic or mock source"):
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
        "uri": str(artefact),
        "access": "temporary test archive",
    }
    payload["checksum_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RealDataManifestError, match="checksum mismatch"):
        load_real_data_manifest(manifest_path, verify_artifact=True)
