# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD declared campaign contract behaviour

"""Exercise actual stored OOD author identities, hashes and inclusive acceptance without installing calibration."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from test_gk_ood_calibration_validation import _valid_calibration_report

from validation.validate_gk_ood_calibration import validate_gk_ood_calibration


def declaration(**changes: Any) -> dict[str, Any]:
    """Copy the original test-only author campaign; no measured dataset or covariance is fabricated."""
    payload: dict[str, Any] = deepcopy(_valid_calibration_report())
    payload.update(changes)
    return payload


def inspect(tmp_path: Path, payload: object) -> dict[str, Any]:
    """Persist supplied JSON and call the public required campaign reader."""
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return validate_gk_ood_calibration(source, require_campaign_artifacts=True)


@pytest.mark.parametrize("payload", [None, [], 17, "campaign"])
def test_object_root_required(tmp_path: Path, payload: object) -> None:
    """Nonobject decoded roots yield a root finding with no accepted metadata flag."""
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["campaign_artifacts"] == 0
    assert report["errors"][0]["field"] == "root"
    assert report["public_claims"]["deployment_calibration_admitted"] is False


@pytest.mark.parametrize("field", ["campaign_id", "source", "evaluated_at"])
@pytest.mark.parametrize("value", [None, [], {}, True, "", "   "])
def test_required_identities(tmp_path: Path, field: str, value: object) -> None:
    """Every identity refuses wrong/empty types, including unhashable source labels, without crashing."""
    report = inspect(tmp_path, declaration(**{field: value}))
    assert report["status"] == "fail" and report["campaign_artifacts"] == 0
    assert field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "v1"), ("source", "mock"), ("feature_schema", []), ("feature_schema", ["beta_e"] * 10)],
)
def test_original_schema_sources_and_feature_order(tmp_path: Path, field: str, value: object) -> None:
    """Original v2 schema, source labels and exact feature ordering remain required."""
    report = inspect(tmp_path, declaration(**{field: value}))
    assert report["status"] == "fail" and field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("source", ["published_gk_campaign", "real_external_gk_campaign", "facility_gk_campaign"])
def test_original_accepted_metadata_and_hash_shape(tmp_path: Path, source: str) -> None:
    """All original source labels retain accepted metadata flags and independent canonical report hashes."""
    payload = declaration(source=source)
    report = inspect(tmp_path, payload)
    assert report["status"] == "pass" and report["campaign_artifacts"] == 1
    assert report["public_claims"]["deployment_calibration_admitted"] is True
    assert report["public_claims"]["full_gk_operating_envelope_admitted"] is False
    entry = report["entries"][0]
    assert entry["source"] == source
    assert (
        entry["canonical_payload_sha256"]
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )
    body = {**report, "payload_sha256": None}
    assert (
        report["payload_sha256"]
        == hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("line_endings", ["LF", "CRLF"])
def test_exact_captured_bytes_and_canonical_metadata(tmp_path: Path, line_endings: str) -> None:
    """Raw identity binds actual LF/CRLF bytes while canonical author metadata remains identical."""
    payload = declaration(unused="ž")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if line_endings == "CRLF":
        text = text.replace("\n", "\r\n")
    raw = text.encode("utf-8")
    source = tmp_path / "campaign.data"
    source.write_bytes(raw)
    report = validate_gk_ood_calibration(source, require_campaign_artifacts=True)
    assert report["status"] == "pass"
    assert report["entries"][0]["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        report["entries"][0]["canonical_payload_sha256"]
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )
    assert source.read_bytes() == raw


@pytest.mark.parametrize("field", ["false_positive_rate", "false_negative_rate", "ood_recall"])
def test_original_acceptance_comparisons(tmp_path: Path, field: str) -> None:
    """Original maximum false-rate and minimum recall comparisons refuse out-of-bound finite rates."""
    payload = declaration()
    payload["acceptance"][field] = 0.9 if field != "ood_recall" else 0.1
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["campaign_artifacts"] == 0
    assert field in {error["field"] for error in report["errors"]}


def test_inclusive_original_acceptance_boundaries(tmp_path: Path) -> None:
    """Equality with every declared acceptance bound remains admitted, including probability endpoints."""
    payload = declaration()
    payload["acceptance"] = {
        "false_positive_rate": 0,
        "false_negative_rate": 1,
        "max_false_positive_rate": 0,
        "max_false_negative_rate": 1,
        "ood_recall": 0,
        "min_ood_recall": 0,
    }
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["acceptance"]["ood_recall"] = payload["acceptance"]["min_ood_recall"] = 1
    assert inspect(tmp_path, payload)["status"] == "pass"


def test_duplicate_campaigns_and_mixed_findings(tmp_path: Path) -> None:
    """One valid campaign is counted once; duplicate IDs and malformed neighbors refuse overall metadata admission."""
    for name in ["a.json", "b.json"]:
        (tmp_path / name).write_text(json.dumps(declaration()))
    report = validate_gk_ood_calibration(tmp_path, require_campaign_artifacts=True)
    assert report["status"] == "fail" and report["campaign_artifacts"] == 1
    assert report["errors"][0]["field"] == "campaign_id"
    assert report["public_claims"]["deployment_calibration_admitted"] is False
    (tmp_path / "b.json").write_text("null")
    report = validate_gk_ood_calibration(tmp_path, require_campaign_artifacts=True)
    assert report["status"] == "fail" and report["campaign_artifacts"] == 1
    assert report["errors"][0]["field"] == "root"
