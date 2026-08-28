# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Append-only validation refresh lineage tests.
"""Validate refresh schemas, self-seals, and immutable-source bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
REFRESH_ROOT = ROOT / "validation" / "report_refreshes"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_refresh_records_are_schema_valid_self_sealed_and_lineage_bound() -> None:
    """Every refresh is separate from, and digest-bound to, its source report."""
    schema = _json(ROOT / "validation" / "report_refresh.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    registry = _json(ROOT / "validation" / "report_lifecycle_registry.json")
    records = {
        record["refresh"]["artifact_path"]: record
        for record in registry["reports"]
        if record["refresh"]["status"] == "refreshed"
    }
    refresh_paths = sorted(REFRESH_ROOT.glob("*/*.json"))

    assert len(refresh_paths) == 10
    assert len(records) == 10
    assert len({_json(path)["lineage_id"] for path in refresh_paths}) == 10

    for path in refresh_paths:
        payload = _json(path)
        validator.validate(payload)
        relative_path = path.relative_to(ROOT).as_posix()
        record = records[relative_path]
        source = ROOT / payload["source_report"]["path"]
        self_seal = payload.pop("payload_sha256")

        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == payload["source_report"]["sha256"]
        assert payload["source_report"]["path"] == record["path"]
        assert payload["source_report"]["sha256"] == record["report_sha256"]
        assert payload["source_commit"] == record["provenance"]["source_commit"]
        assert payload["refresh_evidence_time_utc"] == record["refresh"]["evidence_time_utc"]
        assert payload["producer_command"] == record["refresh"]["commands"][0]
        assert payload["claim_boundary"] == record["claim_boundary"]
        assert payload["failures"] == record["provenance"]["failures"]
        assert payload["sampling"] == {
            "samples": record["provenance"]["samples"],
            "repeats": record["provenance"]["repeats"],
            "warmup": record["provenance"]["warmup"],
        }
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["refresh"]["artifact_sha256"]
        assert _canonical_sha256(payload) == self_seal
