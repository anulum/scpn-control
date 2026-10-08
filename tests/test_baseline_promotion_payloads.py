# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Baseline promotion payload API tests
"""Exercise consumed builder declarations using actual recorded historical values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from baseline_promotion_fixtures import make_corpus

from tools.baseline_promotion_payloads import PromotionInputError, build_baseline, canonical_digest, json_bytes


def test_builder_keeps_report_values_and_does_not_require_reader_envelope(tmp_path: Path) -> None:
    """The public builder accepts its consumed domain and preserves real historical metrics."""
    corpus = make_corpus(tmp_path / "corpus")
    report = cast(dict[str, object], json.loads(corpus.artifact.read_bytes()))
    del report["schema_version"]
    del report["payload_sha256"]
    result = build_baseline(
        report,
        suite="custody",
        source_manifest="selected-manifest",
        source_sha256=corpus.source_digest,
        authority_ref="copy-no-owner-approval",
        hardware_compatibility="initial-baseline",
        promoted_utc="2026-10-08T00:00:00Z",
    )
    assert result["benchmarks"] == report["benchmarks"]
    assert result["provenance"] == report["provenance"]
    assert result["production_claim_allowed"] is False
    assert json.loads(json_bytes(result)) == result
    assert result["baseline_sha256"] == canonical_digest(cast(dict[str, object], report["benchmarks"]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("benchmarks", {}),
        ("benchmarks", []),
        ("provenance", []),
        ("generated_utc", " "),
        ("evidence_class", None),
        ("benchmarks", {"invalid": float("nan")}),
    ],
)
def test_builder_refuses_invalid_consumed_fields(tmp_path: Path, field: str, value: object) -> None:
    """Malformed declared copies cannot produce a baseline with invalid required fields."""
    corpus = make_corpus(tmp_path / "corpus")
    report = cast(dict[str, object], json.loads(corpus.artifact.read_bytes()))
    report[field] = value
    with pytest.raises(PromotionInputError):
        build_baseline(
            report,
            suite="custody",
            source_manifest="selected",
            source_sha256=corpus.source_digest,
            authority_ref="copy",
            hardware_compatibility="initial-baseline",
            promoted_utc="now",
        )


def test_public_json_writer_refuses_nonfinite_or_non_json_declarations() -> None:
    """Public serialization cannot expose invalid JSON in source/history files."""
    with pytest.raises(PromotionInputError):
        json_bytes({"invalid": float("nan")})
    with pytest.raises(PromotionInputError):
        json_bytes({"invalid": object()})
