# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Baseline promotion source API tests
"""Inspect complete real captured source buffers and malformed declared copies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from baseline_promotion_fixtures import make_corpus

from tools.baseline_promotion_payloads import PromotionInputError
from tools.baseline_promotion_records import load_promotion_source


def test_public_source_reader_binds_exact_captured_report_bytes(tmp_path: Path) -> None:
    """Captured source values and digest remain exactly bound through the public reader."""
    corpus = make_corpus(tmp_path / "corpus")
    before = corpus.manifest.read_bytes(), corpus.artifact.read_bytes()
    manifest, artifact, digest, report = load_promotion_source(
        corpus.manifest, "report", corpus.source_digest, corpus.root
    )
    assert artifact == corpus.artifact and digest == hashlib.sha256(before[1]).hexdigest()
    assert report == json.loads(before[1]) and manifest == json.loads(before[0])
    assert report["new_numerical_measurement"] is False
    assert before == (corpus.manifest.read_bytes(), corpus.artifact.read_bytes())


@pytest.mark.parametrize("raw", [b"[]", b'{"duplicated":1,"duplicated":2}', b'{"invalid":NaN}', b'{"invalid":1e999}'])
def test_public_reader_refuses_ambiguous_or_nonfinite_manifest(tmp_path: Path, raw: bytes) -> None:
    """The actual selected manifest file is refused before unsafe report interpretation."""
    corpus = make_corpus(tmp_path / "corpus")
    corpus.manifest.write_bytes(raw)
    with pytest.raises(PromotionInputError):
        load_promotion_source(corpus.manifest, "report", corpus.source_digest, corpus.root)
    assert corpus.manifest.read_bytes() == raw


@pytest.mark.parametrize("kind", ["artifacts-object", "artifact-scalar", "empty-path"])
def test_public_reader_refuses_invalid_artifact_binding_fields(tmp_path: Path, kind: str) -> None:
    """A self-digested malformed envelope cannot be used as a source binding."""
    corpus = make_corpus(tmp_path / "corpus")
    value = cast(dict[str, object], json.loads(corpus.manifest.read_bytes()))
    if kind == "artifacts-object":
        value["artifacts"] = {}
    elif kind == "artifact-scalar":
        value["artifacts"] = [1]
    else:
        cast(list[dict[str, object]], value["artifacts"])[0]["immutable_path"] = " "
    unsigned = {key: item for key, item in value.items() if key != "payload_sha256"}
    value["payload_sha256"] = hashlib.sha256(
        (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    corpus.manifest.write_text(json.dumps(value), encoding="utf-8")
    before = corpus.manifest.read_bytes(), corpus.artifact.read_bytes()
    with pytest.raises(PromotionInputError):
        load_promotion_source(corpus.manifest, "report", corpus.source_digest, corpus.root)
    assert before == (corpus.manifest.read_bytes(), corpus.artifact.read_bytes())
