# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Maintained refresh byte and claim bindings.
"""Refuse actual maintained refresh drift without manufacturing admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from report_lifecycle_fixtures import AUDIT_AS_OF, copy_corpus

from tools.validation_report_freshness import LifecycleRegistryError, build_validation_report_freshness_matrix


def test_maintained_refresh_digest_and_metadata_are_not_scientific_admission(tmp_path: Path) -> None:
    """Bound maintained refreshes retain their false scientific and public flags."""
    corpus = copy_corpus(tmp_path / "corpus")
    matrix = build_validation_report_freshness_matrix(
        corpus.reports,
        registry_path=corpus.registry,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    refreshed = [report for report in matrix.reports if report.lifecycle.refresh_status == "refreshed"]
    assert len(refreshed) == 10
    assert all(not report.lifecycle.scientific_admission for report in refreshed)
    assert all(not report.lifecycle.public_claim_allowed for report in refreshed)
    assert matrix.current_admitted_reports == ()
    name = refreshed[0].lifecycle.refresh_artifact_path
    assert name is not None
    artifact = corpus.root / name
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(LifecycleRegistryError, match="refresh artifact digest drift"):
        build_validation_report_freshness_matrix(
            corpus.reports,
            registry_path=corpus.registry,
            as_of=AUDIT_AS_OF,
            max_age_days=21,
        )


@pytest.mark.parametrize("drift", ["sealed-flags", "sealed-expiry-rationale"])
def test_public_builder_refuses_drift_in_actual_expired_refresh(tmp_path: Path, drift: str) -> None:
    """Changing maintained refresh declarations cannot weaken bound expiry caveats."""
    corpus = copy_corpus(tmp_path / "corpus")
    registry = cast(dict[str, object], json.loads(corpus.registry.read_bytes()))
    records = cast(list[dict[str, object]], registry["reports"])
    record = next(item for item in records if cast(dict[str, object], item["refresh"])["artifact_path"])
    refresh = cast(dict[str, object], record["refresh"])
    artifact = corpus.root / cast(str, refresh["artifact_path"])
    payload = cast(dict[str, object], json.loads(artifact.read_bytes()))
    claim = cast(dict[str, object], payload["claim_boundary"])
    if drift == "sealed-flags":
        claim["current_evidence"] = False
    else:
        claim["rationale"] = "Fresh local evidence without preserved caveats"
    artifact.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    refresh["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    corpus.registry.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    with pytest.raises(
        LifecycleRegistryError, match="refresh claim boundary drift|unsupported refresh expiry rationale"
    ):
        build_validation_report_freshness_matrix(
            corpus.reports,
            registry_path=corpus.registry,
            as_of=AUDIT_AS_OF,
            max_age_days=21,
        )
