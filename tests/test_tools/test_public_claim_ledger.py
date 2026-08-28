# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public claim admission ledger tests
"""Regression tests for fail-closed public-claim admission."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest import CaptureFixture

from tools import public_claim_ledger
from tools.public_claim_ledger import (
    DEFAULT_LIFECYCLE_REGISTRY,
    DEFAULT_REPORTS_ROOT,
    _claim_record,
    _ledger_from_matrix,
    _registry_metadata,
    build_public_claim_ledger,
    main,
)

AUDIT_AS_OF = datetime(2026, 8, 28, 3, 44, 33, tzinfo=timezone.utc)


def _registry(path: Path, source_commit: str = "1" * 40) -> bytes:
    payload = {"registry_source_commit": source_commit}
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _admitted_report(path: str, *, commit: str | None = "2" * 40) -> Any:
    lifecycle = SimpleNamespace(
        path=path,
        report_sha256="3" * 64,
        report_commit=commit,
        evidence_class="local_proxy",
        current_evidence=True,
        scientific_admission=True,
        production_admission=False,
        public_claim_allowed=True,
        claim_rationale="Fresh, independently admitted evidence.",
        provenance={"source_commit": "4" * 40, "dependency_lock_sha256": "5" * 64},
        refresh_artifact_path="validation/report_refreshes/2026-08-28/result.json",
        refresh_artifact_sha256="6" * 64,
    )
    return SimpleNamespace(
        evidence_time=datetime(2026, 8, 28, tzinfo=timezone.utc),
        lifecycle=lifecycle,
    )


def test_live_ledger_is_truthfully_empty() -> None:
    """The audited live corpus currently admits no public scientific claim."""
    ledger = build_public_claim_ledger(as_of=AUDIT_AS_OF)

    assert ledger["schema_version"] == "scpn-control.public-claim-ledger.v1"
    assert ledger["public_claim_count"] == 0
    assert ledger["claims"] == []


def test_ledger_sorts_admitted_reports_and_binds_registry(tmp_path: Path) -> None:
    """Admitted claims are stable, sorted, and bound to registry bytes."""
    registry_path = tmp_path / "registry.json"
    raw = _registry(registry_path)
    matrix = cast(
        Any,
        SimpleNamespace(
            max_age_days=21,
            current_admitted_reports=(
                _admitted_report("validation/reports/z.json"),
                _admitted_report("validation/reports/a.json"),
            ),
        ),
    )

    ledger = _ledger_from_matrix(matrix, registry_path=registry_path)

    assert [claim["report_path"] for claim in ledger["claims"]] == [
        "validation/reports/a.json",
        "validation/reports/z.json",
    ]
    assert ledger["generated_from"]["lifecycle_registry_sha256"] == hashlib.sha256(raw).hexdigest()
    assert ledger["claims"][0]["claim_boundary"]["public_claim_allowed"] is True


def test_claim_record_requires_immutable_report_commit() -> None:
    """A public claim cannot be admitted without its exact report commit."""
    with pytest.raises(ValueError, match="lacks immutable report commit"):
        _claim_record(_admitted_report("validation/reports/unbound.json", commit=None))


@pytest.mark.parametrize("payload", [[], {}, {"registry_source_commit": ""}])
def test_registry_metadata_rejects_malformed_payload(tmp_path: Path, payload: object) -> None:
    """Registry provenance is mandatory and fails closed."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        _registry_metadata(path)


def test_main_writes_checks_and_detects_drift(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI writes deterministically, checks equality, and rejects drift."""
    output = tmp_path / "ledger.json"
    common = [
        "--reports-root",
        str(DEFAULT_REPORTS_ROOT),
        "--registry",
        str(DEFAULT_LIFECYCLE_REGISTRY),
        "--output",
        str(output),
        "--as-of",
        "2026-08-28T03:44:33Z",
    ]

    assert main(common) == 0
    assert main([*common, "--check"]) == 0
    output.write_text("{}\n", encoding="utf-8")
    assert main([*common, "--check"]) == 1
    captured = capsys.readouterr()
    assert "claims=0" in captured.out
    assert "ledger drift" in captured.err


def test_main_reports_validation_failure(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """Validation errors produce a concise non-zero CLI result."""

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("broken lifecycle")

    monkeypatch.setattr(public_claim_ledger, "build_public_claim_ledger", fail)

    assert main([]) == 1
    assert "broken lifecycle" in capsys.readouterr().err
