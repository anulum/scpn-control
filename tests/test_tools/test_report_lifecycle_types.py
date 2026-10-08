# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lifecycle record mutation boundary.
"""Keep validated lifecycle fields frozen while documenting nested provenance."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from report_lifecycle_fixtures import AUDIT_AS_OF

from tools.validation_report_freshness import ROOT, build_validation_report_freshness_matrix


def test_loaded_record_preserves_frozen_fields_and_explicit_nested_mutability() -> None:
    """Field reassignment is refused; changing in-memory provenance never edits source bytes."""
    matrix = build_validation_report_freshness_matrix(ROOT / "validation/reports", as_of=AUDIT_AS_OF, max_age_days=21)
    report = next(report for report in matrix.reports if report.path.is_file())
    original = report.path.read_bytes()
    field = "public_claim_allowed"
    with pytest.raises(FrozenInstanceError):
        setattr(report.lifecycle, field, True)
    assert not report.lifecycle.public_claim_allowed
    report.lifecycle.provenance["host_id"] = "in-memory caller mutation"
    assert report.to_dict()["provenance"] == report.lifecycle.provenance
    assert report.path.read_bytes() == original
    assert matrix.current_admitted_reports == ()
