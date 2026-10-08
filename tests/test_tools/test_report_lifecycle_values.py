# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public lifecycle scalar and timestamp contracts.
"""Exercise scalar refusals through the two public inventory loaders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from report_lifecycle_fixtures import AUDIT_AS_OF, copy_corpus, run_cli

from tools.validation_report_freshness import (
    LifecycleRegistryError,
    build_validation_report_freshness_matrix,
    load_validation_report_lifecycle_registry,
    parse_datetime,
)


@pytest.mark.parametrize("value", [True, False, 1.0, float("nan"), "21", None, -1])
def test_public_loaders_refuse_noninteger_age_windows(tmp_path: Path, value: object) -> None:
    """Both public loaders refuse boolean, noninteger and negative scalar windows."""
    corpus = copy_corpus(tmp_path / "corpus")
    before = corpus.snapshot()
    for loader in (build_validation_report_freshness_matrix, load_validation_report_lifecycle_registry):
        with pytest.raises(LifecycleRegistryError, match="max_age_days must be a non-negative integer"):
            if loader is build_validation_report_freshness_matrix:
                build_validation_report_freshness_matrix(
                    corpus.reports,
                    registry_path=corpus.registry,
                    as_of=AUDIT_AS_OF,
                    max_age_days=cast(int, value),
                )
            else:
                load_validation_report_lifecycle_registry(
                    corpus.registry,
                    reports_root=corpus.reports,
                    as_of=AUDIT_AS_OF,
                    max_age_days=cast(int, value),
                )
    assert corpus.snapshot() == before


@pytest.mark.parametrize("value", [0, 21, 10000])
def test_advisory_age_window_does_not_rewrite_registry_policy(tmp_path: Path, value: int) -> None:
    """Zero and broad caller windows retain the registry's original 21-day bytes."""
    corpus = copy_corpus(tmp_path / "corpus")
    before = corpus.snapshot()
    matrix = build_validation_report_freshness_matrix(
        corpus.reports,
        registry_path=corpus.registry,
        as_of=AUDIT_AS_OF,
        max_age_days=value,
    )
    assert matrix.max_age_days == value
    assert all(report.stale == (report.age_days > value) for report in matrix.reports)
    assert corpus.snapshot() == before


@pytest.mark.parametrize("timestamp", ["20260905T130001Z", "2026-09-05T15:00:01+02:00", " 2026-09-05T13:00:01 "])
def test_public_timestamp_parser_normalises_supported_spellings(timestamp: str) -> None:
    """Compact, offset and naive timestamps denote the same UTC instant."""
    assert parse_datetime(timestamp) == datetime(2026, 9, 5, 13, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize("timestamp", ["", " ", "private malformed timestamp", "2026-02-30T00:00:00Z"])
def test_cli_timestamp_refusal_keeps_existing_output(tmp_path: Path, timestamp: str) -> None:
    """Malformed timestamp text is hidden and refusal precedes output publication."""
    corpus = copy_corpus(tmp_path / "corpus")
    output = corpus.root / "inventory.json"
    output.write_bytes(b"existing output")
    result = run_cli(corpus, ["--as-of", timestamp, "--output-json", str(output)])
    assert result.returncode == 1
    assert result.stderr == "Validation report freshness inputs could not be inspected\n"
    assert output.read_bytes() == b"existing output"
