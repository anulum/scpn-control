# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Inventory representation and public stdout parity.
"""Compare public JSON/Markdown output with actual validated matrix renderers."""

from __future__ import annotations

import json
from pathlib import Path

from report_lifecycle_fixtures import AUDIT_AS_OF, copy_corpus, run_cli

from tools.validation_report_freshness import build_validation_report_freshness_matrix


def test_external_corpus_stdout_and_file_formats_match_public_matrix(tmp_path: Path) -> None:
    """Both supported formats preserve external paths, declarations and deterministic bytes."""
    corpus = copy_corpus(tmp_path / "corpus")
    matrix = build_validation_report_freshness_matrix(
        corpus.reports,
        registry_path=corpus.registry,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    result = run_cli(corpus, ["--json-out", "--markdown-out"])
    assert result.returncode == 0
    assert json.loads(result.stdout) == matrix.to_dict()
    assert corpus.reports.as_posix() in result.stdout
    result = run_cli(corpus, ["--markdown-out", "--output-md", "relative-output/inventory.md"])
    assert result.returncode == 0
    assert result.stdout == matrix.to_markdown()
    assert (corpus.root / "relative-output/inventory.md").read_text(encoding="utf-8") == result.stdout
    assert "Current publicly admitted reports: `0`" in result.stdout
