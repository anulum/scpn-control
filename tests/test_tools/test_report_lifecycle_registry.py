# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Complete maintained report corpus bindings.
"""Validate complete maintained registry membership through the public loader."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from report_lifecycle_fixtures import AUDIT_AS_OF, copy_corpus, run_cli

from tools.validation_report_freshness import ROOT, LifecycleRegistryError, load_validation_report_lifecycle_registry


def test_public_registry_loader_binds_every_available_report(tmp_path: Path) -> None:
    """Every present report matches its frozen digest while owner-local absences remain absent."""
    corpus = copy_corpus(tmp_path / "corpus")
    records = load_validation_report_lifecycle_registry(
        corpus.registry,
        reports_root=corpus.reports,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    assert len(records) == 128
    for name, record in records.items():
        path = corpus.root / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record.report_sha256
        else:
            assert record.storage_class == "owner_local_untracked"
    report = corpus.reports / "gk_interface_artifacts.json"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(LifecycleRegistryError, match="report digest drift"):
        load_validation_report_lifecycle_registry(
            corpus.registry,
            reports_root=corpus.reports,
            as_of=AUDIT_AS_OF,
            max_age_days=21,
        )


def test_cli_refuses_resolved_corpus_escape_before_writing(tmp_path: Path) -> None:
    """An escaping corpus symlink is refused with fixed text and preserves output."""
    corpus = copy_corpus(tmp_path / "corpus")
    outside = corpus.root / "outside.json"
    outside.write_bytes(b"{}")
    (corpus.reports / "escape.json").symlink_to(outside)
    output = corpus.root / "inventory.json"
    output.write_bytes(b"old")
    result = run_cli(corpus, ["--output-json", str(output)])
    assert result.returncode == 1
    assert result.stderr == "Validation report freshness inputs could not be inspected\n"
    assert output.read_bytes() == b"old"
    assert outside.read_bytes() == b"{}"


def test_relative_single_component_corpus_uses_cwd_before_parent_selection(tmp_path: Path) -> None:
    """The actual CLI accepts reports from inside validation without a parent-index traceback."""
    corpus = copy_corpus(tmp_path / "corpus")
    before = corpus.snapshot()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/validation_report_freshness.py"),
            *corpus.arguments(),
            "--reports-root",
            "reports",
            "--json-out",
        ],
        cwd=corpus.root / "validation",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"report_count": 128' in result.stdout
    assert corpus.snapshot() == before
