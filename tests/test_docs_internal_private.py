# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Permanent docs/internal privacy guard tests

"""Real-surface tests for the docs/internal forever-private policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_guard():
    path = REPO_ROOT / "tools" / "check_docs_internal_private.py"
    spec = importlib.util.spec_from_file_location("check_docs_internal_private", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_docs_internal_is_private() -> None:
    """Live repo must keep docs/internal gitignored, untracked, and off MkDocs."""
    guard = _load_guard()
    errors = guard.collect_errors(check_history=True)
    assert errors == [], "\n".join(errors)
    assert guard.main([]) == 0


def test_gitignore_rules_reject_missing_and_negation() -> None:
    """Missing ignore rules or re-include negations fail closed."""
    guard = _load_guard()
    assert guard.check_gitignore_rules("*.pyc\n") != []
    assert guard.check_gitignore_rules("docs/internal/\n") == []
    assert guard.check_gitignore_rules("docs/internal/**\n") == []
    neg = guard.check_gitignore_rules("docs/internal/\n!docs/internal/TODO_CONSOLIDATED.md\n")
    assert any("re-include" in err for err in neg)


def test_tracked_and_history_paths_fail_closed() -> None:
    """Any tracked or historical docs/internal path is a hard failure."""
    guard = _load_guard()
    tracked = guard.check_no_tracked_internal_paths(["README.md", "docs/internal/TODO_CONSOLIDATED.md", "docs/api.md"])
    assert any("TODO_CONSOLIDATED" in err for err in tracked)
    history = guard.check_no_history_internal_paths(["docs/changelog.md", "docs/internal/secret.md"])
    assert any("secret.md" in err for err in history)
    assert guard.check_no_tracked_internal_paths(["README.md", "docs/api.md"]) == []


def test_mkdocs_exclude_required() -> None:
    """MkDocs must exclude the private tree from the public site build."""
    guard = _load_guard()
    ok = "docs_dir: docs\nexclude_docs: |\n  internal/**\n"
    assert guard.check_mkdocs_excludes_internal(ok) == []
    assert guard.check_mkdocs_excludes_internal("docs_dir: docs\n") != []
    assert guard.check_mkdocs_excludes_internal("exclude_docs: |\n  other/**\n") != []
