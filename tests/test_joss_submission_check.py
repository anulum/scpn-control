# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JOSS submission guard tests.

"""Regression tests for the local JOSS submission guard."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools import check_joss_submission

ROOT = Path(__file__).resolve().parents[1]


def _write_valid_joss_tree(root: Path) -> None:
    """Create a minimal JOSS paper tree that satisfies the submission guard."""
    (root / "docs").mkdir()
    submission = root / "papers" / "submissions" / "001_neuro_symbolic_tokamak_control_software"
    submission.mkdir(parents=True)
    (submission / "manuscript.md").write_text(
        "\n".join(
            [
                "---",
                "title: 'SCPN Control: Test Paper'",
                "orcid: 0009-0009-3560-0851",
                "bibliography: references.bib",
                "---",
                "# Summary",
                "quantitative external-code claims remain blocked until real artefacts are admitted [@dimits2000].",
                "# Statement of Need",
                "# Implementation",
                "production runtime claims remain subject to runtime-admission evidence",
                "# Validation",
                "# Acknowledgements",
                "# References",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / "joss_paper.md").write_text(
        "\n".join(
            [
                "# JOSS Paper: SCPN Control",
                "SCPN Control: Test Paper",
                "The canonical manuscript package is",
                "papers/submissions/001_neuro_symbolic_tokamak_control_software/.",
                "It contains manuscript.md, references.bib, and manuscript.pdf.",
                "This review draft has not been submitted.",
            ]
        ),
        encoding="utf-8",
    )
    (submission / "references.bib").write_text(
        "@article{dimits2000,\n  title = {Dimits test},\n  year = {2000}\n}\n",
        encoding="utf-8",
    )


def _point_guard_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Redirect the guard module constants to a temporary repository tree."""
    submission = root / "papers" / "submissions" / "001_neuro_symbolic_tokamak_control_software"
    monkeypatch.setattr(check_joss_submission, "ROOT", root)
    monkeypatch.setattr(check_joss_submission, "SUBMISSION_PATH", submission)
    monkeypatch.setattr(check_joss_submission, "PAPER_PATH", submission / "manuscript.md")
    monkeypatch.setattr(check_joss_submission, "DOCS_PATH", root / "docs" / "joss_paper.md")
    monkeypatch.setattr(check_joss_submission, "BIB_PATH", submission / "references.bib")


def test_joss_submission_guard_accepts_current_repository() -> None:
    """Run the JOSS submission guard through its production CLI path."""
    assert check_joss_submission.main() == 0

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_joss_submission.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: canonical JOSS package and documentation pointer" in result.stdout


def test_joss_submission_guard_reports_missing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing JOSS paper files must fail closed with all file diagnostics."""
    _point_guard_at(monkeypatch, tmp_path)

    assert check_joss_submission.main() == 1
    output = capsys.readouterr().out
    assert "MISSING: papers/submissions/001_neuro_symbolic_tokamak_control_software/manuscript.md" in output
    assert "MISSING: docs/joss_paper.md" in output
    assert "MISSING: papers/submissions/001_neuro_symbolic_tokamak_control_software/references.bib" in output


def test_joss_submission_guard_uses_script_relative_inputs_from_unrelated_cwd(tmp_path: Path) -> None:
    """Use the script's real repository inputs from an unrelated caller directory."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_joss_submission.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "canonical JOSS package and documentation pointer" in result.stdout
    assert not result.stderr


def test_joss_submission_guard_reports_editorial_and_citation_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stale editorial markers and missing citation keys must be reported."""
    _write_valid_joss_tree(tmp_path)
    submission = tmp_path / "papers" / "submissions" / "001_neuro_symbolic_tokamak_control_software"
    (submission / "manuscript.md").write_text(
        "\n".join(
            [
                "---",
                "title: 'SCPN Control: Drifted Paper'",
                "bibliography: references.bib",
                "---",
                "# Summary",
                "A stale claim [@missingkey].",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "joss_paper.md").write_text(
        "# JOSS Paper: SCPN Control\nReference [@missingdoc].\n",
        encoding="utf-8",
    )
    (submission / "references.bib").write_text(
        "\n".join(
            [
                "@article{dimits2000,",
                "  title = {First}",
                "}",
                "@article{dimits2000,",
                "  title = {Duplicate}",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    _point_guard_at(monkeypatch, tmp_path)

    assert check_joss_submission.main() == 1
    output = capsys.readouterr().out
    assert "manuscript.md missing 'orcid: 0009-0009-3560-0851'" in output
    assert "docs/joss_paper.md missing 'canonical manuscript package" in output
    assert "docs/joss_paper.md missing paper title 'SCPN Control: Drifted Paper'" in output
    assert "references.bib duplicate bibliography key 'dimits2000'" in output
    assert "manuscript.md cites missing bibliography keys: missingkey" in output


def test_joss_submission_guard_reports_missing_yaml_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A paper without a YAML title cannot enter the JOSS review workflow."""
    _write_valid_joss_tree(tmp_path)
    paper = tmp_path / "papers" / "submissions" / "001_neuro_symbolic_tokamak_control_software" / "manuscript.md"
    paper.write_text(
        paper.read_text(encoding="utf-8").replace("title: 'SCPN Control: Test Paper'\n", ""), encoding="utf-8"
    )
    _point_guard_at(monkeypatch, tmp_path)

    errors = check_joss_submission.check_repository()
    assert (
        "MISMATCH: papers/submissions/001_neuro_symbolic_tokamak_control_software/"
        "manuscript.md missing YAML title" in errors
    )


def test_joss_submission_guard_skips_missing_markdown_during_citation_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Citation scanning should still run when one markdown mirror is absent."""
    _write_valid_joss_tree(tmp_path)
    (tmp_path / "docs" / "joss_paper.md").unlink()
    _point_guard_at(monkeypatch, tmp_path)

    errors = check_joss_submission.check_repository()

    assert "MISSING: docs/joss_paper.md" in errors
    assert not any("cites missing bibliography keys" in error for error in errors)


def test_joss_submission_guard_accepts_redirected_valid_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Temporary valid paper trees exercise the success path without repo files."""
    _write_valid_joss_tree(tmp_path)
    _point_guard_at(monkeypatch, tmp_path)

    assert check_joss_submission.check_repository() == []


@pytest.fixture
def joss_repository(tmp_path: Path) -> Path:
    """Create real required inputs beside the byte-identical production script.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory.

    Returns
    -------
    pathlib.Path
        Physical repository root discovered from the copied script location.
    """
    root = tmp_path / "joss-repository"
    (root / "tools").mkdir(parents=True)
    source = ROOT / "tools/check_joss_submission.py"
    target = root / "tools/check_joss_submission.py"
    shutil.copyfile(source, target)
    assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    _write_valid_joss_tree(root)
    return root


def _joss_public(root: Path) -> ModuleType:
    """Import the unchanged source from its actual fixture filesystem location.

    Parameters
    ----------
    root : pathlib.Path
        Physical repository root containing the copied script.

    Returns
    -------
    types.ModuleType
        Module whose public no-argument API uses its own real input paths.
        Production globals and providers are not replaced.
    """
    spec = importlib.util.spec_from_file_location("physical_joss_guard", root / "tools/check_joss_submission.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _joss_cli(root: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the physical production script from an unrelated caller directory.

    Parameters
    ----------
    root : pathlib.Path
        Real fixture root, independent of the child working directory.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual status and UTF-8 output. Qualification may measure the entire
        byte-identical child script through an explicit coverage configuration.
    """
    command = [sys.executable]
    config = os.environ.get("SCPN_JOSS_COVERAGE_RC")
    if config:
        command += ["-m", "coverage", "run", "--rcfile=" + config, "--parallel-mode"]
    command += [str(root / "tools/check_joss_submission.py")]
    return subprocess.run(command, cwd=root.parent, capture_output=True, text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize("name", ["manuscript.md", "references.bib", "joss_paper.md", "all"])
@pytest.mark.parametrize("text", ["", " \t\n\n"])
def test_real_joss_blank_required_inputs(joss_repository: Path, name: str, text: str) -> None:
    """Refuse empty and whitespace-only inputs through the real API and CLI.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    name : str
        One required filename or all three inputs.
    text : str
        Empty or whitespace-only content.
    """
    inputs = [joss_repository / "docs/joss_paper.md", *sorted((joss_repository / "papers").rglob("*.*"))]
    selected = [path for path in inputs if name == "all" or path.name == name]
    for path in selected:
        path.write_text(text, encoding="utf-8")
    errors = _joss_public(joss_repository).check_repository()
    assert errors == [
        "EMPTY: " + path.relative_to(joss_repository).as_posix()
        for path in sorted(selected, key=lambda p: ["manuscript.md", "joss_paper.md", "references.bib"].index(p.name))
    ]
    result = _joss_cli(joss_repository)
    assert result.returncode == 1 and "EMPTY:" in result.stdout
    assert "OK:" not in result.stdout and not result.stderr


@pytest.mark.parametrize("name", ["manuscript.md", "references.bib", "joss_paper.md", "all"])
def test_real_joss_missing_required_inputs(joss_repository: Path, name: str) -> None:
    """Report missing real files without bypassing remaining input checks.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    name : str
        Required filename to remove, or all three inputs.
    """
    paths = [joss_repository / "docs/joss_paper.md", *sorted((joss_repository / "papers").rglob("*.*"))]
    removed = [path for path in paths if name == "all" or path.name == name]
    for path in removed:
        path.unlink()
    errors = _joss_public(joss_repository).check_repository()
    assert sum(error.startswith("MISSING:") for error in errors) == len(removed)
    assert _joss_cli(joss_repository).returncode == 1


@pytest.mark.parametrize("drift", ["body_title", "no_opener", "no_closer", "no_title"])
def test_real_joss_front_matter_title(joss_repository: Path, drift: str) -> None:
    """Require title placement in the real initial delimited metadata block.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    drift : str
        Title-placement or delimiter fault.
    """
    paper = next((joss_repository / "papers").rglob("manuscript.md"))
    text = paper.read_text(encoding="utf-8")
    title = "title: 'SCPN Control: Test Paper'"
    if drift == "body_title":
        text = text.replace(title + "\n", "", 1) + "\n" + title + "\n"
    elif drift == "no_opener":
        text = text.removeprefix("---\n")
    elif drift == "no_closer":
        text = text.replace("\n---\n", "\n", 1)
    else:
        text = text.replace(title + "\n", "", 1)
    paper.write_text(text, encoding="utf-8")
    errors = _joss_public(joss_repository).check_repository()
    assert any(error.endswith("missing YAML title") for error in errors)
    assert _joss_cli(joss_repository).returncode == 1


def test_real_joss_editorial_title_and_citation_drift(joss_repository: Path) -> None:
    """Aggregate independent marker, title, duplicate-key, and citation findings.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    """
    paper = next((joss_repository / "papers").rglob("manuscript.md"))
    paper.write_text(paper.read_text().replace("# Validation", "# Other").replace("@dimits2000", "@zeta; @Alpha"))
    docs = joss_repository / "docs/joss_paper.md"
    docs.write_text(
        docs.read_text().replace("SCPN Control: Test Paper", "Another paper").replace("manuscript.pdf", "draft.pdf")
    )
    bib = next((joss_repository / "papers").rglob("references.bib"))
    bib.write_text(bib.read_text() + "\n@book{dimits2000, title = {Duplicate}}\n")
    errors = _joss_public(joss_repository).check_repository()
    assert len(errors) == 5
    assert any("missing '# Validation'" in error for error in errors)
    assert any("missing 'manuscript.pdf'" in error for error in errors)
    assert any("missing paper title" in error for error in errors)
    assert any("duplicate bibliography key 'dimits2000'" in error for error in errors)
    assert any("cites missing bibliography keys: Alpha, zeta" in error for error in errors)
    assert _joss_cli(joss_repository).returncode == 1


def test_real_joss_keyless_bibliography(joss_repository: Path) -> None:
    """Refuse a nonblank keyless bibliography even with no bracketed citations.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    """
    paper = next((joss_repository / "papers").rglob("manuscript.md"))
    paper.write_text(paper.read_text().replace("[@dimits2000]", "reference omitted"))
    bib = next((joss_repository / "papers").rglob("references.bib"))
    bib.write_text("Prose without any bibliography entry.\n")
    errors = _joss_public(joss_repository).check_repository()
    assert len(errors) == 1 and errors[0].endswith("contains no bibliography entry keys")
    assert _joss_cli(joss_repository).returncode == 1


def test_real_joss_lexical_scope_and_fresh_results(joss_repository: Path) -> None:
    """Exercise whitespace, narrative/doc citation limits, and fresh result ownership.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    """
    paper = next((joss_repository / "papers").rglob("manuscript.md"))
    paper.write_text(paper.read_text().replace("# Statement of Need", "# Statement\n of Need") + "\n@not_bracketed\n")
    docs = joss_repository / "docs/joss_paper.md"
    docs.write_text(docs.read_text() + "\nDocumentation-only citation [@not_in_manuscript].\n")
    guard = _joss_public(joss_repository)
    result = guard.check_repository()
    assert result == []
    result.append("caller mutation")
    assert guard.check_repository() == [] and guard.main() == 0
    child = _joss_cli(joss_repository)
    assert child.returncode == 0 and "local editorial and citation checks" in child.stdout
    assert "submission-review aligned" not in child.stdout and not child.stderr
    with pytest.raises(TypeError):
        guard.check_repository(unexpected_argument=True)
    with pytest.raises(TypeError):
        guard.main(unexpected_argument=True)


@pytest.mark.parametrize("name", ["manuscript.md", "references.bib", "joss_paper.md"])
@pytest.mark.parametrize("fault", ["directory", "invalid_utf8"])
def test_real_joss_native_read_failures(joss_repository: Path, name: str, fault: str) -> None:
    """Propagate genuine filesystem and decode failures without printing success.

    Parameters
    ----------
    joss_repository : pathlib.Path
        Physical valid fixture root.
    name : str
        Required input receiving the fault.
    fault : str
        Real directory or invalid UTF-8 bytes.
    """
    target = next(path for path in joss_repository.rglob(name) if path.is_file())
    exception: type[Exception]
    if fault == "directory":
        target.unlink()
        target.mkdir()
        exception = OSError
    else:
        target.write_bytes(b"\xff\xfeinvalid")
        exception = UnicodeError
    with pytest.raises(exception):
        _joss_public(joss_repository).check_repository()
    child = _joss_cli(joss_repository)
    assert child.returncode != 0 and "Traceback" in child.stderr
    assert "OK:" not in child.stdout
