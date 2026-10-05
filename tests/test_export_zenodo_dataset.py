# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual checkout archive and Git file-selection contracts.

"""Exercise real checkout archives and filesystem policy with the actual Git engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools.export_zenodo_dataset import ROOT, collect_files, create_archive, main

SCRIPT = ROOT / "tools/export_zenodo_dataset.py"


def _cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real checkout CLI outside its root without inherited tracer/import flags."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", "COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def actual_archive(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    """Export the actual canonical checkout through the public API, including its uncommitted modules."""
    directory = tmp_path_factory.mktemp("canonical-source-export")
    return create_archive(directory / "candidate.zip")


def test_actual_checkout_zip_has_exact_public_members(actual_archive: tuple[Path, int]) -> None:
    """Compare every real ZIP member against its current checkout input and reject private paths."""
    destination, count = actual_archive
    files = collect_files()
    version = json.loads((ROOT / ".zenodo.json").read_text())["version"]
    prefix = f"scpn-control-{version}/"
    with zipfile.ZipFile(destination) as archive:
        assert count == len(files) == len(archive.namelist())
        assert archive.namelist() == [prefix + path.relative_to(ROOT).as_posix() for path in files]
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            assert not relative.startswith("docs/internal/")
            assert archive.read(prefix + relative) == path.read_bytes()
        assert prefix + "validation/free_boundary_acceptance_campaign.py" in archive.namelist()
        assert prefix + "src/scpn_control/__init__.py" in archive.namelist()
        assert prefix + "tools/export_zenodo_dataset.py" in archive.namelist()
        assert archive.testzip() is None


def test_real_cli_default_uses_caller_directory(tmp_path: Path) -> None:
    """Run the actual default command and inspect its completed source archive outside the checkout."""
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    version = json.loads((ROOT / ".zenodo.json").read_text())["version"]
    output = tmp_path / f"scpn-control-{version}.zip"
    assert str(output) in result.stdout and "files)" in result.stdout
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any("/docs/internal/" in name for name in archive.namelist())


@pytest.mark.parametrize("arguments,expected", [(("--help",), 0), (("--invalid-option",), 2), (("--output",), 2)])
def test_real_parser_contract(tmp_path: Path, arguments: tuple[str, ...], expected: int) -> None:
    """Exercise actual help and usage errors before archive creation."""
    result = _cli(tmp_path, *arguments)
    assert result.returncode == expected and "usage:" in result.stdout + result.stderr
    assert not list(tmp_path.glob("*.zip"))


@pytest.mark.parametrize(
    "alias", ["existing", "source", "metadata", "ignore", "symlink", "hardlink", "git", "git_alias"]
)
def test_real_cli_preserves_existing_inputs_and_git_metadata(tmp_path: Path, alias: str) -> None:
    """Refuse actual output aliases and new Git-metadata paths without changing original bytes."""
    output = tmp_path / "candidate.zip"
    protected = SCRIPT
    if alias == "existing":
        output.write_bytes(b"preserve earlier archive candidate")
        protected = output
    elif alias == "metadata":
        output = protected = ROOT / ".zenodo.json"
    elif alias == "ignore":
        output = protected = ROOT / ".gitignore"
    elif alias == "source":
        output = protected = ROOT / "src/scpn_control/__init__.py"
    elif alias == "symlink":
        output.symlink_to(protected)
    elif alias == "hardlink":
        os.link(protected, output)
    elif alias == "git":
        output = ROOT / ".git/a13-uncreated-archive.zip"
    else:
        parent = tmp_path / "git-alias"
        parent.symlink_to(ROOT / ".git", target_is_directory=True)
        output = parent / "a13-uncreated-archive.zip"
    before = protected.read_bytes()
    result = _cli(tmp_path, "--output", str(output))
    assert result.returncode == 2
    assert result.stderr == "Dataset export refused: metadata, source selection or archive output is invalid.\n"
    assert protected.read_bytes() == before
    if alias in ("git", "git_alias"):
        assert not output.exists()


def test_public_main_reports_success_then_preserves_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Use the genuine public main for creation and repeat-output refusal with an unchanged ZIP."""
    output = tmp_path / "api-main.zip"
    assert main(["--output", str(output)]) == 0
    before = output.read_bytes()
    assert main(["--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "Created" in captured.out and "refused" in captured.err
    assert output.read_bytes() == before


def test_real_cli_missing_parent_is_safe_refusal(tmp_path: Path) -> None:
    """An actual filesystem failure produces the fixed refusal and creates no output directory."""
    output = tmp_path / "absent-parent/candidate.zip"
    result = _cli(tmp_path, "--output", str(output))
    assert result.returncode == 2 and "refused" in result.stderr and "Traceback" not in result.stderr
    assert not output.parent.exists()


@pytest.fixture
def selection_corpus(tmp_path: Path) -> Path:
    """Build a disposable policy corpus for real Git ignore rules, separate from canonical export proof."""
    result = subprocess.run(["git", "init", "--quiet", str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    (tmp_path / ".zenodo.json").write_text('{"version":"1.2.3+local"}')
    for relative in [
        "README.md",
        "src/public.py",
        "src/tracked-private.py",
        "docs/guide.md",
        "docs/ignored.md",
        "docs/internal/plan.md",
        "docs/target/build.md",
        "docs/directory.md/child.md",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    (tmp_path / ".gitignore").write_text("docs/ignored.md\nsrc/tracked-private.py\n")
    added = subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--force", "src/tracked-private.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert added.returncode == 0, added.stderr
    return tmp_path


def test_real_git_selection_excludes_private_tracked_and_symlink_inputs(selection_corpus: Path) -> None:
    """Exercise current ignore rules despite index tracking and reject file/directory symlinks."""
    root = selection_corpus
    (root / "docs/link.md").symlink_to(root / "README.md")
    (root / "examples").symlink_to(root / "src", target_is_directory=True)
    outside = root.parent / (root.name + "-outside.md")
    outside.write_text("outside source bytes")
    (root / "docs/outside.md").symlink_to(outside)
    selected = [path.relative_to(root).as_posix() for path in collect_files(root)]
    assert selected == [".zenodo.json", "README.md", "docs/directory.md/child.md", "docs/guide.md", "src/public.py"]
    destination, count = create_archive(root / "candidate.zip", root=root)
    with zipfile.ZipFile(destination) as archive:
        assert count == len(selected)
        assert archive.namelist() == [f"scpn-control-1.2.3+local/{path}" for path in selected]
    (root / ".gitignore").write_text("")
    refreshed = [path.relative_to(root).as_posix() for path in collect_files(root)]
    assert "docs/ignored.md" in refreshed and "src/tracked-private.py" in refreshed


def test_selection_uses_supplied_checkout_with_inherited_git_root(
    selection_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass genuine inherited Git-root environment inputs without redirecting the real ignore engine."""
    monkeypatch.setenv("GIT_DIR", str(selection_corpus / "absent-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(ROOT))
    monkeypatch.setenv("GIT_INDEX_FILE", str(selection_corpus / "absent-index"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(selection_corpus / "absent-common-dir"))
    assert selection_corpus / "README.md" in collect_files(selection_corpus)
    assert selection_corpus / "src/tracked-private.py" not in collect_files(selection_corpus)


def test_empty_selection_and_nonrepository_refusals(tmp_path: Path) -> None:
    """Distinguish an empty public selection from actual Git failure on eligible nonrepository inputs."""
    assert collect_files(tmp_path) == []
    (tmp_path / ".zenodo.json").write_text('{"version":"1"}')
    with pytest.raises(RuntimeError, match="Git ignore selection"):
        create_archive(tmp_path / "refused.zip", root=tmp_path)
    assert not (tmp_path / "refused.zip").exists()


def test_all_ignored_selection_refuses_archive(selection_corpus: Path) -> None:
    """Use real Git to suppress every eligible file and refuse an empty archive."""
    (selection_corpus / ".gitignore").write_text("*\n")
    assert collect_files(selection_corpus) == []
    with pytest.raises(ValueError, match="selection is empty"):
        create_archive(selection_corpus / "empty.zip", root=selection_corpus)


@pytest.mark.parametrize(
    "metadata",
    [
        "null",
        "[]",
        "{}",
        '{"version":2}',
        '{"version":""}',
        '{"version":"../private"}',
        '{"version":"a/b"}',
        '{"version":"v☃"}',
        "{invalid-json",
    ],
)
def test_public_api_rejects_invalid_metadata(tmp_path: Path, metadata: str) -> None:
    """Refuse real malformed/nonmapping metadata and unsafe archive-prefix versions before any output."""
    (tmp_path / ".zenodo.json").write_text(metadata)
    with pytest.raises(ValueError):
        create_archive(tmp_path / "invalid.zip", root=tmp_path)
    assert not (tmp_path / "invalid.zip").exists()


def test_public_api_rejects_metadata_symlink(tmp_path: Path) -> None:
    """A real metadata symlink cannot admit a version from outside the selected checkout."""
    target = tmp_path / "outside.json"
    target.write_text('{"version":"1"}')
    (tmp_path / ".zenodo.json").symlink_to(target)
    with pytest.raises(ValueError, match="regular checkout file"):
        create_archive(tmp_path / "refused.zip", root=tmp_path)
