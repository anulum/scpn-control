# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual project metadata version mutation contracts.
"""Exercise complete TOML parsing, semantic field ownership and guarded writes."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools.publish_metadata import PublishMetadataError, bump_project_version, read_project_version


@pytest.mark.parametrize(
    "text",
    [
        '[project]\nversion="1.2.3"\n',
        '[project]\nversion="1.2.3"',
        '[project]\nversion\t=\t"1.2.3"\n',
        "[project]\nversion = '1.2.3' # retained comment\n",
        '["project"]\n"version"="1.2.3"\n',
        'project.version="1.2.3"\n',
        'project={version="1.2.3",name="scpn-control"}\n',
        '[project]\nversion="""\n1.2.3"""\n',
        "[project]\nversion='''1.2.3'''\n",
        '[project]\r\nversion="1.2.3"\r\n',
        '[project]\nversion="""1.2.\\\n  3"""\n',
        '[tool.foreign]\nversion="1.2.3"\n[project]\nversion="1.2.3"\n',
        '# version="1.2.3"\n[project]\nversion="1.2.3"\n',
        '[project]\nversion="1.2.3"\n[tool.foreign]\n"1.2.3"="1.2.3"\n',
        '[tool.foreign]\n"1.2.3"="unrelated"\n"1.2.4"="retained"\n[project]\nversion="1.2.3"\n',
        '[project]\nversion="1.2.3"\n[tool.foreign]\nvalues=[nan,inf,-inf,{"nested"=[1,true]}]\n',
        '[project]\nversion="1.2.3"\n[tool.foreign]\nvalue="a\\"#b"\n',
        "[project]\nversion='1.2.3'\n[tool.foreign]\nvalue='literal # value'\n",
        '[project]\nversion="1.2.3"\n[tool.foreign]\nvalue="""two"quote""chars"""\n',
        "[project]\nversion='1.2.3'\n[tool.foreign]\nvalue='''two'quote''chars'''\n",
        '[project]\nversion="1.2.3"\n[tool.foreign]\nvalue="""ends with quote""""\n',
    ],
)
def test_bump_changes_only_the_actual_project_version_token(tmp_path: Path, text: str) -> None:
    """Valid literal/dotted/inline/escaped layouts retain unrelated byte content."""
    path = tmp_path / "pyproject.toml"
    original = text.encode("utf-8")
    path.write_bytes(original)
    assert read_project_version(path) == "1.2.3"
    assert bump_project_version(path, "patch") == ("1.2.3", "1.2.4")
    updated = path.read_bytes().decode("utf-8")
    assert tomllib.loads(updated)["project"]["version"] == "1.2.4"
    assert read_project_version(path) == "1.2.4"
    assert len(list(tmp_path.iterdir())) == 1
    if text.startswith("#"):
        assert updated.startswith('# version="1.2.3"\n')
    if "[tool.foreign]" in text:
        assert (
            updated.split("[tool.foreign]", 1)[1].split("[project]", 1)[0]
            == text.split("[tool.foreign]", 1)[1].split("[project]", 1)[0]
        )
    if "\r\n" in text:
        assert path.read_bytes().count(b"\r\n") == original.count(b"\r\n")


@pytest.mark.parametrize(
    "payload",
    [
        b"[project]\n",
        b"[project]\nversion=42\n",
        b"[project]\nversion=[]\n",
        b'[project]\nversion=""\n',
        b'[project]\nversion=" "\n',
        b'version="1.2.3"\n',
        b'[tool.foreign]\nversion="1.2.3"\n',
        b'[project]\nversion="1.2.3"\nversion="9.8.7"\n',
        b'project="1.2.3"\n',
        b"\xff",
        b"[invalid",
        b"",
    ],
)
def test_invalid_metadata_refuses_without_writing(tmp_path: Path, payload: bytes) -> None:
    """Malformed/duplicate/foreign declarations cannot select a project release field."""
    path = tmp_path / "pyproject.toml"
    path.write_bytes(payload)
    with pytest.raises(PublishMetadataError, match="Cannot parse version"):
        read_project_version(path)
    with pytest.raises(PublishMetadataError, match="Cannot parse version"):
        bump_project_version(path, "patch")
    assert path.read_bytes() == payload


@pytest.mark.parametrize("part,expected", [("major", "2.0.0"), ("minor", "1.3.0"), ("patch", "1.2.4")])
def test_exact_semantic_version_component_updates(tmp_path: Path, part: str, expected: str) -> None:
    """The selected component increments and lower components reset as advertised."""
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nversion="1.2.3"\n', encoding="utf-8")
    assert bump_project_version(path, part) == ("1.2.3", expected)
    assert read_project_version(path) == expected


@pytest.mark.parametrize("version", ["x.2.3", "1.2", "-1.2.3", "01.2.3", "1.2.3rc1", "1.2.3.4"])
def test_bump_refuses_non_semantic_version_but_read_preserves_the_string(tmp_path: Path, version: str) -> None:
    """Reading and the explicitly narrower bump grammar have different contracts."""
    path = tmp_path / "pyproject.toml"
    original = f'[project]\nversion="{version}"\n'.encode()
    path.write_bytes(original)
    assert read_project_version(path) == version
    with pytest.raises(PublishMetadataError, match="not semver"):
        bump_project_version(path, "patch")
    assert path.read_bytes() == original


def test_invalid_part_refuses_before_opening_a_missing_file(tmp_path: Path) -> None:
    """An invalid operation cannot read or create a metadata carrier."""
    path = tmp_path / "missing.toml"
    with pytest.raises(PublishMetadataError, match="Invalid bump part"):
        bump_project_version(path, "invalid")
    assert not path.exists()
    with pytest.raises(PublishMetadataError, match="Cannot parse"):
        read_project_version(path)


def test_integer_conversion_limits_are_authored_refusals(tmp_path: Path) -> None:
    """Native large-integer parsing and increment formatting cannot leak ValueError."""
    path = tmp_path / "pyproject.toml"
    for digits, part in [(4301, "patch"), (4300, "major")]:
        original = ('[project]\nversion="' + "9" * digits + '.2.3"\n').encode()
        path.write_bytes(original)
        with pytest.raises(PublishMetadataError, match="integer conversion limit"):
            bump_project_version(path, part)
        assert path.read_bytes() == original


def test_changed_input_refuses_without_overwriting_the_other_writer(tmp_path: Path) -> None:
    """An actual runtime observer replaces metadata between parsing and publication."""
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b'[project]\nversion="1.2.3"\n')
    script = """
import sys
from pathlib import Path
from tools.publish_metadata import PublishMetadataError, bump_project_version
path = Path(sys.argv[1])
reads = []
def observe(event, args):
    if event == "open" and Path(args[0]) == path and args[1] == "r":
        reads.append(1)
        if len(reads) == 2:
            path.write_bytes(b'[project]\\nversion="1.2.9"\\n')
sys.addaudithook(observe)
try:
    bump_project_version(path, "patch")
except PublishMetadataError as error:
    assert str(error) == "Project metadata changed before version publication"
else:
    raise AssertionError("other-writer metadata was overwritten")
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == b'[project]\nversion="1.2.9"\n'
    assert list(tmp_path.iterdir()) == [path]


def test_bump_preserves_symlink_target_and_protected_inode_alias(tmp_path: Path) -> None:
    """Actual linked output spellings cannot replace protected or foreign files."""
    target = tmp_path / "target.toml"
    original = b'[project]\nversion="1.2.3"\n'
    target.write_bytes(original)
    link = tmp_path / "pyproject.toml"
    link.symlink_to(target)
    with pytest.raises(PublishMetadataError, match="Cannot publish"):
        bump_project_version(link, "patch")
    assert target.read_bytes() == original and link.is_symlink()
    link.unlink()
    link.hardlink_to(target)
    with pytest.raises(PublishMetadataError, match="Cannot publish"):
        bump_project_version(link, "patch", protected_files=(target,))
    assert target.read_bytes() == link.read_bytes() == original


def test_actual_repository_metadata_bumps_without_touching_any_other_field(tmp_path: Path) -> None:
    """The maintained full source metadata is parsed and rewritten in owned custody."""
    repository = Path(__file__).resolve().parents[2]
    source = repository / "pyproject.toml"
    original = source.read_bytes()
    before = tomllib.loads(original.decode("utf-8"))
    path = tmp_path / "pyproject.toml"
    path.write_bytes(original)
    old, new = bump_project_version(path, "patch")
    assert old == before["project"]["version"]
    before["project"]["version"] = new
    assert tomllib.loads(path.read_bytes().decode("utf-8")) == before
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original).digest()


def test_actual_runtime_write_denial_preserves_metadata_and_removes_staging(tmp_path: Path) -> None:
    """A native audit refusal at final replacement leaves the predecessor intact."""
    path = tmp_path / "pyproject.toml"
    original = b'[project]\nversion="1.2.3"\n'
    path.write_bytes(original)
    script = """
import sys
from pathlib import Path
from tools.publish_metadata import PublishMetadataError, bump_project_version
path = Path(sys.argv[1])
def deny(event, args):
    if event == "os.rename" and Path(args[1]) == path:
        raise PermissionError("selected metadata replacement denied")
sys.addaudithook(deny)
try:
    bump_project_version(path, "patch")
except PublishMetadataError as error:
    assert str(error) == "Cannot publish the updated project version"
else:
    raise AssertionError("actual replacement refusal was not enforced")
"""
    process = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_unknown_upload_target_refuses_before_any_process_launch(tmp_path: Path) -> None:
    """An invalid direct API target cannot fall through to production PyPI."""
    script = """
import sys
from pathlib import Path
from tools import publish
publish.DIST = Path(sys.argv[1]) / "missing"
def deny(event, args):
    if event == "subprocess.Popen":
        raise AssertionError("unknown target reached a process launch")
sys.addaudithook(deny)
try:
    publish.upload("unrecognised")
except SystemExit as error:
    assert str(error) == "Invalid upload target (use pypi/testpypi)"
else:
    raise AssertionError("unknown upload target was accepted")
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []
