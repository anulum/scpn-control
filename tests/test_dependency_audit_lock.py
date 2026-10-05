# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Universal dependency advisory export contracts.

"""Exercise the real advisory exporter without resolving or installing packages."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools.export_dependency_audit_lock import locked_packages, main, write_audit_lock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/export_dependency_audit_lock.py"


def _repository(path: Path) -> Path:
    """Create actual universal and requirements locks with distinct marker versions."""
    path.mkdir()
    (path / "requirements").mkdir()
    (path / "tools").mkdir()
    (path / "tools/dependency_advisory_sources.json").write_text("{}\n", encoding="utf-8")
    (path / "pyproject.toml").write_text('[project]\nname = "control-fixture"\n', encoding="utf-8")
    (path / "uv.lock").write_text(
        '[[package]]\nname = "control-fixture"\nversion = "1.0"\nsource = {editable = "."}\n'
        '[[package]]\nname = "Example_Pkg"\nversion = "1.0"\nsource = {registry = "https://pypi.org/simple"}\n',
        encoding="utf-8",
    )
    (path / "requirements/ci-test.txt").write_text(
        "# Locked test dependencies\n\nExample_Pkg==1.0 \\\n    --hash=sha256:123\n"
        'Example_Pkg==2.0; sys_platform == "win32" \\\n    --hash=sha256:456\n'
        'other-pkg==3.0; python_version < "3.12"\n',
        encoding="utf-8",
    )
    return path


def _command(*arguments: str) -> list[str]:
    """Invoke the physical CLI, optionally recording its actual child coverage."""
    config = os.environ.get("SCPN_DEPENDENCY_EXPORT_COVERAGE_RC")
    prefix = [sys.executable]
    if config:
        prefix.extend(["-m", "coverage", "run", "--parallel-mode", f"--rcfile={config}"])
    return [*prefix, str(TOOL), *arguments]


def test_public_inventory_retains_all_versions_and_ignores_environment_markers(tmp_path: Path) -> None:
    """Keep optional and foreign-platform targets while deduplicating identical pairs."""
    root = _repository(tmp_path / "repo")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert locked_packages(root) == [("example-pkg", "1.0"), ("example-pkg", "2.0"), ("other-pkg", "3.0")]
    output = tmp_path / "pylock.toml"
    assert write_audit_lock(root, output) == 3
    data = tomllib.loads(output.read_text(encoding="utf-8"))
    assert data["lock-version"] == "1.0"
    assert [(p["name"], p["version"]) for p in data["packages"]] == locked_packages(root)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_local_build_versions_retain_custody_and_use_upstream_advisory_identity(tmp_path: Path) -> None:
    """Keep complete CPU/vendor pins while querying every public upstream version."""
    root = _repository(tmp_path / "repo")
    (root / "requirements/ci-test.txt").write_text("torch==2.14.0+cpu\ntorch==2.13.0+cpu\n", encoding="utf-8")
    output = tmp_path / "pylock.toml"
    assert write_audit_lock(root, output) == 3
    rows = tomllib.loads(output.read_text(encoding="utf-8"))["packages"]
    assert {(p["name"], p["version"], p["tool"]["scpn-control"]["locked-version"]) for p in rows} == {
        ("example-pkg", "1.0", "1.0"),
        ("torch", "2.13.0", "2.13.0+cpu"),
        ("torch", "2.14.0", "2.14.0+cpu"),
    }


def test_real_cli_uses_default_repository_and_preserves_its_locks(tmp_path: Path) -> None:
    """Export every actual upstream lock identity through the default source-tree entry."""
    paths = [ROOT / "uv.lock", *sorted((ROOT / "requirements").glob("ci-*.txt"))]
    before = {p: p.read_bytes() for p in paths}
    output = tmp_path / "pylock.toml"

    result = subprocess.run(_command("--output", str(output)), capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    data = tomllib.loads(output.read_text(encoding="utf-8"))
    assert {(p["name"], p["tool"]["scpn-control"]["locked-version"]) for p in data["packages"]} == set(
        locked_packages(ROOT)
    )
    assert "Exported" in result.stdout
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("foreign-editable", "not an upstream registry dependency"),
        ("empty-registry", "not an upstream registry dependency"),
        ("range", "must have one exact version"),
        ("two-specifiers", "must have one exact version"),
        ("direct-url", "no verified advisory metadata"),
        ("wildcard", "Invalid version"),
        ("missing-version", "version"),
        ("missing-lock", "uv.lock"),
        ("no-requirements", "no requirements/ci-*.txt"),
        ("empty-inventory", "inventory is empty"),
    ],
)
def test_real_cli_refuses_incomplete_or_unsupported_locks(tmp_path: Path, change: str, diagnostic: str) -> None:
    """Fail closed on actual source files before creating an advisory output."""
    root = _repository(tmp_path / "repo")
    lock = root / "uv.lock"
    requirements = root / "requirements/ci-test.txt"
    if change == "foreign-editable":
        lock.write_text(
            lock.read_text().replace(
                'source = {registry = "https://pypi.org/simple"}', 'source = {editable = "../other"}'
            )
        )
    elif change == "empty-registry":
        lock.write_text(lock.read_text().replace("https://pypi.org/simple", ""))
    elif change == "missing-version":
        lock.write_text(lock.read_text().replace('version = "1.0"\nsource = {registry', "source = {registry"))
    elif change == "missing-lock":
        lock.unlink()
    elif change == "no-requirements":
        requirements.unlink()
    elif change == "empty-inventory":
        lock.write_text("package = []\n")
        requirements.write_text("# No upstream dependencies\n")
    else:
        declarations = {
            "range": "example-pkg>=1.0",
            "two-specifiers": "example-pkg==1.0,!=2.0",
            "direct-url": "example-pkg @ https://example.invalid/package.whl",
            "wildcard": "example-pkg==1.*",
        }
        requirements.write_text(declarations[change] + "\n")
    output = tmp_path / "pylock.toml"

    result = subprocess.run(
        _command("--repo-root", str(root), "--output", str(output)), capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert not output.exists()


def test_public_main_and_cli_refuse_existing_output_and_missing_parent(tmp_path: Path) -> None:
    """Retain caller files and report real filesystem failures as CLI errors."""
    root = _repository(tmp_path / "repo")
    output = tmp_path / "pylock.toml"
    output.write_bytes(b"original caller bytes\n")
    with pytest.raises(SystemExit) as error:
        main(["--repo-root", str(root), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"original caller bytes\n"
    result = subprocess.run(
        _command("--repo-root", str(root), "--output", str(tmp_path / "absent/pylock.toml")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("binding", ["valid", "wrong-name", "wrong-hash"])
def test_source_archives_require_exact_package_and_checksum_bindings(tmp_path: Path, binding: str) -> None:
    """Audit source metadata only for the exact hash-pinned upstream declaration."""
    import json

    root = _repository(tmp_path / "repo")
    url = "https://example.invalid/source.tar.gz"
    digest = "a" * 64
    (root / "requirements/ci-test.txt").write_text(
        f"--index-url https://pypi.org/simple\n--extra-index-url https://example.invalid/simple\n"
        f"archive-pkg @ {url} --hash=sha256:{digest}\n",
        encoding="utf-8",
    )
    metadata = {
        url: {
            "name": "other" if binding == "wrong-name" else "archive-pkg",
            "version": "4.0",
            "sha256": "b" * 64 if binding == "wrong-hash" else digest,
        }
    }
    (root / "tools/dependency_advisory_sources.json").write_text(json.dumps(metadata), encoding="utf-8")
    if binding == "valid":
        assert locked_packages(root) == [("archive-pkg", "4.0"), ("example-pkg", "1.0")]
    else:
        with pytest.raises(ValueError, match="advisory binding differs"):
            locked_packages(root)


def test_actual_cli_help_usage_and_public_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise argparse help, missing required output and a real public success."""
    help_result = subprocess.run(_command("--help"), capture_output=True, text=True, check=False)
    usage_result = subprocess.run(_command(), capture_output=True, text=True, check=False)
    assert help_result.returncode == 0
    assert "--repo-root" in help_result.stdout
    assert usage_result.returncode == 2
    root = _repository(tmp_path / "repo")
    assert main(["--repo-root", str(root), "--output", str(tmp_path / "pylock.toml")]) == 0
    assert "Exported 3 locked upstream package versions" in capsys.readouterr().out
