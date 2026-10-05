# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual release-builder API, archive and frontend contracts.

"""Exercise real archives and the unchanged physical builder through PyPA build."""

from __future__ import annotations

import configparser
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import cast

import pytest

from tools.build_release_artifacts import build_release_artifacts, normalise_sdist, validate_artifact

ROOT = Path(__file__).resolve().parents[1]
EPOCH = 1_700_000_000


def _tar(path: Path, name: str = "demo/module.py", kind: bytes = tarfile.REGTYPE, mtime: int = 1) -> None:
    """Write a physical tar with a directory and selected member/header type."""
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        directory = tarfile.TarInfo("demo")
        directory.type = tarfile.DIRTYPE
        directory.mtime = mtime
        archive.addfile(directory)
        member = tarfile.TarInfo(name)
        member.type = kind
        member.mtime = mtime
        member.mode = 0o640
        member.uid = member.gid = 123
        member.uname = member.gname = "fixture"
        member.pax_headers = {"mtime": str(mtime), "uid": "123"}
        member.linkname = "../../outside"
        payload = b"VALUE = 42\n"
        member.size = len(payload) if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)


def _wheel(
    path: Path,
    *,
    target: str = "demo.cli:main",
    entries: str | None = "[console_scripts]\ndemo = demo.cli:main\n",
    metadata: bytes | None = b"License-Expression: AGPL-3.0-or-later\n",
    extra: str | None = None,
) -> None:
    """Write actual ZIP payload/declarations for public archive inspection."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr("demo/cli.py", "def main():\n    return 0\n")
        if metadata is not None:
            archive.writestr("demo-1.dist-info/METADATA", metadata)
        if entries is not None:
            archive.writestr("demo-1.dist-info/entry_points.txt", entries.replace("demo.cli:main", target))
        if extra is not None:
            archive.writestr(extra, "fixture")


def _project(tmp_path: Path, *, backend: str = "setuptools.build_meta") -> Path:
    """Create a real setuptools project with a byte-identical script-root builder.

    The setup hook records inherited epoch/cwd only when explicitly requested,
    and can emit an extra file to exercise the public artifact-count refusal.
    It invokes actual setuptools; no frontend/subprocess result is substituted.
    """
    root = tmp_path / "fixture"
    (root / "tools").mkdir(parents=True)
    for name in ("build_release_artifacts.py", "__init__.py"):
        source = ROOT / "tools" / name
        target = root / "tools" / name
        target.write_bytes(source.read_bytes())
        assert target.read_bytes() == source.read_bytes()
    (root / "src/demo_fixture").mkdir(parents=True)
    (root / "src/demo_fixture/__init__.py").write_text("", encoding="utf-8")
    (root / "src/demo_fixture/cli.py").write_text(
        'def main():\n    print("REAL_INSTALLED_FIXTURE_CLI")\n    return 0\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools==84.0.0"]\n'
        f'build-backend = "{backend}"\n'
        '[project]\nname = "scpn-build-fixture"\nversion = "1.0.0"\n'
        'license = "AGPL-3.0-or-later"\nrequires-python = ">=3.11"\n'
        '[project.scripts]\nscpn-build-fixture = "demo_fixture.cli:main"\n',
        encoding="utf-8",
    )
    (root / "MANIFEST.in").write_text("include tools/*.py\n", encoding="utf-8")
    (root / "setup.py").write_text(
        "import json,os\nfrom pathlib import Path\nfrom setuptools import setup\n"
        "report=os.environ.get('SCPN_FIXTURE_BUILD_REPORT')\n"
        "if report: Path(report).write_text(json.dumps({'epoch':os.environ.get('SOURCE_DATE_EPOCH'),"
        "'cwd':os.getcwd(),'sentinel':os.environ.get('SCPN_FIXTURE_SENTINEL')}))\n"
        "extra=os.environ.get('SCPN_FIXTURE_EXTRA_ARTIFACT')\n"
        "if extra: Path(extra).write_bytes(b'EXTRA_BUILD_HOOK_OUTPUT')\n"
        "setup()\n",
        encoding="utf-8",
    )
    return root


def _run(
    root: Path, arguments: list[str], *, api: bool = False, changes: dict[str, str | None] | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute the real copied CLI/API with native argv/cwd and optional coverage."""
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(root),
        PIP_INDEX_URL="https://pypi.org/simple",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    environment.pop("PIP_EXTRA_INDEX_URL", None)
    for key, value in (changes or {}).items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    script = root / "tools/build_release_artifacts.py"
    if api:
        script = root / "api_entry.py"
        script.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "from tools.build_release_artifacts import build_release_artifacts\n"
            "rows=build_release_artifacts(Path(sys.argv[1]),epoch=int(sys.argv[2]),sdist_only=True)\n"
            "print('PUBLIC_API_SUMMARY '+json.dumps([{'path':str(x.path),'entries':x.entries,"
            "'sha256':x.sha256} for x in rows]))\n",
            encoding="utf-8",
        )
    command = [sys.executable]
    if config := os.environ.get("SCPN_RELEASE_COVERAGE_RC"):
        command.extend(("-m", "coverage", "run", "--parallel-mode", "--rcfile=" + config))
    return subprocess.run(
        [*command, str(script), *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_normalisation_preserves_payload_mode_and_unrelated_temporary_file(tmp_path: Path) -> None:
    """Two real archives become byte-identical without consuming a legacy temp."""
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _tar(first, mtime=1)
    _tar(second, mtime=2)
    legacy = first.with_name("." + first.name + ".tmp")
    legacy.write_bytes(b"UNRELATED_TEMPORARY_BYTES")
    normalise_sdist(first, EPOCH)
    normalise_sdist(second, EPOCH)
    assert first.read_bytes() == second.read_bytes()
    assert legacy.read_bytes() == b"UNRELATED_TEMPORARY_BYTES"
    assert not list(tmp_path.glob(".scpn-sdist-*.tmp"))
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["demo", "demo/module.py"]
        for member in archive.getmembers():
            assert member.mtime == EPOCH and member.uid == member.gid == 0
            assert member.uname == member.gname == ""
        member = archive.getmember("demo/module.py")
        assert member.mode == 0o640
        payload = archive.extractfile(member)
        assert payload is not None and payload.read() == b"VALUE = 42\n"
    summary = validate_artifact(first)
    assert summary.path == first and summary.entries == 2
    assert summary.sha256 == hashlib.sha256(first.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "name",
    (
        "/absolute.py",
        "demo/../outside.py",
        "C:/outside.py",
        "C:relative.py",
        "demo\\docs\\internal\\secret.py",
        "demo/docs/internal/file",
        "demo/.git/config",
        "demo/.coordination/session",
        "demo/papers/manuscript",
        "demo/site/index.html",
    ),
)
def test_public_archive_apis_refuse_nonportable_and_private_names(tmp_path: Path, name: str) -> None:
    """The same physical bad member refuses in wheel, tar and normalization."""
    wheel = tmp_path / "bad.whl"
    sdist = tmp_path / "bad.tar.gz"
    _wheel(wheel, extra=name)
    _tar(sdist, name)
    original = sdist.read_bytes()
    for path in (wheel, sdist):
        with pytest.raises(ValueError, match="archive member"):
            validate_artifact(path)
    with pytest.raises(ValueError, match="archive member"):
        normalise_sdist(sdist, EPOCH)
    assert sdist.read_bytes() == original and not list(tmp_path.glob(".scpn-sdist-*.tmp"))


@pytest.mark.parametrize("kind", (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE))
def test_public_sdist_apis_refuse_link_and_special_members(tmp_path: Path, kind: bytes) -> None:
    """Link/special headers refuse before any payload rewrite or extraction."""
    path = tmp_path / "special.tar.gz"
    _tar(path, kind=kind)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported sdist member type"):
        validate_artifact(path)
    with pytest.raises(ValueError, match="unsupported sdist member type"):
        normalise_sdist(path, EPOCH)
    assert path.read_bytes() == before


@pytest.mark.parametrize("epoch", (-1, 2**32, True, 0.5))
def test_public_epoch_refusals_preserve_archive_and_avoid_build(tmp_path: Path, epoch: object) -> None:
    """Non-gzip epochs refuse both public writers before creating an output."""
    path = tmp_path / "demo.tar.gz"
    _tar(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="source date epoch"):
        normalise_sdist(path, cast(int, epoch))
    outdir = tmp_path / "not_created"
    with pytest.raises(ValueError, match="source date epoch"):
        build_release_artifacts(outdir, epoch=cast(int, epoch))
    assert path.read_bytes() == before and not outdir.exists()


def test_public_wheel_declarations_and_format_errors(tmp_path: Path) -> None:
    """Physical declarations expose real cardinality, decoding and target refusals."""
    package = tmp_path / "package.whl"
    _wheel(package, target="demo:main")
    assert validate_artifact(package).entries == 4
    _wheel(package, entries="[plugins]\ndemo = demo.cli:main\n")
    assert validate_artifact(package).entries == 4
    _wheel(package, metadata=b"Metadata-Version: 2.4\r\nLicense-Expression: AGPL-3.0-or-later\r\n")
    assert validate_artifact(package).entries == 4
    cases: list[tuple[str, str | None, bytes | None]] = [
        ("demo.cli:main", "[console_scripts]\ndemo = demo.cli:main\n", None),
        ("demo.cli:main", "[console_scripts]\ndemo = demo.cli:main\n", b"License-Expression: MIT\n"),
        ("demo.cli:main", "[console_scripts]\ndemo = demo.cli:main\n", b"\xff"),
        ("demo.cli:main", None, b"License-Expression: AGPL-3.0-or-later\n"),
        ("demo.cli:main", "[invalid\n", b"License-Expression: AGPL-3.0-or-later\n"),
        ("absent.module:main", "[console_scripts]\ndemo = demo.cli:main\n", b"License-Expression: AGPL-3.0-or-later\n"),
    ]
    for i, (target, entries, metadata) in enumerate(cases):
        path = tmp_path / f"bad-{i}.whl"
        _wheel(path, target=target, entries=entries, metadata=metadata)
        with pytest.raises((ValueError, UnicodeError, configparser.Error)):
            validate_artifact(path)
    with pytest.raises(ValueError, match="unsupported release artifact"):
        validate_artifact(tmp_path / "unknown.zip")
    broken = tmp_path / "broken.whl"
    broken.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        validate_artifact(broken)


def test_actual_frontend_reproducible_pair_and_installed_console(tmp_path: Path) -> None:
    """Real setuptools builds two identical pairs and the built wheel's CLI runs."""
    root = _project(tmp_path)
    first = _run(root, ["--outdir", "out-a", "--source-date-epoch", str(EPOCH)])
    second = _run(root, ["--outdir", "out-b", "--source-date-epoch", str(EPOCH)])
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    for suffix in ("*.whl", "*.tar.gz"):
        (a,) = (root / "out-a").glob(suffix)
        (b,) = (root / "out-b").glob(suffix)
        assert a.read_bytes() == b.read_bytes()
        summary = validate_artifact(a)
        assert f"{a.name}\tentries={summary.entries}\tsha256={summary.sha256}" in first.stdout
    (wheel,) = (root / "out-a").glob("*.whl")
    target = tmp_path / "installed"
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)]
    installed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=90)
    assert installed.returncode == 0, installed.stderr
    environment = dict(os.environ, PYTHONPATH=str(target), PYTHONDONTWRITEBYTECODE="1")
    executed = subprocess.run(
        [sys.executable, "-c", "from demo_fixture.cli import main;raise SystemExit(main())"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert executed.returncode == 0 and executed.stdout.strip() == "REAL_INSTALLED_FIXTURE_CLI"
    stale = _run(root, ["--outdir", "out-a", "--source-date-epoch", str(EPOCH)])
    assert stale.returncode != 0 and "stale distributions" in stale.stderr


def test_actual_public_build_api_propagates_epoch_cwd_and_environment(tmp_path: Path) -> None:
    """The public API runs a real sdist backend with observed inherited inputs."""
    root = _project(tmp_path)
    output = tmp_path / "artifacts with spaces"
    report = tmp_path / "backend-inputs.json"
    result = _run(
        root,
        [str(output), str(EPOCH)],
        api=True,
        changes={
            "SCPN_FIXTURE_BUILD_REPORT": str(report),
            "SCPN_FIXTURE_SENTINEL": "inherited-value",
            "SOURCE_DATE_EPOCH": "7",
        },
    )
    assert result.returncode == 0, result.stderr
    observation = json.loads(report.read_text())
    assert observation == {"epoch": str(EPOCH), "cwd": str(root), "sentinel": "inherited-value"}
    (artifact,) = output.glob("*.tar.gz")
    assert not list(output.glob("*.whl"))
    row = validate_artifact(artifact)
    assert '"sha256": "' + row.sha256 + '"' in result.stdout


def test_actual_backend_failure_and_extra_outputs_are_retained(tmp_path: Path) -> None:
    """A failing real backend and an extra setup-hook artifact refuse honestly."""
    root = _project(tmp_path, backend="missing_backend_for_real_refusal")
    failed = _run(root, ["--outdir", "failed", "--source-date-epoch", str(EPOCH)])
    assert failed.returncode != 0 and "CalledProcessError" in failed.stderr
    assert (root / "failed").is_dir()
    project = root / "pyproject.toml"
    project.write_text(project.read_text().replace("missing_backend_for_real_refusal", "setuptools.build_meta"))
    output = root / "extra"
    result = _run(
        root,
        ["--outdir", str(output), "--source-date-epoch", str(EPOCH)],
        changes={
            "SCPN_FIXTURE_EXTRA_ARTIFACT": str(output / "additional.whl"),
        },
    )
    assert result.returncode != 0 and "expected 2 distribution artifact(s), found 3" in result.stderr
    assert (output / "additional.whl").read_bytes() == b"EXTRA_BUILD_HOOK_OUTPUT"
    assert len(list(output.glob("*.whl"))) == 2 and len(list(output.glob("*.tar.gz"))) == 1


def test_actual_cli_epoch_routes_and_usage_refusals(tmp_path: Path) -> None:
    """Real Git/environment/explicit routes select epochs before invoking builds."""
    root = _project(tmp_path)
    git = shutil.which("git")
    assert git is not None, "Git is a required native builder test dependency"
    isolated = dict(
        os.environ,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Fixture",
        GIT_AUTHOR_EMAIL="fixture@example.invalid",
        GIT_COMMITTER_NAME="Fixture",
        GIT_COMMITTER_EMAIL="fixture@example.invalid",
        GIT_AUTHOR_DATE=f"{EPOCH} +0000",
        GIT_COMMITTER_DATE=f"{EPOCH} +0000",
    )
    empty = tmp_path / "git-template"
    empty.mkdir()
    for args in (
        ["init", "--template=" + str(empty)],
        ["add", "--", "pyproject.toml", "src"],
        ["-c", "commit.gpgsign=false", "commit", "-m", "Fixture source epoch"],
    ):
        p = subprocess.run(
            [git, *args], cwd=root, env=isolated, capture_output=True, text=True, check=False, timeout=30
        )
        assert p.returncode == 0, p.stderr
    inferred = _run(root, ["--outdir", "git-epoch", "--sdist-only"], changes={"SOURCE_DATE_EPOCH": None})
    configured = _run(root, ["--outdir", "env-epoch", "--sdist-only"], changes={"SOURCE_DATE_EPOCH": "42"})
    assert inferred.returncode == configured.returncode == 0, inferred.stderr + configured.stderr
    assert f"source_date_epoch={EPOCH}" in inferred.stdout and "source_date_epoch=42" in configured.stdout
    routes: list[tuple[list[str], dict[str, str | None] | None, int]] = [
        (["--help"], None, 0),
        (["--unknown-option"], None, 2),
        (["--source-date-epoch", "-1"], None, 2),
        (["--source-date-epoch", str(2**32)], None, 2),
        ([], {"SOURCE_DATE_EPOCH": "not-an-integer"}, 1),
        ([], {"SOURCE_DATE_EPOCH": None, "PATH": ""}, 1),
    ]
    for arguments, changes, expected in routes:
        result = _run(root, arguments, changes=changes)
        assert result.returncode == expected, result.stderr
    assert not (root / "dist").exists()
