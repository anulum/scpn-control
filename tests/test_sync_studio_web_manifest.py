# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for the Studio Web manifest sync guard.
"""Tests for syncing the generated Studio manifest to Studio Web."""

from __future__ import annotations

import doctest
import html
import json
import os
import pydoc
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

import tools.sync_studio_web_manifest as syncer
from tools.emit_studio_manifest import render

ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict[str, object]:
    """Decode a fresh actual SDK producer carrier for bounded public contract mutations."""
    payload: dict[str, object] = json.loads(render())
    return payload


def _write(path: Path, payload: object) -> None:
    """Write a selected JSON carrier without touching canonical artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_read_manifest_rejects_non_object_json(tmp_path: Path) -> None:
    """Manifest JSON must be an object, not an array or scalar."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        syncer.read_manifest(manifest)


def test_validate_deployed_contract_accepts_expected_payload() -> None:
    """The actual producer declares the expected CONTROL id and fixed UI metadata."""
    syncer.validate_deployed_contract(_manifest())


def test_validate_deployed_contract_rejects_missing_ui_module() -> None:
    """A deployed manifest must include the UI module block."""
    with pytest.raises(ValueError, match="ui_module must be present"):
        syncer.validate_deployed_contract({"studio": "scpn-control"})


def test_validate_deployed_contract_rejects_wrong_studio() -> None:
    """The artifact is specific to the CONTROL Studio id."""
    payload = _manifest()
    payload["studio"] = "other"

    with pytest.raises(ValueError, match="studio must be scpn-control"):
        syncer.validate_deployed_contract(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("remote_entry", "https://example.invalid/remoteEntry.js"),
        ("exposes", ["./Other"]),
        ("federation", "legacy"),
    ],
)
def test_validate_deployed_contract_rejects_wrong_ui_field(field: str, value: object) -> None:
    """The web artifact must match the Hub-facing remote contract exactly."""
    payload = _manifest()
    ui_module = payload["ui_module"]
    assert isinstance(ui_module, dict)
    ui_module[field] = value

    with pytest.raises(ValueError, match=f"ui_module.{field}"):
        syncer.validate_deployed_contract(payload)


def test_sync_manifest_writes_web_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Default sync copies the generated manifest to Studio Web public assets."""
    source = tmp_path / "docs" / "studio_manifest.json"
    web = tmp_path / "studio-web" / "public" / "manifest.json"
    _write(source, _manifest())
    monkeypatch.setattr(syncer, "SOURCE_MANIFEST", source)
    monkeypatch.setattr(syncer, "WEB_MANIFEST", web)

    assert syncer.sync_manifest() == 0

    assert web.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert f"wrote {web}" in capsys.readouterr().out


def test_sync_manifest_check_passes_when_artifact_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check mode accepts a deployed manifest that equals the generated source."""
    source = tmp_path / "docs" / "studio_manifest.json"
    web = tmp_path / "studio-web" / "public" / "manifest.json"
    _write(source, _manifest())
    web.parent.mkdir(parents=True)
    web.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(syncer, "SOURCE_MANIFEST", source)
    monkeypatch.setattr(syncer, "WEB_MANIFEST", web)

    assert syncer.sync_manifest(check=True) == 0


def test_sync_manifest_check_fails_when_web_artifact_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Check mode fails when the deployed manifest has drifted."""
    source = tmp_path / "docs" / "studio_manifest.json"
    web = tmp_path / "studio-web" / "public" / "manifest.json"
    _write(source, _manifest())
    stale_payload = _manifest()
    stale_payload["extra"] = "stale"
    _write(web, stale_payload)
    monkeypatch.setattr(syncer, "SOURCE_MANIFEST", source)
    monkeypatch.setattr(syncer, "WEB_MANIFEST", web)

    assert syncer.sync_manifest(check=True) == 1

    assert "is stale" in capsys.readouterr().out


def test_sync_manifest_check_fails_when_web_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Check mode fails when the deployed manifest does not exist."""
    source = tmp_path / "docs" / "studio_manifest.json"
    web = tmp_path / "studio-web" / "public" / "manifest.json"
    _write(source, _manifest())
    monkeypatch.setattr(syncer, "SOURCE_MANIFEST", source)
    monkeypatch.setattr(syncer, "WEB_MANIFEST", web)

    assert syncer.sync_manifest(check=True) == 1

    assert "invalid or missing" in capsys.readouterr().out


def test_sync_manifest_fails_when_source_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Sync mode refuses an invalid generated source manifest."""
    source = tmp_path / "docs" / "studio_manifest.json"
    web = tmp_path / "studio-web" / "public" / "manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{", encoding="utf-8")
    monkeypatch.setattr(syncer, "SOURCE_MANIFEST", source)
    monkeypatch.setattr(syncer, "WEB_MANIFEST", web)

    assert syncer.sync_manifest() == 1

    assert "is invalid" in capsys.readouterr().out


def test_main_delegates_to_check_mode(tmp_path: Path) -> None:
    """Actual explicit-argv check detects byte drift without writing either producer carrier."""
    source, web = tmp_path / "source.json", tmp_path / "web.json"
    source.write_text(render(), encoding="utf-8")
    web.write_text(render() + "\n", encoding="utf-8")
    before = source.read_bytes(), web.read_bytes()
    assert syncer.main(["--source", str(source), "--destination", str(web), "--check"]) == 1
    assert (source.read_bytes(), web.read_bytes()) == before


def _cli(source: Path, web: Path, *args: str, stdlib_only: bool = False) -> subprocess.CompletedProcess[str]:
    """Run the canonical public CLI from the caller directory with optional site-package absence."""
    return subprocess.run(
        [sys.executable]
        + (["-S"] if stdlib_only else [])
        + [str(ROOT / "tools/sync_studio_web_manifest.py"), "--source", str(source), "--destination", str(web), *args],
        cwd=source.parent,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_real_cli_copies_exact_utf8_crlf_bytes(tmp_path: Path) -> None:
    """Actual producer CRLF bytes survive copy/check and LF-only equivalence is refused as drift."""
    source, web = tmp_path / "source.json", tmp_path / "copied.json"
    text = render().replace("\n", "\r\n")
    source.write_bytes(text.encode("utf-8"))
    decoded, payload = syncer.read_manifest(source)
    assert decoded.encode("utf-8") == source.read_bytes() and payload["studio"] == "scpn-control"
    copied = _cli(source, web)
    assert copied.returncode == 0 and web.read_bytes() == source.read_bytes()
    checked = _cli(source, web, "--check")
    assert checked.returncode == 0 and not checked.stdout and not checked.stderr
    web.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    before = source.read_bytes(), web.read_bytes()
    stale = _cli(source, web, "--check")
    assert stale.returncode == 1 and "is stale" in stale.stdout
    assert (source.read_bytes(), web.read_bytes()) == before


@pytest.mark.parametrize("side", ["source", "web"])
@pytest.mark.parametrize(
    "kind", ["utf8", "syntax", "array", "duplicate", "nested-duplicate", "nonfinite", "overflow", "directory"]
)
def test_real_cli_refuses_invalid_carriers(tmp_path: Path, side: str, kind: str) -> None:
    """Malformed actual producer carriers refuse copy/check while preserving both local paths."""
    source, web = tmp_path / "source.json", tmp_path / "web.json"
    text = render()
    source.write_text(text, encoding="utf-8")
    web.write_text(text, encoding="utf-8")
    invalid = source if side == "source" else web
    if kind == "directory":
        # Keep the original real carrier elsewhere; no repository or other-owner bytes are removed.
        invalid.rename(tmp_path / "original.json")
        invalid.mkdir()
    else:
        versions = {
            "utf8": b"\xff",
            "syntax": text.encode("utf-8") + b"{",
            "array": ("[" + text + "]").encode("utf-8"),
            "duplicate": text.replace("{", '{"studio": "wrong",', 1).encode("utf-8"),
            "nested-duplicate": text.replace('"ui_module": {', '"ui_module": {"exposes": [],', 1).encode("utf-8"),
            "nonfinite": text.encode("utf-8"),
            "overflow": text.encode("utf-8"),
        }
        malformed = versions[kind]
        if kind in {"nonfinite", "overflow"}:
            field = '"studio_version": ' + json.dumps(json.loads(text)["studio_version"])
            assert text.count(field) == 1
            malformed = text.replace(
                field, '"studio_version": ' + ("NaN" if kind == "nonfinite" else "1e400"), 1
            ).encode()
        invalid.write_bytes(malformed)
    before = [p.read_bytes() if p.is_file() else None for p in (source, web)]
    result = _cli(source, web, *(["--check"] if side == "web" else []))
    assert result.returncode == 1 and "invalid" in result.stdout and "Traceback" not in result.stderr
    assert [p.read_bytes() if p.is_file() else None for p in (source, web)] == before


@pytest.mark.parametrize("field", ["studio", "ui_module", "remote_entry", "exposes", "federation"])
def test_real_cli_refuses_changed_declared_ui_contract(tmp_path: Path, field: str) -> None:
    """Actual generated metadata mutations refuse copy before overwriting a valid destination."""
    source, web = tmp_path / "source.json", tmp_path / "web.json"
    payload = _manifest()
    if field in {"studio", "ui_module"}:
        payload[field] = None
    else:
        ui = payload["ui_module"]
        assert isinstance(ui, dict)
        ui[field] = None
    _write(source, payload)
    web.write_text(render(), encoding="utf-8")
    before = source.read_bytes(), web.read_bytes()
    result = _cli(source, web)
    assert result.returncode == 1 and "invalid" in result.stdout
    assert (source.read_bytes(), web.read_bytes()) == before


@pytest.mark.parametrize("blocked", ["directory", "parent-file"])
def test_public_copy_refuses_actual_write_errors(tmp_path: Path, blocked: str) -> None:
    """Actual directory/file parent collisions return fixed refusals without destroying existing bytes."""
    source = tmp_path / "source.json"
    source.write_text(render(), encoding="utf-8")
    existing = tmp_path / "blocked"
    if blocked == "directory":
        existing.mkdir()
        web = existing
    else:
        existing.write_text(render(), encoding="utf-8")
        web = existing / "nested" / "web.json"
    before = existing.read_bytes() if existing.is_file() else None
    result = _cli(source, web)
    assert result.returncode == 1 and "Studio Web manifest refused:" in result.stderr
    assert "Traceback" not in result.stderr
    assert existing.is_dir() if before is None else existing.read_bytes() == before


def test_public_argv_and_stdlib_cli_need_no_producer_dependencies(tmp_path: Path) -> None:
    """Public argv is intact; a real python -S CLI copies and checks prepared producer bytes."""
    source, web = tmp_path / "source.json", tmp_path / "nested" / "ľ" / "web.json"
    payload = _manifest()
    payload["studio_version"] = 1.25
    _write(source, payload)
    argv = sys.argv.copy()
    assert syncer.main(["--source", str(source), "--destination", str(web)]) == 0 and sys.argv == argv
    assert web.read_bytes() == source.read_bytes()
    for flags in ([], ["--check"], ["--help"]):
        result = _cli(source, web, *flags, stdlib_only=True)
        assert result.returncode == 0 and not result.stderr
    assert web.read_bytes() == source.read_bytes()


def test_native_examples_and_actual_html_rendering(tmp_path: Path) -> None:
    """Actual canonical-artifact native examples execute and pydoc renders the byte/state/error contract."""
    result = doctest.testmod(syncer, raise_on_error=True)
    assert result.attempted == 3 and result.failed == 0
    document = pydoc.HTMLDoc().document(syncer)
    artifact = tmp_path / "sync_studio_web_manifest.html"
    artifact.write_text(document, encoding="utf-8")
    visible = html.unescape(artifact.read_text(encoding="utf-8")).replace("\N{NO-BREAK SPACE}", " ")
    assert "read_manifest" in visible and "sync_manifest" in visible and "--source" in visible
    assert "newline translation" in visible and "without atomicity" in visible


def test_actual_script_defaults_belong_to_script_repository(tmp_path: Path) -> None:
    """A copied real stdlib script uses its repository artifacts from an unrelated working directory."""
    repository = tmp_path / "repository"
    script = repository / "tools" / "sync_studio_web_manifest.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "tools/sync_studio_web_manifest.py", script)
    source = repository / "docs/_generated/studio_manifest.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(render().replace("\n", "\r\n").encode("utf-8"))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    for flags in ([], ["--check"]):
        result = subprocess.run(
            [sys.executable, "-S", str(script), *flags],
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": ""},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0 and not result.stderr
    assert (repository / "studio-web/public/manifest.json").read_bytes() == source.read_bytes()
    assert list(cwd.iterdir()) == []


def test_actual_cli_relative_paths_resolve_from_caller(tmp_path: Path) -> None:
    """Explicit relative source/destination paths belong to the real process cwd, not the script root."""
    source, web = tmp_path / "generated.json", tmp_path / "caller-web.json"
    source.write_text(render(), encoding="utf-8")
    result = _cli(source, web, "--source", source.name, "--destination", web.name, stdlib_only=True)
    assert result.returncode == 0 and web.read_bytes() == source.read_bytes()
