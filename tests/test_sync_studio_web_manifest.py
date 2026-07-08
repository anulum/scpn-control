# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for the Studio Web manifest sync guard.
"""Tests for syncing the generated Studio manifest to Studio Web."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

import tools.sync_studio_web_manifest as syncer


def _manifest() -> dict[str, object]:
    return {
        "studio": "scpn-control",
        "ui_module": {
            "remote_entry": "https://www.anulum.org/studios/scpn-control/remoteEntry.js",
            "exposes": ["./Panel"],
            "federation": "module-federation-2",
        },
        "verbs": [],
        "evidence_types": [],
    }


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_read_manifest_rejects_non_object_json(tmp_path: Path) -> None:
    """Manifest JSON must be an object, not an array or scalar."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        syncer.read_manifest(manifest)


def test_validate_deployed_contract_accepts_expected_payload() -> None:
    """The expected CONTROL Studio deployment contract is valid."""
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


def test_main_delegates_to_check_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI passes the requested check mode to the sync helper."""
    calls: list[bool] = []

    def fake_sync_manifest(*, check: bool = False) -> int:
        calls.append(check)
        return 7

    monkeypatch.setattr(syncer, "sync_manifest", fake_sync_manifest)

    assert syncer.main(["--check"]) == 7
    assert calls == [True]
