# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — studio schema-A manifest artifact drift guard
"""Drift guard + schema-A shape checks for the emitted studio CapabilityManifest.

The committed ``docs/_generated/studio_manifest.json`` is the federation-gate artifact
the SCPN-STUDIO keeper reviews with ``validate_studio_manifest``. These tests keep it in
lock-step with :func:`scpn_control.studio.manifest.build_manifest` (so a verb or
evidence-schema change cannot leave a stale federation manifest) and assert the schema-A
shape the keeper's gate requires.
"""

from __future__ import annotations

import doctest
import html
import importlib
import importlib.metadata
import json
import os
import pydoc
import re
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

pytest.importorskip("scpn_studio_platform")

import scpn_control.studio.manifest as manifest_module  # noqa: E402
import tools.emit_studio_manifest as emitter  # noqa: E402
from tools.emit_studio_manifest import _ARTIFACT, render  # noqa: E402

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ROOT = Path(__file__).resolve().parents[1]


def _cli(artifact: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the actual source CLI against a caller-owned artifact with the real Studio producer."""
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/emit_studio_manifest.py"), "--artifact", str(artifact), *args],
        cwd=artifact.parent,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    "kind", ["utf8", "syntax", "array", "duplicate", "nested-duplicate", "nonfinite", "overflow", "directory"]
)
def test_actual_check_refuses_invalid_artifact_bytes(tmp_path: Path, kind: str) -> None:
    """Malformed real producer carriers refuse a positive manifest check without changing the artifact."""
    artifact = tmp_path / "manifest.json"
    text = render()
    if kind == "directory":
        artifact.mkdir()
        before = None
    else:
        values = {
            "utf8": b"\xff",
            "syntax": text.encode("utf-8") + b"{",
            "array": ("[" + text + "]").encode("utf-8"),
            "duplicate": text.replace("{", '{"studio": "conflicting-studio",', 1).encode("utf-8"),
            "nested-duplicate": text.replace('"ui_module": {', '"ui_module": {"exposes": [],', 1).encode("utf-8"),
            "nonfinite": text.encode("utf-8"),
            "overflow": text.encode("utf-8"),
        }
        before = values[kind]
        if kind in {"nonfinite", "overflow"}:
            version = json.loads(text)["studio_version"]
            field = '"studio_version": ' + json.dumps(version)
            assert text.count(field) == 1
            token = "NaN" if kind == "nonfinite" else "1e400"
            before = text.replace(field, '"studio_version": ' + token, 1).encode("utf-8")
        artifact.write_bytes(before)
    result = _cli(artifact, "--check")
    assert result.returncode == 1 and "Studio manifest refused:" in result.stderr
    assert "Traceback" not in result.stderr
    assert artifact.is_dir() if before is None else artifact.read_bytes() == before


def test_committed_artifact_matches_the_producer() -> None:
    """The canonical object matches the actual SDK producer except its local version stamp."""
    # ``studio_version`` is an environment-dependent stamp (the installed distribution
    # version, or "0+unknown" from a non-installed source tree as in CI), so it is
    # excluded — the structural contract (verbs, evidence, digest, era) stays in lock-step,
    # and content_digest is computed over verbs+evidence, not studio_version.
    assert _ARTIFACT.exists(), "run `python tools/emit_studio_manifest.py`"
    committed = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    produced = json.loads(render())
    committed.pop("studio_version", None)
    produced.pop("studio_version", None)
    assert committed == produced, (
        "docs/_generated/studio_manifest.json is stale; run `python tools/emit_studio_manifest.py`"
    )


def test_artifact_is_schema_a_well_formed() -> None:
    """The canonical artifact declares unique verbs/schemas and the expected federation metadata."""
    payload = json.loads(Path(_ARTIFACT).read_text(encoding="utf-8"))
    assert payload["studio"] == "scpn-control"
    assert payload["contract_era"].startswith("v")
    assert _DIGEST_RE.match(payload["content_digest"]), payload["content_digest"]
    verbs = [verb["verb"] if isinstance(verb, dict) else verb for verb in payload["verbs"]]
    assert len(verbs) == len(set(verbs)), "verbs must be unique"
    assert len(verbs) == 12, "the CONTROL vertical advertises twelve verbs"
    evidence_types = payload["evidence_types"]
    assert all(schema.endswith(".v1") for schema in evidence_types)
    assert len(evidence_types) == len(set(evidence_types)) == 12
    ui_module = payload["ui_module"]
    assert ui_module["remote_entry"] == "https://anulum.github.io/scpn-control/studios/scpn-control/remoteEntry.js"
    assert ui_module["exposes"] == ["./Panel"]
    assert ui_module["federation"] == "module-federation-2"


def test_manifest_ui_module_matches_studio_federation_contract() -> None:
    """The producer advertises the declared remote and stable panel exposure."""
    manifest = manifest_module.build_manifest()

    assert manifest.ui_module is not None
    assert manifest.ui_module.remote_entry == manifest_module.UI_REMOTE_ENTRY
    assert manifest.ui_module.exposes == (manifest_module.UI_PANEL_EXPOSE,)


def test_main_check_passes_when_artifact_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--check`` accepts a committed artifact that matches the producer."""
    artifact = tmp_path / "studio_manifest.json"
    artifact.write_text(render(), encoding="utf-8")
    monkeypatch.setattr(emitter, "_ARTIFACT", artifact)

    assert emitter.main(["--check"]) == 0


def test_main_check_ignores_environment_specific_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``studio_version`` differences do not make the drift check fail."""
    artifact = tmp_path / "studio_manifest.json"
    payload = json.loads(render())
    payload["studio_version"] = "different-local-stamp"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(emitter, "_ARTIFACT", artifact)

    assert emitter.main(["--check"]) == 0


def test_main_check_fails_when_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """``--check`` fails closed when the generated artifact is absent."""
    artifact = tmp_path / "missing" / "studio_manifest.json"
    monkeypatch.setattr(emitter, "_ARTIFACT", artifact)

    assert emitter.main(["--check"]) == 1

    assert "is missing" in capsys.readouterr().out


def test_main_check_fails_when_artifact_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """``--check`` fails closed when committed manifest content drifts."""
    artifact = tmp_path / "studio_manifest.json"
    payload = json.loads(render())
    payload["ui_module"]["remote_entry"] = "https://www.anulum.org/studios/scpn-control/stale.js"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(emitter, "_ARTIFACT", artifact)

    assert emitter.main(["--check"]) == 1

    assert "is stale" in capsys.readouterr().out


def test_main_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Default invocation writes the deterministic generated artifact."""
    artifact = tmp_path / "generated" / "studio_manifest.json"
    monkeypatch.setattr(emitter, "_ARTIFACT", artifact)

    assert emitter.main([]) == 0

    assert artifact.read_text(encoding="utf-8") == render()
    assert f"wrote {artifact}" in capsys.readouterr().out


def test_manifest_version_falls_back_when_distribution_metadata_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-tree import stamps manifests with the non-fabricated sentinel."""

    def missing_distribution(distribution_name: str) -> str:
        """Represent absent installed metadata in the historical source-sentinel regression."""
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", missing_distribution)
        importlib.reload(manifest_module)
        assert manifest_module.STUDIO_VERSION == "0+unknown"
        assert manifest_module.build_manifest().studio_version == "0+unknown"

    importlib.reload(manifest_module)


def test_actual_cli_write_and_semantic_check(tmp_path: Path) -> None:
    """The canonical CLI writes the real producer and checks reordered objects without rewriting."""
    artifact = tmp_path / "manifest.json"
    written = _cli(artifact)
    assert written.returncode == 0 and written.stdout.strip() == f"wrote {artifact}"
    assert artifact.read_text(encoding="utf-8") == render()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    # Parity excludes the stamp's value; it does not perform SDK field-type validation.
    payload["studio_version"] = 1.25
    reordered = dict(reversed(list(payload.items())))
    artifact.write_text(json.dumps(reordered, ensure_ascii=False), encoding="utf-8")
    before = artifact.read_bytes()
    checked = _cli(artifact, "--check")
    assert checked.returncode == 0 and not checked.stdout and not checked.stderr
    assert artifact.read_bytes() == before
    payload["studio"] = "different-studio"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    before = artifact.read_bytes()
    stale = _cli(artifact, "--check")
    assert stale.returncode == 1 and "is stale" in stale.stdout and not stale.stderr
    assert artifact.read_bytes() == before


@pytest.mark.parametrize("blocked", ["directory", "parent-file"])
def test_public_write_refuses_actual_filesystem_errors(tmp_path: Path, blocked: str) -> None:
    """Directory destinations and a file in the parent chain refuse writes without deleting bytes."""
    existing = tmp_path / "blocked"
    if blocked == "directory":
        existing.mkdir()
        artifact = existing
    else:
        existing.write_text(render(), encoding="utf-8")
        artifact = existing / "nested" / "manifest.json"
    before = existing.read_bytes() if existing.is_file() else None
    assert emitter.main(["--artifact", str(artifact)]) == 1
    assert existing.is_dir() if before is None else existing.read_bytes() == before


def test_public_main_preserves_argv_and_creates_selected_parents(tmp_path: Path) -> None:
    """Explicit argv writes a nested caller path without consuming pytest's process arguments."""
    argv = sys.argv.copy()
    artifact = tmp_path / "nested" / "unicode-ľ" / "manifest.json"
    assert emitter.main(["--artifact", str(artifact)]) == 0
    assert artifact.read_text(encoding="utf-8") == render() and sys.argv == argv
    assert emitter.main(["--artifact", str(artifact), "--check"]) == 0 and sys.argv == argv


@pytest.mark.parametrize("help_only", [True, False])
def test_actual_stdlib_cli_without_producer_dependencies(tmp_path: Path, help_only: bool) -> None:
    """A real python -S process provides help but refuses producer execution without site packages."""
    artifact = tmp_path / "manifest.json"
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "tools/emit_studio_manifest.py"), "--artifact", str(artifact)]
        + (["--help"] if help_only else []),
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == (0 if help_only else 1)
    if help_only:
        assert "--artifact" in result.stdout and not result.stderr
    else:
        assert "Studio manifest refused:" in result.stderr and "Traceback" not in result.stderr
    assert not artifact.exists()


def test_actual_native_example_and_html_rendering(tmp_path: Path) -> None:
    """The owning-language example executes against the real producer and pydoc renders its contracts."""
    result = doctest.testmod(emitter, raise_on_error=True)
    assert result.attempted == 2 and result.failed == 0
    document = pydoc.HTMLDoc().document(emitter)
    output = tmp_path / "emit_studio_manifest.html"
    output.write_text(document, encoding="utf-8")
    rendered = output.read_text(encoding="utf-8")
    assert "render" in rendered and "main" in rendered and "--artifact" in rendered
    visible = html.unescape(rendered).replace("\N{NO-BREAK SPACE}", " ")
    assert "SDK compatibility" in visible and "trailing LF" in visible
