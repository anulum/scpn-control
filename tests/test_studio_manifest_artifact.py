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

import importlib
import importlib.metadata
import json
import re
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

pytest.importorskip("scpn_studio_platform")

import scpn_control.studio.manifest as manifest_module  # noqa: E402
import tools.emit_studio_manifest as emitter  # noqa: E402
from tools.emit_studio_manifest import _ARTIFACT, render  # noqa: E402

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_committed_artifact_matches_the_producer() -> None:
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
    assert ui_module["remote_entry"] == "https://www.anulum.org/studios/scpn-control/remoteEntry.js"
    assert ui_module["exposes"] == ["./Panel"]
    assert ui_module["federation"] == "module-federation-2"


def test_manifest_ui_module_matches_studio_federation_contract() -> None:
    """The producer advertises the deployed remote and stable panel exposure."""

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
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", missing_distribution)
        importlib.reload(manifest_module)
        assert manifest_module.STUDIO_VERSION == "0+unknown"
        assert manifest_module.build_manifest().studio_version == "0+unknown"

    importlib.reload(manifest_module)
