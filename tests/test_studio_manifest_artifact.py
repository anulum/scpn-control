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

pytest.importorskip("scpn_studio_platform")

import scpn_control.studio.manifest as manifest_module  # noqa: E402
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
