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

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform")

from tools.emit_studio_manifest import _ARTIFACT, render  # noqa: E402

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_committed_artifact_matches_the_producer() -> None:
    assert _ARTIFACT.exists(), "run `python tools/emit_studio_manifest.py`"
    assert _ARTIFACT.read_text(encoding="utf-8") == render(), (
        "docs/_generated/studio_manifest.json is stale; run `python tools/emit_studio_manifest.py`"
    )


def test_artifact_is_schema_a_well_formed() -> None:
    payload = json.loads(Path(_ARTIFACT).read_text(encoding="utf-8"))
    assert payload["studio"] == "scpn-control"
    assert payload["contract_era"].startswith("v")
    assert _DIGEST_RE.match(payload["content_digest"]), payload["content_digest"]
    verbs = [verb["verb"] if isinstance(verb, dict) else verb for verb in payload["verbs"]]
    assert len(verbs) == len(set(verbs)), "verbs must be unique"
    assert len(verbs) == 11, "the CONTROL vertical advertises eleven verbs"
    evidence_types = payload["evidence_types"]
    assert all(schema.endswith(".v1") for schema in evidence_types)
    assert len(evidence_types) == len(set(evidence_types)) == 11
