# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio panel feed tests
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Tests for the studio panel feed serialiser.

The feed is the single wire source the federated panel reads, so these tests check
the verb/claim reductions, that both the explicit-digest and manifest-derived-digest
paths agree with the manifest, and that the representative feed carries the full
honesty spectrum the mappers emit (a validated certificate, a validation-gap
predictor, and bounded claims for the rest).

Skips cleanly when the optional ``scpn-studio-platform`` SDK is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_control.studio import (  # noqa: E402
    CONTROL_VERBS,
    FEED_SCHEMA,
    build_manifest,
    claim_summary,
    render_feed_json,
    representative_bundles,
    representative_feed,
    studio_feed,
    verb_summary,
)
from scpn_control.studio.verbs import evidence_schemas  # noqa: E402


def _verb(name: str):  # type: ignore[no-untyped-def]
    return next(verb for verb in CONTROL_VERBS if verb.name == name)


# ── verb_summary ───────────────────────────────────────────────────────
def test_verb_summary_realtime_carries_deadline() -> None:
    summary = verb_summary(_verb("regulate"))
    assert summary["name"] == "regulate"
    assert summary["safety_tier"] == "certified"
    assert summary["side_effect"] == "live-hardware"
    assert summary["timing_class"] == "realtime"
    assert summary["deadline_us"] == 5.0
    assert summary["domain_distinctive"] is True


def test_verb_summary_interactive_omits_deadline() -> None:
    summary = verb_summary(_verb("reconstruct"))
    assert summary["timing_class"] == "interactive"
    assert "deadline_us" not in summary
    assert summary["domain_distinctive"] is False  # reconstruct is a core-spine verb


# ── claim_summary ──────────────────────────────────────────────────────
def test_claim_summary_reduces_bundle_to_panel_fields() -> None:
    efit = representative_bundles()[0]
    summary = claim_summary(efit)
    assert summary == {
        "schema": "studio.efit-reconstruction.v1",
        "status": "bounded-model",
        "admission": "rejected",
        "kind": "measured",
        "freshness": "traceable-unchecked",
    }


def test_claim_summary_omits_freshness_when_absent() -> None:
    """A bundle without a freshness marker omits the freshness field (branch 131->133).

    claim_summary only records freshness when the bundle carries it, so a bundle
    whose freshness is None yields a summary with the core panel fields and no
    freshness key.
    """
    import dataclasses

    base = representative_bundles()[0]
    bundle = dataclasses.replace(base, freshness=None)

    summary = claim_summary(bundle)

    assert "freshness" not in summary
    assert summary["schema"] == base.schema


# ── studio_feed ────────────────────────────────────────────────────────
def test_studio_feed_uses_explicit_digest() -> None:
    feed = studio_feed([], content_digest="sha256:deadbeef")
    assert feed["feed_schema"] == FEED_SCHEMA
    assert feed["studio"] == "scpn-control"
    assert feed["content_digest"] == "sha256:deadbeef"
    assert feed["claims"] == []
    verbs = feed["verbs"]
    assert isinstance(verbs, list)
    assert len(verbs) == len(CONTROL_VERBS)


def test_studio_feed_derives_digest_from_manifest_when_absent() -> None:
    feed = studio_feed([])
    assert feed["content_digest"] == build_manifest().content_digest


# ── representative_bundles ─────────────────────────────────────────────
def test_representative_bundles_cover_every_schema() -> None:
    bundles = representative_bundles()
    schemas = {bundle.schema for bundle in bundles}
    assert schemas == set(evidence_schemas())
    assert len(bundles) == len(evidence_schemas())


def test_representative_bundles_carry_the_honesty_spectrum() -> None:
    by_schema = {bundle.schema: bundle for bundle in representative_bundles()}
    # The held safety certificate is the one claim that renders as validated.
    assert by_schema["studio.safety-certificate.v1"].renders_as_validated is True
    # The fixed-weight predictor lands on the validation-gap boundary.
    assert by_schema["studio.disruption-prediction.v1"].claim_boundary.status.value == "validation-gap"
    # Every other representative claim is bounded and not validated.
    for schema, bundle in by_schema.items():
        if schema == "studio.safety-certificate.v1":
            continue
        assert bundle.renders_as_validated is False


# ── representative_feed + render_feed_json ─────────────────────────────
def test_representative_feed_is_complete_and_agrees_with_manifest() -> None:
    feed = representative_feed()
    assert feed["feed_schema"] == FEED_SCHEMA
    assert feed["content_digest"] == build_manifest().content_digest
    verbs = feed["verbs"]
    claims = feed["claims"]
    assert isinstance(verbs, list)
    assert isinstance(claims, list)
    assert len(verbs) == len(CONTROL_VERBS)
    assert len(claims) == len(evidence_schemas())
    # Exactly one representative claim renders as validated.
    validated = [c for c in claims if c["status"] == "reference-validated" and c["admission"] == "admitted"]
    assert len(validated) == 1


def test_render_feed_json_is_deterministic_sorted_json() -> None:
    text = render_feed_json()
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert parsed["feed_schema"] == FEED_SCHEMA
    # Sorted-key, indented rendering is stable run to run.
    assert render_feed_json() == text


def test_committed_feed_artifact_matches_the_producer() -> None:
    # The panel's standalone artefact must not drift from the Python producer — this
    # is the whole point of the feed (no second source of truth). Regenerate with
    # `python -m scpn_control.studio.feed > studio-web/public/studio-feed.json`.
    #
    # ``studio_version`` is an environment-dependent stamp (the installed distribution
    # version, or ``"0+unknown"`` when run from a non-installed source tree as in CI),
    # so it is excluded — the structural contract (verbs, claims, digest, schema) is
    # what must stay in lock-step.
    repo_root = Path(__file__).resolve().parents[1]
    artifact = repo_root / "studio-web" / "public" / "studio-feed.json"
    committed = json.loads(artifact.read_text(encoding="utf-8"))
    produced = representative_feed()
    committed.pop("studio_version", None)
    produced.pop("studio_version", None)
    assert committed == produced
