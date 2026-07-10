# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST disruption feature-source audit
"""Offline tests for :mod:`validation.audit_mast_disruption_feature_sources`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation import audit_mast_disruption_feature_sources as audit
from validation.audit_mast_disruption_feature_sources import (
    NPZ_CHANNELS,
    _sha256_json,
    build_feature_source_audit,
    main,
    render_markdown,
)


def test_audit_is_blocked_while_a_channel_is_lookup_needed() -> None:
    report = build_feature_source_audit()
    assert report["schema_version"] == "scpn-control.mast-disruption-feature-source-audit.v1"
    # BT_T has no catalogued level2 signal yet → overall blocked.
    assert report["status"] == "blocked"
    assert report["channels"]["BT_T"]["status"] == "lookup_needed"
    assert set(report["channels"]) == set(NPZ_CHANNELS)


def test_audit_records_label_algorithm_and_floor() -> None:
    report = build_feature_source_audit(ip_max_floor_ka=150.0)
    label = report["label_algorithm"]
    assert label["method"] == "ip_current_quench"
    assert label["drop_fraction"] == 0.8
    assert label["ip_max_floor_ka"] == 150.0


def test_audit_payload_digest_is_self_consistent() -> None:
    report = build_feature_source_audit()
    digest = report["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == _sha256_json({**report, "payload_sha256": None})


def test_audit_status_is_source_ready_when_all_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    patched = {name: dict(entry) for name, entry in audit.FEATURE_SOURCE_POLICY.items()}
    patched["BT_T"]["status"] = "derived"
    monkeypatch.setattr(audit, "FEATURE_SOURCE_POLICY", patched)
    report = build_feature_source_audit()
    assert report["status"] == "source_ready"
    assert report["channel_status_counts"]["lookup_needed"] == 0
    assert "resolved level2 source" in report["blocked_reason"]


def test_audit_rejects_incomplete_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "NPZ_CHANNELS", (*NPZ_CHANNELS, "phantom_channel"))
    with pytest.raises(ValueError, match="missing channels"):
        build_feature_source_audit()


def test_audit_rejects_incomplete_label_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = {k: v for k, v in audit.LABEL_ALGORITHM.items() if k != "quench_criterion"}
    monkeypatch.setattr(audit, "LABEL_ALGORITHM", broken)
    with pytest.raises(ValueError, match="missing keys"):
        build_feature_source_audit()


def test_render_markdown_lists_every_channel() -> None:
    report = build_feature_source_audit()
    markdown = render_markdown(report)
    assert "# FAIR-MAST Disruption Feature-Source Audit" in markdown
    assert "Channel provenance" in markdown
    for channel in NPZ_CHANNELS:
        assert channel in markdown


def test_main_writes_reports(tmp_path: Path) -> None:
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"
    exit_code = main(["--json-out", str(json_out), "--report-out", str(md_out), "--ip-max-floor-ka", "120"])
    assert exit_code == 0
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["status"] == "blocked"
    assert written["label_algorithm"]["ip_max_floor_ka"] == 120.0
    assert md_out.exists()
