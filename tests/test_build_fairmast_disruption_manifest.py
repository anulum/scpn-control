# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST disruption acquisition-campaign builder
"""Offline tests for :mod:`validation.build_fairmast_disruption_manifest`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation.build_fairmast_disruption_manifest import (
    _load_cpf_rows,
    _sha256_json,
    build_campaign,
    main,
    render_markdown,
    select_shots,
)

_FIXED_TS = "2026-07-10T00:00:00+00:00"


def _cpf_rows() -> list[dict[str, Any]]:
    return [
        {"shot_id": 11766, "useful": 1, "abort": 0, "ip_max_ka": 578.0},  # selected
        {"shot_id": 20002, "useful": 0, "abort": 0, "ip_max_ka": 500.0},  # not_useful
        {"shot_id": 20003, "useful": 1, "abort": 1, "ip_max_ka": 500.0},  # aborted
        {"shot_id": 20004, "useful": 1, "abort": 0, "ip_max_ka": 50.0},  # ip_max_below_floor
        {"shot_id": 20005, "useful": 0, "abort": 1, "ip_max_ka": 500.0},  # multiple reasons
    ]


def test_select_shots_applies_quality_filter() -> None:
    split = select_shots(_cpf_rows(), ip_max_floor_ka=100.0)
    assert [s["shot_id"] for s in split["selected"]] == [11766]
    assert split["selected"][0]["s3_key"] == "level2/shots/11766"
    reasons = {e["shot_id"]: e["reasons"] for e in split["excluded"]}
    assert reasons[20002] == ["not_useful"]
    assert reasons[20003] == ["aborted"]
    assert reasons[20004] == ["ip_max_below_floor"]
    assert set(reasons[20005]) == {"not_useful", "aborted"}


def test_select_shots_rejects_incomplete_row() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        select_shots([{"shot_id": 1, "useful": 1}], ip_max_floor_ka=100.0)


def test_build_campaign_is_planned_and_self_digested() -> None:
    report = build_campaign(
        _cpf_rows(), dataset_id="mast-disruption-pilot", ip_max_floor_ka=100.0, generated_at=_FIXED_TS
    )
    assert report["schema_version"] == "scpn-control.mast-disruption-acquisition-campaign.v1"
    assert report["status"] == "planned"
    assert report["synthetic"] is False
    assert report["n_selected"] == 1
    assert report["n_excluded"] == 4
    assert report["label_algorithm"]["ip_max_floor_ka"] == 100.0
    assert report["data_access"]["licence"] == "CC-BY-SA-4.0"
    assert report["data_access"]["licence_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert len(report["data_access"]["citations"]) == 2
    assert "10.1016/j.softx.2024.101869" in report["data_access"]["citation"]
    assert report["data_access"]["source_policy_url"] == "https://mastapp.site/"
    assert report["generated_at"] == _FIXED_TS
    assert report["payload_sha256"] == _sha256_json({**report, "payload_sha256": None})


def test_build_campaign_is_deterministic() -> None:
    kwargs = {"dataset_id": "d", "ip_max_floor_ka": 100.0, "generated_at": _FIXED_TS}
    first = build_campaign(_cpf_rows(), **kwargs)
    second = build_campaign(_cpf_rows(), **kwargs)
    assert first["payload_sha256"] == second["payload_sha256"]


def test_render_markdown_summarises_campaign() -> None:
    report = build_campaign(_cpf_rows(), dataset_id="d", ip_max_floor_ka=100.0, generated_at=_FIXED_TS)
    markdown = render_markdown(report)
    assert "# FAIR-MAST Disruption Acquisition Campaign" in markdown
    assert "Selected" in markdown


def test_load_cpf_rows_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        _load_cpf_rows(path)


def test_main_writes_campaign(tmp_path: Path) -> None:
    cpf = tmp_path / "cpf.json"
    cpf.write_text(json.dumps(_cpf_rows()), encoding="utf-8")
    json_out = tmp_path / "out" / "campaign.json"
    md_out = tmp_path / "out" / "campaign.md"
    exit_code = main(
        [
            "--cpf-json",
            str(cpf),
            "--dataset-id",
            "mast-disruption-pilot",
            "--json-out",
            str(json_out),
            "--report-out",
            str(md_out),
            "--ip-max-floor-ka",
            "100",
            "--generated-at",
            _FIXED_TS,
        ]
    )
    assert exit_code == 0
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["status"] == "planned"
    assert written["n_selected"] == 1
    assert md_out.exists()
