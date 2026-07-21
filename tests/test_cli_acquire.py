# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI acquire-mdsplus-shot command tests
"""Tests for the acquire-mdsplus-shot CLI command and its parse/emit helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@dataclass(frozen=True)
class _AcquisitionResult:
    dataset_id: str
    output_npz: Path
    manifest_json: Path
    checksum_sha256: str


def test_acquire_mdsplus_shot_manual_text_success(runner, tmp_path, monkeypatch):
    import scpn_control.core.mdsplus_acquisition as acquisition

    calls = {}

    def fake_acquire_mdsplus_shot(**kwargs):
        calls.update(kwargs)
        return _AcquisitionResult(
            dataset_id="diii-d-163303-manual",
            output_npz=Path(kwargs["output_npz"]),
            manifest_json=Path(kwargs["manifest_json"]),
            checksum_sha256="e" * 64,
        )

    monkeypatch.setattr(acquisition, "acquire_mdsplus_shot", fake_acquire_mdsplus_shot)
    output_npz = tmp_path / "shot.npz"
    manifest_json = tmp_path / "manifest.json"
    signal = json.dumps({"name": "ip", "node": "\\\\IP", "units": "A", "timebase": "s"})

    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--tree",
            "DIII-D",
            "--shot",
            "163303",
            "--signal",
            signal,
            "--output-npz",
            str(output_npz),
            "--manifest-json",
            str(manifest_json),
        ],
    )

    assert result.exit_code == 0
    assert calls["tree"] == "DIII-D"
    assert calls["shot"] == 163303
    assert calls["source_uri"] == "mdsplus://DIII-D/163303"
    assert calls["signals"][0].name == "ip"
    assert "Dataset: diii-d-163303-manual" in result.output
    assert "Status: pass" in result.output


def test_acquire_mdsplus_shot_spec_json_success(runner, tmp_path, monkeypatch):
    import scpn_control.core.mdsplus_acquisition as acquisition

    signal = acquisition.MDSplusSignalSpec(name="ip", node="\\IP", units="A", timebase="s")
    request = acquisition.MDSplusAcquisitionRequest(
        tree="DIII-D",
        shot=163303,
        source_uri="mdsplus://DIII-D/163303",
        access_policy="facility-approved",
        licence="facility data policy",
        signals=[signal],
    )
    calls = {}

    def fake_load_mdsplus_acquisition_request(path):
        calls["spec_path"] = path
        return request

    def fake_acquire_mdsplus_shot(**kwargs):
        calls.update(kwargs)
        return _AcquisitionResult(
            dataset_id="diii-d-163303-spec",
            output_npz=Path(kwargs["output_npz"]),
            manifest_json=Path(kwargs["manifest_json"]),
            checksum_sha256="f" * 64,
        )

    monkeypatch.setattr(acquisition, "load_mdsplus_acquisition_request", fake_load_mdsplus_acquisition_request)
    monkeypatch.setattr(acquisition, "acquire_mdsplus_shot", fake_acquire_mdsplus_shot)
    spec_path = tmp_path / "request.json"
    spec_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--spec-json",
            str(spec_path),
            "--source-uri",
            "mdsplus://override/163303",
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dataset_id"] == "diii-d-163303-spec"
    assert calls["spec_path"] == str(spec_path)
    assert calls["tree"] == "DIII-D"
    assert calls["shot"] == 163303
    assert calls["source_uri"] == "mdsplus://override/163303"
    assert calls["signals"] == [signal]


def test_acquire_mdsplus_shot_manual_missing_inputs_text_failure(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Status: fail" in result.output
    assert "--tree is required" in result.output


def test_acquire_mdsplus_shot_missing_shot_and_signal_failures(runner, tmp_path):
    base_args = [
        "acquire-mdsplus-shot",
        "--tree",
        "DIII-D",
        "--output-npz",
        str(tmp_path / "shot.npz"),
        "--manifest-json",
        str(tmp_path / "manifest.json"),
        "--json-out",
    ]

    missing_shot = runner.invoke(main, base_args)
    assert missing_shot.exit_code == 1
    assert "--shot is required" in json.loads(missing_shot.output)["error"]

    missing_signal = runner.invoke(main, [*base_args, "--shot", "163303"])
    assert missing_signal.exit_code == 1
    assert "at least one --signal" in json.loads(missing_signal.output)["error"]


def test_acquire_mdsplus_shot_rejects_non_object_signal_json(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--tree",
            "DIII-D",
            "--shot",
            "163303",
            "--signal",
            "[1, 2, 3]",
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "signal specification must be a JSON object" in payload["error"]


def test_acquire_mdsplus_shot_fails_closed_before_external_access(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--shot",
            "163303",
            "--signal",
            json.dumps({"name": "ip", "node": "\\IP", "units": "A", "timebase": "s"}),
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "--tree is required" in payload["error"]
    assert not (tmp_path / "shot.npz").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_parse_mdsplus_signal_spec_rejects_non_object_payload() -> None:
    from scpn_control.cli import _parse_mdsplus_signal_spec

    with pytest.raises(ValueError, match="JSON object"):
        _parse_mdsplus_signal_spec(json.dumps(["not", "a", "signal"]))


def test_emit_campaign_result_skips_execution_line_for_non_dict_execution(capsys) -> None:
    """A campaign summary whose 'execution' entry is not a mapping omits the execution line (branch 104->112)."""
    from scpn_control.cli import _emit_campaign_result

    _emit_campaign_result({"status": "ok", "execution": "not-a-mapping"}, json_out=False)
    out = capsys.readouterr().out
    assert "Hardware campaign completed" in out
    assert "Execution:" not in out
