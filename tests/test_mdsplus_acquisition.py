# SPDX-License-Identifier: AGPL-3.0-or-later
# ----------------------------------------------------------------------
# SCPN Control - MDSplus Acquisition Tests
# Copyright (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ----------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_control.core.mdsplus_acquisition import (
    MDSplusSignalSpec,
    acquire_mdsplus_shot,
    load_mdsplus_acquisition_request,
)
from scpn_control.core.real_data_manifest import load_real_data_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeMDSplusModule:
    def __init__(self) -> None:
        self.opens: list[tuple[str, int]] = []

    def Tree(self, tree: str, shot: int):
        self.opens.append((tree, shot))
        return FakeTree()


class FakeTree:
    def getNode(self, node: str):
        data = {
            "\\\\IP": np.array([1.0, 1.2, 1.1], dtype=np.float64),
            "\\\\BETAN": np.array([2.0, 2.1, 2.2], dtype=np.float64),
        }[node]
        return SimpleNamespace(data=lambda: data)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_acquire_mdsplus_shot_writes_npz_and_validated_manifest(tmp_path: Path) -> None:
    fake = FakeMDSplusModule()
    output_npz = tmp_path / "shot_163303.npz"
    manifest_json = tmp_path / "shot_163303.manifest.json"

    result = acquire_mdsplus_shot(
        tree="DIII-D",
        shot=163303,
        signals=[
            MDSplusSignalSpec(name="plasma_current", node="\\\\IP", units="A", timebase="time_s"),
            MDSplusSignalSpec(name="normalised_beta", node="\\\\BETAN", units="1", timebase="time_s"),
        ],
        output_npz=output_npz,
        manifest_json=manifest_json,
        mdsplus_module=fake,
        source_uri="mdsplus://DIII-D/163303",
        access_policy="facility-approved",
        licence="facility data policy",
        retrieved_at="2026-05-18T01:30:00Z",
    )

    assert fake.opens == [("DIII-D", 163303)]
    assert result.output_npz == output_npz
    assert result.manifest_json == manifest_json

    data = np.load(output_npz)
    assert sorted(data.files) == ["normalised_beta", "plasma_current"]
    np.testing.assert_allclose(data["plasma_current"], [1.0, 1.2, 1.1])

    manifest = load_real_data_manifest(manifest_json, verify_artifact=True)
    assert manifest.kind == "real"
    assert manifest.source.kind == "mdsplus"
    assert manifest.dataset_id == "diii-d-163303-mdsplus"
    assert len(manifest.signals) == 2


def test_acquire_mdsplus_rejects_empty_signal_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one signal"):
        acquire_mdsplus_shot(
            tree="DIII-D",
            shot=163303,
            signals=[],
            output_npz=tmp_path / "shot.npz",
            manifest_json=tmp_path / "shot.manifest.json",
            mdsplus_module=FakeMDSplusModule(),
            source_uri="mdsplus://DIII-D/163303",
            access_policy="facility-approved",
            licence="facility data policy",
            retrieved_at="2026-05-18T01:30:00Z",
        )


def test_load_mdsplus_acquisition_request_from_json(tmp_path: Path) -> None:
    spec_path = tmp_path / "shot_163303_mdsplus.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tree": "DIII-D",
                "shot": 163303,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": [
                    {"name": "plasma_current", "node": "\\\\IP", "units": "A", "timebase": "time_s"},
                    {"name": "normalised_beta", "node": "\\\\BETAN", "units": "1", "timebase": "time_s"},
                ],
            }
        ),
        encoding="utf-8",
    )

    request = load_mdsplus_acquisition_request(spec_path)

    assert request.tree == "DIII-D"
    assert request.shot == 163303
    assert request.signals[0] == MDSplusSignalSpec(
        name="plasma_current",
        node="\\\\IP",
        units="A",
        timebase="time_s",
    )


def test_repository_mdsplus_acquisition_spec_loads() -> None:
    request = load_mdsplus_acquisition_request(
        REPO_ROOT / "validation/reference_data/diiid/acquisition_specs/shot_163303_mdsplus.json"
    )

    assert request.tree == "DIII-D"
    assert request.shot == 163303
    assert [signal.name for signal in request.signals] == ["plasma_current", "normalised_beta"]


def test_cli_acquire_mdsplus_reports_missing_optional_dependency(runner, tmp_path: Path) -> None:
    from scpn_control.cli import main

    signal = json.dumps({"name": "plasma_current", "node": "\\\\IP", "units": "A", "timebase": "time_s"})
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
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "shot.manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "MDSplus" in payload["error"]


def test_cli_acquire_mdsplus_accepts_spec_json(runner, tmp_path: Path) -> None:
    from scpn_control.cli import main

    spec_path = tmp_path / "shot_163303_mdsplus.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tree": "DIII-D",
                "shot": 163303,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": [{"name": "plasma_current", "node": "\\\\IP", "units": "A", "timebase": "time_s"}],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--spec-json",
            str(spec_path),
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "shot.manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "MDSplus" in payload["error"]
