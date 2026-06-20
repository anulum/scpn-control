# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MDSplus Acquisition Tests

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from click.testing import CliRunner

from scpn_control.core.mdsplus_acquisition import (
    MDSplusSignalSpec,
    _validate_signal_specs,
    acquire_mdsplus_shot,
    load_mdsplus_acquisition_request,
)
import scpn_control.core.mdsplus_acquisition as mdsplus_acquisition
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


def test_load_mdsplus_acquisition_request_rejects_duplicate_keys(tmp_path: Path) -> None:
    spec_path = tmp_path / "duplicate_mdsplus.json"
    spec_path.write_text(
        '{"schema_version":"1.0","tree":"DIII-D","tree":"NSTX-U"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: tree"):
        load_mdsplus_acquisition_request(spec_path)


def test_repository_mdsplus_acquisition_spec_loads() -> None:
    request = load_mdsplus_acquisition_request(
        REPO_ROOT / "validation/reference_data/diiid/acquisition_specs/shot_163303_mdsplus.json"
    )

    assert request.tree == "DIII-D"
    assert request.shot == 163303
    assert [signal.name for signal in request.signals] == ["plasma_current", "normalised_beta"]


def _force_mdsplus_client_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make both MDSplus client imports fail without disturbing other imports.

    The local environment may have the ``mdsthin`` compatibility client
    installed (per the install-all-optional-deps policy), in which case
    ``_import_mdsplus`` succeeds and acquisition proceeds to a live-connection
    attempt. Forcing both client imports to fail exercises the documented
    missing-dependency contract deterministically while still driving the real
    ``_import_mdsplus`` fallback chain.
    """
    real_import = mdsplus_acquisition.importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"MDSplus", "mdsthin.MDSplus"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(mdsplus_acquisition.importlib, "import_module", fake_import)


def test_cli_acquire_mdsplus_reports_missing_optional_dependency(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scpn_control.cli import main

    _force_mdsplus_client_absent(monkeypatch)
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


def _valid_signal() -> dict[str, str]:
    return {"name": "plasma_current", "node": "\\\\IP", "units": "A", "timebase": "time_s"}


def _base_request() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "tree": "DIII-D",
        "shot": 163303,
        "source_uri": "mdsplus://DIII-D/163303",
        "access_policy": "facility-approved",
        "licence": "facility data policy",
        "signals": [_valid_signal()],
    }


@pytest.mark.parametrize(
    ("payload_text", "match"),
    [
        ("[]", "root must be a JSON object"),
        (json.dumps({"schema_version": "2.0"}), "schema_version must be '1.0'"),
        (json.dumps({"schema_version": "1.0", "signals": {}}), "requires a signals array"),
        (json.dumps({"schema_version": "1.0", "signals": []}), "at least one signal"),
    ],
)
def test_load_request_rejects_malformed_envelope(tmp_path: Path, payload_text: str, match: str) -> None:
    spec_path = tmp_path / "request.json"
    spec_path.write_text(payload_text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_mdsplus_acquisition_request(spec_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ({"signals": ["not-an-object"]}, "signal specification must be a JSON object"),
        ({"tree": "   "}, "requires non-empty tree"),
        ({"shot": "163303"}, "shot must be an integer"),
        ({"shot": True}, "shot must be an integer"),
    ],
)
def test_load_request_rejects_malformed_fields(tmp_path: Path, mutate: dict[str, Any], match: str) -> None:
    payload = _base_request()
    payload.update(mutate)
    spec_path = tmp_path / "request.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_mdsplus_acquisition_request(spec_path)


@pytest.mark.parametrize(
    ("signals", "match"),
    [
        ([MDSplusSignalSpec(name="  ", node="\\\\IP", units="A", timebase="t")], "name must not be empty"),
        (
            [
                MDSplusSignalSpec(name="ip", node="\\\\IP", units="A", timebase="t"),
                MDSplusSignalSpec(name="ip", node="\\\\IP2", units="A", timebase="t"),
            ],
            "duplicate MDSplus signal name: ip",
        ),
        ([MDSplusSignalSpec(name="ip", node="  ", units="A", timebase="t")], "requires a node path"),
        ([MDSplusSignalSpec(name="ip", node="\\\\IP", units="  ", timebase="t")], "requires units"),
        ([MDSplusSignalSpec(name="ip", node="\\\\IP", units="A", timebase="  ")], "requires a timebase"),
    ],
)
def test_validate_signal_specs_rejects_invalid_specs(signals: list[MDSplusSignalSpec], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_signal_specs(signals)


class _SingleArrayModule:
    """Fake MDSplus module whose every node returns a fixed array."""

    def __init__(self, array: npt.NDArray[np.float64]) -> None:
        self._array = array

    def Tree(self, tree: str, shot: int) -> Any:
        array = self._array

        class _Tree:
            def getNode(self, node: str) -> Any:
                return SimpleNamespace(data=lambda: array)

        return _Tree()


@pytest.mark.parametrize(
    ("array", "match"),
    [
        (np.array([], dtype=np.float64), "returned an empty array"),
        (np.array([1.0, np.nan], dtype=np.float64), "contains non-finite values"),
    ],
)
def test_acquire_rejects_degenerate_signal_arrays(tmp_path: Path, array: npt.NDArray[np.float64], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        acquire_mdsplus_shot(
            tree="DIII-D",
            shot=163303,
            signals=[MDSplusSignalSpec(name="plasma_current", node="\\\\IP", units="A", timebase="time_s")],
            output_npz=tmp_path / "shot.npz",
            manifest_json=tmp_path / "shot.manifest.json",
            mdsplus_module=_SingleArrayModule(array),
            source_uri="mdsplus://DIII-D/163303",
            access_policy="facility-approved",
            licence="facility data policy",
            retrieved_at="2026-05-18T01:30:00Z",
        )


def test_import_mdsplus_falls_back_to_mdsthin_compat(monkeypatch) -> None:
    compat = SimpleNamespace(Tree=object)

    def fake_import_module(name: str) -> Any:
        if name == "MDSplus":
            raise ImportError("native client absent")
        if name == "mdsthin.MDSplus":
            return compat
        raise AssertionError(name)

    monkeypatch.setattr(mdsplus_acquisition.importlib, "import_module", fake_import_module)

    assert mdsplus_acquisition._import_mdsplus() is compat


def test_cli_acquire_mdsplus_accepts_spec_json(runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scpn_control.cli import main

    _force_mdsplus_client_absent(monkeypatch)
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
