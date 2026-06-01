# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — code-to-code benchmark tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Unit coverage for the TORAX code-to-code benchmark adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from validation import code_to_code_benchmark as c2c


class _FakeCoord:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _FakeSeries:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def isel(self, *, time: int) -> _FakeSeries:
        return _FakeSeries(np.asarray(self.values[time]))


class _FakeDataset:
    def __init__(
        self,
        values: dict[str, np.ndarray],
        *,
        rho: np.ndarray | None = None,
    ) -> None:
        self._values = values
        self.coords = {}
        if rho is not None:
            self.coords["rho_norm"] = _FakeCoord(rho)

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def __getitem__(self, name: str) -> _FakeSeries:
        return _FakeSeries(self._values[name])


class _FakeTreeNode:
    def __init__(self, dataset: _FakeDataset) -> None:
        self.dataset = dataset


class _FakeDataTree:
    def __init__(self, profiles: _FakeDataset, scalars: _FakeDataset) -> None:
        self._nodes = {
            "profiles": _FakeTreeNode(profiles),
            "scalars": _FakeTreeNode(scalars),
        }

    def __getitem__(self, name: str) -> _FakeTreeNode:
        return self._nodes[name.strip("/")]


def test_torax_config_maps_scenario_fields() -> None:
    cfg = c2c._torax_config_dict(c2c.ITER_SCENARIO)

    assert cfg["profile_conditions"]["Ip"] == c2c.ITER_SCENARIO["I_p"]
    assert cfg["geometry"]["R_major"] == c2c.ITER_SCENARIO["R0"]
    assert cfg["geometry"]["a_minor"] == c2c.ITER_SCENARIO["a"]
    assert cfg["geometry"]["B_0"] == c2c.ITER_SCENARIO["B0"]
    assert cfg["geometry"]["n_rho"] == c2c.ITER_SCENARIO["n_rho"]
    assert cfg["sources"]["generic_heat"]["P_total"] == c2c.ITER_SCENARIO["P_aux"] * 1.0e6


def test_repo_src_bootstrap_supports_direct_script_execution(monkeypatch) -> None:
    repo_src = str(Path(c2c.__file__).resolve().parents[1] / "src")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_src])

    c2c._ensure_repo_src_on_path()

    assert sys.path[0] == repo_src


def test_extract_torax_result_reads_profiles_and_scalars() -> None:
    rho = np.array([0.0, 0.5, 1.0])
    profiles = _FakeDataset(
        {
            "T_e": np.array([[1.0, 0.8, 0.2], [8.0, 5.0, 1.0]]),
            "T_i": np.array([[1.1, 0.9, 0.2], [7.0, 4.0, 0.8]]),
            "n_e": np.array([[1e20, 8e19, 3e19], [9e19, 7e19, 2e19]]),
        },
        rho=rho,
    )
    scalars = _FakeDataset(
        {
            "W_thermal_total": np.array([1.0e8, 2.5e8]),
            "tau_E": np.array([0.7, 1.2]),
        }
    )
    tree = _FakeDataTree(profiles, scalars)

    result = c2c._extract_torax_result(tree, c2c.ITER_SCENARIO, wall_time=0.25)

    assert result["status"] == "done"
    assert result["Te_avg"] == np.mean([8.0, 5.0, 1.0])
    assert result["Ti_avg"] == np.mean([7.0, 4.0, 0.8])
    assert result["tau_E_s"] == 1.2
    assert result["W_thermal_total_J"] == 2.5e8


def test_compare_results_includes_ion_temperature_metrics() -> None:
    scpn = {
        "rho": [0.0, 0.5, 1.0],
        "Te_final": [8.0, 5.0, 1.0],
        "Ti_final": [7.0, 4.0, 1.0],
    }
    torax = {
        "rho": [0.0, 1.0],
        "Te_final": [7.0, 1.0],
        "Ti_final": [6.0, 1.0],
        "tau_E_s": 1.1,
    }

    result = c2c._compare_results(scpn, torax)

    assert result["comparison"]["Te_rmse_keV"] > 0.0
    assert result["comparison"]["Ti_rmse_keV"] > 0.0
    assert result["comparison"]["torax_tau_E_s"] == 1.1


def test_compare_results_keeps_electron_metrics_when_torax_omits_ion_profile() -> None:
    scpn = {
        "rho": [0.0, 0.5, 1.0],
        "Te_final": [8.0, 5.0, 1.0],
        "Ti_final": [7.0, 4.0, 1.0],
    }
    torax = {
        "rho": [0.0, 1.0],
        "Te_final": [7.0, 1.0],
        "tau_E_s": 1.1,
    }

    result = c2c._compare_results(scpn, torax)

    assert result["comparison"]["Te_rmse_keV"] > 0.0
    assert "Ti_rmse_keV" not in result["comparison"]
    assert result["comparison"]["torax_tau_E_s"] == 1.1


def test_external_reference_report_blocks_when_torax_was_not_requested() -> None:
    scpn = {
        "code": "scpn-control",
        "scenario": c2c.ITER_SCENARIO["name"],
        "rho": [0.0, 1.0],
        "Te_final": [8.0, 1.0],
        "Ti_final": [7.0, 1.0],
    }

    report = c2c._build_external_reference_report(
        c2c._compare_results(scpn, None),
        c2c.ITER_SCENARIO,
        requested_torax=False,
    )

    assert report["schema_version"] == c2c.REPORT_SCHEMA_VERSION
    assert report["external_reference"]["admitted"] is False
    assert report["external_reference"]["status"] == "not_requested"
    assert report["external_reference"]["blocked_reasons"] == ["torax_not_requested"]
    assert c2c._verify_payload_digest(report) is True


def test_external_reference_report_admits_real_torax_comparison() -> None:
    scpn = {
        "code": "scpn-control",
        "scenario": c2c.ITER_SCENARIO["name"],
        "rho": [0.0, 0.5, 1.0],
        "Te_final": [8.0, 5.0, 1.0],
        "Ti_final": [7.0, 4.0, 1.0],
    }
    torax = {
        "code": "torax",
        "scenario": c2c.ITER_SCENARIO["name"],
        "status": "done",
        "rho": [0.0, 0.5, 1.0],
        "Te_final": [7.9, 5.1, 1.1],
        "Ti_final": [7.1, 4.1, 1.0],
        "tau_E_s": 1.1,
    }

    report = c2c._build_external_reference_report(
        c2c._compare_results(scpn, torax),
        c2c.ITER_SCENARIO,
        requested_torax=True,
    )

    assert report["external_reference"]["admitted"] is True
    assert report["external_reference"]["status"] == "admitted"
    assert report["external_reference"]["blocked_reasons"] == []
    assert report["external_reference"]["provider"] == "TORAX"
    assert report["scenario_sha256"] == c2c._sha256_payload(c2c.ITER_SCENARIO)
    assert c2c._verify_payload_digest(report) is True


def test_external_reference_report_rejects_digest_tampering() -> None:
    scpn = {
        "code": "scpn-control",
        "scenario": c2c.ITER_SCENARIO["name"],
        "rho": [0.0, 1.0],
        "Te_final": [8.0, 1.0],
        "Ti_final": [7.0, 1.0],
    }
    report = c2c._build_external_reference_report(
        c2c._compare_results(scpn, None),
        c2c.ITER_SCENARIO,
        requested_torax=True,
    )
    report["benchmark"]["scpn_control"]["Te_final"][0] = 12.0

    assert c2c._verify_payload_digest(report) is False
