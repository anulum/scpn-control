# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Code-to-code TORAX path against an installed provider.
"""The optional TORAX path of the code-to-code command against an installed TORAX 1.4.

These tests run the actual provider: TORAX builds the written configuration,
runs the simulation and returns its output tree. Without TORAX they skip; the
absent-provider path is observed by ``tests/test_code_to_code_commands.py``.
The temporary configuration path is relative to the working directory, so
every test runs in its own directory. TORAX 1.4 imports jaxopt, which warns
on import that it is no longer maintained; the repository turns every
DeprecationWarning into an error, so that one third-party warning is ignored
here and nowhere else.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from validation import code_to_code_benchmark as benchmark
from validation import code_to_code_torax as torax_adapter
from validation.code_to_code_scenario import ITER_SCENARIO

pytestmark = pytest.mark.filterwarnings("ignore:JAXopt is no longer maintained:DeprecationWarning")


@pytest.fixture
def torax_installed() -> None:
    """Skip unless an installed TORAX can be imported."""
    if importlib.util.find_spec("torax") is None:
        pytest.skip("TORAX 1.4 must be installed to run the provider path")


def test_real_torax_run_returns_its_final_profiles(
    torax_installed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TORAX runs the scenario; the result holds finite final profiles, scalars and no leftover config."""
    monkeypatch.chdir(tmp_path)
    result = torax_adapter._run_torax(ITER_SCENARIO)
    assert result is not None and result["code"] == "torax" and result["status"] == "done"
    samples = len(result["rho"])
    assert samples > 1 and len(result["Te_final"]) == len(result["Ti_final"]) == len(result["ne_final"]) == samples
    for name in ("Te_final", "Ti_final", "ne_final"):
        assert all(math.isfinite(value) for value in result[name])
    assert result["Te_avg"] == pytest.approx(sum(result["Te_final"]) / samples)
    assert result["tau_E_s"] > 0.0 and result["W_thermal_total_J"] > 0.0 and result["wall_time_s"] > 0.0
    assert not (tmp_path / torax_adapter.TORAX_TMP_CONFIG).exists()


def test_existing_temporary_config_is_refused_and_kept(
    torax_installed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configuration already at the temporary path is never overwritten or removed."""
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / torax_adapter.TORAX_TMP_CONFIG
    existing.parent.mkdir(parents=True)
    existing.write_text("CONFIG = {}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        torax_adapter._run_torax(ITER_SCENARIO)
    assert existing.read_text(encoding="utf-8") == "CONFIG = {}\n"


def test_command_with_torax_reports_a_blocked_diagnostic_comparison(
    torax_installed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With TORAX the command compares both codes; admission stays blocked by the unmatched physics."""
    monkeypatch.chdir(tmp_path)
    json_out = tmp_path / "out" / "c2c.json"
    markdown_out = tmp_path / "out" / "c2c.md"
    argv = ["--with-torax", "--json-out", str(json_out), "--markdown-out", str(markdown_out)]
    assert benchmark.main(argv) == 0
    printed = capsys.readouterr().out
    assert "Status: done" in printed and "Comparison metrics:" in printed
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["benchmark"]["torax"]["status"] == "done"
    assert sorted(report["benchmark"]["comparison"]) == [
        "Te_max_diff_keV",
        "Te_rmse_keV",
        "Ti_max_diff_keV",
        "Ti_rmse_keV",
        "torax_tau_E_s",
    ]
    external = report["external_reference"]
    assert external["status"] == "blocked" and external["admitted"] is False
    assert external["diagnostic_comparison_available"] is True
    assert external["blocked_reasons"] == [
        "transport_closures_not_matched",
        "current_geometry_not_matched",
        "source_channels_not_matched",
    ]
    assert benchmark.main([*argv, "--require-external"]) == 1
