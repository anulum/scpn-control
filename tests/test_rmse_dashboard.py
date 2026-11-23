# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Rmse Dashboard.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — RMSE Dashboard Tests
# © 1996–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for validation/rmse_dashboard.py helper functions."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

# Windows reports a directory opened as a file as a permission error.
DIRECTORY_AS_FILE_ERROR: type[OSError] = PermissionError if os.name == "nt" else IsADirectoryError

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "rmse_dashboard.py"
SPEC = importlib.util.spec_from_file_location("rmse_dashboard", MODULE_PATH)
assert SPEC and SPEC.loader
rmse_dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rmse_dashboard)


def test_ipb98_regression_point() -> None:
    """Retain the declared fixed-law reference calculation at an explicit design operating point."""
    tau = rmse_dashboard.ipb98_tau_e(
        ip_ma=15.0,
        b_t=5.3,
        n_e19=10.1,
        p_loss_mw=85.0,
        r_m=6.2,
        kappa=1.7,
        epsilon=2.0 / 6.2,
        a_eff_amu=2.5,
    )
    assert abs(tau - 3.6643409641578857) < 1e-9


def test_rmse_basic() -> None:
    """Check paired-error units and the analytical unweighted RMS definition."""
    assert rmse_dashboard.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    val = rmse_dashboard.rmse([1.0, 3.0], [2.0, 5.0])
    assert val == pytest.approx((2.5**0.5))


def test_rmse_raises_on_invalid_input() -> None:
    """Reject empty or differently sampled reference/prediction pairs."""
    with pytest.raises(ValueError):
        rmse_dashboard.rmse([], [])
    with pytest.raises(ValueError):
        rmse_dashboard.rmse([1.0], [1.0, 2.0])


def test_sparc_axis_rmse_smoke() -> None:
    """Read actual repository GEQDSK carriers through the public aggregation API."""
    sparc_dir = ROOT / "validation" / "reference_data" / "sparc"
    result = rmse_dashboard.sparc_axis_rmse(sparc_dir)
    assert result["count"] >= 1
    assert result["axis_rmse_m"] >= 0.0
    assert len(result["rows"]) == result["count"]


def test_forward_diagnostics_rmse_includes_thomson_metrics() -> None:
    """Exercise the supported forward path or its explicit unavailable contract."""
    result = rmse_dashboard.forward_diagnostics_rmse()
    if result.get("skipped", False):
        assert (
            result["count_interferometer_channels"] == 0
            and result["reason"] == "forward diagnostics module unavailable"
        )
        return
    assert result["count_interferometer_channels"] >= 1
    assert result["count_thomson_channels"] >= 1
    assert result["phase_rmse_rad"] >= 0.0
    assert result["neutron_rate_rel_error_pct"] >= 0.0
    assert result["thomson_voltage_rmse_v"] >= 0.0


def test_render_markdown_contains_sections() -> None:
    """Render the supplied dashboard carrier through its actual Markdown entry point."""
    report = {
        "generated_at_utc": "2026-02-12T00:00:00+00:00",
        "runtime_seconds": 1.23,
        "confinement_itpa": {
            "count": 2,
            "tau_rmse_s": 0.1,
            "tau_mae_rel_pct": 5.0,
            "h98_rmse": 0.2,
        },
        "confinement_iter_sparc": {
            "count": 2,
            "tau_rmse_s": 0.3,
        },
        "beta_iter_sparc": {
            "count": 2,
            "beta_n_rmse": 0.4,
        },
        "sparc_axis": {
            "count": 5,
            "axis_rmse_m": 0.01,
        },
        "forward_diagnostics": {
            "count_interferometer_channels": 3,
            "count_thomson_channels": 3,
            "phase_rmse_rad": 1e-4,
            "neutron_rate_rel_error_pct": 2.0,
            "thomson_voltage_rmse_v": 2e-3,
        },
    }
    text = rmse_dashboard.render_markdown(report)
    assert "# SCPN RMSE Dashboard" in text
    assert "Confinement RMSE (ITPA H-mode)" in text
    assert "Beta_N RMSE (ITER + SPARC references)" in text
    assert "Forward Diagnostics RMSE" in text
    assert "Supplied beta_N comparison rows do not establish empirical accuracy or facility validity." in text
    assert "without a design-fit multiplier" in text
    assert "use an actual historical burn-model run" not in text


def test_missing_burn_models_never_substitute_reference_targets() -> None:
    """Actual absent historical models must refuse prediction and yield an unavailable lane."""
    from validation.rmse_dashboard import beta_rmse_iter_sparc, estimate_beta_n_from_burn, load_json

    reference_dir = ROOT / "validation/reference_data"
    reference = load_json(reference_dir / "iter_reference.json")
    try:
        beta, metrics = estimate_beta_n_from_burn(reference, ROOT / "validation/iter_validated_config.json")
    except RuntimeError as exc:
        assert "reference values cannot replace" in str(exc)
        result = beta_rmse_iter_sparc(reference_dir, ROOT / "validation")
        assert result["skipped"] is True and result["count"] == 0
        assert result["beta_n_rmse"] is None and result["rows"] == []
    else:
        assert beta > 0 and metrics["physical_validation_claim"] is False
        assert metrics["model_provenance"]["version"] == "4.0.0"
        assert metrics["native_histories"]


@pytest.mark.parametrize("explicit_plots", [False, True])
def test_real_dashboard_cli_exports_unavailable_beta_and_owned_plots(tmp_path: Path, explicit_plots: bool) -> None:
    """Run the actual report CLI and renderer on tracked references with all outputs in the selected scope."""
    import json
    import os
    import subprocess
    import sys

    from validation.validate_real_shots import validate_disruption

    report = tmp_path / "rmse_dashboard_ci.json"
    markdown = tmp_path / "report.md"
    args = [sys.executable, str(MODULE_PATH), "--output-json", str(report), "--output-md", str(markdown)]
    plot_dir = tmp_path / "figures" if explicit_plots else tmp_path
    if explicit_plots:
        args.extend(["--plot-dir", str(plot_dir)])
    result = subprocess.run(
        args,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    decoded = json.loads(report.read_text())
    assert decoded["schema"] == "scpn-control.rmse-dashboard.v2"
    beta = decoded["beta_iter_sparc"]
    text = markdown.read_text()
    if beta.get("skipped", False):
        assert beta["beta_n_rmse"] is None and beta["count"] == 0
        assert "beta_N RMSE=unavailable" in result.stdout
        assert "No beta_N model prediction was available" in text
        assert not (plot_dir / "beta_n_scatter.png").exists()
    else:
        assert beta["count"] == 2 and beta["design_rmse_is_binding"] is False
        assert beta["physical_validation_claim"] is False
        assert all(row["model_evidence"]["native_histories"] for row in beta["rows"])
        assert "indicative target" in text
    disruption = validate_disruption(ROOT / "validation/reference_data/diiid/disruption_shots")
    (tmp_path / "reference_evidence_validation.json").write_text(
        json.dumps(
            {"schema": "scpn-control.reference-evidence-validation.v1", "lanes": {"disruption_synthetic": disruption}}
        )
    )
    gate = subprocess.run(
        [sys.executable, str(ROOT / "tools/ci_rmse_gate.py"), "--artifact-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if beta.get("skipped", False):
        assert gate.returncode == 1 and "lane must be an available object" in gate.stdout
    else:
        assert "beta_iter_sparc" in gate.stdout
        assert gate.returncode == 0, gate.stdout + gate.stderr
    if (plot_dir / "tau_e_scatter.png").exists():
        prefix = "figures" if explicit_plots else "."
        assert f"{prefix}/tau_e_scatter.png" in text


def test_reference_reader_refuses_nonobject_root(tmp_path: Path) -> None:
    """Wrap an actual reference object in an invalid root array and refuse it at the public reader."""
    import json

    from validation.rmse_dashboard import load_json

    original = load_json(ROOT / "validation/reference_data/iter_reference.json")
    changed = tmp_path / "iter_reference.json"
    changed.write_text(json.dumps([original]))
    with pytest.raises(ValueError, match="reference JSON must be an object"):
        load_json(changed)


def test_native_rmse_example_executes() -> None:
    """Execute the numerical definition example rendered by the owning module's native docs."""
    import doctest

    from validation import rmse_dashboard as module

    result = doctest.testmod(module, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 1


@pytest.mark.parametrize("missing", [False, True])
def test_confinement_reader_marks_actual_empty_or_missing_input_unavailable(tmp_path: Path, missing: bool) -> None:
    """A real missing file or header-only copy yields no invented confinement samples."""
    from validation.rmse_dashboard import confinement_rmse_itpa

    path = tmp_path / "confinement.csv"
    if not missing:
        original = ROOT / "validation/reference_data/itpa/hmode_confinement.csv"
        path.write_text(original.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    result = confinement_rmse_itpa(path)
    assert result["skipped"] is True and result["count"] == 0 and result["rows"] == []
    assert "missing input file" in result["reason"] if missing else "no rows" in result["reason"]


@pytest.mark.parametrize("contents,error", [(b"\xff", UnicodeError), (b"{", ValueError)])
def test_reference_reader_propagates_actual_decode_errors(
    tmp_path: Path, contents: bytes, error: type[Exception]
) -> None:
    """Malformed physical reference bytes propagate through the public UTF-8 JSON reader."""
    from validation.rmse_dashboard import load_json

    path = tmp_path / "invalid.json"
    path.write_bytes(contents)
    with pytest.raises(error):
        load_json(path)
    assert path.read_bytes() == contents


def test_axis_aggregation_refuses_an_actual_empty_reference_directory(tmp_path: Path) -> None:
    """A directory with no GEQDSK/EQDSK records cannot produce a zero-error comparison."""
    from validation.rmse_dashboard import sparc_axis_rmse

    with pytest.raises(ValueError, match="non-empty lists"):
        sparc_axis_rmse(tmp_path)


def test_public_argument_parser_preserves_destinations_without_writes(tmp_path: Path) -> None:
    """The exposed parser retains caller paths, rejects unknown flags and creates no outputs."""
    from validation.rmse_dashboard import ROOT as owner_root
    from validation.rmse_dashboard import parse_args

    defaults = parse_args([])
    assert defaults.output_json == str(owner_root / "validation/reports/rmse_dashboard.json")
    assert defaults.plot_dir is None
    args = parse_args(["--output-json", "chosen.json", "--output-md", "chosen.md", "--plot-dir", str(tmp_path)])
    assert args.output_json == "chosen.json" and args.output_md == "chosen.md" and args.plot_dir == tmp_path
    with pytest.raises(SystemExit) as error:
        parse_args(["--unrecognized-dashboard-flag"])
    assert error.value.code == 2 and list(tmp_path.iterdir()) == []


def test_command_owner_native_destination_example_executes() -> None:
    """Execute the owning command's native example with caller-selected output and plot paths."""
    import doctest

    from validation import rmse_dashboard_command

    result = doctest.testmod(rmse_dashboard_command, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 2


@pytest.mark.parametrize(
    "lane,filename,field",
    [
        ("confinement_itpa", "tau_e_scatter.png", "tau_measured_s"),
        ("sparc_axis", "sparc_axis_error.png", "axis_error_m"),
    ],
)
@pytest.mark.parametrize("failure", ["output", "row"])
def test_renderer_closes_real_figure_after_output_refusal(
    tmp_path: Path,
    failure: str,
    lane: str,
    filename: str,
    field: str,
) -> None:
    """Actual confinement/axis plot failures preserve the pre-existing figure set."""
    import matplotlib.pyplot as plt

    from validation.rmse_dashboard import (
        beta_rmse_iter_sparc,
        confinement_rmse_itpa,
        render_plots,
        sparc_axis_rmse,
    )

    reference_dir = ROOT / "validation/reference_data"
    report = {
        "confinement_itpa": confinement_rmse_itpa(reference_dir / "itpa/hmode_confinement.csv"),
        "beta_iter_sparc": beta_rmse_iter_sparc(reference_dir, ROOT / "validation"),
        "sparc_axis": sparc_axis_rmse(reference_dir / "sparc"),
    }
    expected: type[Exception]
    if failure == "output":
        (tmp_path / filename).mkdir()
        expected = DIRECTORY_AS_FILE_ERROR
    else:
        del report[lane]["rows"][0][field]
        expected = KeyError
    caller_figure = plt.figure()
    try:
        before = plt.get_fignums()
        with pytest.raises(expected):
            render_plots(report, tmp_path)
        assert plt.get_fignums() == before
    finally:
        plt.close(caller_figure)


def _fresh_module(name: str, path: Path) -> Any:
    """Execute a module file under a private name, so its import-time branches run again."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_lane_runs_the_fusion_forward_model_with_the_documented_bias() -> None:
    """The lane runs FUSION's forward model; phase scales with the 0.985 density bias, the rate with 1.03."""
    from validation import rmse_dashboard_metrics as metrics

    try:
        forward = importlib.import_module(metrics.FORWARD_MODEL_MODULE)
    except ImportError:
        assert metrics._forward_model() is None
        return
    assert metrics._forward_model() is forward.generate_forward_channels
    result = metrics.forward_diagnostics_rmse()
    assert "skipped" not in result
    assert result["count_interferometer_channels"] == 3 and result["count_thomson_channels"] == 3
    phase_true = [row["phase_true_rad"] for row in result["rows"]]
    assert all(phase > 0.0 for phase in phase_true)
    for row in result["rows"]:
        assert row["phase_pred_rad"] == pytest.approx(0.985 * row["phase_true_rad"], rel=1e-12)
    rms_true = math.sqrt(sum(phase * phase for phase in phase_true) / len(phase_true))
    assert result["phase_rmse_rad"] == pytest.approx(0.015 * rms_true, rel=1e-9)
    assert result["neutron_rate_rel_error_pct"] == pytest.approx(3.0, rel=1e-9)
    assert result["thomson_voltage_rmse_v"] > 0.0


def test_forward_lane_without_the_fusion_model_reports_null_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the forward model the lane is skipped with its reason and null errors, never zero errors."""
    from validation import rmse_dashboard_metrics as metrics

    monkeypatch.setitem(sys.modules, metrics.FORWARD_MODEL_MODULE, None)
    assert metrics._forward_model() is None
    assert metrics.forward_diagnostics_rmse() == {
        "count_interferometer_channels": 0,
        "phase_rmse_rad": None,
        "neutron_rate_rel_error_pct": None,
        "skipped": True,
        "reason": "forward diagnostics module unavailable",
    }


def test_beta_lane_reports_a_provider_refusal_without_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burn-provider refusal yields the skipped lane with the refusal text and no rows."""
    from validation import rmse_dashboard_metrics as metrics

    def refuse(reference: dict[str, Any], config_path: Path) -> tuple[float, dict[str, Any]]:
        """Refuse the run as the provider does when it cannot produce histories."""
        raise RuntimeError("burn provider refused the run")

    monkeypatch.setattr(metrics, "estimate_beta_n_from_burn", refuse)
    assert metrics.beta_rmse_iter_sparc(ROOT / "validation" / "reference_data", ROOT / "validation") == {
        "count": 0,
        "beta_n_rmse": None,
        "rows": [],
        "skipped": True,
        "reason": "burn provider refused the run",
    }


def test_axis_comparison_follows_the_flux_sign_convention() -> None:
    """Negating psi and its axis and boundary values moves the search to the other extremum, at the same point."""
    from scpn_control.core.eqdsk import read_geqdsk
    from validation import rmse_dashboard_metrics as metrics

    eq = read_geqdsk(ROOT / "validation" / "reference_data" / "sparc" / "lmode_hv.geqdsk")
    flipped = replace(eq, psirz=-eq.psirz, simag=-eq.simag, sibry=-eq.sibry)
    assert (eq.simag < eq.sibry) is not (flipped.simag < flipped.sibry)
    assert metrics.compare_eq_axis(flipped) == metrics.compare_eq_axis(eq)


def test_point_wise_psi_lane_is_reported_unavailable_without_its_module() -> None:
    """Without a point-wise psi module in the repository, the lane reports itself unavailable."""
    from validation import rmse_dashboard_command as command

    assert importlib.util.find_spec("validation.psi_pointwise_rmse") is None
    assert command._sparc_psi_lane(ROOT / "validation" / "reference_data" / "sparc") == {
        "skipped": True,
        "reason": command.PSI_LANE_UNAVAILABLE,
    }


def test_point_wise_psi_lane_returns_the_module_report_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the module importable, the lane returns its report for the given directory unchanged."""
    from validation import rmse_dashboard_command as command

    calls: list[Path] = []
    lane = {"count": 1, "mean_psi_rmse_norm": 0.01}

    def sparc_psi_rmse(directory: Path) -> dict[str, Any]:
        """Record the directory and return the fixed lane report."""
        calls.append(directory)
        return lane

    module = types.ModuleType("validation.psi_pointwise_rmse")
    module.__dict__["sparc_psi_rmse"] = sparc_psi_rmse
    monkeypatch.setitem(sys.modules, "validation.psi_pointwise_rmse", module)
    assert command._sparc_psi_lane(tmp_path) is lane
    assert calls == [tmp_path]


def test_command_without_matplotlib_writes_both_reports_without_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without Matplotlib the command writes JSON and Markdown, no figures and no figure links."""
    from validation import rmse_dashboard_command as command
    from validation import rmse_dashboard_metrics as metrics

    with monkeypatch.context() as blocked:
        blocked.setitem(sys.modules, "matplotlib", None)
        rendering = _fresh_module(
            "rmse_dashboard_rendering_without_matplotlib", ROOT / "validation" / "rmse_dashboard_rendering.py"
        )
    assert rendering._HAS_MPL is False
    monkeypatch.setattr(command, "render_plots", rendering.render_plots)
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    assert command.main(["--output-json", str(json_out), "--output-md", str(markdown_out)]) == 0
    printed = capsys.readouterr().out
    assert "SCPN RMSE dashboard generated." in printed and "Plots:" not in printed
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json", "report.md"]
    report = json.loads(json_out.read_text(encoding="utf-8"))
    text = markdown_out.read_text(encoding="utf-8")
    assert "![" not in text
    assert report["sparc_psi_rmse"] == {"skipped": True, "reason": command.PSI_LANE_UNAVAILABLE}
    assert f"- Unavailable: {command.PSI_LANE_UNAVAILABLE}" in text
    assert report["forward_diagnostics"].get("skipped", False) is (metrics._forward_model() is None)
