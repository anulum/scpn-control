# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bounded claim-admission benchmark producers.
"""The bounded claim producers write their report pair and keep every claim closed.

Each producer runs in process with its output paths moved into a temporary
directory, which the recorded-campaign guard treats as temporary output. The
committed reports under ``validation/reports`` are neither read nor written.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_control.benchmark_records import CAMPAIGN_ENV


def _closed(path: tuple[str, ...]) -> Callable[[dict[str, Any]], bool]:
    """Return a check that the nested boolean at ``path`` is False."""

    def check(payload: dict[str, Any]) -> bool:
        """Follow ``path`` and require a literal False."""
        value: Any = payload
        for key in path:
            value = value[key]
        return value is False

    return check


PRODUCERS: dict[str, tuple[str, str, Callable[[dict[str, Any]], bool]]] = {
    "benchmark_neural_transport_claims": (
        "MD_REPORT",
        "# Neural transport claim-admission benchmark",
        _closed(("claim_evidence", "quantitative_claim_allowed")),
    ),
    "benchmark_neural_turbulence_claims": (
        "MD_REPORT",
        "# Neural turbulence claim-admission benchmark",
        _closed(("claim_evidence", "quantitative_claim_allowed")),
    ),
    "benchmark_disruption_mitigation_claims": (
        "MD_REPORT",
        "# Disruption mitigation claim-admission benchmark",
        _closed(("claim_evidence", "mitigation_claim_allowed")),
    ),
    "benchmark_rwm_claims": ("MD_REPORT", "# RWM Claim-Admission Benchmark", _closed(("facility_claim_allowed",))),
    "benchmark_static_mu_analysis_claims": (
        "MARKDOWN_REPORT",
        "# Static Structured-Mu Claim-Admission Benchmark",
        _closed(("validated_claim_allowed",)),
    ),
    "benchmark_efit_lite_claims": (
        "MD_REPORT",
        "# EFIT-lite Claim-Admission Benchmark",
        _closed(("facility_claim_allowed",)),
    ),
    "benchmark_federated_disruption": (
        "MD_REPORT",
        "# Federated disruption benchmark",
        lambda payload: "no measured cross-facility validation" in payload["claim_boundary"],
    ),
    "benchmark_neural_equilibrium_pretraining": (
        "MD_REPORT",
        "# Neural equilibrium synthetic pretraining benchmark",
        _closed(("claim_evidence", "facility_claim_allowed")),
    ),
    "benchmark_integrated_scenario_coupling": (
        "MD_REPORT",
        "# Integrated Scenario Coupling Benchmark",
        lambda payload: payload["claim_boundary"].startswith("Bounded deterministic replay audit only."),
    ),
    "benchmark_marfe_onset_claims": (
        "MARKDOWN_REPORT",
        "# MARFE Onset Claim-Admission Benchmark",
        _closed(("production_claim_allowed",)),
    ),
    "benchmark_vmec_lite_claims": (
        "MARKDOWN_REPORT",
        "# VMEC-lite Claim-Admission Benchmark",
        _closed(("production_claim_allowed",)),
    ),
    "benchmark_gk_online_learner": (
        "MD_REPORT",
        "# GK online learner benchmark",
        lambda payload: payload["claim_boundary"].startswith("Synthetic online learner evidence only."),
    ),
    "benchmark_digital_twin_online_update": (
        "MD_REPORT",
        "# Digital twin online update benchmark",
        lambda payload: payload["claim_boundary"].startswith("Bounded synthetic online-update benchmark only."),
    ),
}
SCRIPT_PATH_PREFIX: dict[str, tuple[str, ...]] = {
    "benchmark_marfe_onset_claims": ("src", "."),
    "benchmark_vmec_lite_claims": ("src", "."),
    "benchmark_digital_twin_online_update": ("src",),
}


def _redirect(module: Any, markdown_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Move the producer's report directory, report pair and any weight or decision file into ``tmp_path``."""
    out = tmp_path / "reports"
    json_path = out / Path(module.JSON_REPORT).name
    markdown_path = out / Path(getattr(module, markdown_name)).name
    monkeypatch.setattr(module, "REPORT_DIR", out)
    monkeypatch.setattr(module, "JSON_REPORT", json_path)
    monkeypatch.setattr(module, markdown_name, markdown_path)
    for extra in ("WEIGHTS_PATH", "DECISION_REPORT"):
        if hasattr(module, extra):
            monkeypatch.setattr(module, extra, out / Path(getattr(module, extra)).name)
    return json_path, markdown_path


@pytest.mark.parametrize("name", sorted(PRODUCERS))
def test_producer_writes_its_pair_with_the_claim_closed(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSON parses, keeps the claim closed, and the Markdown opens with the producer's title."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    markdown_name, title, claim_closed = PRODUCERS[name]
    module = importlib.import_module(f"validation.{name}")
    json_path, markdown_path = _redirect(module, markdown_name, tmp_path, monkeypatch)

    module.main()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert claim_closed(payload)
    assert title in markdown_path.read_text(encoding="utf-8").splitlines()[:3]


def test_committed_report_paths_need_a_recorded_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a campaign identifier a producer refuses its persistent paths before computing anything."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    module = importlib.import_module("validation.benchmark_rwm_claims")
    before = module.JSON_REPORT.read_bytes()
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        module.main()
    assert module.JSON_REPORT.read_bytes() == before


@pytest.mark.parametrize("name", sorted(PRODUCERS))
def test_script_entry_refuses_persistent_output_without_a_campaign(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executed as a script, each producer stops at the recorded-campaign guard and leaves its report untouched."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    module = importlib.import_module(f"validation.{name}")
    before = Path(module.JSON_REPORT).read_bytes()
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(module.__file__), run_name="__main__")
    assert Path(module.JSON_REPORT).read_bytes() == before


def test_disruption_report_serialiser_unwraps_numpy_scalars_only() -> None:
    """NumPy scalars become Python numbers; any other object is refused with the JSON-style error."""
    module = importlib.import_module("validation.benchmark_disruption_mitigation_claims")
    assert module._json_default(np.float32(1.5)) == 1.5
    with pytest.raises(TypeError, match="Object of type object is not JSON serialisable"):
        module._json_default(object())


def test_vmec_producer_builds_geometry_evidence_when_the_committed_report_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a geometry report the producer validates the geometry itself and binds that payload."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    module = importlib.import_module("validation.benchmark_vmec_lite_claims")
    json_path, _ = _redirect(module, "MARKDOWN_REPORT", tmp_path, monkeypatch)
    monkeypatch.setattr(module, "GEOMETRY_REPORT", tmp_path / "absent_geometry.json")
    module.main()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["geometry_validation_passed"] is True
    assert len(payload["geometry_validation_payload_sha256"]) == 64


@pytest.mark.parametrize("name", sorted(SCRIPT_PATH_PREFIX))
def test_package_import_leaves_the_import_path_unchanged(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Imported as ``validation.<module>``, a producer does not insert the checkout or its sources."""
    module = importlib.import_module(f"validation.{name}")
    root = Path(str(module.__file__)).resolve().parents[1]
    stripped = [entry for entry in sys.path if Path(entry or ".").resolve() not in {root, root / "src"}]
    monkeypatch.setattr(sys, "path", list(stripped))
    importlib.reload(module)
    assert sys.path == stripped


def test_marfe_percentile_interpolates_between_ranks() -> None:
    """The latency percentile interpolates linearly between ordered samples."""
    module = importlib.import_module("validation.benchmark_marfe_onset_claims")
    assert module._percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert module._percentile([5.0], 0.99) == 5.0


def test_failed_coupling_audit_stops_after_writing_its_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed audit still writes the report pair, then exits with the joined violations."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    module = importlib.import_module("validation.benchmark_integrated_scenario_coupling")
    json_path, markdown_path = _redirect(module, "MD_REPORT", tmp_path, monkeypatch)
    real_audit = module.audit_scenario_coupling

    def failing_audit(*args: Any, **kwargs: Any) -> Any:
        """Return the real audit marked as failed with two violations."""
        return dataclasses.replace(real_audit(*args, **kwargs), passed=False, violations=("first", "second"))

    monkeypatch.setattr(module, "audit_scenario_coupling", failing_audit)
    with pytest.raises(SystemExit, match="^first; second$"):
        module.main()
    assert json_path.exists() and "- Passed: `False`" in markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(SCRIPT_PATH_PREFIX))
def test_script_run_puts_the_checkout_and_sources_first(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a script from a path without the checkout, the producer inserts its sources (and the checkout)."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    module = importlib.import_module(f"validation.{name}")
    script = Path(str(module.__file__))
    root = script.resolve().parents[1]
    stripped = [entry for entry in sys.path if Path(entry or ".").resolve() not in {root, root / "src"}]
    monkeypatch.setattr(sys, "path", stripped)
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(script), run_name="__main__")
    expected = [str(root / part) for part in SCRIPT_PATH_PREFIX[name]]
    assert sys.path[: len(expected)] == expected


def test_online_learner_campaign_without_a_decision_writes_no_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the learner records no retrain decision, the producer refuses before writing its summary pair."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    module = importlib.import_module("validation.benchmark_gk_online_learner")
    json_path, markdown_path = _redirect(module, "MD_REPORT", tmp_path, monkeypatch)
    monkeypatch.setattr(module.OnlineLearner, "latest_decision", lambda self: None)
    with pytest.raises(RuntimeError, match="did not produce a retrain decision"):
        module.main()
    assert not json_path.exists() and not markdown_path.exists()
