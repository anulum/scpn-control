# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Import path, report building and command of the nonlinear Cyclone validator.
"""The nonlinear Cyclone validator edits the import path only as a script and reports what it ran.

Imported as ``validation.gk_nonlinear_cyclone`` it must leave the caller's
import path alone; run as a script from a path without the checkout's
sources it must put them first so ``scpn_control`` resolves. Both checks
strip the sources first, because with them present an "insert if absent"
would also leave the path unchanged. The report checks cover NumPy-free
serialisation, digest refusal, every saturation-evidence reason and every
blocked-claim reason; the command test runs the four real diagnostics once.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import pytest

from validation import gk_nonlinear_cyclone as cyclone

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT = ROOT / "validation" / "gk_nonlinear_cyclone.py"


def _without_sources() -> list[str]:
    """Return the current import path without the checkout's sources."""
    return [entry for entry in sys.path if Path(entry or ".").resolve() != SRC]


def test_package_import_leaves_the_import_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reloading the validator as a package module inserts nothing into a path without the sources."""
    stripped = _without_sources()
    monkeypatch.setattr(sys, "path", list(stripped))
    importlib.reload(cyclone)
    assert sys.path == stripped


def test_script_run_puts_the_sources_first_and_answers_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script from a path without the sources, the validator inserts them and prints its usage."""
    monkeypatch.setattr(sys, "path", _without_sources())
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert sys.path[0] == str(SRC)


def test_script_run_with_the_sources_present_adds_no_duplicate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script with the sources already importable, the validator leaves the import path as it was."""
    present = [str(SRC), *_without_sources()]
    monkeypatch.setattr(sys, "path", list(present))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    capsys.readouterr()
    assert sys.path == present


def test_json_safe_turns_numpy_values_into_python_values() -> None:
    """Arrays become lists and NumPy scalars become bool, int or float; other values pass unchanged."""
    import numpy as np

    payload = {"a": np.arange(2), "b": np.bool_(True), "c": np.int32(3), "d": np.float32(0.5), "e": "text"}
    assert cyclone._json_safe(payload) == {"a": [0, 1], "b": True, "c": 3, "d": 0.5, "e": "text"}
    assert type(cyclone._json_safe(np.int32(3))) is int


@pytest.mark.parametrize("digest", [None, 7, "abc"])
def test_digest_check_refuses_a_missing_or_malformed_digest(digest: object) -> None:
    """A report whose digest is absent, not a string, or not 64 characters does not verify."""
    report = cyclone.build_cbc_report([])
    report["payload_sha256"] = digest
    assert cyclone.verify_payload_digest(report) is False


def test_saturation_evidence_lists_every_failing_condition() -> None:
    """A non-converged, short run with a non-finite chi_i and no heat-flux trace fails for each reason."""
    from types import SimpleNamespace

    result = SimpleNamespace(chi_i_gB=float("nan"), Q_i_t=[1.0], converged=False)
    evidence = cyclone.assess_cbc_saturation_evidence(result, SimpleNamespace(n_steps=1))
    assert evidence["passed"] is False
    assert evidence["reasons"] == [
        "solver did not converge",
        "n_steps below saturation evidence minimum",
        "chi_i_gB is not finite",
        "heat-flux time trace is missing or non-finite",
    ]
    assert evidence["tail_relative_drift"] == float("inf")


@pytest.mark.parametrize(
    ("results", "reason"),
    [
        ([{"test": "V1_linear_recovery", "passed": True}], "cbc saturation result missing"),
        (
            [{"test": "V1_linear_recovery", "passed": True}, {"test": "V4_cbc_saturated", "passed": False}],
            "cbc saturation evidence missing",
        ),
        (
            [
                {"test": "V1_linear_recovery", "passed": False},
                {"test": "V4_cbc_saturated", "saturation_evidence": {"passed": True, "reasons": []}},
            ],
            "diagnostic checks failed",
        ),
    ],
)
def test_report_blocks_the_claim_and_names_why(results: list[dict[str, object]], reason: str) -> None:
    """Without the V4 result, without its evidence, or with a failed diagnostic, the claim stays blocked."""
    report = cyclone.build_cbc_report(results)
    assert report["saturation_claim_admitted"] is False
    assert reason in report["blocked_reasons"]
    assert cyclone.verify_payload_digest(report) is True


def test_command_runs_all_four_cases_and_writes_the_report_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command runs the four diagnostics, writes a digest-bound report and its summary, and exits by them."""
    import json

    json_out = tmp_path / "out" / "cyclone.json"
    markdown_out = tmp_path / "md" / "cyclone.md"
    code = cyclone.main(["--json-out", str(json_out), "--markdown-out", str(markdown_out)])
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert [result["test"] for result in report["results"]] == [
        "V1_linear_recovery",
        "V2_energy_conservation",
        "V3_zonal_flow_generation",
        "V4_cbc_saturated",
    ]
    assert cyclone.verify_payload_digest(report) is True
    assert code == (0 if report["diagnostic_checks_passed"] else 1)
    summary = markdown_out.read_text(encoding="utf-8")
    assert f"- Payload SHA-256: `{report['payload_sha256']}`" in summary
    out = capsys.readouterr().out
    assert f"Report: {json_out}" in out and "Saturation claim:" in out


def test_required_saturation_fails_the_command_while_the_claim_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --require-saturation the command exits one when diagnostics pass but saturation is not admitted."""
    passing = {"passed": True}
    for name, test in (
        ("run_linear_recovery", "V1_linear_recovery"),
        ("run_energy_conservation", "V2_energy_conservation"),
        ("run_zonal_flow", "V3_zonal_flow_generation"),
    ):
        monkeypatch.setattr(cyclone, name, lambda test=test: {"test": test, **passing})
    monkeypatch.setattr(
        cyclone,
        "run_cbc_saturated",
        lambda: {
            "test": "V4_cbc_saturated",
            "passed": False,
            "saturation_evidence": {"passed": False, "reasons": ["x"]},
        },
    )
    argv = ["--json-out", str(tmp_path / "r.json"), "--markdown-out", str(tmp_path / "r.md")]
    assert cyclone.main(argv) == 0
    assert cyclone.main([*argv, "--require-saturation"]) == 1
    assert "Saturation claim: BLOCKED" in capsys.readouterr().out
