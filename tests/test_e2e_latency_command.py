# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation

"""Actual standalone/imported-main E2E validator commands over a real observation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from e2e_latency_observation import ROOT
from e2e_latency_observation import measured_report as measured_report


@pytest.mark.parametrize("entry", ["script", "imported-main"])
@pytest.mark.parametrize("mode", ["local-json", "qualified-text", "zero-budget", "nan-budget", "help", "syntax"])
def test_actual_public_latency_command(measured_report: Path, tmp_path: Path, entry: str, mode: str) -> None:
    """Real CLI entry points retain output/status and refuse nonfinite budgets."""
    if entry == "script":
        command = [sys.executable, str(ROOT / "validation/validate_e2e_latency_evidence.py")]
    else:
        command = [sys.executable, "-c", "from validation.validate_e2e_latency_evidence import main; main()"]
    arguments = [str(measured_report)]
    if mode == "local-json":
        arguments += ["--allow-local-unqualified", "--json-out"]
    elif mode == "zero-budget":
        arguments += ["--allow-local-unqualified", "--max-e2e-p95-us", "0", "--json-out"]
    elif mode == "nan-budget":
        arguments += ["--allow-local-unqualified", "--max-e2e-p95-us", "nan"]
    elif mode == "help":
        arguments = ["--help"]
    elif mode == "syntax":
        arguments += ["--unknown-option"]
    result = subprocess.run(
        [*command, *arguments],
        cwd=tmp_path,
        env=dict(
            os.environ, PYTHONPATH=(str(ROOT) + os.pathsep if entry == "imported-main" else "") + str(ROOT / "src")
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    (tmp_path / "command.json").write_text(
        json.dumps(
            {
                "argv": [*command, *arguments],
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        result.returncode
        == {"local-json": 0, "qualified-text": 1, "zero-budget": 1, "nan-budget": 1, "help": 0, "syntax": 2}[mode]
    )
    if mode in {"local-json", "zero-budget"}:
        payload = json.loads(result.stdout)
        assert set(payload) == {
            "status",
            "errors",
            "p95_us",
            "target_hardware_id",
            "target_hardware_class",
            "rt_kernel",
        }
        assert payload["status"] == ("pass" if mode == "local-json" else "fail")
        assert payload["p95_us"] > 0
        assert payload["target_hardware_id"] is None
    elif mode == "qualified-text":
        assert "E2E latency evidence: fail" in result.stdout and "ERROR target_hardware.id" in result.stdout
    elif mode == "nan-budget":
        assert "max_e2e_p95_us must be a finite non-negative number" in result.stderr
    elif mode == "help":
        assert "--max-e2e-p95-us" in result.stdout
    else:
        assert "unrecognized arguments" in result.stderr
