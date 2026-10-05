# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable latency public command tests.

"""Real standalone/imported-main differentiable report validator command outcomes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from differentiable_latency_observation import ROOT
from differentiable_latency_observation import observed_transport_reports as observed_transport_reports


@pytest.mark.parametrize("entry", ["script", "imported-main"])
@pytest.mark.parametrize("mode", ["json", "text", "missing", "help", "syntax"])
def test_actual_differentiable_latency_command(
    observed_transport_reports: tuple[Path, Path, Path],
    tmp_path: Path,
    entry: str,
    mode: str,
) -> None:
    """Actual parser/readers preserve local-only pass and native missing-report refusals."""
    one, rollout, readiness = observed_transport_reports
    if entry == "script":
        command = [sys.executable, str(ROOT / "validation/validate_differentiable_transport_latency.py")]
    else:
        command = [
            sys.executable,
            "-c",
            "from validation.validate_differentiable_transport_latency import main; main()",
        ]
    args = [
        "--one-step-report",
        str(one),
        "--rollout-report",
        str(rollout),
        "--readiness-report",
        str(readiness),
        "--require-admitted",
    ]
    if mode == "json":
        args.append("--json-out")
    elif mode == "missing":
        args[1] = str(tmp_path / "absent.json")
    elif mode == "help":
        args = ["--help"]
    elif mode == "syntax":
        args = ["--unknown"]
    env = dict(os.environ, PYTHONPATH=(str(ROOT) + os.pathsep if entry == "imported-main" else "") + str(ROOT / "src"))
    result = subprocess.run(
        [*command, *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=60
    )
    (tmp_path / "command.json").write_text(
        json.dumps(
            {
                "argv": [*command, *args],
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert result.returncode == {"json": 0, "text": 0, "missing": 1, "help": 0, "syntax": 2}[mode]
    if mode == "json":
        payload = json.loads(result.stdout)
        assert payload["status"] == "pass" and payload["admitted_reports"] == 2
        assert payload["full_fidelity_ready"] is False
    elif mode in {"text", "missing"}:
        assert "Differentiable transport latency evidence:" in result.stdout
        assert ("ERROR " in result.stdout) is (mode == "missing")
    elif mode == "help":
        assert "--require-admitted" in result.stdout
    else:
        assert "unrecognized arguments" in result.stderr
