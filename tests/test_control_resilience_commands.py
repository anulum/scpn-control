# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Resilience campaign real API and commands.

"""Exercise genuine deterministic fault campaigns, refusals and output/exit order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from campaign_command_observation import campaign_command, observe_command

from validation.control_resilience_campaign import generate_campaign_report, render_markdown

MODULE = "validation.control_resilience_campaign"


@pytest.mark.parametrize("entry", ["script", "imported-main"])
@pytest.mark.parametrize("scenario", ["pass-strict", "fail-strict", "fail-nonstrict"])
def test_actual_metrics_and_reports_precede_strict_exit(tmp_path: Path, entry: str, scenario: str) -> None:
    """Measured reports are reproducible and written even for a strict failure.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary UTF-8 report and process custody.
    entry, scenario : str
        Public entry and actual mild/severe perturbation with strict policy.
    """
    output = tmp_path / "nested/report.json"
    markdown = tmp_path / "nested/report.md"
    fail = scenario.startswith("fail")
    kwargs = dict(
        seed=7, episodes=4, window=32, noise_std=10.0 if fail else 0.01, recovery_epsilon=1e-12 if fail else 0.03
    )
    args = [
        "--seed",
        "7",
        "--episodes",
        "4",
        "--window",
        "32",
        "--noise-std",
        str(kwargs["noise_std"]),
        "--recovery-epsilon",
        str(kwargs["recovery_epsilon"]),
        "--output-json",
        str(output),
        "--output-md",
        str(markdown),
    ]
    if scenario != "fail-nonstrict":
        args.append("--strict")
    result = observe_command(tmp_path, campaign_command(MODULE, entry) + args, include_repository=entry != "script")
    assert result.returncode == (2 if scenario == "fail-strict" else 0), result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    expected = generate_campaign_report(
        seed=7, episodes=4, window=32, noise_std=10.0 if fail else 0.01, recovery_epsilon=1e-12 if fail else 0.03
    )
    assert report["campaign"] == expected["campaign"]
    assert report["campaign"]["passes_thresholds"] is (not fail)
    assert report["runtime_seconds"] >= 0
    assert report["generated_at_utc"].endswith("+00:00")
    assert markdown.read_text(encoding="utf-8") == render_markdown(report)
    assert "Control resilience campaign complete." in result.stdout


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"episodes": 0}, "episodes"),
        ({"window": 15}, "window"),
        ({"noise_std": -1.0}, "noise_std"),
        ({"noise_std": float("inf")}, "noise_std"),
        ({"bit_flip_interval": 0}, "bit_flip_interval"),
        ({"recovery_window": 0}, "recovery_window"),
        ({"recovery_epsilon": 0.0}, "recovery_epsilon"),
        ({"recovery_epsilon": float("nan")}, "recovery_epsilon"),
        ({"seed": -1}, "seed"),
    ],
)
def test_actual_public_producer_input_refusals(kwargs: dict[str, Any], diagnostic: str) -> None:
    """Named invalid scalars fail through the public adapter, never a private normalizer.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Authored invalid domain inputs; no measurements or backend substitution.
    diagnostic : str
        Named field in the actual refusal.
    """
    with pytest.raises(ValueError, match=diagnostic):
        generate_campaign_report(**kwargs)


def test_actual_campaign_metrics_repeat_and_leave_global_rng_unchanged() -> None:
    """Local-seed campaigns repeat metrics while preserving the caller's global RNG."""
    before = np.random.get_state()
    first = generate_campaign_report(seed=19, episodes=2, window=16)
    second = generate_campaign_report(seed=19, episodes=2, window=16)
    after = np.random.get_state()
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert first["campaign"] == second["campaign"]
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize(
    "args,status,diagnostic",
    [
        (["--help"], 0, "--strict"),
        (["--unknown"], 2, "unrecognized arguments"),
        (["--episodes", "0"], 1, "episodes must be >= 1"),
    ],
)
def test_actual_cli_help_syntax_and_domain_refusals(
    tmp_path: Path, args: list[str], status: int, diagnostic: str
) -> None:
    """CLI refusal occurs before explicit temporary reports are written.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Unwritten temporary report destinations.
    args : list[str]
        Help, invalid option or invalid episode count.
    status, diagnostic : int and str
        Expected actual process status/observable message.
    """
    output = tmp_path / "invalid.json"
    markdown = tmp_path / "invalid.md"
    result = observe_command(
        tmp_path,
        campaign_command(MODULE, "script") + args + ["--output-json", str(output), "--output-md", str(markdown)],
    )
    assert result.returncode == status and diagnostic in result.stdout + result.stderr
    assert not output.exists() and not markdown.exists()
