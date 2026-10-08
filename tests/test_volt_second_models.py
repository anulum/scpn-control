# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second analytic model record tests
"""Exercise the original public configuration and relocated immutable records."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import pytest

from validation import validate_volt_second as facade
from validation import volt_second_models as models


@pytest.mark.parametrize(
    "field",
    [
        "flux_budget_vs",
        "plasma_inductance_uh",
        "plasma_resistance_uohm",
        "major_radius_m",
        "plasma_current_ma",
        "bootstrap_current_ma",
        "ramp_duration_s",
        "flat_duration_s",
        "ramp_down_duration_s",
        "standalone_ramp_flux_vs",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "1", 10**400])
def test_config_refuses_nonfinite_or_coerced_numbers(field: str, value: object) -> None:
    """Reject invalid numbers through the public configuration constructor."""
    values: dict[str, object] = asdict(models.default_config())
    values[field] = value
    constructor: Callable[..., models.VoltSecondConfig] = models.VoltSecondConfig
    with pytest.raises(ValueError, match="finite"):
        constructor(**values)


@pytest.mark.parametrize(
    "field", ["flux_budget_vs", "plasma_inductance_uh", "plasma_resistance_uohm", "major_radius_m", "plasma_current_ma"]
)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_config_requires_positive_circuit_and_current(field: str, value: float) -> None:
    """Refuse nonpositive physical circuit/current quantities."""
    values = asdict(models.default_config())
    values[field] = value
    with pytest.raises(ValueError, match="positive"):
        models.VoltSecondConfig(**values)


@pytest.mark.parametrize(
    "field",
    ["bootstrap_current_ma", "ramp_duration_s", "flat_duration_s", "ramp_down_duration_s", "standalone_ramp_flux_vs"],
)
def test_config_requires_nonnegative_stages(field: str) -> None:
    """Refuse negative bootstrap, duration and standalone ramp values."""
    values = asdict(models.default_config())
    values[field] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        models.VoltSecondConfig(**values)


def test_config_exports_and_budget_are_preserved() -> None:
    """Keep the original public record address and circuit conversion."""
    config = models.default_config()
    assert facade.VoltSecondConfig is models.VoltSecondConfig
    assert type(config).__module__ == "validation.validate_volt_second"
    assert config.budget().Phi_CS_Vs == 300.0
    assert config.budget().L_plasma_H == pytest.approx(10e-6)
    assert config.budget().R_plasma_Ohm == pytest.approx(5e-6)
    revised = replace(config, bootstrap_current_ma=0.0, flat_duration_s=0.0)
    assert revised.bootstrap_current_ma == revised.flat_duration_s == 0.0


def test_bootstrap_must_be_smaller_than_total_current() -> None:
    """Refuse a nonpositive driven-current interval."""
    with pytest.raises(ValueError, match="bootstrap current"):
        replace(models.default_config(), bootstrap_current_ma=15.0)


def test_original_record_addresses_restore_in_fresh_interpreter(tmp_path: Path) -> None:
    """Restore every actual result record through the preserved public facade."""
    observed = facade.validate_volt_second()
    payload = tmp_path / "result.pickle"
    payload.write_bytes(pickle.dumps(observed))
    script = tmp_path / "restore.py"
    script.write_text(
        "import pickle, sys\nfrom pathlib import Path\nfrom validation.validate_volt_second import VoltSecondValidationResult\nobserved=pickle.loads(Path(sys.argv[1]).read_bytes())\nassert isinstance(observed,VoltSecondValidationResult)\nassert observed.passed and observed.config.flux_budget_vs==300.0\nassert len(observed.scaling)==4\nprint('restored')\n",
        encoding="utf-8",
    )
    root = Path(facade.__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(script), str(payload)],
        env={
            **os.environ,
            "PYTHONPATH": str(root) + os.pathsep + str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "restored"
