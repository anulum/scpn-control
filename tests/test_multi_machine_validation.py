# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Facade contract tests.

"""Exercise the public multi-machine facade refusal and native documentation workflow."""

from __future__ import annotations

import doctest
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import validation.confinement_reference as confinement_module
import validation.equilibrium_execution as equilibrium_module
import validation.machine_inputs as inputs_module
import validation.synthetic_diagnostics as diagnostics_module
from validation.multi_machine_validation import (
    MachineConfig,
    MultiMachineValidator,
    ValidationResult,
    diiid_h_mode,
    iter_15ma,
    jet_high_performance,
    nstx_u_standard,
    sparc_baseline,
)


@pytest.mark.parametrize("factory", [iter_15ma, jet_high_performance, diiid_h_mode, sparc_baseline, nstx_u_standard])
def test_machine_preset_cannot_generate_unmeasured_passes(factory: Callable[[], MachineConfig]) -> None:
    """Every named machine refuses unsupported validation; no RNG-derived residual is published."""
    validator = MultiMachineValidator([factory()])
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="real model runs and machine-specific"):
        validator.run_all(seed=123)
    assert validator.results == []
    after = np.random.get_state()
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("format_name", ["save_json", "save_markdown"])
@pytest.mark.parametrize("preexisting", [False, True])
def test_export_refuses_before_touching_destination(tmp_path: Path, format_name: str, preexisting: bool) -> None:
    """Legacy caller-supplied PASS records cannot bypass the unavailable evidence workflow."""
    validator = MultiMachineValidator([iter_15ma()])
    validator.results.append(
        ValidationResult("equilibrium_convergence", "ITER", "NRMSE", 0.01, 0.02, True, "unverified")
    )
    destination = tmp_path / "report"
    if preexisting:
        destination.write_bytes(b"retain previous artifact")
    export = validator.save_json if format_name == "save_json" else validator.save_markdown
    with pytest.raises(RuntimeError, match="verified model/reference evidence"):
        export(destination)
    if preexisting:
        assert destination.read_bytes() == b"retain previous artifact"
    else:
        assert not destination.exists()


def test_native_machine_examples_execute_actual_public_apis() -> None:
    """Run input-capture and real calibration examples from their defining docstrings."""
    results = [
        doctest.testmod(module)
        for module in (inputs_module, confinement_module, equilibrium_module, diagnostics_module)
    ]
    assert sum(result.failed for result in results) == 0
    assert sum(result.attempted for result in results) == 14
