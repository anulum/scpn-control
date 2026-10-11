# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Equilibrium execution contract tests.

"""Exercise real fixed-boundary execution and evidence refusal through the public adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from validation.multi_machine_validation import (
    EquilibriumExecution,
)


@pytest.fixture
def equilibrium_execution() -> EquilibriumExecution:
    """Bind the actual shipped SOR configuration without modifying iteration controls."""
    path = Path(__file__).resolve().parents[1] / "validation/iter_validated_config.json"
    return EquilibriumExecution(path, hashlib.sha256(path.read_bytes()).hexdigest(), "ITER-Validated", 1e-4)


def test_equilibrium_actual_runtime_retains_failed_convergence(
    equilibrium_execution: EquilibriumExecution, tmp_path: Path
) -> None:
    """Run the full actual solver and retain negative evidence with independent field residual."""
    encoded = equilibrium_execution.evaluate(tmp_path)
    result = json.loads(encoded)
    assert result["config_utf8"].encode() == equilibrium_execution.config_path.read_bytes()
    assert result["solver_result"]["iterations"] == 1000
    assert result["solver_result"]["converged"] is False
    assert result["numerical_pass"] is False
    assert result["independent_gs_rms"] > equilibrium_execution.max_gs_rms
    assert result["independent_gs_rms"] == pytest.approx(result["solver_result"]["gs_residual"], rel=1e-12)
    assert np.asarray(result["fields"]["psi"]).shape == (65, 65)
    assert np.asarray(result["fields"]["J_phi"]).shape == (65, 65)
    assert result["external_reference_compared"] is result["actionable"] is result["facility_validated"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf"), True])
def test_equilibrium_rejects_invalid_admission_threshold(
    equilibrium_execution: EquilibriumExecution, tmp_path: Path, threshold: float
) -> None:
    """Reject invalid residual policies before starting solver work."""
    with pytest.raises(ValueError, match="max_gs_rms"):
        dataclasses.replace(equilibrium_execution, max_gs_rms=threshold).evaluate(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_equilibrium_rejects_hash_and_identity_changes(
    equilibrium_execution: EquilibriumExecution, tmp_path: Path
) -> None:
    """A machine label or changed file cannot silently select another numerical case."""
    with pytest.raises(ValueError, match="SHA256"):
        dataclasses.replace(equilibrium_execution, config_sha256="0" * 64).evaluate(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        dataclasses.replace(equilibrium_execution, reactor_name="JET").evaluate(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("boundary,budget", [("free_boundary", 1000), ("fixed_boundary", 0)])
def test_equilibrium_refuses_unsupported_execution_contract(
    equilibrium_execution: EquilibriumExecution, tmp_path: Path, boundary: str, budget: int
) -> None:
    """Reject a real copied free-boundary or zero-budget configuration before fixed-boundary execution."""
    config = json.loads(equilibrium_execution.config_path.read_bytes())
    config["solver"]["boundary_variant"] = boundary
    config["solver"]["max_iterations"] = budget
    copied = tmp_path / "config.json"
    copied.write_text(json.dumps(config))
    scratch = tmp_path / "run"
    scratch.mkdir()
    case = dataclasses.replace(
        equilibrium_execution, config_path=copied, config_sha256=hashlib.sha256(copied.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="fixed-boundary configuration"):
        case.evaluate(scratch)
    assert list(scratch.iterdir()) == []
    assert json.loads(copied.read_bytes()) == config


def test_equilibrium_refuses_blank_identity_before_execution(
    equilibrium_execution: EquilibriumExecution, tmp_path: Path
) -> None:
    """Require explicit reactor identity before copying files or starting a solver."""
    with pytest.raises(ValueError, match="reactor_name"):
        dataclasses.replace(equilibrium_execution, reactor_name=" ").evaluate(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "relaxation,message",
    [
        (float("nan"), "equilibrium returned nonfinite fields"),
        (1e160, "equilibrium GS residual is not finite"),
    ],
)
def test_equilibrium_refuses_actual_nonfinite_runtime_evidence(
    equilibrium_execution: EquilibriumExecution, tmp_path: Path, relaxation: float, message: str
) -> None:
    """Reject genuine solver output from malformed/extreme copies, without replacing the solver."""
    original = equilibrium_execution.config_path.read_bytes()
    config = json.loads(original)
    config["solver"]["relaxation_factor"] = relaxation
    # A loose copied stopping policy permits the actual first-step output to
    # reach the adapter. It grants no validation and changes no canonical policy.
    config["solver"]["convergence_threshold"] = 1e300
    assert config["solver"]["max_iterations"] == 1000
    copied = tmp_path / "stress.json"
    captured = json.dumps(config).encode("utf-8")
    copied.write_bytes(captured)
    scratch = tmp_path / "run"
    scratch.mkdir()
    case = dataclasses.replace(
        equilibrium_execution, config_path=copied, config_sha256=hashlib.sha256(captured).hexdigest()
    )
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(RuntimeError, match=message):
        case.evaluate(scratch)
    assert list(scratch.iterdir()) == []
    assert copied.read_bytes() == captured
    assert equilibrium_execution.config_path.read_bytes() == original
