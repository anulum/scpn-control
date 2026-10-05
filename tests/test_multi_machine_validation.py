# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Multi Machine Validation.

"""Exercise synthetic diagnostic shapes and prevent unsupported machine-validation claims."""

from __future__ import annotations

import dataclasses
import doctest
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import validation.confinement_reference as confinement_module
import validation.equilibrium_execution as equilibrium_module
import validation.machine_inputs as inputs_module
import validation.synthetic_diagnostics as diagnostics_module
from validation.multi_machine_validation import (
    ConfinementReference,
    EquilibriumExecution,
    MachineConfig,
    MultiMachineValidator,
    ValidationResult,
    diiid_h_mode,
    iter_15ma,
    jet_high_performance,
    nstx_u_standard,
    sparc_baseline,
)


def test_synthetic_diagnostics() -> None:
    """Exercise actual synthetic profile sampling; shapes/positivity do not prove diagnostic fidelity."""
    machine = iter_15ma()
    rho = np.linspace(0, 1, 50, dtype=np.float64)
    temperature = machine.Te_profile(rho)
    density = machine.ne_profile(rho)
    thomson = machine.diagnostics.thomson_scattering(temperature, density, 20)
    assert len(thomson["Te_keV"]) == len(thomson["ne_19"]) == 20
    sxr = machine.diagnostics.soft_xray(temperature, density, rho, 40)
    assert len(sxr) == 40 and np.all(sxr > 0)


def test_ece_legacy_samples_ignore_magnetic_profile() -> None:
    """Keep the documented temperature-only sampling distinct from an ECE forward model."""
    diagnostics = iter_15ma().diagnostics
    temperature = np.linspace(1.0, 10.0, 50, dtype=np.float64)
    field = np.full(50, 5.3)
    state = np.random.get_state()
    try:
        np.random.seed(17)
        first = diagnostics.ece_radiometer(temperature, field, 12)
        np.random.seed(17)
        second = diagnostics.ece_radiometer(temperature, field * 2.0, 12)
    finally:
        np.random.set_state(state)
    assert first.shape == (12,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(temperature, np.linspace(1.0, 10.0, 50, dtype=np.float64))


@pytest.mark.parametrize("diagnostic", ["interferometer", "bolometer"])
def test_legacy_chord_examples_scale_with_radius_and_ignore_rho(diagnostic: str) -> None:
    """Exercise the documented geometric proxy without claiming radial integration fidelity."""
    diagnostics = iter_15ma().diagnostics
    sample = diagnostics.interferometer if diagnostic == "interferometer" else diagnostics.bolometer
    profile = np.linspace(1.0, 10.0, 50, dtype=np.float64)
    rho = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    state = np.random.get_state()
    try:
        np.random.seed(17)
        first = sample(profile, rho, 0.5, 8)
        np.random.seed(17)
        doubled_radius = sample(profile, rho, 1.0, 8)
        np.random.seed(17)
        reversed_grid = sample(profile, rho[::-1], 0.5, 8)
    finally:
        np.random.set_state(state)
    assert first.shape == (8,)
    np.testing.assert_allclose(doubled_radius, first * 2.0)
    np.testing.assert_array_equal(first, reversed_grid)


def test_legacy_magnetic_examples_ignore_machine_geometry() -> None:
    """Random constants retain their stated shapes and cannot establish machine-dependent magnetics."""
    diagnostics = iter_15ma().diagnostics
    state = np.random.get_state()
    try:
        np.random.seed(17)
        first = diagnostics.magnetics(6.2, 2.0)
        np.random.seed(17)
        second = diagnostics.magnetics(1.67, 0.67)
    finally:
        np.random.set_state(state)
    assert first["flux_loops"].shape == (20,)
    assert first["b_probes"].shape == (30,)
    np.testing.assert_array_equal(first["flux_loops"], second["flux_loops"])
    np.testing.assert_array_equal(first["b_probes"], second["b_probes"])
    assert first["Ip"] == second["Ip"]


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


@pytest.mark.parametrize("factory", [iter_15ma, jet_high_performance, diiid_h_mode, sparc_baseline, nstx_u_standard])
def test_snapshot_binds_actual_machine_values(factory: Callable[[], MachineConfig]) -> None:
    """Capture actual preset profiles and units without silently changing the zero edge."""
    machine = factory()
    rho = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    encoded = machine.snapshot(rho)
    decoded = json.loads(encoded)
    assert decoded["machine"] == machine.name
    assert decoded["scalars"]["Ip_MA"] == machine.Ip_MA
    assert decoded["units"]["ne"] == "1e19 m^-3"
    assert decoded["units"]["Te"] == "keV"
    np.testing.assert_array_equal(decoded["profiles"]["Te"], machine.Te_profile(rho))
    assert decoded["profiles"]["ne"][-1] == decoded["profiles"]["Ti"][-1] == 0.0
    assert machine.snapshot(rho) == encoded
    changed = dataclasses.replace(machine, Ip_MA=machine.Ip_MA * 0.9).snapshot(rho)
    assert hashlib.sha256(changed.encode()).digest() != hashlib.sha256(encoded.encode()).digest()


def test_snapshot_isolates_callback_inputs_and_outputs() -> None:
    """Mutating callbacks cannot rewrite an earlier sample, the caller grid or captured scalars."""
    machine = iter_15ma()
    rho = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    shared = np.zeros(5)

    def density(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Deliberately modify callback-owned input and shared output."""
        grid[:] = 7.0
        shared[:] = 2.0
        machine.Ip_MA = 3.0
        return shared

    def temperature(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Require the original grid despite mutation by the preceding callback."""
        np.testing.assert_array_equal(grid, rho)
        shared[:] = 4.0
        return shared

    machine.ne_profile = density
    machine.Te_profile = temperature
    result = json.loads(machine.snapshot(rho))
    assert result["scalars"]["Ip_MA"] == 15.0
    assert result["profiles"]["ne"] == [2.0] * 5
    assert result["profiles"]["Te"] == [4.0] * 5
    np.testing.assert_array_equal(rho, np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize(
    "field,value",
    [
        ("B0", -1.0),
        ("Ip_MA", -10.0),
        ("a", 6.2),
        ("P_aux_MW", -1.0),
        ("delta", 1.0),
        ("R0", float("nan")),
        ("kappa", True),
    ],
)
def test_snapshot_rejects_invalid_machine_scalars(field: str, value: float) -> None:
    """Reject malformed machine state before a model can consume it."""
    machine = iter_15ma()
    setattr(machine, field, value)
    with pytest.raises(ValueError):
        machine.snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize("values", [[0.0, 0.5, float("nan")], [0.0, 0.5, 0.5, 1.0], [0.1, 0.5, 1.0], [0.0, 1.0]])
def test_snapshot_rejects_invalid_grid(values: list[float]) -> None:
    """Require explicit axis-to-edge sampling without implicit grid repairs."""
    with pytest.raises(ValueError, match="rho"):
        iter_15ma().snapshot(np.asarray(values, dtype=np.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_snapshot_rejects_invalid_profile_values(bad: float) -> None:
    """Reject original nonfinite/negative-profile reproducer inputs at capture time."""
    machine = iter_15ma()
    machine.ne_profile = lambda grid: np.full_like(grid, bad)
    with pytest.raises(ValueError, match="ne profile"):
        machine.snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.mark.parametrize("kind", ["shape", "complex", "bool"])
def test_snapshot_rejects_invalid_profile_representation(kind: str) -> None:
    """Prevent broadcasting and lossy conversion from changing supplied profile values."""
    machine = iter_15ma()
    values = {"shape": np.ones(1), "complex": np.ones(5, dtype=complex), "bool": np.ones(5, dtype=bool)}
    machine.Te_profile = lambda grid: values[kind]
    with pytest.raises(ValueError, match="Te profile"):
        machine.snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


@pytest.fixture
def confinement_reference() -> ConfinementReference:
    """Bind the actual shipped calibration CSV and coefficients, without fabricated outputs."""
    root = Path(__file__).resolve().parents[1] / "validation/reference_data/itpa"
    csv_path = root / "hmode_confinement.csv"
    coeff = root / "ipb98y2_coefficients.json"
    return ConfinementReference(
        csv_path,
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        coeff,
        hashlib.sha256(coeff.read_bytes()).hexdigest(),
        0,
        "ITER",
        "design",
        "derived_calibration",
        0.1,
    )


def test_confinement_actual_reference_comparison(confinement_reference: ConfinementReference) -> None:
    """Real model results differ by operating point and may fail the declared comparison."""
    first = json.loads(confinement_reference.evaluate())
    second = json.loads(dataclasses.replace(confinement_reference, row_index=1, machine="JET", shot="92436").evaluate())
    assert first["predicted_tau_E_s"] == pytest.approx(3.6060077194126654, rel=1e-12)
    assert second["predicted_tau_E_s"] == pytest.approx(0.5201485510391346, rel=1e-12)
    assert first["comparison_pass"] is True
    assert second["comparison_pass"] is False
    assert first["inputs"]["Ploss_MW"] == 87.0
    assert first["reference_tau_E_s"] == 3.7
    assert first["source_row"]["shot"] == "design"
    assert first["reference_sha256"] == confinement_reference.csv_sha256
    for record in (first, second):
        assert record["actionable"] is record["facility_validated"] is record["source_class_verified"] is False
        assert record["training_domain"] == "not_assessed"


@pytest.mark.parametrize("which", ["csv", "coefficients"])
def test_confinement_refuses_changed_bound_bytes(
    confinement_reference: ConfinementReference, tmp_path: Path, which: str
) -> None:
    """A content change cannot keep the original reference/coefficient admission digest."""
    original = confinement_reference.csv_path if which == "csv" else confinement_reference.coefficients_path
    changed = tmp_path / original.name
    changed.write_bytes(original.read_bytes() + b"\n")
    case = (
        dataclasses.replace(confinement_reference, csv_path=changed)
        if which == "csv"
        else dataclasses.replace(confinement_reference, coefficients_path=changed)
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        case.evaluate()


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf"), True, np.bool_(True), np.bool_(False)])
def test_confinement_refuses_invalid_tolerance(confinement_reference: ConfinementReference, tolerance: Any) -> None:
    """A caller cannot manufacture success using an unbounded/nonfinite tolerance."""
    with pytest.raises(ValueError, match="tolerance"):
        dataclasses.replace(confinement_reference, relative_tolerance=tolerance).evaluate()


def test_confinement_refuses_wrong_operating_point(confinement_reference: ConfinementReference) -> None:
    """Selecting another row requires updating its explicit machine/shot identity."""
    with pytest.raises(ValueError, match="identity mismatch"):
        dataclasses.replace(confinement_reference, row_index=1).evaluate()


@pytest.mark.parametrize(
    "defect",
    [
        "negative_power",
        "nonfinite_reference",
        "duplicate_header",
        "extra_field",
        "missing_field",
        "missing_header",
        "blank_source",
        "comparison_overflow",
    ],
)
def test_confinement_valid_digest_does_not_admit_malformed_data(
    confinement_reference: ConfinementReference, tmp_path: Path, defect: str
) -> None:
    """Hash agreement establishes byte identity, while the reader still checks content."""
    lines = confinement_reference.csv_path.read_text().splitlines()
    fields = lines[0].split(",")
    values = lines[1].split(",")
    if defect == "negative_power":
        values[fields.index("Ploss_MW")] = "-87"
    elif defect == "nonfinite_reference":
        values[fields.index("tau_E_s")] = "nan"
    elif defect == "duplicate_header":
        fields[-1] = fields[0]
    elif defect == "extra_field":
        values.append("unexpected")
    elif defect == "missing_field":
        values.pop()
    elif defect == "missing_header":
        fields[-1] = "unrecognized"
    elif defect == "blank_source":
        values[fields.index("source")] = " "
    else:
        values[fields.index("tau_E_s")] = "1e-320"
    path = tmp_path / "reference.csv"
    path.write_text(",".join(fields) + "\n" + ",".join(values) + "\n")
    case = dataclasses.replace(
        confinement_reference, csv_path=path, csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        case.evaluate()


@pytest.mark.parametrize(
    "mutation", ["duplicate_root", "duplicate_exponent", "nonfinite_metadata", "overflow_metadata"]
)
def test_confinement_rejects_ambiguous_coefficients(
    confinement_reference: ConfinementReference, tmp_path: Path, mutation: str
) -> None:
    """Matching bytes must still describe a unique finite JSON coefficient object."""
    original = confinement_reference.coefficients_path.read_text()
    if mutation == "duplicate_root":
        changed = original.replace('"C": 0.0562', '"C": 99.0, "C": 0.0562', 1)
    elif mutation == "duplicate_exponent":
        changed = original.replace('"Ip_MA": 0.93', '"Ip_MA": 99.0, "Ip_MA": 0.93', 1)
    elif mutation == "nonfinite_metadata":
        changed = original.replace('"total_data_points": 5920', '"total_data_points": NaN', 1)
    else:
        changed = original.replace('"total_data_points": 5920', '"total_data_points": 1e9999', 1)
    assert changed != original
    path = tmp_path / "coefficients.json"
    path.write_text(changed)
    case = dataclasses.replace(
        confinement_reference,
        coefficients_path=path,
        coefficients_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="coefficient JSON"):
        case.evaluate()


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


def test_native_machine_examples_execute_actual_public_apis() -> None:
    """Run input-capture and real calibration examples from their defining docstrings."""
    results = [
        doctest.testmod(module)
        for module in (inputs_module, confinement_module, equilibrium_module, diagnostics_module)
    ]
    assert sum(result.failed for result in results) == 0
    assert sum(result.attempted for result in results) == 14


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source_class": "held_out_empirical"}, "classification"),
        ({"row_index": True}, "integer"),
        ({"row_index": -1}, "integer"),
        ({"row_index": 1000000}, "outside"),
        ({"csv_sha256": "not-a-digest"}, "SHA256"),
        ({"machine": " "}, "identity"),
    ],
)
def test_confinement_rejects_invalid_reference_selection(
    confinement_reference: ConfinementReference, changes: dict[str, Any], message: str
) -> None:
    """Refuse unsupported evidence categories and malformed explicit reference selectors."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(confinement_reference, **changes).evaluate()


@pytest.mark.parametrize("values", [np.array([False, True, True]), np.array([0.0, 0.5, 1.0], dtype=complex)])
def test_snapshot_rejects_lossy_grid_conversion(values: npt.NDArray[Any]) -> None:
    """Boolean and complex radius samples cannot be silently converted into physical coordinates."""
    with pytest.raises(ValueError, match="real numeric"):
        iter_15ma().snapshot(values)


def test_snapshot_rejects_blank_machine_identity() -> None:
    """Require a declared machine identity before invoking profile callbacks."""
    with pytest.raises(ValueError, match="machine name"):
        dataclasses.replace(iter_15ma(), name=" ").snapshot(np.linspace(0.0, 1.0, 5, dtype=np.float64))


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
