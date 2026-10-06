# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-API docstring coverage gate tests

"""Regression tests for the docstring-coverage gate and per-module debt ratchet."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_docstring_gate as rdg

_DIAGNOSTICS_SAMPLE: list[dict[str, object]] = [
    {"filename": "/abs/repo/src/scpn_control/core/locked_mode.py", "code": "D102"},
    {"filename": "/abs/repo/src/scpn_control/core/locked_mode.py", "code": "D101"},
    {"filename": "src/scpn_control/control/density_controller.py", "code": "D103"},
]


def test_file_to_module_strips_absolute_src_prefix() -> None:
    """An absolute path is reduced to its dotted import path below ``src``."""
    path = "/abs/repo/src/scpn_control/core/elm_model.py"
    assert rdg.file_to_module(path) == "scpn_control.core.elm_model"


def test_file_to_module_strips_relative_src_prefix() -> None:
    """A ``src``-relative path maps to its dotted import path."""
    assert rdg.file_to_module("src/scpn_control/scpn/artifact.py") == "scpn_control.scpn.artifact"


def test_file_to_module_normalises_backslashes() -> None:
    """Windows-style separators are normalised before conversion."""
    assert rdg.file_to_module("src\\scpn_control\\core\\marfe.py") == "scpn_control.core.marfe"


def test_file_to_module_uses_last_src_marker() -> None:
    """A spurious earlier ``src/`` segment does not shadow the package root."""
    path = "/home/src/checkout/src/scpn_control/core/pedestal.py"
    assert rdg.file_to_module(path) == "scpn_control.core.pedestal"


def test_parse_module_counts_groups_per_module() -> None:
    """Diagnostics are grouped by their owning module."""
    counts = rdg.parse_module_counts(_DIAGNOSTICS_SAMPLE)

    assert counts == {
        "scpn_control.core.locked_mode": 2,
        "scpn_control.control.density_controller": 1,
    }


def test_parse_module_counts_empty() -> None:
    """No diagnostics yields an empty mapping (full coverage)."""
    assert rdg.parse_module_counts([]) == {}


def test_parse_module_counts_rejects_missing_filename() -> None:
    """A diagnostic without a string filename is a hard failure."""
    with pytest.raises(ValueError):
        rdg.parse_module_counts([{"code": "D102"}])


def test_ledger_round_trip() -> None:
    """A ledger survives serialisation and deserialisation unchanged."""
    ledger = rdg.DocstringDebtLedger(total=5, per_module={"a.b": 3, "c.d": 2})
    restored = rdg.DocstringDebtLedger.from_dict(ledger.to_dict())

    assert restored == ledger


def test_ledger_to_dict_sorts_modules_and_records_rules() -> None:
    """Serialised counts are key-sorted and the rule set is recorded."""
    ledger = rdg.DocstringDebtLedger(total=2, per_module={"z.z": 1, "a.a": 1})
    payload = ledger.to_dict()

    modules = payload["per_module"]
    assert isinstance(modules, dict)
    assert list(modules) == ["a.a", "z.z"]
    assert payload["rules"] == list(rdg.COVERAGE_RULES)


def test_ledger_from_dict_rejects_wrong_schema() -> None:
    """An unrecognised schema marker is refused."""
    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict({"schema": "other", "total": 0, "per_module": {}})


def test_ledger_from_dict_rejects_non_object_per_module() -> None:
    """A non-object ``per_module`` payload is refused."""
    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict({"schema": rdg.LEDGER_SCHEMA, "total": 0, "per_module": []})


def test_ledger_from_dict_rejects_non_integer_total() -> None:
    """A non-integer ``total`` payload is refused."""
    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict({"schema": rdg.LEDGER_SCHEMA, "total": "many", "per_module": {}})


def _ledger(total: int, per_module: dict[str, int]) -> rdg.DocstringDebtLedger:
    """Construct the explicit historical debt snapshot used by ratchet comparisons."""
    return rdg.DocstringDebtLedger(total=total, per_module=per_module)


def test_ratchet_clean_when_unchanged() -> None:
    """Identical current and baseline debt holds the ratchet."""
    result = rdg.evaluate_ratchet(5, {"a.b": 3, "c.d": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert result.ok
    assert not result.regressions
    assert not result.improvements
    assert result.total_delta == 0


def test_ratchet_detects_module_regression_even_with_flat_total() -> None:
    """A module rising while another falls is still a regression."""
    result = rdg.evaluate_ratchet(5, {"a.b": 4, "c.d": 1}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert result.regressions == {"a.b": (3, 4)}
    assert result.improvements == {"c.d": (2, 1)}
    assert result.total_delta == 0


def test_ratchet_detects_new_module_with_debt() -> None:
    """A module absent from the baseline that now has debt regresses."""
    result = rdg.evaluate_ratchet(7, {"a.b": 3, "new.mod": 2}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert not result.ok
    assert result.regressions == {"new.mod": (0, 2)}
    assert result.total_delta == 2


def test_ratchet_passes_on_overall_improvement() -> None:
    """Falling total with no per-module regression holds the ratchet."""
    result = rdg.evaluate_ratchet(3, {"a.b": 3}, _ledger(5, {"a.b": 3, "c.d": 2}))

    assert result.ok
    assert result.improvements == {"c.d": (2, 0)}
    assert result.total_delta == -2


@pytest.fixture
def patched_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ruff probe deterministic for ``main`` tests."""
    monkeypatch.setattr(rdg, "run_ruff_probe", lambda: list(_DIAGNOSTICS_SAMPLE))


@pytest.fixture
def ratchet_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the ratchet machinery with the hard-zero floor disabled."""
    monkeypatch.setattr(rdg, "ENFORCE_ZERO", False)


def _point_ledger_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect legacy ratchet tests to a temporary ledger so the committed baseline cannot change."""
    ledger_path = tmp_path / "docstring_debt.json"
    monkeypatch.setattr(rdg, "LEDGER_PATH", ledger_path)
    return ledger_path


def test_main_passes_when_debt_within_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None, ratchet_mode: None
) -> None:
    """Current debt equal to the committed baseline passes (ratchet mode)."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(
        _ledger(3, {"scpn_control.core.locked_mode": 2, "scpn_control.control.density_controller": 1}),
        ledger_path,
    )

    assert rdg.main([]) == 0


def test_main_fails_on_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None, ratchet_mode: None
) -> None:
    """Debt above the baseline fails the ratchet."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(1, {"baseline.only": 1}), ledger_path)

    assert rdg.main([]) == 1


def test_main_update_baseline_creates_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None, ratchet_mode: None
) -> None:
    """``--update-baseline`` writes a ledger reflecting the current probe."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)

    assert rdg.main(["--update-baseline"]) == 0

    payload = json.loads(ledger_path.read_text())
    assert payload["schema"] == rdg.LEDGER_SCHEMA
    assert payload["total"] == 3
    assert payload["per_module"]["scpn_control.core.locked_mode"] == 2


def test_main_update_baseline_refuses_increase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None, ratchet_mode: None
) -> None:
    """Raising the recorded total needs the explicit increase flag."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(1, {"baseline.only": 1}), ledger_path)

    assert rdg.main(["--update-baseline"]) == 3
    assert json.loads(ledger_path.read_text())["total"] == 1


def test_main_update_baseline_allows_increase_with_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None, ratchet_mode: None
) -> None:
    """The increase flag permits recording higher debt deliberately (ratchet mode)."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(1, {"baseline.only": 1}), ledger_path)

    assert rdg.main(["--update-baseline", "--allow-baseline-increase"]) == 0
    assert json.loads(ledger_path.read_text())["total"] == 3


def test_main_hard_zero_fails_on_any_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """With the hard-zero floor active, any missing docstring fails the run."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(
        _ledger(3, {"scpn_control.core.locked_mode": 2, "scpn_control.control.density_controller": 1}), ledger_path
    )

    assert rdg.ENFORCE_ZERO is True
    assert rdg.main([]) == 1


def test_main_hard_zero_refuses_update_baseline_even_with_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_probe: None
) -> None:
    """The hard-zero floor refuses to record non-zero debt, flag notwithstanding."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)

    assert rdg.main(["--update-baseline", "--allow-baseline-increase"]) == 3
    assert not ledger_path.exists()


def test_main_passes_when_probe_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty probe passes the hard-zero gate against a zero ledger."""
    ledger_path = _point_ledger_at(tmp_path, monkeypatch)
    rdg.write_ledger(_ledger(0, {}), ledger_path)
    monkeypatch.setattr(rdg, "run_ruff_probe", list)

    assert rdg.main([]) == 0


def test_main_fails_on_unparsable_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ruff probe that cannot run is a hard failure, not a silent pass."""

    def _explode() -> list[dict[str, object]]:
        """Represent the existing legacy test failure when the Ruff probe cannot produce diagnostics."""
        raise RuntimeError("ruff produced no JSON output")

    monkeypatch.setattr(rdg, "run_ruff_probe", _explode)
    _point_ledger_at(tmp_path, monkeypatch)

    assert rdg.main([]) == 2


def test_committed_ledger_matches_schema() -> None:
    """The committed ledger loads cleanly under the current schema."""
    ledger = rdg.load_ledger()

    assert ledger.total == sum(ledger.per_module.values())


@pytest.mark.parametrize("helper", ["_instrumented_launcher", "interrupt", "signal_when_ready"])
def test_cli_rejects_missing_private_or_nested_test_helper_docstring(tmp_path: Path, helper: str) -> None:
    """Remove one real helper docstring in a copied test owner and exercise the unmocked CLI."""
    original = rdg.REPO_ROOT / "tests/test_tglf_launcher.py"
    tree = ast.parse(original.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == helper:
            assert ast.get_docstring(node)
            node.body.pop(0)
            break
    else:
        pytest.fail(f"Real helper {helper} missing from source")
    mutated = tmp_path / "test_tglf_launcher.py"
    mutated.write_text(ast.unparse(tree))
    result = subprocess.run(
        [sys.executable, str(rdg.REPO_ROOT / "tools/run_docstring_gate.py"), "--all-definitions", str(mutated)],
        cwd=rdg.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert helper in result.stderr and str(mutated) in result.stderr
    assert "missing all-definition docstrings" in result.stderr


@pytest.fixture
def copied_gate_graph(tmp_path: Path) -> Path:
    """Copy the actual default AST/Ruff graph without runtime dependency trees."""
    shutil.copytree(
        rdg.REPO_ROOT / "src/scpn_control", tmp_path / "src/scpn_control", ignore=shutil.ignore_patterns("__pycache__")
    )
    targets = set(rdg.ALL_DEFINITION_TARGETS) | {
        "tools/check_test_quality_policy.py",
        "tests/test_check_test_quality_policy.py",
        "tools/check_docstring_debt.py",
        "tests/test_docstring_debt_ratchet.py",
        "tools/docstring_debt.json",
        "pyproject.toml",
    }
    for name in targets:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rdg.REPO_ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize(
    "owner,symbol",
    [
        ("tools/train_rl_tokamak.py", "main"),
        ("tools/rl_training_config.py", "TrainingConfig"),
        ("tools/rl_tokamak_evaluation.py", "evaluate_agent"),
        ("tools/rl_training_results.py", "select_best"),
        ("benchmarks/rl_vs_classical.py", "SimpleMPC"),
        ("examples/tutorial_03_ppo_rl_agent.py", "main"),
        ("tests/test_rl_tokamak.py", "TestPPOAgent"),
        ("tests/test_rl_training_workflow.py", "test_actual_three_seed_api_and_native_shell_plans"),
        ("tests/test_rl_training_results.py", "test_real_json_selection_below_old_sentinel_and_equal_reward"),
        ("tests/test_rl_benchmark_command.py", "test_actual_native_benchmark_and_independent_public_policies"),
        ("tests/test_rl_tutorial_command.py", "test_actual_default_tutorial_retained_artifact_and_no_learning"),
        (
            "tests/test_rl_benchmark_public_claims.py",
            "test_retained_public_rl_claims_match_committed_benchmark_artifact",
        ),
        ("tools/jarvislabs_train.py", "_check_artifact_paths"),
        ("tools/jarvislabs_client.py", "destroy_instance"),
        ("tools/jarvislabs_transport.py", "_endpoint"),
        ("tests/test_jarvislabs_train.py", "test_actual_missing_token_api_and_cli"),
        ("tests/test_jarvislabs_transport.py", "test_actual_native_transport_failure_preserves_files"),
        ("tools/gpu_cbc_benchmark.py", "run_nonlinear_numpy"),
        ("tests/test_gpu_cbc_benchmark.py", "test_actual_fixed_model_reports"),
        ("tools/gk_convergence_benchmark.py", "run_benchmark"),
        ("tests/test_gk_convergence_benchmark.py", "test_actual_provider_workflow_and_public_scratch"),
        ("tools/generate_phase_video.py", "generate"),
        ("tools/phase_video_model.py", "capture_trajectory"),
        ("tools/phase_video_rendering.py", "render_trajectory"),
        ("tests/test_phase_video_model.py", "test_actual_monitor_error_snapshot_refuses_ordinary_trajectory"),
        ("tests/test_generate_phase_video.py", "actual_bundle"),
        ("tools/export_zenodo_dataset.py", "create_archive"),
        ("tests/test_export_zenodo_dataset.py", "test_actual_checkout_zip_has_exact_public_members"),
        ("tools/check_test_quality_policy.py", "collect_violations"),
        ("tests/test_check_test_quality_policy.py", "test_real_cli_scans_named_contract_file"),
        ("tools/ci_rmse_gate.py", "main"),
        ("validation/rmse_dashboard_metrics.py", "beta_rmse_iter_sparc"),
        ("validation/rmse_dashboard_rendering.py", "render_plots"),
        ("validation/rmse_dashboard_command.py", "parse_args"),
        ("tests/test_rmse_dashboard_rendering.py", "actual_report"),
        ("tools/check_version_sync.py", "main"),
        ("tools/check_docs_coverage.py", "iter_public_modules"),
        ("tests/test_check_docs_coverage.py", "_cli"),
        ("tools/check_python_lint_contract.py", "lint_contract_errors"),
        ("tests/test_python_lint_contract.py", "test_actual_hook_refuses_static_governance_drift"),
        ("tools/check_public_surface_hygiene.py", "scan_repository"),
        ("tests/test_public_surface_hygiene.py", "test_actual_index_path_spellings_reach_api_and_cli"),
        ("tools/check_api_contracts.py", "check_contracts"),
        ("tools/api_contract_inventory.py", "build_inventory"),
        ("validation/density_reference_contracts.py", "_validate_artifact"),
        ("validation/density_reference_domains.py", "_is_positive_finite"),
        ("tests/test_density_reference_command.py", "test_actual_cold_script"),
        ("tools/benchmark_full_stack.py", "benchmark_full_stack"),
        ("tests/test_benchmark_full_stack_public.py", "test_real_native_probes_and_readout_export"),
        ("validation/validate_density_reference.py", "validate_density_reference"),
        ("validation/validate_release_evidence.py", "validate_release_evidence"),
        ("validation/validate_runtime_admission_evidence.py", "validate_runtime_admission_evidence"),
        ("validation/validate_multi_shot_campaign_evidence.py", "validate_multi_shot_campaign_evidence"),
        ("tests/test_multi_shot_campaign_evidence_validation.py", "test_public_campaign_report_domain_refusals"),
        ("validation/validate_physics_traceability.py", "validate_physics_traceability"),
        ("validation/validate_data_manifests.py", "validate_manifest_directory"),
        ("validation/validate_public_data_acquisition.py", "validate_public_data_acquisition_manifest"),
        ("validation/plan_neural_equilibrium_training_campaign.py", "build_plan"),
        ("validation/train_mast_efm_neural_equilibrium.py", "build_training_report"),
        ("validation/audit_mast_efm_feature_provenance.py", "build_audit"),
        ("validation/audit_mast_efm_original_feature_sources.py", "build_original_feature_source_audit"),
        ("validation/mast_efm_zarr_store.py", "capture_zarr_store"),
        ("validation/mast_efm_original_source_policy.py", "classify_feature_sources"),
        ("validation/mast_efm_original_source_inputs.py", "inspect_original_sources"),
        ("validation/mast_efm_original_source_reporting.py", "validate_original_audit_report"),
        ("validation/convert_mast_efm_neural_equilibrium_reference.py", "read_reference_zarr"),
        ("validation/mast_efm_reference_arrays.py", "_status_mask"),
        ("validation/evaluate_mast_efm_neural_equilibrium.py", "main"),
        ("validation/mast_efm_evaluation_features.py", "build_feature_projection"),
        ("validation/mast_efm_evaluation_geometry.py", "evaluate_flux_geometry"),
        ("validation/mast_efm_evaluation_geometry.py", "_interpolate_crossing"),
        ("tests/test_mast_efm_evaluation_features.py", "reference_arrays"),
        ("tests/test_mast_efm_evaluation_geometry.py", "geometry_reference"),
        ("tools/generate_native_api_reference.py", "render"),
        ("tools/generate_native_api_reference.py", "_c_declarations"),
        ("tools/generate_native_api_reference.py", "_lean_declarations"),
        ("tests/test_tools/test_generate_native_api_reference.py", "selected_sources"),
        ("tools/check_benchmark_producers.py", "audit_registry"),
        ("tools/check_benchmark_producers.py", "_discover"),
        ("tests/test_benchmark_producer_registry.py", "selected_repository"),
        ("validation/validate_neural_equilibrium_reference.py", "main"),
        ("validation/reference_uri.py", "reference_artifact_uri_error"),
        ("validation/validate_orbit_reference.py", "write_orbit_reference_report"),
        ("validation/orbit_reference_contracts.py", "_validate_artifact"),
        ("validation/orbit_reference_contracts.py", "_finite_number"),
        ("tests/test_orbit_reference_contracts.py", "declaration"),
        ("validation/validate_uncertainty_reference.py", "write_uncertainty_reference_report"),
        ("validation/uncertainty_reference_contracts.py", "_validate_artifact"),
        ("validation/uncertainty_reference_contracts.py", "_finite_number"),
        ("tests/test_uncertainty_reference_contracts.py", "declaration"),
        ("validation/validate_vmec_reference.py", "write_vmec_reference_report"),
        ("validation/vmec_reference_contracts.py", "_validate_artifact"),
        ("validation/vmec_reference_contracts.py", "_finite_number"),
        ("tests/test_vmec_reference_contracts.py", "declaration"),
        ("validation/validate_eped_reference.py", "write_eped_reference_report"),
        ("validation/eped_reference_contracts.py", "_validate_artifact"),
        ("validation/eped_reference_geometry.py", "_is_finite"),
        ("tests/test_eped_reference_contracts.py", "declaration"),
        ("tests/test_eped_reference_geometry.py", "test_grid_domains"),
        ("tests/test_eped_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_marfe_reference.py", "write_marfe_reference_report"),
        ("validation/marfe_reference_contracts.py", "_validate_artifact"),
        ("validation/marfe_reference_domains.py", "_is_finite"),
        ("tests/test_marfe_reference_contracts.py", "declaration"),
        ("tests/test_marfe_reference_domains.py", "test_positive_scan_domains"),
        ("tests/test_marfe_reference_command.py", "test_actual_cold_script"),
        ("tests/test_reference_output_path_refusal.py", "test_real_output_cycle_refusal"),
        ("validation/validate_static_mu_analysis_reference.py", "write_static_mu_analysis_reference_report"),
        ("validation/validate_mu_synthesis_reference.py", "validate_mu_synthesis_reference"),
        ("validation/static_mu_reference_contracts.py", "_validate_source_provenance"),
        ("validation/static_mu_reference_domains.py", "_valid_plant_metadata"),
        ("tests/test_static_mu_reference_contracts.py", "test_external_code_identity"),
        ("tests/test_static_mu_reference_domains.py", "test_uncapped_plant_dimensions"),
        ("tests/test_static_mu_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_elm_reference.py", "write_elm_reference_report"),
        ("validation/elm_reference_contracts.py", "canonical_artifact_sha256"),
        ("validation/elm_reference_domains.py", "_positive_ordered_pair"),
        ("tests/test_elm_reference_validation.py", "test_elm_gate_accepts_measured_hmode_campaign"),
        ("tests/test_elm_reference_contracts.py", "test_original_lexical_uri_policy"),
        ("tests/test_elm_reference_domains.py", "test_original_type_i_fraction_domain"),
        ("tests/test_elm_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_soc_reference.py", "write_soc_reference_report"),
        ("validation/soc_reference_contracts.py", "_validate_source_provenance"),
        ("validation/soc_reference_domains.py", "_valid_lattice_metadata"),
        ("tests/test_soc_reference_validation.py", "test_soc_gate_accepts_measured_turbulence_replay"),
        ("tests/test_soc_reference_contracts.py", "test_external_code_identity"),
        ("tests/test_soc_reference_domains.py", "test_nonnegative_lattice_couplings"),
        ("tests/test_soc_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_neural_turbulence_reference.py", "write_neural_turbulence_reference_report"),
        ("validation/neural_turbulence_reference_contracts.py", "_validate_artifact"),
        ("validation/neural_turbulence_reference_domains.py", "_is_finite_number"),
        (
            "tests/test_neural_turbulence_reference_validation.py",
            "test_neural_turbulence_gate_accepts_real_gk_artifact",
        ),
        ("tests/test_neural_turbulence_reference_contracts.py", "test_format_only_byte_hashes"),
        ("tests/test_neural_turbulence_reference_domains.py", "test_campaign_and_public_presence"),
        ("tests/test_neural_turbulence_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_gk_species_reference.py", "write_gk_species_reference_report"),
        ("validation/gk_species_reference_contracts.py", "verify_payload_digest"),
        ("validation/gk_species_reference_cases.py", "_validate_case"),
        ("validation/gk_species_reference_operators.py", "_validate_pitch_angle_check"),
        ("validation/gk_species_reference_numeric.py", "_numeric_scalar"),
        ("tests/test_gk_species_reference_validation.py", "test_repository_species_reference_cases_pass"),
        ("tests/test_gk_species_reference_domains.py", "test_required_scalar_findings"),
        ("tests/test_gk_species_reference_contracts.py", "test_CRLF_input_identity"),
        ("tests/test_gk_species_reference_operators.py", "test_original_count_coercion"),
        ("tests/test_gk_species_reference_command.py", "test_actual_cold_script"),
        ("src/scpn_control/core/gk_species.py", "collision_frequencies"),
        ("validation/gk_geometry_independent_reference.py", "independent_miller_metric"),
        ("validation/validate_gk_geometry_independent.py", "validate_gk_geometry_independent"),
        ("validation/validate_gk_geometry_reference.py", "write_gk_geometry_reference_report"),
        ("validation/gk_geometry_reference_contracts.py", "_finalise_report"),
        ("validation/gk_geometry_reference_cases.py", "_geometry_arguments"),
        ("tests/test_gk_geometry_reference_validation.py", "test_repository_geometry_reference_cases_pass"),
        ("tests/test_gk_geometry_reference_contracts.py", "test_exact_CRLF_byte_custody"),
        ("tests/test_gk_geometry_reference_domains.py", "test_actual_physical_domain_findings"),
        ("tests/test_gk_geometry_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_neural_transport_reference.py", "write_neural_transport_reference_report"),
        ("validation/neural_transport_reference_contracts.py", "canonical_artifact_sha256"),
        ("validation/neural_transport_reference_domains.py", "_is_unit_interval"),
        ("tests/test_neural_transport_reference_validation.py", "_valid_qualikiz_reference_artifact"),
        ("tests/test_neural_transport_reference_contracts.py", "test_original_lexical_uri_policy"),
        ("tests/test_neural_transport_reference_domains.py", "test_inclusive_score_domains"),
        ("tests/test_neural_transport_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_blob_transport_reference.py", "write_blob_transport_reference_report"),
        ("validation/blob_transport_reference_contracts.py", "canonical_artifact_sha256"),
        ("validation/blob_transport_reference_domains.py", "_positive_ordered_pair"),
        ("tests/test_blob_transport_reference_validation.py", "_valid_blob_reference_artifact"),
        ("tests/test_blob_transport_reference_contracts.py", "test_original_lexical_uri_policy"),
        ("tests/test_blob_transport_reference_domains.py", "test_zero_lower_ordered_pairs"),
        ("tests/test_blob_transport_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_free_boundary_reference.py", "write_free_boundary_reference_report"),
        ("validation/free_boundary_reference_contracts.py", "_valid_equilibrium_metadata"),
        ("tests/test_free_boundary_reference_contracts.py", "test_source_presence"),
        ("tests/test_free_boundary_reference_domains.py", "test_uncapped_positive_metadata_counts"),
        ("tests/test_free_boundary_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_digital_twin_reference.py", "write_digital_twin_reference_report"),
        ("validation/digital_twin_reference_contracts.py", "_validate_source_provenance"),
        ("validation/digital_twin_reference_domains.py", "_valid_grid_metadata"),
        ("tests/test_digital_twin_reference_validation.py", "_valid_digital_twin_reference_artifact"),
        ("tests/test_digital_twin_reference_contracts.py", "test_source_presence"),
        ("tests/test_digital_twin_reference_domains.py", "test_original_grid_domains"),
        ("tests/test_digital_twin_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_disruption_reference.py", "write_disruption_reference_report"),
        ("validation/disruption_reference_contracts.py", "_valid_signal_window"),
        ("tests/test_disruption_reference_validation.py", "_valid_disruption_reference_artifact"),
        ("tests/test_disruption_reference_contracts.py", "test_source_presence"),
        ("tests/test_disruption_reference_domains.py", "test_signal_sample_count"),
        ("tests/test_disruption_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_rzip_reference.py", "write_rzip_reference_report"),
        ("validation/rzip_reference_contracts.py", "_is_finite_number"),
        ("tests/test_rzip_reference_validation.py", "_valid_rzip_reference_artifact"),
        ("tests/test_rzip_reference_contracts.py", "test_external_uri_policy"),
        ("tests/test_rzip_reference_domains.py", "test_signed_vertical_index"),
        ("tests/test_rzip_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_burn_reference.py", "write_burn_reference_report"),
        ("validation/burn_reference_contracts.py", "_is_finite_number"),
        ("tests/test_burn_reference_validation.py", "valid_burn_declaration"),
        ("tests/test_burn_reference_contracts.py", "test_source_presence"),
        ("tests/test_burn_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_volt_second_reference.py", "write_volt_second_reference_report"),
        ("validation/volt_second_reference_contracts.py", "_is_finite_number"),
        ("tests/test_volt_second_reference_validation.py", "valid_volt_second_declaration"),
        ("tests/test_volt_second_reference_contracts.py", "test_source_presence"),
        ("tests/test_volt_second_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_ntm_reference.py", "write_ntm_reference_report"),
        ("validation/ntm_reference_contracts.py", "_validate_artifact"),
        ("validation/ntm_reference_domains.py", "_is_finite"),
        ("tests/test_ntm_reference_contracts.py", "declaration"),
        ("tests/test_ntm_reference_domains.py", "test_grid_domains"),
        ("tests/test_ntm_reference_command.py", "test_actual_cold_script"),
        ("validation/reference_uri.py", "external_executable_path_error"),
        ("validation/reference_uri.py", "_has_control_characters"),
        ("tests/test_reference_uri.py", "test_native_public_examples_execute"),
        ("validation/validate_neural_equilibrium_reference.py", "write_neural_equilibrium_reference_report"),
        ("src/scpn_control/cli_reference_paths.py", "convert"),
        ("validation/validate_gk_crosscode.py", "write_gk_crosscode_report"),
        ("validation/validate_gk_interface_artifacts.py", "write_gk_interface_artifacts_report"),
        ("validation/gk_interface_reference_contracts.py", "canonical_artifact_sha256"),
        ("tests/test_gk_interface_artifact_validation.py", "_valid_real_executable_artifact"),
        ("tests/test_gk_interface_reference_contracts.py", "declaration"),
        ("tests/test_gk_interface_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_gk_ood_calibration.py", "write_gk_ood_calibration_report"),
        ("validation/gk_ood_reference_contracts.py", "_validate_artifact"),
        ("validation/gk_ood_reference_domains.py", "_is_number"),
        ("tests/test_gk_ood_calibration_validation.py", "_valid_calibration_report"),
        ("tests/test_gk_ood_reference_contracts.py", "declaration"),
        ("tests/test_gk_ood_reference_domains.py", "test_finite_positive_thresholds"),
        ("tests/test_gk_ood_reference_command.py", "test_actual_cold_script"),
        ("validation/gk_crosscode_reference_contracts.py", "_is_finite_number"),
        ("tests/test_gk_crosscode_validation.py", "_valid_gene_report"),
        ("tests/test_gk_crosscode_reference_contracts.py", "declaration"),
        ("tests/test_gk_crosscode_reference_command.py", "test_actual_cold_script"),
        ("tests/test_neural_reference_operational_refusals.py", "test_fixed_public_decode_findings"),
        ("src/scpn_control/cli_reference_kinetic.py", "validate_gk_crosscode_command"),
        ("src/scpn_control/cli_reference_transport.py", "validate_blob_transport_reference_command"),
        ("src/scpn_control/cli_reference_engineering.py", "validate_current_drive_reference_command"),
        ("src/scpn_control/cli_reference_instabilities.py", "validate_elm_reference_command"),
        ("src/scpn_control/cli_reference_tracking.py", "validate_rzip_reference_command"),
        ("src/scpn_control/cli_reference_static_mu.py", "_run_static_mu_analysis_reference"),
        ("tests/test_cli_reference_families.py", "test_actual_jax_cli_csv_requirements"),
        ("validation/neural_equilibrium_reference_contracts.py", "_validate_artifact"),
        ("validation/neural_equilibrium_reference_metrics.py", "_is_positive_finite"),
        ("tests/test_neural_equilibrium_reference_validation.py", "_valid_pefit_reference_artifact"),
        ("tests/test_neural_equilibrium_reference_contracts.py", "declaration_file"),
        ("tools/check_history_exposure.py", "collect_current_paths"),
        ("tools/check_history_exposure.py", "collect_history_paths"),
        ("tools/check_history_exposure.py", "_git"),
        ("tests/test_check_history_exposure.py", "_history_path"),
        ("tests/mast_efm_zarr_fixtures.py", "dataset_from_reference"),
        ("validation/mast_efm_feature_audit_inputs.py", "inspect_reference_sources"),
        ("validation/mast_efm_feature_audit_reporting.py", "validate_audit_report"),
        ("validation/mast_efm_feature_audit_reporting.py", "validate_audit_dataset_bindings"),
        ("validation/neural_equilibrium_dataset_contracts.py", "ensure_distinct_outputs"),
        ("tests/test_mast_efm_feature_provenance_audit.py", "_write_reference"),
        ("validation/build_mast_efm_neural_equilibrium_dataset.py", "build_dataset"),
        ("validation/neural_equilibrium_dataset_contracts.py", "DatasetInput"),
        ("validation/neural_equilibrium_dataset_features.py", "build_feature_matrix"),
        ("validation/neural_equilibrium_dataset_features.py", "validate_feature_matrix"),
        ("validation/neural_equilibrium_dataset_tensors.py", "load_reference"),
        ("validation/neural_equilibrium_dataset_tensors.py", "load_verified_npz"),
        ("validation/neural_equilibrium_dataset_reporting.py", "validate_dataset_report"),
        ("tests/test_mast_efm_neural_equilibrium_dataset.py", "_write_reference"),
        ("validation/neural_equilibrium_training_inputs.py", "_validate_reports"),
        ("validation/neural_equilibrium_training_arrays.py", "_dataset_metadata"),
        ("validation/neural_equilibrium_training_evidence.py", "validate_training_report"),
        ("validation/neural_equilibrium_training_rendering.py", "write_report"),
        ("tests/test_mast_efm_neural_equilibrium_training.py", "test_actual_canonical_dry_run_and_templates"),
        ("validation/neural_equilibrium_campaign_inputs.py", "read_campaign_dataset_report"),
        ("tests/test_neural_equilibrium_training_campaign_plan.py", "test_public_plan_refuses_failed_acquisition"),
        ("tests/test_public_data_acquisition.py", "test_public_file_custody_refusals"),
        ("tests/test_validate_data_manifests.py", "test_public_directory_refuses_identity_only_acquisition"),
        ("src/scpn_control/core/real_data_manifest.py", "resolve_manifest_artifact"),
        ("tests/test_real_data_manifest.py", "test_public_manifest_refuses_nonidentity_shots"),
        ("tests/test_cli_validate_manifest.py", "test_actual_registered_manifest_cli_reports_decode_refusal"),
        ("tests/test_physics_traceability.py", "test_repository_physics_traceability_records_open_fidelity_gaps"),
        ("validation/generate_physics_traceability_report.py", "generate_physics_traceability_markdown"),
        (
            "tests/test_generate_physics_traceability_report.py",
            "test_generate_physics_traceability_markdown_bounds_public_claims",
        ),
        ("tools/check_generated_traceability.py", "generated_traceability_is_current"),
        ("tests/test_check_generated_traceability.py", "test_generated_traceability_check_passes_for_repository_state"),
        ("validation/jax_gk_parity_contracts.py", "_validate_artifact"),
        ("validation/jax_gk_parity_domains.py", "_finite_json_float"),
        ("validation/jax_gk_parity_summary.py", "_attach_summary_fields"),
        ("tests/test_jax_gk_parity_command.py", "test_actual_cold_script"),
        ("validation/validate_jax_gk_parity.py", "validate_jax_gk_parity"),
        ("tests/test_jax_gk_parity_validation.py", "test_copied_public_declaration_domain_refusals"),
        ("validation/validate_tracker53_evidence.py", "validate_tracker53_evidence"),
        ("tests/test_tracker53_evidence_gate.py", "test_aggregate_actual_decoder_refusals"),
        ("validation/validate_native_formal_certificate_evidence.py", "validate_native_formal_certificate_evidence"),
        ("tests/test_native_formal_certificate_evidence.py", "test_real_native_context_refusals"),
        ("src/scpn_control/cli_evidence_validators.py", "validate"),
        ("tests/test_cli_validate.py", "_fresh_cli"),
        ("tests/test_runtime_admission_evidence_validation.py", "test_public_report_domain_refusals"),
        ("tests/test_release_evidence_validation.py", "test_public_reader_refuses_malformed_declarations"),
        ("validation/synthetic_diagnostics.py", "ece_radiometer"),
        ("validation/machine_inputs.py", "snapshot"),
        ("validation/confinement_reference.py", "evaluate"),
        ("validation/equilibrium_execution.py", "evaluate"),
        ("tests/test_density_reference_validation.py", "test_strict_density_gate_requires_reference_artifacts"),
        ("tests/test_tools/test_api_contracts.py", "test_actual_cli_malformed_registry_refuses"),
        ("tests/test_tools/test_api_contract_inventory.py", "test_actual_root_export_shape_refusal"),
        ("tools/check_runtime_wiring.py", "find_orphans"),
        ("tests/test_tools/test_runtime_wiring.py", "_cli"),
        ("tools/emit_studio_manifest.py", "render"),
        ("tests/test_studio_manifest_artifact.py", "test_actual_cli_write_and_semantic_check"),
        ("tools/sync_studio_web_manifest.py", "read_manifest"),
        ("tests/test_sync_studio_web_manifest.py", "test_real_cli_copies_exact_utf8_crlf_bytes"),
        ("validation/current_drive_reference_contracts.py", "_validate_artifact"),
        ("validation/current_drive_reference_domains.py", "_is_positive_finite"),
        ("validation/report_output_paths.py", "checked_report_destination"),
        ("tools/benchmark_regression_gate.py", "main"),
        ("tools/em_and_dimits.py", "run_jax"),
        ("tools/dimits_256_fixed.py", "run"),
        ("tools/dimits_256_fixed.py", "main"),
        ("tools/dimits_long.py", "main"),
        ("validation/benchmark_density_control_claims.py", "main"),
        ("validation/benchmark_current_drive_claims.py", "main"),
        ("validation/benchmark_burn_control_claims.py", "main"),
        ("validation/benchmark_volt_second_claims.py", "main"),
        ("validation/benchmark_orbit_following_claims.py", "main"),
        ("validation/benchmark_disturbance_rejection.py", "main"),
        ("validation/free_boundary_tracking_acceptance.py", "main"),
        ("validation/free_boundary_acceptance_presets.py", "_build_tracking_template"),
        ("validation/free_boundary_acceptance_presets.py", "_make_coil_kick_disturbance"),
        ("validation/free_boundary_acceptance_evaluations.py", "_combine_evaluations"),
        ("validation/free_boundary_acceptance_evaluations.py", "_evaluate_topology"),
        ("validation/free_boundary_acceptance_sweeps.py", "_run_measurement_sweep"),
        ("validation/free_boundary_acceptance_topology_sweeps.py", "_run_topology_measurement_sweep"),
        ("validation/free_boundary_acceptance_campaign.py", "run_campaign"),
        ("validation/free_boundary_acceptance_reports.py", "generate_report"),
        ("validation/free_boundary_acceptance_reports.py", "render_markdown"),
        ("tests/test_free_boundary_acceptance_commands.py", "_command"),
        ("tests/test_free_boundary_acceptance_commands.py", "observed_report"),
        ("validation/benchmark_disturbance_rejection.py", "cli"),
        ("validation/disturbance_inputs.py", "_finite"),
        ("validation/disturbance_inputs.py", "_state"),
        ("validation/disturbance_controllers.py", "PIDController"),
        ("validation/disturbance_controllers.py", "HInfinityErrorController"),
        ("validation/disturbance_controllers.py", "SNNControllerWrapper"),
        ("validation/disturbance_runtime.py", "run_scenario"),
        ("validation/disturbance_runtime.py", "LinearPlant"),
        ("validation/disturbance_runtime.py", "ScenarioMetrics"),
        ("validation/disturbance_reports.py", "generate_json_results"),
        ("validation/disturbance_reports.py", "save_overlay_plots"),
        ("tests/test_disturbance_runtime.py", "_cfg"),
        ("tests/test_disturbance_commands.py", "_command"),
        ("validation/code_to_code_benchmark.py", "main"),
        ("validation/code_to_code_scenario.py", "validate_scenario"),
        ("validation/code_to_code_scenario.py", "initial_profiles"),
        ("validation/code_to_code_scenario.py", "_torax_config_dict"),
        ("validation/code_to_code_local.py", "run_local_transport"),
        ("validation/code_to_code_torax.py", "write_torax_config"),
        ("validation/code_to_code_torax.py", "_run_torax"),
        ("validation/code_to_code_comparison.py", "compare_transport_profiles"),
        ("validation/code_to_code_comparison.py", "_finite_number"),
        ("validation/code_to_code_reports.py", "build_comparison_report"),
        ("validation/code_to_code_reports.py", "_external_reference_status"),
        ("tests/test_code_to_code_benchmark.py", "_FakeDataTree"),
        ("tests/test_code_to_code_commands.py", "test_actual_code_to_code_command"),
        ("tests/test_code_to_code_profiles.py", "_declared_reference"),
        ("validation/benchmark_kuramoto_runtime_evidence.py", "_deterministic_case"),
        ("validation/benchmark_kuramoto_runtime_evidence.py", "main"),
        ("validation/control_resilience_campaign.py", "generate_campaign_report"),
        ("validation/control_resilience_campaign.py", "main"),
        ("validation/resilience_campaign_inputs.py", "_normalize_campaign_inputs"),
        ("validation/disruption_roc_analysis.py", "generate_scenario_batch"),
        ("validation/disruption_roc_analysis.py", "evaluate_batch"),
        ("validation/disruption_roc_analysis.py", "main"),
        ("tests/campaign_command_observation.py", "observe_command"),
        ("tests/test_kuramoto_evidence_command.py", "test_actual_existing_native_claim_and_target_refusal"),
        ("tests/test_control_resilience_commands.py", "test_actual_metrics_and_reports_precede_strict_exit"),
        ("tests/test_disruption_roc_analysis_command.py", "generated_shots"),
        ("validation/mesh_convergence_study.py", "run_solovev_benchmark"),
        ("validation/validate_differentiable_transport_latency.py", "validate_differentiable_transport_latency"),
        ("validation/differentiable_latency_fields.py", "_require_value"),
        ("validation/differentiable_latency_audit.py", "_validate_indices"),
        ("validation/differentiable_latency_context.py", "_validate_runtime_metadata"),
        ("validation/differentiable_latency_reports.py", "_validate_report"),
        ("validation/differentiable_latency_readiness.py", "_validate_readiness_report"),
        ("tests/differentiable_latency_observation.py", "observed_transport_reports"),
        ("tests/test_differentiable_latency_contracts.py", "test_bad_one_step_is_fail_and_never_counted_as_admitted"),
        ("tests/test_differentiable_latency_command.py", "test_actual_differentiable_latency_command"),
        ("tests/test_mesh_convergence_commands.py", "test_actual_completed_sweep_count"),
        ("validation/validate_e2e_latency_evidence.py", "validate_e2e_latency_evidence"),
        ("validation/e2e_latency_payload.py", "_validate_percentiles"),
        ("validation/e2e_latency_context.py", "_valid_utc_timestamp"),
        ("tests/e2e_latency_observation.py", "measured_report"),
        ("tests/test_e2e_latency_command.py", "test_actual_public_latency_command"),
        ("tests/test_e2e_latency_contracts.py", "test_invalid_budget_refused_before_report_io"),
        ("validation/validate_scpn_lean_formal.py", "validate_lean_formal_evidence"),
        ("validation/validate_scpn_lean_formal.py", "_result_payload"),
        ("validation/validate_scpn_z3_formal.py", "publish_report"),
        ("validation/validate_scpn_z3_formal.py", "_write_blocked"),
        ("tests/formal_validator_declaration_fixtures.py", "declared_lean_case"),
        ("tests/test_lean_formal_validation_command.py", "test_actual_named_report_and_artifact_must_agree"),
        ("tests/test_z3_formal_publisher_command.py", "test_actual_bounded_z3_publisher_preserves_proof_scope"),
        ("validation/benchmark_uq_claims.py", "build_reference_scenario"),
        ("validation/benchmark_uq_claims.py", "main"),
        ("validation/benchmark_kinetic_efit_claims.py", "build_reference_case"),
        ("validation/benchmark_kinetic_efit_claims.py", "main"),
        ("validation/benchmark_free_boundary_tracking_claims.py", "_sample_flux_at_points"),
        (
            "tests/test_equilibrium_claim_producer_commands.py",
            "test_actual_recorded_public_producer_preserves_bounded_claims",
        ),
        ("validation/benchmark_free_boundary.py", "main"),
        ("tests/test_free_boundary_benchmark_command.py", "test_actual_public_writer_does_not_admit_off_axis_field"),
        (
            "tests/test_scalar_claim_producer_commands.py",
            "test_actual_recorded_public_producer_preserves_bounded_claims",
        ),
        (
            "tests/test_particle_claim_producer_commands.py",
            "test_actual_recorded_public_producer_preserves_bounded_claims",
        ),
        ("tests/test_em_and_dimits.py", "test_empty_actual_cpu_history_is_refused"),
        ("tools/benchmark_gate_policy.py", "validate_report"),
        ("tools/benchmark_gate_verdict.py", "gate"),
        ("tests/test_benchmark_regression_gate.py", "test_main_passes_on_clean_run"),
        ("tests/test_benchmark_regression_gate_command.py", "test_real_command_preserves_each_selected_input_alias"),
        ("tests/test_report_output_paths.py", "test_public_path_check_refuses_real_aliases"),
        ("tests/current_drive_declaration_fixtures.py", "declaration"),
        ("tests/test_current_drive_reference_command.py", "test_actual_cold_script"),
        ("validation/validate_current_drive_reference.py", "validate_current_drive_reference"),
        ("tests/test_current_drive_reference_validation.py", "test_decoder_refusals_use_authored_text"),
        ("validation/validate_benchmark_regression_gates.py", "validate_benchmark_regression_gates"),
        ("tests/test_benchmark_regression_gates.py", "test_repository_benchmark_regression_manifest_is_admitted"),
        ("tests/test_benchmark_regression_cli.py", "test_actual_relative_cli_admits_copied_historical_metadata"),
        ("tools/check_changelog_sync.py", "main"),
        ("tools/check_commit_authorship.py", "main"),
        ("tools/capability_manifest_inventory.py", "build_manifest"),
        ("tools/capability_manifest_rendering.py", "write_outputs"),
        ("tools/check_docstring_debt.py", "read_ceiling"),
        ("tests/test_docstring_debt_ratchet.py", "_cli"),
        ("tools/check_docs_internal_private.py", "check_mkdocs_excludes_internal"),
        ("tests/test_docs_internal_private_guard.py", "test_real_index_and_noncurrent_history_paths_cli"),
        ("tools/check_joss_submission.py", "check_repository"),
        ("tests/test_joss_submission_check.py", "test_real_joss_blank_required_inputs"),
        ("tools/check_rust_toolchain_contract.py", "check_rust_toolchain_contract"),
        ("tests/test_tools/test_rust_toolchain_contract.py", "test_real_policy_action_input_binding"),
        ("tools/check_coverage_pragmas.py", "find_unreasoned_pragmas"),
        ("tests/test_tools/test_pragma_reason_gate.py", "test_real_marker_reason_suffix"),
        ("tools/check_source_headers.py", "load_policy"),
        ("tests/test_source_header_policy.py", "test_policy_classifies_every_live_tracked_path"),
        ("tests/test_source_header_command.py", "test_public_loader_refuses_non_string_arrays"),
        ("tools/check_test_module_linkage.py", "collect_unlinked_modules"),
        ("tests/test_check_test_module_linkage.py", "test_only_reachable_helpers_link_owner"),
        ("tests/test_tools/test_module_linkage_command.py", "test_real_lexical_scope_and_reachability"),
        ("tools/coverage_exception_ledger.py", "build_ledger"),
        ("tests/test_tools/test_exception_ledger_contract.py", "test_live_coverage_exception_inventory_is_complete"),
        ("tests/test_tools/test_exception_ledger_command.py", "test_rule_required_metadata_is_never_coerced"),
    ],
)
def test_ordinary_cli_enforces_maintained_guard_docs(copied_gate_graph: Path, owner: str, symbol: str) -> None:
    """The ordinary CI command rejects a removed real function/class/test contract without extra flags."""
    target = copied_gate_graph / owner
    tree = ast.parse(target.read_text())
    selected = next(
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol
    )
    assert ast.get_docstring(selected)
    selected.body.pop(0)
    target.write_text(ast.unparse(tree))
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py")],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert symbol in result.stderr and str(target) in result.stderr


def test_ordinary_cli_enforces_validation_package_contract(copied_gate_graph: Path) -> None:
    """The actual default gate refuses a missing contract on the maintained validation package."""
    target = copied_gate_graph / "validation/__init__.py"
    tree = ast.parse(target.read_text())
    assert ast.get_docstring(tree)
    tree.body.pop(0)
    target.write_text(ast.unparse(tree), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py")],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert str(target) in result.stderr and "module" in result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("total", True),
        ("total", False),
        ("total", -1),
        ("total", 0.5),
        ("total", "0"),
        ("total", None),
        ("per_module", []),
        ("per_module", {"x": True}),
        ("per_module", {"x": 1.9}),
        ("per_module", {"x": "1"}),
        ("per_module", {"x": -1}),
        ("per_module", {"x": None}),
        ("per_module", {"x": 1}),
        ("per_module", {"": 0}),
        ("per_module", {1: 0}),
        ("rules", ["D999"]),
        ("rules", list(reversed(rdg.COVERAGE_RULES))),
    ],
)
def test_public_ledger_decoder_refuses_invalid_counts_and_rules(field: str, value: object) -> None:
    """The actual JSON-value API refuses coercion, malformed labels and incoherent counts."""
    payload: dict[str, object] = {"schema": rdg.LEDGER_SCHEMA, "total": 0, "per_module": {}}
    payload[field] = value
    with pytest.raises(ValueError):
        rdg.DocstringDebtLedger.from_dict(payload)


def test_public_legacy_ledger_roundtrip_retains_exact_counts_and_rules(tmp_path: Path) -> None:
    """Legacy omitted rules remain readable, and native writes bind the current rules."""
    payload: dict[str, object] = {"schema": rdg.LEDGER_SCHEMA, "total": 3, "per_module": {"z": 1, "a": 2}}
    ledger = rdg.DocstringDebtLedger.from_dict(payload)
    target = tmp_path / "ledger.json"
    rdg.write_ledger(ledger, target)
    assert rdg.load_ledger(target) == ledger
    saved = json.loads(target.read_text())
    assert saved["rules"] == list(rdg.COVERAGE_RULES)
    assert list(saved["per_module"]) == ["a", "z"]
    assert saved["total"] == 3


def test_public_ledger_writer_refuses_invalid_constructor_before_overwrite(tmp_path: Path) -> None:
    """An unchecked Python value cannot overwrite an existing valid count snapshot."""
    target = tmp_path / "ledger.json"
    rdg.write_ledger(rdg.DocstringDebtLedger(0, {}), target)
    original = target.read_bytes()
    with pytest.raises(ValueError):
        rdg.write_ledger(rdg.DocstringDebtLedger(True, {}), target)
    assert target.read_bytes() == original


@pytest.mark.parametrize("flags", [[], ["--update-baseline"], ["--update-baseline", "--allow-baseline-increase"]])
@pytest.mark.parametrize(
    "fault", ["boolean", "float_count", "negative_count", "wrong_rules", "duplicate", "list", "utf8"]
)
def test_actual_cli_refuses_malformed_ledger_without_overwrite(
    copied_gate_graph: Path, fault: str, flags: list[str]
) -> None:
    """Real AST/Ruff inspection cannot admit or rewrite malformed persisted counts."""
    target = copied_gate_graph / "tools/docstring_debt.json"
    payload: dict[str, object] = {"schema": rdg.LEDGER_SCHEMA, "total": 0, "per_module": {}}
    if fault == "boolean":
        payload["total"] = True
    elif fault == "float_count":
        payload.update(total=1, per_module={"x": 1.9})
    elif fault == "negative_count":
        payload["per_module"] = {"x": -1}
    elif fault == "wrong_rules":
        payload["rules"] = ["D999"]
    if fault == "duplicate":
        target.write_text('{"schema": "scpn-control.docstring-debt.v1", "total": 0, "total": 0, "per_module": {}}')
    elif fault == "list":
        target.write_text("[]")
    elif fault == "utf8":
        target.write_bytes(b"\xff")
    else:
        target.write_text(json.dumps(payload))
    original = target.read_bytes()
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py"), *flags],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stderr == "[docstrings] FAILED: docstring debt ledger could not be read.\n"
    assert target.read_bytes() == original
    assert "baseline updated" not in result.stdout


@pytest.mark.parametrize("fault", ["missing", "config", "unlisted_syntax", "extra_syntax"])
def test_actual_cli_inspection_failure_uses_authored_refusal(copied_gate_graph: Path, fault: str) -> None:
    """Real missing files, invalid configuration and source syntax refuse without exception text."""
    args: list[str] = []
    if fault == "missing":
        target = copied_gate_graph / "tools/document_link_audit.py"
        target.rename(target.with_suffix(".saved"))
    elif fault == "config":
        (copied_gate_graph / "pyproject.toml").write_text("[tool.ruff\n")
    elif fault == "unlisted_syntax":
        (copied_gate_graph / "src/scpn_control/invalid_doc_gate_source.py").write_text("def broken(\n")
    else:
        extra = copied_gate_graph / "extra.py"
        extra.write_text("def broken(\n")
        args = ["--all-definitions", str(extra)]
    ledger = copied_gate_graph / "tools/docstring_debt.json"
    original = ledger.read_bytes()
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py"), *args],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    expected = (
        "Ruff docstring probe failed."
        if fault in ("config", "unlisted_syntax")
        else "all-definition probe could not be read."
    )
    assert result.stderr == "[docstrings] FAILED: " + expected + "\n"
    assert ledger.read_bytes() == original


def test_actual_probe_refuses_missing_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public probe reports an actual native executable absence without raw OS text."""
    missing = tmp_path / "absent-interpreter"
    assert not missing.exists()
    monkeypatch.setattr(sys, "executable", str(missing))
    with pytest.raises(RuntimeError, match="^ruff docstring probe could not launch or decode$"):
        rdg.run_ruff_probe()


@pytest.mark.parametrize("flags", [[], ["--update-baseline"], ["--update-baseline", "--allow-baseline-increase"]])
def test_actual_cli_hard_zero_refuses_unlisted_public_debt(copied_gate_graph: Path, flags: list[str]) -> None:
    """Native Ruff finds new public classes outside the AST list; updates cannot waive the zero floor."""
    source = copied_gate_graph / "src/scpn_control/unlisted_public_api.py"
    source.write_text(
        '"""An additional source file inspected by the real package-wide Ruff probe."""\nclass Undocumented:\n    pass\n'
    )
    ledger = copied_gate_graph / "tools/docstring_debt.json"
    original = ledger.read_bytes()
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py"), *flags],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == (3 if flags else 1), result.stdout + result.stderr
    assert "scpn_control.unlisted_public_api" in result.stdout
    assert (
        "hard-zero gate is active" if flags else "missing public-API docstrings: 1"
    ) in result.stdout + result.stderr
    assert ledger.read_bytes() == original


@pytest.mark.parametrize("existing", [False, True])
def test_actual_cli_update_records_clean_native_probe(copied_gate_graph: Path, existing: bool) -> None:
    """An actual zero probe creates or tightens a valid persisted ledger without changing the rule sequence."""
    ledger = copied_gate_graph / "tools/docstring_debt.json"
    if existing:
        ledger.write_text(json.dumps({"schema": rdg.LEDGER_SCHEMA, "total": 2, "per_module": {"historical.only": 2}}))
    else:
        ledger.unlink()
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py"), "--update-baseline"],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "baseline updated" in result.stdout
    assert json.loads(ledger.read_text()) == rdg.DocstringDebtLedger(0, {}).to_dict()


def test_actual_main_refuses_native_write_to_missing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public Python entry point runs real inspections before returning a fixed filesystem-write refusal."""
    ledger = tmp_path / "missing-parent" / "ledger.json"
    monkeypatch.setattr(rdg, "LEDGER_PATH", ledger)
    assert rdg.main(["--update-baseline"]) == 2
    assert capsys.readouterr().err == "[docstrings] FAILED: docstring debt ledger could not be written.\n"
    assert not ledger.exists() and not ledger.parent.exists()


def test_public_file_to_module_accepts_non_src_label() -> None:
    """The public path mapper retains a non-src prefix instead of inventing a package root."""
    assert rdg.file_to_module("tools/run_docstring_gate.py") == "tools.run_docstring_gate"


@pytest.mark.parametrize("baseline_debt", [0, 2])
def test_actual_cli_clean_probe_retains_or_improves_baseline(copied_gate_graph: Path, baseline_debt: int) -> None:
    """The configured zero-floor command passes real clean sources and reports historical improvement without writing."""
    ledger = copied_gate_graph / "tools/docstring_debt.json"
    counts = {"historical.only": baseline_debt} if baseline_debt else {}
    ledger.write_text(json.dumps({"schema": rdg.LEDGER_SCHEMA, "total": baseline_debt, "per_module": counts}))
    original = ledger.read_bytes()
    result = subprocess.run(
        [sys.executable, str(copied_gate_graph / "tools/run_docstring_gate.py")],
        cwd=copied_gate_graph,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"OK: missing-docstring debt within baseline (0 <= {baseline_debt})" in result.stdout
    assert ("debt fell by 2" in result.stdout) is bool(baseline_debt)
    assert ledger.read_bytes() == original


@pytest.mark.parametrize(
    "payload,refusal",
    [
        ("  \n", "ruff produced no JSON output"),
        ("{", "ruff output could not be decoded as JSON"),
        ("{}", "ruff JSON output was not an array of diagnostics"),
        ("[1]", "ruff diagnostic requires an object with a string filename"),
        ('[{"filename": 1, "code": "D101"}]', "ruff diagnostic requires an object with a string filename"),
        ('[{"filename": "x.py"}]', "ruff docstring probe returned an unsupported diagnostic"),
        ('[{"filename": "x.py", "code": "D999"}]', "ruff docstring probe returned an unsupported diagnostic"),
    ],
)
def test_public_ruff_json_decoder_refuses_invalid_payload(payload: str, refusal: str) -> None:
    """The public pure decoder rejects malformed wire shapes with authored messages; this is no provider claim."""
    with pytest.raises(RuntimeError) as error:
        rdg.parse_ruff_diagnostics(payload)
    assert str(error.value) == refusal


def test_public_ruff_json_decoder_retains_selected_diagnostics_and_extra_fields() -> None:
    """Literal JSON decoding preserves all selected rules and opaque native fields without asserting file existence."""
    expected = [
        {"filename": "not-an-existing-source.py", "code": code, "location": {"row": 1}} for code in rdg.COVERAGE_RULES
    ]
    assert rdg.parse_ruff_diagnostics(json.dumps(expected)) == expected
    assert rdg.parse_ruff_diagnostics(" \n [] \n") == []


def test_public_ruff_json_decoder_maps_native_integer_limit_refusal() -> None:
    """An actual interpreter conversion limit maps to the authored decoder refusal and is restored afterward."""
    previous = sys.get_int_max_str_digits()
    limit = sys.int_info.str_digits_check_threshold
    try:
        sys.set_int_max_str_digits(limit)
        payload = '[{"filename":"x.py","code":"D101","opaque":' + "9" * (limit + 1) + "}]"
        with pytest.raises(RuntimeError) as error:
            rdg.parse_ruff_diagnostics(payload)
        assert str(error.value) == "ruff output could not be decoded as JSON"
    finally:
        sys.set_int_max_str_digits(previous)
    assert sys.get_int_max_str_digits() == previous


@pytest.mark.parametrize("case", ["increase_refused", "increase_allowed", "total_regression", "flat_total_regression"])
def test_actual_historical_python_policy_uses_native_ruff(copied_gate_graph: Path, case: str) -> None:
    """A cloned historical Python policy uses actual Ruff counts; the configured CLI's zero floor stays unchanged."""
    source = copied_gate_graph / "src/scpn_control/unlisted_public_api.py"
    source.write_text('"""A real additional package source inspected by Ruff."""\nclass Undocumented:\n    pass\n')
    ledger = copied_gate_graph / "tools/docstring_debt.json"
    baseline = {"historical.only": 1} if case == "flat_total_regression" else {}
    ledger.write_text(
        json.dumps({"schema": rdg.LEDGER_SCHEMA, "total": sum(baseline.values()), "per_module": baseline})
    )
    before = ledger.read_bytes()
    spec = importlib.util.spec_from_file_location(
        "historical_docstring_policy", copied_gate_graph / "tools/run_docstring_gate.py"
    )
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    assert gate.ENFORCE_ZERO is True and rdg.ENFORCE_ZERO is True
    vars(gate)["ENFORCE_ZERO"] = False
    flags = ["--update-baseline"] if case.startswith("increase_") else []
    if case == "increase_allowed":
        flags.append("--allow-baseline-increase")
    result = gate.main(flags)
    assert result == (0 if case == "increase_allowed" else 3 if case == "increase_refused" else 1)
    assert rdg.ENFORCE_ZERO is True
    if case == "increase_allowed":
        saved = json.loads(ledger.read_text())
        assert saved["total"] == 1 and saved["per_module"] == {"scpn_control.unlisted_public_api": 1}
    else:
        assert ledger.read_bytes() == before
