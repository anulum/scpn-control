# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-API docstring coverage gate with per-module debt accounting
"""Enforce zero public docstring debt and explicit all-definition owner contracts.

``tools/check_docs_coverage.py`` separately checks module docstring presence and
lexical ``docs/api.md`` coverage. This gate checks the next layer: every public
**class**, **method**, and **function** in SOURCE_TARGET must carry a
docstring. The current floor is zero. The historical per-module ratchet remains
available to Python callers; no CLI flag disables ``ENFORCE_ZERO``.

Detection is delegated to ruff's pydocstyle rules so the gate reuses a
maintained, well-tested implementation rather than re-deriving public-API
discovery from the AST:

* ``D101`` — undocumented public class
* ``D102`` — undocumented public method
* ``D103`` — undocumented public function
* ``D104`` — undocumented public package (``__init__.py``)
* ``D106`` — undocumented public nested class

``D105`` (magic methods) and ``D107`` (``__init__``) are deliberately excluded:
the NumPy docstring convention documents ``__init__`` in the class docstring and
does not require docstrings on dunder methods, so requiring them would invite
filler. The main ruff configuration mirrors that decision: ``pyproject.toml``
selects the ``D`` rule set under ``convention = "numpy"`` and ignores ``D105`` /
``D107`` for the same reason, so the docstring *quality* rules (``D2xx`` /
``D4xx``) are enforced there while this gate owns the *coverage* floor above.

The committed ledger ``tools/docstring_debt.json`` records the remaining count
per module. On every run the current count must stay equal or fall, and no
module may exceed its recorded count — mirroring ``run_mypy_strict.py``. The
ledger is rewritten only with ``--update-baseline``, which refuses to raise the
recorded total unless ``--allow-baseline-increase`` is also given.

The migration is complete: the recorded total is zero. With ``ENFORCE_ZERO`` set
the gate is now a hard floor — any missing public-API docstring fails the run
regardless of the ledger, and ``--update-baseline`` refuses to record a non-zero
total even with ``--allow-baseline-increase``. The per-module ledger is retained
for diagnostic reporting.

A separate AST probe enforces every definition in ALL_DEFINITION_TARGETS,
including private/nested callables and test helpers. Historical files outside
that explicit scope retain the public-API gate; their private/test debt is not
claimed cleared. Extra files can be checked with --all-definitions.

The script-relative root owns default files and the Ruff working directory;
additional relative AST paths resolve from caller cwd. Read/update ledger JSON
requires unique keys, nonnegative integer counts excluding booleans, matching
total/module sums and matching rules when that legacy-optional field is present.
This is presence accounting, not semantic or scientific admission. Ledger writes
replace text directly without a source-content digest or atomic publication.
The AST stage precedes Ruff and both precede ledger reads/writes. Refusals use
status one for missing docs, two for inspection/ledger failure and three for an
attempted baseline increase; argparse retains its own zero/two exits.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TARGET = "src/scpn_control/"
LEDGER_PATH = REPO_ROOT / "tools" / "docstring_debt.json"
LEDGER_SCHEMA = "scpn-control.docstring-debt.v1"
COVERAGE_RULES = ("D101", "D102", "D103", "D104", "D106")

# The migration reached zero missing docstrings; the gate now enforces that as a
# hard floor rather than a movable ratchet baseline.
ENFORCE_ZERO = True

# All definitions in these transport and gate owners are covered, including
# private/nested functions, constructors and test helpers. This does not claim
# that historical private/test definitions elsewhere have reached zero debt.
ALL_DEFINITION_TARGETS = (
    "tools/document_link_audit.py",
    "tests/test_tools/test_document_link_audit.py",
    "tools/run_fuzz_campaign.py",
    "tests/test_fuzz_campaign.py",
    "tools/kinetic_e_dual_test.py",
    "tools/kinetic_electron_test.py",
    "tools/sugama_comparison.py",
    "tools/train_neural_transport_qlknn.py",
    "tests/test_train_neural_transport_qlknn.py",
    "tools/run_python_preflight.py",
    "tests/test_run_python_preflight.py",
    "tools/stress_test_oscillators.py",
    "tests/test_stress_test_oscillators.py",
    "tools/train_rl_tokamak.py",
    "tools/rl_training_config.py",
    "tools/rl_tokamak_evaluation.py",
    "tools/rl_training_results.py",
    "benchmarks/rl_vs_classical.py",
    "examples/tutorial_03_ppo_rl_agent.py",
    "tests/test_rl_tokamak.py",
    "tests/test_rl_training_workflow.py",
    "tests/test_rl_training_results.py",
    "tests/test_rl_benchmark_command.py",
    "tests/test_rl_tutorial_command.py",
    "tests/test_rl_benchmark_public_claims.py",
    "tools/jarvislabs_train.py",
    "tools/jarvislabs_client.py",
    "tools/jarvislabs_transport.py",
    "tests/test_jarvislabs_train.py",
    "tests/test_jarvislabs_transport.py",
    "tools/gpu_cbc_benchmark.py",
    "tests/test_gpu_cbc_benchmark.py",
    "tools/gk_convergence_benchmark.py",
    "tests/test_gk_convergence_benchmark.py",
    "tools/generate_phase_video.py",
    "tools/phase_video_model.py",
    "tools/phase_video_rendering.py",
    "tests/test_phase_video_model.py",
    "tests/test_generate_phase_video.py",
    "tools/export_zenodo_dataset.py",
    "tests/test_export_zenodo_dataset.py",
    "validation/free_boundary_tracking_acceptance.py",
    "validation/free_boundary_acceptance_presets.py",
    "validation/free_boundary_acceptance_evaluations.py",
    "validation/free_boundary_acceptance_sweeps.py",
    "validation/free_boundary_acceptance_topology_sweeps.py",
    "validation/free_boundary_acceptance_campaign.py",
    "validation/free_boundary_acceptance_reports.py",
    "tests/test_free_boundary_acceptance_commands.py",
    "validation/benchmark_disturbance_rejection.py",
    "validation/disturbance_inputs.py",
    "validation/disturbance_controllers.py",
    "validation/disturbance_runtime.py",
    "validation/disturbance_reports.py",
    "tests/test_disturbance_runtime.py",
    "tests/test_disturbance_commands.py",
    "validation/code_to_code_benchmark.py",
    "validation/code_to_code_scenario.py",
    "validation/code_to_code_local.py",
    "validation/code_to_code_torax.py",
    "validation/code_to_code_comparison.py",
    "validation/code_to_code_reports.py",
    "tests/test_code_to_code_benchmark.py",
    "tests/test_code_to_code_commands.py",
    "tests/test_code_to_code_profiles.py",
    "validation/benchmark_kuramoto_runtime_evidence.py",
    "validation/control_resilience_campaign.py",
    "validation/disruption_roc_analysis.py",
    "tests/campaign_command_observation.py",
    "tests/test_kuramoto_evidence_command.py",
    "tests/test_control_resilience_commands.py",
    "tests/test_disruption_roc_analysis_command.py",
    "validation/resilience_campaign_inputs.py",
    "validation/mesh_convergence_study.py",
    "tests/test_mesh_convergence_commands.py",
    "validation/validate_differentiable_transport_latency.py",
    "validation/differentiable_latency_fields.py",
    "validation/differentiable_latency_audit.py",
    "validation/differentiable_latency_context.py",
    "validation/differentiable_latency_reports.py",
    "validation/differentiable_latency_readiness.py",
    "tests/differentiable_latency_observation.py",
    "tests/test_differentiable_latency_contracts.py",
    "tests/test_differentiable_latency_command.py",
    "validation/validate_e2e_latency_evidence.py",
    "validation/e2e_latency_payload.py",
    "validation/e2e_latency_context.py",
    "tests/e2e_latency_observation.py",
    "tests/test_e2e_latency_command.py",
    "tests/test_e2e_latency_contracts.py",
    "validation/validate_scpn_lean_formal.py",
    "validation/validate_scpn_z3_formal.py",
    "tests/formal_validator_declaration_fixtures.py",
    "tests/test_lean_formal_validation_command.py",
    "tests/test_z3_formal_publisher_command.py",
    "validation/benchmark_uq_claims.py",
    "validation/benchmark_kinetic_efit_claims.py",
    "validation/benchmark_free_boundary_tracking_claims.py",
    "tests/test_equilibrium_claim_producer_commands.py",
    "validation/benchmark_free_boundary.py",
    "tests/test_free_boundary_benchmark_command.py",
    "validation/benchmark_burn_control_claims.py",
    "validation/benchmark_volt_second_claims.py",
    "validation/benchmark_orbit_following_claims.py",
    "tests/test_scalar_claim_producer_commands.py",
    "validation/benchmark_density_control_claims.py",
    "validation/benchmark_current_drive_claims.py",
    "tests/test_particle_claim_producer_commands.py",
    "tools/dimits_256_fixed.py",
    "tools/dimits_long.py",
    "tools/em_and_dimits.py",
    "tests/test_em_and_dimits.py",
    "validation/__init__.py",
    "tools/benchmark_regression_gate.py",
    "tools/benchmark_gate_policy.py",
    "tools/benchmark_gate_verdict.py",
    "tests/test_benchmark_regression_gate.py",
    "tests/test_benchmark_regression_gate_command.py",
    "validation/report_output_paths.py",
    "tests/test_report_output_paths.py",
    "tests/current_drive_declaration_fixtures.py",
    "tools/benchmark_full_stack.py",
    "tests/test_benchmark_full_stack_public.py",
    "validation/current_drive_reference_contracts.py",
    "validation/current_drive_reference_domains.py",
    "tests/test_current_drive_reference_command.py",
    "validation/density_reference_contracts.py",
    "validation/density_reference_domains.py",
    "tests/test_density_reference_command.py",
    "validation/jax_gk_parity_contracts.py",
    "validation/jax_gk_parity_domains.py",
    "validation/jax_gk_parity_summary.py",
    "tests/test_jax_gk_parity_command.py",
    "validation/validate_gk_interface_artifacts.py",
    "validation/gk_interface_reference_contracts.py",
    "tests/test_gk_interface_artifact_validation.py",
    "tests/test_gk_interface_reference_contracts.py",
    "tests/test_gk_interface_reference_command.py",
    "validation/validate_gk_ood_calibration.py",
    "validation/gk_ood_reference_contracts.py",
    "validation/gk_ood_reference_domains.py",
    "tests/test_gk_ood_calibration_validation.py",
    "tests/test_gk_ood_reference_contracts.py",
    "tests/test_gk_ood_reference_domains.py",
    "tests/test_gk_ood_reference_command.py",
    "validation/validate_gk_crosscode.py",
    "validation/gk_crosscode_reference_contracts.py",
    "tests/test_gk_crosscode_validation.py",
    "tests/test_gk_crosscode_reference_contracts.py",
    "tests/test_gk_crosscode_reference_command.py",
    "src/scpn_control/cli_reference_paths.py",
    "src/scpn_control/cli_reference_kinetic.py",
    "src/scpn_control/cli_reference_equilibrium.py",
    "src/scpn_control/cli_reference_transport.py",
    "src/scpn_control/cli_reference_engineering.py",
    "src/scpn_control/cli_reference_instabilities.py",
    "src/scpn_control/cli_reference_tracking.py",
    "src/scpn_control/cli_reference_static_mu.py",
    "tests/test_cli_reference_families.py",
    "src/scpn_control/cli_reference_validators.py",
    "tests/test_neural_reference_operational_refusals.py",
    "validation/validate_static_mu_analysis_reference.py",
    "validation/validate_mu_synthesis_reference.py",
    "validation/static_mu_reference_contracts.py",
    "validation/static_mu_reference_domains.py",
    "tests/test_static_mu_reference_contracts.py",
    "tests/test_static_mu_reference_domains.py",
    "tests/test_static_mu_reference_command.py",
    "validation/validate_elm_reference.py",
    "tests/test_elm_reference_validation.py",
    "validation/elm_reference_contracts.py",
    "validation/elm_reference_domains.py",
    "tests/test_elm_reference_contracts.py",
    "tests/test_elm_reference_domains.py",
    "tests/test_elm_reference_command.py",
    "validation/validate_soc_reference.py",
    "tests/test_soc_reference_validation.py",
    "validation/soc_reference_contracts.py",
    "validation/soc_reference_domains.py",
    "tests/test_soc_reference_contracts.py",
    "tests/test_soc_reference_domains.py",
    "tests/test_soc_reference_command.py",
    "validation/validate_neural_turbulence_reference.py",
    "tests/test_neural_turbulence_reference_validation.py",
    "validation/neural_turbulence_reference_contracts.py",
    "validation/neural_turbulence_reference_domains.py",
    "tests/test_neural_turbulence_reference_contracts.py",
    "tests/test_neural_turbulence_reference_domains.py",
    "tests/test_neural_turbulence_reference_command.py",
    "validation/validate_gk_species_reference.py",
    "tests/test_gk_species_reference_validation.py",
    "validation/gk_species_reference_contracts.py",
    "validation/gk_species_reference_cases.py",
    "validation/gk_species_reference_operators.py",
    "validation/gk_species_reference_numeric.py",
    "tests/test_gk_species_reference_contracts.py",
    "tests/test_gk_species_reference_domains.py",
    "tests/test_gk_species_reference_operators.py",
    "tests/test_gk_species_reference_command.py",
    "src/scpn_control/core/gk_species.py",
    "validation/gk_geometry_independent_reference.py",
    "validation/validate_gk_geometry_independent.py",
    "validation/validate_gk_geometry_reference.py",
    "validation/gk_geometry_reference_contracts.py",
    "validation/gk_geometry_reference_cases.py",
    "tests/test_gk_geometry_reference_validation.py",
    "tests/test_gk_geometry_reference_contracts.py",
    "tests/test_gk_geometry_reference_domains.py",
    "tests/test_gk_geometry_reference_command.py",
    "validation/validate_neural_transport_reference.py",
    "validation/neural_transport_reference_contracts.py",
    "validation/neural_transport_reference_domains.py",
    "tests/test_neural_transport_reference_validation.py",
    "tests/test_neural_transport_reference_contracts.py",
    "tests/test_neural_transport_reference_domains.py",
    "tests/test_neural_transport_reference_command.py",
    "validation/validate_blob_transport_reference.py",
    "validation/blob_transport_reference_contracts.py",
    "validation/blob_transport_reference_domains.py",
    "tests/test_blob_transport_reference_validation.py",
    "tests/test_blob_transport_reference_contracts.py",
    "tests/test_blob_transport_reference_domains.py",
    "tests/test_blob_transport_reference_command.py",
    "validation/validate_free_boundary_reference.py",
    "validation/free_boundary_reference_contracts.py",
    "tests/test_free_boundary_reference_contracts.py",
    "tests/test_free_boundary_reference_domains.py",
    "tests/test_free_boundary_reference_command.py",
    "validation/validate_digital_twin_reference.py",
    "validation/digital_twin_reference_contracts.py",
    "validation/digital_twin_reference_domains.py",
    "tests/test_digital_twin_reference_validation.py",
    "tests/test_digital_twin_reference_contracts.py",
    "tests/test_digital_twin_reference_domains.py",
    "tests/test_digital_twin_reference_command.py",
    "validation/validate_disruption_reference.py",
    "validation/disruption_reference_contracts.py",
    "tests/test_disruption_reference_validation.py",
    "tests/test_disruption_reference_contracts.py",
    "tests/test_disruption_reference_domains.py",
    "tests/test_disruption_reference_command.py",
    "validation/validate_rzip_reference.py",
    "validation/rzip_reference_contracts.py",
    "tests/test_rzip_reference_validation.py",
    "tests/test_rzip_reference_contracts.py",
    "tests/test_rzip_reference_domains.py",
    "tests/test_rzip_reference_command.py",
    "validation/validate_burn_reference.py",
    "validation/burn_reference_contracts.py",
    "tests/test_burn_reference_validation.py",
    "tests/test_burn_reference_contracts.py",
    "tests/test_burn_reference_command.py",
    "validation/validate_volt_second_reference.py",
    "validation/volt_second_reference_contracts.py",
    "tests/test_volt_second_reference_validation.py",
    "tests/test_volt_second_reference_contracts.py",
    "tests/test_volt_second_reference_command.py",
    "tests/test_reference_output_path_refusal.py",
    "validation/validate_orbit_reference.py",
    "validation/orbit_reference_contracts.py",
    "tests/test_orbit_reference_validation.py",
    "tests/test_orbit_reference_contracts.py",
    "validation/validate_uncertainty_reference.py",
    "validation/uncertainty_reference_contracts.py",
    "tests/test_uncertainty_reference_validation.py",
    "tests/test_uncertainty_reference_contracts.py",
    "validation/validate_vmec_reference.py",
    "validation/vmec_reference_contracts.py",
    "tests/test_vmec_reference_validation.py",
    "tests/test_vmec_reference_contracts.py",
    "validation/validate_eped_reference.py",
    "validation/eped_reference_contracts.py",
    "validation/eped_reference_geometry.py",
    "tests/test_eped_reference_validation.py",
    "tests/test_eped_reference_contracts.py",
    "tests/test_eped_reference_geometry.py",
    "tests/test_eped_reference_command.py",
    "validation/validate_marfe_reference.py",
    "validation/marfe_reference_contracts.py",
    "validation/marfe_reference_domains.py",
    "tests/test_marfe_reference_validation.py",
    "tests/test_marfe_reference_contracts.py",
    "tests/test_marfe_reference_domains.py",
    "tests/test_marfe_reference_command.py",
    "validation/validate_ntm_reference.py",
    "validation/ntm_reference_contracts.py",
    "validation/ntm_reference_domains.py",
    "tests/test_ntm_reference_validation.py",
    "tests/test_ntm_reference_contracts.py",
    "tests/test_ntm_reference_domains.py",
    "tests/test_ntm_reference_command.py",
    "validation/reference_uri.py",
    "tests/test_reference_uri.py",
    "src/scpn_control/core/integrated_transport_solver.py",
    "tests/test_integrated_transport_solver.py",
    "src/scpn_control/core/radial_diffusion.py",
    "tests/test_radial_diffusion.py",
    "src/scpn_control/core/species_evolution.py",
    "tests/test_species_evolution.py",
    "validation/multi_machine_validation.py",
    "validation/synthetic_diagnostics.py",
    "validation/machine_inputs.py",
    "validation/confinement_reference.py",
    "validation/equilibrium_execution.py",
    "tests/test_multi_machine_validation.py",
    "src/scpn_control/core/tglf_flux.py",
    "validation/tglf_launcher.py",
    "tests/test_tglf_launcher.py",
    "tests/child_coverage.py",
    "src/scpn_control/core/tglf_units.py",
    "src/scpn_control/core/tglf_miller.py",
    "tests/test_tglf_miller.py",
    "tests/test_miller_volume_transport.py",
    "tests/test_tglf_numeric_range.py",
    "src/scpn_control/core/transport_flux.py",
    "tests/test_tglf_flux.py",
    "tests/test_tglf_units.py",
    "tests/test_transport_flux.py",
    "tools/check_test_quality_policy.py",
    "tests/test_check_test_quality_policy.py",
    "tools/check_docstring_debt.py",
    "tests/test_docstring_debt_ratchet.py",
    "tools/ci_rmse_gate.py",
    "tests/test_ci_rmse_gate.py",
    "validation/rmse_dashboard.py",
    "validation/rmse_dashboard_metrics.py",
    "validation/rmse_dashboard_rendering.py",
    "validation/rmse_dashboard_command.py",
    "tests/test_rmse_dashboard_rendering.py",
    "tests/test_rmse_dashboard.py",
    "tests/test_cli_validate_rmse.py",
    "tools/check_version_sync.py",
    "tests/test_check_version_sync.py",
    "tests/test_project_metadata.py",
    "tools/check_docs_coverage.py",
    "tests/test_check_docs_coverage.py",
    "tools/check_docs_internal_private.py",
    "tests/test_docs_internal_private_guard.py",
    "tools/check_joss_submission.py",
    "tests/test_joss_submission_check.py",
    "tools/check_rust_toolchain_contract.py",
    "tests/test_tools/test_rust_toolchain_contract.py",
    "tools/check_coverage_pragmas.py",
    "tests/test_tools/test_pragma_reason_gate.py",
    "tools/check_python_lint_contract.py",
    "tests/test_python_lint_contract.py",
    "tools/check_public_surface_hygiene.py",
    "tests/test_public_surface_hygiene.py",
    "tools/check_api_contracts.py",
    "tools/api_contract_inventory.py",
    "tests/test_tools/test_api_contracts.py",
    "tests/test_tools/test_api_contract_inventory.py",
    "tools/check_runtime_wiring.py",
    "tests/test_tools/test_runtime_wiring.py",
    "tools/emit_studio_manifest.py",
    "tests/test_studio_manifest_artifact.py",
    "tools/sync_studio_web_manifest.py",
    "tests/test_sync_studio_web_manifest.py",
    "validation/validate_current_drive_reference.py",
    "tests/test_current_drive_reference_validation.py",
    "validation/validate_density_reference.py",
    "tests/test_density_reference_validation.py",
    "validation/validate_release_evidence.py",
    "tests/test_release_evidence_validation.py",
    "validation/validate_runtime_admission_evidence.py",
    "tests/test_runtime_admission_evidence_validation.py",
    "validation/validate_multi_shot_campaign_evidence.py",
    "tests/test_multi_shot_campaign_evidence_validation.py",
    "validation/validate_physics_traceability.py",
    "validation/validate_data_manifests.py",
    "validation/validate_public_data_acquisition.py",
    "validation/plan_neural_equilibrium_training_campaign.py",
    "validation/train_mast_efm_neural_equilibrium.py",
    "validation/neural_equilibrium_training_inputs.py",
    "validation/neural_equilibrium_training_arrays.py",
    "validation/neural_equilibrium_training_evidence.py",
    "validation/neural_equilibrium_training_rendering.py",
    "validation/audit_mast_efm_feature_provenance.py",
    "validation/mast_efm_feature_audit_inputs.py",
    "validation/mast_efm_feature_audit_reporting.py",
    "tests/test_mast_efm_feature_provenance_audit.py",
    "validation/build_mast_efm_neural_equilibrium_dataset.py",
    "validation/neural_equilibrium_dataset_contracts.py",
    "validation/neural_equilibrium_dataset_features.py",
    "validation/neural_equilibrium_dataset_tensors.py",
    "validation/neural_equilibrium_dataset_reporting.py",
    "tests/test_mast_efm_neural_equilibrium_dataset.py",
    "tests/test_mast_efm_neural_equilibrium_training.py",
    "validation/neural_equilibrium_campaign_inputs.py",
    "tests/test_neural_equilibrium_training_campaign_plan.py",
    "tests/test_public_data_acquisition.py",
    "tests/test_validate_data_manifests.py",
    "src/scpn_control/core/real_data_manifest.py",
    "tests/test_real_data_manifest.py",
    "tests/test_cli_validate_manifest.py",
    "tests/test_physics_traceability.py",
    "validation/generate_physics_traceability_report.py",
    "tests/test_generate_physics_traceability_report.py",
    "tools/check_generated_traceability.py",
    "tests/test_check_generated_traceability.py",
    "validation/validate_jax_gk_parity.py",
    "tests/test_jax_gk_parity_validation.py",
    "validation/validate_tracker53_evidence.py",
    "tests/test_tracker53_evidence_gate.py",
    "validation/validate_native_formal_certificate_evidence.py",
    "tests/test_native_formal_certificate_evidence.py",
    "src/scpn_control/cli_evidence_validators.py",
    "tests/test_cli_validate.py",
    "validation/validate_benchmark_regression_gates.py",
    "tests/test_benchmark_regression_gates.py",
    "tests/test_benchmark_regression_cli.py",
    "tools/check_changelog_sync.py",
    "tests/test_changelog_sync.py",
    "tools/check_commit_authorship.py",
    "tests/test_check_commit_authorship.py",
    "tools/capability_manifest.py",
    "tools/capability_manifest_inventory.py",
    "tools/capability_manifest_rendering.py",
    "tests/test_tools/test_capability_manifest.py",
    "tools/check_source_headers.py",
    "tests/test_source_header_policy.py",
    "tests/test_source_header_command.py",
    "tools/check_test_module_linkage.py",
    "tests/test_check_test_module_linkage.py",
    "tests/test_tools/test_module_linkage_command.py",
    "tools/coverage_exception_ledger.py",
    "tests/test_tools/test_exception_ledger_contract.py",
    "tests/test_tools/test_exception_ledger_command.py",
    "tools/run_docstring_gate.py",
    "tests/test_run_docstring_gate.py",
    "tools/run_mypy_strict.py",
    "tests/test_run_mypy_strict.py",
    "validation/convert_mast_efm_neural_equilibrium_reference.py",
    "validation/mast_efm_reference_arrays.py",
    "validation/mast_efm_reference_contracts.py",
    "tests/test_mast_efm_neural_equilibrium_reference.py",
    "tests/mast_efm_zarr_fixtures.py",
    "validation/audit_mast_efm_original_feature_sources.py",
    "validation/mast_efm_zarr_store.py",
    "validation/mast_efm_original_source_policy.py",
    "validation/mast_efm_original_source_inputs.py",
    "validation/mast_efm_original_source_reporting.py",
    "tests/test_mast_efm_original_feature_source_audit.py",
    "tests/test_mast_efm_original_source_contracts.py",
    "tests/test_mast_efm_zarr_store.py",
    "validation/evaluate_mast_efm_neural_equilibrium.py",
    "validation/mast_efm_evaluation_features.py",
    "validation/mast_efm_evaluation_geometry.py",
    "tests/test_mast_efm_neural_equilibrium_evaluation.py",
    "tests/test_mast_efm_evaluation_features.py",
    "tests/test_mast_efm_evaluation_geometry.py",
    "tools/generate_native_api_reference.py",
    "tests/test_tools/test_generate_native_api_reference.py",
    "tools/check_benchmark_producers.py",
    "tests/test_benchmark_producer_registry.py",
    "validation/validate_neural_equilibrium_reference.py",
    "validation/neural_equilibrium_reference_contracts.py",
    "validation/neural_equilibrium_reference_metrics.py",
    "tests/test_neural_equilibrium_reference_validation.py",
    "tests/test_neural_equilibrium_reference_contracts.py",
    "tools/check_history_exposure.py",
    "tests/test_check_history_exposure.py",
)


def all_definition_findings(paths: list[Path]) -> list[str]:
    """Find undocumented modules, classes and every nested/private callable in explicit files.

    No name, visibility or dunder exemption is applied. Missing files and parse
    failures propagate to the CLI as a failed probe rather than silent coverage.
    """
    findings: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    findings.append(f"{path}:{getattr(node, 'lineno', 1)}: {getattr(node, 'name', '<module>')}")
    return findings


def file_to_module(path: str) -> str:
    """Return the dotted import path for a source file.

    Parameters
    ----------
    path
        A source path as emitted by ruff. Absolute paths and ``src``-relative
        paths are both accepted; everything up to and including ``src/`` is
        stripped.

    Returns
    -------
    str
        The import path, for example ``scpn_control.core.elm_model``.
    """
    normalised = path.replace("\\", "/")
    marker = "src/"
    index = normalised.rfind(marker)
    if index != -1:
        normalised = normalised[index + len(marker) :]
    return normalised.removesuffix(".py").replace("/", ".")


def parse_module_counts(diagnostics: list[dict[str, object]]) -> dict[str, int]:
    """Count missing-docstring diagnostics per module.

    Parameters
    ----------
    diagnostics
        The decoded ruff JSON diagnostics array. Each entry must carry a
        ``filename`` string.

    Returns
    -------
    dict[str, int]
        Mapping of dotted module import path to the number of diagnostics
        attributed to it. Modules with no diagnostics are omitted.
    """
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        filename = diagnostic.get("filename")
        if not isinstance(filename, str):
            raise ValueError("ruff diagnostic missing a string 'filename' field")
        module = file_to_module(filename)
        counts[module] = counts.get(module, 0) + 1
    return counts


def parse_ruff_diagnostics(payload: str) -> list[dict[str, object]]:
    """Decode Ruff's JSON array and validate the selected diagnostic fields.

    Parameters
    ----------
    payload
        Ruff stdout text. Surrounding whitespace is accepted; an empty body,
        invalid JSON, non-array root or unsupported entry is refused.

    Returns
    -------
    list[dict[str, object]]
        Fresh decoded objects carrying string filenames and one of the five
        selected rule codes. Additional diagnostic fields are retained.

    Raises
    ------
    RuntimeError
        Fixed authored refusal for invalid JSON or diagnostic fields. Decoding
        validates shape and rule membership, not the authenticity of stdout or
        the existence of the named files.
    """
    if not payload.strip():
        raise RuntimeError("ruff produced no JSON output")
    try:
        diagnostics = json.loads(payload)
    except ValueError:
        raise RuntimeError("ruff output could not be decoded as JSON") from None
    if not isinstance(diagnostics, list):
        raise RuntimeError("ruff JSON output was not an array of diagnostics")
    checked: list[dict[str, object]] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict) or not isinstance(diagnostic.get("filename"), str):
            raise RuntimeError("ruff diagnostic requires an object with a string filename")
        if diagnostic.get("code") not in COVERAGE_RULES:
            raise RuntimeError("ruff docstring probe returned an unsupported diagnostic")
        checked.append(diagnostic)
    return checked


def run_ruff_probe() -> list[dict[str, object]]:
    """Run the actual package-wide Ruff subprocess and decode its diagnostics.

    Exit zero or one supplies stdout to ``parse_ruff_diagnostics``. Other exits,
    native launch/decoding failures and malformed diagnostic JSON raise a fixed
    authored RuntimeError. The command uses this interpreter and REPO_ROOT;
    extra AST paths never replace the package-wide Ruff source target.

    Returns
    -------
    list[dict[str, object]]
        The validated native diagnostics, including any additional Ruff fields.

    Raises
    ------
    RuntimeError
        Authored refusal for native launch/decoding failure, an unexpected exit
        status or invalid diagnostic JSON.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                ",".join(COVERAGE_RULES),
                "--output-format",
                "json",
                SOURCE_TARGET,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, UnicodeError):
        raise RuntimeError("ruff docstring probe could not launch or decode") from None
    if result.returncode not in (0, 1):
        raise RuntimeError("ruff docstring probe could not run")
    return parse_ruff_diagnostics(result.stdout)


@dataclass(frozen=True)
class DocstringDebtLedger:
    """Committed snapshot of remaining missing-docstring debt.

    Attributes
    ----------
    total
        Total count of missing public-API docstrings across the package.
    per_module
        Mapping of dotted module import path to its recorded missing count.

    Notes
    -----
    Direct construction is an unchecked Python value; the frozen dataclass does
    not freeze its dictionary. JSON reconstruction and writes validate integer
    counts and their sum. Module labels need not be importable Python names.
    """

    total: int
    per_module: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable representation of the ledger."""
        return {
            "schema": LEDGER_SCHEMA,
            "rules": list(COVERAGE_RULES),
            "total": self.total,
            "per_module": dict(sorted(self.per_module.items())),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DocstringDebtLedger:
        """Build a ledger from its JSON representation.

        Raises
        ------
        ValueError
            If the schema marker is absent or unrecognised, or the payload
            count fields are not nonnegative integers, totals disagree, or a
            present rules field differs from the selected rule sequence.
            Booleans and numerical coercion are refused. An omitted rules field
            remains supported for legacy snapshots; other unknown fields are
            ignored. The returned module-count dictionary is a new copy.
        """
        if payload.get("schema") != LEDGER_SCHEMA:
            raise ValueError("unexpected ledger schema")
        per_module = payload.get("per_module", {})
        if not isinstance(per_module, dict):
            raise ValueError("ledger per_module must be an object")
        total = payload.get("total")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError("ledger total must be a nonnegative integer")
        counts: dict[str, int] = {}
        for module, count in per_module.items():
            if not isinstance(module, str) or not module:
                raise ValueError("ledger module labels must be nonempty strings")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("ledger module counts must be nonnegative integers")
            counts[module] = count
        if sum(counts.values()) != total:
            raise ValueError("ledger module counts must sum to total")
        if "rules" in payload and payload["rules"] != list(COVERAGE_RULES):
            raise ValueError("ledger rules must match the selected rules")
        return cls(
            total=total,
            per_module=counts,
        )


@dataclass(frozen=True)
class RatchetResult:
    """Outcome of comparing current docstring debt against the ledger.

    Attributes
    ----------
    regressions
        Modules whose current missing count exceeds the recorded count, mapped
        to ``(recorded, current)`` pairs.
    total_delta
        ``current_total - ledger_total``; negative means debt fell.
    improvements
        Modules whose current count is strictly below the recorded count, mapped
        to ``(recorded, current)`` pairs.
    """

    regressions: dict[str, tuple[int, int]]
    total_delta: int
    improvements: dict[str, tuple[int, int]]

    @property
    def ok(self) -> bool:
        """Return whether the ratchet holds (no regressions, total not risen)."""
        return not self.regressions and self.total_delta <= 0


def evaluate_ratchet(
    current_total: int,
    current_modules: dict[str, int],
    ledger: DocstringDebtLedger,
) -> RatchetResult:
    """Compare current docstring debt against the committed ledger.

    Parameters
    ----------
    current_total
        Total missing-docstring count from this run.
    current_modules
        Per-module missing-docstring counts from this run.
    ledger
        The committed baseline ledger.

    Returns
    -------
    RatchetResult
        The regressions, improvements, and total delta versus the baseline.
    """
    regressions: dict[str, tuple[int, int]] = {}
    improvements: dict[str, tuple[int, int]] = {}
    for module, current in current_modules.items():
        recorded = ledger.per_module.get(module, 0)
        if current > recorded:
            regressions[module] = (recorded, current)
    for module, recorded in ledger.per_module.items():
        current = current_modules.get(module, 0)
        if current < recorded:
            improvements[module] = (recorded, current)
    return RatchetResult(
        regressions=regressions,
        total_delta=current_total - ledger.total,
        improvements=improvements,
    )


def _unique_ledger_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse duplicate JSON object keys at every depth without displaying values."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate docstring ledger key")
        result[key] = value
    return result


def load_ledger(path: Path = LEDGER_PATH) -> DocstringDebtLedger:
    """Read UTF-8 unique-key ledger JSON and validate its count/rule contract.

    The root must be an object. IO/decoding errors propagate; invalid JSON,
    duplicate keys and schema/count violations raise ValueError. No missing-file
    default, numeric coercion or source authenticity claim is supplied.
    """
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_ledger_object)
    if not isinstance(payload, dict):
        raise ValueError("docstring ledger must be an object")
    return DocstringDebtLedger.from_dict(payload)


def write_ledger(ledger: DocstringDebtLedger, path: Path = LEDGER_PATH) -> None:
    """Validate counts before overwriting UTF-8 JSON with a trailing newline.

    Invalid values raise ValueError before opening the destination. Valid writes
    use direct Path.write_text without an atomic rename, fsync or symlink/alias
    protection. Native filesystem errors propagate; callers own path custody.
    """
    payload = ledger.to_dict()
    DocstringDebtLedger.from_dict(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _report_debt(total: int, modules: dict[str, int], *, top: int = 10) -> None:
    """Print the current docstring debt and the worst-offending modules."""
    print(f"[docstrings] remaining missing public-API docstrings: {total} across {len(modules)} modules")
    worst = sorted(modules.items(), key=lambda item: (-item[1], item[0]))[:top]
    for module, count in worst:
        print(f"[docstrings]   {count:>4}  {module}")


def main(argv: list[str] | None = None) -> int:
    """Run the AST/public-doc floor before validating or updating its ledger.

    Parameters
    ----------
    argv
        Optional argument vector; defaults to ``sys.argv`` when ``None``.

    Returns
    -------
    int
        Zero when all selected inspection and ledger checks pass, one for
        missing documentation, two for inspection/ledger refusal and three for
        a refused baseline increase. Argparse help/invalid flags exit zero/two.

    Notes
    -----
    --allow-baseline-increase does not bypass the enabled zero floor. An update
    cannot rewrite an existing malformed ledger. Writes occur only after the
    selected AST/Ruff inspection and valid baseline comparison. Explicit extra
    files are added to default owners, never substituted for them.
    """
    parser = argparse.ArgumentParser(description="Public-API docstring coverage gate with per-module debt accounting.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite tools/docstring_debt.json from the current ruff probe.",
    )
    parser.add_argument(
        "--allow-baseline-increase",
        action="store_true",
        help="Permit --update-baseline to record a higher total than the existing ledger.",
    )
    parser.add_argument(
        "--all-definitions",
        nargs="+",
        type=Path,
        default=[],
        help="Check additional files for all definitions, including private/nested helpers; default owners stay enforced.",
    )
    args = parser.parse_args(argv)
    targets = [REPO_ROOT / path for path in ALL_DEFINITION_TARGETS] + args.all_definitions
    try:
        missing = all_definition_findings(targets)
    except (OSError, SyntaxError, UnicodeError):
        print("[docstrings] FAILED: all-definition probe could not be read.", file=sys.stderr)
        return 2
    if missing:
        print("[docstrings] FAILED: missing all-definition docstrings:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    print(f"[docstrings] all-definition scope: {len(targets)} files, zero missing module/class/callable docstrings")

    print(f"[docstrings] running probe: ruff check --select {','.join(COVERAGE_RULES)} {SOURCE_TARGET}")
    try:
        diagnostics = run_ruff_probe()
    except RuntimeError:
        print("[docstrings] FAILED: Ruff docstring probe failed.", file=sys.stderr)
        return 2
    current_modules = parse_module_counts(diagnostics)
    current_total = sum(current_modules.values())

    if args.update_baseline:
        if ENFORCE_ZERO and current_total > 0:
            print(
                f"[docstrings] REFUSED: hard-zero gate is active; cannot record {current_total} "
                "missing docstrings. Document the listed APIs instead of recording debt.",
                file=sys.stderr,
            )
            _report_debt(current_total, current_modules)
            return 3
        new_ledger = DocstringDebtLedger(total=current_total, per_module=current_modules)
        if LEDGER_PATH.exists():
            try:
                existing = load_ledger(LEDGER_PATH)
            except (OSError, UnicodeError, ValueError):
                print("[docstrings] FAILED: docstring debt ledger could not be read.", file=sys.stderr)
                return 2
            if current_total > existing.total and not args.allow_baseline_increase:
                print(
                    f"[docstrings] REFUSED: missing-docstring debt {current_total} exceeds recorded "
                    f"{existing.total}; pass --allow-baseline-increase to accept new debt.",
                    file=sys.stderr,
                )
                return 3
        try:
            write_ledger(new_ledger, LEDGER_PATH)
        except (OSError, UnicodeError, ValueError):
            print("[docstrings] FAILED: docstring debt ledger could not be written.", file=sys.stderr)
            return 2
        print(f"[docstrings] baseline updated: total {current_total} missing docstrings recorded.")
        return 0

    if ENFORCE_ZERO and current_total > 0:
        _report_debt(current_total, current_modules)
        print(
            f"[docstrings] FAILED: hard-zero gate is active and {current_total} public-API "
            f"docstrings are missing across {len(current_modules)} module(s). Document them.",
            file=sys.stderr,
        )
        return 1

    try:
        ledger = load_ledger(LEDGER_PATH)
    except (OSError, UnicodeError, ValueError):
        print("[docstrings] FAILED: docstring debt ledger could not be read.", file=sys.stderr)
        return 2
    result = evaluate_ratchet(current_total, current_modules, ledger)
    _report_debt(current_total, current_modules)

    if result.improvements and result.total_delta < 0:
        print(
            f"[docstrings] debt fell by {-result.total_delta} since the baseline; "
            "run --update-baseline to tighten the ratchet."
        )

    if not result.ok:
        # Coherent ledger/current sums make a total increase imply a module increase.
        print("[docstrings] FAILED: missing-docstring debt regressed in these modules:", file=sys.stderr)
        for module, (recorded, current) in sorted(result.regressions.items()):
            print(f"[docstrings]   {module}: {recorded} -> {current}", file=sys.stderr)
        if result.total_delta > 0:
            print(
                f"[docstrings] FAILED: total missing-docstring debt rose by {result.total_delta} "
                f"(baseline {ledger.total}, current {current_total}).",
                file=sys.stderr,
            )
        return 1

    print(f"[docstrings] OK: missing-docstring debt within baseline ({current_total} <= {ledger.total}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
