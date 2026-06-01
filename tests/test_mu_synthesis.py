# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Mu-synthesis tests
"""Module-specific tests for bounded static mu-analysis contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.control.mu_synthesis import (
    MuSynthesisController,
    StructuredUncertainty,
    UncertaintyBlock,
    assert_mu_synthesis_validated_claim_admissible,
    compute_mu_upper_bound,
    dk_iteration,
    load_mu_synthesis_claim_evidence,
    mu_synthesis_claim_evidence,
    save_mu_synthesis_claim_evidence,
)


def _plant() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([[-1.4, 0.2], [-0.1, -0.9]], dtype=float)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2), dtype=float)
    return A, B, C, D


def _uncertainty() -> StructuredUncertainty:
    return StructuredUncertainty(
        [
            UncertaintyBlock("plasma_position", 1, 0.02, "real_scalar"),
            UncertaintyBlock("plasma_current", 1, 0.03, "real_scalar"),
        ]
    )


def test_mu_upper_bound_respects_block_scaling_and_domains() -> None:
    M = np.array([[0.2, 0.05], [0.02, 0.15]], dtype=float)
    structure = [(1, "real_scalar"), (1, "real_scalar")]
    mu = compute_mu_upper_bound(M, structure)
    assert 0.0 < mu <= np.linalg.svd(M, compute_uv=False)[0] + 1.0e-12

    with pytest.raises(ValueError, match="M must be a square"):
        compute_mu_upper_bound(np.ones((2, 3)), structure)
    with pytest.raises(ValueError, match="Delta block sizes must sum"):
        compute_mu_upper_bound(M, [(1, "real_scalar")])
    with pytest.raises(ValueError, match="Delta structure"):
        compute_mu_upper_bound(np.zeros((0, 0)), [])
    with pytest.raises(ValueError, match="finite values"):
        compute_mu_upper_bound(np.array([[np.nan]]), [(1, "real_scalar")])
    with pytest.raises(ValueError, match="Delta block_type"):
        compute_mu_upper_bound(np.eye(1), [(1, "bad")])
    with pytest.raises(ValueError, match="Delta block size"):
        compute_mu_upper_bound(np.eye(1), [(0, "real_scalar")])


def test_mu_synthesis_controller_synthesizes_stable_static_controller() -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=3)
    assert controller.K is not None
    assert controller.mu_peak > 0.0
    assert controller.robustness_margin() > 0.0
    action = controller.step(np.array([0.1, -0.2]), dt=0.01)
    assert action.shape == (2,)

    with pytest.raises(ValueError, match="x must have shape"):
        controller.step(np.array([1.0]), dt=0.01)
    with pytest.raises(ValueError, match="dt must be positive"):
        controller.step(np.array([0.1, -0.2]), dt=0.0)
    with pytest.raises(ValueError, match="x must contain only finite"):
        controller.step(np.array([np.nan, -0.2]), dt=0.01)


def test_mu_synthesis_controller_fails_closed_before_synthesis_and_invalid_iterations() -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())

    with pytest.raises(RuntimeError, match="not synthesized"):
        controller.step(np.array([0.1, -0.2]), dt=0.01)
    with pytest.raises(ValueError, match="n_iter must be positive"):
        dk_iteration(_plant(), _uncertainty(), n_iter=0)
    with pytest.raises(ValueError, match="gamma_bisect_tol must be positive"):
        dk_iteration(_plant(), _uncertainty(), gamma_bisect_tol=0.0)


def test_uncertainty_blocks_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        UncertaintyBlock(" ", 1, 0.1, "real_scalar")
    with pytest.raises(ValueError, match="size must be a positive integer"):
        UncertaintyBlock("bad", 0, 0.1, "real_scalar")
    with pytest.raises(ValueError, match="bound must be positive"):
        UncertaintyBlock("bad", 1, 0.0, "real_scalar")
    with pytest.raises(ValueError, match="block_type must be one"):
        UncertaintyBlock("bad", 1, 0.1, "unstructured")


def test_structured_uncertainty_exposes_physical_bound_matrix_contract() -> None:
    uncertainty = _uncertainty()

    assert uncertainty.build_Delta_structure() == [(1, "real_scalar"), (1, "real_scalar")]
    assert uncertainty.total_size() == 2
    assert np.allclose(uncertainty.bound_matrix(), np.diag([0.02, 0.03]))
    with pytest.raises(ValueError, match="at least one"):
        StructuredUncertainty([])


@pytest.mark.parametrize(
    ("plant", "message"),
    [
        (
            (np.ones((2, 3)), np.eye(2), np.eye(2), np.zeros((2, 2))),
            "A must be square",
        ),
        (
            (np.eye(2), np.ones((3, 2)), np.eye(2), np.zeros((2, 2))),
            "B row count",
        ),
        (
            (np.eye(2), np.eye(2), np.ones((2, 3)), np.zeros((2, 2))),
            "C column count",
        ),
        (
            (np.eye(2), np.eye(2), np.eye(2), np.zeros((1, 2))),
            "D must have shape",
        ),
        (
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.eye(2), np.eye(2), np.zeros((2, 2))),
            "A must contain only finite",
        ),
        (
            (np.eye(2), np.ones((2, 1)), np.eye(2), np.zeros((2, 1))),
            "uncertainty size",
        ),
    ],
)
def test_dk_iteration_rejects_invalid_state_space_contracts(plant, message) -> None:
    with pytest.raises(ValueError, match=message):
        dk_iteration(plant, _uncertainty(), n_iter=1)


def test_mu_claim_evidence_records_bounded_boundary(tmp_path) -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=3)
    evidence = mu_synthesis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="mu-static-regression-v1",
    )
    assert evidence.claim_status == "bounded_static_mu_evidence"
    assert evidence.validated_claim_allowed is False
    assert evidence.static_dc_analysis_only is True
    assert evidence.closed_loop_spectral_abscissa < 0.0
    with pytest.raises(ValueError, match="validated mu-synthesis claim requires matched"):
        assert_mu_synthesis_validated_claim_admissible(evidence)

    output = tmp_path / "mu_claim.json"
    save_mu_synthesis_claim_evidence(evidence, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["claim_status"] == "bounded_static_mu_evidence"
    assert persisted["payload_sha256"]
    assert load_mu_synthesis_claim_evidence(output) == evidence
    with pytest.raises(ValueError, match="validated mu-synthesis claim requires matched"):
        load_mu_synthesis_claim_evidence(output, require_validated_claim=True)

    with pytest.raises(ValueError, match="evidence must"):
        save_mu_synthesis_claim_evidence(object(), tmp_path / "bad.json")


def test_mu_claim_evidence_loader_rejects_tampering_and_duplicate_keys(tmp_path) -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=3)
    evidence = mu_synthesis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="mu-static-regression-v1",
    )
    output = tmp_path / "mu_claim.json"
    save_mu_synthesis_claim_evidence(evidence, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["mu_peak_upper_bound"] = payload["mu_peak_upper_bound"] * 2.0
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload_sha256"):
        load_mu_synthesis_claim_evidence(output)

    duplicate = tmp_path / "duplicate_mu_claim.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_mu_synthesis_claim_evidence(duplicate)


def test_mu_claim_evidence_rejects_unsynthesized_controller_and_bad_claim_domains() -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    with pytest.raises(ValueError, match="synthesized"):
        mu_synthesis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="mu-static-regression-v1",
        )

    controller.synthesize(n_dk_iter=1)
    with pytest.raises(ValueError, match="source must be one"):
        mu_synthesis_claim_evidence(controller, source="internal_claim", source_id="mu-static-regression-v1")
    with pytest.raises(ValueError, match="source_id"):
        mu_synthesis_claim_evidence(controller, source="repository_static_mu_regression", source_id=" ")
    with pytest.raises(ValueError, match="model_id"):
        mu_synthesis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="mu-static-regression-v1",
            model_id=" ",
        )
    with pytest.raises(ValueError, match="mu_upper_bound_relative_tolerance"):
        mu_synthesis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="mu-static-regression-v1",
            mu_upper_bound_relative_tolerance=0.0,
        )


def test_mu_validated_claim_requires_reference_artifact() -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=3)
    artifact = {
        "source": "external_mu_toolbox_benchmark",
        "reference_dataset_id": "mu-toolbox-static-fixture-v1",
        "reference_artifact_sha256": "e" * 64,
        "reference_case_count": 2,
        "units": {
            "mu": "1",
            "robustness_margin": "1",
            "controller_gain": "1",
            "d_scaling": "1",
            "spectral_abscissa": "s^-1",
        },
        "metrics": {
            "mu_upper_bound_relative_error": 0.01,
            "robustness_margin_abs_error": 0.02,
            "controller_gain_relative_error": 0.03,
            "d_scaling_relative_error": 0.04,
            "closed_loop_spectral_abscissa_abs_error": 0.01,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 0.05,
            "robustness_margin_abs_error": 0.05,
            "controller_gain_relative_error": 0.10,
            "d_scaling_relative_error": 0.10,
            "closed_loop_spectral_abscissa_abs_error": 0.05,
        },
    }
    evidence = mu_synthesis_claim_evidence(
        controller,
        source="external_mu_toolbox_benchmark",
        source_id="mu-toolbox-static-fixture-v1",
        reference_artifact=artifact,
    )
    assert evidence.validated_claim_allowed is True
    assert assert_mu_synthesis_validated_claim_admissible(evidence) is evidence

    bad_artifact = dict(artifact)
    bad_artifact["metrics"] = dict(artifact["metrics"])
    bad_artifact["metrics"]["mu_upper_bound_relative_error"] = 0.5
    with pytest.raises(ValueError, match="mu_upper_bound_relative_error exceeds declared tolerance"):
        mu_synthesis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="mu-toolbox-static-fixture-v1",
            reference_artifact=bad_artifact,
        )


def _valid_reference_artifact() -> dict[str, object]:
    return {
        "source": "external_mu_toolbox_benchmark",
        "reference_dataset_id": "mu-toolbox-static-fixture-v1",
        "reference_artifact_sha256": "b" * 64,
        "reference_case_count": 3,
        "units": {
            "mu": "1",
            "robustness_margin": "1",
            "controller_gain": "1",
            "d_scaling": "1",
            "spectral_abscissa": "s^-1",
        },
        "metrics": {
            "mu_upper_bound_relative_error": 0.01,
            "robustness_margin_abs_error": 0.01,
            "controller_gain_relative_error": 0.02,
            "d_scaling_relative_error": 0.02,
            "closed_loop_spectral_abscissa_abs_error": 0.01,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 0.05,
            "robustness_margin_abs_error": 0.05,
            "controller_gain_relative_error": 0.10,
            "d_scaling_relative_error": 0.10,
            "closed_loop_spectral_abscissa_abs_error": 0.05,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact.update(source="repository_static_mu_regression"), "source must be one"),
        (lambda artifact: artifact.update(reference_artifact_sha256="not-a-digest"), "SHA-256"),
        (lambda artifact: artifact.update(reference_case_count=0), "positive integer"),
        (lambda artifact: artifact.update(metrics=[]), "metrics and tolerances"),
        (lambda artifact: artifact["metrics"].update(mu_upper_bound_relative_error=-0.1), "non-negative"),
        (lambda artifact: artifact["tolerances"].update(mu_upper_bound_relative_error=0.0), "positive"),
    ],
)
def test_mu_reference_artifact_rejects_invalid_validation_payloads(mutation, message) -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=1)
    artifact = _valid_reference_artifact()
    mutation(artifact)

    with pytest.raises(ValueError, match=message):
        mu_synthesis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="static-two-state-reference",
            reference_artifact=artifact,
        )


def test_mu_reference_artifact_rejects_non_mapping_reference_payload() -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=1)

    with pytest.raises(ValueError, match="reference_artifact must be a dictionary"):
        mu_synthesis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="static-two-state-reference",
            reference_artifact=["not", "a", "mapping"],
        )


def test_mu_reference_artifact_rejects_unit_mismatches() -> None:
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.synthesize(n_dk_iter=1)
    artifact = {
        "source": "external_mu_toolbox_benchmark",
        "reference_dataset_id": "mu-toolbox-static-fixture-v1",
        "reference_artifact_sha256": "b" * 64,
        "reference_case_count": 3,
        "units": {"mu": "wrong"},
        "metrics": {
            "mu_upper_bound_relative_error": 0.01,
            "robustness_margin_abs_error": 0.01,
            "controller_gain_relative_error": 0.02,
            "d_scaling_relative_error": 0.02,
            "closed_loop_spectral_abscissa_abs_error": 0.01,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 0.05,
            "robustness_margin_abs_error": 0.05,
            "controller_gain_relative_error": 0.10,
            "d_scaling_relative_error": 0.10,
            "closed_loop_spectral_abscissa_abs_error": 0.05,
        },
    }

    with pytest.raises(ValueError, match="unit contracts"):
        mu_synthesis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="static-two-state-reference",
            reference_artifact=artifact,
        )
