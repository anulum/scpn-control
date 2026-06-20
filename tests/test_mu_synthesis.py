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
from dataclasses import asdict, replace

import numpy as np
import pytest

import scpn_control.control.mu_synthesis as mu
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


# ── Validator-helper, claim-payload and numeric-guard branch contracts ────────


def _bounded_evidence():
    evidence = mu.MuSynthesisClaimEvidence(
        schema_version=1,
        source="repository_static_mu_regression",
        source_id="sid",
        model_id="mid",
        state_dimension=2,
        control_dimension=1,
        output_dimension=1,
        uncertainty_block_count=1,
        uncertainty_total_size=1,
        max_uncertainty_bound=0.5,
        block_structure=[(1, "full")],
        mu_peak_upper_bound=0.8,
        robustness_margin=1.25,
        controller_gain_frobenius_norm=2.0,
        d_scalings=[1.0],
        closed_loop_spectral_abscissa=-0.5,
        static_dc_analysis_only=True,
        reference_source=None,
        reference_dataset_id=None,
        reference_artifact_sha256=None,
        reference_case_count=None,
        mu_upper_bound_relative_error=None,
        robustness_margin_abs_error=None,
        controller_gain_relative_error=None,
        d_scaling_relative_error=None,
        closed_loop_spectral_abscissa_abs_error=None,
        mu_upper_bound_relative_tolerance=0.05,
        robustness_margin_abs_tolerance=0.05,
        controller_gain_relative_tolerance=0.10,
        d_scaling_relative_tolerance=0.10,
        closed_loop_spectral_abscissa_abs_tolerance=0.05,
        validated_claim_allowed=False,
        claim_status="bounded_static_mu_evidence",
    )
    return mu._with_payload_digest(evidence)


def _validated_evidence():
    evidence = replace(
        _bounded_evidence(),
        source="documented_public_reference",
        reference_source="public-ref",
        reference_dataset_id="dataset-1",
        reference_artifact_sha256="a" * 64,
        reference_case_count=3,
        mu_upper_bound_relative_error=0.01,
        robustness_margin_abs_error=0.01,
        controller_gain_relative_error=0.01,
        d_scaling_relative_error=0.01,
        closed_loop_spectral_abscissa_abs_error=0.01,
        validated_claim_allowed=True,
        claim_status="validated_static_mu_reference_matched",
    )
    return mu._with_payload_digest(evidence)


def _reseal(payload):
    payload["payload_sha256"] = mu._claim_payload_sha256(payload)
    return payload


def test_finite_scalar_rejects_non_finite():
    with pytest.raises(ValueError, match="must be finite"):
        mu._finite_scalar("x", float("nan"))


@pytest.mark.parametrize(
    ("fn_name", "value", "match"),
    [
        ("_positive_reference_scalar", float("inf"), "finite and positive"),
        ("_positive_reference_scalar", True, "finite and positive"),
        ("_nonnegative_reference_scalar", float("nan"), "finite and non-negative"),
        ("_require_positive_claim_int", 0, "positive integer"),
    ],
)
def test_reference_scalar_validators_reject(fn_name, value, match):
    with pytest.raises(ValueError, match=match):
        getattr(mu, fn_name)("field", value)


def test_require_bool_rejects_non_bool():
    with pytest.raises(ValueError, match="must be boolean"):
        mu._require_bool("field", 1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"block_structure": "notlist"}, "block_structure must be a list"),
        ({"block_structure": []}, "length must match uncertainty_block_count"),
        ({"block_structure": [(1, "full", 9)]}, "entries must be \\[size, block_type\\]"),
        ({"block_structure": [(1, "bogus_type")]}, "block_type must be one of"),
        ({"block_structure": [(2, "full")]}, "must sum to uncertainty_total_size"),
    ],
)
def test_validate_claim_structure_rejects(overrides, match):
    evidence = replace(_bounded_evidence(), **overrides)
    with pytest.raises(ValueError, match=match):
        mu._validate_claim_structure(evidence)


def test_validate_payload_rejects_missing_field():
    payload = asdict(_bounded_evidence())
    del payload["source"]
    with pytest.raises(ValueError, match="missing fields"):
        mu._validate_mu_synthesis_claim_payload(payload, require_validated_claim=False)


def test_validate_payload_rejects_unsupported_field():
    payload = asdict(_bounded_evidence())
    payload["bogus_field"] = 1
    with pytest.raises(ValueError, match="unsupported fields"):
        mu._validate_mu_synthesis_claim_payload(payload, require_validated_claim=False)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 2}, "schema_version is unsupported"),
        ({"source": "unlisted_source"}, "source must be one of"),
        ({"static_dc_analysis_only": False}, "must declare static_dc_analysis_only"),
        ({"closed_loop_spectral_abscissa": float("inf")}, "closed_loop_spectral_abscissa must be finite"),
        ({"closed_loop_spectral_abscissa": 0.1}, "must be negative"),
        ({"d_scalings": [1.0, 2.0]}, "one positive value per uncertainty block"),
        ({"claim_status": "wrong"}, "claim_status does not match"),
        ({"reference_case_count": 5}, "cannot carry partial reference fields"),
    ],
)
def test_validate_bounded_payload_rejects(overrides, match):
    payload = asdict(replace(_bounded_evidence(), **overrides))
    _reseal(payload)
    with pytest.raises(ValueError, match=match):
        mu._validate_mu_synthesis_claim_payload(payload, require_validated_claim=False)


def test_validate_validated_payload_rejects_non_validated_source():
    payload = asdict(replace(_validated_evidence(), source="repository_static_mu_regression"))
    _reseal(payload)
    with pytest.raises(ValueError, match="require a validated source"):
        mu._validate_mu_synthesis_claim_payload(payload, require_validated_claim=True)


def test_validate_validated_payload_rejects_metric_over_tolerance():
    payload = asdict(replace(_validated_evidence(), mu_upper_bound_relative_error=0.5))
    _reseal(payload)
    with pytest.raises(ValueError, match="exceeds declared tolerance"):
        mu._validate_mu_synthesis_claim_payload(payload, require_validated_claim=True)


def test_riccati_state_feedback_rejects_unstabilisable_plant():
    A = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float)
    B = np.array([[1.0], [0.0]], dtype=float)  # second unstable mode uncontrollable
    C = np.eye(2)
    with pytest.raises(RuntimeError, match="not stabilisable"):
        mu._riccati_state_feedback(A, B, C)


def test_closed_loop_dc_map_rejects_singular_system():
    identity = np.eye(2)
    with pytest.raises(RuntimeError, match="singular"):
        mu._closed_loop_dc_uncertainty_map(identity, identity, identity, np.zeros((2, 2)), identity)


def test_robustness_margin_infinite_for_nonpositive_mu_peak():
    controller = MuSynthesisController(_plant(), _uncertainty())
    controller.mu_peak = 0.0
    assert controller.robustness_margin() == float("inf")


def test_assert_validated_claim_rejects_non_evidence():
    with pytest.raises(ValueError, match="must be MuSynthesisClaimEvidence"):
        assert_mu_synthesis_validated_claim_admissible(object())  # type: ignore[arg-type]


def test_load_claim_evidence_rejects_non_object(tmp_path):
    path = tmp_path / "claim.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_mu_synthesis_claim_evidence(path)
