# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu public contract tests.

"""Public-contract tests for Riccati feedback with static mu analysis."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from scpn_control.control.static_mu_analysis import (
    RiccatiStateFeedbackController,
    StaticMuAnalysisClaimEvidence,
    StaticMuAnalysisResult,
    StructuredUncertainty,
    UncertaintyBlock,
    compute_static_mu_upper_bound,
    design_riccati_state_feedback_with_static_mu_analysis,
    static_mu_analysis_claim_evidence,
)


def _plant() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([[-1.4, 0.2], [-0.1, -0.9]], dtype=float),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2), dtype=float),
    )


def _uncertainty() -> StructuredUncertainty:
    return StructuredUncertainty(
        [
            UncertaintyBlock("plasma_position", 1, 0.02, "real_scalar"),
            UncertaintyBlock("plasma_current", 1, 0.03, "real_scalar"),
        ]
    )


def test_canonical_design_has_no_fake_iteration_or_hinf_tolerance_parameters() -> None:
    """The canonical API must expose only controls that affect its algorithm."""
    signature = inspect.signature(design_riccati_state_feedback_with_static_mu_analysis)
    assert tuple(signature.parameters) == ("plant_ss", "uncertainty")
    assert tuple(inspect.signature(RiccatiStateFeedbackController.design).parameters) == ("self",)


def test_canonical_design_result_matches_independent_dc_analysis() -> None:
    """The result must bind the actual CARE gain and single-frequency mu bound."""
    plant = _plant()
    uncertainty = _uncertainty()
    result = design_riccati_state_feedback_with_static_mu_analysis(plant, uncertainty)

    assert isinstance(result, StaticMuAnalysisResult)
    assert result.analysis_frequency_rad_s == 0.0
    closed_loop = plant[0] - plant[1] @ result.controller_gain
    expected_map = (plant[2] @ np.linalg.solve(-closed_loop, plant[1]) + plant[3]) @ uncertainty.bound_matrix()
    expected_bound = compute_static_mu_upper_bound(
        expected_map,
        uncertainty.build_delta_structure(),
    )
    np.testing.assert_allclose(result.mu_upper_bound, expected_bound)
    np.testing.assert_allclose(
        result.closed_loop_spectral_abscissa,
        np.max(np.real(np.linalg.eigvals(closed_loop))),
    )


def test_canonical_controller_design_and_evidence_use_static_names() -> None:
    """Controller state and claim evidence must describe the executed analysis."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    result = controller.design()
    assert controller.analysis_result is result
    np.testing.assert_allclose(
        controller.step(np.array([0.1, -0.2]), dt=0.01),
        -result.controller_gain @ np.array([0.1, -0.2]),
    )
    evidence = static_mu_analysis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="static-mu-contract-v1",
    )
    assert isinstance(evidence, StaticMuAnalysisClaimEvidence)
    assert evidence.model_id == "bounded_static_mu_analysis"
    assert evidence.static_dc_analysis_only is True


def test_legacy_module_is_a_warning_facade_with_no_iteration_claim() -> None:
    """Old calls must remain usable while stating that their controls are ignored."""
    from scpn_control.control.mu_synthesis import MuSynthesisController, dk_iteration

    with pytest.warns(DeprecationWarning, match="removed in 0.25.0"):
        gain_one, bound_one, scales_one = dk_iteration(
            _plant(),
            _uncertainty(),
            n_iter=1,
            gamma_bisect_tol=0.5,
        )
    with pytest.warns(DeprecationWarning, match="does not execute D-K iterations"):
        gain_many, bound_many, scales_many = dk_iteration(
            _plant(),
            _uncertainty(),
            n_iter=99,
            gamma_bisect_tol=1.0e-8,
        )
    np.testing.assert_allclose(gain_one, gain_many)
    np.testing.assert_allclose(bound_one, bound_many)
    np.testing.assert_allclose(scales_one, scales_many)

    with pytest.warns(DeprecationWarning, match="RiccatiStateFeedbackController"):
        controller = MuSynthesisController(_plant(), _uncertainty())
    with pytest.warns(DeprecationWarning, match="does not execute D-K iterations"):
        controller.synthesize(n_dk_iter=7)
    assert controller.analysis_result is not None


def test_legacy_facade_preserves_bounded_access_and_persistence(tmp_path) -> None:
    """Every retained legacy symbol must warn and forward to the static owner."""
    from scpn_control.control.mu_synthesis import (
        MuSynthesisController,
        StructuredUncertainty as LegacyStructuredUncertainty,
        assert_mu_synthesis_validated_claim_admissible,
        compute_mu_upper_bound,
        load_mu_synthesis_claim_evidence,
        mu_synthesis_claim_evidence,
        save_mu_synthesis_claim_evidence,
    )

    legacy_uncertainty = LegacyStructuredUncertainty(_uncertainty().blocks)
    with pytest.warns(DeprecationWarning, match="build_delta_structure"):
        assert legacy_uncertainty.build_Delta_structure() == legacy_uncertainty.build_delta_structure()
    with pytest.warns(DeprecationWarning, match="compute_static_mu_upper_bound"):
        bound = compute_mu_upper_bound(np.eye(2), legacy_uncertainty.build_delta_structure())
    assert bound == pytest.approx(1.0)

    with pytest.warns(DeprecationWarning, match="RiccatiStateFeedbackController"):
        controller = MuSynthesisController(_plant(), legacy_uncertainty)
    assert controller.K is None
    assert controller.D_scalings is None
    assert controller.mu_peak == float("inf")
    controller.mu_peak = float("inf")
    assert controller.mu_peak == float("inf")
    with pytest.raises(ValueError, match="positive integer"):
        controller.synthesize(n_dk_iter=False)
    with pytest.warns(DeprecationWarning, match="does not execute D-K iterations"):
        controller.synthesize(n_dk_iter=1)
    assert controller.K is not None
    assert controller.D_scalings is not None
    assert controller.mu_peak > 0.0
    with pytest.warns(DeprecationWarning, match="inverse_static_mu_upper_bound"):
        assert controller.robustness_margin() == pytest.approx(1.0 / controller.mu_peak)

    with pytest.warns(DeprecationWarning, match="static_mu_analysis_claim_evidence"):
        evidence = mu_synthesis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="legacy-static-mu-contract-v1",
        )
    path = tmp_path / "legacy-static-mu.json"
    with pytest.warns(DeprecationWarning, match="save_static_mu_analysis_claim_evidence"):
        save_mu_synthesis_claim_evidence(evidence, path)
    with pytest.warns(DeprecationWarning, match="load_static_mu_analysis_claim_evidence"):
        assert load_mu_synthesis_claim_evidence(path) == evidence
    with pytest.warns(DeprecationWarning, match="assert_static_mu_analysis"):
        with pytest.raises(ValueError, match="validated static mu-analysis claim"):
            assert_mu_synthesis_validated_claim_admissible(evidence)

    controller.mu_peak = 0.0
    with pytest.warns(DeprecationWarning, match="inverse_static_mu_upper_bound"):
        assert controller.robustness_margin() == float("inf")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iter": 0}, "positive integer"),
        ({"n_iter": True}, "positive integer"),
        ({"gamma_bisect_tol": 0.0}, "must be positive"),
        ({"gamma_bisect_tol": float("nan")}, "must be finite"),
    ],
)
def test_legacy_iteration_parameters_fail_closed(kwargs, message) -> None:
    """Deprecated inert controls must still reject invalid caller input."""
    from scpn_control.control.mu_synthesis import dk_iteration

    with pytest.raises(ValueError, match=message):
        dk_iteration(_plant(), _uncertainty(), **kwargs)


def test_static_design_rejects_nonstable_returned_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public design must reject an unstable gain even after the CARE step."""
    import scpn_control.control.static_mu_analysis as static_mu

    unstable_plant = (
        np.eye(2),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )
    monkeypatch.setattr(static_mu, "_riccati_state_feedback", lambda *_args: np.zeros((2, 2)))
    with pytest.raises(RuntimeError, match="finite stable closed loop"):
        design_riccati_state_feedback_with_static_mu_analysis(unstable_plant, _uncertainty())


def test_deprecated_reference_validator_function_forwards(tmp_path) -> None:
    """The historical Python validator symbol must warn and use the new owner."""
    from validation.validate_mu_synthesis_reference import validate_mu_synthesis_reference

    with pytest.warns(DeprecationWarning, match="validate_static_mu_analysis_reference"):
        report = validate_mu_synthesis_reference(tmp_path)
    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 0


def test_deprecated_benchmark_entrypoint_is_visible(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The historical script must print its replacement before forwarding."""
    import validation.benchmark_mu_synthesis_claims as legacy_benchmark

    called: list[bool] = []
    monkeypatch.setenv("SCPN_BENCHMARK_CAMPAIGN_ID", "test-legacy-wrapper")
    monkeypatch.setattr(legacy_benchmark, "_canonical_main", lambda: called.append(True))
    legacy_benchmark.main()
    assert called == [True]
    assert "DEPRECATED: benchmark_mu_synthesis_claims.py" in capsys.readouterr().err


def test_canonical_module_does_not_export_synthesis_or_dk_names() -> None:
    """Current discovery surfaces must not advertise an unimplemented algorithm."""
    import scpn_control.control.static_mu_analysis as static_mu

    assert "dk_iteration" not in static_mu.__all__
    assert "MuSynthesisController" not in static_mu.__all__
    assert all("synthesis" not in name.lower() for name in static_mu.__all__)
