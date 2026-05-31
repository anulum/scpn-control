# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin online update tests
"""Behavioural tests for digital twin online model updating."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.control.digital_twin_online_update import (
    BayesianUpdateConfig,
    BayesianUpdateResult,
    TwinObservation,
    TwinParameterPrior,
    artifact_payload_sha256,
    bayesian_update_digital_twin,
    digital_twin_update_evidence,
    digital_twin_loss,
    assert_digital_twin_update_claim_admissible,
    load_external_simulator_artifact,
    synthetic_online_update_benchmark,
    validate_external_simulator_artifact,
)
from scpn_control.control.tokamak_digital_twin import run_digital_twin


def _artifact_payload(code: str = "TRANSP") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "simulator_code": code,
        "artifact_uri": f"file:///validation/reports/digital_twin/{code.lower()}_case.nc",
        "artifact_sha256": "a" * 64,
        "case_id": f"{code}-shot-001",
        "time_base_s": [0.0, 0.01, 0.02],
        "signal_units": {
            "final_avg_temp": "keV",
            "final_reward": "1",
            "mean_abs_actuator_lag": "1",
        },
    }


def test_external_simulator_artifact_accepts_transp_and_tsc(tmp_path):
    for code in ("TRANSP", "TSC"):
        path = tmp_path / f"{code.lower()}.json"
        path.write_text(json.dumps(_artifact_payload(code)), encoding="utf-8")
        artifact = load_external_simulator_artifact(path)
        assert artifact.simulator_code == code
        assert artifact.time_base_s == (0.0, 0.01, 0.02)


def test_external_simulator_artifact_rejects_bad_time_base():
    payload = _artifact_payload()
    payload["time_base_s"] = [0.0, 0.02, 0.01]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_external_simulator_artifact(payload)


def test_external_simulator_artifact_rejects_unsupported_code():
    payload = _artifact_payload("UNVETTED")
    with pytest.raises(ValueError, match="TRANSP or TSC"):
        validate_external_simulator_artifact(payload)


def test_artifact_payload_digest_is_stable():
    payload = _artifact_payload()
    assert artifact_payload_sha256(payload) == artifact_payload_sha256(dict(reversed(list(payload.items()))))


def test_digital_twin_physical_update_knobs_change_reference_loss():
    target = run_digital_twin(time_steps=10, seed=41, save_plot=False, verbose=False, n_e=1.25e20, Z_eff=2.5)
    observation = TwinObservation(
        targets={"final_avg_temp": float(target["final_avg_temp"]), "final_reward": float(target["final_reward"])},
        tolerances={"final_avg_temp": 0.05, "final_reward": 0.05},
    )
    matched = run_digital_twin(time_steps=10, seed=41, save_plot=False, verbose=False, n_e=1.25e20, Z_eff=2.5)
    mismatched = run_digital_twin(time_steps=10, seed=41, save_plot=False, verbose=False, n_e=0.8e20, Z_eff=1.0)
    assert digital_twin_loss(matched, observation) == pytest.approx(0.0)
    assert digital_twin_loss(mismatched, observation) > 0.0


def test_bayesian_update_improves_over_nominal_baseline():
    target = run_digital_twin(
        time_steps=12,
        seed=101,
        save_plot=False,
        verbose=False,
        n_e=1.16e20,
        Z_eff=2.0,
        actuator_tau_steps=2.0,
        actuator_rate_limit=0.08,
    )
    observation = TwinObservation(
        targets={
            "final_avg_temp": float(target["final_avg_temp"]),
            "mean_abs_actuator_lag": float(target["mean_abs_actuator_lag"]),
        },
        tolerances={"final_avg_temp": 0.05, "mean_abs_actuator_lag": 0.02},
        source="synthetic_regression_reference",
    )
    priors = (
        TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),
        TwinParameterPrior("Z_eff", 1.0, 3.0, 1.5),
        TwinParameterPrior("actuator_tau_steps", 0.0, 5.0, 0.0),
        TwinParameterPrior("actuator_rate_limit", 0.03, 0.3, 0.12),
    )

    result = bayesian_update_digital_twin(
        observation,
        priors,
        config=BayesianUpdateConfig(n_initial=6, n_iterations=5, n_candidates=64, time_steps=12, seed=4),
    )

    assert result.evidence_kind == "bounded_online_update"
    assert result.evaluated_points == 11
    assert result.best_loss <= result.baseline_loss
    assert set(result.best_parameters) == {"n_e", "Z_eff", "actuator_tau_steps", "actuator_rate_limit"}
    assert np.all(np.isfinite(result.loss_history))


def test_synthetic_online_update_benchmark_is_bounded_and_deterministic():
    a = synthetic_online_update_benchmark(seed=8)
    b = synthetic_online_update_benchmark(seed=8)
    assert a.best_loss == pytest.approx(b.best_loss)
    assert a.best_parameters == pytest.approx(b.best_parameters)
    assert a.best_loss <= a.baseline_loss


def test_digital_twin_update_evidence_requires_transp_tsc_and_improvement():
    artifacts = tuple(validate_external_simulator_artifact(_artifact_payload(code)) for code in ("TRANSP", "TSC"))
    observation = TwinObservation(
        targets={"final_avg_temp": 2.0},
        tolerances={"final_avg_temp": 0.05},
        source="paired_transp_tsc_reference",
    )
    priors = (
        TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),
        TwinParameterPrior("Z_eff", 1.0, 3.0, 1.5),
    )
    result = BayesianUpdateResult(
        best_parameters={"n_e": 1.1e20, "Z_eff": 2.0},
        best_loss=0.2,
        baseline_loss=0.8,
        evaluated_points=3,
        loss_history=(0.8, 0.5, 0.2),
        source=observation.source,
        evidence_kind="bounded_online_update",
    )

    evidence = digital_twin_update_evidence(
        observation,
        priors,
        result,
        artifacts,
        controller_formal_artifact_sha256="c" * 64,
    )

    assert evidence.simulator_codes == ("TRANSP", "TSC")
    assert evidence.improved_over_baseline
    assert len(evidence.observation_sha256) == 64
    assert_digital_twin_update_claim_admissible(evidence, observation, priors, result, artifacts)


def test_digital_twin_update_evidence_rejects_missing_simulator_and_non_improvement():
    artifacts = (validate_external_simulator_artifact(_artifact_payload("TRANSP")),)
    observation = TwinObservation(targets={"final_avg_temp": 2.0}, tolerances={"final_avg_temp": 0.05})
    priors = (TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),)
    result = BayesianUpdateResult(
        best_parameters={"n_e": 1.0e20},
        best_loss=0.8,
        baseline_loss=0.8,
        evaluated_points=2,
        loss_history=(0.8, 0.8),
        source=observation.source,
        evidence_kind="bounded_online_update",
    )
    evidence = digital_twin_update_evidence(observation, priors, result, artifacts)

    with pytest.raises(ValueError, match="TRANSP and TSC"):
        assert_digital_twin_update_claim_admissible(evidence, observation, priors, result, artifacts)

    tsc_artifact = validate_external_simulator_artifact(_artifact_payload("TSC"))
    with pytest.raises(ValueError, match="improve"):
        assert_digital_twin_update_claim_admissible(
            digital_twin_update_evidence(observation, priors, result, artifacts + (tsc_artifact,)),
            observation,
            priors,
            result,
            artifacts + (tsc_artifact,),
        )


def test_digital_twin_update_evidence_rejects_malformed_bayesian_results():
    artifacts = tuple(validate_external_simulator_artifact(_artifact_payload(code)) for code in ("TRANSP", "TSC"))
    observation = TwinObservation(
        targets={"final_avg_temp": 2.0},
        tolerances={"final_avg_temp": 0.05},
        source="paired_transp_tsc_reference",
    )
    priors = (
        TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),
        TwinParameterPrior("Z_eff", 1.0, 3.0, 1.5),
    )
    valid_result = BayesianUpdateResult(
        best_parameters={"n_e": 1.1e20, "Z_eff": 2.0},
        best_loss=0.2,
        baseline_loss=0.8,
        evaluated_points=3,
        loss_history=(0.8, 0.5, 0.2),
        source=observation.source,
        evidence_kind="bounded_online_update",
    )

    with pytest.raises(ValueError, match="source"):
        digital_twin_update_evidence(
            observation,
            priors,
            BayesianUpdateResult(
                best_parameters=valid_result.best_parameters,
                best_loss=valid_result.best_loss,
                baseline_loss=valid_result.baseline_loss,
                evaluated_points=valid_result.evaluated_points,
                loss_history=valid_result.loss_history,
                source="unbound_reference",
                evidence_kind=valid_result.evidence_kind,
            ),
            artifacts,
        )

    with pytest.raises(ValueError, match="prior bounds"):
        digital_twin_update_evidence(
            observation,
            priors,
            BayesianUpdateResult(
                best_parameters={"n_e": 2.0e20, "Z_eff": 2.0},
                best_loss=valid_result.best_loss,
                baseline_loss=valid_result.baseline_loss,
                evaluated_points=valid_result.evaluated_points,
                loss_history=valid_result.loss_history,
                source=valid_result.source,
                evidence_kind=valid_result.evidence_kind,
            ),
            artifacts,
        )

    with pytest.raises(ValueError, match="minimum loss_history"):
        digital_twin_update_evidence(
            observation,
            priors,
            BayesianUpdateResult(
                best_parameters=valid_result.best_parameters,
                best_loss=0.1,
                baseline_loss=valid_result.baseline_loss,
                evaluated_points=valid_result.evaluated_points,
                loss_history=valid_result.loss_history,
                source=valid_result.source,
                evidence_kind=valid_result.evidence_kind,
            ),
            artifacts,
        )

    missing_units_payload = _artifact_payload("TRANSP")
    missing_units_payload["signal_units"] = {"final_reward": "1"}
    with pytest.raises(ValueError, match="signal_units"):
        digital_twin_update_evidence(
            observation,
            priors,
            valid_result,
            (
                validate_external_simulator_artifact(missing_units_payload),
                validate_external_simulator_artifact(_artifact_payload("TSC")),
            ),
        )


def test_bayesian_update_config_rejects_bool_and_float_integer_fields():
    with pytest.raises(ValueError, match="n_initial"):
        BayesianUpdateConfig(n_initial=6.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed"):
        BayesianUpdateConfig(seed=True)  # type: ignore[arg-type]


def test_twin_observation_requires_matching_positive_tolerances() -> None:
    with pytest.raises(ValueError, match="missing tolerance"):
        TwinObservation(targets={"final_avg_temp": 2.0}, tolerances={})

    with pytest.raises(ValueError, match="tolerances"):
        TwinObservation(targets={"final_avg_temp": 2.0}, tolerances={"final_avg_temp": 0.0})


def test_digital_twin_update_evidence_rejects_duplicate_priors() -> None:
    artifacts = tuple(validate_external_simulator_artifact(_artifact_payload(code)) for code in ("TRANSP", "TSC"))
    observation = TwinObservation(targets={"final_avg_temp": 2.0}, tolerances={"final_avg_temp": 0.05})
    priors = (
        TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),
        TwinParameterPrior("n_e", 0.7e20, 1.4e20, 1.0e20),
    )
    result = BayesianUpdateResult(
        best_parameters={"n_e": 1.0e20},
        best_loss=0.1,
        baseline_loss=0.2,
        evaluated_points=2,
        loss_history=(0.2, 0.1),
        source=observation.source,
        evidence_kind="bounded_online_update",
    )

    with pytest.raises(ValueError, match="unique parameter names"):
        digital_twin_update_evidence(observation, priors, result, artifacts)
