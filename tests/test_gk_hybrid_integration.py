# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Hybrid GK Integration Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.gk_corrector import CorrectionRecord, CorrectorConfig, GKCorrector
from scpn_control.core.gk_ood_detector import OODResult
from scpn_control.core.gk_online_learner import LearnerConfig, OnlineLearner
from scpn_control.core.gk_scheduler import GKScheduler, SchedulerConfig
from scpn_control.core.gk_verification_report import VerificationReport


def test_full_hybrid_pipeline():
    """End-to-end: scheduler → OOD → corrector → report."""
    nr = 20
    rho = np.linspace(0.05, 0.95, nr)

    scheduler = GKScheduler(SchedulerConfig(strategy="periodic", period=2, budget=3))
    corrector = GKCorrector(nr=nr, config=CorrectorConfig(smoothing_alpha=0.5))
    report = VerificationReport()

    chi_i = np.ones(nr) * 2.0
    chi_e = np.ones(nr) * 1.5
    D_e = np.ones(nr) * 0.3

    for step in range(6):
        req = scheduler.step(rho, chi_i)
        if req is not None:
            records = [
                CorrectionRecord(
                    rho_idx=idx,
                    rho=rho[idx],
                    chi_i_surrogate=chi_i[idx],
                    chi_i_gk=chi_i[idx] * 1.3,
                    chi_e_surrogate=chi_e[idx],
                    chi_e_gk=chi_e[idx] * 1.2,
                    D_e_surrogate=D_e[idx],
                    D_e_gk=D_e[idx] * 1.1,
                )
                for idx in req.rho_indices
            ]
            corrector.update(records, rho)
            report.add_step(verified=True, n_spot_checks=len(records))
            report.add_records(records)
        else:
            report.add_step(verified=False)

        chi_i_corr, chi_e_corr, D_e_corr = corrector.correct(chi_i, chi_e, D_e)
        report.add_correction_factor(corrector.mean_correction_factor)

    assert report.total_steps == 6
    assert report.steps_verified == 3  # period=2, steps 2,4,6
    assert report.total_spot_checks > 0
    assert np.all(np.isfinite(chi_i_corr))

    d = report.to_dict()
    assert d["verification_fraction"] == 0.5


def test_adaptive_with_ood():
    """Adaptive scheduler fires on OOD-flagged surfaces."""
    nr = 10
    rho = np.linspace(0.05, 0.95, nr)
    scheduler = GKScheduler(SchedulerConfig(strategy="adaptive", budget=5))
    chi = np.ones(nr)

    ood_results = [OODResult(is_ood=False, confidence=0.0, method="combined", details={}) for _ in range(nr)]
    ood_results[5] = OODResult(is_ood=True, confidence=0.9, method="mahalanobis", details={})

    req = scheduler.step(rho, chi, ood_results)
    assert req is not None
    assert 5 in req.rho_indices


def test_online_learner_in_pipeline():
    """Online learner accumulates data and triggers retraining."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=2))
    rng = np.random.default_rng(42)

    for _ in range(5):
        learner.add_sample(rng.random(10), rng.random(3))

    assert learner.buffer_full
    result = learner.try_retrain()
    assert result is not None
    assert learner.generation == 1
    assert len(learner.buffer) == 0


def test_corrector_converges():
    """Repeated corrections converge the correction factor."""
    nr = 10
    rho = np.linspace(0, 1, nr)
    corrector = GKCorrector(nr=nr, config=CorrectorConfig(smoothing_alpha=0.3))

    records = [
        CorrectionRecord(
            rho_idx=5,
            rho=0.5,
            chi_i_surrogate=1.0,
            chi_i_gk=2.0,
            chi_e_surrogate=1.0,
            chi_e_gk=1.5,
            D_e_surrogate=0.1,
            D_e_gk=0.1,
        )
    ]

    factors = []
    for _ in range(10):
        corrector.update(records, rho)
        chi_corr, _, _ = corrector.correct(np.ones(nr), np.ones(nr), np.ones(nr) * 0.1)
        factors.append(chi_corr[5])

    # Should converge toward GK value (2.0)
    assert factors[-1] > factors[0]
    assert factors[-1] > 1.5


def test_verification_report_json_valid():
    report = VerificationReport()
    for i in range(5):
        report.add_step(verified=(i % 2 == 0), n_spot_checks=i, n_ood=0)
    text = report.to_json()
    import json

    parsed = json.loads(text)
    assert parsed["total_steps"] == 5
    assert parsed["steps_verified"] == 3
