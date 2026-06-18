# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin online model updating
"""Fail-closed external-simulator coupling and Bayesian twin updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from scpn_control.control.tokamak_digital_twin import run_digital_twin

SUPPORTED_EXTERNAL_SIMULATORS = ("TRANSP", "TSC")
TUNABLE_PARAMETER_NAMES = ("n_e", "Z_eff", "actuator_tau_steps", "actuator_rate_limit")


@dataclass(frozen=True)
class ExternalSimulatorArtifact:
    """Validated external simulator coupling artifact metadata."""

    simulator_code: str
    artifact_uri: str
    artifact_sha256: str
    case_id: str
    time_base_s: tuple[float, ...]
    signal_units: dict[str, str]


@dataclass(frozen=True)
class TwinParameterPrior:
    """Bounded tunable digital-twin model parameter."""

    name: str
    lower: float
    upper: float
    nominal: float

    def __post_init__(self) -> None:
        if self.name not in TUNABLE_PARAMETER_NAMES:
            raise ValueError(f"Unsupported tunable parameter {self.name!r}")
        lower = _require_finite("lower", self.lower)
        upper = _require_finite("upper", self.upper)
        nominal = _require_finite("nominal", self.nominal)
        if not lower < upper:
            raise ValueError("parameter prior lower bound must be < upper bound")
        if not lower <= nominal <= upper:
            raise ValueError("parameter nominal value must lie inside bounds")


@dataclass(frozen=True)
class TwinObservation:
    """Target digital-twin summary values from an external reference."""

    targets: dict[str, float]
    tolerances: dict[str, float]
    source: str = "synthetic_reference"

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("targets must not be empty")
        for key, value in self.targets.items():
            _require_finite(f"targets[{key}]", value)
            tolerance = self.tolerances.get(key)
            if tolerance is None:
                raise ValueError(f"missing tolerance for target {key!r}")
            _require_positive(f"tolerances[{key}]", tolerance)


@dataclass(frozen=True)
class BayesianUpdateConfig:
    """Deterministic Gaussian-process-style Bayesian update settings."""

    n_initial: int = 6
    n_iterations: int = 8
    n_candidates: int = 96
    exploration_weight: float = 0.25
    kernel_length_scale: float = 0.45
    noise: float = 1.0e-8
    time_steps: int = 24
    seed: int = 20240531

    def __post_init__(self) -> None:
        _require_positive_int("n_initial", self.n_initial)
        _require_positive_int("n_iterations", self.n_iterations)
        _require_positive_int("n_candidates", self.n_candidates)
        _require_positive("kernel_length_scale", self.kernel_length_scale)
        _require_positive("noise", self.noise)
        _require_positive_int("time_steps", self.time_steps)
        if not math.isfinite(float(self.exploration_weight)) or self.exploration_weight < 0.0:
            raise ValueError("exploration_weight must be finite and >= 0")
        _require_int("seed", self.seed)


@dataclass(frozen=True)
class BayesianUpdateResult:
    """Bayesian online-update result."""

    best_parameters: dict[str, float]
    best_loss: float
    baseline_loss: float
    evaluated_points: int
    loss_history: tuple[float, ...]
    source: str
    evidence_kind: str


@dataclass(frozen=True)
class DigitalTwinUpdateEvidence:
    """Tamper-evident admission evidence for bounded online twin updates."""

    schema_version: int
    simulator_codes: tuple[str, ...]
    external_artifacts_sha256: str
    observation_sha256: str
    priors_sha256: str
    result_sha256: str
    best_loss: float
    baseline_loss: float
    improved_over_baseline: bool
    evaluated_points: int
    controller_formal_artifact_sha256: str | None
    claim_status: str


def validate_external_simulator_artifact(payload: dict[str, Any]) -> ExternalSimulatorArtifact:
    """Validate TRANSP/TSC-style simulator artefact metadata before coupling."""
    if payload.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    simulator_code = str(payload.get("simulator_code", "")).upper()
    if simulator_code not in SUPPORTED_EXTERNAL_SIMULATORS:
        raise ValueError("simulator_code must be TRANSP or TSC")
    artifact_uri = _require_nonempty_string(payload, "artifact_uri")
    artifact_sha256 = _require_nonempty_string(payload, "artifact_sha256")
    if len(artifact_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in artifact_sha256):
        raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
    case_id = _require_nonempty_string(payload, "case_id")
    time_base_raw = payload.get("time_base_s")
    if not isinstance(time_base_raw, list | tuple) or len(time_base_raw) < 2:
        raise ValueError("time_base_s must contain at least two finite samples")
    time_base = tuple(float(item) for item in time_base_raw)
    if not all(math.isfinite(item) for item in time_base):
        raise ValueError("time_base_s must be finite")
    if any(b <= a for a, b in zip(time_base, time_base[1:])):
        raise ValueError("time_base_s must be strictly increasing")
    signal_units = payload.get("signal_units")
    if not isinstance(signal_units, dict) or not signal_units:
        raise ValueError("signal_units must be a non-empty object")
    for key, value in signal_units.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError("signal_units entries must be non-empty strings")
    return ExternalSimulatorArtifact(
        simulator_code=simulator_code,
        artifact_uri=artifact_uri,
        artifact_sha256=artifact_sha256.lower(),
        case_id=case_id,
        time_base_s=time_base,
        signal_units={str(key): str(value) for key, value in signal_units.items()},
    )


def load_external_simulator_artifact(path: str | Path) -> ExternalSimulatorArtifact:
    """Load and validate an external simulator artifact metadata JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("external simulator artifact must be a JSON object")
    return validate_external_simulator_artifact(payload)


def digital_twin_loss(summary: dict[str, Any], observation: TwinObservation) -> float:
    """Weighted squared residual between a twin summary and reference targets."""
    loss = 0.0
    for key, target in observation.targets.items():
        if key not in summary:
            raise ValueError(f"digital twin summary missing target key {key!r}")
        value = _require_finite(f"summary[{key}]", float(summary[key]))
        scale = observation.tolerances[key]
        loss += ((value - target) / scale) ** 2
    return float(loss / len(observation.targets))


def bayesian_update_digital_twin(
    observation: TwinObservation,
    priors: tuple[TwinParameterPrior, ...],
    *,
    config: BayesianUpdateConfig | None = None,
) -> BayesianUpdateResult:
    """Run deterministic Bayesian optimisation for online twin calibration."""
    if not priors:
        raise ValueError("at least one parameter prior is required")
    cfg = config or BayesianUpdateConfig()
    rng = np.random.default_rng(cfg.seed)
    baseline = {prior.name: prior.nominal for prior in priors}
    baseline_loss = _evaluate_parameters(baseline, observation, cfg)

    x_seen: list[AnyFloatArray] = []
    y_seen: list[float] = []
    loss_history: list[float] = []

    for _ in range(cfg.n_initial):
        x = rng.random(len(priors))
        params = _denormalise(x, priors)
        y = _evaluate_parameters(params, observation, cfg)
        x_seen.append(x)
        y_seen.append(y)
        loss_history.append(y)

    for _ in range(cfg.n_iterations):
        candidates = rng.random((cfg.n_candidates, len(priors)))
        mean, std = _gp_predict(np.vstack(x_seen), np.asarray(y_seen), candidates, cfg)
        acquisition = mean - cfg.exploration_weight * std
        idx = int(np.argmin(acquisition))
        x = candidates[idx]
        params = _denormalise(x, priors)
        y = _evaluate_parameters(params, observation, cfg)
        x_seen.append(x)
        y_seen.append(y)
        loss_history.append(y)

    best_idx = int(np.argmin(y_seen))
    return BayesianUpdateResult(
        best_parameters=_denormalise(x_seen[best_idx], priors),
        best_loss=float(y_seen[best_idx]),
        baseline_loss=float(baseline_loss),
        evaluated_points=len(y_seen),
        loss_history=tuple(float(item) for item in loss_history),
        source=observation.source,
        evidence_kind="bounded_online_update",
    )


def synthetic_online_update_benchmark(seed: int = 20240531) -> BayesianUpdateResult:
    """Deterministic local benchmark for the online-update contract."""
    target_summary = run_digital_twin(
        time_steps=24,
        seed=101,
        save_plot=False,
        verbose=False,
        n_e=1.18e20,
        Z_eff=2.2,
        actuator_tau_steps=2.0,
        actuator_rate_limit=0.08,
    )
    observation = TwinObservation(
        targets={
            "final_avg_temp": float(target_summary["final_avg_temp"]),
            "final_reward": float(target_summary["final_reward"]),
            "mean_abs_actuator_lag": float(target_summary["mean_abs_actuator_lag"]),
        },
        tolerances={
            "final_avg_temp": 0.05,
            "final_reward": 0.05,
            "mean_abs_actuator_lag": 0.02,
        },
        source="synthetic_digital_twin_reference",
    )
    priors = (
        TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),
        TwinParameterPrior("Z_eff", 1.0, 3.0, 1.5),
        TwinParameterPrior("actuator_tau_steps", 0.0, 5.0, 0.0),
        TwinParameterPrior("actuator_rate_limit", 0.03, 0.3, 0.12),
    )
    return bayesian_update_digital_twin(
        observation,
        priors,
        config=BayesianUpdateConfig(n_initial=8, n_iterations=10, n_candidates=128, time_steps=24, seed=seed),
    )


def artifact_payload_sha256(payload: dict[str, Any]) -> str:
    """Stable SHA-256 digest for a simulator metadata payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _is_sha256_hex(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def digital_twin_update_evidence(
    observation: TwinObservation,
    priors: tuple[TwinParameterPrior, ...],
    result: BayesianUpdateResult,
    external_artifacts: tuple[ExternalSimulatorArtifact, ...],
    *,
    controller_formal_artifact_sha256: str | None = None,
) -> DigitalTwinUpdateEvidence:
    """Build tamper-evident evidence for bounded online digital-twin updates."""
    if not isinstance(observation, TwinObservation):
        raise ValueError("observation must be TwinObservation")
    if not priors or any(not isinstance(prior, TwinParameterPrior) for prior in priors):
        raise ValueError("priors must contain TwinParameterPrior entries")
    if not isinstance(result, BayesianUpdateResult):
        raise ValueError("result must be BayesianUpdateResult")
    if not external_artifacts or any(not isinstance(item, ExternalSimulatorArtifact) for item in external_artifacts):
        raise ValueError("external_artifacts must contain ExternalSimulatorArtifact entries")
    _validate_update_inputs(observation, priors, result, external_artifacts)
    proof_digest = _validate_optional_sha256(
        "controller_formal_artifact_sha256",
        controller_formal_artifact_sha256,
    )
    simulator_codes = tuple(sorted({artifact.simulator_code for artifact in external_artifacts}))
    artifact_payload = sorted(
        (asdict(artifact) for artifact in external_artifacts), key=lambda item: item["simulator_code"]
    )
    return DigitalTwinUpdateEvidence(
        schema_version=1,
        simulator_codes=simulator_codes,
        external_artifacts_sha256=_canonical_sha256(artifact_payload),
        observation_sha256=_canonical_sha256(observation),
        priors_sha256=_canonical_sha256(tuple(priors)),
        result_sha256=_canonical_sha256(result),
        best_loss=float(result.best_loss),
        baseline_loss=float(result.baseline_loss),
        improved_over_baseline=bool(result.best_loss < result.baseline_loss),
        evaluated_points=int(result.evaluated_points),
        controller_formal_artifact_sha256=proof_digest,
        claim_status=(
            "bounded online Bayesian digital-twin update evidence only; "
            "external simulator artifacts are provenance inputs, not facility validation"
        ),
    )


def assert_digital_twin_update_claim_admissible(
    evidence: DigitalTwinUpdateEvidence,
    observation: TwinObservation,
    priors: tuple[TwinParameterPrior, ...],
    result: BayesianUpdateResult,
    external_artifacts: tuple[ExternalSimulatorArtifact, ...],
    *,
    require_simulators: tuple[str, ...] = SUPPORTED_EXTERNAL_SIMULATORS,
) -> DigitalTwinUpdateEvidence:
    """Fail closed unless online twin-update evidence matches replay inputs."""
    if not isinstance(evidence, DigitalTwinUpdateEvidence):
        raise ValueError("evidence must be DigitalTwinUpdateEvidence")
    if evidence.schema_version != 1:
        raise ValueError("digital twin update evidence schema_version is unsupported")
    required = tuple(sorted(code.upper() for code in require_simulators))
    if tuple(sorted(evidence.simulator_codes)) != required:
        raise ValueError("digital twin update evidence requires TRANSP and TSC simulator artifacts")
    recomputed = digital_twin_update_evidence(
        observation,
        priors,
        result,
        external_artifacts,
        controller_formal_artifact_sha256=evidence.controller_formal_artifact_sha256,
    )
    if evidence.external_artifacts_sha256 != recomputed.external_artifacts_sha256:
        raise ValueError("digital twin update evidence external_artifacts_sha256 mismatch")
    if evidence.observation_sha256 != recomputed.observation_sha256:
        raise ValueError("digital twin update evidence observation_sha256 mismatch")
    if evidence.priors_sha256 != recomputed.priors_sha256:
        raise ValueError("digital twin update evidence priors_sha256 mismatch")
    if evidence.result_sha256 != recomputed.result_sha256:
        raise ValueError("digital twin update evidence result_sha256 mismatch")
    if result.evidence_kind != "bounded_online_update":
        raise ValueError("digital twin update evidence requires bounded_online_update result")
    _validate_update_inputs(observation, priors, result, external_artifacts)
    if not evidence.improved_over_baseline or result.best_loss >= result.baseline_loss:
        raise ValueError("digital twin update evidence must improve over baseline")
    if evidence.evaluated_points != result.evaluated_points or evidence.evaluated_points < len(priors) + 1:
        raise ValueError("digital twin update evidence evaluated_points is inconsistent")
    if not math.isclose(evidence.best_loss, result.best_loss, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("digital twin update evidence best_loss mismatch")
    if not math.isclose(evidence.baseline_loss, result.baseline_loss, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("digital twin update evidence baseline_loss mismatch")
    _validate_optional_sha256("controller_formal_artifact_sha256", evidence.controller_formal_artifact_sha256)
    return evidence


def _evaluate_parameters(
    params: dict[str, float],
    observation: TwinObservation,
    config: BayesianUpdateConfig,
) -> float:
    summary = run_digital_twin(
        time_steps=config.time_steps,
        seed=101,
        save_plot=False,
        verbose=False,
        n_e=float(params.get("n_e", 1.0e20)),
        Z_eff=float(params.get("Z_eff", 1.5)),
        actuator_tau_steps=float(params.get("actuator_tau_steps", 0.0)),
        actuator_rate_limit=float(params.get("actuator_rate_limit", 2.0)),
    )
    return digital_twin_loss(summary, observation)


def _validate_update_inputs(
    observation: TwinObservation,
    priors: tuple[TwinParameterPrior, ...],
    result: BayesianUpdateResult,
    external_artifacts: tuple[ExternalSimulatorArtifact, ...],
) -> None:
    prior_names = tuple(prior.name for prior in priors)
    if len(set(prior_names)) != len(prior_names):
        raise ValueError("priors must use unique parameter names")
    result_names = set(result.best_parameters)
    if result_names != set(prior_names):
        raise ValueError("result best_parameters must match prior names")
    for prior in priors:
        value = _require_finite(f"best_parameters[{prior.name}]", result.best_parameters[prior.name])
        if value < prior.lower or value > prior.upper:
            raise ValueError("result best_parameters must lie inside prior bounds")
    best_loss = _require_nonnegative("best_loss", result.best_loss)
    baseline_loss = _require_nonnegative("baseline_loss", result.baseline_loss)
    _ = baseline_loss
    if result.source != observation.source:
        raise ValueError("result source must match observation source")
    if result.evidence_kind != "bounded_online_update":
        raise ValueError("result evidence_kind must be bounded_online_update")
    evaluated_points = _require_positive_int("evaluated_points", result.evaluated_points)
    if not result.loss_history:
        raise ValueError("result loss_history must not be empty")
    if evaluated_points != len(result.loss_history):
        raise ValueError("result evaluated_points must equal loss_history length")
    losses = tuple(_require_nonnegative("loss_history", item) for item in result.loss_history)
    if not math.isclose(best_loss, min(losses), rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("result best_loss must match the minimum loss_history value")
    target_names = set(observation.targets)
    for artifact in external_artifacts:
        missing_units = sorted(target_names - set(artifact.signal_units))
        if missing_units:
            raise ValueError(
                "external simulator artifact signal_units missing observation targets: " + ", ".join(missing_units)
            )


def _denormalise(x: AnyFloatArray, priors: tuple[TwinParameterPrior, ...]) -> dict[str, float]:
    params: dict[str, float] = {}
    for value, prior in zip(x, priors):
        params[prior.name] = float(prior.lower + np.clip(value, 0.0, 1.0) * (prior.upper - prior.lower))
    return params


def _gp_predict(
    x_seen: AnyFloatArray,
    y_seen: AnyFloatArray,
    candidates: AnyFloatArray,
    config: BayesianUpdateConfig,
) -> tuple[FloatArray, FloatArray]:
    length = float(config.kernel_length_scale)
    k_xx = _rbf_kernel(x_seen, x_seen, length) + config.noise * np.eye(len(x_seen))
    k_cx = _rbf_kernel(candidates, x_seen, length)
    jitter = 1.0e-10 * np.eye(len(x_seen))
    chol = np.linalg.cholesky(k_xx + jitter)
    centered = y_seen - float(np.mean(y_seen))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
    mean = float(np.mean(y_seen)) + k_cx @ alpha
    v = np.linalg.solve(chol, k_cx.T)
    variance = np.maximum(1.0 - np.sum(v * v, axis=0), 0.0)
    return np.asarray(mean, dtype=np.float64), np.sqrt(variance)


def _rbf_kernel(a: AnyFloatArray, b: AnyFloatArray, length: float) -> FloatArray:
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.asarray(np.exp(-0.5 * dist2 / (length**2)), dtype=np.float64)


def _require_nonempty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_positive(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _require_nonnegative(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_positive_int(name: str, value: int) -> int:
    if _require_int(name, value) < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


__all__ = [
    "BayesianUpdateConfig",
    "BayesianUpdateResult",
    "DigitalTwinUpdateEvidence",
    "ExternalSimulatorArtifact",
    "SUPPORTED_EXTERNAL_SIMULATORS",
    "TUNABLE_PARAMETER_NAMES",
    "TwinObservation",
    "TwinParameterPrior",
    "assert_digital_twin_update_claim_admissible",
    "artifact_payload_sha256",
    "bayesian_update_digital_twin",
    "digital_twin_update_evidence",
    "digital_twin_loss",
    "load_external_simulator_artifact",
    "synthetic_online_update_benchmark",
    "validate_external_simulator_artifact",
]
