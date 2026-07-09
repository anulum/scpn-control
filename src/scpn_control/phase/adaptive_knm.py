# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Adaptive Knm
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Real-Time Adaptive Knm Engine
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
r"""
Online-adaptive coupling matrix K(t) driven by tokamak diagnostics.

Each control tick, diagnostic signals (β_N, disruption risk, Mirnov RMS,
per-layer order parameters) modulate the baseline Knm through independent
channels:

  1. Beta channel:     K *= (1 + β_scale · β_N)       clamped to β_max_boost
  2. Risk channel:     K[MHD pairs] += risk_gain · diagnostic_stress
  3. Coherence PI:     K[m,m] += PI(R_target − R_m, V_m) per-layer control
  4. Rate limiter:     |ΔK_ij| ≤ max_delta per tick
  5. Symmetry/pos:     K = ½(K+Kᵀ), K ≥ 0
  6. Guard veto:       if guard_approved=False → revert to last known-good K

The adaptation gains are bounded local-control heuristics. They are not
facility-calibrated stability laws and should be admitted for public claims only
through benchmark or validation evidence that records its operating context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_control.phase.knm import KnmSpec


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Per-tick plasma diagnostic bundle.

    Attributes
    ----------
    R_layer
        Per-layer Kuramoto order parameters, dimensionless in ``[0, 1]``.
    V_layer
        Per-layer Lyapunov candidate values, dimensionless in ``[0, 2]`` for
        the Kuramoto candidate used by this repository.
    lambda_exp
        Global Lyapunov exponent estimate in inverse seconds.
    beta_n
        Normalised beta, dimensionless and non-negative.
    q95
        Edge safety factor, dimensionless and positive.
    disruption_risk
        Upstream disruption-risk score, dimensionless in ``[0, 1]``.
    mirnov_rms
        Normalised Mirnov RMS amplitude, non-negative and dimensionless in this
        engine contract.
    guard_approved
        Whether the preceding Lyapunov guard verdict allowed adaptation.
    """

    R_layer: NDArray[np.float64]
    V_layer: NDArray[np.float64]
    lambda_exp: float
    beta_n: float
    q95: float
    disruption_risk: float
    mirnov_rms: float
    guard_approved: bool


@dataclass(frozen=True)
class AdaptiveKnmConfig:
    """Tuning knobs for each adaptation channel.

    All gains are dimensionless except ``lambda_risk_gain_s``, which has units
    of seconds so that ``lambda_exp`` in ``1/s`` maps to a dimensionless risk
    contribution. ``q95_reference`` is a dimensionless low-safety-factor
    reference, and ``mirnov_rms_reference`` is the normalisation scale for the
    dimensionless Mirnov RMS diagnostic.
    """

    beta_scale: float = 0.3
    beta_max_boost: float = 0.5
    risk_pairs: tuple[tuple[int, int], ...] = ((2, 5), (3, 5), (2, 4))
    risk_gain: float = 0.4
    lambda_risk_gain_s: float = 0.1
    q95_reference: float = 3.0
    q95_low_risk_gain: float = 0.2
    mirnov_rms_reference: float = 1.0
    mirnov_risk_gain: float = 0.2
    coherence_Kp: float = 0.15
    coherence_Ki: float = 0.02
    coherence_R_target: float = 0.6
    lyapunov_v_gain: float = 0.1
    coherence_max_boost: float = 0.3
    max_delta_per_tick: float = 0.02
    revert_on_guard_refusal: bool = True


class AdaptiveKnmEngine:
    """Diagnostic-driven online adaptation of the Knm coupling matrix.

    Holds a baseline K from a KnmSpec and produces K_adapted each tick.
    """

    def __init__(
        self,
        baseline_spec: KnmSpec,
        config: AdaptiveKnmConfig | None = None,
    ) -> None:
        self._baseline = np.asarray(baseline_spec.K, dtype=np.float64).copy()
        self._L = self._baseline.shape[0]
        self._cfg = config or AdaptiveKnmConfig()
        self._validate_config(self._cfg)
        self._K_current = self._baseline.copy()
        self._K_last_good = self._baseline.copy()
        self._integral = np.zeros(self._L, dtype=np.float64)

    def update(self, snap: DiagnosticSnapshot) -> NDArray[np.float64]:
        """Apply all adaptation channels and return K_adapted."""
        cfg = self._cfg
        self._validate_snapshot(snap)

        # Guard veto: revert before computing if previous tick was refused
        if not snap.guard_approved and cfg.revert_on_guard_refusal:
            self._K_current[:] = self._K_last_good
            self._integral[:] = 0.0
            return self._K_current.copy()

        K_new = self._baseline.copy()

        # 1. Beta channel: scale entire matrix
        beta_boost = min(cfg.beta_scale * snap.beta_n, cfg.beta_max_boost)
        K_new *= 1.0 + beta_boost

        # 2. Risk channel: amplify MHD-relevant pairs
        diagnostic_stress = self._diagnostic_stress(snap)
        for i, j in cfg.risk_pairs:
            if i < self._L and j < self._L:
                delta = cfg.risk_gain * diagnostic_stress
                K_new[i, j] += delta
                K_new[j, i] += delta

        # 3. Coherence PI: per-layer diagonal boost
        R = np.asarray(snap.R_layer, dtype=np.float64)
        V = np.asarray(snap.V_layer, dtype=np.float64)
        error = np.maximum(cfg.coherence_R_target - R[: self._L], 0.0)
        v_scale = 1.0 + cfg.lyapunov_v_gain * np.clip(0.5 * V[: self._L], 0.0, 1.0)
        coherence_drive = error * v_scale
        self._integral += cfg.coherence_Ki * coherence_drive
        np.clip(self._integral, 0.0, cfg.coherence_max_boost, out=self._integral)
        for m in range(self._L):
            boost = cfg.coherence_Kp * coherence_drive[m] + self._integral[m]
            K_new[m, m] += min(boost, cfg.coherence_max_boost)

        # 4. Invariants: symmetry + non-negativity
        K_new = 0.5 * (K_new + K_new.T)
        np.maximum(K_new, 0.0, out=K_new)

        # 5. Rate limit: per-element clamp
        dK = K_new - self._K_current
        np.clip(dK, -cfg.max_delta_per_tick, cfg.max_delta_per_tick, out=dK)
        K_new = self._K_current + dK

        # Re-enforce invariants after rate limiting
        K_new = 0.5 * (K_new + K_new.T)
        np.maximum(K_new, 0.0, out=K_new)

        # 6. Commit — K_last_good tracks the most recent approved K
        self._K_current[:] = K_new
        self._K_last_good[:] = K_new
        return self._K_current.copy()

    def _validate_snapshot(self, snap: DiagnosticSnapshot) -> None:
        """Reject malformed diagnostic input before it can mutate K state."""
        r_layer = np.asarray(snap.R_layer, dtype=np.float64)
        if r_layer.shape != (self._L,):
            raise ValueError(f"R_layer must have shape ({self._L},), got {r_layer.shape}")
        if not np.isfinite(r_layer).all():
            raise ValueError("R_layer must contain only finite values")

        v_layer = np.asarray(snap.V_layer, dtype=np.float64)
        if v_layer.shape != (self._L,):
            raise ValueError(f"V_layer must have shape ({self._L},), got {v_layer.shape}")
        if not np.isfinite(v_layer).all():
            raise ValueError("V_layer must contain only finite values")

        scalar_fields = {
            "lambda_exp": snap.lambda_exp,
            "beta_n": snap.beta_n,
            "q95": snap.q95,
            "disruption_risk": snap.disruption_risk,
            "mirnov_rms": snap.mirnov_rms,
        }
        for name, value in scalar_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if snap.beta_n < 0.0:
            raise ValueError("beta_n must be non-negative")
        if snap.q95 <= 0.0:
            raise ValueError("q95 must be positive")
        if not 0.0 <= snap.disruption_risk <= 1.0:
            raise ValueError("disruption_risk must be in [0, 1]")
        if snap.mirnov_rms < 0.0:
            raise ValueError("mirnov_rms must be non-negative")

    def _validate_config(self, cfg: AdaptiveKnmConfig) -> None:
        """Reject invalid gain or reference constants before adaptation."""
        nonnegative_fields = {
            "beta_scale": cfg.beta_scale,
            "beta_max_boost": cfg.beta_max_boost,
            "risk_gain": cfg.risk_gain,
            "lambda_risk_gain_s": cfg.lambda_risk_gain_s,
            "q95_low_risk_gain": cfg.q95_low_risk_gain,
            "mirnov_risk_gain": cfg.mirnov_risk_gain,
            "coherence_Kp": cfg.coherence_Kp,
            "coherence_Ki": cfg.coherence_Ki,
            "lyapunov_v_gain": cfg.lyapunov_v_gain,
            "coherence_max_boost": cfg.coherence_max_boost,
            "max_delta_per_tick": cfg.max_delta_per_tick,
        }
        for name, value in nonnegative_fields.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        positive_fields = {
            "q95_reference": cfg.q95_reference,
            "mirnov_rms_reference": cfg.mirnov_rms_reference,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(cfg.coherence_R_target) or not 0.0 <= cfg.coherence_R_target <= 1.0:
            raise ValueError("coherence_R_target must be finite and in [0, 1]")
        for pair in cfg.risk_pairs:
            if len(pair) != 2 or pair[0] < 0 or pair[1] < 0:
                raise ValueError("risk_pairs must contain non-negative index pairs")

    def _diagnostic_stress(self, snap: DiagnosticSnapshot) -> float:
        """Combine scalar instability diagnostics into a bounded risk drive."""
        cfg = self._cfg
        lambda_drive = max(snap.lambda_exp, 0.0) * cfg.lambda_risk_gain_s
        q95_drive = cfg.q95_low_risk_gain * max((cfg.q95_reference - snap.q95) / cfg.q95_reference, 0.0)
        mirnov_drive = cfg.mirnov_risk_gain * (snap.mirnov_rms / cfg.mirnov_rms_reference)
        return float(np.clip(snap.disruption_risk + lambda_drive + q95_drive + mirnov_drive, 0.0, 1.0))

    def reset(self) -> None:
        """Revert to baseline and clear integral state."""
        self._K_current[:] = self._baseline
        self._K_last_good[:] = self._baseline
        self._integral[:] = 0.0

    @property
    def K_current(self) -> NDArray[np.float64]:
        """A copy of the current adaptive inter-layer coupling matrix K."""
        return self._K_current.copy()

    @property
    def adaptation_summary(self) -> dict[str, float]:
        """Snapshot of engine state for dashboard export."""
        diff = self._K_current - self._baseline
        k_values = [float(value) for value in self._K_current.ravel()]
        diff_values = [abs(float(value)) for value in diff.ravel()]
        integral_values = [float(value) for value in self._integral.ravel()]
        return {
            "L": self._L,
            "K_mean": math.fsum(k_values) / len(k_values),
            "K_max": max(k_values),
            "delta_frobenius": math.sqrt(math.fsum(value * value for value in diff_values)),
            "delta_max_element": max(diff_values),
            "integral_sum": math.fsum(integral_values),
        }
