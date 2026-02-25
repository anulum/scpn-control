# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Multi-Layer UPDE Engine (Paper 27)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Unified Phase Dynamics Equation — multi-layer evolution parameterised
by the Knm coupling matrix from Paper 27.

Per-layer equation:
    dθ_{m,i}/dt = ω_{m,i}
                + K_{mm} · R_m · sin(ψ_m − θ_{m,i} − α_{mm})
                + Σ_{n≠m} K_{nm} · R_n · sin(ψ_n − θ_{m,i} − α_{nm})
                + ζ_m · sin(Ψ − θ_{m,i})

K_{mm} (diagonal):     intra-layer synchronisation
K_{nm} (off-diagonal): inter-layer bidirectional causality
ζ_m sin(Ψ − θ):       global field driver (reviewer request)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_control.phase.knm import KnmSpec
from scpn_control.phase.kuramoto import order_parameter, wrap_phase

FloatArray = NDArray[np.float64]


@dataclass
class UPDESystem:
    """Multi-layer UPDE driven by a KnmSpec."""
    spec: KnmSpec
    dt: float = 1e-3
    psi_mode: str = "external"
    wrap: bool = True

    def step(
        self,
        theta_layers: Sequence[FloatArray],
        omega_layers: Sequence[FloatArray],
        *,
        psi_driver: Optional[float] = None,
        actuation_gain: float = 1.0,
        pac_gamma: float = 0.0,
    ) -> dict:
        """Advance all L layers by one Euler step.

        Parameters
        ----------
        theta_layers : sequence of 1D arrays
            Phase vectors per layer.
        omega_layers : sequence of 1D arrays
            Natural frequencies per layer.
        psi_driver : float or None
            External global field phase Ψ (required if psi_mode="external").
        actuation_gain : float
            Multiplicative gain on all coupling terms.
        pac_gamma : float
            PAC-like gating: boost inter-layer coupling by
            (1 + pac_gamma·(1 − R_source)).
        """
        K = np.asarray(self.spec.K, dtype=np.float64)
        L = K.shape[0]
        if len(theta_layers) != L or len(omega_layers) != L:
            raise ValueError(f"Expected {L} layers, got {len(theta_layers)}")

        g = float(actuation_gain)

        # Per-layer order parameters
        Rm = np.empty(L)
        Psim = np.empty(L)
        for m in range(L):
            Rm[m], Psim[m] = order_parameter(theta_layers[m])

        # Resolve global Ψ
        if self.psi_mode == "external":
            if psi_driver is None:
                raise ValueError("psi_driver required when psi_mode='external'")
            Psi_global = float(psi_driver)
        elif self.psi_mode == "global_mean_field":
            z = np.sum(Rm * np.exp(1j * Psim))
            Psi_global = float(np.angle(z))
        else:
            raise ValueError(f"Unknown psi_mode: {self.psi_mode}")

        alpha = np.zeros_like(K) if self.spec.alpha is None else np.asarray(self.spec.alpha, dtype=np.float64)
        zeta = np.zeros(L) if self.spec.zeta is None else np.asarray(self.spec.zeta, dtype=np.float64)

        theta1: List[FloatArray] = []
        dtheta_all: List[FloatArray] = []

        for m in range(L):
            th = np.asarray(theta_layers[m], dtype=np.float64).ravel()
            om = np.asarray(omega_layers[m], dtype=np.float64).ravel()

            # Intra-layer: K_{mm} R_m sin(ψ_m − θ − α_{mm})
            dth = om + g * K[m, m] * Rm[m] * np.sin(Psim[m] - th - alpha[m, m])

            # Inter-layer: Σ_{n≠m} K_{nm} R_n sin(ψ_n − θ − α_{nm})
            for n in range(L):
                if n == m:
                    continue
                pac_gate = 1.0 + pac_gamma * (1.0 - Rm[n])
                dth += g * pac_gate * K[n, m] * Rm[n] * np.sin(Psim[n] - th - alpha[n, m])

            # Global driver: ζ_m sin(Ψ − θ)
            if zeta[m] != 0.0:
                dth += zeta[m] * np.sin(Psi_global - th)

            th_next = th + self.dt * dth
            if self.wrap:
                th_next = wrap_phase(th_next)

            theta1.append(th_next)
            dtheta_all.append(dth)

        R_global, Psi_r_global = order_parameter(
            np.concatenate([np.asarray(t).ravel() for t in theta_layers])
        )

        return {
            "theta1": theta1,
            "dtheta": dtheta_all,
            "R_layer": Rm.copy(),
            "Psi_layer": Psim.copy(),
            "R_global": R_global,
            "Psi_global": Psi_global,
        }

    def run(
        self,
        n_steps: int,
        theta_layers: Sequence[FloatArray],
        omega_layers: Sequence[FloatArray],
        *,
        psi_driver: Optional[float] = None,
        actuation_gain: float = 1.0,
        pac_gamma: float = 0.0,
    ) -> dict:
        """Run n_steps and return trajectory of per-layer R and global R."""
        R_layer_hist = []
        R_global_hist = []
        current = [np.asarray(t, dtype=np.float64).copy() for t in theta_layers]

        for _ in range(n_steps):
            out = self.step(
                current, omega_layers,
                psi_driver=psi_driver,
                actuation_gain=actuation_gain,
                pac_gamma=pac_gamma,
            )
            current = out["theta1"]
            R_layer_hist.append(out["R_layer"].copy())
            R_global_hist.append(out["R_global"])

        return {
            "theta_final": current,
            "R_layer_hist": np.array(R_layer_hist),
            "R_global_hist": np.array(R_global_hist),
        }
