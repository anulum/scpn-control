# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin telemetry and emulation.

"""Telemetry contracts and deterministic emulation for the digital twin."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control.core._validators import require_int

_VALID_MACHINES = {"NSTX-U", "SPARC"}


@dataclass(frozen=True)
class TelemetryPacket:
    """A single digital-twin telemetry sample.

    Attributes
    ----------
    t_ms
        Timestamp in milliseconds.
    machine
        Source device name (``"NSTX-U"`` or ``"SPARC"``).
    ip_ma
        Plasma current in MA.
    beta_n
        Normalised beta.
    q95
        Edge safety factor q95.
    density_1e19
        Line-averaged density in 10¹⁹ m⁻³.

    Notes
    -----
    The timestamp is a nonnegative integer and the machine name is canonical.
    All four measured scalars must be finite; this contract does not establish
    measured-source provenance or facility calibration.
    """

    t_ms: int
    machine: str
    ip_ma: float
    beta_n: float
    q95: float
    density_1e19: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_ms", require_int("t_ms", self.t_ms, 0))
        if self.machine != _normalize_machine(self.machine):
            raise ValueError("TelemetryPacket machine must use a canonical device name")
        # A non-finite field (NaN/inf) would poison the risk signal — max(nan, 0) is nan
        # and nan comparisons fail OPEN in the mitigation gate — so reject it at
        # construction rather than let it reach risk scoring.
        for value, name in (
            (self.ip_ma, "ip_ma"),
            (self.beta_n, "beta_n"),
            (self.q95, "q95"),
            (self.density_1e19, "density_1e19"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"TelemetryPacket {name} must be finite")


def _normalize_machine(machine: str) -> str:
    if not isinstance(machine, str):
        raise ValueError("machine must be 'NSTX-U' or 'SPARC'")
    machine_key = machine.strip().upper()
    if machine_key not in _VALID_MACHINES:
        raise ValueError("machine must be 'NSTX-U' or 'SPARC'")
    return machine_key


def generate_emulated_stream(
    machine: str,
    *,
    samples: int = 320,
    dt_ms: int = 5,
    seed: int = 42,
) -> list[TelemetryPacket]:
    """Generate a synthetic telemetry stream for a device.

    Parameters
    ----------
    machine
        Device name (``"NSTX-U"`` or ``"SPARC"``).
    samples
        Integral number of telemetry packets to generate, at least 32.
    dt_ms
        Integral sample spacing in milliseconds, at least 1.
    seed
        Nonnegative integer random seed for reproducibility.

    Returns
    -------
    list[TelemetryPacket]
        The emulated telemetry packets.
    """
    machine_key = _normalize_machine(machine)

    seed = require_int("seed", seed, 0)
    samples = require_int("samples", samples)
    if samples < 32:
        raise ValueError("samples must be >= 32.")
    dt_ms = require_int("dt_ms", dt_ms)
    if dt_ms < 1:
        raise ValueError("dt_ms must be >= 1.")
    rng = np.random.default_rng(seed)

    # Menard et al., Nucl. Fusion 52, 083015 (2012): NSTX-U H-mode baseline
    if machine_key == "NSTX-U":
        ip_base, beta_base, q95_base, dens_base = 1.2, 1.95, 4.7, 6.5
    else:
        # Creely et al., J. Plasma Phys. 86, 865860502 (2020): SPARC V2C design
        ip_base, beta_base, q95_base, dens_base = 8.7, 1.65, 3.9, 8.2

    packets: list[TelemetryPacket] = []
    for k in range(samples):
        phase = k / max(samples - 1, 1)
        disruption_burst = 0.0
        if 0.58 <= phase <= 0.76:
            disruption_burst = 0.18 * np.sin(np.pi * (phase - 0.58) / 0.18)

        packets.append(
            TelemetryPacket(
                t_ms=k * dt_ms,
                machine=machine_key,
                ip_ma=float(ip_base + 0.03 * np.sin(2.0 * np.pi * phase) + rng.normal(0.0, 0.004)),
                beta_n=float(beta_base + 0.05 * np.cos(2.0 * np.pi * 1.4 * phase) + disruption_burst),
                q95=float(q95_base - 0.12 * disruption_burst + rng.normal(0.0, 0.01)),
                density_1e19=float(dens_base + 0.10 * np.sin(2.0 * np.pi * 0.6 * phase)),
            )
        )
    return packets


def _apply_chaos_monkey(
    packet: TelemetryPacket,
    *,
    rng: np.random.Generator,
    dropout_prob: float,
    gaussian_noise_std: float,
) -> tuple[TelemetryPacket, int, int]:
    drop = float(np.clip(dropout_prob, 0.0, 1.0))
    sigma = max(float(gaussian_noise_std), 0.0)
    dropouts = 0
    noise_injections = 0

    def channel(value: float) -> float:
        nonlocal dropouts, noise_injections
        out = float(value)
        if drop > 0.0 and float(rng.random()) < drop:
            out = 0.0
            dropouts += 1
        if sigma > 0.0:
            out += float(rng.normal(0.0, sigma))
            noise_injections += 1
        return out

    noisy_packet = TelemetryPacket(
        t_ms=int(packet.t_ms),
        machine=str(packet.machine),
        ip_ma=channel(packet.ip_ma),
        beta_n=channel(packet.beta_n),
        q95=channel(packet.q95),
        density_1e19=max(0.0, channel(packet.density_1e19)),
    )
    return noisy_packet, int(dropouts), int(noise_injections)
