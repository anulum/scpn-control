# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Control-owned capacitor-bank RLC state model.
"""Bounded capacitor-bank state model for CONTROL pulsed-shot contracts."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class _CompatStrEnum(str, Enum):
        """Local Python 3.10-compatible subset of enum.StrEnum."""

    StrEnum = _CompatStrEnum


from typing import Literal

from scpn_control.control.pulsed_scenario_scheduler_v2 import CapacitorBankTelemetry


class RLCRegime(StrEnum):
    """Canonical damping regimes for the series RLC bank model."""

    UNDERDAMPED = "underdamped"
    CRITICAL = "critical"
    OVERDAMPED = "overdamped"


WaveformName = Literal["rect", "half_sine", "exp_decay"]
ENERGY_BALANCE_REL_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class CapacitorBankSpec:
    """Physical parameters for a bounded series RLC capacitor bank."""

    capacitance_F: float
    inductance_H: float
    series_resistance_ohm: float
    voltage_max_V: float
    recharge_power_kW: float = 0.0

    def __post_init__(self) -> None:
        capacitance = _positive("capacitance_F", self.capacitance_F)
        inductance = _positive("inductance_H", self.inductance_H)
        resistance = _non_negative("series_resistance_ohm", self.series_resistance_ohm)
        voltage_max = _positive("voltage_max_V", self.voltage_max_V)
        recharge_power = _non_negative("recharge_power_kW", self.recharge_power_kW)
        object.__setattr__(self, "capacitance_F", capacitance)
        object.__setattr__(self, "inductance_H", inductance)
        object.__setattr__(self, "series_resistance_ohm", resistance)
        object.__setattr__(self, "voltage_max_V", voltage_max)
        object.__setattr__(self, "recharge_power_kW", recharge_power)

    @property
    def natural_impedance_ohm(self) -> float:
        r"""Return :math:`Z_0 = \sqrt{L/C}`."""
        return math.sqrt(self.inductance_H / self.capacitance_F)

    @property
    def undamped_angular_frequency_rad_s(self) -> float:
        r"""Return :math:`\omega_0 = 1/\sqrt{LC}`."""
        return 1.0 / math.sqrt(self.inductance_H * self.capacitance_F)

    @property
    def damping_ratio(self) -> float:
        """Return the dimensionless series-RLC damping ratio."""
        return 0.5 * self.series_resistance_ohm * math.sqrt(self.capacitance_F / self.inductance_H)

    @property
    def regime(self) -> RLCRegime:
        """Classify the bank damping regime from the physical coefficients."""
        critical_r = 2.0 * self.natural_impedance_ohm
        tolerance = 1e-12 * max(1.0, critical_r)
        if abs(self.series_resistance_ohm - critical_r) <= tolerance:
            return RLCRegime.CRITICAL
        if self.series_resistance_ohm < critical_r:
            return RLCRegime.UNDERDAMPED
        return RLCRegime.OVERDAMPED


@dataclass(frozen=True)
class CapacitorBankState:
    """Instantaneous bank state in SI units."""

    t: float
    voltage_V: float
    current_A: float
    di_dt_A_s: float
    capacitance_F: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _non_negative("t", self.t))
        object.__setattr__(self, "voltage_V", _finite("voltage_V", self.voltage_V))
        object.__setattr__(self, "current_A", _finite("current_A", self.current_A))
        object.__setattr__(self, "di_dt_A_s", _finite("di_dt_A_s", self.di_dt_A_s))
        object.__setattr__(self, "capacitance_F", _positive("capacitance_F", self.capacitance_F))

    @property
    def energy_J(self) -> float:
        """Return capacitor stored energy :math:`0.5 C V^2`."""
        return 0.5 * self.capacitance_F * self.voltage_V * self.voltage_V


@dataclass(frozen=True)
class PulseSpec:
    """Prescribed load-current waveform for bounded discharge simulations."""

    peak_current_A: float
    duration_s: float
    waveform: WaveformName = "half_sine"

    def __post_init__(self) -> None:
        object.__setattr__(self, "peak_current_A", _positive("peak_current_A", self.peak_current_A))
        object.__setattr__(self, "duration_s", _positive("duration_s", self.duration_s))
        if self.waveform not in ("rect", "half_sine", "exp_decay"):
            raise ValueError("waveform must be one of: rect, half_sine, exp_decay")


@dataclass(frozen=True)
class EnergyReport:
    """Energy bookkeeping returned by a discharge simulation."""

    energy_delivered_J: float
    energy_initial_J: float
    energy_remaining_J: float
    capacitor_energy_remaining_J: float
    inductor_energy_remaining_J: float
    resistive_loss_J: float
    load_energy_J: float
    energy_balance_residual_J: float
    energy_balance_relative_error: float
    energy_balance_passed: bool
    peak_voltage_V: float
    peak_current_A: float
    discharge_duration_s: float
    rlc_regime: RLCRegime

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_delivered_J", _finite("energy_delivered_J", self.energy_delivered_J))
        object.__setattr__(self, "energy_initial_J", _non_negative("energy_initial_J", self.energy_initial_J))
        object.__setattr__(self, "energy_remaining_J", _non_negative("energy_remaining_J", self.energy_remaining_J))
        object.__setattr__(
            self,
            "capacitor_energy_remaining_J",
            _non_negative("capacitor_energy_remaining_J", self.capacitor_energy_remaining_J),
        )
        object.__setattr__(
            self,
            "inductor_energy_remaining_J",
            _non_negative("inductor_energy_remaining_J", self.inductor_energy_remaining_J),
        )
        object.__setattr__(self, "resistive_loss_J", _non_negative("resistive_loss_J", self.resistive_loss_J))
        object.__setattr__(self, "load_energy_J", _finite("load_energy_J", self.load_energy_J))
        object.__setattr__(
            self,
            "energy_balance_residual_J",
            _finite("energy_balance_residual_J", self.energy_balance_residual_J),
        )
        object.__setattr__(
            self,
            "energy_balance_relative_error",
            _non_negative("energy_balance_relative_error", self.energy_balance_relative_error),
        )
        object.__setattr__(self, "peak_voltage_V", _non_negative("peak_voltage_V", self.peak_voltage_V))
        object.__setattr__(self, "peak_current_A", _non_negative("peak_current_A", self.peak_current_A))
        object.__setattr__(
            self, "discharge_duration_s", _non_negative("discharge_duration_s", self.discharge_duration_s)
        )
        object.__setattr__(self, "energy_balance_passed", bool(self.energy_balance_passed))


def free_response(spec: CapacitorBankSpec, v0: float, i0: float, t: float) -> CapacitorBankState:
    """Evaluate the closed-form homogeneous series-RLC response at time ``t``."""
    time = _non_negative("t", t)
    voltage0 = _finite("v0", v0)
    current0 = _finite("i0", i0)
    capacitance = spec.capacitance_F
    inductance = spec.inductance_H
    resistance = spec.series_resistance_ohm
    alpha = resistance / (2.0 * inductance)
    omega0 = spec.undamped_angular_frequency_rad_s
    dv0 = -current0 / capacitance

    if spec.regime is RLCRegime.UNDERDAMPED:
        omega_d = math.sqrt(max(omega0 * omega0 - alpha * alpha, 0.0))
        exp_term = math.exp(-alpha * time)
        if omega_d == 0.0:
            return _critical_response(spec, voltage0, current0, time)
        coeff_b = (dv0 + alpha * voltage0) / omega_d
        cos_term = math.cos(omega_d * time)
        sin_term = math.sin(omega_d * time)
        voltage = exp_term * (voltage0 * cos_term + coeff_b * sin_term)
        dv_dt = exp_term * (
            -alpha * (voltage0 * cos_term + coeff_b * sin_term)
            + (-voltage0 * omega_d * sin_term + coeff_b * omega_d * cos_term)
        )
    elif spec.regime is RLCRegime.CRITICAL:
        return _critical_response(spec, voltage0, current0, time)
    else:
        root_delta = math.sqrt(max(alpha * alpha - omega0 * omega0, 0.0))
        root_1 = -alpha + root_delta
        root_2 = -alpha - root_delta
        if root_1 == root_2:
            return _critical_response(spec, voltage0, current0, time)
        coeff_a = (dv0 - root_2 * voltage0) / (root_1 - root_2)
        coeff_b = voltage0 - coeff_a
        exp_1 = math.exp(root_1 * time)
        exp_2 = math.exp(root_2 * time)
        voltage = coeff_a * exp_1 + coeff_b * exp_2
        dv_dt = coeff_a * root_1 * exp_1 + coeff_b * root_2 * exp_2

    current = -capacitance * dv_dt
    di_dt = voltage / inductance - resistance * current / inductance
    return CapacitorBankState(
        t=time,
        voltage_V=voltage,
        current_A=current,
        di_dt_A_s=di_dt,
        capacitance_F=capacitance,
    )


@dataclass(frozen=True)
class _ExactRLCStep:
    r"""Cached exact zero-order-hold discretisation of the series-RLC bank.

    The bank obeys the linear time-invariant system

    .. math::

        \frac{d}{dt}\begin{bmatrix} v \\ i \end{bmatrix}
        = A \begin{bmatrix} v \\ i \end{bmatrix} + B\,u,
        \quad
        A = \begin{bmatrix} 0 & -1/C \\ 1/L & -R/L \end{bmatrix},
        \quad
        B = \begin{bmatrix} -1/C \\ 0 \end{bmatrix},

    with ``u`` the prescribed load current. The exact zero-order-hold update over
    a step ``dt`` with constant load is :math:`x_{n+1} = \Phi x_n + \Gamma u`,
    :math:`\Phi = e^{A\,dt}`, :math:`\Gamma = A^{-1}(\Phi - I)B`. The state
    transition ``Phi`` is taken from the closed-form :func:`free_response`, so the
    stepper reproduces the analytic homogeneous solution to machine precision.
    The energy ledger is closed analytically as well: ``intv_*`` give
    :math:`\int_0^{dt} v\,dt` and ``w_*`` is the finite-horizon current gramian
    :math:`\int_0^{dt} i(t)^2\,dt` of the augmented state :math:`[v, i, u]`.
    """

    dt: float
    phi11: float
    phi12: float
    phi21: float
    phi22: float
    gamma_v: float
    gamma_i: float
    intv_v0: float
    intv_i0: float
    intv_u: float
    w_vv: float
    w_vi: float
    w_vu: float
    w_ii: float
    w_iu: float
    w_uu: float


def _build_exact_rlc_step(spec: CapacitorBankSpec, dt: float) -> _ExactRLCStep:
    """Build the exact zero-order-hold discretisation for ``spec`` at step ``dt``."""
    import numpy as np
    from scipy.linalg import expm

    capacitance = spec.capacitance_F
    inductance = spec.inductance_H
    resistance = spec.series_resistance_ohm

    # Phi = exp(A dt) from the closed-form homogeneous response columns.
    col_v = free_response(spec, 1.0, 0.0, dt)
    col_i = free_response(spec, 0.0, 1.0, dt)
    phi11, phi12 = col_v.voltage_V, col_i.voltage_V
    phi21, phi22 = col_v.current_A, col_i.current_A

    # A^{-1} = [[-RC, L], [-C, 0]]; B = [-1/C, 0]. M1 = A^{-1}(Phi - I) = integral of Phi.
    ainv_11, ainv_12 = -resistance * capacitance, inductance
    ainv_21, ainv_22 = -capacitance, 0.0
    pm11, pm12, pm21, pm22 = phi11 - 1.0, phi12, phi21, phi22 - 1.0
    m1_11 = ainv_11 * pm11 + ainv_12 * pm21
    m1_12 = ainv_11 * pm12 + ainv_12 * pm22
    m1_21 = ainv_21 * pm11 + ainv_22 * pm21
    m1_22 = ainv_21 * pm12 + ainv_22 * pm22

    b0 = -1.0 / capacitance
    gamma_v = m1_11 * b0
    gamma_i = m1_21 * b0

    # N = A^{-1}(M1 - dt I) B, giving the constant-load contribution to int v dt.
    md_11, md_12, md_21, md_22 = m1_11 - dt, m1_12, m1_21, m1_22 - dt
    mdb0 = md_11 * b0
    mdb1 = md_21 * b0
    n0 = ainv_11 * mdb0 + ainv_12 * mdb1

    # Finite-horizon current gramian W via Van Loan's augmented matrix exponential,
    # so int_0^dt i(t)^2 dt = z0^T W z0 with z0 = [v, i, u].
    a_mat = np.array([[0.0, b0], [1.0 / inductance, -resistance / inductance]], dtype=np.float64)
    a_tilde = np.zeros((3, 3), dtype=np.float64)
    a_tilde[:2, :2] = a_mat
    a_tilde[0, 2] = b0
    pick = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    weight = np.outer(pick, pick)
    block = np.zeros((6, 6), dtype=np.float64)
    block[:3, :3] = -a_tilde.T
    block[:3, 3:] = weight
    block[3:, 3:] = a_tilde
    expo = expm(block * dt)
    gramian = expo[3:, 3:].T @ expo[:3, 3:]
    gramian = 0.5 * (gramian + gramian.T)

    return _ExactRLCStep(
        dt=float(dt),
        phi11=phi11,
        phi12=phi12,
        phi21=phi21,
        phi22=phi22,
        gamma_v=float(gamma_v),
        gamma_i=float(gamma_i),
        intv_v0=float(m1_11),
        intv_i0=float(m1_12),
        intv_u=float(n0),
        w_vv=float(gramian[0, 0]),
        w_vi=float(gramian[0, 1]),
        w_vu=float(gramian[0, 2]),
        w_ii=float(gramian[1, 1]),
        w_iu=float(gramian[1, 2]),
        w_uu=float(gramian[2, 2]),
    )


class CapacitorBank:
    """Mutable bounded series-RLC bank with analytical and numerical stepping."""

    def __init__(
        self,
        spec: CapacitorBankSpec,
        initial_voltage_V: float = 0.0,
        initial_current_A: float = 0.0,
    ) -> None:
        self._spec = spec
        self._t = 0.0
        self._v = 0.0
        self._i = 0.0
        self._di_dt = 0.0
        self._disc: _ExactRLCStep | None = None
        self.reset(initial_voltage_V, initial_current_A=initial_current_A)

    def _discretization(self, dt: float) -> _ExactRLCStep:
        """Return the cached exact discretisation for ``dt``, rebuilding on change."""
        cached = self._disc
        if cached is None or cached.dt != dt:
            cached = _build_exact_rlc_step(self._spec, dt)
            self._disc = cached
        return cached

    @property
    def spec(self) -> CapacitorBankSpec:
        """Return the immutable bank specification."""
        return self._spec

    @property
    def state(self) -> CapacitorBankState:
        """Return an immutable snapshot of the current bank state."""
        return CapacitorBankState(
            t=self._t,
            voltage_V=self._v,
            current_A=self._i,
            di_dt_A_s=self._di_dt,
            capacitance_F=self._spec.capacitance_F,
        )

    def telemetry(self) -> CapacitorBankTelemetry:
        """Return scheduler-compatible bank telemetry using absolute voltage magnitude."""
        voltage_abs = abs(self._v)
        if voltage_abs > self._spec.voltage_max_V:
            raise ValueError("bank voltage magnitude exceeds voltage_max_V")
        return CapacitorBankTelemetry(
            voltage_V=voltage_abs,
            voltage_max_V=self._spec.voltage_max_V,
            energy_J=self.state.energy_J,
        )

    def reset(self, initial_voltage_V: float = 0.0, *, initial_current_A: float = 0.0) -> CapacitorBankState:
        """Reset bank state at ``t = 0`` with bounded initial voltage."""
        voltage = _non_negative("initial_voltage_V", initial_voltage_V)
        current = _finite("initial_current_A", initial_current_A)
        if voltage > self._spec.voltage_max_V:
            raise ValueError("initial_voltage_V exceeds bank max")
        self._t = 0.0
        self._v = voltage
        self._i = current
        self._di_dt = voltage / self._spec.inductance_H - (
            self._spec.series_resistance_ohm * current / self._spec.inductance_H
        )
        return self.state

    def step(self, dt: float, *, external_load_current_A: float = 0.0) -> CapacitorBankState:
        """Advance the bank one exact zero-order-hold step with optional load current.

        The update is the exact state-transition solution of the series-RLC system
        for a load current held constant over the step, so it reproduces the
        closed-form :func:`free_response` to machine precision rather than the
        second-order Crank-Nicolson truncation it replaces.
        """
        step_s = _positive("dt", dt)
        load_current = _finite("external_load_current_A", external_load_current_A)
        disc = self._discretization(step_s)
        inductance = self._spec.inductance_H
        resistance = self._spec.series_resistance_ohm
        voltage_next = disc.phi11 * self._v + disc.phi12 * self._i + disc.gamma_v * load_current
        current_next = disc.phi21 * self._v + disc.phi22 * self._i + disc.gamma_i * load_current
        di_dt_next = voltage_next / inductance - resistance * current_next / inductance
        for name, value in (
            ("voltage_next", voltage_next),
            ("current_next", current_next),
            ("di_dt_next", di_dt_next),
        ):
            _finite(name, value)
        self._t += step_s
        self._v = voltage_next
        self._i = current_next
        self._di_dt = di_dt_next
        return self.state

    def discharge(self, pulse: PulseSpec, dt: float, n_steps: int) -> EnergyReport:
        """Drive the bank with ``pulse`` using midpoint-sampled load current.

        The bank dynamics advance with the exact zero-order-hold stepper, and each
        step's ohmic dissipation :math:`R\\int i^2\\,dt` and load extraction
        :math:`\\int v\\,u\\,dt` are integrated in closed form, so the energy ledger
        closes to machine precision instead of carrying second-order quadrature
        error.
        """
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        step_s = _positive("dt", dt)
        disc = self._discretization(step_s)
        resistance = self._spec.series_resistance_ohm
        energy_initial = self._total_stored_energy_J()
        resistive_loss = 0.0
        load_energy = 0.0
        peak_voltage = abs(self._v)
        peak_current = abs(self._i)
        pulse_t = 0.0
        for _ in range(n_steps):
            load_current = _sample_waveform(pulse, pulse_t + step_s / 2.0)
            v0 = self._v
            i0 = self._i
            int_v = disc.intv_v0 * v0 + disc.intv_i0 * i0 + disc.intv_u * load_current
            int_i_squared = (
                disc.w_vv * v0 * v0
                + disc.w_ii * i0 * i0
                + disc.w_uu * load_current * load_current
                + 2.0 * (disc.w_vi * v0 * i0 + disc.w_vu * v0 * load_current + disc.w_iu * i0 * load_current)
            )
            state = self.step(step_s, external_load_current_A=load_current)
            resistive_loss += resistance * int_i_squared
            load_energy += load_current * int_v
            peak_voltage = max(peak_voltage, abs(state.voltage_V))
            peak_current = max(peak_current, abs(state.current_A))
            pulse_t += step_s
        energy_remaining = self._total_stored_energy_J()
        capacitor_energy_remaining = self.state.energy_J
        inductor_energy_remaining = 0.5 * self._spec.inductance_H * self._i * self._i
        energy_delivered = energy_initial - energy_remaining
        residual = energy_delivered - resistive_loss - load_energy
        relative_error = _energy_balance_relative_error(
            energy_initial,
            energy_remaining,
            resistive_loss,
            load_energy,
            residual,
        )
        return EnergyReport(
            energy_delivered_J=energy_delivered,
            energy_initial_J=energy_initial,
            energy_remaining_J=energy_remaining,
            capacitor_energy_remaining_J=capacitor_energy_remaining,
            inductor_energy_remaining_J=inductor_energy_remaining,
            resistive_loss_J=resistive_loss,
            load_energy_J=load_energy,
            energy_balance_residual_J=residual,
            energy_balance_relative_error=relative_error,
            energy_balance_passed=relative_error <= ENERGY_BALANCE_REL_TOLERANCE,
            peak_voltage_V=peak_voltage,
            peak_current_A=peak_current,
            discharge_duration_s=n_steps * step_s,
            rlc_regime=self._spec.regime,
        )

    def _total_stored_energy_J(self) -> float:
        capacitor_energy = 0.5 * self._spec.capacitance_F * self._v * self._v
        inductor_energy = 0.5 * self._spec.inductance_H * self._i * self._i
        return capacitor_energy + inductor_energy

    def feasibility(self, pulse: PulseSpec) -> tuple[bool, str]:
        """Run conservative pulse admissibility guards against current bank state."""
        total_energy = self._total_stored_energy_J()
        if total_energy > 0.0:
            max_natural_current = math.sqrt(2.0 * total_energy / self._spec.inductance_H)
            if pulse.peak_current_A > max_natural_current:
                return (
                    False,
                    (
                        f"requested peak current {pulse.peak_current_A:.3g} A exceeds bank natural peak "
                        f"{max_natural_current:.3g} A from stored RLC energy {total_energy:.3g} J"
                    ),
                )
        rms_squared_factor = _waveform_rms_squared_fraction(pulse.waveform)
        rough_resistive_loss = (
            self._spec.series_resistance_ohm * pulse.peak_current_A**2 * rms_squared_factor * pulse.duration_s
        )
        available_energy = total_energy
        if rough_resistive_loss > available_energy:
            return (
                False,
                f"resistive dissipation {rough_resistive_loss:.3g} J exceeds available {available_energy:.3g} J",
            )
        return True, "ok"

    def recharge_status(self, t: float) -> dict[str, float]:
        """Project constant-power recharge state after non-negative time ``t``."""
        time = _non_negative("t", t)
        capacitance = self._spec.capacitance_F
        voltage_target = self._spec.voltage_max_V
        power_w = self._spec.recharge_power_kW * 1000.0
        energy_now = self.state.energy_J
        energy_target = 0.5 * capacitance * voltage_target * voltage_target
        if power_w <= 0.0:
            return {
                "target_voltage_V": voltage_target,
                "projected_voltage_V": abs(self._v),
                "time_to_full_s": float("inf"),
            }
        deficit = max(energy_target - energy_now, 0.0)
        time_to_full = deficit / power_w
        if time >= time_to_full:
            projected_voltage = voltage_target
        else:
            projected_energy = energy_now + power_w * time
            projected_voltage = math.sqrt(2.0 * projected_energy / capacitance)
        return {
            "target_voltage_V": voltage_target,
            "projected_voltage_V": projected_voltage,
            "time_to_full_s": time_to_full,
        }


def _critical_response(
    spec: CapacitorBankSpec,
    voltage0: float,
    current0: float,
    time: float,
) -> CapacitorBankState:
    capacitance = spec.capacitance_F
    inductance = spec.inductance_H
    resistance = spec.series_resistance_ohm
    alpha = resistance / (2.0 * inductance)
    dv0 = -current0 / capacitance
    coeff = dv0 + alpha * voltage0
    exp_term = math.exp(-alpha * time)
    voltage = exp_term * (voltage0 + coeff * time)
    dv_dt = exp_term * (coeff - alpha * (voltage0 + coeff * time))
    current = -capacitance * dv_dt
    di_dt = voltage / inductance - resistance * current / inductance
    return CapacitorBankState(
        t=time,
        voltage_V=voltage,
        current_A=current,
        di_dt_A_s=di_dt,
        capacitance_F=capacitance,
    )


def _sample_waveform(pulse: PulseSpec, t: float) -> float:
    """Return the load current at time ``t`` since pulse start."""
    time = _finite("t", t)
    if time < 0.0 or time > pulse.duration_s:
        return 0.0
    if pulse.waveform == "rect":
        return pulse.peak_current_A
    if pulse.waveform == "half_sine":
        return pulse.peak_current_A * math.sin(math.pi * time / pulse.duration_s)
    if pulse.waveform == "exp_decay":
        tau = pulse.duration_s / 5.0
        return pulse.peak_current_A * math.exp(-time / tau)
    raise ValueError(f"unknown waveform: {pulse.waveform!r}")


def _waveform_rms_squared_fraction(waveform: WaveformName) -> float:
    """Return the waveform mean-square factor relative to peak current."""
    if waveform == "rect":
        return 1.0
    if waveform == "half_sine":
        return 0.5
    if waveform == "exp_decay":
        return 0.25 * (1.0 - math.exp(-10.0))
    raise ValueError(f"unknown waveform: {waveform!r}")


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _energy_balance_relative_error(
    energy_initial: float,
    energy_remaining: float,
    resistive_loss: float,
    load_energy: float,
    residual: float,
) -> float:
    scale = max(
        abs(energy_initial),
        abs(energy_remaining),
        abs(resistive_loss) + abs(load_energy) + abs(energy_initial - energy_remaining),
        1.0,
    )
    return abs(residual) / scale


__all__ = [
    "CapacitorBank",
    "CapacitorBankSpec",
    "CapacitorBankState",
    "ENERGY_BALANCE_REL_TOLERANCE",
    "EnergyReport",
    "PulseSpec",
    "RLCRegime",
    "WaveformName",
    "free_response",
]
