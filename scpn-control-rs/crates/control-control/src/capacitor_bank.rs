// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Control-owned capacitor-bank RLC state model.
//! Bounded capacitor-bank state model for CONTROL pulsed-shot contracts.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ndarray::Array2;

use crate::h_infinity::matrix_exp;

/// Relative residual tolerance for the exact-discretisation RLC energy-balance admission.
pub const ENERGY_BALANCE_REL_TOLERANCE: f64 = 1.0e-8;

/// Canonical damping regimes for the series RLC bank model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlcRegime {
    /// Resistance below critical damping.
    Underdamped,
    /// Resistance at the critical-damping boundary.
    Critical,
    /// Resistance above critical damping.
    Overdamped,
}

impl RlcRegime {
    /// Stable string identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Underdamped => "underdamped",
            Self::Critical => "critical",
            Self::Overdamped => "overdamped",
        }
    }
}

/// Supported prescribed load-current waveforms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PulseWaveform {
    /// Constant load current over the pulse duration.
    Rect,
    /// Half-sine current pulse.
    HalfSine,
    /// Exponential decay with tau = duration / 5.
    ExpDecay,
}

impl PulseWaveform {
    /// Parse a stable waveform identifier.
    pub fn parse(value: &str) -> Result<Self, CapacitorBankError> {
        match value.trim() {
            "rect" => Ok(Self::Rect),
            "half_sine" => Ok(Self::HalfSine),
            "exp_decay" => Ok(Self::ExpDecay),
            _ => Err(CapacitorBankError::UnknownWaveform),
        }
    }

    /// Stable string identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rect => "rect",
            Self::HalfSine => "half_sine",
            Self::ExpDecay => "exp_decay",
        }
    }
}

/// Physical parameters for a bounded series RLC capacitor bank.
#[derive(Clone, Copy, Debug)]
pub struct CapacitorBankSpec {
    /// Capacitance in farads.
    pub capacitance_f: f64,
    /// Series inductance in henries.
    pub inductance_h: f64,
    /// Series resistance in ohms.
    pub series_resistance_ohm: f64,
    /// Maximum permitted absolute bank voltage in volts.
    pub voltage_max_v: f64,
    /// Constant recharge power in kilowatts.
    pub recharge_power_kw: f64,
}

impl CapacitorBankSpec {
    /// Construct a validated bank specification.
    pub fn new(
        capacitance_f: f64,
        inductance_h: f64,
        series_resistance_ohm: f64,
        voltage_max_v: f64,
        recharge_power_kw: f64,
    ) -> Result<Self, CapacitorBankError> {
        validate_positive("capacitance_f", capacitance_f)?;
        validate_positive("inductance_h", inductance_h)?;
        validate_non_negative("series_resistance_ohm", series_resistance_ohm)?;
        validate_positive("voltage_max_v", voltage_max_v)?;
        validate_non_negative("recharge_power_kw", recharge_power_kw)?;
        Ok(Self {
            capacitance_f,
            inductance_h,
            series_resistance_ohm,
            voltage_max_v,
            recharge_power_kw,
        })
    }

    /// Natural impedance Z0 = sqrt(L/C).
    pub fn natural_impedance_ohm(self) -> f64 {
        (self.inductance_h / self.capacitance_f).sqrt()
    }

    /// Undamped angular frequency omega0 = 1/sqrt(LC).
    pub fn undamped_angular_frequency_rad_s(self) -> f64 {
        1.0 / (self.inductance_h * self.capacitance_f).sqrt()
    }

    /// Dimensionless damping ratio.
    pub fn damping_ratio(self) -> f64 {
        0.5 * self.series_resistance_ohm * (self.capacitance_f / self.inductance_h).sqrt()
    }

    /// Classify the bank damping regime.
    pub fn regime(self) -> RlcRegime {
        let critical_r = 2.0 * self.natural_impedance_ohm();
        let tolerance = 1e-12_f64 * critical_r.max(1.0);
        if (self.series_resistance_ohm - critical_r).abs() <= tolerance {
            RlcRegime::Critical
        } else if self.series_resistance_ohm < critical_r {
            RlcRegime::Underdamped
        } else {
            RlcRegime::Overdamped
        }
    }
}

/// Instantaneous bank state in SI units.
#[derive(Clone, Copy, Debug)]
pub struct CapacitorBankState {
    /// State time in seconds.
    pub t: f64,
    /// Capacitor voltage in volts.
    pub voltage_v: f64,
    /// Series current in amperes.
    pub current_a: f64,
    /// Current derivative in amperes per second.
    pub di_dt_a_s: f64,
    /// Capacitance in farads.
    pub capacitance_f: f64,
}

impl CapacitorBankState {
    /// Stored capacitor energy in joules.
    pub fn energy_j(self) -> f64 {
        0.5 * self.capacitance_f * self.voltage_v * self.voltage_v
    }
}

/// Prescribed load-current waveform for bounded discharge simulations.
#[derive(Clone, Copy, Debug)]
pub struct PulseSpec {
    /// Peak load current in amperes.
    pub peak_current_a: f64,
    /// Pulse duration in seconds.
    pub duration_s: f64,
    /// Waveform family.
    pub waveform: PulseWaveform,
}

impl PulseSpec {
    /// Construct a validated pulse descriptor.
    pub fn new(
        peak_current_a: f64,
        duration_s: f64,
        waveform: PulseWaveform,
    ) -> Result<Self, CapacitorBankError> {
        validate_positive("peak_current_a", peak_current_a)?;
        validate_positive("duration_s", duration_s)?;
        Ok(Self {
            peak_current_a,
            duration_s,
            waveform,
        })
    }
}

/// Energy bookkeeping returned by a discharge simulation.
#[derive(Clone, Copy, Debug)]
pub struct EnergyReport {
    /// Energy removed from total RLC storage.
    pub energy_delivered_j: f64,
    /// Initial total RLC stored energy.
    pub energy_initial_j: f64,
    /// Remaining total RLC stored energy.
    pub energy_remaining_j: f64,
    /// Remaining capacitor electric energy.
    pub capacitor_energy_remaining_j: f64,
    /// Remaining inductor magnetic energy.
    pub inductor_energy_remaining_j: f64,
    /// Integrated ohmic dissipation over the discharge.
    pub resistive_loss_j: f64,
    /// Integrated prescribed load extraction over the discharge.
    pub load_energy_j: f64,
    /// Energy ledger residual: delivered - resistive_loss - load_energy.
    pub energy_balance_residual_j: f64,
    /// Scale-normalised absolute residual.
    pub energy_balance_relative_error: f64,
    /// Whether the residual passes the CONTROL admission tolerance.
    pub energy_balance_passed: bool,
    /// Peak absolute capacitor voltage observed.
    pub peak_voltage_v: f64,
    /// Peak absolute current observed.
    pub peak_current_a: f64,
    /// Simulated discharge duration.
    pub discharge_duration_s: f64,
    /// Damping regime active for the bank.
    pub rlc_regime: RlcRegime,
}

/// Cached exact zero-order-hold discretisation of the series-RLC bank.
///
/// The bank obeys `d/dt [v; i] = A [v; i] + B u` with
/// `A = [[0, -1/C], [1/L, -R/L]]`, `B = [-1/C, 0]^T`, and load current `u`.
/// The exact step over `dt` with constant load is `x_{n+1} = Phi x_n + Gamma u`,
/// `Phi = exp(A dt)` taken from the closed-form [`free_response`]. The ledger is
/// closed analytically: `intv` gives the coefficients of `int_0^dt v dt` and `w`
/// is the finite-horizon current gramian for `int_0^dt i(t)^2 dt` over the
/// augmented state `[v, i, u]`.
#[derive(Clone, Copy, Debug)]
struct ExactRlcStep {
    dt: f64,
    phi: [[f64; 2]; 2],
    gamma: [f64; 2],
    intv: [f64; 3],
    w: [[f64; 3]; 3],
}

/// Mutable bounded series-RLC bank with analytical and exact discrete stepping.
#[derive(Clone, Debug)]
pub struct CapacitorBank {
    spec: CapacitorBankSpec,
    t: f64,
    voltage_v: f64,
    current_a: f64,
    di_dt_a_s: f64,
    disc: Option<ExactRlcStep>,
}

impl CapacitorBank {
    /// Construct a bank and reset to the requested initial state.
    pub fn new(
        spec: CapacitorBankSpec,
        initial_voltage_v: f64,
        initial_current_a: f64,
    ) -> Result<Self, CapacitorBankError> {
        let mut bank = Self {
            spec,
            t: 0.0,
            voltage_v: 0.0,
            current_a: 0.0,
            di_dt_a_s: 0.0,
            disc: None,
        };
        bank.reset(initial_voltage_v, initial_current_a)?;
        Ok(bank)
    }

    /// Return the immutable bank specification.
    pub fn spec(&self) -> CapacitorBankSpec {
        self.spec
    }

    /// Return the current state snapshot.
    pub fn state(&self) -> CapacitorBankState {
        CapacitorBankState {
            t: self.t,
            voltage_v: self.voltage_v,
            current_a: self.current_a,
            di_dt_a_s: self.di_dt_a_s,
            capacitance_f: self.spec.capacitance_f,
        }
    }

    /// Return scheduler-compatible telemetry fields: |voltage|, voltage_max, energy.
    pub fn telemetry_fields(&self) -> Result<(f64, f64, f64), CapacitorBankError> {
        let voltage_abs = self.voltage_v.abs();
        if voltage_abs > self.spec.voltage_max_v {
            return Err(CapacitorBankError::VoltageExceedsMax);
        }
        Ok((
            voltage_abs,
            self.spec.voltage_max_v,
            self.state().energy_j(),
        ))
    }

    /// Reset bank state at t = 0.
    pub fn reset(
        &mut self,
        initial_voltage_v: f64,
        initial_current_a: f64,
    ) -> Result<CapacitorBankState, CapacitorBankError> {
        validate_non_negative("initial_voltage_v", initial_voltage_v)?;
        validate_finite("initial_current_a", initial_current_a)?;
        if initial_voltage_v > self.spec.voltage_max_v {
            return Err(CapacitorBankError::InitialVoltageExceedsMax);
        }
        self.t = 0.0;
        self.voltage_v = initial_voltage_v;
        self.current_a = initial_current_a;
        self.di_dt_a_s = initial_voltage_v / self.spec.inductance_h
            - (self.spec.series_resistance_ohm * initial_current_a / self.spec.inductance_h);
        Ok(self.state())
    }

    /// Return the cached exact discretisation for `dt`, rebuilding on change.
    fn discretization(&mut self, dt: f64) -> Result<ExactRlcStep, CapacitorBankError> {
        if let Some(disc) = self.disc {
            if disc.dt == dt {
                return Ok(disc);
            }
        }
        let disc = build_exact_rlc_step(self.spec, dt)?;
        self.disc = Some(disc);
        Ok(disc)
    }

    /// Advance the bank one exact zero-order-hold step with optional load current.
    ///
    /// The update is the exact state-transition solution of the series-RLC system
    /// for a load current held constant over the step, so it reproduces the
    /// closed-form [`free_response`] to machine precision rather than the
    /// second-order Crank-Nicolson truncation it replaces.
    pub fn step(
        &mut self,
        dt: f64,
        external_load_current_a: f64,
    ) -> Result<CapacitorBankState, CapacitorBankError> {
        validate_positive("dt", dt)?;
        validate_finite("external_load_current_a", external_load_current_a)?;
        let disc = self.discretization(dt)?;
        let inductance = self.spec.inductance_h;
        let resistance = self.spec.series_resistance_ohm;
        let voltage_next = disc.phi[0][0] * self.voltage_v
            + disc.phi[0][1] * self.current_a
            + disc.gamma[0] * external_load_current_a;
        let current_next = disc.phi[1][0] * self.voltage_v
            + disc.phi[1][1] * self.current_a
            + disc.gamma[1] * external_load_current_a;
        let di_dt_next = voltage_next / inductance - resistance * current_next / inductance;
        validate_finite("voltage_next", voltage_next)?;
        validate_finite("current_next", current_next)?;
        validate_finite("di_dt_next", di_dt_next)?;
        self.t += dt;
        self.voltage_v = voltage_next;
        self.current_a = current_next;
        self.di_dt_a_s = di_dt_next;
        Ok(self.state())
    }

    /// Drive the bank with a midpoint-sampled prescribed load-current pulse.
    pub fn discharge(
        &mut self,
        pulse: PulseSpec,
        dt: f64,
        n_steps: usize,
    ) -> Result<EnergyReport, CapacitorBankError> {
        validate_positive("dt", dt)?;
        if n_steps == 0 {
            return Err(CapacitorBankError::NonPositiveInteger("n_steps"));
        }
        let disc = self.discretization(dt)?;
        let resistance = self.spec.series_resistance_ohm;
        let energy_initial = self.total_stored_energy_j();
        let mut resistive_loss = 0.0;
        let mut load_energy = 0.0;
        let mut peak_voltage = self.voltage_v.abs();
        let mut peak_current = self.current_a.abs();
        let mut pulse_t = 0.0;
        for _ in 0..n_steps {
            let load_current = sample_waveform(pulse, pulse_t + dt / 2.0)?;
            let v0 = self.voltage_v;
            let i0 = self.current_a;
            let int_v = disc.intv[0] * v0 + disc.intv[1] * i0 + disc.intv[2] * load_current;
            let int_i_squared = disc.w[0][0] * v0 * v0
                + disc.w[1][1] * i0 * i0
                + disc.w[2][2] * load_current * load_current
                + 2.0
                    * (disc.w[0][1] * v0 * i0
                        + disc.w[0][2] * v0 * load_current
                        + disc.w[1][2] * i0 * load_current);
            let state = self.step(dt, load_current)?;
            resistive_loss += resistance * int_i_squared;
            load_energy += load_current * int_v;
            peak_voltage = peak_voltage.max(state.voltage_v.abs());
            peak_current = peak_current.max(state.current_a.abs());
            pulse_t += dt;
        }
        let energy_remaining = self.total_stored_energy_j();
        let capacitor_energy_remaining = self.state().energy_j();
        let inductor_energy_remaining =
            0.5 * self.spec.inductance_h * self.current_a * self.current_a;
        let energy_delivered = energy_initial - energy_remaining;
        let residual = energy_delivered - resistive_loss - load_energy;
        let relative_error = energy_balance_relative_error(
            energy_initial,
            energy_remaining,
            resistive_loss,
            load_energy,
            residual,
        );
        Ok(EnergyReport {
            energy_delivered_j: energy_delivered,
            energy_initial_j: energy_initial,
            energy_remaining_j: energy_remaining,
            capacitor_energy_remaining_j: capacitor_energy_remaining,
            inductor_energy_remaining_j: inductor_energy_remaining,
            resistive_loss_j: resistive_loss,
            load_energy_j: load_energy,
            energy_balance_residual_j: residual,
            energy_balance_relative_error: relative_error,
            energy_balance_passed: relative_error <= ENERGY_BALANCE_REL_TOLERANCE,
            peak_voltage_v: peak_voltage,
            peak_current_a: peak_current,
            discharge_duration_s: n_steps as f64 * dt,
            rlc_regime: self.spec.regime(),
        })
    }

    /// Run conservative pulse admissibility guards against current bank state.
    pub fn feasibility(&self, pulse: PulseSpec) -> Result<(bool, String), CapacitorBankError> {
        let total_energy = self.total_stored_energy_j();
        if total_energy > 0.0 {
            let max_natural_current = (2.0 * total_energy / self.spec.inductance_h).sqrt();
            if pulse.peak_current_a > max_natural_current {
                return Ok((
                    false,
                    format!(
                        "requested peak current {:.3e} A exceeds bank natural peak {:.3e} A from stored RLC energy {:.3e} J",
                        pulse.peak_current_a, max_natural_current, total_energy
                    ),
                ));
            }
        }
        let rms_squared_factor = waveform_rms_squared_fraction(pulse.waveform);
        let rough_resistive_loss = self.spec.series_resistance_ohm
            * pulse.peak_current_a
            * pulse.peak_current_a
            * rms_squared_factor
            * pulse.duration_s;
        let available_energy = total_energy;
        if rough_resistive_loss > available_energy {
            return Ok((
                false,
                format!(
                    "resistive dissipation {:.3e} J exceeds available {:.3e} J",
                    rough_resistive_loss, available_energy
                ),
            ));
        }
        Ok((true, "ok".to_string()))
    }

    fn total_stored_energy_j(&self) -> f64 {
        let capacitor_energy = 0.5 * self.spec.capacitance_f * self.voltage_v * self.voltage_v;
        let inductor_energy = 0.5 * self.spec.inductance_h * self.current_a * self.current_a;
        capacitor_energy + inductor_energy
    }

    /// Project constant-power recharge state after non-negative time t.
    pub fn recharge_status(&self, t: f64) -> Result<RechargeStatus, CapacitorBankError> {
        validate_non_negative("t", t)?;
        let capacitance = self.spec.capacitance_f;
        let target_voltage_v = self.spec.voltage_max_v;
        let power_w = self.spec.recharge_power_kw * 1000.0;
        let energy_now = self.state().energy_j();
        let energy_target = 0.5 * capacitance * target_voltage_v * target_voltage_v;
        if power_w <= 0.0 {
            return Ok(RechargeStatus {
                target_voltage_v,
                projected_voltage_v: self.voltage_v.abs(),
                time_to_full_s: f64::INFINITY,
            });
        }
        let deficit = (energy_target - energy_now).max(0.0);
        let time_to_full_s = deficit / power_w;
        let projected_voltage_v = if t >= time_to_full_s {
            target_voltage_v
        } else {
            let projected_energy = energy_now + power_w * t;
            (2.0 * projected_energy / capacitance).sqrt()
        };
        Ok(RechargeStatus {
            target_voltage_v,
            projected_voltage_v,
            time_to_full_s,
        })
    }
}

/// Recharge projection fields.
#[derive(Clone, Copy, Debug)]
pub struct RechargeStatus {
    /// Full-charge voltage target.
    pub target_voltage_v: f64,
    /// Projected voltage after elapsed recharge time.
    pub projected_voltage_v: f64,
    /// Time required to reach target from current energy.
    pub time_to_full_s: f64,
}

/// Evaluate the closed-form homogeneous series-RLC response at time t.
pub fn free_response(
    spec: CapacitorBankSpec,
    v0: f64,
    i0: f64,
    t: f64,
) -> Result<CapacitorBankState, CapacitorBankError> {
    validate_finite("v0", v0)?;
    validate_finite("i0", i0)?;
    validate_non_negative("t", t)?;
    let capacitance = spec.capacitance_f;
    let inductance = spec.inductance_h;
    let resistance = spec.series_resistance_ohm;
    let alpha = resistance / (2.0 * inductance);
    let omega0 = spec.undamped_angular_frequency_rad_s();
    let dv0 = -i0 / capacitance;

    let (voltage, dv_dt) = match spec.regime() {
        RlcRegime::Underdamped => {
            let omega_d = (omega0 * omega0 - alpha * alpha).max(0.0).sqrt();
            if omega_d == 0.0 {
                return Ok(critical_response(spec, v0, i0, t));
            }
            let exp_term = (-alpha * t).exp();
            let coeff_b = (dv0 + alpha * v0) / omega_d;
            let cos_term = (omega_d * t).cos();
            let sin_term = (omega_d * t).sin();
            let voltage = exp_term * (v0 * cos_term + coeff_b * sin_term);
            let dv_dt = exp_term
                * (-alpha * (v0 * cos_term + coeff_b * sin_term)
                    + (-v0 * omega_d * sin_term + coeff_b * omega_d * cos_term));
            (voltage, dv_dt)
        }
        RlcRegime::Critical => return Ok(critical_response(spec, v0, i0, t)),
        RlcRegime::Overdamped => {
            let root_delta = (alpha * alpha - omega0 * omega0).max(0.0).sqrt();
            let root_1 = -alpha + root_delta;
            let root_2 = -alpha - root_delta;
            if root_1 == root_2 {
                return Ok(critical_response(spec, v0, i0, t));
            }
            let coeff_a = (dv0 - root_2 * v0) / (root_1 - root_2);
            let coeff_b = v0 - coeff_a;
            let exp_1 = (root_1 * t).exp();
            let exp_2 = (root_2 * t).exp();
            let voltage = coeff_a * exp_1 + coeff_b * exp_2;
            let dv_dt = coeff_a * root_1 * exp_1 + coeff_b * root_2 * exp_2;
            (voltage, dv_dt)
        }
    };
    let current = -capacitance * dv_dt;
    let di_dt = voltage / inductance - resistance * current / inductance;
    validate_finite("voltage", voltage)?;
    validate_finite("current", current)?;
    validate_finite("di_dt", di_dt)?;
    Ok(CapacitorBankState {
        t,
        voltage_v: voltage,
        current_a: current,
        di_dt_a_s: di_dt,
        capacitance_f: capacitance,
    })
}

/// Build the exact zero-order-hold discretisation for `spec` at step `dt`.
fn build_exact_rlc_step(
    spec: CapacitorBankSpec,
    dt: f64,
) -> Result<ExactRlcStep, CapacitorBankError> {
    let capacitance = spec.capacitance_f;
    let inductance = spec.inductance_h;
    let resistance = spec.series_resistance_ohm;

    // Phi = exp(A dt) from the closed-form homogeneous response columns.
    let col_v = free_response(spec, 1.0, 0.0, dt)?;
    let col_i = free_response(spec, 0.0, 1.0, dt)?;
    let phi = [
        [col_v.voltage_v, col_i.voltage_v],
        [col_v.current_a, col_i.current_a],
    ];

    // A^{-1} = [[-RC, L], [-C, 0]]; B = [-1/C, 0]. M1 = A^{-1}(Phi - I) = integral of Phi.
    let ainv = [[-resistance * capacitance, inductance], [-capacitance, 0.0]];
    let pm = [[phi[0][0] - 1.0, phi[0][1]], [phi[1][0], phi[1][1] - 1.0]];
    let m1 = [
        [
            ainv[0][0] * pm[0][0] + ainv[0][1] * pm[1][0],
            ainv[0][0] * pm[0][1] + ainv[0][1] * pm[1][1],
        ],
        [
            ainv[1][0] * pm[0][0] + ainv[1][1] * pm[1][0],
            ainv[1][0] * pm[0][1] + ainv[1][1] * pm[1][1],
        ],
    ];
    let b0 = -1.0 / capacitance;
    let gamma = [m1[0][0] * b0, m1[1][0] * b0];

    // N = A^{-1}(M1 - dt I) B: the constant-load contribution to int v dt.
    let md = [[m1[0][0] - dt, m1[0][1]], [m1[1][0], m1[1][1] - dt]];
    let mdb0 = md[0][0] * b0;
    let mdb1 = md[1][0] * b0;
    let n0 = ainv[0][0] * mdb0 + ainv[0][1] * mdb1;
    let intv = [m1[0][0], m1[0][1], n0];

    // Finite-horizon current gramian via Van Loan's augmented matrix exponential.
    // Augmented A_tilde = [[A, B], [0, 0]] on state [v, i, u]; weight picks i.
    let a_tilde = [
        [0.0_f64, b0, b0],
        [1.0 / inductance, -resistance / inductance, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let mut block = Array2::<f64>::zeros((6, 6));
    for r in 0..3 {
        for c in 0..3 {
            block[[r, c]] = -a_tilde[c][r];
            block[[3 + r, 3 + c]] = a_tilde[r][c];
        }
    }
    block[[1, 4]] = 1.0; // (e_i e_i^T)[i][i] with i the current index
    let expo = matrix_exp(&(block * dt));
    let mut w = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += expo[[3 + k, 3 + r]] * expo[[k, 3 + c]];
            }
            w[r][c] = acc;
        }
    }
    let w_sym = [
        [
            w[0][0],
            0.5 * (w[0][1] + w[1][0]),
            0.5 * (w[0][2] + w[2][0]),
        ],
        [
            0.5 * (w[1][0] + w[0][1]),
            w[1][1],
            0.5 * (w[1][2] + w[2][1]),
        ],
        [
            0.5 * (w[2][0] + w[0][2]),
            0.5 * (w[2][1] + w[1][2]),
            w[2][2],
        ],
    ];

    Ok(ExactRlcStep {
        dt,
        phi,
        gamma,
        intv,
        w: w_sym,
    })
}

/// Return the load current at time t since pulse start.
pub fn sample_waveform(pulse: PulseSpec, t: f64) -> Result<f64, CapacitorBankError> {
    validate_finite("t", t)?;
    if t < 0.0 || t > pulse.duration_s {
        return Ok(0.0);
    }
    let current = match pulse.waveform {
        PulseWaveform::Rect => pulse.peak_current_a,
        PulseWaveform::HalfSine => {
            pulse.peak_current_a * (std::f64::consts::PI * t / pulse.duration_s).sin()
        }
        PulseWaveform::ExpDecay => {
            let tau = pulse.duration_s / 5.0;
            pulse.peak_current_a * (-t / tau).exp()
        }
    };
    Ok(current)
}

/// Errors raised by the bounded capacitor-bank model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapacitorBankError {
    /// Field was not finite.
    NonFinite(&'static str),
    /// Field was not strictly positive.
    NonPositive(&'static str),
    /// Field was negative.
    Negative(&'static str),
    /// Integer field was not positive.
    NonPositiveInteger(&'static str),
    /// Initial voltage exceeded the declared maximum.
    InitialVoltageExceedsMax,
    /// Runtime voltage magnitude exceeded the declared maximum.
    VoltageExceedsMax,
    /// Step state matrix was singular. Retained for API stability; the exact
    /// zero-order-hold discretisation no longer constructs this variant.
    SingularStepMatrix,
    /// Unknown waveform string.
    UnknownWaveform,
}

impl Display for CapacitorBankError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::NonPositive(field) => write!(f, "{field} must be strictly positive"),
            Self::Negative(field) => write!(f, "{field} must be non-negative"),
            Self::NonPositiveInteger(field) => write!(f, "{field} must be a positive integer"),
            Self::InitialVoltageExceedsMax => write!(f, "initial_voltage_v exceeds bank max"),
            Self::VoltageExceedsMax => write!(f, "bank voltage magnitude exceeds voltage_max_v"),
            Self::SingularStepMatrix => write!(f, "RLC step matrix is singular"),
            Self::UnknownWaveform => write!(f, "unknown waveform"),
        }
    }
}

impl Error for CapacitorBankError {}

fn critical_response(
    spec: CapacitorBankSpec,
    voltage0: f64,
    current0: f64,
    t: f64,
) -> CapacitorBankState {
    let capacitance = spec.capacitance_f;
    let inductance = spec.inductance_h;
    let resistance = spec.series_resistance_ohm;
    let alpha = resistance / (2.0 * inductance);
    let dv0 = -current0 / capacitance;
    let coeff = dv0 + alpha * voltage0;
    let exp_term = (-alpha * t).exp();
    let voltage = exp_term * (voltage0 + coeff * t);
    let dv_dt = exp_term * (coeff - alpha * (voltage0 + coeff * t));
    let current = -capacitance * dv_dt;
    let di_dt = voltage / inductance - resistance * current / inductance;
    CapacitorBankState {
        t,
        voltage_v: voltage,
        current_a: current,
        di_dt_a_s: di_dt,
        capacitance_f: capacitance,
    }
}

fn waveform_rms_squared_fraction(waveform: PulseWaveform) -> f64 {
    match waveform {
        PulseWaveform::Rect => 1.0,
        PulseWaveform::HalfSine => 0.5,
        PulseWaveform::ExpDecay => 0.25 * (1.0 - (-10.0_f64).exp()),
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), CapacitorBankError> {
    if !value.is_finite() {
        return Err(CapacitorBankError::NonFinite(field));
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), CapacitorBankError> {
    validate_finite(field, value)?;
    if value <= 0.0 {
        return Err(CapacitorBankError::NonPositive(field));
    }
    Ok(())
}

fn validate_non_negative(field: &'static str, value: f64) -> Result<(), CapacitorBankError> {
    validate_finite(field, value)?;
    if value < 0.0 {
        return Err(CapacitorBankError::Negative(field));
    }
    Ok(())
}

fn energy_balance_relative_error(
    energy_initial: f64,
    energy_remaining: f64,
    resistive_loss: f64,
    load_energy: f64,
    residual: f64,
) -> f64 {
    let scale = energy_initial.abs().max(energy_remaining.abs()).max(
        (resistive_loss.abs() + load_energy.abs() + (energy_initial - energy_remaining).abs())
            .max(1.0),
    );
    residual.abs() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn underdamped_spec() -> CapacitorBankSpec {
        CapacitorBankSpec::new(100e-6, 100e-6, 0.5, 10_000.0, 20.0).expect("valid underdamped spec")
    }

    fn critical_spec() -> CapacitorBankSpec {
        CapacitorBankSpec::new(100e-6, 100e-6, 2.0, 10_000.0, 20.0).expect("valid critical spec")
    }

    fn overdamped_spec() -> CapacitorBankSpec {
        CapacitorBankSpec::new(100e-6, 100e-6, 5.0, 10_000.0, 20.0).expect("valid overdamped spec")
    }

    #[test]
    fn classifies_rlc_regimes() {
        assert_eq!(underdamped_spec().regime(), RlcRegime::Underdamped);
        assert_eq!(critical_spec().regime(), RlcRegime::Critical);
        assert_eq!(overdamped_spec().regime(), RlcRegime::Overdamped);
    }

    #[test]
    fn free_response_preserves_initial_state_at_zero_time() {
        let state = free_response(underdamped_spec(), 5_000.0, 12.0, 0.0)
            .expect("free response must evaluate");
        assert!((state.voltage_v - 5_000.0).abs() < 1e-9);
        assert!((state.current_a - 12.0).abs() < 1e-9);
    }

    #[test]
    fn exact_step_reproduces_closed_form_response_across_regimes() {
        for spec in [underdamped_spec(), critical_spec(), overdamped_spec()] {
            let mut bank = CapacitorBank::new(spec, 5_000.0, 12.0).expect("valid bank");
            let dt = 1e-7;
            for _ in 0..200 {
                bank.step(dt, 0.0).expect("step must evaluate");
            }
            let expected = free_response(spec, 5_000.0, 12.0, 200.0 * dt).expect("closed form");
            assert!((bank.state().voltage_v - expected.voltage_v).abs() < 1e-7);
            assert!((bank.state().current_a - expected.current_a).abs() < 1e-7);
        }
    }

    #[test]
    fn discharge_energy_ledger_closes_to_machine_precision() {
        let mut bank = CapacitorBank::new(overdamped_spec(), 5_000.0, 20.0).expect("valid bank");
        let pulse = PulseSpec::new(300.0, 1e-3, PulseWaveform::HalfSine).expect("valid pulse");
        let report = bank
            .discharge(pulse, 1e-6, 1000)
            .expect("discharge evaluates");
        assert!(report.energy_balance_relative_error < 1e-11);
        assert!(report.energy_balance_passed);
        assert!(report.resistive_loss_j >= 0.0);
    }

    #[test]
    fn feasibility_rejects_unreachable_natural_peak() {
        let bank = CapacitorBank::new(underdamped_spec(), 5_000.0, 0.0).expect("valid bank");
        let pulse = PulseSpec::new(20_000.0, 1e-6, PulseWaveform::HalfSine).expect("valid pulse");
        let (feasible, reason) = bank.feasibility(pulse).expect("feasibility evaluates");
        assert!(!feasible);
        assert!(reason.contains("natural peak"));
    }

    #[test]
    fn feasibility_uses_total_rlc_energy_when_inductor_already_stores_current() {
        let spec = underdamped_spec();
        let bank = CapacitorBank::new(spec, 100.0, 700.0).expect("valid bank");
        let pulse = PulseSpec::new(650.0, 1e-6, PulseWaveform::Rect).expect("valid pulse");
        let (feasible, reason) = bank.feasibility(pulse).expect("feasibility evaluates");
        assert!(feasible);
        assert_eq!(reason, "ok");
        assert!(
            pulse.peak_current_a < (2.0 * bank.total_stored_energy_j() / spec.inductance_h).sqrt()
        );
    }

    #[test]
    fn discharge_preserves_total_rlc_energy_bookkeeping() {
        let mut bank = CapacitorBank::new(overdamped_spec(), 5_000.0, 0.0).expect("valid bank");
        let initial_energy = bank.total_stored_energy_j();
        let pulse = PulseSpec::new(500.0, 1e-3, PulseWaveform::HalfSine).expect("valid pulse");
        let report = bank
            .discharge(pulse, 1e-6, 1000)
            .expect("discharge evaluates");
        assert!(
            (report.energy_delivered_j + report.energy_remaining_j - initial_energy).abs() < 1e-9
        );
        assert!(report.resistive_loss_j >= 0.0);
        assert!(report.load_energy_j > 0.0);
        assert!(report.energy_balance_passed);
        assert!(report.energy_balance_relative_error <= ENERGY_BALANCE_REL_TOLERANCE);
        assert_eq!(report.rlc_regime, RlcRegime::Overdamped);
    }

    #[test]
    fn discharge_balance_includes_inductor_energy() {
        let mut bank = CapacitorBank::new(underdamped_spec(), 4_000.0, 75.0).expect("valid bank");
        let capacitor_only_initial = bank.state().energy_j();
        let pulse = PulseSpec::new(0.1, 2e-5, PulseWaveform::Rect).expect("valid pulse");
        let report = bank
            .discharge(pulse, 1e-7, 200)
            .expect("discharge evaluates");
        assert!(report.energy_initial_j > capacitor_only_initial);
        assert!(
            (report.energy_delivered_j + report.energy_remaining_j - report.energy_initial_j).abs()
                < 1e-9
        );
        assert!(report.energy_balance_passed);
    }

    #[test]
    fn recharge_status_uses_linear_energy_growth() {
        let spec = underdamped_spec();
        let bank = CapacitorBank::new(spec, 0.0, 0.0).expect("valid bank");
        let status = bank.recharge_status(0.1).expect("status evaluates");
        let energy =
            0.5 * spec.capacitance_f * status.projected_voltage_v * status.projected_voltage_v;
        assert!((energy - spec.recharge_power_kw * 1000.0 * 0.1).abs() < 1e-9);
    }

    #[test]
    fn discharge_with_extreme_finite_parameters_fails_closed_without_hang() {
        // Regression for the libFuzzer `capacitor_bank` finding
        // (timeout-be2a810f...): a denormal capacitance with ~1e103 H/Ω
        // inductance and resistance overflowed the Van Loan gramian matrix norm
        // to +inf, which — before the matrix_exp scaling-exponent guard —
        // saturated the squaring count to u32::MAX and spun the squaring loop
        // for minutes. The discharge must now return promptly and fail closed:
        // it may report a non-admitted (or non-finite) ledger, but it must
        // neither hang nor panic.
        let spec = CapacitorBankSpec::new(
            5e-323,
            1.175_865_990_282_161_7e103,
            1.194_530_529_161_495_5e103,
            5.241_383_972_045_715e-304,
            0.0,
        )
        .expect("spec validates: all parameters are finite and in range");
        let mut bank =
            CapacitorBank::new(spec, 0.0, 4.329_103_699_111_611e-304).expect("valid bank");
        let pulse = PulseSpec::new(
            1.194_676_345_876_481e103,
            1.194_530_529_161_495_5e103,
            PulseWaveform::HalfSine,
        )
        .expect("pulse validates");
        // Reaching this assertion at all proves the call returned (no hang). The
        // fail-closed contract admits two outcomes: an explicit error (the step
        // detects the non-finite state and rejects it), or an Ok ledger that —
        // if admitted — carries only finite quantities. Both are acceptable; a
        // hang, panic, or admitted non-finite ledger are not.
        match bank.discharge(pulse, 1.341_801_598_285_21e-309, 1) {
            Err(_) => {}
            Ok(report) if report.energy_balance_passed => {
                assert!(report.energy_initial_j.is_finite() && report.resistive_loss_j.is_finite());
            }
            Ok(_) => {}
        }
    }
}
