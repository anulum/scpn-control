// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Project: SCPN Control
// Description: Control-owned capacitor-bank RLC state model.
//! Bounded capacitor-bank state model for CONTROL pulsed-shot contracts.

use std::error::Error;
use std::fmt::{Display, Formatter};

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
    /// Energy removed from capacitor storage.
    pub energy_delivered_j: f64,
    /// Remaining capacitor energy.
    pub energy_remaining_j: f64,
    /// Peak absolute capacitor voltage observed.
    pub peak_voltage_v: f64,
    /// Peak absolute current observed.
    pub peak_current_a: f64,
    /// Simulated discharge duration.
    pub discharge_duration_s: f64,
    /// Damping regime active for the bank.
    pub rlc_regime: RlcRegime,
}

/// Mutable bounded series-RLC bank with analytical and numerical stepping.
#[derive(Clone, Debug)]
pub struct CapacitorBank {
    spec: CapacitorBankSpec,
    t: f64,
    voltage_v: f64,
    current_a: f64,
    di_dt_a_s: f64,
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

    /// Advance the bank one Crank-Nicolson step with optional load current.
    pub fn step(
        &mut self,
        dt: f64,
        external_load_current_a: f64,
    ) -> Result<CapacitorBankState, CapacitorBankError> {
        validate_positive("dt", dt)?;
        validate_finite("external_load_current_a", external_load_current_a)?;
        let capacitance = self.spec.capacitance_f;
        let inductance = self.spec.inductance_h;
        let resistance = self.spec.series_resistance_ohm;
        let a12 = -1.0 / capacitance;
        let a21 = 1.0 / inductance;
        let a22 = -resistance / inductance;
        let h = dt / 2.0;
        let lhs_11 = 1.0;
        let lhs_12 = -h * a12;
        let lhs_21 = -h * a21;
        let lhs_22 = 1.0 - h * a22;
        let rhs_v =
            self.voltage_v + h * a12 * self.current_a - dt * external_load_current_a / capacitance;
        let rhs_i = h * a21 * self.voltage_v + (1.0 + h * a22) * self.current_a;
        let determinant = lhs_11 * lhs_22 - lhs_12 * lhs_21;
        if !determinant.is_finite() || determinant.abs() <= 1e-30 {
            return Err(CapacitorBankError::SingularStepMatrix);
        }
        let voltage_next = (lhs_22 * rhs_v - lhs_12 * rhs_i) / determinant;
        let current_next = (-lhs_21 * rhs_v + lhs_11 * rhs_i) / determinant;
        let di_dt_next = a21 * voltage_next + a22 * current_next;
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
        let energy_initial = self.state().energy_j();
        let mut peak_voltage = self.voltage_v.abs();
        let mut peak_current = self.current_a.abs();
        let mut pulse_t = 0.0;
        for _ in 0..n_steps {
            let load_current = sample_waveform(pulse, pulse_t + dt / 2.0)?;
            let state = self.step(dt, load_current)?;
            peak_voltage = peak_voltage.max(state.voltage_v.abs());
            peak_current = peak_current.max(state.current_a.abs());
            pulse_t += dt;
        }
        let energy_remaining = self.state().energy_j();
        Ok(EnergyReport {
            energy_delivered_j: energy_initial - energy_remaining,
            energy_remaining_j: energy_remaining,
            peak_voltage_v: peak_voltage,
            peak_current_a: peak_current,
            discharge_duration_s: n_steps as f64 * dt,
            rlc_regime: self.spec.regime(),
        })
    }

    /// Run conservative pulse admissibility guards against current bank state.
    pub fn feasibility(&self, pulse: PulseSpec) -> Result<(bool, String), CapacitorBankError> {
        let voltage_now = self.voltage_v.abs();
        if voltage_now > 0.0 {
            let max_natural_current = voltage_now / self.spec.natural_impedance_ohm();
            if pulse.peak_current_a > max_natural_current {
                return Ok((
                    false,
                    format!(
                        "requested peak current {:.3e} A exceeds bank natural peak {:.3e} A at |v0| = {:.3e} V",
                        pulse.peak_current_a, max_natural_current, voltage_now
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
        let available_energy = self.state().energy_j();
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
    /// Crank-Nicolson matrix was singular.
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
    fn crank_nicolson_tracks_closed_form_response() {
        let spec = underdamped_spec();
        let mut bank = CapacitorBank::new(spec, 5_000.0, 0.0).expect("valid bank");
        let dt = 1e-7;
        for _ in 0..200 {
            bank.step(dt, 0.0).expect("step must evaluate");
        }
        let expected = free_response(spec, 5_000.0, 0.0, 200.0 * dt).expect("closed form");
        assert!((bank.state().voltage_v - expected.voltage_v).abs() < 5e-2);
        assert!((bank.state().current_a - expected.current_a).abs() < 5e-2);
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
    fn discharge_preserves_energy_bookkeeping() {
        let mut bank = CapacitorBank::new(overdamped_spec(), 5_000.0, 0.0).expect("valid bank");
        let initial_energy = bank.state().energy_j();
        let pulse = PulseSpec::new(500.0, 1e-3, PulseWaveform::HalfSine).expect("valid pulse");
        let report = bank
            .discharge(pulse, 1e-6, 1000)
            .expect("discharge evaluates");
        assert!(
            (report.energy_delivered_j + report.energy_remaining_j - initial_energy).abs() < 1e-9
        );
        assert_eq!(report.rlc_regime, RlcRegime::Overdamped);
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
}
