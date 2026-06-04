// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Project: SCPN Control
// Description: Pulsed-scenario scheduler v2 Rust kernel.
//! Reusable pulsed-fusion lifecycle scheduler.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Canonical pulsed-fusion lifecycle states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PulsedScenarioState {
    /// Waiting for capacitor precharge.
    Idle,
    /// Ramping formation or compression field current.
    RampUp,
    /// Holding the field while phase and spatial guards close.
    FlatTop,
    /// Compression and burn interval.
    Burn,
    /// Plasma expansion against the recovery field.
    Expansion,
    /// Residual-energy dump.
    Dump,
    /// Capacitor-bank recharge.
    Recharge,
    /// Plasma and coil-current cooldown.
    CoolDown,
}

impl PulsedScenarioState {
    /// Stable string identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::RampUp => "ramp_up",
            Self::FlatTop => "flat_top",
            Self::Burn => "burn",
            Self::Expansion => "expansion",
            Self::Dump => "dump",
            Self::Recharge => "recharge",
            Self::CoolDown => "cool_down",
        }
    }
}

/// Command action emitted for the active lifecycle state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PulsedScenarioAction {
    /// Arm or wait for precharge.
    ArmPrecharge,
    /// Ramp field current.
    RampField,
    /// Hold flat-top field.
    HoldFlatTop,
    /// Fire compression.
    FireCompression,
    /// Recover expansion energy.
    RecoverEnergy,
    /// Dump residual energy.
    DumpResidual,
    /// Recharge the bank.
    RechargeBank,
    /// Cool down.
    CoolDown,
}

impl PulsedScenarioAction {
    /// Stable string identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ArmPrecharge => "arm_precharge",
            Self::RampField => "ramp_field",
            Self::HoldFlatTop => "hold_flat_top",
            Self::FireCompression => "fire_compression",
            Self::RecoverEnergy => "recover_energy",
            Self::DumpResidual => "dump_residual",
            Self::RechargeBank => "recharge_bank",
            Self::CoolDown => "cool_down",
        }
    }
}

/// Guard thresholds for a pulsed-fusion shot lifecycle.
#[derive(Clone, Copy, Debug)]
pub struct PulsedScenarioSpec {
    /// Minimum bank energy before leaving idle.
    pub min_precharge_energy_j: f64,
    /// Absolute coil-current threshold for ramp completion.
    pub ramp_current_a: f64,
    /// Maximum phase-lock error for burn entry.
    pub phase_tolerance_rad: f64,
    /// Maximum chamber-reference error for burn entry.
    pub spatial_tolerance_m: f64,
    /// Minimum plasma temperature for burn entry.
    pub burn_temperature_ev: f64,
    /// Minimum fusion power for burn exit.
    pub min_fusion_power_w: f64,
    /// Minimum radial expansion velocity for dump entry.
    pub expansion_velocity_m_s: f64,
    /// Bank energy floor for dump exit.
    pub dump_energy_floor_j: f64,
    /// Required bank-voltage fraction for recharge exit.
    pub recharge_voltage_fraction: f64,
    /// Maximum plasma temperature for idle re-entry.
    pub cooldown_temperature_ev: f64,
    /// Maximum absolute coil current for idle re-entry.
    pub cooldown_current_a: f64,
    /// Minimum burn dwell before expansion may start.
    pub min_burn_duration_s: f64,
}

impl PulsedScenarioSpec {
    /// Construct a validated scheduler specification.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_precharge_energy_j: f64,
        ramp_current_a: f64,
        phase_tolerance_rad: f64,
        spatial_tolerance_m: f64,
        burn_temperature_ev: f64,
        min_fusion_power_w: f64,
        expansion_velocity_m_s: f64,
        dump_energy_floor_j: f64,
        recharge_voltage_fraction: f64,
        cooldown_temperature_ev: f64,
        cooldown_current_a: f64,
        min_burn_duration_s: f64,
    ) -> Result<Self, PulsedScenarioError> {
        for (field, value) in [
            ("min_precharge_energy_j", min_precharge_energy_j),
            ("ramp_current_a", ramp_current_a),
            ("phase_tolerance_rad", phase_tolerance_rad),
            ("spatial_tolerance_m", spatial_tolerance_m),
            ("burn_temperature_ev", burn_temperature_ev),
            ("min_fusion_power_w", min_fusion_power_w),
            ("expansion_velocity_m_s", expansion_velocity_m_s),
            ("dump_energy_floor_j", dump_energy_floor_j),
            ("cooldown_temperature_ev", cooldown_temperature_ev),
            ("cooldown_current_a", cooldown_current_a),
            ("min_burn_duration_s", min_burn_duration_s),
        ] {
            validate_non_negative(field, value)?;
        }
        validate_positive("phase_tolerance_rad", phase_tolerance_rad)?;
        validate_positive("spatial_tolerance_m", spatial_tolerance_m)?;
        if !recharge_voltage_fraction.is_finite() {
            return Err(PulsedScenarioError::NonFinite("recharge_voltage_fraction"));
        }
        if recharge_voltage_fraction <= 0.0 || recharge_voltage_fraction > 1.0 {
            return Err(PulsedScenarioError::FractionOutOfRange(
                "recharge_voltage_fraction",
            ));
        }
        Ok(Self {
            min_precharge_energy_j,
            ramp_current_a,
            phase_tolerance_rad,
            spatial_tolerance_m,
            burn_temperature_ev,
            min_fusion_power_w,
            expansion_velocity_m_s,
            dump_energy_floor_j,
            recharge_voltage_fraction,
            cooldown_temperature_ev,
            cooldown_current_a,
            min_burn_duration_s,
        })
    }
}

/// Plasma telemetry consumed by lifecycle transition guards.
#[derive(Clone, Copy, Debug)]
pub struct PulsedPlasmaTelemetry {
    /// Coil current in amperes.
    pub coil_current_a: f64,
    /// Plasma temperature in electron-volts.
    pub temperature_ev: f64,
    /// Circular phase-lock error in radians.
    pub phase_lock_error_rad: f64,
    /// Maximum chamber-reference error in metres.
    pub reference_error_m: f64,
    /// Fusion power in watts.
    pub fusion_power_w: f64,
    /// Radial expansion velocity in metres per second.
    pub radial_velocity_m_s: f64,
}

impl PulsedPlasmaTelemetry {
    /// Construct validated plasma telemetry.
    pub fn new(
        coil_current_a: f64,
        temperature_ev: f64,
        phase_lock_error_rad: f64,
        reference_error_m: f64,
        fusion_power_w: f64,
        radial_velocity_m_s: f64,
    ) -> Result<Self, PulsedScenarioError> {
        validate_finite("coil_current_a", coil_current_a)?;
        validate_non_negative("temperature_ev", temperature_ev)?;
        validate_non_negative("phase_lock_error_rad", phase_lock_error_rad)?;
        validate_non_negative("reference_error_m", reference_error_m)?;
        validate_non_negative("fusion_power_w", fusion_power_w)?;
        validate_finite("radial_velocity_m_s", radial_velocity_m_s)?;
        Ok(Self {
            coil_current_a,
            temperature_ev,
            phase_lock_error_rad,
            reference_error_m,
            fusion_power_w,
            radial_velocity_m_s,
        })
    }
}

/// Capacitor-bank telemetry consumed by lifecycle transition guards.
#[derive(Clone, Copy, Debug)]
pub struct CapacitorBankTelemetry {
    /// Bank voltage in volts.
    pub voltage_v: f64,
    /// Declared maximum bank voltage in volts.
    pub voltage_max_v: f64,
    /// Bank energy in joules.
    pub energy_j: f64,
}

impl CapacitorBankTelemetry {
    /// Construct validated bank telemetry.
    pub fn new(
        voltage_v: f64,
        voltage_max_v: f64,
        energy_j: f64,
    ) -> Result<Self, PulsedScenarioError> {
        validate_non_negative("voltage_v", voltage_v)?;
        validate_positive("voltage_max_v", voltage_max_v)?;
        validate_non_negative("energy_j", energy_j)?;
        if voltage_v > voltage_max_v {
            return Err(PulsedScenarioError::VoltageExceedsMax);
        }
        Ok(Self {
            voltage_v,
            voltage_max_v,
            energy_j,
        })
    }

    /// Voltage fraction relative to the declared maximum.
    pub fn voltage_fraction(self) -> f64 {
        self.voltage_v / self.voltage_max_v
    }
}

/// Single lifecycle transition audit entry.
#[derive(Clone, Debug)]
pub struct PulsedScenarioTransition {
    /// Transition time in seconds.
    pub t_s: f64,
    /// State before transition.
    pub from_state: PulsedScenarioState,
    /// State after transition.
    pub to_state: PulsedScenarioState,
    /// Human-readable guard reason.
    pub reason: String,
}

/// Command emitted by one scheduler step.
#[derive(Clone, Debug)]
pub struct PulsedScenarioCommand {
    /// Sample time in seconds.
    pub t_s: f64,
    /// Active state after evaluating guards.
    pub state: PulsedScenarioState,
    /// Command action for the active state.
    pub action: PulsedScenarioAction,
    /// Guard reason for transition or hold.
    pub reason: String,
    /// True when this step changed state.
    pub transition: bool,
    /// Dwell time in the previous state before transition evaluation.
    pub dwell_s: f64,
}

/// Scheduler errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PulsedScenarioError {
    /// A value was not finite.
    NonFinite(&'static str),
    /// A value was negative.
    Negative(&'static str),
    /// A value was not strictly positive.
    NonPositive(&'static str),
    /// A fraction was outside `(0, 1]`.
    FractionOutOfRange(&'static str),
    /// Bank voltage exceeded the declared maximum.
    VoltageExceedsMax,
    /// Time moved backwards.
    NonMonotoneTime,
    /// Time was negative.
    NegativeTime,
    /// Manual transition was not adjacent.
    InvalidTransition,
    /// Manual transition reason was empty.
    EmptyReason,
}

impl Display for PulsedScenarioError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::Negative(field) => write!(f, "{field} must be non-negative"),
            Self::NonPositive(field) => write!(f, "{field} must be strictly positive"),
            Self::FractionOutOfRange(field) => write!(f, "{field} must lie in (0, 1]"),
            Self::VoltageExceedsMax => write!(f, "voltage_v must not exceed voltage_max_v"),
            Self::NonMonotoneTime => write!(f, "t_s must be monotone"),
            Self::NegativeTime => write!(f, "t_s must be non-negative"),
            Self::InvalidTransition => write!(f, "manual transition must be adjacent"),
            Self::EmptyReason => write!(f, "reason must not be empty"),
        }
    }
}

impl Error for PulsedScenarioError {}

/// Reusable eight-state scheduler for pulsed-fusion shots.
#[derive(Clone, Debug)]
pub struct PulsedScenarioScheduler {
    /// Guard thresholds.
    pub spec: PulsedScenarioSpec,
    /// Active state.
    pub state: PulsedScenarioState,
    last_sample_t_s: Option<f64>,
    last_transition_t_s: f64,
    audit_log: Vec<PulsedScenarioTransition>,
}

impl PulsedScenarioScheduler {
    /// Construct a scheduler in `idle`.
    pub fn new(spec: PulsedScenarioSpec) -> Self {
        Self {
            spec,
            state: PulsedScenarioState::Idle,
            last_sample_t_s: None,
            last_transition_t_s: 0.0,
            audit_log: Vec::new(),
        }
    }

    /// Return immutable audit entries.
    pub fn audit_log(&self) -> &[PulsedScenarioTransition] {
        &self.audit_log
    }

    /// Reset to `idle` and clear audit state.
    pub fn reset(&mut self) {
        self.state = PulsedScenarioState::Idle;
        self.last_sample_t_s = None;
        self.last_transition_t_s = 0.0;
        self.audit_log.clear();
    }

    /// Perform a validated manual adjacent transition.
    pub fn transition_to(
        &mut self,
        next_state: PulsedScenarioState,
        t_s: f64,
        reason: &str,
    ) -> Result<PulsedScenarioTransition, PulsedScenarioError> {
        let time = self.validate_timestamp(t_s)?;
        if next_state != next_state_for(self.state) {
            return Err(PulsedScenarioError::InvalidTransition);
        }
        if reason.trim().is_empty() {
            return Err(PulsedScenarioError::EmptyReason);
        }
        let record = self.record_transition(time, next_state, reason);
        self.last_sample_t_s = Some(time);
        Ok(record)
    }

    /// Evaluate lifecycle guards and emit the active command.
    pub fn step(
        &mut self,
        t_s: f64,
        plasma: PulsedPlasmaTelemetry,
        bank: CapacitorBankTelemetry,
    ) -> Result<PulsedScenarioCommand, PulsedScenarioError> {
        let time = self.validate_timestamp(t_s)?;
        let dwell = time - self.last_transition_t_s;
        let (next_state, reason) = self.guard(plasma, bank, dwell);
        let transition = next_state.is_some();
        if let Some(state) = next_state {
            self.record_transition(time, state, reason);
        }
        self.last_sample_t_s = Some(time);
        Ok(PulsedScenarioCommand {
            t_s: time,
            state: self.state,
            action: action_for(self.state),
            reason: reason.to_string(),
            transition,
            dwell_s: dwell,
        })
    }

    fn validate_timestamp(&self, t_s: f64) -> Result<f64, PulsedScenarioError> {
        validate_finite("t_s", t_s)?;
        if t_s < 0.0 {
            return Err(PulsedScenarioError::NegativeTime);
        }
        if let Some(last) = self.last_sample_t_s {
            if t_s < last {
                return Err(PulsedScenarioError::NonMonotoneTime);
            }
        }
        Ok(t_s)
    }

    fn record_transition(
        &mut self,
        t_s: f64,
        next_state: PulsedScenarioState,
        reason: &str,
    ) -> PulsedScenarioTransition {
        let previous = self.state;
        self.state = next_state;
        self.last_transition_t_s = t_s;
        let record = PulsedScenarioTransition {
            t_s,
            from_state: previous,
            to_state: next_state,
            reason: reason.to_string(),
        };
        self.audit_log.push(record.clone());
        record
    }

    fn guard(
        &self,
        plasma: PulsedPlasmaTelemetry,
        bank: CapacitorBankTelemetry,
        dwell_s: f64,
    ) -> (Option<PulsedScenarioState>, &'static str) {
        let spec = self.spec;
        match self.state {
            PulsedScenarioState::Idle => {
                if bank.energy_j >= spec.min_precharge_energy_j {
                    (
                        Some(PulsedScenarioState::RampUp),
                        "precharge energy available",
                    )
                } else {
                    (None, "waiting for precharge energy")
                }
            }
            PulsedScenarioState::RampUp => {
                if plasma.coil_current_a.abs() >= spec.ramp_current_a {
                    (Some(PulsedScenarioState::FlatTop), "ramp current reached")
                } else {
                    (None, "waiting for ramp current")
                }
            }
            PulsedScenarioState::FlatTop => {
                let phase_ok = plasma.phase_lock_error_rad <= spec.phase_tolerance_rad;
                let spatial_ok = plasma.reference_error_m <= spec.spatial_tolerance_m;
                let temperature_ok = plasma.temperature_ev >= spec.burn_temperature_ev;
                let energy_ok = bank.energy_j >= spec.min_precharge_energy_j;
                if phase_ok && spatial_ok && temperature_ok && energy_ok {
                    (
                        Some(PulsedScenarioState::Burn),
                        "phase, spatial, temperature, and energy guards passed",
                    )
                } else if !phase_ok || !spatial_ok {
                    (None, "waiting for phase and spatial lock")
                } else if !temperature_ok {
                    (None, "waiting for burn temperature")
                } else {
                    (None, "waiting for burn energy")
                }
            }
            PulsedScenarioState::Burn => {
                if dwell_s < spec.min_burn_duration_s {
                    (None, "waiting for minimum burn dwell")
                } else if plasma.fusion_power_w >= spec.min_fusion_power_w {
                    (
                        Some(PulsedScenarioState::Expansion),
                        "fusion power threshold reached",
                    )
                } else {
                    (None, "waiting for fusion power")
                }
            }
            PulsedScenarioState::Expansion => {
                if plasma.radial_velocity_m_s >= spec.expansion_velocity_m_s {
                    (
                        Some(PulsedScenarioState::Dump),
                        "expansion velocity reached",
                    )
                } else {
                    (None, "waiting for expansion velocity")
                }
            }
            PulsedScenarioState::Dump => {
                if bank.energy_j <= spec.dump_energy_floor_j {
                    (
                        Some(PulsedScenarioState::Recharge),
                        "residual energy dumped",
                    )
                } else {
                    (None, "waiting for dump energy floor")
                }
            }
            PulsedScenarioState::Recharge => {
                if bank.voltage_fraction() >= spec.recharge_voltage_fraction {
                    (Some(PulsedScenarioState::CoolDown), "bank recharged")
                } else {
                    (None, "waiting for recharge voltage")
                }
            }
            PulsedScenarioState::CoolDown => {
                let temperature_ok = plasma.temperature_ev <= spec.cooldown_temperature_ev;
                let current_ok = plasma.coil_current_a.abs() <= spec.cooldown_current_a;
                if temperature_ok && current_ok {
                    (
                        Some(PulsedScenarioState::Idle),
                        "plasma cooled and coil current cleared",
                    )
                } else {
                    (None, "waiting for plasma cooldown")
                }
            }
        }
    }
}

fn next_state_for(state: PulsedScenarioState) -> PulsedScenarioState {
    match state {
        PulsedScenarioState::Idle => PulsedScenarioState::RampUp,
        PulsedScenarioState::RampUp => PulsedScenarioState::FlatTop,
        PulsedScenarioState::FlatTop => PulsedScenarioState::Burn,
        PulsedScenarioState::Burn => PulsedScenarioState::Expansion,
        PulsedScenarioState::Expansion => PulsedScenarioState::Dump,
        PulsedScenarioState::Dump => PulsedScenarioState::Recharge,
        PulsedScenarioState::Recharge => PulsedScenarioState::CoolDown,
        PulsedScenarioState::CoolDown => PulsedScenarioState::Idle,
    }
}

fn action_for(state: PulsedScenarioState) -> PulsedScenarioAction {
    match state {
        PulsedScenarioState::Idle => PulsedScenarioAction::ArmPrecharge,
        PulsedScenarioState::RampUp => PulsedScenarioAction::RampField,
        PulsedScenarioState::FlatTop => PulsedScenarioAction::HoldFlatTop,
        PulsedScenarioState::Burn => PulsedScenarioAction::FireCompression,
        PulsedScenarioState::Expansion => PulsedScenarioAction::RecoverEnergy,
        PulsedScenarioState::Dump => PulsedScenarioAction::DumpResidual,
        PulsedScenarioState::Recharge => PulsedScenarioAction::RechargeBank,
        PulsedScenarioState::CoolDown => PulsedScenarioAction::CoolDown,
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), PulsedScenarioError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PulsedScenarioError::NonFinite(field))
    }
}

fn validate_non_negative(field: &'static str, value: f64) -> Result<(), PulsedScenarioError> {
    validate_finite(field, value)?;
    if value < 0.0 {
        Err(PulsedScenarioError::Negative(field))
    } else {
        Ok(())
    }
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), PulsedScenarioError> {
    validate_finite(field, value)?;
    if value <= 0.0 {
        Err(PulsedScenarioError::NonPositive(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CapacitorBankTelemetry, PulsedPlasmaTelemetry, PulsedScenarioScheduler, PulsedScenarioSpec,
        PulsedScenarioState,
    };

    fn spec() -> PulsedScenarioSpec {
        PulsedScenarioSpec::new(
            100.0, 2.0e6, 0.01, 0.002, 1.0e3, 2.0e6, 1.0e3, 40.0, 0.95, 20.0, 1.0e3, 0.0,
        )
        .expect("valid spec")
    }

    fn plasma(
        coil_current_a: f64,
        temperature_ev: f64,
        phase_lock_error_rad: f64,
        reference_error_m: f64,
        fusion_power_w: f64,
        radial_velocity_m_s: f64,
    ) -> PulsedPlasmaTelemetry {
        PulsedPlasmaTelemetry::new(
            coil_current_a,
            temperature_ev,
            phase_lock_error_rad,
            reference_error_m,
            fusion_power_w,
            radial_velocity_m_s,
        )
        .expect("valid plasma")
    }

    fn bank(voltage_v: f64, energy_j: f64) -> CapacitorBankTelemetry {
        CapacitorBankTelemetry::new(voltage_v, 10_000.0, energy_j).expect("valid bank")
    }

    #[test]
    fn campaign_traverses_eight_states() {
        let mut scheduler = PulsedScenarioScheduler::new(spec());
        let samples = [
            (
                0.0,
                plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            (
                1.0e-3,
                plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            (
                2.0e-3,
                plasma(2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            (
                3.0e-3,
                plasma(2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0),
                bank(9800.0, 200.0),
            ),
            (
                4.0e-3,
                plasma(0.0, 200.0, 0.02, 0.01, 0.0, 1500.0),
                bank(9800.0, 200.0),
            ),
            (
                5.0e-3,
                plasma(0.0, 120.0, 0.02, 0.01, 0.0, 0.0),
                bank(2000.0, 20.0),
            ),
            (
                6.0e-3,
                plasma(0.0, 40.0, 0.02, 0.01, 0.0, 0.0),
                bank(9700.0, 180.0),
            ),
            (
                7.0e-3,
                plasma(100.0, 15.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
        ];
        let mut states = Vec::new();
        for (time, p, b) in samples {
            states.push(scheduler.step(time, p, b).expect("step").state);
        }
        assert_eq!(
            states,
            [
                PulsedScenarioState::RampUp,
                PulsedScenarioState::FlatTop,
                PulsedScenarioState::Burn,
                PulsedScenarioState::Expansion,
                PulsedScenarioState::Dump,
                PulsedScenarioState::Recharge,
                PulsedScenarioState::CoolDown,
                PulsedScenarioState::Idle,
            ]
        );
        assert_eq!(scheduler.audit_log().len(), 8);
    }
}
