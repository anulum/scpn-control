// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — PID
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

//! PID controller for tokamak position control.
//!
//! Port of `tokamak_flight_sim.py`.
//! Implements decoupled radial and vertical position PID.

use control_types::error::{FusionError, FusionResult};

/// Radial PID gains. Python: Kp=2.0, Ki=0.1, Kd=0.5.
const PID_R_KP: f64 = 2.0;
const PID_R_KI: f64 = 0.1;
const PID_R_KD: f64 = 0.5;

/// Vertical PID gains. Python: Kp=5.0, Ki=0.2, Kd=2.0.
const PID_Z_KP: f64 = 5.0;
const PID_Z_KI: f64 = 0.2;
const PID_Z_KD: f64 = 2.0;

/// Generic PID controller.
///
/// The output envelope (saturation limits and per-step slew limit) is optional
/// and OFF by default: an unconfigured controller behaves exactly as an ideal
/// `kp*e + ki*Σe + kd*Δe` law, preserving legacy behaviour and Python parity.
/// When an output envelope is configured via [`PIDController::with_output_limits`]
/// or [`PIDController::with_slew_limit`], the controller applies **conditional-
/// integration anti-windup**: the integrator is frozen for any step whose raw
/// output is clamped in the same direction as the error, so a sustained
/// saturating error can never wind the integral up without bound. This closes
/// the integral-windup hazard for a position PID driving a saturating actuator.
#[derive(Debug, Clone)]
pub struct PIDController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    err_sum: f64,
    last_err: f64,
    /// Lower output saturation limit; `-inf` when no saturation is configured.
    output_min: f64,
    /// Upper output saturation limit; `+inf` when no saturation is configured.
    output_max: f64,
    /// Maximum permitted `|Δoutput|` per step; `+inf` when no slew limit is set.
    slew_max: f64,
    /// Output applied on the previous step (seed for the slew limit).
    last_output: f64,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> FusionResult<Self> {
        if !kp.is_finite() || !ki.is_finite() || !kd.is_finite() {
            return Err(FusionError::ConfigError(
                "pid gains must be finite".to_string(),
            ));
        }
        Ok(PIDController {
            kp,
            ki,
            kd,
            err_sum: 0.0,
            last_err: 0.0,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
            slew_max: f64::INFINITY,
            last_output: 0.0,
        })
    }

    /// Configure the output saturation envelope `[output_min, output_max]`.
    ///
    /// Both bounds must be finite and ordered. Enabling saturation also enables
    /// the anti-windup behaviour documented on the struct.
    pub fn with_output_limits(mut self, output_min: f64, output_max: f64) -> FusionResult<Self> {
        if !output_min.is_finite() || !output_max.is_finite() {
            return Err(FusionError::ConfigError(
                "pid output limits must be finite".to_string(),
            ));
        }
        if output_min > output_max {
            return Err(FusionError::ConfigError(
                "pid output_min must not exceed output_max".to_string(),
            ));
        }
        self.output_min = output_min;
        self.output_max = output_max;
        Ok(self)
    }

    /// Configure the maximum per-step output change `slew_max` (`|Δoutput|`).
    ///
    /// Must be finite and strictly positive. A slew-limited step that clamps the
    /// output against the error direction also freezes the integrator.
    pub fn with_slew_limit(mut self, slew_max: f64) -> FusionResult<Self> {
        if !slew_max.is_finite() || slew_max <= 0.0 {
            return Err(FusionError::ConfigError(
                "pid slew_max must be finite and > 0".to_string(),
            ));
        }
        self.slew_max = slew_max;
        Ok(self)
    }

    /// Default radial position controller.
    pub fn radial() -> FusionResult<Self> {
        Self::new(PID_R_KP, PID_R_KI, PID_R_KD)
    }

    /// Default vertical position controller.
    pub fn vertical() -> FusionResult<Self> {
        Self::new(PID_Z_KP, PID_Z_KI, PID_Z_KD)
    }

    /// Clamp `raw` to the configured saturation envelope and slew limit.
    fn saturate_and_slew(&self, raw: f64) -> f64 {
        let saturated = raw.clamp(self.output_min, self.output_max);
        let delta = saturated - self.last_output;
        if delta.abs() > self.slew_max {
            self.last_output + delta.signum() * self.slew_max
        } else {
            saturated
        }
    }

    /// One PID step. Returns the applied control output.
    ///
    /// With no envelope configured this is the ideal `kp*e + ki*Σe + kd*Δe`.
    /// With an envelope, the output is saturated and slew-limited, and the
    /// integrator is frozen whenever the raw output is clamped in the same
    /// direction as the error (conditional-integration anti-windup).
    pub fn step(&mut self, error: f64) -> FusionResult<f64> {
        if !error.is_finite() {
            return Err(FusionError::ConfigError(
                "pid error input must be finite".to_string(),
            ));
        }
        let d_err = error - self.last_err;
        let candidate_sum = self.err_sum + error;
        let raw = self.kp * error + self.ki * candidate_sum + self.kd * d_err;
        let applied = self.saturate_and_slew(raw);

        // Anti-windup: `(raw - applied) * error > 0` means the output was clamped
        // in the same direction the error is pushing, so committing the integral
        // would only wind it further into the limit. Freeze it in that case.
        let winding = (raw - applied) * error > 0.0;
        self.err_sum = if winding { self.err_sum } else { candidate_sum };
        self.last_err = error;
        self.last_output = applied;
        Ok(applied)
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        self.err_sum = 0.0;
        self.last_err = 0.0;
        self.last_output = 0.0;
    }
}

/// Isoflux position controller with R and Z PIDs.
pub struct IsoFluxController {
    pub pid_r: PIDController,
    pub pid_z: PIDController,
    pub target_r: f64,
    pub target_z: f64,
    pub r_history: Vec<f64>,
    pub z_history: Vec<f64>,
}

impl IsoFluxController {
    pub fn new(target_r: f64, target_z: f64) -> FusionResult<Self> {
        if !target_r.is_finite() || !target_z.is_finite() {
            return Err(FusionError::ConfigError(
                "isoflux targets must be finite".to_string(),
            ));
        }
        Ok(IsoFluxController {
            pid_r: PIDController::radial()?,
            pid_z: PIDController::vertical()?,
            target_r,
            target_z,
            r_history: Vec::new(),
            z_history: Vec::new(),
        })
    }

    /// Apply a symmetric coil-current actuator envelope to both position PIDs.
    ///
    /// `current_delta_limit` bounds `|ctrl|` per axis (saturation) and, when
    /// `slew_max` is `Some`, the per-step change. This mirrors the Python
    /// flight-sim `FirstOrderActuator` limits and engages the PID anti-windup so
    /// a saturating position error cannot wind either integrator up.
    pub fn with_actuator_limits(
        mut self,
        current_delta_limit: f64,
        slew_max: Option<f64>,
    ) -> FusionResult<Self> {
        if !current_delta_limit.is_finite() || current_delta_limit <= 0.0 {
            return Err(FusionError::ConfigError(
                "isoflux current_delta_limit must be finite and > 0".to_string(),
            ));
        }
        self.pid_r = self
            .pid_r
            .clone()
            .with_output_limits(-current_delta_limit, current_delta_limit)?;
        self.pid_z = self
            .pid_z
            .clone()
            .with_output_limits(-current_delta_limit, current_delta_limit)?;
        if let Some(slew) = slew_max {
            self.pid_r = self.pid_r.clone().with_slew_limit(slew)?;
            self.pid_z = self.pid_z.clone().with_slew_limit(slew)?;
        }
        Ok(self)
    }

    /// Compute coil corrections given measured position.
    /// Returns (ctrl_radial, ctrl_vertical).
    pub fn step(&mut self, measured_r: f64, measured_z: f64) -> FusionResult<(f64, f64)> {
        if !measured_r.is_finite() || !measured_z.is_finite() {
            return Err(FusionError::ConfigError(
                "isoflux measured position must be finite".to_string(),
            ));
        }
        let err_r = self.target_r - measured_r;
        let err_z = self.target_z - measured_z;
        let ctrl_r = self.pid_r.step(err_r)?;
        let ctrl_z = self.pid_z.step(err_z)?;
        self.r_history.push(measured_r);
        self.z_history.push(measured_z);
        Ok((ctrl_r, ctrl_z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_zero_error() {
        let mut pid = PIDController::new(1.0, 0.1, 0.5).expect("valid gains");
        let out = pid.step(0.0).expect("valid finite input");
        assert!((out).abs() < 1e-10, "Zero error → zero output: {out}");
    }

    #[test]
    fn test_pid_proportional() {
        let mut pid = PIDController::new(2.0, 0.0, 0.0).expect("valid gains");
        let out = pid.step(5.0).expect("valid finite input");
        assert!(
            (out - 10.0).abs() < 1e-10,
            "Pure P: 2.0 * 5.0 = 10.0: {out}"
        );
    }

    #[test]
    fn test_pid_integral_accumulates() {
        let mut pid = PIDController::new(0.0, 1.0, 0.0).expect("valid gains");
        pid.step(1.0).expect("valid finite input");
        pid.step(1.0).expect("valid finite input");
        let out = pid.step(1.0).expect("valid finite input");
        assert!((out - 3.0).abs() < 1e-10, "Integral of three 1s = 3: {out}");
    }

    #[test]
    fn test_isoflux_converges() {
        let mut ctrl = IsoFluxController::new(6.2, 0.0).expect("valid targets");
        // Simple plant model: position moves toward target under control
        let mut r = 5.0;
        let mut z = 1.0;
        for _ in 0..100 {
            let (cr, cz) = ctrl.step(r, z).expect("valid finite position inputs");
            // Simple plant: position += gain * control
            r += 0.01 * cr;
            z += 0.01 * cz;
        }
        assert!((r - 6.2).abs() < 0.5, "R should approach target: {r}");
        assert!((z).abs() < 0.5, "Z should approach 0: {z}");
    }

    #[test]
    fn test_pid_rejects_non_finite_gains_and_error() {
        assert!(PIDController::new(f64::NAN, 0.1, 0.2).is_err());
        let mut pid = PIDController::new(1.0, 0.1, 0.2).expect("valid gains");
        assert!(pid.step(f64::INFINITY).is_err());
    }

    #[test]
    fn test_isoflux_rejects_non_finite_targets_and_measurements() {
        assert!(IsoFluxController::new(f64::NAN, 0.0).is_err());
        let mut ctrl = IsoFluxController::new(6.2, 0.0).expect("valid targets");
        assert!(ctrl.step(6.0, f64::NAN).is_err());
    }

    #[test]
    fn test_unconfigured_envelope_is_bit_identical_to_ideal_law() {
        // The default (no envelope) PID must reproduce kp*e + ki*Σe + kd*Δe exactly.
        let mut pid = PIDController::new(2.0, 0.1, 0.5).expect("valid gains");
        let mut err_sum = 0.0;
        let mut last_err = 0.0;
        for &e in &[1.0, -0.5, 3.0, -2.0, 0.25] {
            err_sum += e;
            let d_err = e - last_err;
            last_err = e;
            let expected = 2.0 * e + 0.1 * err_sum + 0.5 * d_err;
            let got = pid.step(e).expect("finite");
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "legacy path must be exact"
            );
        }
    }

    #[test]
    fn test_output_saturation_clamps_symmetrically() {
        let mut pid = PIDController::new(10.0, 0.0, 0.0)
            .expect("valid gains")
            .with_output_limits(-5.0, 5.0)
            .expect("valid limits");
        assert_eq!(
            pid.step(100.0).expect("finite"),
            5.0,
            "clamps to output_max"
        );
        assert_eq!(
            pid.step(-100.0).expect("finite"),
            -5.0,
            "clamps to output_min"
        );
    }

    #[test]
    fn test_anti_windup_freezes_integrator_under_sustained_saturation() {
        // A sustained saturating error must NOT wind the integral up: recovery
        // is immediate once the error clears, with no post-saturation overshoot.
        let limit = 5.0;
        let mut guarded = PIDController::new(1.0, 0.5, 0.0)
            .expect("valid gains")
            .with_output_limits(-limit, limit)
            .expect("valid limits");
        let mut naive = PIDController::new(1.0, 0.5, 0.0).expect("valid gains");
        // Drive both hard into saturation for many steps.
        for _ in 0..200 {
            let g = guarded.step(10.0).expect("finite");
            assert!(g <= limit + 1e-12, "guarded output stays within the limit");
            naive.step(10.0).expect("finite");
        }
        // Now the error reverses sign: the guarded controller responds at once,
        // the naive one must first unwind a massive integral (windup).
        let guarded_recovery = guarded.step(-1.0).expect("finite");
        let naive_recovery = naive.step(-1.0).expect("finite");
        assert!(
            guarded_recovery < 0.0,
            "anti-windup controller reverses immediately: {guarded_recovery}"
        );
        assert!(
            naive_recovery > 100.0,
            "naive controller is still stuck at a wound-up positive output: {naive_recovery}"
        );
    }

    #[test]
    fn test_anti_windup_still_integrates_away_from_the_limit() {
        // Freezing must be directional: an error that de-saturates the output
        // must still be integrated.
        let mut pid = PIDController::new(1.0, 1.0, 0.0)
            .expect("valid gains")
            .with_output_limits(-5.0, 5.0)
            .expect("valid limits");
        for _ in 0..20 {
            pid.step(10.0).expect("finite"); // saturate high, integrator frozen
        }
        // A small negative error de-saturates without hitting the opposite rail:
        // the integrator must accumulate it, so successive outputs keep falling.
        let first = pid.step(-1.0).expect("finite");
        let second = pid.step(-1.0).expect("finite");
        assert!(first < 5.0, "output leaves the upper rail: {first}");
        assert!(
            second < first,
            "de-saturating error keeps integrating: {second} < {first}"
        );
    }

    #[test]
    fn test_slew_limit_bounds_per_step_change() {
        let mut pid = PIDController::new(100.0, 0.0, 0.0)
            .expect("valid gains")
            .with_slew_limit(2.0)
            .expect("valid slew");
        let first = pid.step(1.0).expect("finite");
        assert_eq!(first, 2.0, "first step is slew-limited from 0 to +2");
        let second = pid.step(1.0).expect("finite");
        assert_eq!(
            second, 4.0,
            "second step advances by at most the slew limit"
        );
    }

    #[test]
    fn test_output_limit_validation() {
        let base = PIDController::new(1.0, 0.0, 0.0).expect("valid gains");
        assert!(base.clone().with_output_limits(f64::NAN, 1.0).is_err());
        assert!(
            base.clone().with_output_limits(1.0, -1.0).is_err(),
            "min > max rejected"
        );
        assert!(
            base.clone().with_slew_limit(0.0).is_err(),
            "non-positive slew rejected"
        );
        assert!(
            base.clone().with_slew_limit(f64::INFINITY).is_err(),
            "infinite slew rejected"
        );
    }

    #[test]
    fn test_reset_clears_slew_seed() {
        let mut pid = PIDController::new(100.0, 0.0, 0.0)
            .expect("valid gains")
            .with_slew_limit(2.0)
            .expect("valid slew");
        pid.step(1.0).expect("finite");
        pid.reset();
        // After reset the slew reference is back at 0, so the first step is +2 again.
        assert_eq!(pid.step(1.0).expect("finite"), 2.0);
    }

    #[test]
    fn test_isoflux_actuator_limits_bound_and_reject() {
        let mut ctrl = IsoFluxController::new(6.2, 0.0)
            .expect("valid targets")
            .with_actuator_limits(3.0, Some(1.0))
            .expect("valid envelope");
        let (cr, cz) = ctrl.step(0.0, 5.0).expect("finite position");
        assert!(cr.abs() <= 1.0 + 1e-12, "radial slew-bounded first step");
        assert!(cz.abs() <= 1.0 + 1e-12, "vertical slew-bounded first step");
        assert!(IsoFluxController::new(6.2, 0.0)
            .expect("valid targets")
            .with_actuator_limits(-1.0, None)
            .is_err());
    }
}
