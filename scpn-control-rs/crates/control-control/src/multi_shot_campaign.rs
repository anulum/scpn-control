// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust multi-shot pulsed-campaign orchestration.
//! Multi-shot campaign admission over pulsed scheduler and bank telemetry.

use crate::capacitor_bank::{CapacitorBank, CapacitorBankSpec};
use crate::pulsed_scenario::{
    CapacitorBankTelemetry, PulsedPlasmaTelemetry, PulsedScenarioCommand, PulsedScenarioError,
    PulsedScenarioScheduler, PulsedScenarioSpec,
};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

const MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION: &str = "scpn-control.multi-shot-campaign.v1";
const PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION: &str =
    "scpn-control.pulsed-mpc-decision-evidence.v1";
const EXPECTED_TRANSITION_STATES: [&str; 8] = [
    "ramp_up",
    "flat_top",
    "burn",
    "expansion",
    "dump",
    "recharge",
    "cool_down",
    "idle",
];

/// One timestamped sample for a pulsed shot.
#[derive(Clone, Copy, Debug)]
pub struct CampaignShotSample {
    pub t_s: f64,
    pub plasma: PulsedPlasmaTelemetry,
    pub bank: CapacitorBankTelemetry,
}

impl CampaignShotSample {
    /// Construct a validated sample.
    pub fn new(
        t_s: f64,
        plasma: PulsedPlasmaTelemetry,
        bank: CapacitorBankTelemetry,
    ) -> Result<Self, MultiShotCampaignError> {
        validate_non_negative("t_s", t_s)?;
        Ok(Self { t_s, plasma, bank })
    }
}

/// Input contract for one shot in a campaign.
#[derive(Clone, Debug)]
pub struct CampaignShotPlan {
    pub shot_id: String,
    pub samples: Vec<CampaignShotSample>,
    pub initial_bank_voltage_v: f64,
    pub initial_bank_current_a: f64,
    pub pulsed_mpc_admission_digest: Option<String>,
}

impl CampaignShotPlan {
    /// Construct a validated shot plan.
    pub fn new(
        shot_id: &str,
        samples: Vec<CampaignShotSample>,
        initial_bank_voltage_v: f64,
        initial_bank_current_a: f64,
    ) -> Result<Self, MultiShotCampaignError> {
        let id = shot_id.trim();
        if id.is_empty() {
            return Err(MultiShotCampaignError::EmptyShotId);
        }
        if samples.is_empty() {
            return Err(MultiShotCampaignError::EmptySamples);
        }
        validate_non_negative("initial_bank_voltage_v", initial_bank_voltage_v)?;
        validate_finite("initial_bank_current_a", initial_bank_current_a)?;
        Ok(Self {
            shot_id: id.to_string(),
            samples,
            initial_bank_voltage_v,
            initial_bank_current_a,
            pulsed_mpc_admission_digest: None,
        })
    }

    /// Attach digest-bound pulsed-MPC admission evidence to this shot plan.
    pub fn with_pulsed_mpc_admission_digest(
        mut self,
        digest: &str,
    ) -> Result<Self, MultiShotCampaignError> {
        self.pulsed_mpc_admission_digest = Some(validate_sha256_digest(
            "pulsed_mpc_admission_digest",
            digest,
        )?);
        Ok(self)
    }
}

/// Stable command log row.
#[derive(Clone, Debug)]
pub struct CampaignCommandLog {
    pub t_s: f64,
    pub state: String,
    pub action: String,
    pub reason: String,
    pub transition: bool,
    pub dwell_s: f64,
}

impl From<PulsedScenarioCommand> for CampaignCommandLog {
    fn from(command: PulsedScenarioCommand) -> Self {
        Self {
            t_s: command.t_s,
            state: command.state.as_str().to_string(),
            action: command.action.as_str().to_string(),
            reason: command.reason,
            transition: command.transition,
            dwell_s: command.dwell_s,
        }
    }
}

/// Stable phase-log row compatible with replay v1.1 shot metadata.
#[derive(Clone, Debug)]
pub struct CampaignPhaseLog {
    pub t_s: f64,
    pub state: String,
    pub reason: String,
}

/// Admission result for one shot.
#[derive(Clone, Debug)]
pub struct CampaignShotResult {
    pub shot_id: String,
    pub pulse_id: String,
    pub passed: bool,
    pub failure_reason: Option<String>,
    pub terminal_state: String,
    pub transition_states: Vec<String>,
    pub command_log: Vec<CampaignCommandLog>,
    pub shot_phase_log: Vec<CampaignPhaseLog>,
    pub capacitor_state_initial_j: f64,
    pub capacitor_state_final_j: f64,
    pub energy_recovered_j: f64,
    pub trigger_timestamp_ns: Option<u64>,
    pub pulsed_mpc_admission_digest: Option<String>,
}

/// Campaign report summary.
#[derive(Clone, Debug)]
pub struct MultiShotCampaignReport {
    pub schema_version: String,
    pub campaign_id: String,
    pub shot_count: usize,
    pub passed_count: usize,
    pub failed_count: usize,
    pub require_complete_lifecycle: bool,
    pub expected_transition_states: Vec<String>,
    pub pulsed_mpc_evidence_schema_version: String,
    pub pulsed_mpc_admission_digest_count: usize,
    pub shots: Vec<CampaignShotResult>,
    pub payload_sha256: String,
}

/// Multi-shot campaign orchestrator.
#[derive(Clone, Debug)]
pub struct MultiShotCampaignOrchestrator {
    campaign_id: String,
    scheduler_spec: PulsedScenarioSpec,
    bank_spec: CapacitorBankSpec,
    require_complete_lifecycle: bool,
}

impl MultiShotCampaignOrchestrator {
    /// Construct a validated orchestrator.
    pub fn new(
        campaign_id: &str,
        scheduler_spec: PulsedScenarioSpec,
        bank_spec: CapacitorBankSpec,
        require_complete_lifecycle: bool,
    ) -> Result<Self, MultiShotCampaignError> {
        let id = campaign_id.trim();
        if id.is_empty() {
            return Err(MultiShotCampaignError::EmptyCampaignId);
        }
        Ok(Self {
            campaign_id: id.to_string(),
            scheduler_spec,
            bank_spec,
            require_complete_lifecycle,
        })
    }

    /// Run all shots and return a digest-bound report.
    pub fn run(
        &self,
        shots: &[CampaignShotPlan],
    ) -> Result<MultiShotCampaignReport, MultiShotCampaignError> {
        if shots.is_empty() {
            return Err(MultiShotCampaignError::EmptyCampaign);
        }
        let mut seen = HashSet::with_capacity(shots.len());
        let mut results = Vec::with_capacity(shots.len());
        for shot in shots {
            if !seen.insert(shot.shot_id.clone()) {
                return Err(MultiShotCampaignError::DuplicateShotId(
                    shot.shot_id.clone(),
                ));
            }
            results.push(self.run_shot(shot)?);
        }
        let passed_count = results.iter().filter(|shot| shot.passed).count();
        let mut report = MultiShotCampaignReport {
            schema_version: MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION.to_string(),
            campaign_id: self.campaign_id.clone(),
            shot_count: results.len(),
            passed_count,
            failed_count: results.len() - passed_count,
            require_complete_lifecycle: self.require_complete_lifecycle,
            expected_transition_states: EXPECTED_TRANSITION_STATES
                .iter()
                .map(|value| value.to_string())
                .collect(),
            pulsed_mpc_evidence_schema_version: PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
                .to_string(),
            pulsed_mpc_admission_digest_count: results
                .iter()
                .filter(|shot| shot.pulsed_mpc_admission_digest.is_some())
                .count(),
            shots: results,
            payload_sha256: String::new(),
        };
        report.payload_sha256 = report_digest(&report);
        Ok(report)
    }

    fn run_shot(
        &self,
        shot: &CampaignShotPlan,
    ) -> Result<CampaignShotResult, MultiShotCampaignError> {
        let mut scheduler = PulsedScenarioScheduler::new(self.scheduler_spec);
        let bank = CapacitorBank::new(
            self.bank_spec,
            shot.initial_bank_voltage_v,
            shot.initial_bank_current_a,
        )
        .map_err(|err| MultiShotCampaignError::Bank(err.to_string()))?;
        let initial_energy = bank.state().energy_j();
        let mut last_energy = initial_energy;
        let mut min_energy = initial_energy;
        let mut command_log = Vec::with_capacity(shot.samples.len());
        let mut failure_reason: Option<String> = None;

        for sample in &shot.samples {
            last_energy = sample.bank.energy_j;
            min_energy = min_energy.min(last_energy);
            match scheduler.step(sample.t_s, sample.plasma, sample.bank) {
                Ok(command) => command_log.push(command.into()),
                Err(err) => {
                    failure_reason = Some(err.to_string());
                    break;
                }
            }
        }

        let shot_phase_log: Vec<CampaignPhaseLog> = scheduler
            .audit_log()
            .iter()
            .map(|record| CampaignPhaseLog {
                t_s: record.t_s,
                state: record.to_state.as_str().to_string(),
                reason: record.reason.clone(),
            })
            .collect();
        let transition_states: Vec<String> =
            shot_phase_log.iter().map(|row| row.state.clone()).collect();
        let terminal_state = scheduler.state.as_str().to_string();
        if failure_reason.is_none() {
            failure_reason = self.admission_failure_reason(&terminal_state, &transition_states);
        }
        let trigger_timestamp_ns = shot_phase_log
            .iter()
            .find(|row| row.state == "burn")
            .map(|row| (row.t_s * 1_000_000_000.0).round() as u64);
        Ok(CampaignShotResult {
            shot_id: shot.shot_id.clone(),
            pulse_id: pulse_id(&self.campaign_id, &shot.shot_id),
            passed: failure_reason.is_none(),
            failure_reason,
            terminal_state,
            transition_states,
            command_log,
            shot_phase_log,
            capacitor_state_initial_j: initial_energy,
            capacitor_state_final_j: last_energy,
            energy_recovered_j: (last_energy - min_energy).max(0.0),
            trigger_timestamp_ns,
            pulsed_mpc_admission_digest: shot.pulsed_mpc_admission_digest.clone(),
        })
    }

    fn admission_failure_reason(
        &self,
        terminal_state: &str,
        transition_states: &[String],
    ) -> Option<String> {
        if self.require_complete_lifecycle
            && transition_states
                != EXPECTED_TRANSITION_STATES
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<String>>()
        {
            return Some("shot did not traverse the complete pulsed lifecycle".to_string());
        }
        if terminal_state != "idle" {
            return Some("shot did not return to idle".to_string());
        }
        None
    }
}

/// Campaign orchestration errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MultiShotCampaignError {
    EmptyCampaignId,
    EmptyCampaign,
    EmptyShotId,
    EmptySamples,
    DuplicateShotId(String),
    NonFinite(&'static str),
    Negative(&'static str),
    InvalidDigest(&'static str),
    Bank(String),
    Scheduler(String),
}

impl From<PulsedScenarioError> for MultiShotCampaignError {
    fn from(value: PulsedScenarioError) -> Self {
        Self::Scheduler(value.to_string())
    }
}

impl Display for MultiShotCampaignError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCampaignId => write!(f, "campaign_id must not be empty"),
            Self::EmptyCampaign => write!(f, "shots must not be empty"),
            Self::EmptyShotId => write!(f, "shot_id must not be empty"),
            Self::EmptySamples => write!(f, "samples must not be empty"),
            Self::DuplicateShotId(shot_id) => write!(f, "duplicate shot_id: {shot_id}"),
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::Negative(field) => write!(f, "{field} must be non-negative"),
            Self::InvalidDigest(field) => {
                write!(f, "{field} must be a lowercase SHA-256 hex digest")
            }
            Self::Bank(reason) => write!(f, "{reason}"),
            Self::Scheduler(reason) => write!(f, "{reason}"),
        }
    }
}

impl Error for MultiShotCampaignError {}

fn validate_finite(name: &'static str, value: f64) -> Result<(), MultiShotCampaignError> {
    if !value.is_finite() {
        return Err(MultiShotCampaignError::NonFinite(name));
    }
    Ok(())
}

fn validate_non_negative(name: &'static str, value: f64) -> Result<(), MultiShotCampaignError> {
    validate_finite(name, value)?;
    if value < 0.0 {
        return Err(MultiShotCampaignError::Negative(name));
    }
    Ok(())
}

fn validate_sha256_digest(
    name: &'static str,
    value: &str,
) -> Result<String, MultiShotCampaignError> {
    let digest = value.trim();
    if digest.len() != 64
        || !digest
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (*byte >= b'a' && *byte <= b'f'))
    {
        return Err(MultiShotCampaignError::InvalidDigest(name));
    }
    Ok(digest.to_string())
}

fn pulse_id(campaign_id: &str, shot_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"scpn-control.multi-shot-campaign.pulse-id.v1");
    hasher.update(campaign_id.as_bytes());
    hasher.update([0]);
    hasher.update(shot_id.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

fn report_digest(report: &MultiShotCampaignReport) -> String {
    let mut hasher = Sha256::new();
    hasher.update(report.schema_version.as_bytes());
    hasher.update(report.campaign_id.as_bytes());
    hasher.update(report.shot_count.to_string().as_bytes());
    hasher.update(report.passed_count.to_string().as_bytes());
    hasher.update(report.failed_count.to_string().as_bytes());
    hasher.update(report.pulsed_mpc_evidence_schema_version.as_bytes());
    hasher.update(
        report
            .pulsed_mpc_admission_digest_count
            .to_string()
            .as_bytes(),
    );
    for shot in &report.shots {
        hasher.update(shot.shot_id.as_bytes());
        hasher.update(shot.pulse_id.as_bytes());
        hasher.update(shot.passed.to_string().as_bytes());
        hasher.update(shot.terminal_state.as_bytes());
        if let Some(digest) = &shot.pulsed_mpc_admission_digest {
            hasher.update(digest.as_bytes());
        }
        for state in &shot.transition_states {
            hasher.update(state.as_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}
