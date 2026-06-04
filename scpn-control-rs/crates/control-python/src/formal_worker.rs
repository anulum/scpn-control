// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Native Formal Worker

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use sha2::{Digest, Sha256};
use z3::ast::{Bool, Int};
use z3::{SatResult, Solver};

const NUM_PLACES: usize = 4;
const NUM_TRANSITIONS: usize = 3;
const RECV_TIMEOUT_MS: u64 = 10;
const AOT_CERTIFICATE_SCHEMA_VERSION: &str = "scpn-control.native-formal.aot-certificate.v1";
const AOT_CERTIFICATE_ID: &str = "bounded-petri-marking-sufficient-invariant";
const AOT_CERTIFICATE_CONTRACT: &str =
    "non_negative_markings_and_bounded_core_total_imply_depth_bounded_marking_safety";

const W_IN: [[i64; NUM_PLACES]; NUM_TRANSITIONS] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];

const W_OUT: [[i64; NUM_PLACES]; NUM_TRANSITIONS] = [[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeFormalMode {
    AsyncDrop,
    SyncStride,
    AotCertificate,
}

impl NativeFormalMode {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "async" | "async_drop" | "drop" => Some(Self::AsyncDrop),
            "sync" | "sync_stride" | "strict" | "blocking" => Some(Self::SyncStride),
            "aot" | "aot_certificate" | "certificate" | "runtime_certificate" => {
                Some(Self::AotCertificate)
            }
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::AsyncDrop => "async_drop",
            Self::SyncStride => "sync_stride",
            Self::AotCertificate => "aot_certificate",
        }
    }

    fn code(self) -> u64 {
        match self {
            Self::AsyncDrop => 1,
            Self::SyncStride => 2,
            Self::AotCertificate => 3,
        }
    }

    fn from_code(value: u64) -> Option<Self> {
        match value {
            1 => Some(Self::AsyncDrop),
            2 => Some(Self::SyncStride),
            3 => Some(Self::AotCertificate),
            _ => None,
        }
    }

    pub(crate) fn backend_label(self) -> &'static str {
        match self {
            Self::AsyncDrop | Self::SyncStride => "rust-z3",
            Self::AotCertificate => "compiled-certificate",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PetriNetSnapshot {
    pub(crate) step_index: u64,
    pub(crate) active_markings: [i64; NUM_PLACES],
}

impl PetriNetSnapshot {
    pub(crate) fn from_control_outputs(
        step_index: u64,
        error_r: f64,
        error_z: f64,
        r_command: f64,
        z_command: f64,
    ) -> Self {
        Self {
            step_index,
            active_markings: [
                finite_ceiling(error_r),
                finite_ceiling(error_z),
                finite_ceiling(r_command),
                finite_ceiling(z_command),
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeFormalConfig {
    pub(crate) mode: NativeFormalMode,
    pub(crate) max_marking: i64,
    pub(crate) max_depth: usize,
    pub(crate) channel_capacity: usize,
    pub(crate) core_z3: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct NativeFormalReport {
    pub(crate) mode_code: u64,
    pub(crate) generated: u64,
    pub(crate) submitted: u64,
    pub(crate) dropped: u64,
    pub(crate) checked: u64,
    pub(crate) failures: u64,
    pub(crate) last_checked_step: u64,
    pub(crate) last_failed_step: u64,
    pub(crate) last_status_code: u64,
    pub(crate) sync_wait_count: u64,
    pub(crate) sync_wait_total_ns: u64,
    pub(crate) sync_wait_min_ns: u64,
    pub(crate) sync_wait_p50_ns: u64,
    pub(crate) sync_wait_p95_ns: u64,
    pub(crate) sync_wait_p99_ns: u64,
    pub(crate) sync_wait_max_ns: u64,
    pub(crate) certificate_admitted: bool,
    pub(crate) certificate_schema_version: String,
    pub(crate) certificate_id: String,
    pub(crate) certificate_contract: String,
    pub(crate) certificate_assumption_sha256: String,
}

impl Default for NativeFormalReport {
    fn default() -> Self {
        Self::disabled()
    }
}

impl NativeFormalReport {
    pub(crate) fn disabled() -> Self {
        Self {
            mode_code: 0,
            generated: 0,
            submitted: 0,
            dropped: 0,
            checked: 0,
            failures: 0,
            last_checked_step: 0,
            last_failed_step: 0,
            last_status_code: 0,
            sync_wait_count: 0,
            sync_wait_total_ns: 0,
            sync_wait_min_ns: 0,
            sync_wait_p50_ns: 0,
            sync_wait_p95_ns: 0,
            sync_wait_p99_ns: 0,
            sync_wait_max_ns: 0,
            certificate_admitted: false,
            certificate_schema_version: String::new(),
            certificate_id: String::new(),
            certificate_contract: String::new(),
            certificate_assumption_sha256: String::new(),
        }
    }

    pub(crate) fn mode_label(&self) -> &'static str {
        NativeFormalMode::from_code(self.mode_code).map_or("disabled", NativeFormalMode::label)
    }

    pub(crate) fn status_label(&self) -> &'static str {
        match self.last_status_code {
            1 => "safe",
            2 => "violation",
            3 => "unknown",
            _ => "pending",
        }
    }

    pub(crate) fn backend_label(&self) -> &'static str {
        NativeFormalMode::from_code(self.mode_code)
            .map_or("disabled", NativeFormalMode::backend_label)
    }

    pub(crate) fn effective_verification_rate(&self) -> f64 {
        let denominator = self.submitted.saturating_add(self.dropped);
        if denominator == 0 {
            0.0
        } else {
            100.0 * self.checked as f64 / denominator as f64
        }
    }
}

struct FormalRequest {
    snapshot: PetriNetSnapshot,
    response: Option<Sender<VerificationOutcome>>,
}

pub(crate) struct NativeFormalRuntime {
    mode: NativeFormalMode,
    sender: Option<Sender<FormalRequest>>,
    shutdown_flag: Arc<AtomicBool>,
    violation_flag: Arc<AtomicBool>,
    generated: Arc<AtomicU64>,
    submitted: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    checked: Arc<AtomicU64>,
    failures: Arc<AtomicU64>,
    last_checked_step: Arc<AtomicU64>,
    last_failed_step: Arc<AtomicU64>,
    last_status_code: Arc<AtomicU64>,
    sync_wait_total_ns: Arc<AtomicU64>,
    sync_wait_max_ns: Arc<AtomicU64>,
    sync_wait_samples: Arc<Mutex<Vec<u64>>>,
    certificate_monitor: Option<CompiledCertificateMonitor>,
    handle: Option<JoinHandle<()>>,
}

impl NativeFormalRuntime {
    pub(crate) fn spawn(
        config: NativeFormalConfig,
        shutdown_flag: Arc<AtomicBool>,
    ) -> std::io::Result<Self> {
        let violation_flag = Arc::new(AtomicBool::new(false));
        let generated = Arc::new(AtomicU64::new(0));
        let submitted = Arc::new(AtomicU64::new(0));
        let dropped = Arc::new(AtomicU64::new(0));
        let checked = Arc::new(AtomicU64::new(0));
        let failures = Arc::new(AtomicU64::new(0));
        let last_checked_step = Arc::new(AtomicU64::new(0));
        let last_failed_step = Arc::new(AtomicU64::new(0));
        let last_status_code = Arc::new(AtomicU64::new(0));
        let sync_wait_total_ns = Arc::new(AtomicU64::new(0));
        let sync_wait_max_ns = Arc::new(AtomicU64::new(0));
        let sync_wait_samples = Arc::new(Mutex::new(Vec::new()));
        let certificate_monitor = if config.mode == NativeFormalMode::AotCertificate {
            Some(CompiledCertificateMonitor::admit(
                config.max_marking,
                config.max_depth,
            )?)
        } else {
            None
        };
        let (sender, handle) = if config.mode == NativeFormalMode::AotCertificate {
            (None, None)
        } else {
            let (sender, receiver) = bounded::<FormalRequest>(config.channel_capacity);
            let worker = FormalWorker {
                receiver,
                shutdown_flag: Arc::clone(&shutdown_flag),
                violation_flag: Arc::clone(&violation_flag),
                checked: Arc::clone(&checked),
                failures: Arc::clone(&failures),
                last_checked_step: Arc::clone(&last_checked_step),
                last_failed_step: Arc::clone(&last_failed_step),
                last_status_code: Arc::clone(&last_status_code),
                max_marking: config.max_marking,
                max_depth: config.max_depth,
                core_z3: config.core_z3,
            };

            let handle = std::thread::Builder::new()
                .name("scpn-native-z3-worker".to_string())
                .spawn(move || worker.run())?;
            (Some(sender), Some(handle))
        };

        Ok(Self {
            mode: config.mode,
            sender,
            shutdown_flag,
            violation_flag,
            generated,
            submitted,
            dropped,
            checked,
            failures,
            last_checked_step,
            last_failed_step,
            last_status_code,
            sync_wait_total_ns,
            sync_wait_max_ns,
            sync_wait_samples,
            certificate_monitor,
            handle,
        })
    }

    pub(crate) fn try_submit_async(&self, snapshot: PetriNetSnapshot) -> bool {
        self.generated.fetch_add(1, Ordering::Relaxed);
        let Some(sender) = self.sender.as_ref() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        };

        match sender.try_send(FormalRequest {
            snapshot,
            response: None,
        }) {
            Ok(()) => {
                self.submitted.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    pub(crate) fn submit_sync(&self, snapshot: PetriNetSnapshot) -> bool {
        self.generated.fetch_add(1, Ordering::Relaxed);
        let Some(sender) = self.sender.as_ref() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            self.fail_closed(snapshot.step_index);
            return false;
        };

        let wait_start = Instant::now();
        let (response_tx, response_rx) = bounded::<VerificationOutcome>(0);
        if sender
            .send(FormalRequest {
                snapshot,
                response: Some(response_tx),
            })
            .is_err()
        {
            self.record_sync_wait(wait_start.elapsed().as_nanos() as u64);
            self.dropped.fetch_add(1, Ordering::Relaxed);
            self.fail_closed(snapshot.step_index);
            return false;
        }
        self.submitted.fetch_add(1, Ordering::Relaxed);

        let outcome = response_rx.recv().unwrap_or(VerificationOutcome::Unknown);
        self.record_sync_wait(wait_start.elapsed().as_nanos() as u64);
        matches!(outcome, VerificationOutcome::Safe)
    }

    pub(crate) fn check_aot_certificate(&self, snapshot: PetriNetSnapshot) -> bool {
        self.generated.fetch_add(1, Ordering::Relaxed);
        self.submitted.fetch_add(1, Ordering::Relaxed);
        self.checked.fetch_add(1, Ordering::Relaxed);
        self.last_checked_step
            .store(snapshot.step_index, Ordering::Relaxed);

        let outcome = self
            .certificate_monitor
            .as_ref()
            .map_or(VerificationOutcome::Unknown, |monitor| {
                monitor.evaluate(&snapshot)
            });
        self.record_runtime_outcome(snapshot.step_index, outcome);
        matches!(outcome, VerificationOutcome::Safe)
    }

    pub(crate) fn violation_detected(&self) -> bool {
        self.violation_flag.load(Ordering::Acquire)
    }

    pub(crate) fn snapshot_report(&self) -> NativeFormalReport {
        let samples = match self.sync_wait_samples.lock() {
            Ok(samples) => samples.clone(),
            Err(_) => Vec::new(),
        };
        let wait_count = samples.len() as u64;
        let certificate_admission = self
            .certificate_monitor
            .as_ref()
            .map(|monitor| &monitor.admission);
        NativeFormalReport {
            mode_code: self.mode.code(),
            generated: self.generated.load(Ordering::Relaxed),
            submitted: self.submitted.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            checked: self.checked.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            last_checked_step: self.last_checked_step.load(Ordering::Relaxed),
            last_failed_step: self.last_failed_step.load(Ordering::Relaxed),
            last_status_code: self.last_status_code.load(Ordering::Relaxed),
            sync_wait_count: wait_count,
            sync_wait_total_ns: self.sync_wait_total_ns.load(Ordering::Relaxed),
            sync_wait_min_ns: percentile_from_samples(&samples, 0),
            sync_wait_p50_ns: percentile_from_samples(&samples, 50),
            sync_wait_p95_ns: percentile_from_samples(&samples, 95),
            sync_wait_p99_ns: percentile_from_samples(&samples, 99),
            sync_wait_max_ns: self.sync_wait_max_ns.load(Ordering::Relaxed),
            certificate_admitted: certificate_admission.is_some(),
            certificate_schema_version: certificate_admission
                .map_or_else(String::new, |admission| {
                    admission.schema_version.to_string()
                }),
            certificate_id: certificate_admission.map_or_else(String::new, |admission| {
                admission.certificate_id.to_string()
            }),
            certificate_contract: certificate_admission
                .map_or_else(String::new, |admission| admission.contract.to_string()),
            certificate_assumption_sha256: certificate_admission
                .map_or_else(String::new, |admission| admission.assumption_sha256.clone()),
        }
    }

    pub(crate) fn shutdown(mut self) -> NativeFormalReport {
        self.shutdown_flag.store(true, Ordering::Release);
        self.sender.take();

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }

        self.snapshot_report()
    }

    fn fail_closed(&self, step_index: u64) {
        self.last_status_code.store(3, Ordering::Release);
        self.failures.fetch_add(1, Ordering::Relaxed);
        self.last_failed_step.store(step_index, Ordering::Relaxed);
        self.violation_flag.store(true, Ordering::Release);
        self.shutdown_flag.store(true, Ordering::Release);
    }

    fn record_sync_wait(&self, wait_ns: u64) {
        self.sync_wait_total_ns
            .fetch_add(wait_ns, Ordering::Relaxed);
        atomic_max(&self.sync_wait_max_ns, wait_ns);
        if let Ok(mut samples) = self.sync_wait_samples.lock() {
            samples.push(wait_ns);
        }
    }

    fn record_runtime_outcome(&self, step_index: u64, outcome: VerificationOutcome) {
        match outcome {
            VerificationOutcome::Safe => {
                self.last_status_code.store(1, Ordering::Release);
            }
            VerificationOutcome::Violation => {
                self.last_status_code.store(2, Ordering::Release);
                self.failures.fetch_add(1, Ordering::Relaxed);
                self.last_failed_step.store(step_index, Ordering::Relaxed);
                self.violation_flag.store(true, Ordering::Release);
                self.shutdown_flag.store(true, Ordering::Release);
            }
            VerificationOutcome::Unknown => {
                self.fail_closed(step_index);
            }
        }
    }
}

struct FormalWorker {
    receiver: Receiver<FormalRequest>,
    shutdown_flag: Arc<AtomicBool>,
    violation_flag: Arc<AtomicBool>,
    checked: Arc<AtomicU64>,
    failures: Arc<AtomicU64>,
    last_checked_step: Arc<AtomicU64>,
    last_failed_step: Arc<AtomicU64>,
    last_status_code: Arc<AtomicU64>,
    max_marking: i64,
    max_depth: usize,
    core_z3: usize,
}

impl FormalWorker {
    fn run(self) {
        pin_current_thread(self.core_z3);

        while !self.shutdown_flag.load(Ordering::Acquire) {
            match self
                .receiver
                .recv_timeout(Duration::from_millis(RECV_TIMEOUT_MS))
            {
                Ok(request) => self.verify_request(request),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    fn verify_request(&self, request: FormalRequest) {
        self.checked.fetch_add(1, Ordering::Relaxed);
        self.last_checked_step
            .store(request.snapshot.step_index, Ordering::Relaxed);

        let outcome = verify_snapshot_with_z3(&request.snapshot, self.max_marking, self.max_depth);
        match outcome {
            VerificationOutcome::Safe => {
                self.last_status_code.store(1, Ordering::Release);
            }
            VerificationOutcome::Violation => {
                self.last_status_code.store(2, Ordering::Release);
                self.failures.fetch_add(1, Ordering::Relaxed);
                self.last_failed_step
                    .store(request.snapshot.step_index, Ordering::Relaxed);
                self.violation_flag.store(true, Ordering::Release);
                self.shutdown_flag.store(true, Ordering::Release);
            }
            VerificationOutcome::Unknown => {
                self.last_status_code.store(3, Ordering::Release);
                self.failures.fetch_add(1, Ordering::Relaxed);
                self.last_failed_step
                    .store(request.snapshot.step_index, Ordering::Relaxed);
                self.violation_flag.store(true, Ordering::Release);
                self.shutdown_flag.store(true, Ordering::Release);
            }
        }

        if let Some(response) = request.response {
            let _ = response.send(outcome);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VerificationOutcome {
    Safe,
    Violation,
    Unknown,
}

#[derive(Clone, Debug)]
struct CertificateAdmission {
    schema_version: &'static str,
    certificate_id: &'static str,
    contract: &'static str,
    assumption_sha256: String,
}

impl CertificateAdmission {
    fn new(max_marking: i64, max_depth: usize) -> std::io::Result<Self> {
        if max_marking <= 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "AOT certificate admission requires a positive max_marking",
            ));
        }
        let bounded_depth = i64::try_from(max_depth).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "AOT certificate admission requires max_depth to fit in i64",
            )
        })?;
        if bounded_depth <= 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "AOT certificate admission requires a positive max_depth",
            ));
        }

        let payload = build_certificate_assumption_payload(max_marking, bounded_depth);
        Ok(Self {
            schema_version: AOT_CERTIFICATE_SCHEMA_VERSION,
            certificate_id: AOT_CERTIFICATE_ID,
            contract: AOT_CERTIFICATE_CONTRACT,
            assumption_sha256: sha256_hex(payload.as_bytes()),
        })
    }
}

#[derive(Clone, Debug)]
struct CompiledCertificateMonitor {
    max_marking: i64,
    max_depth: i64,
    admission: CertificateAdmission,
}

impl CompiledCertificateMonitor {
    fn admit(max_marking: i64, max_depth: usize) -> std::io::Result<Self> {
        let admission = CertificateAdmission::new(max_marking, max_depth)?;
        let bounded_depth = i64::try_from(max_depth).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "AOT certificate max_depth must fit in i64",
            )
        })?;
        Ok(Self {
            max_marking,
            max_depth: bounded_depth,
            admission,
        })
    }

    fn evaluate(&self, snapshot: &PetriNetSnapshot) -> VerificationOutcome {
        verify_snapshot_with_compiled_certificate(snapshot, self.max_marking, self.max_depth)
    }
}

fn build_certificate_assumption_payload(max_marking: i64, max_depth: i64) -> String {
    [
        format!("schema_version={AOT_CERTIFICATE_SCHEMA_VERSION}"),
        format!("certificate_id={AOT_CERTIFICATE_ID}"),
        format!("contract={AOT_CERTIFICATE_CONTRACT}"),
        format!("num_places={NUM_PLACES}"),
        format!("num_transitions={NUM_TRANSITIONS}"),
        format!("max_marking={max_marking}"),
        format!("max_depth={max_depth}"),
        format!("w_in={}", matrix_to_canonical(&W_IN)),
        format!("w_out={}", matrix_to_canonical(&W_OUT)),
        String::new(),
    ]
    .join("\n")
}

fn matrix_to_canonical<const R: usize, const C: usize>(matrix: &[[i64; C]; R]) -> String {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(i64::to_string)
                .collect::<Vec<String>>()
                .join(",")
        })
        .collect::<Vec<String>>()
        .join(";")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

fn verify_snapshot_with_compiled_certificate(
    snapshot: &PetriNetSnapshot,
    max_marking: i64,
    depth: i64,
) -> VerificationOutcome {
    if max_marking <= 0 || depth < 0 {
        return VerificationOutcome::Unknown;
    }
    if snapshot.active_markings.iter().any(|marking| *marking < 0) {
        return VerificationOutcome::Violation;
    }

    let core_total = match snapshot.active_markings[..3]
        .iter()
        .try_fold(0_i64, |acc, marking| acc.checked_add(*marking))
    {
        Some(value) => value,
        None => return VerificationOutcome::Unknown,
    };
    let p3_reachable_bound = match snapshot.active_markings[3].checked_add(depth) {
        Some(value) => value,
        None => return VerificationOutcome::Unknown,
    };

    if core_total <= max_marking && p3_reachable_bound <= max_marking {
        VerificationOutcome::Safe
    } else {
        VerificationOutcome::Violation
    }
}

fn verify_snapshot_with_z3(
    snapshot: &PetriNetSnapshot,
    max_marking: i64,
    depth: usize,
) -> VerificationOutcome {
    z3::with_z3_config(&z3::Config::new(), || {
        let solver = Solver::new();

        let mut markings: Vec<Vec<Int>> = Vec::with_capacity(depth + 1);
        for step in 0..=depth {
            let mut row = Vec::with_capacity(NUM_PLACES);
            for place in 0..NUM_PLACES {
                row.push(Int::new_const(format!("M_{step}_{place}")));
            }
            markings.push(row);
        }

        let mut firings: Vec<Vec<Bool>> = Vec::with_capacity(depth);
        for step in 0..depth {
            let mut row = Vec::with_capacity(NUM_TRANSITIONS);
            for transition in 0..NUM_TRANSITIONS {
                row.push(Bool::new_const(format!("F_{step}_{transition}")));
            }
            firings.push(row);
        }

        for (place, marking) in markings[0].iter().enumerate() {
            solver.assert(marking.eq(Int::from_i64(snapshot.active_markings[place])));
        }

        let zero = Int::from_i64(0);
        let one = Int::from_i64(1);

        for step in 0..depth {
            for place in 0..NUM_PLACES {
                let mut next_marking = markings[step][place].clone();

                for transition in 0..NUM_TRANSITIONS {
                    let firing_as_int = firings[step][transition].ite(&one, &zero);

                    if W_IN[transition][place] > 0 {
                        next_marking -=
                            Int::from_i64(W_IN[transition][place]) * firing_as_int.clone();
                    }

                    if W_OUT[transition][place] > 0 {
                        next_marking += Int::from_i64(W_OUT[transition][place]) * firing_as_int;
                    }
                }

                solver.assert(markings[step + 1][place].eq(next_marking));
                solver.assert(markings[step + 1][place].ge(Int::from_i64(0)));
            }
        }

        let bound = Int::from_i64(max_marking);
        let mut violations: Vec<Bool> = Vec::with_capacity(depth * NUM_PLACES * 2);
        for row in markings.iter().take(depth + 1).skip(1) {
            for marking in row.iter().take(NUM_PLACES) {
                violations.push(marking.gt(&bound));
                violations.push(marking.lt(&zero));
            }
        }

        let violation_refs: Vec<&Bool> = violations.iter().collect();
        solver.assert(Bool::or(&violation_refs));

        match solver.check() {
            SatResult::Unsat => VerificationOutcome::Safe,
            SatResult::Sat => VerificationOutcome::Violation,
            SatResult::Unknown => VerificationOutcome::Unknown,
        }
    })
}

fn finite_ceiling(value: f64) -> i64 {
    if !value.is_finite() {
        return i64::MAX;
    }

    let magnitude = value.abs().ceil();
    if magnitude >= i64::MAX as f64 {
        i64::MAX
    } else {
        magnitude as i64
    }
}

fn pin_current_thread(core_index: usize) {
    let Some(core_ids) = core_affinity::get_core_ids() else {
        return;
    };

    let Some(core_id) = core_ids.into_iter().nth(core_index) else {
        return;
    };

    core_affinity::set_for_current(core_id);
}

fn atomic_max(target: &AtomicU64, value: u64) {
    let mut observed = target.load(Ordering::Relaxed);
    while value > observed {
        match target.compare_exchange(observed, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(current) => observed = current,
        }
    }
}

fn percentile_from_samples(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let bounded_percentile = percentile.min(100) as f64 / 100.0;
    let index = ((sorted.len().saturating_sub(1)) as f64 * bounded_percentile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z3_worker_accepts_bounded_snapshot() {
        let snapshot = PetriNetSnapshot {
            step_index: 1,
            active_markings: [1, 1, 1, 0],
        };

        assert_eq!(
            verify_snapshot_with_z3(&snapshot, 100, 4),
            VerificationOutcome::Safe
        );
    }

    #[test]
    fn z3_worker_rejects_reachable_bound_violation() {
        let snapshot = PetriNetSnapshot {
            step_index: 1,
            active_markings: [101, 0, 0, 0],
        };

        assert_eq!(
            verify_snapshot_with_z3(&snapshot, 100, 1),
            VerificationOutcome::Violation
        );
    }

    #[test]
    fn runtime_sets_violation_flag_fail_closed() {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let runtime = NativeFormalRuntime::spawn(
            NativeFormalConfig {
                mode: NativeFormalMode::AsyncDrop,
                max_marking: 10,
                max_depth: 1,
                channel_capacity: 1,
                core_z3: 4095,
            },
            Arc::clone(&shutdown_flag),
        )
        .expect("native formal runtime should spawn");

        assert!(runtime.try_submit_async(PetriNetSnapshot {
            step_index: 30,
            active_markings: [11, 0, 0, 0],
        }));

        for _ in 0..500 {
            if runtime.violation_detected() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        assert!(runtime.violation_detected());
        let report = runtime.shutdown();
        assert_eq!(report.failures, 1);
        assert_eq!(report.last_failed_step, 30);
    }

    #[test]
    fn runtime_sync_stride_waits_for_worker_result() {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let runtime = NativeFormalRuntime::spawn(
            NativeFormalConfig {
                mode: NativeFormalMode::SyncStride,
                max_marking: 10,
                max_depth: 1,
                channel_capacity: 1,
                core_z3: 4095,
            },
            shutdown_flag,
        )
        .expect("native formal runtime should spawn");

        assert!(runtime.submit_sync(PetriNetSnapshot {
            step_index: 20,
            active_markings: [1, 0, 0, 0],
        }));

        let report = runtime.shutdown();
        assert_eq!(report.generated, 1);
        assert_eq!(report.submitted, 1);
        assert_eq!(report.checked, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(report.failures, 0);
        assert_eq!(report.sync_wait_count, 1);
        assert!(report.sync_wait_total_ns > 0);
    }

    #[test]
    fn compiled_certificate_accepts_sound_invariant() {
        let snapshot = PetriNetSnapshot {
            step_index: 1,
            active_markings: [3, 2, 1, 4],
        };

        assert_eq!(
            verify_snapshot_with_compiled_certificate(&snapshot, 10, 4),
            VerificationOutcome::Safe
        );
    }

    #[test]
    fn compiled_certificate_rejects_depth_reachable_p3_violation() {
        let snapshot = PetriNetSnapshot {
            step_index: 1,
            active_markings: [0, 0, 0, 8],
        };

        assert_eq!(
            verify_snapshot_with_compiled_certificate(&snapshot, 10, 4),
            VerificationOutcome::Violation
        );
    }

    #[test]
    fn compiled_certificate_safe_region_is_z3_safe_for_small_domain() {
        for p0 in 0..=3 {
            for p1 in 0..=3 {
                for p2 in 0..=3 {
                    for p3 in 0..=3 {
                        let snapshot = PetriNetSnapshot {
                            step_index: 1,
                            active_markings: [p0, p1, p2, p3],
                        };
                        if verify_snapshot_with_compiled_certificate(&snapshot, 6, 3)
                            == VerificationOutcome::Safe
                        {
                            assert_eq!(
                                verify_snapshot_with_z3(&snapshot, 6, 3),
                                VerificationOutcome::Safe
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn runtime_aot_certificate_checks_inline_without_worker_queue() {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let runtime = NativeFormalRuntime::spawn(
            NativeFormalConfig {
                mode: NativeFormalMode::AotCertificate,
                max_marking: 10,
                max_depth: 4,
                channel_capacity: 1,
                core_z3: 4095,
            },
            shutdown_flag,
        )
        .expect("native formal runtime should spawn");

        assert!(runtime.check_aot_certificate(PetriNetSnapshot {
            step_index: 30,
            active_markings: [1, 1, 1, 1],
        }));

        let report = runtime.shutdown();
        assert_eq!(report.mode_label(), "aot_certificate");
        assert_eq!(report.backend_label(), "compiled-certificate");
        assert!(report.certificate_admitted);
        assert_eq!(
            report.certificate_schema_version,
            AOT_CERTIFICATE_SCHEMA_VERSION
        );
        assert_eq!(report.certificate_id, AOT_CERTIFICATE_ID);
        assert_eq!(report.certificate_contract, AOT_CERTIFICATE_CONTRACT);
        assert_eq!(report.certificate_assumption_sha256.len(), 64);
        assert_eq!(report.generated, 1);
        assert_eq!(report.submitted, 1);
        assert_eq!(report.checked, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(report.failures, 0);
        assert_eq!(report.sync_wait_count, 0);
    }

    #[test]
    fn aot_certificate_admission_digest_binds_assumptions() {
        let baseline = CompiledCertificateMonitor::admit(10, 4)
            .expect("valid certificate assumptions should admit");
        let changed_marking = CompiledCertificateMonitor::admit(11, 4)
            .expect("valid certificate assumptions should admit");
        let changed_depth = CompiledCertificateMonitor::admit(10, 5)
            .expect("valid certificate assumptions should admit");

        assert_eq!(baseline.admission.assumption_sha256.len(), 64);
        assert!(baseline
            .admission
            .assumption_sha256
            .chars()
            .all(|ch| ch.is_ascii_hexdigit()));
        assert_ne!(
            baseline.admission.assumption_sha256,
            changed_marking.admission.assumption_sha256
        );
        assert_ne!(
            baseline.admission.assumption_sha256,
            changed_depth.admission.assumption_sha256
        );
    }
}
