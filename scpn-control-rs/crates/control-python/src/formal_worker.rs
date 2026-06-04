// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Native Formal Worker

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use z3::ast::{Bool, Int};
use z3::{SatResult, Solver};

const NUM_PLACES: usize = 4;
const NUM_TRANSITIONS: usize = 3;
const RECV_TIMEOUT_MS: u64 = 10;

const W_IN: [[i64; NUM_PLACES]; NUM_TRANSITIONS] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];

const W_OUT: [[i64; NUM_PLACES]; NUM_TRANSITIONS] = [[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]];

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
    pub(crate) max_marking: i64,
    pub(crate) max_depth: usize,
    pub(crate) channel_capacity: usize,
    pub(crate) core_z3: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct NativeFormalReport {
    pub(crate) submitted: u64,
    pub(crate) dropped: u64,
    pub(crate) checked: u64,
    pub(crate) failures: u64,
    pub(crate) last_checked_step: u64,
    pub(crate) last_failed_step: u64,
    pub(crate) last_status_code: u64,
}

impl NativeFormalReport {
    pub(crate) fn disabled() -> Self {
        Self::default()
    }

    pub(crate) fn status_label(&self) -> &'static str {
        match self.last_status_code {
            1 => "safe",
            2 => "violation",
            3 => "unknown",
            _ => "pending",
        }
    }
}

pub(crate) struct NativeFormalRuntime {
    sender: Option<Sender<PetriNetSnapshot>>,
    shutdown_flag: Arc<AtomicBool>,
    violation_flag: Arc<AtomicBool>,
    submitted: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    checked: Arc<AtomicU64>,
    failures: Arc<AtomicU64>,
    last_checked_step: Arc<AtomicU64>,
    last_failed_step: Arc<AtomicU64>,
    last_status_code: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl NativeFormalRuntime {
    pub(crate) fn spawn(
        config: NativeFormalConfig,
        shutdown_flag: Arc<AtomicBool>,
    ) -> std::io::Result<Self> {
        let (sender, receiver) = bounded::<PetriNetSnapshot>(config.channel_capacity);
        let violation_flag = Arc::new(AtomicBool::new(false));
        let submitted = Arc::new(AtomicU64::new(0));
        let dropped = Arc::new(AtomicU64::new(0));
        let checked = Arc::new(AtomicU64::new(0));
        let failures = Arc::new(AtomicU64::new(0));
        let last_checked_step = Arc::new(AtomicU64::new(0));
        let last_failed_step = Arc::new(AtomicU64::new(0));
        let last_status_code = Arc::new(AtomicU64::new(0));

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

        Ok(Self {
            sender: Some(sender),
            shutdown_flag,
            violation_flag,
            submitted,
            dropped,
            checked,
            failures,
            last_checked_step,
            last_failed_step,
            last_status_code,
            handle: Some(handle),
        })
    }

    pub(crate) fn try_submit(&self, snapshot: PetriNetSnapshot) -> bool {
        let Some(sender) = self.sender.as_ref() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        };

        match sender.try_send(snapshot) {
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

    pub(crate) fn violation_detected(&self) -> bool {
        self.violation_flag.load(Ordering::Acquire)
    }

    pub(crate) fn snapshot_report(&self) -> NativeFormalReport {
        NativeFormalReport {
            submitted: self.submitted.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            checked: self.checked.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            last_checked_step: self.last_checked_step.load(Ordering::Relaxed),
            last_failed_step: self.last_failed_step.load(Ordering::Relaxed),
            last_status_code: self.last_status_code.load(Ordering::Relaxed),
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
}

struct FormalWorker {
    receiver: Receiver<PetriNetSnapshot>,
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
                Ok(snapshot) => {
                    self.checked.fetch_add(1, Ordering::Relaxed);
                    self.last_checked_step
                        .store(snapshot.step_index, Ordering::Relaxed);

                    match verify_snapshot_with_z3(&snapshot, self.max_marking, self.max_depth) {
                        VerificationOutcome::Safe => {
                            self.last_status_code.store(1, Ordering::Release);
                        }
                        VerificationOutcome::Violation => {
                            self.last_status_code.store(2, Ordering::Release);
                            self.failures.fetch_add(1, Ordering::Relaxed);
                            self.last_failed_step
                                .store(snapshot.step_index, Ordering::Relaxed);
                            self.violation_flag.store(true, Ordering::Release);
                            self.shutdown_flag.store(true, Ordering::Release);
                        }
                        VerificationOutcome::Unknown => {
                            self.last_status_code.store(3, Ordering::Release);
                            self.failures.fetch_add(1, Ordering::Relaxed);
                            self.last_failed_step
                                .store(snapshot.step_index, Ordering::Relaxed);
                            self.violation_flag.store(true, Ordering::Release);
                            self.shutdown_flag.store(true, Ordering::Release);
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VerificationOutcome {
    Safe,
    Violation,
    Unknown,
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
                max_marking: 10,
                max_depth: 1,
                channel_capacity: 1,
                core_z3: 4095,
            },
            Arc::clone(&shutdown_flag),
        )
        .expect("native formal runtime should spawn");

        assert!(runtime.try_submit(PetriNetSnapshot {
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
}
