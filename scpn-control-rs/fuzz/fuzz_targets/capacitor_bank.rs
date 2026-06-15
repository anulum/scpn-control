// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Fuzz target: series-RLC capacitor-bank discharge ledger.
//
// The capacitor bank is the safety-critical numeric adapter the pulsed-shot
// scheduler and pulsed-MPC adapter call before admitting a shot, and it is the
// same code reached through the PyO3 `PyCapacitorBankModel` FFI boundary. This
// target drives it with adversarial-but-structured parameters and asserts the
// admission contract: an *admitted* energy balance (`energy_balance_passed`)
// must carry only finite, physically-ordered quantities — a passed admission
// must never smuggle a NaN/Inf ledger or a negative dissipation. Construction
// validators already reject non-finite inputs, so a violation here means a
// finite-but-extreme parameter overflowed the ledger while still being
// admitted. Work is bounded (step count, pulse selection) so the fuzzer probes
// the numerics rather than the allocator.

#![no_main]

use arbitrary::Arbitrary;
use control_control::capacitor_bank::{
    CapacitorBank, CapacitorBankSpec, PulseSpec, PulseWaveform,
};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    capacitance_f: f64,
    inductance_h: f64,
    series_resistance_ohm: f64,
    voltage_max_v: f64,
    recharge_power_kw: f64,
    initial_voltage_v: f64,
    initial_current_a: f64,
    peak_current_a: f64,
    duration_s: f64,
    waveform_sel: u8,
    dt: f64,
    n_steps: u16,
}

fuzz_target!(|input: Input| {
    let Ok(spec) = CapacitorBankSpec::new(
        input.capacitance_f,
        input.inductance_h,
        input.series_resistance_ohm,
        input.voltage_max_v,
        input.recharge_power_kw,
    ) else {
        return;
    };
    let Ok(mut bank) = CapacitorBank::new(spec, input.initial_voltage_v, input.initial_current_a)
    else {
        return;
    };

    let waveform = match input.waveform_sel % 3 {
        0 => PulseWaveform::Rect,
        1 => PulseWaveform::HalfSine,
        _ => PulseWaveform::ExpDecay,
    };
    let Ok(pulse) = PulseSpec::new(input.peak_current_a, input.duration_s, waveform) else {
        return;
    };

    // Pre-admission feasibility must never panic on any admitted pulse.
    let _ = bank.feasibility(pulse);

    // Bound the integration work: 1..=4096 steps keeps a fuzz iteration short.
    let n_steps = (input.n_steps % 4096) as usize + 1;
    if let Ok(report) = bank.discharge(pulse, input.dt, n_steps) {
        if report.energy_balance_passed {
            assert!(
                report.energy_initial_j.is_finite()
                    && report.energy_remaining_j.is_finite()
                    && report.capacitor_energy_remaining_j.is_finite()
                    && report.inductor_energy_remaining_j.is_finite()
                    && report.resistive_loss_j.is_finite()
                    && report.load_energy_j.is_finite()
                    && report.energy_delivered_j.is_finite()
                    && report.energy_balance_residual_j.is_finite(),
                "admitted energy balance carried a non-finite ledger quantity"
            );
            assert!(
                report.resistive_loss_j >= -1.0e-6,
                "admitted ledger reported negative ohmic dissipation"
            );
        }
    }
});
