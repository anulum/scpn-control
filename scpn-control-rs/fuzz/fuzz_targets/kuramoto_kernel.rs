// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Fuzz target: Kuramoto-Sakaguchi phase vector kernel.
//
// `order_parameter`, `wrap_phase`, and `kuramoto_sakaguchi_step` are the
// high-volume phase-array kernel of the SCPN phase runtime, exposed through the
// PyO3 FFI boundary. This target drives them with adversarial phase arrays and
// coupling parameters and asserts the dimensionless invariants that hold for
// finite input: the order-parameter magnitude r is finite and in [0, 1], the
// order-parameter phase is finite, and `wrap_phase` lands in (-pi, pi]. When
// input contains non-finite phases the kernel is only required not to panic
// (non-finite output is the documented garbage-in/garbage-out behaviour). The
// `omega` driver is built to match the phase length, honouring the documented
// equal-length precondition; array length and step count are bounded.

#![no_main]

use arbitrary::Arbitrary;
use control_math::kuramoto::{kuramoto_sakaguchi_step, order_parameter, wrap_phase};
use libfuzzer_sys::fuzz_target;

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Arbitrary)]
struct Input {
    phases: Vec<f64>,
    omega_seed: f64,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
    steps: u8,
}

fuzz_target!(|input: Input| {
    let n = input.phases.len();
    if n > 4096 {
        return;
    }
    let theta = &input.phases;

    let (r, psi) = order_parameter(theta);
    if theta.iter().all(|x| x.is_finite()) {
        assert!(
            r.is_finite() && (-1.0e-9..=1.0 + 1.0e-9).contains(&r),
            "order-parameter magnitude escaped [0, 1] for finite phases"
        );
        assert!(psi.is_finite(), "order-parameter phase non-finite for finite phases");
    }

    for &x in theta {
        if x.is_finite() {
            let w = wrap_phase(x);
            assert!(
                w.is_finite() && w > -PI - 1.0e-9 && w <= PI + 1.0e-9,
                "wrap_phase left (-pi, pi] for a finite input"
            );
        }
    }

    // `omega` must match `theta` length (documented precondition of the step).
    let omega: Vec<f64> = (0..n).map(|i| input.omega_seed + i as f64 * 1.0e-3).collect();
    if input.dt.is_finite()
        && input.k.is_finite()
        && input.alpha.is_finite()
        && input.zeta.is_finite()
    {
        let steps = (input.steps % 8) as usize;
        let mut cur = theta.clone();
        for _ in 0..steps {
            let res = kuramoto_sakaguchi_step(
                &cur,
                &omega,
                input.dt,
                input.k,
                input.alpha,
                input.zeta,
                input.psi_external,
            );
            cur = res.theta.to_vec();
        }
    }
});
