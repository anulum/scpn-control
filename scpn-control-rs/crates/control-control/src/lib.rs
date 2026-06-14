// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust control-kernel module registry.

//! Control systems modules.
//!
//! Stage 7: PID, optimal, MPC, SNN, digital twin, SPI, SOC-learning, analytic.

pub mod analytic;
pub mod capacitor_bank;
pub mod digital_twin;
pub mod h_infinity;
pub mod mpc;
pub mod multi_shot_campaign;
pub mod optimal;
pub mod pid;
pub mod pulsed_scenario;
pub mod snn;
pub mod soc_learning;
pub mod spi;

/// Lowercase hex-encode a byte slice (e.g. a SHA-256 digest).
///
/// `digest` 0.11 returns an array type that no longer implements `LowerHex`,
/// so digests are encoded explicitly; the output is byte-identical to the
/// previous `format!("{:x}", finalize())`.
pub fn to_hex(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
