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
