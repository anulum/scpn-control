// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust Crate.

//! Grad-Shafranov kernel and equilibrium solver.
//!
//! Stage 3: core kernel modules
//! Stage 4: ignition, transport, stability, RF heating

#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub mod amr_kernel;
pub mod bfield;
pub mod bout_interface;
pub mod ignition;
pub mod inverse;
pub mod jacobian;
pub mod jit;
pub mod kernel;
pub mod memory_transport;
pub mod mpi_domain;
pub mod particles;
pub mod pedestal;
pub mod rf_heating;
pub mod source;
pub mod spike_buffer;
pub mod stability;
pub mod transport;
pub mod vacuum;
pub mod vmec_interface;
pub mod xpoint;
