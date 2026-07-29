// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Fusion Types
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

#![deny(missing_docs, rustdoc::broken_intra_doc_links)]
//! Shared configuration, physical constants, errors, and state records for the
//! native SCPN CONTROL workspace.

/// Reactor, grid, profile, coil, and solver configuration schemas.
pub mod config;
/// Physical and numerical constants shared by native kernels.
pub mod constants;
/// Typed failures returned by native configuration and solver operations.
pub mod error;
/// Grid and plasma-state data contracts.
pub mod state;
