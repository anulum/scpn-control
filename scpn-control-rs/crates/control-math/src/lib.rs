// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust Crate.

#![deny(missing_docs, rustdoc::broken_intra_doc_links)]
//! Mathematical primitives for SCPN Control.

pub mod amr;
pub mod chebyshev;
pub mod elliptic;
pub mod fft;
pub mod gmres;
pub mod iga;
pub mod interp;
pub mod kuramoto;
pub mod linalg;
pub mod multigrid;
pub mod sor;
pub mod symplectic;
pub mod tridiag;
