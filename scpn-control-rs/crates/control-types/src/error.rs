// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Error
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

use thiserror::Error;

#[derive(Error, Debug)]
/// Failures produced by native CONTROL configuration, numerics, and physics gates.
pub enum FusionError {
    /// An iterative solver diverged or otherwise failed at a known iteration.
    #[error("Solver diverged at iteration {iteration}: {message}")]
    SolverDiverged {
        /// Zero-based or solver-reported iteration of failure.
        iteration: usize,
        /// Human-readable divergence diagnostic.
        message: String,
    },

    /// Configuration content is invalid or internally inconsistent.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// A requested row/column index lies outside the active grid.
    #[error("Grid index out of bounds: row={row}, col={col}")]
    GridOutOfBounds {
        /// Requested row index.
        row: usize,
        /// Requested column index.
        col: usize,
    },

    /// A computed or supplied value violates a physical admissibility condition.
    #[error("Physics constraint violated: {0}")]
    PhysicsViolation(String),

    /// Filesystem or stream I/O failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON configuration or evidence decoding failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A linear-algebra operation failed or produced an inadmissible result.
    #[error("Linear algebra error: {0}")]
    LinAlg(String),
}

/// Native CONTROL result type using [`FusionError`].
pub type FusionResult<T> = Result<T, FusionError>;
