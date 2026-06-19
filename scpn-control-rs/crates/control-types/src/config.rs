// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Config
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

use crate::error::FusionResult;
use crate::state::Grid2D;
use serde::{Deserialize, Deserializer, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Minimum grid points per axis: a finite-difference grid needs at least two
/// nodes per axis so the spacing `(max - min) / (n - 1)` is well defined.
/// `Grid2D::new` computes `n - 1` on `usize`, so a resolution of 0 underflows
/// (panic) and 1 divides by zero — both are rejected here at deserialisation so
/// an untrusted config never reaches grid construction with a degenerate axis.
const MIN_GRID_RESOLUTION: usize = 2;

/// Deserialise and validate `grid_resolution`, rejecting any axis below
/// [`MIN_GRID_RESOLUTION`] with a serde error rather than letting a degenerate
/// value panic later in `Grid2D::new`.
fn deserialize_grid_resolution<'de, D>(deserializer: D) -> Result<[usize; 2], D::Error>
where
    D: Deserializer<'de>,
{
    let resolution = <[usize; 2]>::deserialize(deserializer)?;
    if resolution[0] < MIN_GRID_RESOLUTION || resolution[1] < MIN_GRID_RESOLUTION {
        return Err(serde::de::Error::custom(format!(
            "grid_resolution components must each be >= {MIN_GRID_RESOLUTION}, got {resolution:?}"
        )));
    }
    Ok(resolution)
}

/// Top-level reactor configuration.
/// Maps 1:1 to iter_config.json schema.
/// Must deserialize ALL 6 existing JSON config files without modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorConfig {
    pub reactor_name: String,
    #[serde(deserialize_with = "deserialize_grid_resolution")]
    pub grid_resolution: [usize; 2],
    pub dimensions: GridDimensions,
    pub physics: PhysicsParams,
    pub coils: Vec<CoilConfig>,
    pub solver: SolverConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDimensions {
    #[serde(rename = "R_min")]
    pub r_min: f64,
    #[serde(rename = "R_max")]
    pub r_max: f64,
    #[serde(rename = "Z_min")]
    pub z_min: f64,
    #[serde(rename = "Z_max")]
    pub z_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    pub plasma_current_target: f64,
    pub vacuum_permeability: f64,
    /// Optional H-mode pedestal profile configuration.
    /// When absent, the solver uses L-mode linear profiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profiles: Option<ProfileConfig>,
}

/// H-mode pedestal profile parameters (optional in JSON config).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Profile mode: "l-mode" or "h-mode"
    pub mode: String,
    /// Pressure gradient pedestal parameters
    pub params_p: Option<ProfileParams>,
    /// Poloidal current function (ff') pedestal parameters
    pub params_ff: Option<ProfileParams>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProfileParams {
    pub ped_top: f64,
    pub ped_width: f64,
    pub ped_height: f64,
    pub core_alpha: f64,
}

impl Default for ProfileParams {
    fn default() -> Self {
        ProfileParams {
            ped_top: 0.9,
            ped_width: 0.05,
            ped_height: 1.0,
            core_alpha: 1.5,
        }
    }
}

/// Coil geometric and physical configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilConfig {
    pub name: String,
    pub r: f64,
    pub z: f64,
    pub current: f64,
}

fn default_solver_method() -> String {
    "sor".to_string()
}

fn default_sor_omega() -> f64 {
    1.8
}

/// Linear solver configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    #[serde(default = "default_solver_method")]
    pub solver_method: String,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    #[serde(default = "default_sor_omega")]
    pub sor_omega: f64,
}

impl ReactorConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> FusionResult<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let config: ReactorConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub fn create_grid(&self) -> Grid2D {
        Grid2D::new(
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.dimensions.r_min,
            self.dimensions.r_max,
            self.dimensions.z_min,
            self.dimensions.z_max,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    #[test]
    fn test_load_default_config() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("default_config.json");
        let cfg = ReactorConfig::from_file(path).unwrap();
        assert_eq!(cfg.reactor_name, "SCPN-Standard-Model");
    }

    #[test]
    fn test_load_iter_config() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        assert_eq!(cfg.reactor_name, "ITER-Like-Demo");
    }

    #[test]
    fn test_load_validated_config() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        assert_eq!(cfg.reactor_name, "ITER-Validated");
    }

    #[test]
    fn test_load_all_six_configs() {
        let configs = [
            "iter_config.json",
            "validation/iter_validated_config.json",
            "validation/iter_genetic_config.json",
            "validation/iter_force_balanced.json",
            "scpn-control-rs/crates/control-types/test_data/default_config.json",
        ];
        for path in configs.iter() {
            let full_path = if path.contains("test_data") {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/default_config.json")
            } else {
                PathBuf::from(config_path(path))
            };
            let cfg = ReactorConfig::from_file(full_path).expect(path);
            assert!(!cfg.reactor_name.is_empty());
        }
    }

    #[test]
    fn test_roundtrip_serialization() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: ReactorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.reactor_name, cfg2.reactor_name);
        assert_eq!(cfg.grid_resolution, cfg2.grid_resolution);
        assert_eq!(cfg.coils.len(), cfg2.coils.len());
    }

    fn config_json_with_resolution(resolution: &str) -> String {
        format!(
            r#"{{
                "reactor_name": "fuzz",
                "grid_resolution": {resolution},
                "dimensions": {{"R_min": 2.0, "R_max": 10.0, "Z_min": -6.0, "Z_max": 6.0}},
                "physics": {{"plasma_current_target": 15.0, "vacuum_permeability": 1.0}},
                "coils": [],
                "solver": {{"max_iterations": 1000, "convergence_threshold": 1e-4}}
            }}"#
        )
    }

    /// Regression for the fuzz crash (Fuzz nightly run 27815219303): a config
    /// with `grid_resolution = [0, 165]` reached `Grid2D::new`, where `nr - 1`
    /// underflowed on `usize` and panicked. A degenerate axis must now be
    /// rejected at deserialisation with an error, never panic.
    #[test]
    fn test_rejects_degenerate_grid_resolution() {
        for resolution in ["[0, 165]", "[165, 0]", "[1, 165]", "[165, 1]", "[0, 0]"] {
            let json = config_json_with_resolution(resolution);
            let parsed = serde_json::from_str::<ReactorConfig>(&json);
            assert!(
                parsed.is_err(),
                "grid_resolution {resolution} must be rejected, not accepted"
            );
            let message = parsed.unwrap_err().to_string();
            assert!(
                message.contains("grid_resolution"),
                "error should name grid_resolution, got: {message}"
            );
        }
    }

    #[test]
    fn test_accepts_minimal_grid_resolution_and_builds_grid() {
        let json = config_json_with_resolution("[2, 2]");
        let cfg = serde_json::from_str::<ReactorConfig>(&json).expect("res >= 2 must parse");
        assert_eq!(cfg.grid_resolution, [2, 2]);
        // The smallest valid grid constructs without panicking and has finite spacing.
        let grid = cfg.create_grid();
        assert_eq!(grid.nr, 2);
        assert_eq!(grid.nz, 2);
        assert!(grid.dr.is_finite() && grid.dr > 0.0);
        assert!(grid.dz.is_finite() && grid.dz > 0.0);
    }
}
