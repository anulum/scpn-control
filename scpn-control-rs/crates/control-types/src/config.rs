// ─────────────────────────────────────────────────────────────────────
// SCPN Control — Config
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ─────────────────────────────────────────────────────────────────────
use crate::error::FusionResult;
use crate::state::Grid2D;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Top-level reactor configuration.
/// Maps 1:1 to iter_config.json schema.
/// Must deserialize ALL 6 existing JSON config files without modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorConfig {
    pub reactor_name: String,
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

/// Linear solver configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub solver_method: String,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
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
        assert_eq!(cfg.reactor_name, "ITER-Like");
    }

    #[test]
    fn test_load_iter_config() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        assert_eq!(cfg.reactor_name, "ITER-Like");
    }

    #[test]
    fn test_load_validated_config() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        assert_eq!(cfg.reactor_name, "ITER-V1");
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
}
