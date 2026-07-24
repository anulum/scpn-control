# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for integrated scenario presets leaf

"""Drive production scenario config presets leaf and owner re-exports."""

from __future__ import annotations

import pytest

import scpn_control.core.integrated_scenario as owner
import scpn_control.core.integrated_scenario_presets as presets


def test_owner_reexports_presets_surface_by_identity() -> None:
    """Owner product surface re-exports the presets leaf by identity."""
    assert owner.ScenarioConfig is presets.ScenarioConfig
    assert owner.iter_baseline_scenario is presets.iter_baseline_scenario
    assert owner.iter_hybrid_scenario is presets.iter_hybrid_scenario
    assert owner.nstx_u_scenario is presets.nstx_u_scenario
    assert owner.scenario_config_sha256 is presets.scenario_config_sha256
    assert owner.enabled_scenario_modules is presets.enabled_scenario_modules


def test_facility_presets_have_physical_ordering() -> None:
    """Facility presets keep tokamak ordering a < R0 and positive actuators."""
    for factory in (
        presets.iter_baseline_scenario,
        presets.iter_hybrid_scenario,
        presets.nstx_u_scenario,
    ):
        cfg = factory()
        assert cfg.a < cfg.R0
        assert cfg.Ip_MA > 0.0
        assert cfg.t_end > cfg.t_start
        assert cfg.dt > 0.0
        presets._validate_config(cfg)


def test_validate_config_rejects_bad_triangularity() -> None:
    """Triangularity outside (-1, 1) fails closed."""
    cfg = presets.nstx_u_scenario()
    cfg.delta = 1.5
    with pytest.raises(ValueError, match="triangularity"):
        presets._validate_config(cfg)


def test_config_sha256_is_stable_hex() -> None:
    """Payload hash is a deterministic 64-character hex digest."""
    cfg = presets.iter_baseline_scenario()
    digest = presets.scenario_config_sha256(cfg)
    assert len(digest) == 64
    int(digest, 16)
    assert digest == presets.scenario_config_sha256(cfg)


def test_enabled_modules_track_flags() -> None:
    """Module inventory includes optional physics when flags/power are set."""
    cfg = presets.iter_baseline_scenario()
    modules = presets.enabled_scenario_modules(cfg)
    assert "transport" in modules
    assert "current_drive" in modules  # ECCD + NBI power on baseline
    assert "sawtooth" in modules
    cfg.include_sawteeth = False
    cfg.P_eccd_MW = 0.0
    cfg.P_nbi_MW = 0.0
    modules = presets.enabled_scenario_modules(cfg)
    assert "sawtooth" not in modules
    assert "current_drive" not in modules
