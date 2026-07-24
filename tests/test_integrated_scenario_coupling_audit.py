# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for scenario coupling audit leaf

"""Drive production coupling-audit leaf contracts on real scenario trajectories."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scpn_control.core.integrated_scenario as owner
import scpn_control.core.integrated_scenario_coupling_audit as leaf
from scpn_control.core.integrated_scenario import IntegratedScenarioSimulator
from scpn_control.core.integrated_scenario_presets import (
    ScenarioConfig,
    enabled_scenario_modules,
    scenario_config_sha256,
)


def _minimal_config(**overrides: object) -> ScenarioConfig:
    """Short closed-loop-free scenario config used for coupling-audit contracts."""
    base = ScenarioConfig(
        R0=0.93,
        a=0.58,
        B0=1.0,
        kappa=2.0,
        delta=0.4,
        Ip_MA=1.0,
        P_aux_MW=0.5,
        P_eccd_MW=0.0,
        P_nbi_MW=0.0,
        t_start=0.0,
        t_end=0.05,
        dt=0.01,
        include_sawteeth=False,
        include_ntm=False,
        include_sol=False,
        include_elm=False,
        include_stability=False,
        include_phase_bridge=False,
    )
    if not overrides:
        return base
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_public_owner_symbols_bind_to_leaf() -> None:
    """Facade re-exports are the production coupling-audit leaf objects."""
    assert owner.ScenarioModuleExchange is leaf.ScenarioModuleExchange
    assert owner.ScenarioCouplingMetadata is leaf.ScenarioCouplingMetadata
    assert owner.ScenarioCouplingAudit is leaf.ScenarioCouplingAudit
    assert owner.audit_scenario_coupling is leaf.audit_scenario_coupling
    assert owner.scenario_coupling_audit_to_dict is leaf.scenario_coupling_audit_to_dict
    assert owner.save_scenario_coupling_report is leaf.save_scenario_coupling_report


def test_audit_records_module_exchange_and_bounds_via_leaf() -> None:
    """Leaf audit preserves finite module exchange and conservation bounds."""
    cfg = _minimal_config(P_aux_MW=2.0, P_eccd_MW=0.5, include_stability=True)
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    states = sim.run()
    audit = leaf.audit_scenario_coupling(states, cfg, scenario_name="minimal_control_replay")
    assert audit.passed is True
    assert audit.violations == ()
    assert audit.metadata.config_sha256 == scenario_config_sha256(cfg)
    assert audit.metadata.strictly_monotonic_time is True
    assert audit.metadata.all_profiles_finite is True
    assert audit.metadata.max_abs_current_deviation_MA <= cfg.Ip_MA * 1e-9
    modules = {exchange.module for exchange in audit.module_exchanges}
    assert {"transport", "current_diffusion", "bootstrap_current", "mhd_stability"} <= modules


def test_audit_rejects_empty_states() -> None:
    """Empty trajectories fail closed with an explicit violation."""
    cfg = _minimal_config()
    audit = leaf.audit_scenario_coupling([], cfg, scenario_name="empty")
    assert audit.passed is False
    assert audit.violations == ("states must not be empty",)
    assert audit.metadata.n_steps == 0
    assert audit.metadata.exchange_count == 0


def test_audit_rejects_nonmonotonic_and_timestep_drift() -> None:
    """Replay order and dt consistency fail closed."""
    cfg = _minimal_config(P_aux_MW=0.5)
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    states = sim.run()
    broken = states[:-1] + [replace(states[-1], time=states[0].time)]
    audit = leaf.audit_scenario_coupling(broken, cfg)
    assert audit.passed is False
    assert "scenario replay time must be strictly monotonic" in audit.violations


def test_audit_rejects_nonfinite_profiles_and_positive_domain_violations() -> None:
    """Non-finite profiles and non-positive Te/q/W fail closed."""
    cfg = _minimal_config(P_aux_MW=0.5)
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    state = sim.run()[-1]
    poisoned_te = np.asarray(state.Te, dtype=float).copy()
    poisoned_te[3] = np.nan
    audit = leaf.audit_scenario_coupling([replace(state, Te=poisoned_te)], cfg)
    assert audit.passed is False
    assert audit.metadata.all_profiles_finite is False

    audit_w = leaf.audit_scenario_coupling([replace(state, W_thermal=-1.0)], cfg)
    assert any("thermal energy" in v for v in audit_w.violations)

    bad_q = np.asarray(state.q, dtype=float).copy()
    bad_q[0] = -0.1
    audit_q = leaf.audit_scenario_coupling([replace(state, q=bad_q)], cfg)
    assert any("safety factor" in v for v in audit_q.violations)

    bad_te = np.asarray(state.Te, dtype=float).copy()
    bad_te[0] = -0.01
    audit_te = leaf.audit_scenario_coupling([replace(state, Te=bad_te)], cfg)
    assert any("temperature and density" in v for v in audit_te.violations)


def test_audit_rejects_current_and_energy_step_bounds() -> None:
    """Ip and thermal-energy step bounds fail closed on hostile replays."""
    cfg = _minimal_config(P_aux_MW=0.5)
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    state = sim.run()[-1]
    deviated = replace(state, Ip_MA=float(cfg.Ip_MA) + 1.0)
    audit = leaf.audit_scenario_coupling([deviated], cfg)
    assert any("current deviates" in v for v in audit.violations)

    low = replace(state, W_thermal=1.0, time=0.0)
    high = replace(state, W_thermal=100.0, time=float(cfg.dt))
    audit_e = leaf.audit_scenario_coupling([low, high], cfg, energy_relative_step_limit=0.1)
    assert any("thermal-energy step" in v for v in audit_e.violations)


def test_exchange_for_module_covers_optional_lanes() -> None:
    """Each enabled optional module produces a structured exchange record."""
    cfg = _minimal_config(
        P_aux_MW=1.0,
        P_eccd_MW=0.5,
        P_nbi_MW=0.5,
        include_sawteeth=True,
        include_ntm=True,
        include_sol=True,
        include_elm=True,
        include_stability=True,
        include_phase_bridge=True,
    )
    modules = enabled_scenario_modules(cfg)
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    state = sim.run()[-1]
    for module in modules:
        exchange = leaf._exchange_for_module(module, state, cfg)
        assert exchange.module == module
        assert exchange.inputs["dt_s"] == pytest.approx(float(cfg.dt))
        assert exchange.outputs
        assert exchange.diagnostics
    # Unknown module name still yields a structured fallback record.
    unknown = leaf._exchange_for_module("not_a_real_module", state, cfg)
    assert unknown.module == "not_a_real_module"
    assert unknown.diagnostics["module"] == "not_a_real_module"


def test_save_report_and_dict_serialisation(tmp_path: Path) -> None:
    """JSON report persistence keeps hash, exchanges, and claim boundary."""
    cfg = _minimal_config(P_aux_MW=1.0)
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    states = sim.run()
    audit = leaf.audit_scenario_coupling(states, cfg, scenario_name="persisted_contract")
    out = tmp_path / "nested" / "scenario_coupling.json"
    leaf.save_scenario_coupling_report(audit, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    payload = leaf.scenario_coupling_audit_to_dict(audit)
    # JSON round-trip normalises tuples to lists; compare on serialised form.
    assert data == json.loads(json.dumps(payload, sort_keys=True))
    assert data["passed"] is True
    assert len(data["metadata"]["config_sha256"]) == 64
    assert data["metadata"]["scenario_name"] == "persisted_contract"
    assert data["metadata"]["claim_status"].startswith("bounded replay audit only")
    assert len(data["module_exchanges"]) == data["metadata"]["exchange_count"]


def test_relative_energy_steps_empty_and_single() -> None:
    """Energy-step helper returns empty for single-sample trajectories."""
    cfg = _minimal_config()
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    state = sim.run()[-1]
    steps = leaf._relative_energy_steps([state])
    assert steps.shape == (0,)
    assert leaf._finite_profile_set(state) is True
