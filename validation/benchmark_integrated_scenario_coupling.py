# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated scenario coupling benchmark
"""Publish deterministic bounded integrated-scenario coupling evidence."""

from __future__ import annotations

import json
from pathlib import Path

from scpn_control.core.integrated_scenario import (
    IntegratedScenarioSimulator,
    ScenarioConfig,
    audit_scenario_coupling,
    save_scenario_coupling_report,
    scenario_coupling_audit_to_dict,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "integrated_scenario_coupling.json"
MD_REPORT = REPORT_DIR / "integrated_scenario_coupling.md"


def benchmark_config() -> ScenarioConfig:
    """Return a deterministic compact scenario exercising all control exchanges."""

    return ScenarioConfig(
        R0=0.93,
        a=0.58,
        B0=1.0,
        kappa=2.0,
        delta=0.4,
        Ip_MA=1.0,
        P_aux_MW=1.5,
        P_eccd_MW=0.25,
        rho_eccd=0.45,
        P_nbi_MW=0.2,
        E_nbi_keV=80.0,
        t_start=0.0,
        t_end=0.03,
        dt=0.01,
        include_sawteeth=False,
        include_ntm=True,
        include_sol=True,
        include_elm=False,
        include_stability=True,
        include_phase_bridge=False,
    )


def main() -> None:
    """Run deterministic scenario replay audit and write reports."""

    cfg = benchmark_config()
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    states = sim.run()
    audit = audit_scenario_coupling(states, cfg, scenario_name="compact_control_coupling_benchmark")
    payload = scenario_coupling_audit_to_dict(audit)
    payload["claim_boundary"] = (
        "Bounded deterministic replay audit only. Measured-discharge and external "
        "integrated-modelling validation remain required before predictive scenario claims."
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_scenario_coupling_report(audit, JSON_REPORT)
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Integrated scenario coupling benchmark report -->",
                "",
                "# Integrated Scenario Coupling Benchmark",
                "",
                "This report is bounded deterministic replay evidence for the",
                "`IntegratedScenarioSimulator` coupling contract. It is not measured",
                "discharge validation and does not enable predictive full-scenario claims.",
                "",
                f"- Passed: `{audit.passed}`",
                f"- Scenario: `{audit.metadata.scenario_name}`",
                f"- Steps: `{audit.metadata.n_steps}`",
                f"- Enabled modules: `{', '.join(audit.metadata.enabled_modules)}`",
                f"- Exchange records: `{audit.metadata.exchange_count}`",
                f"- Max current deviation [MA]: `{audit.metadata.max_abs_current_deviation_MA:.6e}`",
                f"- Max relative thermal-energy step: `{audit.metadata.max_relative_thermal_energy_step:.6e}`",
                f"- Claim boundary: `{payload['claim_boundary']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not audit.passed:
        raise SystemExit("; ".join(audit.violations))


if __name__ == "__main__":
    main()
