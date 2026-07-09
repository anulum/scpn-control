# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI closed-loop demo tests

"""CLI tests for the bounded integrated-scenario closed-loop demo surface."""

from __future__ import annotations

import json
from typing import Any, cast

from click.testing import CliRunner

from scpn_control.cli import main


def test_combined_demo_json_includes_integrated_scenario_closed_loop() -> None:
    """Combined demo JSON exposes controller-to-plant closed-loop evidence."""
    result = CliRunner().invoke(main, ["demo", "--scenario", "combined", "--steps", "3", "--json-out"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    closed_loop = cast(dict[str, Any], data["integrated_scenario_closed_loop"])
    assert closed_loop["schema_version"] == "closed-loop-integrated-scenario-v1"
    assert closed_loop["coupling_audit"]["passed"] is True
    assert len(closed_loop["steps"]) == 3
    assert closed_loop["steps"][0]["applied_p_aux_mw"] >= 0.0


def test_combined_demo_text_reports_integrated_scenario_audit() -> None:
    """Combined demo text names the bounded integrated-scenario audit."""
    result = CliRunner().invoke(main, ["demo", "--scenario", "combined", "--steps", "2"])

    assert result.exit_code == 0, result.output
    assert "Integrated scenario closed loop: steps=2 audit_passed=True" in result.output
