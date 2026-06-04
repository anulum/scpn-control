# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware Campaign CLI Tests

from __future__ import annotations

from click.testing import CliRunner

from scpn_control.cli import main


def test_run_hardware_campaign_help_exposes_execution_backend() -> None:
    result = CliRunner().invoke(main, ["run-hardware-campaign", "--help"])

    assert result.exit_code == 0
    assert "--execution-backend" in result.output
