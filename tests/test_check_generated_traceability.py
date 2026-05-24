# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Generated traceability freshness check tests

from __future__ import annotations

from tools.check_generated_traceability import main


def test_generated_traceability_check_passes_for_repository_state(capsys) -> None:
    assert main() == 0
    assert "Generated traceability documentation is current." in capsys.readouterr().out
