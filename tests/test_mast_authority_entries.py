# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Script entries of the MAST signal-authority commands.
"""Each MAST signal-authority module hands its command line to ``main`` when run as a script.

The help run reaches the ``__main__`` line without reading a source
manifest or writing an authority report.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

AUTHORITIES = (
    "mast_dbdt_authority",
    "mast_locked_mode_authority",
    "mast_normalised_beta_authority",
    "mast_saddle_modal_authority",
    "mast_toroidal_field_authority",
)


@pytest.mark.parametrize("name", AUTHORITIES)
def test_script_entry_answers_help_without_reading_sources(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script with ``--help``, the authority prints its usage and exits zero."""
    script = ROOT / "validation" / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
