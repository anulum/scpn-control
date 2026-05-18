# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Project Metadata Tests

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_facility_optional_extra_declares_mdsplus_thin_client() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    extras = pyproject["project"]["optional-dependencies"]
    assert "facility" in extras
    assert "mdsthin>=1.6.3" in extras["facility"]
    assert "mdsthin>=1.6.3" in extras["all"]
