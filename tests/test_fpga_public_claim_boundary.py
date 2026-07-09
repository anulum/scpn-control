# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FPGA public claim boundary tests

"""Regression tests for public FPGA HDL export claim boundaries."""

from __future__ import annotations

from pathlib import Path


def test_public_surface_does_not_claim_fpga_bitstream_export() -> None:
    """Public docs must not claim that HDL export emits FPGA bitstreams."""
    repo_root = Path(__file__).resolve().parents[1]
    checked_paths = (
        repo_root / "paper.md",
        repo_root / "docs" / "joss_paper.md",
        repo_root / "src" / "scpn_control" / "scpn" / "fpga_export.py",
    )

    for path in checked_paths:
        assert "FPGA bitstream export" not in path.read_text(encoding="utf-8")
