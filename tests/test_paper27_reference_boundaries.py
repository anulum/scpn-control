# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Paper 27 public-reference boundary tests.
"""Regression tests for public Paper 27 reference wording."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER27_URL = "https://www.academia.edu/143833534/27_SCPN_The_Knm_Matrix"
KURAMOTO_ARXIV_URL = "https://arxiv.org/abs/2004.06344"


def _read(relative_path: str) -> str:
    """Return a repository text file as UTF-8."""

    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_pitch_separates_paper27_from_kuramoto_arxiv_reference() -> None:
    """The pitch must not attribute the Kuramoto reference as Paper 27."""

    pitch = _read("docs/pitch.md")

    assert f"[Paper 27 manuscript]({PAPER27_URL})" in pitch
    assert f"[arXiv:2004.06344]({KURAMOTO_ARXIV_URL})" in pitch
    assert (
        'Paper 27:** "The Knm Matrix" — 16-layer Kuramoto-Sakaguchi phase dynamics\n  with exogenous global field driver. [arXiv:2004.06344]'
        not in pitch
    )


def test_readme_publication_caveat_keeps_references_distinct() -> None:
    """The README caveat names the arXiv paper as a related reference."""

    readme = _read("README.md")

    assert "Kuramoto-Sakaguchi reference" in readme
    assert "Paper 27\n  (arXiv:2004.06344)" not in readme
    assert "arXiv:2004.06344 is a related" in readme


def test_reviewer_handoff_uses_paper27_source_url() -> None:
    """Reviewer handoff documents cite the real Paper 27 manuscript URL."""

    handoff = _read("docs/REVIEWER_PAPER27_INTEGRATION.md")
    handoff_tex = _read("docs/REVIEWER_PAPER27_INTEGRATION.tex")

    assert PAPER27_URL in handoff
    assert PAPER27_URL in handoff_tex
    assert "Related Kuramoto" in handoff
    assert "Related Kuramoto--Sakaguchi reference" in handoff_tex
