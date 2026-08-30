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


def test_phase_materials_do_not_claim_reactor_control_closure() -> None:
    """Phase examples state their model boundary without plant-control claims."""
    paths = (
        "docs/REVIEWER_PAPER27_INTEGRATION.md",
        "docs/REVIEWER_PAPER27_INTEGRATION.tex",
        "docs/paper27_phase_dynamics.md",
        "examples/paper27_phase_dynamics_demo.ipynb",
        "examples/tutorial_05_adaptive_phase_dynamics.py",
    )
    forbidden = tuple(
        "".join(parts)
        for parts in (
            ("authority over ", "plasma modes"),
            ("entry point a real ", "control loop would call"),
            ("maps directly to SNN or ", "PID output amplitude"),
            ("complete real-time ", "monitoring loop"),
        )
    )
    for path in paths:
        text = _read(path)
        normalised = " ".join(text.split())
        assert "not a reactor feedback loop" in normalised
        for claim in forbidden:
            assert claim not in text


def test_phase_runtime_surfaces_state_their_non_reactor_boundary() -> None:
    """Runtime-facing phase APIs cannot imply an identified plant loop."""
    required_markers = {
        "README.md": "not a reactor feedback loop",
        "docs/api.md": "do not identify oscillator states from reactor observations",
        "docs/architecture.md": "do not currently identify oscillators from reactor signals",
        "src/scpn_control/phase/adaptive_knm.py": "not facility-calibrated stability laws or a reactor feedback loop",
        "src/scpn_control/phase/realtime_monitor.py": "It is not a reactor feedback loop",
        "src/scpn_control/phase/ws_phase_stream.py": "does not ingest a reactor plant state",
    }

    for path, marker in required_markers.items():
        assert marker in " ".join(_read(path).split())


def test_public_materials_separate_abstract_and_plasma_labelled_layers() -> None:
    """Paper-27 indices cannot be presented as the plasma-labelled ontology."""
    notebook = " ".join(_read("examples/scpn_full_stack_demo_2026.ipynb").split())
    api = " ".join(_read("docs/api.md").split())
    architecture = " ".join(_read("docs/architecture.md").split())

    assert "16 plasma layers defined in Paper 27" not in notebook
    assert "Paper 27 Hierarchy" not in notebook
    assert "distinct from Paper 27's abstract layer indices" in notebook
    assert "separate ontology from the abstract Paper-27 Knm construction" in api
    assert "L=1..8" in api
    assert "L=16" in api
    assert "scpn-phase-orchestrator" in architecture
