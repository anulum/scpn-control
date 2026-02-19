# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Configuration
# ──────────────────────────────────────────────────────────────────────
"""Shared fixtures for scpn-control test suite."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG seeded at 42."""
    return np.random.default_rng(42)


@pytest.fixture
def minimal_petri_net():
    """A minimal 3-place, 2-transition Petri net for smoke tests."""
    from scpn_control.scpn.structure import StochasticPetriNet

    net = StochasticPetriNet()
    net.add_place("p0", tokens=1.0)
    net.add_place("p1", tokens=0.0)
    net.add_place("p2", tokens=0.0)
    net.add_transition("t0", rate=1.0)
    net.add_transition("t1", rate=1.0)
    net.add_arc("p0", "t0", weight=1.0)
    net.add_arc("t0", "p1", weight=1.0)
    net.add_arc("p1", "t1", weight=1.0)
    net.add_arc("t1", "p2", weight=1.0)
    return net


@pytest.fixture
def reference_data_dir():
    """Path to validation reference data."""
    from pathlib import Path
    return Path(__file__).parent.parent / "validation" / "reference_data"


@pytest.fixture
def weights_dir():
    """Path to pretrained weights."""
    from pathlib import Path
    return Path(__file__).parent.parent / "weights"
