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


# ── Shared controller fixtures ────────────────────────────────────────


@pytest.fixture
def petri_net_std():
    """Standard 8-place, 4-transition Petri net used by controller tests."""
    from scpn_control.scpn.structure import StochasticPetriNet

    net = StochasticPetriNet()
    for name in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        net.add_place(name)
    net.add_transition("T0", threshold=0.5)
    net.add_transition("T1", threshold=0.5)
    net.add_transition("T2", threshold=0.5)
    net.add_transition("T3", threshold=0.5)
    net.add_arc("P0", "T0", 0.8)
    net.add_arc("P1", "T0", 0.6)
    net.add_arc("T0", "P4", 0.7)
    net.add_arc("P2", "T1", 0.8)
    net.add_arc("P3", "T1", 0.6)
    net.add_arc("T1", "P5", 0.7)
    net.add_arc("P4", "T2", 0.5)
    net.add_arc("T2", "P6", 0.9)
    net.add_arc("P5", "T3", 0.5)
    net.add_arc("T3", "P7", 0.9)
    return net


@pytest.fixture
def controller_artifact(tmp_path, petri_net_std):
    """Compile petri_net_std and save artifact, return path string."""
    from scpn_control.scpn.artifact import save_artifact
    from scpn_control.scpn.compiler import FusionCompiler

    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(petri_net_std)
    readout_config = {
        "actions": [
            {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
            {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
        ],
        "gains": [1000.0, 1000.0],
        "abs_max": [5000.0, 5000.0],
        "slew_per_s": [1e6, 1e6],
    }
    injection_config = [
        {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    ]
    art = compiled.export_artifact(
        name="test-shared",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "shared.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


@pytest.fixture
def controller_instance(controller_artifact):
    """Ready-to-use NeuroSymbolicController from shared artifact."""
    from scpn_control.scpn.artifact import load_artifact
    from scpn_control.scpn.controller import (
        ControlScales,
        ControlTargets,
        NeuroSymbolicController,
    )

    art = load_artifact(controller_artifact)
    return NeuroSymbolicController(
        artifact=art,
        seed_base=42,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        sc_n_passes=1,
        sc_bitflip_rate=0.0,
        runtime_backend="numpy",
    )
