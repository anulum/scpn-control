# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Compiler sc_neurocore Inference Path Tests
"""Real sc_neurocore-backed coverage for CompiledNet.dense_forward.

Exercises the v3.16 ``sc_forward`` fast path, the legacy bit-ops path (forced by
disabling the v3 flag so the genuine ``vec_and``/``vec_popcount`` loop runs), and
the LIF-neuron firing path built by the compiler. All tests require sc_neurocore;
they skip where it is absent (e.g. CI without the dependency) but run wherever it
is installed, so the accelerated paths are covered by the real dependency rather
than a stand-in.
"""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.scpn.compiler as compiler_module
from scpn_control.scpn.compiler import (
    _HAS_NEUROCORE_V3,
    _HAS_SC_NEUROCORE,
    CompiledNet,
    FusionCompiler,
)
from scpn_control.scpn.structure import StochasticPetriNet


def _compiled(bitstream_length: int = 4096) -> CompiledNet:
    net = StochasticPetriNet()
    net.add_place("Red", initial_tokens=1.0)
    net.add_place("Green", initial_tokens=0.0)
    net.add_place("Yellow", initial_tokens=0.0)
    net.add_transition("T_r2g", threshold=0.5)
    net.add_transition("T_g2y", threshold=0.5)
    net.add_transition("T_y2r", threshold=0.5)
    net.add_arc("Red", "T_r2g", weight=1.0)
    net.add_arc("Green", "T_g2y", weight=1.0)
    net.add_arc("Yellow", "T_y2r", weight=1.0)
    net.add_arc("T_r2g", "Green", weight=1.0)
    net.add_arc("T_g2y", "Yellow", weight=1.0)
    net.add_arc("T_y2r", "Red", weight=1.0)
    net.compile()
    return FusionCompiler(bitstream_length=bitstream_length, seed=42).compile(net)


@pytest.mark.skipif(not _HAS_NEUROCORE_V3, reason="sc_neurocore v3.16 sc_forward not installed")
class TestScForwardPath:
    def test_dense_forward_matches_float_reference(self) -> None:
        compiled = _compiled()
        assert compiled.W_in_packed is not None
        marking = np.array([0.8, 0.3, 0.0])
        stochastic = compiled.dense_forward(compiled.W_in_packed, marking)
        reference = compiled.dense_forward_float(compiled.W_in, marking)
        np.testing.assert_allclose(stochastic, reference, atol=0.1)


@pytest.mark.skipif(not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed")
class TestLegacyBitOpsPath:
    def test_legacy_loop_matches_float_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the pre-v3.16 path so the genuine vec_and/vec_popcount loop runs.
        monkeypatch.setattr(compiler_module, "_HAS_NEUROCORE_V3", False)
        compiled = _compiled()
        assert compiled.W_in_packed is not None
        marking = np.array([0.8, 0.3, 0.0])
        stochastic = compiled.dense_forward(compiled.W_in_packed, marking)
        reference = compiled.dense_forward_float(compiled.W_in, marking)
        np.testing.assert_allclose(stochastic, reference, atol=0.1)

    def test_legacy_loop_without_vectorised_popcount(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Drop numpy.bitwise_count to drive the per-element vec_popcount fallback
        # (the path taken on numpy < 2.0).
        monkeypatch.setattr(compiler_module, "_HAS_NEUROCORE_V3", False)
        monkeypatch.delattr(np, "bitwise_count", raising=False)
        compiled = _compiled()
        assert compiled.W_in_packed is not None
        marking = np.array([1.0, 0.0, 0.0])
        stochastic = compiled.dense_forward(compiled.W_in_packed, marking)
        reference = compiled.dense_forward_float(compiled.W_in, marking)
        np.testing.assert_allclose(stochastic, reference, atol=0.1)


@pytest.mark.skipif(not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed")
class TestNeuronFiringPath:
    def test_binary_lif_fire_uses_compiled_neurons(self) -> None:
        compiled = _compiled(bitstream_length=256)
        assert compiled.neurons  # compiler built one StochasticLIFNeuron per transition
        currents = np.array([1.0, 0.0, 0.0])
        fired = compiled.lif_fire(currents)
        assert fired[0] == 1.0
        assert fired[1] == 0.0
        assert fired[2] == 0.0
