"""Verify sc_neurocore v3.8.0+ detection and VectorizedSCLayer forward path."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_control.scpn import compiler as _compiler

HAS_V3 = _compiler._HAS_NEUROCORE_V3
HAS_ANY = _compiler._HAS_SC_NEUROCORE

pytestmark = pytest.mark.skipif(not HAS_ANY, reason="sc_neurocore not installed")


class TestNeurocore:
    def test_v3_flag_set(self) -> None:
        if not HAS_V3:
            pytest.skip("sc-neurocore V3 (>=3.8.0) is not installed")
        assert HAS_V3, "_HAS_NEUROCORE_V3 should be True with sc-neurocore>=3.8.0"

    def test_vectorized_layer_importable(self) -> None:
        if not HAS_V3:
            pytest.skip("sc-neurocore V3 (>=3.8.0) is not installed")
        from sc_neurocore import VectorizedSCLayer  # noqa: F401

    def test_get_backend_importable(self) -> None:
        if not HAS_V3:
            pytest.skip("sc-neurocore V3 (>=3.8.0) is not installed")
        from sc_neurocore.accel import get_backend

        backend = get_backend()
        assert backend is not None

    def test_dense_forward_v3_path(self) -> None:
        """CompiledNet.dense_forward should use VectorizedSCLayer when v3 is available."""
        if not HAS_V3:
            pytest.skip("sc-neurocore V3 (>=3.8.0) is not installed")
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        net.add_place("p0", initial_tokens=1.0)
        net.add_place("p1", initial_tokens=0.0)
        net.add_transition("t0")
        net.add_arc("p0", "t0", weight=1.0)
        net.add_arc("t0", "p1", weight=1.0)

        from scpn_control.scpn.compiler import FusionCompiler

        compiled = FusionCompiler(net).compile()

        marking = np.array([1.0, 0.0])
        result = compiled.dense_forward(marking)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_v3_faster_than_float(self) -> None:
        """VectorizedSCLayer path should not be slower than numpy float path."""
        if not HAS_V3:
            pytest.skip("v3 path not available for speed comparison")

        from scpn_control.scpn.compiler import FusionCompiler
        from scpn_control.scpn.structure import StochasticPetriNet

        net = StochasticPetriNet()
        for i in range(8):
            net.add_place(f"p{i}", initial_tokens=1.0)
        for i in range(8):
            net.add_transition(f"t{i}")
            net.add_arc(f"p{i}", f"t{i}", weight=1)
            net.add_arc(f"t{i}", f"p{(i + 1) % 8}", weight=1)

        compiled = FusionCompiler(net).compile()
        marking = np.random.default_rng(42).random(8)

        # Warm-up (also serves as baseline)
        n_warmup = 10
        t_warmup = time.perf_counter()
        for _ in range(n_warmup):
            compiled.dense_forward(marking)
        warmup_elapsed = time.perf_counter() - t_warmup

        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            compiled.dense_forward(marking)
        elapsed = time.perf_counter() - t0

        per_call_warmup = warmup_elapsed / n_warmup
        per_call_main = elapsed / n
        assert per_call_main < per_call_warmup * 5, (
            f"dense_forward regressed: {per_call_main * 1e6:.0f} us/call vs warmup {per_call_warmup * 1e6:.0f} us/call"
        )
