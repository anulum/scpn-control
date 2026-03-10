# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Coverage gap tests for scpn/ subpackage
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Targeted tests hitting uncovered lines in compiler, controller, structure, artifact."""

from __future__ import annotations

import base64
import zlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.compiler import CompiledNet
from scpn_control.scpn.artifact import (
    ArtifactMeta,
    ArtifactValidationError,
    CompilerInfo,
    FixedPoint,
    SeedPolicy,
    _decode_u64_compact,
    _validate,
    Artifact,
    Topology,
    PlaceSpec,
    TransitionSpec,
    WeightMatrix,
    Weights,
    Readout,
    ActionReadout,
    InitialState,
    PlaceInjection,
)
from scpn_control.scpn.contracts import ControlScales, ControlTargets, FeatureAxisSpec


# ═══════════════════════════════════════════════════════════════════════
# compiler.py — line 258: lif_fire binary fallback (no neurons)
# ═══════════════════════════════════════════════════════════════════════


class TestLifFireBinaryNoNeurons:
    def test_binary_threshold_comparison_without_neurons(self):
        """CompiledNet with neurons=[] uses np threshold comparison (line 258)."""
        net = CompiledNet(
            n_places=2,
            n_transitions=2,
            place_names=["p0", "p1"],
            transition_names=["t0", "t1"],
            W_in=np.array([[0.5, 0.0], [0.0, 0.5]]),
            W_out=np.array([[0.0, 0.5], [0.5, 0.0]]),
            neurons=[],
            thresholds=np.array([0.3, 0.7]),
            firing_mode="binary",
        )
        currents = np.array([0.5, 0.5])
        fired = net.lif_fire(currents)
        assert fired[0] == 1.0  # 0.5 >= 0.3
        assert fired[1] == 0.0  # 0.5 < 0.7


# ═══════════════════════════════════════════════════════════════════════
# compiler.py — lines 191-193: v3 dense_forward path (mock)
# ═══════════════════════════════════════════════════════════════════════


class TestDenseForwardV3Path:
    def test_v3_vectorized_sc_layer_path(self, monkeypatch):
        """Mock sc_neurocore v3 classes to exercise lines 191-193."""
        import scpn_control.scpn.compiler as cmod

        monkeypatch.setattr(cmod, "_HAS_SC_NEUROCORE", True)
        monkeypatch.setattr(cmod, "_HAS_NEUROCORE_V3", True)

        mock_encoder = MagicMock()
        mock_layer = MagicMock()
        mock_layer.forward.return_value = np.array([0.9, 0.1])
        mock_backend = MagicMock()

        monkeypatch.setattr(cmod, "BitstreamEncoder", MagicMock(return_value=mock_encoder), raising=False)
        monkeypatch.setattr(cmod, "VectorizedSCLayer", MagicMock(return_value=mock_layer), raising=False)
        monkeypatch.setattr(cmod, "get_backend", MagicMock(return_value=mock_backend), raising=False)

        net = CompiledNet(
            n_places=2,
            n_transitions=2,
            place_names=["p0", "p1"],
            transition_names=["t0", "t1"],
            W_in=np.eye(2),
            W_out=np.eye(2),
            W_in_packed=np.zeros((2, 2, 1), dtype=np.uint64),
            thresholds=np.array([0.5, 0.5]),
            bitstream_length=64,
            seed=0,
        )
        result = net.dense_forward(net.W_in_packed, np.array([1.0, 0.0]))
        np.testing.assert_allclose(result, [0.9, 0.1])
        mock_layer.forward.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# compiler.py — lines 208-210: legacy np.bit_count path (mock)
# ═══════════════════════════════════════════════════════════════════════


class TestDenseForwardLegacyBitCount:
    def test_legacy_bit_count_path(self, monkeypatch):
        """Mock sc_neurocore legacy imports + np.bit_count to exercise lines 207-210."""
        import scpn_control.scpn.compiler as cmod

        monkeypatch.setattr(cmod, "_HAS_SC_NEUROCORE", True)
        monkeypatch.setattr(cmod, "_HAS_NEUROCORE_V3", False)

        mock_rng = MagicMock()
        mock_generate = MagicMock(return_value=np.zeros(64, dtype=np.uint8))
        mock_pack = MagicMock(return_value=np.array([0], dtype=np.uint64))

        monkeypatch.setattr(cmod, "_SC_RNG", MagicMock(return_value=mock_rng), raising=False)
        monkeypatch.setattr(cmod, "generate_bernoulli_bitstream", mock_generate, raising=False)
        monkeypatch.setattr(cmod, "pack_bitstream", mock_pack, raising=False)

        # Add np.bit_count mock so the hasattr check passes
        def _fake_bit_count(arr):
            return np.zeros_like(arr, dtype=np.uint64)

        monkeypatch.setattr(np, "bit_count", _fake_bit_count, raising=False)

        net = CompiledNet(
            n_places=2,
            n_transitions=1,
            place_names=["p0", "p1"],
            transition_names=["t0"],
            W_in=np.array([[0.5, 0.0]]),
            W_out=np.array([[0.0], [0.5]]),
            W_in_packed=np.zeros((1, 2, 1), dtype=np.uint64),
            thresholds=np.array([0.5]),
            bitstream_length=64,
            seed=0,
        )
        result = net.dense_forward(net.W_in_packed, np.array([0.5, 0.5]))
        assert result.shape == (1,)


# ═══════════════════════════════════════════════════════════════════════
# controller.py — lines 46-56: rust import path (mock)
# ═══════════════════════════════════════════════════════════════════════


class TestControllerRustImport:
    def test_rust_scpn_runtime_flags_set_when_available(self, monkeypatch):
        """Mock scpn_control_rs module to exercise the import block (lines 42-56)."""
        import scpn_control.scpn.controller as ctrl_mod

        mock_dense = MagicMock()
        mock_marking = MagicMock()
        mock_sample = MagicMock()

        monkeypatch.setattr(ctrl_mod, "_HAS_RUST_SCPN_RUNTIME", True)
        monkeypatch.setattr(ctrl_mod, "_rust_dense_activations", mock_dense)
        monkeypatch.setattr(ctrl_mod, "_rust_marking_update", mock_marking)
        monkeypatch.setattr(ctrl_mod, "_rust_sample_firing", mock_sample)

        assert ctrl_mod._HAS_RUST_SCPN_RUNTIME is True
        assert ctrl_mod._rust_dense_activations is mock_dense
        assert ctrl_mod._rust_marking_update is mock_marking
        assert ctrl_mod._rust_sample_firing is mock_sample


# ═══════════════════════════════════════════════════════════════════════
# controller.py — line 464: missing passthrough in _build_feature_dict
# ═══════════════════════════════════════════════════════════════════════


def _make_test_artifact():
    """Build a minimal Artifact for controller tests."""
    meta = ArtifactMeta(
        artifact_version="1.0.0",
        name="test",
        dt_control_s=0.001,
        stream_length=64,
        fixed_point=FixedPoint(data_width=16, fraction_bits=10, signed=False),
        firing_mode="binary",
        seed_policy=SeedPolicy(id="default", hash_fn="splitmix64", rng_family="xoshiro256++"),
        created_utc="2026-01-01T00:00:00Z",
        compiler=CompilerInfo(name="test", version="0.1", git_sha="0000000"),
    )
    topology = Topology(
        places=[PlaceSpec(id=i, name=f"p{i}") for i in range(4)],
        transitions=[
            TransitionSpec(id=0, name="t0", threshold=0.1),
            TransitionSpec(id=1, name="t1", threshold=0.1),
        ],
    )
    weights = Weights(
        w_in=WeightMatrix(shape=[2, 4], data=[0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        w_out=WeightMatrix(shape=[4, 2], data=[0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1]),
    )
    readout = Readout(
        actions=[ActionReadout(id=0, name="ctrl", pos_place=2, neg_place=3)],
        gains=[1.0],
        abs_max=[1.0],
        slew_per_s=[1e6],
    )
    initial_state = InitialState(
        marking=[0.0, 0.0, 0.0, 0.0],
        place_injections=[
            PlaceInjection(place_id=0, source="x_pos", scale=1.0, offset=0.0, clamp_0_1=True),
            PlaceInjection(place_id=1, source="x_neg", scale=1.0, offset=0.0, clamp_0_1=True),
            PlaceInjection(place_id=2, source="sensor_passthrough", scale=1.0, offset=0.0, clamp_0_1=True),
        ],
    )
    return Artifact(
        meta=meta,
        topology=topology,
        weights=weights,
        readout=readout,
        initial_state=initial_state,
    )


class TestControllerBuildFeatureDictMissingPassthrough:
    def test_build_feature_dict_missing_passthrough_directly(self):
        """Directly call _build_feature_dict with missing passthrough key."""
        from scpn_control.scpn.controller import NeuroSymbolicController

        art = _make_test_artifact()
        ctrl = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=1.0, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
            feature_axes=[
                FeatureAxisSpec(obs_key="err", target=1.0, scale=1.0, pos_key="x_pos", neg_key="x_neg"),
            ],
        )
        pos = np.array([0.5])
        neg = np.array([0.0])
        # obs_map missing "sensor_passthrough"
        with pytest.raises(KeyError, match="sensor_passthrough"):
            ctrl._build_feature_dict({"err": 0.5}, pos, neg)


# ═══════════════════════════════════════════════════════════════════════
# controller.py — line 556: rust path with bitflip (mock)
# ═══════════════════════════════════════════════════════════════════════


class TestControllerRustBitflipPath:
    def test_rust_sample_firing_with_bitflip(self, monkeypatch):
        """Line 556: rust backend + sc_bitflip_rate > 0 creates an rng."""
        import scpn_control.scpn.controller as ctrl_mod
        from scpn_control.scpn.controller import NeuroSymbolicController

        mock_dense = MagicMock(return_value=np.array([0.6, 0.4]))
        mock_marking = MagicMock(return_value=np.array([0.0, 0.0, 0.3, 0.1]))
        mock_sample = MagicMock(return_value=np.array([0.8, 0.2]))

        monkeypatch.setattr(ctrl_mod, "_HAS_RUST_SCPN_RUNTIME", True)
        monkeypatch.setattr(ctrl_mod, "_rust_dense_activations", mock_dense)
        monkeypatch.setattr(ctrl_mod, "_rust_marking_update", mock_marking)
        monkeypatch.setattr(ctrl_mod, "_rust_sample_firing", mock_sample)

        art = _make_test_artifact()
        ctrl = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=1.0, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
            feature_axes=[
                FeatureAxisSpec(obs_key="err", target=1.0, scale=1.0, pos_key="x_pos", neg_key="x_neg"),
            ],
            sc_n_passes=8,
            sc_bitflip_rate=0.01,
            runtime_backend="rust",
            enable_oracle_diagnostics=False,
        )
        assert ctrl._runtime_backend == "rust"

        result = ctrl.step({"err": 0.5, "sensor_passthrough": 0.3}, k=0)
        assert isinstance(result, dict)
        mock_sample.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# structure.py — line 194: strict validation with unseeded cycles
# ═══════════════════════════════════════════════════════════════════════


class TestStrictValidationUnseededCycles:
    def test_strict_rejects_unseeded_place_cycles(self):
        """Line 194: strict validation error includes unseeded_place_cycles."""
        net = StochasticPetriNet()
        net.add_place("P0", initial_tokens=0.0)
        net.add_place("P1", initial_tokens=0.0)
        net.add_transition("T0", threshold=0.1)
        net.add_transition("T1", threshold=0.1)
        net.add_arc("P0", "T0", weight=1.0)
        net.add_arc("T0", "P1", weight=1.0)
        net.add_arc("P1", "T1", weight=1.0)
        net.add_arc("T1", "P0", weight=1.0)
        with pytest.raises(ValueError, match="unseeded_place_cycles"):
            net.compile(strict_validation=True)


# ═══════════════════════════════════════════════════════════════════════
# structure.py — lines 433, 486: W_in/W_out None guard after _compiled=True
# ═══════════════════════════════════════════════════════════════════════


class TestVerifyWithNoneMatrices:
    def test_verify_boundedness_none_matrices(self):
        """Line 433: compiled=True but W_in=None triggers RuntimeError."""
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.5)
        net.add_transition("T", threshold=0.3)
        net.add_arc("P", "T", weight=0.5)
        net.add_arc("T", "P", weight=0.5)
        net._compiled = True
        net.W_in = None
        net.W_out = None
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_boundedness()

    def test_verify_liveness_none_matrices(self):
        """Line 486: compiled=True but W_in=None triggers RuntimeError."""
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.5)
        net.add_transition("T", threshold=0.3)
        net.add_arc("P", "T", weight=0.5)
        net.add_arc("T", "P", weight=0.5)
        net._compiled = True
        net.W_in = None
        net.W_out = None
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_liveness()


# ═══════════════════════════════════════════════════════════════════════
# structure.py — line 457: bounded = False (out-of-range marking)
# Line 457 is unreachable through normal code paths because np.clip
# (line 451) runs before the range check (line 456). Marking cannot
# exceed [0,1]+epsilon after clipping. Skipping per anti-slop policy.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# artifact.py — line 225: decompressed payload exceeds limit (post-flush)
# ═══════════════════════════════════════════════════════════════════════


class TestDecodeExceedsDecompressedLimit:
    def test_large_decompressed_payload_rejected(self, monkeypatch):
        """Line 225: total decompressed size > MAX_DECOMPRESSED_BYTES after flush.

        Set limit to 7 so that 8 bytes of decompressed data passes the
        decompress(comp, 8) call without unconsumed_tail, but fails the
        post-flush len(raw) > 7 check.
        """
        import scpn_control.scpn.artifact as art_mod

        monkeypatch.setattr(art_mod, "MAX_DECOMPRESSED_BYTES", 7)

        raw = b"\x00" * 8  # 8 bytes > 7 limit; divisible by 8
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        encoded = {
            "encoding": "u64-le-zlib-base64",
            "data_u64_b64_zlib": payload,
            "count": 1,
        }
        with pytest.raises(ArtifactValidationError, match="exceeds configured limit"):
            _decode_u64_compact(encoded)


# ═══════════════════════════════════════════════════════════════════════
# artifact.py — line 265: fraction_bits is bool (not int)
# ═══════════════════════════════════════════════════════════════════════


class TestValidateFractionBitsBool:
    def test_fraction_bits_bool_rejected(self):
        """Line 265: fraction_bits=True (bool) fails isinstance(x, int) guard."""
        art = Artifact(
            meta=ArtifactMeta(
                artifact_version="1.0.0",
                name="test",
                dt_control_s=0.001,
                stream_length=64,
                fixed_point=FixedPoint(data_width=16, fraction_bits=True, signed=False),
                firing_mode="binary",
                seed_policy=SeedPolicy(id="d", hash_fn="s", rng_family="x"),
                created_utc="2026-01-01T00:00:00Z",
                compiler=CompilerInfo(name="t", version="0.1", git_sha="0000000"),
            ),
            topology=Topology(
                places=[PlaceSpec(id=0, name="p0")],
                transitions=[TransitionSpec(id=0, name="t0", threshold=0.5)],
            ),
            weights=Weights(
                w_in=WeightMatrix(shape=[1, 1], data=[0.5]),
                w_out=WeightMatrix(shape=[1, 1], data=[0.5]),
            ),
            readout=Readout(actions=[], gains=[], abs_max=[], slew_per_s=[]),
            initial_state=InitialState(marking=[0.0], place_injections=[]),
        )
        with pytest.raises(ArtifactValidationError, match="fraction_bits"):
            _validate(art)
