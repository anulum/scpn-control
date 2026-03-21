# ──────────────────────────────────────────────────────────────────────
# SCPN Control — FPGA Export Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for the FPGA bitstream export pipeline."""

from __future__ import annotations

import re

import pytest

from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.fpga_export import (
    FPGAConfig,
    _leak_shift,
    compile_to_verilog,
    compile_to_vhdl,
    estimate_resources,
    export_bitstream_project,
)
from scpn_control.scpn.structure import StochasticPetriNet


def _make_small_net() -> StochasticPetriNet:
    """3 places, 2 transitions — minimal compilable net."""
    net = StochasticPetriNet()
    net.add_place("P0", initial_tokens=0.8)
    net.add_place("P1", initial_tokens=0.0)
    net.add_place("P2", initial_tokens=0.3)
    net.add_transition("T0", threshold=0.5)
    net.add_transition("T1", threshold=0.7)
    net.add_arc("P0", "T0", weight=0.6)
    net.add_arc("P2", "T1", weight=0.4)
    net.add_arc("T0", "P1", weight=0.9)
    net.add_arc("T1", "P0", weight=0.5)
    return net


def _compile_small_net():
    net = _make_small_net()
    compiler = FusionCompiler(bitstream_length=256, seed=7)
    return compiler.compile(net)


class TestCompileToVerilog:
    def test_contains_lif_neuron_module(self):
        cnet = _compile_small_net()
        verilog = compile_to_verilog(cnet, FPGAConfig())
        assert "module lif_neuron" in verilog

    def test_contains_weight_rom_module(self):
        cnet = _compile_small_net()
        verilog = compile_to_verilog(cnet, FPGAConfig())
        assert "module weight_rom" in verilog

    def test_weight_rom_depth(self):
        cnet = _compile_small_net()
        verilog = compile_to_verilog(cnet, FPGAConfig())
        # 2 transitions -> depth = 2*2 = 4
        assert "DEPTH     = 4" in verilog or "DEPTH = 4" in verilog or ".DEPTH(4)" in verilog

    def test_contains_top_module(self):
        cnet = _compile_small_net()
        verilog = compile_to_verilog(cnet, FPGAConfig())
        assert "module scpn_top" in verilog

    def test_contains_readmemh(self):
        cnet = _compile_small_net()
        verilog = compile_to_verilog(cnet, FPGAConfig())
        assert '$readmemh("weights.mem"' in verilog

    def test_bitwidth_8(self):
        cnet = _compile_small_net()
        cfg = FPGAConfig(lif_bit_width=8)
        verilog = compile_to_verilog(cnet, cfg)
        assert "BW         = 8" in verilog or "BW = 8" in verilog

    def test_bitwidth_32(self):
        cnet = _compile_small_net()
        cfg = FPGAConfig(lif_bit_width=32)
        verilog = compile_to_verilog(cnet, cfg)
        assert "BW         = 32" in verilog or "BW = 32" in verilog


class TestCompileToVHDL:
    def test_contains_entity_and_architecture(self):
        cnet = _compile_small_net()
        vhdl = compile_to_vhdl(cnet, FPGAConfig())
        assert "entity lif_neuron" in vhdl.lower() or "entity lif_neuron" in vhdl
        assert "architecture rtl" in vhdl.lower() or "architecture rtl" in vhdl

    def test_contains_weight_rom_entity(self):
        cnet = _compile_small_net()
        vhdl = compile_to_vhdl(cnet, FPGAConfig())
        assert "entity weight_rom" in vhdl


class TestEstimateResources:
    def test_positive_counts(self):
        cnet = _compile_small_net()
        res = estimate_resources(cnet, FPGAConfig())
        assert res["lut_count"] > 0
        assert res["ff_count"] > 0
        assert res["bram_blocks"] > 0
        assert res["estimated_fmax_mhz"] > 0

    def test_neuron_count_matches(self):
        cnet = _compile_small_net()
        res = estimate_resources(cnet, FPGAConfig())
        assert res["n_neurons"] == 2

    def test_rom_depth(self):
        cnet = _compile_small_net()
        res = estimate_resources(cnet, FPGAConfig())
        assert res["weight_rom_depth"] == 4  # 2*2


class TestExportProject:
    def test_xilinx_creates_files(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="xilinx", clock_mhz=200.0)
        out = tmp_path / "fpga_out"
        export_bitstream_project(cnet, cfg, out)
        assert (out / "top.v").exists()
        assert (out / "weights.mem").exists()
        assert (out / "constraints.xdc").exists()
        assert (out / "Makefile").exists()

    def test_intel_creates_sdc(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="intel")
        out = tmp_path / "fpga_intel"
        export_bitstream_project(cnet, cfg, out)
        assert (out / "top.v").exists()
        assert (out / "constraints.sdc").exists()

    def test_lattice_creates_vhdl(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="lattice")
        out = tmp_path / "fpga_lattice"
        export_bitstream_project(cnet, cfg, out)
        assert (out / "top.vhd").exists()
        assert not (out / "top.v").exists()

    def test_clock_constraint_xilinx(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="xilinx", clock_mhz=150.0)
        out = tmp_path / "fpga_clk"
        export_bitstream_project(cnet, cfg, out)
        xdc = (out / "constraints.xdc").read_text()
        period = 1000.0 / 150.0
        assert f"{period:.3f}" in xdc

    def test_weights_mem_line_count(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig()
        out = tmp_path / "fpga_wm"
        export_bitstream_project(cnet, cfg, out)
        lines = [ln for ln in (out / "weights.mem").read_text().splitlines() if ln.strip()]
        # 2 neurons * 2 neurons = 4 entries
        assert len(lines) == 4


class TestRoundTrip:
    def test_neuron_count_round_trip(self):
        """Compile SPN -> export Verilog -> parse back N_NEURONS parameter."""
        cnet = _compile_small_net()
        verilog = compile_to_verilog(cnet, FPGAConfig())
        match = re.search(r"parameter\s+N_NEURONS\s*=\s*(\d+)", verilog)
        assert match is not None
        assert int(match.group(1)) == cnet.n_transitions


class TestFPGAConfigValidation:
    def test_invalid_target(self):
        with pytest.raises(ValueError, match="target"):
            FPGAConfig(target="altera")

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="lif_bit_width"):
            FPGAConfig(lif_bit_width=12)

    def test_invalid_clock(self):
        with pytest.raises(ValueError, match="clock_mhz"):
            FPGAConfig(clock_mhz=-10.0)

    def test_exceeds_neuron_max(self):
        cnet = _compile_small_net()
        cfg = FPGAConfig(n_neurons_max=1)
        with pytest.raises(ValueError, match="exceeds"):
            compile_to_verilog(cnet, cfg)

    def test_invalid_n_neurons_max(self):
        with pytest.raises(ValueError, match="n_neurons_max"):
            FPGAConfig(n_neurons_max=0)

    def test_invalid_fifo_depth(self):
        with pytest.raises(ValueError, match="fifo_depth"):
            FPGAConfig(fifo_depth=0)


class TestLeakShift:
    def test_tau_mem_zero_returns_zero(self):
        assert _leak_shift(0.0, 1.0) == 0

    def test_tau_mem_negative_returns_zero(self):
        assert _leak_shift(-1.0, 1.0) == 0

    def test_dt_zero_returns_zero(self):
        assert _leak_shift(1.0, 0.0) == 0

    def test_ratio_leq_one_returns_zero(self):
        assert _leak_shift(1.0, 1.0) == 0
        assert _leak_shift(0.5, 1.0) == 0

    def test_normal_ratio(self):
        shift = _leak_shift(8.0, 1.0)
        assert shift == 3


class TestCompileToVHDLExceedsMax:
    def test_vhdl_exceeds_neuron_max(self):
        cnet = _compile_small_net()
        cfg = FPGAConfig(n_neurons_max=1)
        with pytest.raises(ValueError, match="exceeds"):
            compile_to_vhdl(cnet, cfg)


class TestWeightsMemZeroPad:
    def test_zero_padded_weight_entries(self, tmp_path):
        net = StochasticPetriNet()
        net.add_place("P0", initial_tokens=0.8)
        net.add_transition("T0", threshold=0.5)
        net.add_transition("T1", threshold=0.6)
        net.add_transition("T2", threshold=0.7)
        net.add_arc("P0", "T0", weight=0.6)
        net.add_arc("T0", "P0", weight=0.9)
        net.add_arc("P0", "T1", weight=0.4)
        net.add_arc("T1", "P0", weight=0.3)
        net.add_arc("P0", "T2", weight=0.5)
        net.add_arc("T2", "P0", weight=0.2)
        compiler = FusionCompiler(bitstream_length=256, seed=7)
        cnet = compiler.compile(net)
        assert cnet.n_transitions > cnet.n_places

        cfg = FPGAConfig()
        out = tmp_path / "fpga_pad"
        export_bitstream_project(cnet, cfg, out)
        lines = [ln for ln in (out / "weights.mem").read_text().splitlines() if ln.strip()]
        n = cnet.n_transitions
        hw = (cfg.lif_bit_width + 3) // 4
        assert len(lines) == n * n
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if j >= cnet.n_places:
                    assert lines[idx].strip() == "0" * hw
