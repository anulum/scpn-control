# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Fpga Export
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — FPGA Export Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for the FPGA bitstream export pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

import pytest

from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.fpga_export import (
    FPGAConfig,
    HDL_EXPORT_EVIDENCE_LOCAL_ONLY,
    HDL_EXPORT_EVIDENCE_SYNTHESIS_QUALIFIED,
    _leak_shift,
    assert_hdl_export_claim_admissible,
    compile_to_verilog,
    compile_to_vhdl,
    estimate_resources,
    export_bitstream_project,
    hdl_export_evidence,
    load_hdl_export_evidence,
    save_hdl_export_evidence,
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


class TestHDLExportEvidence:
    def test_local_evidence_binds_generated_project_files(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="xilinx", clock_mhz=125.0)
        out = tmp_path / "fpga_project"
        export_bitstream_project(cnet, cfg, out)

        evidence = hdl_export_evidence(
            cnet,
            cfg,
            out,
            controller_artifact_sha256="a" * 64,
            target_part="xc7a35tcpg236-1",
            generated_utc="2026-05-31T00:00:00Z",
        )

        assert evidence.claim_status == HDL_EXPORT_EVIDENCE_LOCAL_ONLY
        assert evidence.hdl_format == "verilog"
        assert evidence.hdl_sha256 == hashlib.sha256((out / "top.v").read_bytes()).hexdigest()
        assert evidence.weights_mem_sha256 == hashlib.sha256((out / "weights.mem").read_bytes()).hexdigest()
        assert evidence.constraints_sha256 == hashlib.sha256((out / "constraints.xdc").read_bytes()).hexdigest()
        assert evidence.makefile_sha256 == hashlib.sha256((out / "Makefile").read_bytes()).hexdigest()
        assert evidence.estimated_lut_count > 0
        assert len(evidence.payload_sha256) == 64
        with pytest.raises(ValueError, match="local-only"):
            assert_hdl_export_claim_admissible(evidence)

    def test_qualified_evidence_round_trips_with_synthesis_report_binding(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="xilinx", clock_mhz=100.0)
        out = tmp_path / "fpga_project"
        export_bitstream_project(cnet, cfg, out)
        report = tmp_path / "validation" / "reports" / "hardware" / "fpga_synth.rpt"
        report.parent.mkdir(parents=True)
        report.write_text("timing met\nslack 1.250 ns\n", encoding="utf-8")
        report_digest = hashlib.sha256(report.read_bytes()).hexdigest()

        evidence = hdl_export_evidence(
            cnet,
            cfg,
            out,
            controller_artifact_sha256="a" * 64,
            target_part="xc7a35tcpg236-1",
            synthesis_toolchain="vivado",
            synthesis_report_sha256=report_digest,
            synthesis_report_uri="validation/reports/hardware/fpga_synth.rpt",
            timing_slack_ns=1.25,
            generated_utc="2026-05-31T00:00:00Z",
            facility_claim_allowed=True,
        )
        assert evidence.claim_status == HDL_EXPORT_EVIDENCE_SYNTHESIS_QUALIFIED
        assert_hdl_export_claim_admissible(evidence, artifact_root=tmp_path)

        path = tmp_path / "validation" / "reports" / "hardware" / "hdl_export.json"
        save_hdl_export_evidence(evidence, path)
        loaded = load_hdl_export_evidence(path, require_facility_claim=True, artifact_root=tmp_path)
        assert loaded == evidence

    def test_qualified_evidence_rejects_unsafe_or_missing_synthesis_boundary(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="xilinx")
        out = tmp_path / "fpga_project"
        export_bitstream_project(cnet, cfg, out)

        with pytest.raises(ValueError, match="requires synthesis"):
            hdl_export_evidence(
                cnet,
                cfg,
                out,
                controller_artifact_sha256="a" * 64,
                target_part="xc7a35tcpg236-1",
                timing_slack_ns=1.0,
                facility_claim_allowed=True,
            )
        with pytest.raises(ValueError, match="safe relative path"):
            hdl_export_evidence(
                cnet,
                cfg,
                out,
                controller_artifact_sha256="a" * 64,
                target_part="xc7a35tcpg236-1",
                synthesis_toolchain="vivado",
                synthesis_report_sha256="b" * 64,
                synthesis_report_uri="../fpga_synth.rpt",
                timing_slack_ns=1.0,
                facility_claim_allowed=True,
            )

    def test_evidence_loader_rejects_tampering_and_duplicate_keys(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="intel")
        out = tmp_path / "fpga_project"
        export_bitstream_project(cnet, cfg, out)
        evidence = hdl_export_evidence(
            cnet,
            cfg,
            out,
            controller_artifact_sha256="a" * 64,
            target_part="10CL025YU256C8G",
            generated_utc="2026-05-31T00:00:00Z",
        )
        path = tmp_path / "hdl_export.json"
        save_hdl_export_evidence(evidence, path)
        payload = asdict(evidence)
        payload["target_part"] = "tampered"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="payload_sha256"):
            load_hdl_export_evidence(path)

        inconsistent = asdict(evidence)
        inconsistent["estimated_lut_count"] = int(inconsistent["estimated_lut_count"]) + 1
        unsigned = dict(inconsistent)
        unsigned["payload_sha256"] = ""
        inconsistent["payload_sha256"] = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        path.write_text(json.dumps(inconsistent, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="resource_estimate_sha256"):
            load_hdl_export_evidence(path)

        dup = tmp_path / "dup_hdl_export.json"
        dup.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate JSON key"):
            load_hdl_export_evidence(dup)

    def test_missing_project_file_rejects_evidence_creation(self, tmp_path):
        cnet = _compile_small_net()
        cfg = FPGAConfig(target="xilinx")
        out = tmp_path / "fpga_project"
        export_bitstream_project(cnet, cfg, out)
        (out / "weights.mem").unlink()

        with pytest.raises(ValueError, match="project file is missing"):
            hdl_export_evidence(
                cnet,
                cfg,
                out,
                controller_artifact_sha256="a" * 64,
                target_part="xc7a35tcpg236-1",
            )
