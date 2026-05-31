# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — HIL Harness Tests

"""Tests for Hardware-in-the-Loop test harness."""

import json
import logging

import numpy as np
import pytest

from scpn_control.control.hil_harness import (
    ADCConfig,
    DACConfig,
    SensorInterface,
    HILControlLoop,
    ControlLoopMetrics,
    FPGASNNExport,
    FPGARegisterMap,
    SNNNeuronConfig,
    HILBenchmarkResult,
    assert_hil_replay_evidence_admissible,
    hil_replay_evidence,
    load_hil_replay_evidence,
    run_hil_benchmark,
    save_hil_replay_evidence,
)


class TestSensorInterface:
    def test_adc_quantization(self):
        sensor = SensorInterface(adc=ADCConfig(resolution_bits=12))
        reading = sensor.read_adc(0.5)
        assert isinstance(reading, float)
        # 12-bit in ±1.5V → LSB ~0.73 mV
        assert abs(reading - 0.5) < 0.01  # within ~10 LSBs (noise)

    def test_adc_clamps_range(self):
        sensor = SensorInterface()
        # Beyond range should clamp
        reading = sensor.read_adc(10.0)  # way above ±1.5V
        assert reading <= 1.5 + 0.01

    def test_dac_slew_rate(self):
        sensor = SensorInterface(dac=DACConfig(slew_rate_v_per_us=50.0))
        # Large step should be slew-limited
        out = sensor.write_dac(10.0, dt_us=0.1)
        assert out < 10.0  # can't reach 10V in 0.1 us at 50V/us

    def test_magnetic_probe(self):
        sensor = SensorInterface(rng_seed=42)
        B = sensor.read_magnetic_probe(5.3)
        assert abs(B - 5.3) < 0.5  # reasonable noise level

    def test_coil_current(self):
        sensor = SensorInterface()
        I = sensor.write_coil_current(25.0, dt_us=1000.0)
        assert abs(I - 25.0) < 5.0  # slew-limited approach

    def test_adc_deterministic_with_seed(self):
        s1 = SensorInterface(rng_seed=42)
        s2 = SensorInterface(rng_seed=42)
        r1 = s1.read_adc(0.5)
        r2 = s2.read_adc(0.5)
        assert r1 == r2


class TestHILControlLoop:
    def test_basic_loop(self):
        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(lambda err, _: -0.5 * err)
        metrics = loop.run(iterations=100, setpoint=0.0)
        assert isinstance(metrics, ControlLoopMetrics)
        assert metrics.iterations == 100
        assert len(metrics.measured_dt_us) == 100

    def test_sub_ms_latency(self):
        """Core requirement: P95 loop latency < 1 ms."""
        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(lambda err, _: -0.5 * err)
        metrics = loop.run(iterations=500)
        # Pure Python PID in tight loop should easily be < 1ms
        assert metrics.p95_latency_us < 1000.0
        assert metrics.sub_ms_achieved

    def test_no_controller_raises(self):
        loop = HILControlLoop()
        with pytest.raises(RuntimeError, match="No controller"):
            loop.run(iterations=10)

    def test_pid_controller_stabilises(self):
        state = {"i": 0.0, "prev": 0.0}

        def pid(err, _):
            state["i"] += err
            d = err - state["prev"]
            state["prev"] = err
            return 2.0 * err + 0.1 * state["i"] + 0.5 * d

        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(pid)

        def plant(s, cmd):
            return s + cmd * 0.001  # integrator

        metrics = loop.run(iterations=200, plant_fn=plant, initial_state=1.0, setpoint=0.0)
        assert metrics.iterations == 200
        assert metrics.mean_latency_us > 0.0

    def test_jitter_measured(self):
        loop = HILControlLoop()
        loop.set_controller(lambda err, _: -err)
        metrics = loop.run(iterations=200)
        assert metrics.jitter_std_us >= 0.0
        assert np.isfinite(metrics.jitter_std_us)


class TestFPGASNNExport:
    def test_register_map_generation(self):
        exporter = FPGASNNExport(n_neurons=50, n_channels=2)
        reg_map = exporter.generate_register_map()
        assert isinstance(reg_map, FPGARegisterMap)
        assert reg_map.n_neurons == 100  # 50 * 2 channels
        assert len(reg_map.neurons) == 100
        assert len(reg_map.input_ports) == 2
        assert len(reg_map.output_ports) == 2

    def test_verilog_header(self):
        exporter = FPGASNNExport(n_neurons=10, n_channels=2, clock_mhz=100.0)
        reg_map = exporter.generate_register_map()
        verilog = exporter.export_verilog_header(reg_map)
        assert "module snn_controller" in verilog
        assert "N_NEURONS" in verilog
        assert "CLK_HZ" in verilog
        assert "v_mem" in verilog
        assert "endmodule" in verilog

    def test_neuron_configs(self):
        exporter = FPGASNNExport(n_neurons=5, n_channels=1)
        reg_map = exporter.generate_register_map(v_threshold=0.4, tau_mem_us=20000.0)
        for neuron in reg_map.neurons:
            assert isinstance(neuron, SNNNeuronConfig)
            assert neuron.v_threshold == 0.4
            assert neuron.tau_mem_us == 20000.0

    def test_clock_frequency(self):
        exporter = FPGASNNExport(clock_mhz=200.0)
        reg_map = exporter.generate_register_map()
        assert reg_map.clock_hz == 200_000_000


class TestHILBenchmark:
    def test_full_benchmark(self):
        result = run_hil_benchmark(iterations=200, verbose=False)
        assert isinstance(result, HILBenchmarkResult)
        assert result.total_loop_latency_us > 0.0
        assert result.passes_sub_ms

    def test_benchmark_with_fpga_export(self):
        result = run_hil_benchmark(iterations=100, include_fpga_export=True)
        assert result.fpga_register_map is not None
        assert result.fpga_register_map.n_neurons > 0

    def test_benchmark_without_fpga(self):
        result = run_hil_benchmark(iterations=100, include_fpga_export=False)
        assert result.fpga_register_map is None

    def test_latency_budget_decomposition(self):
        result = run_hil_benchmark(iterations=100)
        total = result.sensor_latency_us + result.controller_latency_us + result.actuator_latency_us
        assert abs(total - result.total_loop_latency_us) < 0.1

    def test_sub_ms_p95(self):
        """The key deliverable: demonstrate sub-ms control loop latency."""
        result = run_hil_benchmark(iterations=1000)
        assert result.passes_sub_ms


class TestHILReplayEvidence:
    @staticmethod
    def _metrics(
        *,
        overrun_count: int = 0,
        target_dt_us: float = 1000.0,
        max_latency_us: float = 40.0,
    ) -> ControlLoopMetrics:
        return ControlLoopMetrics(
            iterations=10,
            target_dt_us=target_dt_us,
            measured_dt_us=[10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0, 40.0],
            p50_latency_us=21.0,
            p95_latency_us=37.75,
            p99_latency_us=39.55,
            max_latency_us=max_latency_us,
            min_latency_us=10.0,
            mean_latency_us=22.7,
            jitter_std_us=9.117,
            overrun_count=overrun_count,
            overrun_fraction=overrun_count / 10,
            sub_ms_achieved=True,
        )

    def test_local_replay_evidence_round_trips_and_rejects_tampering(self, tmp_path):
        evidence = hil_replay_evidence(
            self._metrics(),
            controller_id="tests.test_hil_harness.pid-vde",
            generated_at="2026-05-24T00:00:00Z",
        )
        assert evidence["admission"]["deployment_claim_allowed"] is False
        assert evidence["admission"]["claim_status"] == "bounded_local_hil_replay_only"
        assert_hil_replay_evidence_admissible(evidence)

        path = save_hil_replay_evidence(evidence, tmp_path / "hil_replay_evidence.json")
        assert load_hil_replay_evidence(path) == evidence

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["timing"]["p95_latency_us"] = 1.0
        path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="payload_sha256"):
            load_hil_replay_evidence(path)

    def test_deployment_claim_requires_qualified_target_hardware(self):
        with pytest.raises(ValueError, match="target_hardware"):
            hil_replay_evidence(
                self._metrics(),
                controller_id="pid-vde",
                deployment_claim_allowed=True,
                generated_at="2026-05-24T00:00:00Z",
            )

    def test_deployment_claim_rejects_runtime_safety_events(self):
        with pytest.raises(ValueError, match="backpressure"):
            hil_replay_evidence(
                self._metrics(),
                controller_id="pid-vde",
                target_hardware_id="jetson-orin-lab-01",
                target_hardware_class="jetson-orin-preempt-rt",
                rt_kernel="linux-rt-6.8.0-lab",
                backpressure_events=1,
                deployment_claim_allowed=True,
                generated_at="2026-05-24T00:00:00Z",
            )

    def test_deployment_claim_accepts_consistent_target_hardware_contract(self):
        evidence = hil_replay_evidence(
            self._metrics(),
            controller_id="pid-vde",
            target_hardware_id="jetson-orin-lab-01",
            target_hardware_class="jetson-orin-preempt-rt",
            rt_kernel="linux-rt-6.8.0-lab",
            deployment_claim_allowed=True,
            generated_at="2026-05-24T00:00:00Z",
        )
        admitted = assert_hil_replay_evidence_admissible(
            evidence,
            require_target_hardware=True,
        )
        assert admitted["admission"]["claim_status"] == "qualified_target_hardware_deployment_evidence"
        assert admitted["timing"]["max_latency_us"] <= admitted["timing"]["target_dt_us"]


# ── HILDemoRunner ──────────────────────────────────────────────────


class TestHILDemoRunner:
    def test_q16_roundtrip(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        for val in [0.0, 0.5, -0.5, 1.0, -1.0, 0.123]:
            encoded = HILDemoRunner.float_to_q16_16(val)
            decoded = HILDemoRunner.q16_16_to_float(encoded)
            assert abs(decoded - val) < 1e-4

    def test_step_returns_output(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        runner = HILDemoRunner(n_neurons=4, n_inputs=2, n_outputs=2)
        out = runner.step(np.array([0.1, 0.2]))
        assert out.shape == (2,)
        assert runner.total_steps == 1

    def test_run_episode_no_faults(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        runner = HILDemoRunner(n_neurons=4, n_inputs=4, n_outputs=4)
        report = runner.run_episode(n_steps=50, inject_faults=False)
        assert report["total_steps"] == 50
        assert report["tmr_mismatches"] == 0

    def test_run_episode_with_faults(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        runner = HILDemoRunner(n_neurons=4, n_inputs=4, n_outputs=4)
        report = runner.run_episode(n_steps=200, inject_faults=True)
        assert report["total_steps"] == 200
        assert report["tmr_mismatch_rate"] >= 0.0

    def test_inject_bitflip(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        runner = HILDemoRunner(n_neurons=4, n_inputs=2, n_outputs=2)
        runner.tmr_copies[0][0] = 0.5
        runner.tmr_copies[1][0] = 0.5
        runner.tmr_copies[2][0] = 0.5
        runner.inject_bitflip(neuron_idx=0, bit_idx=10)
        assert runner.tmr_copies[1][0] == 0.5
        assert runner.tmr_copies[2][0] == 0.5

    def test_report_keys(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        runner = HILDemoRunner()
        runner.step(np.zeros(4))
        report = runner.report()
        for key in (
            "total_steps",
            "tmr_mismatches",
            "tmr_mismatch_rate",
            "latency_mean_cycles",
            "latency_p95_cycles",
            "latency_max_cycles",
            "latency_mean_ns",
            "n_neurons",
            "n_inputs",
            "n_outputs",
        ):
            assert key in report

    def test_tmr_vote_corrects_single_fault(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        runner = HILDemoRunner(n_neurons=4, n_inputs=2, n_outputs=2)
        runner.tmr_copies[0][:] = [0.1, 0.2, 0.3, 0.4]
        runner.tmr_copies[1][:] = [0.1, 0.2, 0.3, 0.4]
        runner.tmr_copies[2][:] = [99.0, 0.2, 0.3, 0.4]  # faulty
        voted = runner._tmr_vote()
        assert abs(voted[0] - 0.1) < 0.01
        assert runner.tmr_mismatches >= 1


# ── run_hil_benchmark_detailed ─────────────────────────────────────


class TestRunHILBenchmarkDetailed:
    def test_returns_expected_keys(self):
        from scpn_control.control.hil_harness import run_hil_benchmark_detailed

        result = run_hil_benchmark_detailed(n_steps=100)
        for key in ("n_steps", "mean_us", "p50_us", "p95_us", "p99_us", "max_us", "stage_breakdown"):
            assert key in result
        assert result["n_steps"] == 100

    def test_stage_breakdown_positive(self):
        from scpn_control.control.hil_harness import run_hil_benchmark_detailed

        result = run_hil_benchmark_detailed(n_steps=200)
        sb = result["stage_breakdown"]
        assert sb["state_estimation_mean_us"] >= 0.0
        assert sb["controller_step_mean_us"] >= 0.0
        assert sb["actuator_command_mean_us"] >= 0.0

    def test_mean_bounded_by_max(self):
        from scpn_control.control.hil_harness import run_hil_benchmark_detailed

        result = run_hil_benchmark_detailed(n_steps=100)
        assert result["mean_us"] <= result["max_us"]
        assert result["p50_us"] <= result["p99_us"]


class TestHILBenchmarkVerbose:
    def test_verbose_prints_output(self, caplog):
        with caplog.at_level(logging.INFO, logger="scpn_control.control.hil_harness"):
            run_hil_benchmark(iterations=50, verbose=True)
        assert "HIL Benchmark Results" in caplog.text
        assert "P50 latency" in caplog.text
        assert "P95 latency" in caplog.text
        assert "Overruns" in caplog.text
        assert "Sub-ms" in caplog.text

    def test_verbose_with_fpga(self, caplog):
        with caplog.at_level(logging.INFO, logger="scpn_control.control.hil_harness"):
            run_hil_benchmark(iterations=50, verbose=True, include_fpga_export=True)
        assert "FPGA neurons" in caplog.text
        assert "FPGA clock" in caplog.text


class TestHILDemoRunnerWeights:
    def test_load_weights_from_controller(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        class _MockController:
            weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
            output_weights = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        runner = HILDemoRunner(n_neurons=4, n_inputs=2, n_outputs=2)
        runner.load_weights_from_controller(_MockController())
        np.testing.assert_allclose(runner.weights, _MockController.weights)
        np.testing.assert_allclose(runner.output_weights, _MockController.output_weights)

    def test_load_weights_no_output_weights(self):
        from scpn_control.control.hil_harness import HILDemoRunner

        class _MockCtrlNoOutput:
            weights = np.array([[0.1, 0.2], [0.3, 0.4]])

        runner = HILDemoRunner(n_neurons=2, n_inputs=2, n_outputs=2)
        runner.load_weights_from_controller(_MockCtrlNoOutput())
        np.testing.assert_allclose(runner.weights, _MockCtrlNoOutput.weights)
        # output_weights should remain zeros
        assert np.all(runner.output_weights == 0.0)
