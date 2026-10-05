# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware-in-the-Loop Test Harness

"""HIL software FPGA SNN demo runner."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray


class HILDemoRunner:
    """Simulate FPGA register-mapped SNN controller for demo/testing.

    Maps Python SNN controller state to/from Q16.16 fixed-point registers,
    injects bit-flip faults, and verifies TMR recovery. See docs/hil_demo.md.
    """

    CLOCK_HZ = 250_000_000
    Q16_SCALE = 65536.0

    def __init__(self, n_neurons: int = 8, n_inputs: int = 4, n_outputs: int = 4):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # Register file (simulated as uint32 array)
        self.registers = np.zeros(512, dtype=np.uint32)
        # TMR: 3 copies of neuron state
        self.tmr_copies: list[AnyFloatArray] = [np.zeros(n_neurons, dtype=np.float64) for _ in range(3)]
        self.weights: AnyFloatArray = np.zeros((n_neurons, n_inputs), dtype=np.float64)
        self.output_weights: AnyFloatArray = np.zeros((n_outputs, n_neurons), dtype=np.float64)
        self.tmr_mismatches = 0
        self.total_steps = 0
        self.latency_cycles: list[int] = []

    @staticmethod
    def float_to_q16_16(x: float) -> int:
        """Convert a float to a 32-bit Q16.16 fixed-point integer.

        Parameters
        ----------
        x
            The value to convert.

        Returns
        -------
        int
            The Q16.16 representation masked to 32 bits.
        """
        return int(round(x * HILDemoRunner.Q16_SCALE)) & 0xFFFFFFFF

    @staticmethod
    def q16_16_to_float(x: int) -> float:
        """Convert a 32-bit Q16.16 fixed-point integer to a float.

        Parameters
        ----------
        x
            The Q16.16 representation (two's-complement 32-bit).

        Returns
        -------
        float
            The decoded floating-point value.
        """
        if x & 0x80000000:
            x -= 0x100000000
        return x / HILDemoRunner.Q16_SCALE

    def load_weights_from_controller(self, controller: object) -> None:
        """Load weights from a Python SNN controller object."""
        if hasattr(controller, "weights"):
            w = np.asarray(controller.weights, dtype=np.float64)
            self.weights = w[: self.n_neurons, : self.n_inputs]
        if hasattr(controller, "output_weights"):
            ow = np.asarray(controller.output_weights, dtype=np.float64)
            self.output_weights = ow[: self.n_outputs, : self.n_neurons]

    def _lif_step(
        self, state: AnyFloatArray, inputs: AnyFloatArray, dt_s: float = 0.001
    ) -> tuple[AnyFloatArray, AnyFloatArray]:
        """Leaky Integrate-and-Fire neuron update."""
        tau = 0.02  # 20ms membrane time constant
        threshold = 1.0
        reset = 0.0
        current = self.weights @ inputs
        state = state * (1.0 - dt_s / tau) + current * dt_s
        spikes = (state >= threshold).astype(np.float64)
        state = np.where(spikes > 0, reset, state)
        return state, spikes

    def _tmr_vote(self) -> FloatArray:
        """Majority vote across TMR copies. Returns voted state."""
        # TMR median voter
        stacked = np.stack(self.tmr_copies, axis=0)
        voted = np.median(stacked, axis=0)
        for i in range(3):
            if not np.allclose(self.tmr_copies[i], voted, atol=0.01):
                self.tmr_mismatches += 1
                self.tmr_copies[i] = voted.copy()
                break
        return np.asarray(voted)

    def step(self, inputs: AnyFloatArray) -> FloatArray:
        """Execute one SNN inference step through simulated register pipeline."""
        t0 = time.perf_counter_ns()
        inp = np.asarray(inputs[: self.n_inputs], dtype=np.float64)

        # Write input registers
        for i in range(self.n_inputs):
            self.registers[0x60 // 4 + i] = self.float_to_q16_16(float(inp[i]))

        # TMR: update all 3 copies
        for c in range(3):
            self.tmr_copies[c], spikes = self._lif_step(self.tmr_copies[c], inp)

        # Vote
        voted = self._tmr_vote()

        # Write neuron state registers
        for i in range(min(self.n_neurons, 8)):
            self.registers[0x20 // 4 + i] = self.float_to_q16_16(float(voted[i]))

        # Compute output
        output = np.asarray(self.output_weights @ voted)
        for i in range(self.n_outputs):
            self.registers[0x70 // 4 + i] = self.float_to_q16_16(float(output[i]))

        elapsed_ns = time.perf_counter_ns() - t0
        cycles = max(1, int(elapsed_ns * self.CLOCK_HZ / 1e9))
        self.latency_cycles.append(cycles)
        self.registers[0x200 // 4] = np.uint32(cycles)
        self.total_steps += 1

        return output

    def inject_bitflip(self, neuron_idx: int = 0, bit_idx: int = 15) -> None:
        """Inject a single-bit fault into one TMR copy."""
        state_val = self.tmr_copies[0][neuron_idx]
        raw = np.array([state_val], dtype=np.float64).view(np.uint64)[0]
        flipped = np.uint64(raw ^ (np.uint64(1) << np.uint64(bit_idx)))
        new_val = np.array([flipped], dtype=np.uint64).view(np.float64)[0]
        if np.isfinite(new_val):
            self.tmr_copies[0][neuron_idx] = new_val

    def run_episode(self, n_steps: int = 1000, inject_faults: bool = False) -> dict[str, Any]:
        """Run a demo episode with optional fault injection."""
        rng = np.random.default_rng(42)
        outputs = []

        for t in range(n_steps):
            inputs = rng.normal(0, 0.1, size=self.n_inputs)
            if inject_faults and t % 100 == 50:
                self.inject_bitflip(neuron_idx=t % self.n_neurons, bit_idx=int(rng.integers(0, 52)))
            out = self.step(inputs)
            outputs.append(out.copy())

        return self.report()

    def report(self) -> dict[str, Any]:
        """Generate benchmark report."""
        lat = np.array(self.latency_cycles) if self.latency_cycles else np.array([0])
        return {
            "total_steps": self.total_steps,
            "tmr_mismatches": self.tmr_mismatches,
            "tmr_mismatch_rate": self.tmr_mismatches / max(self.total_steps, 1),
            "latency_mean_cycles": float(np.mean(lat)),
            "latency_p95_cycles": float(np.percentile(lat, 95)),
            "latency_max_cycles": float(np.max(lat)),
            "latency_mean_ns": float(np.mean(lat) / self.CLOCK_HZ * 1e9),
            "n_neurons": self.n_neurons,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }
