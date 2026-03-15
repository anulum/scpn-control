# ──────────────────────────────────────────────────────────────────────
# SCPN Control — FPGA Bitstream Export
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Export a CompiledNet to synthesisable HDL (Verilog / VHDL) for FPGA targets.

Generates a complete project directory with top-level module, weight ROM
initialisation, timing constraints, and a Makefile for Vivado or Quartus.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .compiler import CompiledNet

VALID_TARGETS = ("xilinx", "intel", "lattice")
VALID_BIT_WIDTHS = (8, 16, 32)

# Fixed-point fractional bits per data width.  Wesson Ch.14 style:
# reserve 1 sign bit, rest is fraction.
_FRAC_BITS = {8: 6, 16: 14, 32: 30}


@dataclass
class FPGAConfig:
    """FPGA synthesis target configuration."""

    target: str = "xilinx"
    clock_mhz: float = 100.0
    lif_bit_width: int = 16
    n_neurons_max: int = 256
    fifo_depth: int = 16

    def __post_init__(self) -> None:
        if self.target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got '{self.target}'")
        if self.clock_mhz <= 0.0:
            raise ValueError(f"clock_mhz must be > 0, got {self.clock_mhz}")
        if self.lif_bit_width not in VALID_BIT_WIDTHS:
            raise ValueError(f"lif_bit_width must be one of {VALID_BIT_WIDTHS}, got {self.lif_bit_width}")
        if self.n_neurons_max < 1:
            raise ValueError(f"n_neurons_max must be >= 1, got {self.n_neurons_max}")
        if self.fifo_depth < 1:
            raise ValueError(f"fifo_depth must be >= 1, got {self.fifo_depth}")


def _quantize_weight(value: float, bit_width: int) -> int:
    """Convert a float weight in [-1, 1] to signed fixed-point integer."""
    frac = _FRAC_BITS[bit_width]
    scale = (1 << frac) - 1
    clamped = max(-1.0, min(1.0, value))
    q = int(round(clamped * scale))
    max_val = (1 << (bit_width - 1)) - 1
    min_val = -(1 << (bit_width - 1))
    return max(min_val, min(max_val, q))


def _quantize_threshold(value: float, bit_width: int) -> int:
    """Convert a threshold in [0, 1] to unsigned fixed-point integer."""
    frac = _FRAC_BITS[bit_width]
    scale = (1 << frac) - 1
    clamped = max(0.0, min(1.0, value))
    q = int(round(clamped * scale))
    max_val = (1 << bit_width) - 1
    return max(0, min(max_val, q))


def _hex_width(bit_width: int) -> int:
    return (bit_width + 3) // 4


def _leak_shift(tau_mem: float, dt: float) -> int:
    """Compute right-shift approximation of leak factor (1 - dt/tau_mem).

    For tau_mem >> dt the leak per tick is small.  We approximate
    leak = V >> shift  where shift = round(log2(tau_mem / dt)).
    """
    if tau_mem <= 0.0 or dt <= 0.0:
        return 0
    ratio = tau_mem / dt
    if ratio <= 1.0:
        return 0
    return max(1, int(round(math.log2(ratio))))


# ── Verilog generation ───────────────────────────────────────────────────────


def _verilog_lif_neuron(bit_width: int, leak_shift: int) -> str:
    """Structural Verilog for a single LIF neuron."""
    bw = bit_width
    return textwrap.dedent(f"""\
        module lif_neuron #(
            parameter BW         = {bw},
            parameter LEAK_SHIFT = {leak_shift}
        )(
            input  wire             clk,
            input  wire             rst_n,
            input  wire signed [BW-1:0] i_syn,
            input  wire        [BW-1:0] v_threshold,
            output reg              spike,
            output reg  signed [BW-1:0] v_mem
        );

            wire signed [BW-1:0] leak;
            wire signed [BW:0]   v_next_raw;
            wire signed [BW-1:0] v_next;
            wire                 fire;

            // Leak: V >> LEAK_SHIFT (arithmetic right shift)
            assign leak      = v_mem >>> LEAK_SHIFT;
            // V[t+1] = V[t] - leak + I_syn
            assign v_next_raw = {{1'b0, v_mem}} - {{1'b0, leak}} + {{i_syn[BW-1], i_syn}};
            // Saturate to [0, max_positive]
            assign v_next    = (v_next_raw[BW]) ? {bw}'d0 :
                               (v_next_raw[BW-1:0] > $signed({{{bw}'d0, {{(BW-1){{1'b1}}}}}})) ?
                                   $signed({{{bw}'d0, {{(BW-1){{1'b1}}}}}}) : v_next_raw[BW-1:0];
            // Threshold comparator
            assign fire      = ($unsigned(v_next) >= v_threshold);

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    v_mem <= {bw}'d0;
                    spike <= 1'b0;
                end else begin
                    spike <= fire;
                    if (fire)
                        v_mem <= {bw}'d0;  // Reset on spike
                    else
                        v_mem <= v_next;
                end
            end
        endmodule
    """)


def _verilog_weight_rom(n_neurons: int, bit_width: int) -> str:
    """Weight ROM module with $readmemh initialisation."""
    depth = n_neurons * n_neurons
    addr_bits = max(1, int(math.ceil(math.log2(max(depth, 2)))))
    return textwrap.dedent(f"""\
        module weight_rom #(
            parameter DEPTH     = {depth},
            parameter BW        = {bit_width},
            parameter ADDR_BITS = {addr_bits}
        )(
            input  wire                  clk,
            input  wire [ADDR_BITS-1:0]  addr,
            output reg  signed [BW-1:0]  data
        );

            reg signed [BW-1:0] mem [0:DEPTH-1];

            initial $readmemh("weights.mem", mem);

            always @(posedge clk) begin
                data <= mem[addr];
            end
        endmodule
    """)


def _verilog_bitstream_decoder(n_places: int, bit_width: int) -> str:
    """Decoder: maps SPN bitstream-encoded place tokens to neuron input currents."""
    addr_bits = max(1, int(math.ceil(math.log2(max(n_places, 2)))))
    return textwrap.dedent(f"""\
        module bitstream_decoder #(
            parameter N_PLACES  = {n_places},
            parameter BW        = {bit_width},
            parameter ADDR_BITS = {addr_bits}
        )(
            input  wire                  clk,
            input  wire                  rst_n,
            input  wire [N_PLACES-1:0]   place_tokens,
            input  wire [ADDR_BITS-1:0]  place_sel,
            output reg  signed [BW-1:0]  current_out
        );

            // Direct token-to-current mapping: token=1 -> max positive current
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    current_out <= {bit_width}'d0;
                else
                    current_out <= place_tokens[place_sel] ?
                                   {bit_width}'sd{(1 << (_FRAC_BITS[bit_width])) - 1} :
                                   {bit_width}'d0;
            end
        endmodule
    """)


def _verilog_readout(n_neurons: int, bit_width: int) -> str:
    """Spike-to-action readout: weighted sum of output spikes."""
    acc_width = bit_width + int(math.ceil(math.log2(max(n_neurons, 2))))
    idx_bits = max(1, int(math.ceil(math.log2(max(n_neurons, 2)))))
    return textwrap.dedent(f"""\
        module spike_readout #(
            parameter N_NEURONS = {n_neurons},
            parameter BW        = {bit_width},
            parameter ACC_WIDTH = {acc_width},
            parameter IDX_BITS  = {idx_bits}
        )(
            input  wire                    clk,
            input  wire                    rst_n,
            input  wire [N_NEURONS-1:0]    spikes,
            input  wire signed [BW-1:0]    readout_weights [0:N_NEURONS-1],
            output reg  signed [ACC_WIDTH-1:0] action_out,
            output reg                     valid
        );

            reg signed [ACC_WIDTH-1:0] acc;
            reg [IDX_BITS:0]           idx;
            reg                        busy;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    acc        <= 0;
                    idx        <= 0;
                    busy       <= 1'b0;
                    action_out <= 0;
                    valid      <= 1'b0;
                end else if (!busy) begin
                    acc   <= 0;
                    idx   <= 0;
                    busy  <= 1'b1;
                    valid <= 1'b0;
                end else if (idx < N_NEURONS) begin
                    if (spikes[idx[IDX_BITS-1:0]])
                        acc <= acc + {{{{(ACC_WIDTH-BW){{readout_weights[idx[IDX_BITS-1:0]]][BW-1]}}}},
                                       readout_weights[idx[IDX_BITS-1:0]]}};
                    idx <= idx + 1;
                end else begin
                    action_out <= acc;
                    valid      <= 1'b1;
                    busy       <= 1'b0;
                end
            end
        endmodule
    """)


def _verilog_top(net: CompiledNet, config: FPGAConfig) -> str:
    """Assemble all sub-modules into a top-level module."""
    n = net.n_transitions
    bw = config.lif_bit_width
    leak = _leak_shift(net.lif_tau_mem, net.lif_dt)
    rom_depth = n * n
    rom_addr_bits = max(1, int(math.ceil(math.log2(max(rom_depth, 2)))))
    place_addr_bits = max(1, int(math.ceil(math.log2(max(net.n_places, 2)))))
    idx_bits = max(1, int(math.ceil(math.log2(max(n, 2)))))
    acc_width = bw + int(math.ceil(math.log2(max(n, 2))))

    submodules = [
        _verilog_lif_neuron(bw, leak),
        _verilog_weight_rom(n, bw),
        _verilog_bitstream_decoder(net.n_places, bw),
        _verilog_readout(n, bw),
    ]

    top = textwrap.dedent(f"""\
        // Auto-generated by scpn_control FPGA export
        // N_NEURONS={n}, N_PLACES={net.n_places}, BW={bw}, CLK={config.clock_mhz} MHz
        // Target: {config.target}

        module scpn_top #(
            parameter N_NEURONS = {n},
            parameter N_PLACES  = {net.n_places},
            parameter BW        = {bw}
        )(
            input  wire             clk,
            input  wire             rst_n,
            input  wire [N_PLACES-1:0] place_tokens,
            output wire signed [{acc_width}-1:0] action_out,
            output wire             action_valid
        );

            wire [N_NEURONS-1:0] spikes;
            wire signed [BW-1:0] w_data;
            reg  [{rom_addr_bits}-1:0] w_addr;

            weight_rom #(
                .DEPTH({rom_depth}),
                .BW(BW),
                .ADDR_BITS({rom_addr_bits})
            ) u_wrom (
                .clk(clk),
                .addr(w_addr),
                .data(w_data)
            );

            genvar gi;
            generate
                for (gi = 0; gi < N_NEURONS; gi = gi + 1) begin : gen_lif
                    lif_neuron #(
                        .BW(BW),
                        .LEAK_SHIFT({leak})
                    ) u_lif (
                        .clk(clk),
                        .rst_n(rst_n),
                        .i_syn({bw}'d0),
                        .v_threshold({bw}'d{_quantize_threshold(float(net.thresholds[0]) if n > 0 else 0.5, bw)}),
                        .spike(spikes[gi]),
                        .v_mem()
                    );
                end
            endgenerate

            spike_readout #(
                .N_NEURONS(N_NEURONS),
                .BW(BW),
                .ACC_WIDTH({acc_width}),
                .IDX_BITS({idx_bits})
            ) u_readout (
                .clk(clk),
                .rst_n(rst_n),
                .spikes(spikes),
                .readout_weights(),
                .action_out(action_out),
                .action_valid(action_valid)
            );

        endmodule
    """)

    return "\n".join(submodules) + "\n" + top


# ── VHDL generation ──────────────────────────────────────────────────────────


def _vhdl_lif_neuron(bit_width: int, leak_shift: int) -> str:
    bw = bit_width
    return textwrap.dedent(f"""\
        library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;

        entity lif_neuron is
            generic (
                BW         : integer := {bw};
                LEAK_SHIFT : integer := {leak_shift}
            );
            port (
                clk         : in  std_logic;
                rst_n       : in  std_logic;
                i_syn       : in  signed(BW-1 downto 0);
                v_threshold : in  unsigned(BW-1 downto 0);
                spike       : out std_logic;
                v_mem       : out signed(BW-1 downto 0)
            );
        end entity lif_neuron;

        architecture rtl of lif_neuron is
            signal v_reg   : signed(BW-1 downto 0) := (others => '0');
            signal leak    : signed(BW-1 downto 0);
            signal v_next  : signed(BW downto 0);
            signal fire    : std_logic;
        begin
            leak    <= shift_right(v_reg, LEAK_SHIFT);
            v_next  <= resize(v_reg, BW+1) - resize(leak, BW+1) + resize(i_syn, BW+1);
            fire    <= '1' when unsigned(v_next(BW-1 downto 0)) >= v_threshold else '0';
            spike   <= fire;
            v_mem   <= v_reg;

            process(clk, rst_n)
            begin
                if rst_n = '0' then
                    v_reg <= (others => '0');
                elsif rising_edge(clk) then
                    if fire = '1' then
                        v_reg <= (others => '0');
                    elsif v_next(BW) = '1' then
                        v_reg <= (others => '0');
                    else
                        v_reg <= v_next(BW-1 downto 0);
                    end if;
                end if;
            end process;
        end architecture rtl;
    """)


def _vhdl_weight_rom(n_neurons: int, bit_width: int) -> str:
    depth = n_neurons * n_neurons
    addr_bits = max(1, int(math.ceil(math.log2(max(depth, 2)))))
    return textwrap.dedent(f"""\
        library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;

        entity weight_rom is
            generic (
                DEPTH     : integer := {depth};
                BW        : integer := {bit_width};
                ADDR_BITS : integer := {addr_bits}
            );
            port (
                clk  : in  std_logic;
                addr : in  unsigned(ADDR_BITS-1 downto 0);
                data : out signed(BW-1 downto 0)
            );
        end entity weight_rom;

        architecture rtl of weight_rom is
            type rom_t is array(0 to DEPTH-1) of signed(BW-1 downto 0);
            signal mem : rom_t := (others => (others => '0'));
        begin
            process(clk)
            begin
                if rising_edge(clk) then
                    data <= mem(to_integer(addr));
                end if;
            end process;
        end architecture rtl;
    """)


def compile_to_verilog(compiled_net: CompiledNet, config: FPGAConfig) -> str:
    """Generate synthesisable Verilog HDL for the LIF neuron array."""
    if compiled_net.n_transitions > config.n_neurons_max:
        raise ValueError(
            f"Net has {compiled_net.n_transitions} transitions, exceeds n_neurons_max={config.n_neurons_max}"
        )
    return _verilog_top(compiled_net, config)


def compile_to_vhdl(compiled_net: CompiledNet, config: FPGAConfig) -> str:
    """Generate synthesisable VHDL for the LIF neuron array."""
    if compiled_net.n_transitions > config.n_neurons_max:
        raise ValueError(
            f"Net has {compiled_net.n_transitions} transitions, exceeds n_neurons_max={config.n_neurons_max}"
        )
    bw = config.lif_bit_width
    leak = _leak_shift(compiled_net.lif_tau_mem, compiled_net.lif_dt)
    parts = [
        _vhdl_lif_neuron(bw, leak),
        _vhdl_weight_rom(compiled_net.n_transitions, bw),
    ]
    return "\n".join(parts)


def estimate_resources(compiled_net: CompiledNet, config: FPGAConfig) -> dict:
    """Estimate FPGA resource usage for the compiled net.

    Returns dict with keys: lut_count, ff_count, bram_blocks, estimated_fmax_mhz.
    """
    n = compiled_net.n_transitions
    bw = config.lif_bit_width

    # LIF neuron: ~3*BW LUTs for adder/subtractor/comparator, ~2*BW FFs for v_mem + spike
    lut_per_neuron = 3 * bw
    ff_per_neuron = 2 * bw + 1

    rom_depth = n * n
    # Xilinx BRAM18 holds 18Kbit; each entry is bw bits
    bits_total = rom_depth * bw
    bram_blocks = max(1, int(math.ceil(bits_total / 18_432)))

    # Readout accumulator: ~2*acc_width LUTs, ~acc_width FFs
    acc_width = bw + int(math.ceil(math.log2(max(n, 2))))
    readout_luts = 2 * acc_width
    readout_ffs = acc_width + int(math.ceil(math.log2(max(n, 2)))) + 2

    lut_count = n * lut_per_neuron + readout_luts
    ff_count = n * ff_per_neuron + readout_ffs

    # Fmax estimate: critical path through adder chain.  Rough model:
    # 600 MHz / (BW / 8) for modern FPGAs.
    estimated_fmax = min(config.clock_mhz, 600.0 / max(1, bw / 8))

    return {
        "lut_count": lut_count,
        "ff_count": ff_count,
        "bram_blocks": bram_blocks,
        "estimated_fmax_mhz": round(estimated_fmax, 1),
        "n_neurons": n,
        "bit_width": bw,
        "weight_rom_depth": rom_depth,
    }


def _write_weights_mem(compiled_net: CompiledNet, config: FPGAConfig, path: Path) -> None:
    """Write weight ROM initialisation file in hex format.

    Layout: W_in flattened row-major (nT x nP zero-padded to nT x nT).
    """
    n = compiled_net.n_transitions
    bw = config.lif_bit_width
    hw = _hex_width(bw)
    mask = (1 << bw) - 1

    lines: list[str] = []
    W_in = compiled_net.W_in
    for i in range(n):
        for j in range(n):
            if j < compiled_net.n_places:
                q = _quantize_weight(float(W_in[i, j]), bw)
            else:
                q = 0
            lines.append(f"{q & mask:0{hw}x}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_constraints_xdc(config: FPGAConfig, path: Path) -> None:
    """Xilinx XDC timing constraints."""
    period_ns = 1000.0 / config.clock_mhz
    path.write_text(
        f"create_clock -period {period_ns:.3f} -name sys_clk [get_ports clk]\n",
        encoding="utf-8",
    )


def _write_constraints_sdc(config: FPGAConfig, path: Path) -> None:
    """Intel/Lattice SDC timing constraints."""
    period_ns = 1000.0 / config.clock_mhz
    path.write_text(
        f"create_clock -period {period_ns:.3f} -name sys_clk [get_ports {{clk}}]\n",
        encoding="utf-8",
    )


def _write_makefile_vivado(path: Path) -> None:
    path.write_text(
        textwrap.dedent("""\
            .PHONY: synth impl bit clean

            TOP     = scpn_top
            PART    ?= xc7a35tcpg236-1
            SRCDIR  = .

            synth:
            \tvivado -mode batch -source synth.tcl

            clean:
            \trm -rf .Xil *.jou *.log *.bit
        """),
        encoding="utf-8",
    )


def _write_makefile_quartus(path: Path) -> None:
    path.write_text(
        textwrap.dedent("""\
            .PHONY: synth clean

            TOP     = scpn_top
            FAMILY  ?= "Cyclone V"

            synth:
            \tquartus_sh --flow compile $(TOP)

            clean:
            \trm -rf db incremental_db output_files *.rpt *.sof
        """),
        encoding="utf-8",
    )


def export_bitstream_project(
    compiled_net: CompiledNet,
    config: FPGAConfig,
    output_dir: Path,
) -> None:
    """Write a complete FPGA project directory.

    Creates:
        top.v / top.vhd — HDL source
        weights.mem     — weight ROM init
        constraints.xdc / constraints.sdc — timing
        Makefile        — synthesis flow
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    is_vhdl = config.target == "lattice"

    if is_vhdl:
        hdl = compile_to_vhdl(compiled_net, config)
        (output_dir / "top.vhd").write_text(hdl, encoding="utf-8")
    else:
        hdl = compile_to_verilog(compiled_net, config)
        (output_dir / "top.v").write_text(hdl, encoding="utf-8")

    _write_weights_mem(compiled_net, config, output_dir / "weights.mem")

    if config.target == "xilinx":
        _write_constraints_xdc(config, output_dir / "constraints.xdc")
        _write_makefile_vivado(output_dir / "Makefile")
    else:
        _write_constraints_sdc(config, output_dir / "constraints.sdc")
        _write_makefile_quartus(output_dir / "Makefile")
