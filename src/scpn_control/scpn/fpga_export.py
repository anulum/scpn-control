# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fpga Export
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — FPGA HDL Project Export
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Export a CompiledNet to generated HDL project files for FPGA toolchains.

Generates a bounded project directory with HDL source, weight ROM
initialisation, timing constraints, and a Makefile for Vivado or Quartus. The
repository does not claim to emit a device bitstream; hardware claims require
separate synthesis, timing, and bitstream evidence through
``HDLExportEvidence``.
"""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .compiler import CompiledNet

VALID_TARGETS = ("xilinx", "intel", "lattice")
VALID_BIT_WIDTHS = (8, 16, 32)
HDL_EXPORT_EVIDENCE_SCHEMA_VERSION = "scpn-control.hdl-export-evidence.v1"
HDL_EXPORT_EVIDENCE_LOCAL_ONLY = "generated_hdl_export_evidence_only"
HDL_EXPORT_EVIDENCE_SYNTHESIS_QUALIFIED = "qualified_hdl_synthesis_evidence"
QUALIFIED_SYNTHESIS_TOOLCHAINS = ("vivado", "quartus", "radiant", "yosys-nextpnr")

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


@dataclass(frozen=True)
class HDLExportEvidence:
    """Tamper-evident FPGA HDL export and synthesis-admission evidence."""

    schema_version: str
    generated_utc: str
    controller_artifact_sha256: str
    target: str
    target_part: str
    clock_mhz: float
    lif_bit_width: int
    n_neurons: int
    n_places: int
    bitstream_length: int
    hdl_format: str
    hdl_sha256: str
    weights_mem_sha256: str
    constraints_sha256: str
    makefile_sha256: str
    project_sha256: str
    resource_estimate_sha256: str
    estimated_lut_count: int
    estimated_ff_count: int
    estimated_bram_blocks: int
    weight_rom_depth: int
    estimated_fmax_mhz: float
    synthesis_toolchain: str | None
    synthesis_report_sha256: str | None
    synthesis_report_uri: str | None
    timing_slack_ns: float | None
    facility_claim_allowed: bool
    claim_status: str
    payload_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in HDL export evidence: {key}")
        seen.add(key)
        out[key] = value
    return out


def _safe_relative_uri(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty safe relative path")
    if "\\" in value or "://" in value or "//" in value or value.startswith(("file:", "/", "~")):
        raise ValueError(f"{name} must be a safe relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"{name} must be a safe relative path")
    return value


def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _project_file_paths(project_dir: Path, config: FPGAConfig) -> dict[str, Path]:
    hdl_name = "top.vhd" if config.target == "lattice" else "top.v"
    constraints_name = "constraints.xdc" if config.target == "xilinx" else "constraints.sdc"
    return {
        "hdl": project_dir / hdl_name,
        "weights_mem": project_dir / "weights.mem",
        "constraints": project_dir / constraints_name,
        "makefile": project_dir / "Makefile",
    }


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"HDL export project file is missing: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_digest(resources: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(resources)).encode("utf-8")).hexdigest()


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_synthesis_report(
    synthesis_report_uri: str,
    synthesis_report_sha256: str,
    artifact_root: str | Path,
) -> None:
    rel = _safe_relative_uri("synthesis_report_uri", synthesis_report_uri)
    root = Path(artifact_root).resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("synthesis_report_uri escapes artifact_root") from exc
    if not candidate.is_file():
        raise ValueError("synthesis_report_uri does not resolve to a file")
    observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if observed != synthesis_report_sha256:
        raise ValueError("synthesis_report_sha256 does not match synthesis report bytes")


def _validate_hdl_export_payload(
    payload: Mapping[str, Any],
    *,
    require_facility_claim: bool,
    artifact_root: str | Path | None = None,
) -> HDLExportEvidence:
    if payload.get("schema_version") != HDL_EXPORT_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("HDL export evidence schema_version is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not _is_sha256(declared_digest):
        raise ValueError("HDL export evidence payload_sha256 must be a SHA-256 hex digest")
    if declared_digest != _payload_sha256(payload):
        raise ValueError("HDL export evidence payload_sha256 does not match payload")
    generated_utc = payload.get("generated_utc")
    if not isinstance(generated_utc, str) or not generated_utc.endswith("Z"):
        raise ValueError("HDL export evidence generated_utc must be a UTC timestamp ending in Z")
    controller_artifact_sha256 = payload.get("controller_artifact_sha256")
    if not _is_sha256(controller_artifact_sha256):
        raise ValueError("controller_artifact_sha256 must be a SHA-256 hex digest")
    target = payload.get("target")
    if target not in VALID_TARGETS:
        raise ValueError("HDL export evidence target is unsupported")
    target_part = payload.get("target_part")
    if not isinstance(target_part, str) or not target_part.strip():
        raise ValueError("target_part must be a non-empty string")
    clock_mhz = _require_finite_nonnegative("clock_mhz", payload.get("clock_mhz"))
    if clock_mhz <= 0.0:
        raise ValueError("clock_mhz must be positive")
    lif_bit_width = payload.get("lif_bit_width")
    if isinstance(lif_bit_width, bool) or lif_bit_width not in VALID_BIT_WIDTHS:
        raise ValueError("lif_bit_width must be a supported FPGA bit width")
    n_neurons = _require_positive_integer("n_neurons", payload.get("n_neurons"))
    n_places = _require_positive_integer("n_places", payload.get("n_places"))
    bitstream_length = _require_positive_integer("bitstream_length", payload.get("bitstream_length"))
    hdl_format = payload.get("hdl_format")
    if hdl_format not in {"verilog", "vhdl"}:
        raise ValueError("hdl_format must be verilog or vhdl")
    for name in (
        "hdl_sha256",
        "weights_mem_sha256",
        "constraints_sha256",
        "makefile_sha256",
        "project_sha256",
        "resource_estimate_sha256",
    ):
        if not _is_sha256(payload.get(name)):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    estimated_lut_count = _require_positive_integer("estimated_lut_count", payload.get("estimated_lut_count"))
    estimated_ff_count = _require_positive_integer("estimated_ff_count", payload.get("estimated_ff_count"))
    estimated_bram_blocks = _require_positive_integer("estimated_bram_blocks", payload.get("estimated_bram_blocks"))
    weight_rom_depth = _require_positive_integer("weight_rom_depth", payload.get("weight_rom_depth"))
    estimated_fmax_mhz = _require_finite_nonnegative("estimated_fmax_mhz", payload.get("estimated_fmax_mhz"))
    if estimated_fmax_mhz <= 0.0:
        raise ValueError("estimated_fmax_mhz must be positive")
    expected_project_digest = hashlib.sha256(
        _canonical_json(
            {
                "target": target,
                "clock_mhz": clock_mhz,
                "lif_bit_width": int(lif_bit_width),
                "files": {
                    "constraints": payload["constraints_sha256"],
                    "hdl": payload["hdl_sha256"],
                    "makefile": payload["makefile_sha256"],
                    "weights_mem": payload["weights_mem_sha256"],
                },
            }
        ).encode("utf-8")
    ).hexdigest()
    if payload["project_sha256"] != expected_project_digest:
        raise ValueError("project_sha256 does not match HDL export file digest bundle")
    expected_resource_digest = _resource_digest(
        {
            "bit_width": int(lif_bit_width),
            "bram_blocks": int(estimated_bram_blocks),
            "estimated_fmax_mhz": estimated_fmax_mhz,
            "ff_count": int(estimated_ff_count),
            "lut_count": int(estimated_lut_count),
            "n_neurons": int(n_neurons),
            "weight_rom_depth": int(weight_rom_depth),
        }
    )
    if payload["resource_estimate_sha256"] != expected_resource_digest:
        raise ValueError("resource_estimate_sha256 does not match HDL resource evidence")

    synthesis_toolchain = payload.get("synthesis_toolchain")
    synthesis_report_sha256 = payload.get("synthesis_report_sha256")
    synthesis_report_uri = payload.get("synthesis_report_uri")
    timing_slack_ns_raw = payload.get("timing_slack_ns")
    if synthesis_toolchain is not None and synthesis_toolchain not in QUALIFIED_SYNTHESIS_TOOLCHAINS:
        raise ValueError("synthesis_toolchain is unsupported")
    if synthesis_report_sha256 is not None and not _is_sha256(synthesis_report_sha256):
        raise ValueError("synthesis_report_sha256 must be null or a SHA-256 hex digest")
    if synthesis_report_uri is not None:
        if not isinstance(synthesis_report_uri, str):
            raise ValueError("synthesis_report_uri must be null or a safe relative path")
        _safe_relative_uri("synthesis_report_uri", synthesis_report_uri)
    timing_slack_ns = None
    if timing_slack_ns_raw is not None:
        timing_slack_ns = _require_finite_nonnegative("timing_slack_ns", timing_slack_ns_raw)

    facility_claim_allowed = payload.get("facility_claim_allowed")
    claim_status = payload.get("claim_status")
    if not isinstance(facility_claim_allowed, bool):
        raise ValueError("facility_claim_allowed must be boolean")
    expected_status = (
        HDL_EXPORT_EVIDENCE_SYNTHESIS_QUALIFIED if facility_claim_allowed else HDL_EXPORT_EVIDENCE_LOCAL_ONLY
    )
    if claim_status != expected_status:
        raise ValueError("HDL export evidence claim_status does not match facility_claim_allowed")
    if require_facility_claim:
        if not facility_claim_allowed:
            raise ValueError("HDL export evidence is local-only and cannot support a facility claim")
        if synthesis_toolchain is None or synthesis_report_sha256 is None or synthesis_report_uri is None:
            raise ValueError("qualified HDL evidence requires synthesis toolchain, report URI, and report digest")
        if timing_slack_ns is None:
            raise ValueError("qualified HDL evidence requires non-negative timing slack")
        if artifact_root is not None:
            _resolve_synthesis_report(synthesis_report_uri, synthesis_report_sha256, artifact_root)

    return HDLExportEvidence(
        schema_version=str(payload["schema_version"]),
        generated_utc=generated_utc,
        controller_artifact_sha256=str(controller_artifact_sha256),
        target=str(target),
        target_part=str(target_part),
        clock_mhz=clock_mhz,
        lif_bit_width=int(lif_bit_width),
        n_neurons=int(n_neurons),
        n_places=int(n_places),
        bitstream_length=int(bitstream_length),
        hdl_format=str(hdl_format),
        hdl_sha256=str(payload["hdl_sha256"]),
        weights_mem_sha256=str(payload["weights_mem_sha256"]),
        constraints_sha256=str(payload["constraints_sha256"]),
        makefile_sha256=str(payload["makefile_sha256"]),
        project_sha256=str(payload["project_sha256"]),
        resource_estimate_sha256=str(payload["resource_estimate_sha256"]),
        estimated_lut_count=int(estimated_lut_count),
        estimated_ff_count=int(estimated_ff_count),
        estimated_bram_blocks=int(estimated_bram_blocks),
        weight_rom_depth=int(weight_rom_depth),
        estimated_fmax_mhz=estimated_fmax_mhz,
        synthesis_toolchain=None if synthesis_toolchain is None else str(synthesis_toolchain),
        synthesis_report_sha256=None if synthesis_report_sha256 is None else str(synthesis_report_sha256),
        synthesis_report_uri=None if synthesis_report_uri is None else str(synthesis_report_uri),
        timing_slack_ns=timing_slack_ns,
        facility_claim_allowed=facility_claim_allowed,
        claim_status=str(claim_status),
        payload_sha256=str(declared_digest),
    )


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


def estimate_resources(compiled_net: CompiledNet, config: FPGAConfig) -> dict[str, float]:
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
    """Write a bounded FPGA HDL project directory.

    Creates:
        top.v / top.vhd — HDL source
        weights.mem     — weight ROM init
        constraints.xdc / constraints.sdc — timing
        Makefile        — synthesis flow

    The historical function name is retained for API compatibility. This
    function does not run a vendor toolchain or emit a device bitstream.
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


def hdl_export_evidence(
    compiled_net: CompiledNet,
    config: FPGAConfig,
    project_dir: str | Path,
    *,
    controller_artifact_sha256: str,
    target_part: str,
    generated_utc: str | None = None,
    synthesis_toolchain: str | None = None,
    synthesis_report_sha256: str | None = None,
    synthesis_report_uri: str | None = None,
    timing_slack_ns: float | None = None,
    facility_claim_allowed: bool = False,
) -> HDLExportEvidence:
    """Build tamper-evident FPGA HDL export evidence for a generated project."""

    if not _is_sha256(controller_artifact_sha256):
        raise ValueError("controller_artifact_sha256 must be a SHA-256 hex digest")
    if not isinstance(target_part, str) or not target_part.strip():
        raise ValueError("target_part must be a non-empty string")
    project_path = Path(project_dir)
    files = _project_file_paths(project_path, config)
    digests = {name: _file_sha256(path) for name, path in files.items()}
    resources = estimate_resources(compiled_net, config)
    hdl_format = "vhdl" if config.target == "lattice" else "verilog"
    project_digest_payload = {
        "target": config.target,
        "clock_mhz": float(config.clock_mhz),
        "lif_bit_width": int(config.lif_bit_width),
        "files": {
            "constraints": digests["constraints"],
            "hdl": digests["hdl"],
            "makefile": digests["makefile"],
            "weights_mem": digests["weights_mem"],
        },
    }
    payload: dict[str, Any] = {
        "schema_version": HDL_EXPORT_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": generated_utc or _utc_now(),
        "controller_artifact_sha256": controller_artifact_sha256.lower(),
        "target": config.target,
        "target_part": target_part.strip(),
        "clock_mhz": float(config.clock_mhz),
        "lif_bit_width": int(config.lif_bit_width),
        "n_neurons": int(compiled_net.n_transitions),
        "n_places": int(compiled_net.n_places),
        "bitstream_length": int(compiled_net.bitstream_length),
        "hdl_format": hdl_format,
        "hdl_sha256": digests["hdl"],
        "weights_mem_sha256": digests["weights_mem"],
        "constraints_sha256": digests["constraints"],
        "makefile_sha256": digests["makefile"],
        "project_sha256": hashlib.sha256(_canonical_json(project_digest_payload).encode("utf-8")).hexdigest(),
        "resource_estimate_sha256": _resource_digest(resources),
        "estimated_lut_count": int(resources["lut_count"]),
        "estimated_ff_count": int(resources["ff_count"]),
        "estimated_bram_blocks": int(resources["bram_blocks"]),
        "weight_rom_depth": int(resources["weight_rom_depth"]),
        "estimated_fmax_mhz": float(resources["estimated_fmax_mhz"]),
        "synthesis_toolchain": synthesis_toolchain,
        "synthesis_report_sha256": synthesis_report_sha256,
        "synthesis_report_uri": synthesis_report_uri,
        "timing_slack_ns": timing_slack_ns,
        "facility_claim_allowed": bool(facility_claim_allowed),
        "claim_status": (
            HDL_EXPORT_EVIDENCE_SYNTHESIS_QUALIFIED if facility_claim_allowed else HDL_EXPORT_EVIDENCE_LOCAL_ONLY
        ),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _validate_hdl_export_payload(payload, require_facility_claim=bool(facility_claim_allowed))


def assert_hdl_export_claim_admissible(
    evidence: HDLExportEvidence,
    *,
    artifact_root: str | Path | None = None,
) -> HDLExportEvidence:
    """Fail closed unless HDL export evidence supports a facility hardware claim."""

    return _validate_hdl_export_payload(
        asdict(evidence),
        require_facility_claim=True,
        artifact_root=artifact_root,
    )


def save_hdl_export_evidence(evidence: HDLExportEvidence, output_path: str | Path) -> None:
    """Persist HDL export evidence as sorted JSON."""

    if not isinstance(evidence, HDLExportEvidence):
        raise ValueError("evidence must be HDLExportEvidence")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_hdl_export_evidence(
    path: str | Path,
    *,
    require_facility_claim: bool = False,
    artifact_root: str | Path | None = None,
) -> HDLExportEvidence:
    """Load HDL export evidence with duplicate-key and digest admission."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("HDL export evidence must be a JSON object")
    return _validate_hdl_export_payload(
        payload,
        require_facility_claim=require_facility_claim,
        artifact_root=artifact_root,
    )
