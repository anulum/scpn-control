# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware-in-the-Loop Test Harness

"""Stable public facade for HIL simulation, evidence, and benchmarks."""

from __future__ import annotations

from scpn_control.control.hil_benchmark import run_hil_benchmark, run_hil_benchmark_detailed
from scpn_control.control.hil_demo import HILDemoRunner
from scpn_control.control.hil_evidence import (
    HILBenchmarkResult,
    _control_metrics_from_evidence_source,
    _hil_timing_payload,
    assert_hil_replay_evidence_admissible,
    hil_replay_evidence,
    load_hil_replay_evidence,
    save_hil_replay_evidence,
)
from scpn_control.control.hil_evidence_contracts import (
    HIL_REPLAY_EVIDENCE_BOUNDARY,
    HIL_REPLAY_EVIDENCE_SCHEMA_VERSION,
    _is_hex_sha256,
    _reject_duplicate_json_keys,
    _require_finite_float,
    _require_mapping,
    _require_non_empty_text,
    _require_non_negative_int,
    _require_positive_int,
    _require_qualified_hardware,
    _sha256_json,
    _utc_now_iso,
)
from scpn_control.control.hil_fpga import FPGARegisterMap, FPGASNNExport, SNNNeuronConfig
from scpn_control.control.hil_io import ADCConfig, DACConfig, SensorInterface
from scpn_control.control.hil_loop import ControlLoopMetrics, HILControlLoop, PipelineProfile

__all__ = [
    "ADCConfig",
    "DACConfig",
    "SensorInterface",
    "ControlLoopMetrics",
    "HILControlLoop",
    "PipelineProfile",
    "SNNNeuronConfig",
    "FPGARegisterMap",
    "FPGASNNExport",
    "HILBenchmarkResult",
    "hil_replay_evidence",
    "assert_hil_replay_evidence_admissible",
    "save_hil_replay_evidence",
    "load_hil_replay_evidence",
    "run_hil_benchmark",
    "run_hil_benchmark_detailed",
    "HILDemoRunner",
    "HIL_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "HIL_REPLAY_EVIDENCE_BOUNDARY",
    "_sha256_json",
    "_utc_now_iso",
    "_reject_duplicate_json_keys",
    "_require_mapping",
    "_require_non_empty_text",
    "_require_positive_int",
    "_require_non_negative_int",
    "_require_finite_float",
    "_is_hex_sha256",
    "_require_qualified_hardware",
    "_control_metrics_from_evidence_source",
    "_hil_timing_payload",
]
