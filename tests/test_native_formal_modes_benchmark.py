# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Formal Modes Benchmark Tests

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_native_formal_modes",
        REPO_ROOT / "scripts" / "benchmark_native_formal_modes.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_preserves_aot_certificate_admission_fields() -> None:
    module = _load_benchmark_module()
    formal = {
        "generated": 10,
        "submitted": 10,
        "checked": 10,
        "dropped": 0,
        "failures": 0,
        "certificate_admitted": True,
        "certificate_schema_version": "scpn-control.native-formal.aot-certificate.v1",
        "certificate_id": "bounded-petri-marking-sufficient-invariant",
        "certificate_assumption_sha256": "a" * 64,
        "sync_wait_count": 0,
        "sync_wait_p99_ns": 0,
    }
    rows = [
        {
            "avg_cycle_us": 1.0,
            "effective_step_us": 100.0,
            "dropped": 0,
            "publish_failures": 0,
            "udp_sink_packets": 10,
            "native": {"formal_verification": dict(formal)},
        },
        {
            "avg_cycle_us": 2.0,
            "effective_step_us": 100.0,
            "dropped": 0,
            "publish_failures": 0,
            "udp_sink_packets": 10,
            "native": {"formal_verification": dict(formal)},
        },
    ]

    summary = module._summarise(rows)

    assert summary["certificate_admitted_total"] == 2
    assert summary["certificate_schema_versions"] == [
        "scpn-control.native-formal.aot-certificate.v1",
    ]
    assert summary["certificate_ids"] == [
        "bounded-petri-marking-sufficient-invariant",
    ]
    assert summary["certificate_assumption_sha256_values"] == ["a" * 64]
