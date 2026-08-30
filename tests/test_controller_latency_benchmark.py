# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — controller latency benchmark contract tests.

"""Ensure the H-infinity polyglot lane compares one controller realization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_benchmark() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "controller_latency.py"
    spec = importlib.util.spec_from_file_location("controller_latency_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load controller latency benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h_infinity_entries_declare_same_normalized_realization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both latency rows declare the same normalized DGKF realization."""
    module = _load_benchmark()

    def deterministic_measure(step: Any, *, iterations: int, warmup: int) -> dict[str, float]:
        for sample in range(warmup + iterations):
            step(sample)
        return {"median_us": 1.0, "p95_us": 1.0, "p99_us": 1.0, "throughput_hz": 1.0}

    monkeypatch.setattr(module, "_measure", deterministic_measure)
    entries = module._h_infinity_entries(iterations=4, warmup=2)
    numpy_entry = next(entry for entry in entries if entry["backend"] == "numpy")
    assert numpy_entry["status"] == "measured"
    assert numpy_entry["note"] == "normalized 2-state DGKF realization"
    rust_entry = next(entry for entry in entries if entry["backend"] == "rust")
    if rust_entry["status"] == "measured":
        assert rust_entry["note"] == "same normalized 2-state DGKF realization"
    else:
        assert rust_entry["status"].startswith("unavailable")
