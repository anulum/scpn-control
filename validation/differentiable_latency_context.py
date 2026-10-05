# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Declared local differentiable latency context contracts; no audit replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from validation.differentiable_latency_fields import _finite_non_negative, _finite_positive, _require_value


def _validate_latency_order(path: Path, payload: dict[str, Any], errors: list[dict[str, object]]) -> None:
    """Require finite nonnegative millisecond p50/p95/max declarations in order.

    Parameters
    ----------
    path : pathlib.Path
        Report path for diagnostics.
    payload : dict[str, Any]
        Mapping containing p50_ms, p95_ms and max_ms.
    errors : list[dict[str, object]]
        Ordered diagnostics appended in place.

    Returns
    -------
    None
        Each metric may be zero and ordering is inclusive. No budget, sample
        distribution, percentile estimator or measured timing is checked.
    """
    p50 = _finite_non_negative(payload.get("p50_ms"))
    p95 = _finite_non_negative(payload.get("p95_ms"))
    max_ms = _finite_non_negative(payload.get("max_ms"))
    if p50 is None:
        errors.append({"path": str(path), "field": "p50_ms", "error": "p50_ms must be finite and non-negative"})
    if p95 is None:
        errors.append({"path": str(path), "field": "p95_ms", "error": "p95_ms must be finite and non-negative"})
    if max_ms is None:
        errors.append({"path": str(path), "field": "max_ms", "error": "max_ms must be finite and non-negative"})
    if p50 is not None and p95 is not None and max_ms is not None and not (p50 <= p95 <= max_ms):
        errors.append(
            {
                "path": str(path),
                "field": "latency",
                "error": "latency percentiles must satisfy p50_ms <= p95_ms <= max_ms",
            }
        )


def _validate_runtime_metadata(path: Path, metadata: object, errors: list[dict[str, object]]) -> None:
    """Check required runtime provenance declarations without inspecting that host.

    Parameters
    ----------
    path : pathlib.Path
        Report path for diagnostics.
    metadata : object
        Expected schema1 mapping of timestamp/version/platform/device fields.
    errors : list[dict[str, object]]
        Ordered diagnostics appended in place.

    Returns
    -------
    None
        Require positive finite timestamp, nonempty version/platform/machine/
        backend strings, string processor, nonempty string device list and
        boolean x64 state. False x64 and whitespace-only strings retain the
        original declaration policy.

    Notes
    -----
    No freshness, installed version, device identity, x64 execution or actual
    backend is authenticated. Declared metadata does not qualify performance.
    """
    if not isinstance(metadata, dict):
        errors.append({"path": str(path), "field": "runtime_metadata", "error": "runtime metadata must be an object"})
        return
    _require_value(path, metadata, "schema_version", 1, errors)
    measured = _finite_positive(metadata.get("measured_at_unix_s"))
    if measured is None:
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.measured_at_unix_s",
                "error": "measurement timestamp must be positive and finite",
            }
        )
    for field in (
        "python_version",
        "platform",
        "machine",
        "jax_version",
        "jaxlib_version",
        "jax_default_backend",
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            errors.append(
                {
                    "path": str(path),
                    "field": f"runtime_metadata.{field}",
                    "error": "field must be a non-empty string",
                }
            )
    if not isinstance(metadata.get("processor"), str):
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.processor",
                "error": "field must be a string",
            }
        )
    devices = metadata.get("jax_devices")
    if (
        not isinstance(devices, list)
        or not devices
        or not all(isinstance(device, str) and device for device in devices)
    ):
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.jax_devices",
                "error": "field must be a non-empty string list",
            }
        )
    if not isinstance(metadata.get("jax_enable_x64"), bool):
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.jax_enable_x64",
                "error": "field must be boolean",
            }
        )
