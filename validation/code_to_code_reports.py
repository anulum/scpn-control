# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Code-to-code report declarations and admission limits.

"""Bind diagnostic comparison reports while retaining unresolved physics limits.

The schema describes declared inputs and finite arithmetic, not authenticated
facility measurements. The configured physics models remain different even
after their initial profile and fixed-timestep controls have been aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from validation.code_to_code_comparison import _benchmark_numeric_payload_is_finite, _finite_number, _sha256_payload
from validation.code_to_code_scenario import validate_scenario

REPORT_SCHEMA_VERSION = "scpn-control.code-to-code-benchmark.v3"
REPORT_PATH = Path("validation/reports/code_to_code_benchmark.json")
MARKDOWN_REPORT_PATH = Path("validation/reports/code_to_code_benchmark.md")
PHYSICS_LIMITATIONS = (
    "transport_closures_not_matched",
    "current_geometry_not_matched",
    "source_channels_not_matched",
)


def _external_reference_status(comparison: dict[str, Any], *, requested_torax: bool) -> dict[str, Any]:
    """Classify declared payload diagnostics without promoting unmatched physics.

    Not-requested and unavailable providers retain their prior statuses/reasons.
    Identity labels, finite values and nonempty metrics are checked. Even a
    structurally valid comparison retains the concrete configured-physics
    limitations; this adapter cannot admit a physical transport reference.
    The diagnostic_comparison_available flag denotes usable declared metrics,
    not independent proof of an actual TORAX execution.
    """
    torax = comparison.get("torax")
    metrics = comparison.get("comparison", {})
    reasons: list[str] = []
    status = "blocked"
    diagnostic = False
    if not requested_torax:
        reasons.append("torax_not_requested")
        status = "not_requested"
    elif torax is None:
        reasons.append("torax_not_available_or_failed")
    elif not isinstance(torax, dict) or torax.get("code") != "torax":
        reasons.append("torax_payload_identity")
    elif not _benchmark_numeric_payload_is_finite(torax):
        reasons.append("torax_numeric_payload")
    elif not isinstance(metrics, dict) or not metrics:
        reasons.append("comparison_metrics_missing")
    elif any(not _finite_number(value) for value in metrics.values()):
        reasons.append("comparison_metrics_non_finite")
    else:
        diagnostic = True
        reasons.extend(PHYSICS_LIMITATIONS)
    scpn = comparison.get("scpn_control")
    if not isinstance(scpn, dict) or scpn.get("code") != "scpn-control":
        reasons.append("scpn_payload_identity")
        diagnostic = False
    elif not _benchmark_numeric_payload_is_finite(scpn):
        reasons.append("scpn_numeric_payload")
        diagnostic = False
    return {
        "provider": "TORAX",
        "artifact_kind": "code_to_code_transport_reference",
        "requested": requested_torax,
        "admitted": False,
        "status": status,
        "diagnostic_comparison_available": diagnostic,
        "blocked_reasons": reasons,
    }


def build_comparison_report(
    comparison: dict[str, Any], scenario: dict[str, Any], *, requested_torax: bool
) -> dict[str, Any]:
    """Bind declared inputs and comparison arithmetic to a schema-v3 report.

    Parameters
    ----------
    comparison : dict
        Payload from compare_transport_profiles; supplied values remain caller
        declarations. No provider authentication is inferred.
    scenario : dict
        Required high-level fields accepted by validate_scenario.
    requested_torax : bool
        Whether the producing command requested the optional provider.

    Returns
    -------
    dict
        Canonical scenario/payload SHA-256 bindings, benchmark, status/reasons
        and the configured model differences that prevent physical admission.

    Raises
    ------
    TypeError, ValueError
        Scenario validation or canonical JSON encoding refuses input.

    Notes
    -----
    A digest authenticates no source and is not an independent physical audit.
    Local/optional transport closures, current/geometry and source channels
    differ. Availability of finite diagnostic metrics does not repair them.
    """
    if not isinstance(requested_torax, bool):
        raise ValueError("requested_torax must be a boolean")
    scenario = validate_scenario(scenario)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scenario_name": scenario["name"],
        "scenario_sha256": _sha256_payload(scenario),
        "scenario": scenario,
        "external_reference": _external_reference_status(comparison, requested_torax=requested_torax),
        "benchmark": comparison,
        "model_contract": {
            "local_transport": "gyro_bohm",
            "torax_transport": "constant",
            "local_unmapped_fields": ["B0", "delta"],
            "limitations": list(PHYSICS_LIMITATIONS),
        },
        "claim_boundary": (
            "Finite profile comparisons are declared arithmetic diagnostics. "
            "Configured transport, current/geometry and source models differ; "
            "physical external-reference admission remains blocked. "
            "Self-digests do not authenticate a provider or independent reference."
        ),
    }
    report["payload_sha256"] = _sha256_payload(report)
    return report


_build_external_reference_report = build_comparison_report


def _write_markdown_report(report: dict[str, Any], path: Path = MARKDOWN_REPORT_PATH) -> None:
    """Write the declared status, digests and diagnostic metrics as UTF-8 Markdown.

    Parent directories are created; existing contents are overwritten. No
    validation, transaction or source authentication occurs here. OSError and
    malformed-report key/type errors propagate. The CLI guards output aliases
    and recorded-campaign custody before any solver run.
    """
    external = report["external_reference"]
    metrics = report["benchmark"].get("comparison", {})
    metric_lines = (
        [f"- `{name}`: `{value}`" for name, value in sorted(metrics.items())]
        if metrics
        else ["- No comparison metrics admitted."]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# TORAX Code-to-Code Benchmark Evidence",
                "",
                f"- Schema: `{report['schema_version']}`",
                f"- Scenario: `{report['scenario_name']}`",
                f"- Scenario digest: `{report['scenario_sha256']}`",
                f"- External provider: `{external['provider']}`",
                f"- External status: `{external['status']}`",
                f"- External admitted: `{external['admitted']}`",
                f"- Blocked reasons: `{', '.join(external['blocked_reasons']) or 'none'}`",
                f"- Payload digest: `{report['payload_sha256']}`",
                f"- Claim boundary: {report['claim_boundary']}",
                "",
                "## Comparison metrics",
                "",
                *metric_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
