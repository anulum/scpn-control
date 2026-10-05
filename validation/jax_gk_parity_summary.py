# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity requirements and report aggregation

"""Preserve original normalized case/backend coverage, admitted-entry multiset and report digest semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from validation.jax_gk_parity_domains import _display_path, _sha256_json


def _normalise_required_values(
    values: tuple[str, ...] | list[str] | set[str] | None, allowed: set[str], label: str
) -> set[str]:
    """Normalise supported requirement names, ignoring blanks and repeats.

    None means no requirement. Each value is converted to text and stripped;
    a nonblank unsupported value raises ValueError rather than a report finding.
    """
    if values is None:
        return set()
    out: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in allowed:
            raise ValueError(f"unsupported required {label}: {text}")
        out.add(text)
    return out


def _validate_required_coverage(
    root: Path,
    entries: list[dict[str, object]],
    errors: list[dict[str, object]],
    *,
    required_cases: set[str],
    required_backends: set[str],
) -> None:
    """Append missing named values and Cartesian pairs from admitted entries.

    Entries have the defining reader's fixed case/backend shape. Both nonempty
    requirements demand every pair; one requirement alone demands its names.
    """
    observed_cases = {str(entry["case"]) for entry in entries}
    observed_backends = {str(entry["backend"]) for entry in entries}
    observed_pairs = {(str(entry["case"]), str(entry["backend"])) for entry in entries}

    for case in sorted(required_cases - observed_cases):
        errors.append(
            {"path": _display_path(root), "field": "required_case", "error": f"missing required case: {case}"}
        )
    for backend in sorted(required_backends - observed_backends):
        errors.append(
            {"path": _display_path(root), "field": "required_backend", "error": f"missing required backend: {backend}"}
        )
    if required_cases and required_backends:
        for case in sorted(required_cases):
            for backend in sorted(required_backends):
                if (case, backend) not in observed_pairs:
                    errors.append(
                        {
                            "path": _display_path(root),
                            "field": "required_case_backend",
                            "error": f"missing required case/backend evidence: {case}/{backend}",
                        }
                    )


def _attach_summary_fields(report: dict[str, Any]) -> None:
    """Sort admitted entries and attach mutable aggregate fields in place.

    Counts/pair lists retain duplicate entries; required pair coverage uses set
    inclusion. Empty drift maxima are None. Entry digests bind a sorted
    multiset of admitted self digests; report digest excludes both self keys.
    """
    entries = sorted(
        report["entries"], key=lambda entry: (str(entry["case"]), str(entry["backend"]), str(entry["path"]))
    )
    report["entries"] = entries
    backend_counts: dict[str, int] = {}
    case_counts: dict[str, int] = {}
    observed_pairs: list[str] = []
    gamma_errors: list[float] = []
    omega_errors: list[float] = []
    payload_digests: list[str] = []
    for entry in entries:
        backend = str(entry["backend"])
        case = str(entry["case"])
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        case_counts[case] = case_counts.get(case, 0) + 1
        observed_pairs.append(f"{case}/{backend}")
        gamma_errors.append(float(entry["gamma_relative_error"]))
        omega_errors.append(float(entry["omega_absolute_error"]))
        payload_digests.append(str(entry["payload_sha256"]))
    required_pairs = [
        f"{case}/{backend}" for case in report["required_cases"] for backend in report["required_backends"]
    ]
    report["backend_counts"] = dict(sorted(backend_counts.items()))
    report["case_counts"] = dict(sorted(case_counts.items()))
    report["observed_case_backend_pairs"] = sorted(observed_pairs)
    report["required_case_backend_pairs"] = required_pairs
    report["complete_required_case_backend_coverage"] = (
        set(required_pairs).issubset(set(observed_pairs)) if required_pairs else None
    )
    report["max_gamma_relative_error"] = max(gamma_errors) if gamma_errors else None
    report["max_omega_absolute_error"] = max(omega_errors) if omega_errors else None
    report["entries_payload_sha256"] = _sha256_json(
        {"payload_sha256_values": sorted(payload_digests)}, include_payload_field=True
    )
    report["report_payload_sha256"] = _sha256_json(report)
