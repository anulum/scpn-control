# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Z3 Formal Report I/O

"""Schema-versioned evidence I/O for bounded Z3 formal-verification reports.

This module owns the persistence, payload construction, and strict validation of
the JSON/Markdown evidence artifacts produced by the bounded SMT model checker in
:mod:`scpn_control.scpn.z3_model_checking`. It runs the combined safety and
temporal proof obligations via :func:`verify_z3_formal_contracts`, serialises the
result into a self-digesting schema-versioned payload, and re-validates any loaded
payload against the same contract so tampered or malformed evidence is rejected.

The report surface is deliberately separated from the
:class:`~scpn_control.scpn.z3_model_checking.Z3BoundedModelChecker` engine: the
engine proves obligations, this module turns proofs into durable, verifiable
evidence. The dependency is one-way — this module imports the checker, never the
reverse.

Notes
-----
The emitted schema version is ``scpn-control.z3-formal-report.v2``. The scope and
claim-boundary strings are pinned constants: the evidence is bounded SMT proof for
compiled Petri-net control logic, and explicitly **not** hardware-timing evidence,
PCS certification, or an unbounded liveness proof.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    EventuallyFires,
    FireLeadsToMarking,
    NeverCoMarked,
)
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_model_checking import Z3BoundedModelChecker, Z3ModelCheckingReport

Z3_FORMAL_REPORT_SCHEMA_VERSION = "scpn-control.z3-formal-report.v2"
Z3_FORMAL_REPORT_SCOPE = "bounded SMT evidence for compiled Petri-net control logic"
Z3_FORMAL_REPORT_CLAIM_BOUNDARY = "not hardware timing evidence, PCS certification, or unbounded liveness proof"
_Z3_BLOCKED_SOLVER_LABEL = "z3-solver unavailable"
_Z3_BLOCKED_CHECKED_SPECS = ["z3_solver_available"]
_Z3_SECTION_SOLVER_STATUSES = frozenset({"mixed", "not-run", "sat", "unknown", "unsat"})

_Z3_REPORT_PASS_FAIL_KEYS = frozenset(
    {
        "backend",
        "checked_specs",
        "claim_boundary",
        "holds",
        "max_depth",
        "payload_sha256",
        "safety",
        "schema_version",
        "scope",
        "solver",
        "status",
        "temporal",
    }
)
_Z3_REPORT_BLOCKED_KEYS = _Z3_REPORT_PASS_FAIL_KEYS | {"reason"}
_Z3_REPORT_SECTION_KEYS = frozenset(
    {
        "backend",
        "checked_specs",
        "holds",
        "max_depth",
        "solver_status",
        "violations",
    }
)
_Z3_REPORT_VIOLATION_KEYS = frozenset(
    {
        "marking",
        "message",
        "path",
        "place",
        "property_name",
        "transition",
    }
)


@dataclass(frozen=True)
class Z3FormalVerificationReport:
    """Combined bounded SMT formal-verification report."""

    holds: bool
    backend: str
    max_depth: int
    safety: Z3ModelCheckingReport
    temporal: Z3ModelCheckingReport


def verify_z3_formal_contracts(
    net: StochasticPetriNet,
    *,
    max_depth: int,
    marking_bounds: dict[str, tuple[float, float]],
    temporal_specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking]
    | None = None,
    weight_bounds: dict[str, tuple[float, float]] | None = None,
) -> Z3FormalVerificationReport:
    """Run bounded Z3 safety and temporal proof obligations."""
    checker = Z3BoundedModelChecker(net)
    safety = checker.prove_marking_bounds(marking_bounds, max_depth=max_depth, weight_bounds=weight_bounds)
    temporal = checker.verify_temporal_specs(temporal_specs or [], max_depth=max_depth, weight_bounds=weight_bounds)
    return Z3FormalVerificationReport(
        holds=safety.holds and temporal.holds,
        backend="z3",
        max_depth=max_depth,
        safety=safety,
        temporal=temporal,
    )


def write_z3_formal_report(
    report: Z3FormalVerificationReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Persist a bounded SMT report as JSON and Markdown evidence artifacts."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_z3_formal_report_payload(report)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_render_markdown(payload), encoding="utf-8")


def build_z3_formal_report_payload(report: Z3FormalVerificationReport) -> dict[str, Any]:
    """Return the schema-versioned JSON payload for a bounded Z3 report."""
    safety = asdict(report.safety)
    temporal = asdict(report.temporal)
    checked_specs = _checked_specs(safety, temporal)
    payload: dict[str, Any] = {
        "schema_version": Z3_FORMAL_REPORT_SCHEMA_VERSION,
        "status": "pass" if report.holds else "fail",
        "backend": report.backend,
        "solver": _z3_solver_label(),
        "holds": report.holds,
        "max_depth": report.max_depth,
        "checked_specs": checked_specs,
        "scope": Z3_FORMAL_REPORT_SCOPE,
        "claim_boundary": Z3_FORMAL_REPORT_CLAIM_BOUNDARY,
        "safety": safety,
        "temporal": temporal,
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_z3_formal_report_payload(payload)


def build_blocked_z3_formal_report_payload(reason: str) -> dict[str, Any]:
    """Return a schema-versioned blocked report for unavailable SMT evidence."""
    payload: dict[str, Any] = {
        "schema_version": Z3_FORMAL_REPORT_SCHEMA_VERSION,
        "status": "blocked",
        "backend": "z3",
        "solver": _Z3_BLOCKED_SOLVER_LABEL,
        "holds": False,
        "max_depth": 0,
        "checked_specs": list(_Z3_BLOCKED_CHECKED_SPECS),
        "scope": Z3_FORMAL_REPORT_SCOPE,
        "claim_boundary": Z3_FORMAL_REPORT_CLAIM_BOUNDARY,
        "reason": reason,
        "safety": None,
        "temporal": None,
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_z3_formal_report_payload(payload)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_unknown_keys(payload: dict[str, Any], *, allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{context} unknown fields: {', '.join(unknown)}")


def _reject_missing_keys(payload: dict[str, Any], *, required: frozenset[str], context: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(missing)}")


def _is_finite_json_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _validate_z3_violation_record(violation: Any, *, context: str) -> None:
    if not isinstance(violation, dict):
        raise ValueError(f"{context} violation must be an object")
    _reject_unknown_keys(violation, allowed=_Z3_REPORT_VIOLATION_KEYS, context=f"{context} violation")
    _reject_missing_keys(violation, required=_Z3_REPORT_VIOLATION_KEYS, context=f"{context} violation")
    if not isinstance(violation["property_name"], str) or not violation["property_name"]:
        raise ValueError(f"{context} violation property_name must be a non-empty string")
    if not isinstance(violation["message"], str) or not violation["message"]:
        raise ValueError(f"{context} violation message must be a non-empty string")
    marking = violation["marking"]
    if not isinstance(marking, dict):
        raise ValueError(f"{context} violation marking must be an object")
    for place, value in marking.items():
        if not isinstance(place, str) or not place:
            raise ValueError(f"{context} violation marking keys must be non-empty strings")
        if not _is_finite_json_number(value):
            raise ValueError(f"{context} violation marking values must be finite numbers")
    path = violation["path"]
    if not isinstance(path, list) or any(not isinstance(step, str) or not step for step in path):
        raise ValueError(f"{context} violation path must contain transition-name strings")
    for optional_name in ("place", "transition"):
        value = violation[optional_name]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{context} violation {optional_name} must be null or a non-empty string")


def _validate_z3_section_solver_consistency(section: dict[str, Any], *, context: str, section_name: str) -> None:
    solver_status = section["solver_status"]
    holds = section["holds"]
    violations = section["violations"]
    checked_specs = section["checked_specs"]
    if section_name == "safety":
        if solver_status == "unsat":
            if not holds:
                raise ValueError(f"{context} unsat section must hold")
            if violations:
                raise ValueError(f"{context} unsat section must not carry violations")
            return
        if solver_status == "sat":
            if holds:
                raise ValueError(f"{context} sat section must not hold")
            if not violations:
                raise ValueError(f"{context} sat section must carry counterexample violations")
            return
        if solver_status in {"mixed", "not-run"}:
            raise ValueError(f"{context} solver_status must describe one counterexample query")
        if holds:
            raise ValueError(f"{context} unknown section must not hold")
        if violations:
            raise ValueError(f"{context} unknown section must not carry violations")
        return

    if solver_status == "not-run":
        if not holds:
            raise ValueError(f"{context} not-run section must hold vacuously")
        if checked_specs:
            raise ValueError(f"{context} not-run section must not carry checked specs")
        if violations:
            raise ValueError(f"{context} not-run section must not carry violations")
        return
    if solver_status == "unknown":
        if holds:
            raise ValueError(f"{context} unknown section must not hold")
        if violations:
            raise ValueError(f"{context} unknown section must not carry violations")
        return
    if solver_status == "mixed" and len(checked_specs) < 2:
        raise ValueError(f"{context} mixed section must carry at least two checked specs")
    if holds and violations:
        raise ValueError(f"{context} holding section must not carry violations")
    if not holds and not violations:
        raise ValueError(f"{context} non-holding solver result must carry a violation")


def _validate_z3_section_checked_specs(checked_specs: list[Any], *, context: str) -> None:
    if any(not isinstance(spec, str) or not spec for spec in checked_specs):
        raise ValueError(f"{context} checked_specs must contain non-empty strings")
    if len(checked_specs) != len(set(checked_specs)):
        raise ValueError(f"{context} checked_specs must be unique")


def load_z3_formal_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a duplicate-key-safe Z3 formal evidence report."""
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("Z3 formal report must be a JSON object")
    return validate_z3_formal_report_payload(payload)


def validate_z3_formal_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a schema-versioned Z3 formal evidence report."""
    if payload.get("schema_version") != Z3_FORMAL_REPORT_SCHEMA_VERSION:
        raise ValueError("Z3 formal report schema_version is unsupported")
    if payload.get("backend") != "z3":
        raise ValueError("Z3 formal report backend must be 'z3'")
    status = payload.get("status")
    if status not in {"pass", "fail", "blocked"}:
        raise ValueError("Z3 formal report status must be pass, fail, or blocked")
    _reject_unknown_keys(
        payload,
        allowed=_Z3_REPORT_BLOCKED_KEYS if status == "blocked" else _Z3_REPORT_PASS_FAIL_KEYS,
        context="Z3 formal report",
    )
    if not isinstance(payload.get("solver"), str) or not payload["solver"]:
        raise ValueError("Z3 formal report solver must be a non-empty string")
    if not isinstance(payload.get("holds"), bool):
        raise ValueError("Z3 formal report holds must be a boolean")
    if isinstance(payload.get("max_depth"), bool) or not isinstance(payload.get("max_depth"), int):
        raise ValueError("Z3 formal report max_depth must be an integer")
    if payload["max_depth"] < 0:
        raise ValueError("Z3 formal report max_depth must be non-negative")
    if not isinstance(payload.get("checked_specs"), list) or not payload["checked_specs"]:
        raise ValueError("Z3 formal report checked_specs must be a non-empty list")
    if any(not isinstance(spec, str) or not spec for spec in payload["checked_specs"]):
        raise ValueError("Z3 formal report checked_specs must contain non-empty strings")
    if len(payload["checked_specs"]) != len(set(payload["checked_specs"])):
        raise ValueError("Z3 formal report checked_specs must be unique")
    if payload.get("scope") != Z3_FORMAL_REPORT_SCOPE:
        raise ValueError("Z3 formal report scope is unsupported")
    if payload.get("claim_boundary") != Z3_FORMAL_REPORT_CLAIM_BOUNDARY:
        raise ValueError("Z3 formal report claim_boundary is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise ValueError("Z3 formal report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("Z3 formal report payload_sha256 does not match payload")
    if status == "blocked":
        if payload["holds"]:
            raise ValueError("blocked Z3 formal report must not hold")
        if payload["solver"] != _Z3_BLOCKED_SOLVER_LABEL:
            raise ValueError("blocked Z3 formal report solver must be z3-solver unavailable")
        if payload["max_depth"] != 0:
            raise ValueError("blocked Z3 formal report max_depth must be 0")
        if payload["checked_specs"] != _Z3_BLOCKED_CHECKED_SPECS:
            raise ValueError("blocked Z3 formal report checked_specs must only contain z3_solver_available")
        if not isinstance(payload.get("reason"), str) or not payload["reason"]:
            raise ValueError("blocked Z3 formal report must include a reason")
        if payload.get("safety") is not None or payload.get("temporal") is not None:
            raise ValueError("blocked Z3 formal report must not carry proof sections")
        return payload
    if status == "pass" and not payload["holds"]:
        raise ValueError("passing Z3 formal report must hold")
    if status == "fail" and payload["holds"]:
        raise ValueError("failed Z3 formal report must not hold")
    if payload["solver"] == _Z3_BLOCKED_SOLVER_LABEL:
        raise ValueError("unavailable Z3 solver reports must use blocked status")
    if not payload["solver"].startswith("z3-solver "):
        raise ValueError("Z3 formal report solver must identify z3-solver")
    for section_name in ("safety", "temporal"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Z3 formal report {section_name} section must be an object")
        _reject_unknown_keys(section, allowed=_Z3_REPORT_SECTION_KEYS, context=f"Z3 formal report {section_name}")
        if section.get("backend") != "z3":
            raise ValueError(f"Z3 formal report {section_name} backend must be 'z3'")
        if section.get("max_depth") != payload["max_depth"]:
            raise ValueError(f"Z3 formal report {section_name} depth must match report depth")
        if section.get("solver_status") not in _Z3_SECTION_SOLVER_STATUSES:
            raise ValueError(f"Z3 formal report {section_name} solver_status is unsupported")
        if not isinstance(section.get("holds"), bool):
            raise ValueError(f"Z3 formal report {section_name} holds must be a boolean")
        if not isinstance(section.get("violations"), list):
            raise ValueError(f"Z3 formal report {section_name} violations must be a list")
        for violation in section["violations"]:
            _validate_z3_violation_record(violation, context=f"Z3 formal report {section_name}")
        if not isinstance(section.get("checked_specs"), list):
            raise ValueError(f"Z3 formal report {section_name} checked_specs must be a list")
        _validate_z3_section_checked_specs(section["checked_specs"], context=f"Z3 formal report {section_name}")
        _validate_z3_section_solver_consistency(
            section,
            context=f"Z3 formal report {section_name}",
            section_name=section_name,
        )
    if payload["holds"] != (payload["safety"]["holds"] and payload["temporal"]["holds"]):
        raise ValueError("Z3 formal report holds must match safety and temporal sections")
    if payload["checked_specs"] != _checked_specs(payload["safety"], payload["temporal"]):
        raise ValueError("Z3 formal report checked_specs must match proof sections")
    return payload


def _checked_specs(safety: dict[str, Any], temporal: dict[str, Any]) -> list[str]:
    specs = ["marking_bounds"]
    for section in (safety, temporal):
        for spec in section.get("checked_specs", []):
            if spec not in specs:
                specs.append(spec)
    return specs


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _z3_solver_label() -> str:
    try:
        import z3
    except ModuleNotFoundError:
        return "z3-solver unavailable"
    version = getattr(z3, "get_version_string", lambda: "unknown")()
    return f"z3-solver {version}"


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SCPN Z3 Formal Verification Report",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Backend: `{payload['backend']}`",
        f"- Solver: `{payload['solver']}`",
        f"- Max depth: `{payload['max_depth']}`",
        f"- Payload SHA-256: `{payload['payload_sha256']}`",
        f"- Scope: {payload['scope']}.",
        f"- Claim boundary: {payload['claim_boundary']}.",
        "",
    ]
    if payload["status"] == "blocked":
        lines.extend([f"- Reason: {payload['reason']}", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## Safety",
            "",
            f"- Holds: `{payload['safety']['holds']}`",
            f"- Solver status: `{payload['safety']['solver_status']}`",
            "",
            "## Temporal",
            "",
            f"- Holds: `{payload['temporal']['holds']}`",
            f"- Aggregate solver status: `{payload['temporal']['solver_status']}`",
            f"- Checked specs: `{', '.join(payload['temporal']['checked_specs'])}`",
            "",
        ]
    )
    violations = [*payload["safety"]["violations"], *payload["temporal"]["violations"]]
    if violations:
        lines.extend(["## Counterexamples", ""])
        for violation in violations:
            lines.append(f"- `{violation['property_name']}` path={violation['path']} message={violation['message']}")
        lines.append("")
    return "\n".join(lines)
