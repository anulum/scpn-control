# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Z3 formal report I/O, schema validation and markdown branches

"""Schema, validation and rendering branches of the Z3 formal-report evidence I/O.

Covers the solver-label degradation when z3 is absent, the JSON/Markdown
round-trip through :func:`verify_z3_formal_contracts` and
:func:`write_z3_formal_report`, the markdown renderer for blocked and
counterexample reports, and the single-field tampering rejections of the
schema-versioned Z3 formal report (top-level, section-solver consistency,
structural, and violation-record checks).
"""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_control.scpn.formal_verification import FormalViolation
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_formal_report import (
    Z3FormalVerificationReport,
    _render_markdown,
    _z3_solver_label,
    build_blocked_z3_formal_report_payload,
    build_z3_formal_report_payload,
    load_z3_formal_report,
    validate_z3_formal_report_payload,
    verify_z3_formal_contracts,
    write_z3_formal_report,
)
from scpn_control.scpn.z3_model_checking import Z3ModelCheckingReport

requires_z3 = pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3-solver optional dependency absent")


def _transfer_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _block_z3_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "z3":
            raise ModuleNotFoundError("No module named 'z3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute the payload digest exactly as :func:`_payload_digest` does."""
    import hashlib

    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def _valid_pass_payload() -> dict[str, Any]:
    return build_z3_formal_report_payload(
        Z3FormalVerificationReport(
            holds=True,
            backend="z3",
            max_depth=2,
            safety=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["marking_bounds"]),
            temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
        )
    )


def _valid_fail_payload() -> dict[str, Any]:
    violation = FormalViolation(
        property_name="unsafe_bound",
        message="sink exceeds admitted control envelope",
        marking={"sink": 1.0},
        path=["move"],
        place="sink",
    )
    return build_z3_formal_report_payload(
        Z3FormalVerificationReport(
            holds=False,
            backend="z3",
            max_depth=2,
            safety=Z3ModelCheckingReport(False, "z3", 2, "sat", [violation], ["unsafe_bound"]),
            temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
        )
    )


def _fail_with_violation(mutate: Any) -> dict[str, Any]:
    payload = _valid_fail_payload()
    mutate(payload["safety"]["violations"])
    return _reseal(payload)


def test_solver_label_degrades_when_z3_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_z3_import(monkeypatch)
    assert _z3_solver_label() == "z3-solver unavailable"


@requires_z3
def test_verify_write_and_load_round_trip(tmp_path: Path) -> None:
    report = verify_z3_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"sink": (0.0, 1.0)},
    )
    json_path = tmp_path / "z3_formal.json"
    markdown_path = tmp_path / "z3_formal.md"
    write_z3_formal_report(report, json_path=json_path, markdown_path=markdown_path)
    payload = load_z3_formal_report(json_path)
    assert payload["status"] in {"pass", "fail"}
    assert payload["solver"].startswith("z3-solver ")
    assert validate_z3_formal_report_payload(payload) == payload
    assert markdown_path.read_text(encoding="utf-8").startswith("# SCPN Z3 Formal Verification Report")


def test_markdown_renders_blocked_report_reason() -> None:
    payload = build_blocked_z3_formal_report_payload("z3 unavailable in deployment image")
    markdown = _render_markdown(payload)
    assert "Reason: z3 unavailable in deployment image" in markdown
    assert "## Safety" not in markdown


def test_markdown_renders_counterexample_section() -> None:
    markdown = _render_markdown(_valid_fail_payload())
    assert "## Counterexamples" in markdown
    assert "`unsafe_bound`" in markdown
    assert "path=['move']" in markdown


def test_load_rejects_non_object_payload(tmp_path: Path) -> None:
    from scpn_control.scpn.z3_formal_report import load_z3_formal_report

    path = tmp_path / "array-report.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_z3_formal_report(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "scpn-control.z3-formal-report.v0", "schema_version is unsupported"),
        ("max_depth", -1, "max_depth must be non-negative"),
        ("checked_specs", ["marking_bounds", "marking_bounds"], "checked_specs must be unique"),
        ("payload_sha256", "abc", "payload_sha256 must be a SHA-256 hex digest"),
    ],
)
def test_validator_rejects_top_level_tampering_before_digest(field: str, value: Any, match: str) -> None:
    payload = _valid_pass_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        validate_z3_formal_report_payload(payload)


@pytest.mark.parametrize(
    ("solver_status", "holds", "match"),
    [
        ("unsat", False, "unsat section must hold"),
        ("sat", False, "sat section must carry counterexample violations"),
        ("unknown", True, "unknown section must not hold"),
    ],
)
def test_validator_rejects_section_solver_inconsistency(solver_status: str, holds: bool, match: str) -> None:
    payload = _valid_pass_payload()
    payload["safety"]["solver_status"] = solver_status
    payload["safety"]["holds"] = holds
    _reseal(payload)
    with pytest.raises(ValueError, match=match):
        validate_z3_formal_report_payload(payload)


def test_validator_rejects_section_with_empty_checked_spec() -> None:
    payload = _valid_pass_payload()
    payload["safety"]["checked_specs"] = ["marking_bounds", ""]
    # keep the top-level list non-empty and valid so the section check is reached
    payload["checked_specs"] = ["marking_bounds", "response_ok"]
    _reseal(payload)
    with pytest.raises(ValueError, match="checked_specs must contain non-empty strings"):
        validate_z3_formal_report_payload(payload)


def test_validator_rejects_top_level_holds_inconsistent_with_sections() -> None:
    payload = _valid_fail_payload()
    payload["safety"]["solver_status"] = "unsat"
    payload["safety"]["holds"] = True
    payload["safety"]["violations"] = []
    _reseal(payload)
    with pytest.raises(ValueError, match="holds must match safety and temporal sections"):
        validate_z3_formal_report_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("safety", 42), "safety section must be an object"),
        (lambda p: p["safety"].__setitem__("backend", "smt"), "safety backend must be 'z3'"),
        (lambda p: p["safety"].__setitem__("holds", "yes"), "safety holds must be a boolean"),
        (lambda p: p["safety"].__setitem__("violations", "none"), "safety violations must be a list"),
        (lambda p: p["safety"].__setitem__("checked_specs", "all"), "safety checked_specs must be a list"),
    ],
)
def test_validator_rejects_section_structural_tampering(mutate: Any, match: str) -> None:
    payload = _valid_pass_payload()
    mutate(payload)
    _reseal(payload)
    with pytest.raises(ValueError, match=match):
        validate_z3_formal_report_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda v: v.__setitem__(0, 42), "violation must be an object"),
        (lambda v: v[0].pop("transition"), "violation missing fields"),
        (lambda v: v[0].__setitem__("property_name", ""), "violation property_name must be a non-empty string"),
        (lambda v: v[0].__setitem__("message", ""), "violation message must be a non-empty string"),
        (lambda v: v[0].__setitem__("marking", "n/a"), "violation marking must be an object"),
        (lambda v: v[0].__setitem__("marking", {"": 1.0}), "violation marking keys must be non-empty strings"),
        (
            lambda v: v[0].__setitem__("marking", {"sink": float("inf")}),
            "violation marking values must be finite numbers",
        ),
        (lambda v: v[0].__setitem__("path", [""]), "violation path must contain transition-name strings"),
        (lambda v: v[0].__setitem__("place", 5), "violation place must be null or a non-empty string"),
    ],
)
def test_validator_rejects_violation_record_tampering(mutate: Any, match: str) -> None:
    payload = _fail_with_violation(mutate)
    with pytest.raises(ValueError, match=match):
        validate_z3_formal_report_payload(payload)
