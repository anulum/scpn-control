# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second analytic evidence contract tests
"""Exercise complete content-hashed analytic reports through public APIs."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable

import pytest

from validation import validate_volt_second as facade
from validation import volt_second_evidence as evidence


def report() -> dict[str, object]:
    """Build an actual default analytic report through the producer."""
    return dict(facade.build_evidence(facade.validate_volt_second(), target_id="owned-target"))


def seal(payload: dict[str, object]) -> dict[str, object]:
    """Apply the public historical hash to an ordinary input-contract case."""
    payload["payload_sha256"] = evidence.evidence_digest(payload)
    return payload


def group(payload: dict[str, object], name: str) -> dict[str, object]:
    """Return a copied nested report object for deliberate malformed inputs."""
    value = payload[name]
    assert isinstance(value, dict)
    return dict(value)


@pytest.mark.parametrize("budget", [300.0, 140.0, 155.0])
def test_real_budget_configurations_round_trip(budget: float) -> None:
    """Preserve closed-form agreement across declared circuit budgets."""
    result = facade.validate_volt_second(config=replace(facade.default_config(), flux_budget_vs=budget))
    payload = facade.build_evidence(result, target_id="actual-case")
    assert evidence.validate_evidence_payload(payload) is result.passed
    assert facade.validate_evidence_payload(payload) is result.passed
    assert result.passed is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", "false"),
        ("inductive_rel_error", 1.0),
        ("inductive_rel_error", float("nan")),
        ("fluxes_passed", False),
        ("passed", False),
        ("exact_tol", 0.0),
        ("margin_abs_tol", -1.0),
        ("target_id", " "),
        ("target_id", 3),
        ("generated_utc", 3),
        ("generated_utc", "bad"),
        ("generated_utc", "2026-10-08T00:00:00"),
        ("generated_utc", "2026-10-08T00:00:00+01:00"),
    ],
)
def test_sealed_invalid_fields_are_refused(field: str, value: object) -> None:
    """Refuse invalid or inconsistent declarations despite their valid hash."""
    payload = report()
    payload[field] = value
    with pytest.raises(ValueError):
        evidence.validate_evidence_payload(seal(payload))


@pytest.mark.parametrize("change", ["missing", "extra", "unsupported", "invalid-hash", "mismatch", "non-json"])
def test_schema_and_digest_are_required(change: str) -> None:
    """Refuse incomplete schemas, mismatched hashes and unserialisable values."""
    payload = report()
    if change == "missing":
        del payload["config"]
    elif change == "extra":
        payload["unknown"] = 1
    elif change == "unsupported":
        payload["schema_version"] = "other"
    elif change == "invalid-hash":
        payload["payload_sha256"] = "z" * 64
    elif change == "mismatch":
        payload["target_id"] = "changed"
    else:
        payload["unknown"] = object()
    if change in {"missing", "extra", "unsupported"}:
        seal(payload)
    with pytest.raises(ValueError):
        evidence.validate_evidence_payload(payload)


@pytest.mark.parametrize("value", [None, [], "text"])
def test_runtime_nonobject_roots_are_refused(value: object) -> None:
    """Reject ordinary JSON roots that are not report objects."""
    validator: Callable[..., bool] = evidence.validate_evidence_payload
    with pytest.raises(ValueError):
        validator(value)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("config", "flux_budget_vs", -1.0),
        ("config", "bootstrap_current_ma", 15.0),
        ("decomposition", "max_rel_error", 0.5),
        ("monitor", "max_rel_error", 0.5),
        ("ramp_optimizer", "is_linear", False),
        ("ramp_optimizer", "is_linear", "true"),
        ("ramp_optimizer", "spacing_max_rel_error", float("inf")),
        ("monitor", "fraction_rel_error", -1.0),
    ],
)
def test_nested_domains_and_maxima_are_checked(section: str, field: str, value: object) -> None:
    """Reject nested invalid values and maxima/linearity contradictions."""
    payload = report()
    changed = group(payload, section)
    changed[field] = value
    payload[section] = changed
    with pytest.raises(ValueError):
        evidence.validate_evidence_payload(seal(payload))


@pytest.mark.parametrize("value", [None, [], [{"name": "only"}]])
def test_scaling_requires_complete_array(value: object) -> None:
    """Require every declared scaling law rather than a partial collection."""
    payload = report()
    payload["scaling"] = value
    with pytest.raises(ValueError):
        evidence.validate_evidence_payload(seal(payload))


@pytest.mark.parametrize(
    "field,value",
    [("name", "unknown"), ("name", 3), ("expected_ratio", 3.0), ("measured_ratio", 3.0), ("rel_error", True)],
)
def test_scaling_law_identity_and_arithmetic_are_checked(field: str, value: object) -> None:
    """Refuse unknown names, invalid ratios and wrong derived errors."""
    payload = report()
    raw = payload["scaling"]
    assert isinstance(raw, list)
    changed = copy.deepcopy(raw)
    changed[0][field] = value
    payload["scaling"] = changed
    with pytest.raises(ValueError):
        evidence.validate_evidence_payload(seal(payload))


def test_duplicate_scaling_laws_are_refused() -> None:
    """Reject duplication even when array cardinality remains four."""
    payload = report()
    raw = payload["scaling"]
    assert isinstance(raw, list)
    payload["scaling"] = [raw[0], raw[0], *raw[2:]]
    with pytest.raises(ValueError, match="known and unique"):
        evidence.validate_evidence_payload(seal(payload))


def test_scaling_maximum_is_recomputed() -> None:
    """Reject a declared maximum that differs from its actual members."""
    payload = report()
    payload["max_scaling_rel_error"] = 0.5
    with pytest.raises(ValueError, match="scaling maximum"):
        evidence.validate_evidence_payload(seal(payload))


def test_builder_rejects_invalid_target_and_inconsistent_result() -> None:
    """Refuse invalid producer input instead of publishing contradictory data."""
    result = facade.validate_volt_second()
    with pytest.raises(ValueError, match="target_id"):
        facade.build_evidence(result, target_id=" ")
    with pytest.raises(ValueError, match="stage verdict"):
        facade.build_evidence(replace(result, inductive_rel_error=1.0), target_id="bad-result")


def test_real_stricter_precision_failure_remains_false() -> None:
    """Retain a genuine failed arithmetic tolerance rather than coercing it."""
    result = facade.validate_volt_second(exact_tol=1e-30)
    assert result.passed is False
    assert result.monitor_passed is False
    payload = facade.build_evidence(result, target_id="strict-precision")
    assert evidence.validate_evidence_payload(payload) is False
