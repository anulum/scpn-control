# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Codac Interface.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CODAC Interface Tests
# ──────────────────────────────────────────────────────────────────────
"""CODAC runtime-evidence contract and validation tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scpn_control.control.codac_evidence import (
    CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY,
    CODAC_RUNTIME_EVIDENCE_QUALIFIED,
    CODACRuntimeEvidence,
    _payload_sha256,
    _percentile,
    _require_finite_nonnegative,
    _require_nonnegative_int,
)
from scpn_control.control.codac_interface import (
    CODACConfig,
    CODACInterface,
    assert_codac_runtime_claim_admissible,
    codac_runtime_evidence,
    load_codac_runtime_evidence,
    save_codac_runtime_evidence,
)


def _make_interface() -> CODACInterface:
    """Build a bounded adapter for local evidence tests."""
    return CODACInterface(CODACConfig(), controller=object())


# ── Runtime evidence ─────────────────────────────────────────────────


def test_codac_runtime_evidence_binds_exports_and_payload(tmp_path: Path) -> None:
    """Check codac runtime evidence binds exports and payload."""
    iface = _make_interface()
    evidence = codac_runtime_evidence(
        iface,
        controller_id="nsc-v0.19.2",
        observed_cycle_us=[410.0, 420.0, 430.0, 440.0],
        interlock_checks=2,
        interlock_blocks=1,
        backpressure_events=0,
        generated_utc="2026-05-24T00:00:00Z",
        facility_claim_allowed=False,
    )
    assert evidence.claim_status == CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY
    assert evidence.input_channel_count == 12
    assert evidence.output_channel_count == 11
    assert evidence.interlock_pv_count == len(iface.config.interlock_pvs)
    assert evidence.schema_version == "scpn-control.codac-runtime-evidence.v3"
    assert evidence.output_limits_enforced is True
    assert evidence.epics_drive_limits_exported is True
    assert evidence.interlock_fail_closed is True
    assert len(evidence.epics_db_sha256) == 64
    assert len(evidence.opcua_nodeset_sha256) == 64
    assert len(evidence.payload_sha256) == 64
    with pytest.raises(ValueError, match="local-only"):
        assert_codac_runtime_claim_admissible(evidence)

    path = tmp_path / "codac-runtime-evidence.json"
    save_codac_runtime_evidence(evidence, path)
    loaded = load_codac_runtime_evidence(path)
    assert loaded == evidence


def test_codac_runtime_evidence_keeps_local_runs_out_of_facility_claims() -> None:
    """Check codac runtime evidence keeps local runs out of facility claims."""
    iface = _make_interface()
    evidence = codac_runtime_evidence(
        iface,
        controller_id="local-loopback",
        observed_cycle_us=[410.0, 420.0],
        interlock_checks=1,
        interlock_blocks=1,
        facility_claim_allowed=False,
    )
    assert evidence.claim_status == CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY
    with pytest.raises(ValueError, match="local-only"):
        assert_codac_runtime_claim_admissible(evidence)


def test_codac_runtime_evidence_requires_interlock_block_for_facility_claim() -> None:
    """Check codac runtime evidence requires interlock block for facility claim."""
    iface = _make_interface()
    with pytest.raises(ValueError, match="exercise and block"):
        codac_runtime_evidence(
            iface,
            controller_id="nsc-v0.19.2",
            observed_cycle_us=[410.0, 420.0],
            interlock_checks=2,
            interlock_blocks=0,
            facility_claim_allowed=True,
        )


def test_codac_runtime_evidence_rejects_deadline_overrun_for_facility_claim() -> None:
    """Check codac runtime evidence rejects deadline overrun for facility claim."""
    iface = _make_interface()
    with pytest.raises(ValueError, match="cycle deadline"):
        codac_runtime_evidence(
            iface,
            controller_id="nsc-v0.19.2",
            observed_cycle_us=[410.0, 1500.0],
            interlock_checks=2,
            interlock_blocks=1,
            facility_claim_allowed=True,
        )


def test_codac_runtime_evidence_rejects_tampered_json(tmp_path: Path) -> None:
    """Check codac runtime evidence rejects tampered json."""
    iface = _make_interface()
    evidence = codac_runtime_evidence(
        iface,
        controller_id="nsc-v0.19.2",
        observed_cycle_us=[410.0, 420.0],
        interlock_checks=2,
        interlock_blocks=1,
        facility_claim_allowed=False,
    )
    path = tmp_path / "codac-runtime-evidence.json"
    save_codac_runtime_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observed_cycle_max_us"] = 421.0
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="payload_sha256"):
        load_codac_runtime_evidence(path)


def test_codac_runtime_evidence_rejects_duplicate_json_key(tmp_path: Path) -> None:
    """Check codac runtime evidence rejects duplicate json key."""
    path = tmp_path / "codac-runtime-evidence.json"
    path.write_text('{"schema_version": "x", "schema_version": "y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_codac_runtime_evidence(path)


# ── Numeric validation helpers ───────────────────────────────────────


@pytest.mark.parametrize("value", [[1.0], {"a": 1}, None])
def test_require_finite_nonnegative_rejects_non_numeric_types(value: Any) -> None:
    """Check require finite nonnegative rejects non numeric types."""
    with pytest.raises(ValueError, match="finite non-negative"):
        _require_finite_nonnegative("metric", value)


def test_require_finite_nonnegative_rejects_bool() -> None:
    """Check require finite nonnegative rejects bool."""
    with pytest.raises(ValueError, match="finite non-negative"):
        _require_finite_nonnegative("metric", True)


def test_require_finite_nonnegative_rejects_unparsable_string() -> None:
    """Check require finite nonnegative rejects unparsable string."""
    with pytest.raises(ValueError, match="finite non-negative"):
        _require_finite_nonnegative("metric", "not-a-number")


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_require_finite_nonnegative_rejects_negative_or_non_finite(value: Any) -> None:
    """Check require finite nonnegative rejects negative or non finite."""
    with pytest.raises(ValueError, match="finite non-negative"):
        _require_finite_nonnegative("metric", value)


def test_require_finite_nonnegative_accepts_numeric_string() -> None:
    """Check require finite nonnegative accepts numeric string."""
    assert _require_finite_nonnegative("metric", "410.5") == 410.5


@pytest.mark.parametrize("value", [True, -1, 1.5, "3"])
def test_require_nonnegative_int_rejects_bad_values(value: Any) -> None:
    """Check require nonnegative int rejects bad values."""
    with pytest.raises(ValueError, match="non-negative integer"):
        _require_nonnegative_int("count", value)


def test_percentile_single_sample_returns_only_value() -> None:
    """Check percentile single sample returns only value."""
    assert _percentile([5.0], 0.5) == 5.0


def test_percentile_integer_position_returns_exact_element() -> None:
    """Check percentile integer position returns exact element."""
    # q=0.5 over three samples lands exactly on index 1 (lo == hi).
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0


# ── Evidence payload validation (re-sealed tamper matrix) ─────────────


def _sealed_payload(iface: CODACInterface, **overrides: Any) -> dict[str, Any]:
    """Build a forged facility payload for validation guards, then re-seal."""
    evidence = codac_runtime_evidence(
        iface,
        controller_id="nsc-v0.19.2",
        observed_cycle_us=[410.0, 420.0, 430.0, 440.0],
        interlock_checks=2,
        interlock_blocks=1,
        facility_claim_allowed=False,
    )
    payload = asdict(evidence)
    payload["facility_claim_allowed"] = True
    payload["claim_status"] = CODAC_RUNTIME_EVIDENCE_QUALIFIED
    payload.update(overrides)
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _load_payload(tmp_path: Path, payload: dict[str, Any], **kwargs: Any) -> CODACRuntimeEvidence:
    path = tmp_path / "codac-runtime-evidence.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return load_codac_runtime_evidence(path, **kwargs)


def test_payload_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    """Check payload rejects unsupported schema version."""
    iface = _make_interface()
    payload = _sealed_payload(iface, schema_version="codac-runtime-evidence.v0")
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_legacy_v1_schema(tmp_path: Path) -> None:
    """Check payload rejects legacy v1 schema."""
    iface = _make_interface()
    payload = _sealed_payload(iface, schema_version="scpn-control.codac-runtime-evidence.v1")
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_non_sha256_digest(tmp_path: Path) -> None:
    """Check payload rejects non sha256 digest."""
    iface = _make_interface()
    payload = _sealed_payload(iface)
    payload["payload_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="payload_sha256 must be a SHA-256"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_timestamp_without_z(tmp_path: Path) -> None:
    """Check payload rejects timestamp without z."""
    iface = _make_interface()
    payload = _sealed_payload(iface, generated_utc="2026-05-24T00:00:00")
    with pytest.raises(ValueError, match="ending in Z"):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("controller_id", "controller_id must be non-empty"),
        ("plant_system", "plant_system must be non-empty"),
        ("pv_prefix", "pv_prefix must be non-empty"),
    ],
)
def test_payload_rejects_blank_identity_fields(tmp_path: Path, field: str, match: str) -> None:
    """Check payload rejects blank identity fields."""
    iface = _make_interface()
    payload = _sealed_payload(iface, **{field: "   "})
    with pytest.raises(ValueError, match=match):
        _load_payload(tmp_path, payload)


def test_payload_rejects_non_positive_cycle_hz(tmp_path: Path) -> None:
    """Check payload rejects non positive cycle hz."""
    iface = _make_interface()
    payload = _sealed_payload(iface, cycle_hz=0.0)
    with pytest.raises(ValueError, match="cycle_hz must be positive"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_deadline_mismatch(tmp_path: Path) -> None:
    """Check payload rejects deadline mismatch."""
    iface = _make_interface()
    payload = _sealed_payload(iface, deadline_us=123.0)
    with pytest.raises(ValueError, match="deadline_us must equal"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_unordered_percentiles(tmp_path: Path) -> None:
    """Check payload rejects unordered percentiles."""
    iface = _make_interface()
    payload = _sealed_payload(iface, observed_cycle_p50_us=500.0)
    with pytest.raises(ValueError, match="percentiles must be ordered"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_channel_count_drift(tmp_path: Path) -> None:
    """Check payload rejects channel count drift."""
    iface = _make_interface()
    payload = _sealed_payload(iface, input_channel_count=99)
    with pytest.raises(ValueError, match="channel counts do not match"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_zero_interlock_pv_count(tmp_path: Path) -> None:
    """Check payload rejects zero interlock pv count."""
    iface = _make_interface()
    payload = _sealed_payload(iface, interlock_pv_count=0)
    with pytest.raises(ValueError, match="at least one interlock PV"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_blocks_exceeding_checks(tmp_path: Path) -> None:
    """Check payload rejects blocks exceeding checks."""
    iface = _make_interface()
    payload = _sealed_payload(iface, interlock_blocks=5, interlock_checks=2)
    with pytest.raises(ValueError, match="interlock_blocks cannot exceed"):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "field",
    ["output_limits_enforced", "epics_drive_limits_exported", "interlock_fail_closed"],
)
def test_payload_rejects_non_boolean_boundary_guard(tmp_path: Path, field: str) -> None:
    """Check payload rejects non boolean boundary guard."""
    iface = _make_interface()
    payload = _sealed_payload(iface, **{field: "yes"})
    with pytest.raises(ValueError, match=f"{field} must be boolean"):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "field",
    ["output_limits_enforced", "epics_drive_limits_exported", "interlock_fail_closed"],
)
def test_payload_requires_every_boundary_guard_for_facility_claim(tmp_path: Path, field: str) -> None:
    """Check payload requires every boundary guard for facility claim."""
    iface = _make_interface()
    payload = _sealed_payload(iface, **{field: False})
    with pytest.raises(ValueError, match="all fail-closed boundary guards"):
        _load_payload(tmp_path, payload, require_facility_claim=True)


@pytest.mark.parametrize("field", ["epics_db_sha256", "opcua_nodeset_sha256"])
def test_payload_rejects_non_sha256_export_digests(tmp_path: Path, field: str) -> None:
    """Check payload rejects non sha256 export digests."""
    iface = _make_interface()
    payload = _sealed_payload(iface, **{field: "not-a-digest"})
    with pytest.raises(ValueError, match=f"{field} must be a SHA-256"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_non_boolean_facility_flag(tmp_path: Path) -> None:
    """Check payload rejects non boolean facility flag."""
    iface = _make_interface()
    payload = _sealed_payload(iface, facility_claim_allowed="yes")
    with pytest.raises(ValueError, match="facility_claim_allowed must be boolean"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_claim_status_inconsistent_with_flag(tmp_path: Path) -> None:
    """Check payload rejects claim status inconsistent with flag."""
    iface = _make_interface()
    payload = _sealed_payload(iface, claim_status="totally-wrong")
    with pytest.raises(ValueError, match="claim_status does not match"):
        _load_payload(tmp_path, payload)


def test_payload_rejects_non_object_json(tmp_path: Path) -> None:
    """Check payload rejects non object json."""
    path = tmp_path / "codac-runtime-evidence.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_codac_runtime_evidence(path)


# ── Builder argument guards ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"controller_id": "  "}, "controller_id must be non-empty"),
        ({"plant_system": "  "}, "plant_system must be non-empty"),
        ({"observed_cycle_us": []}, "at least one sample"),
        ({"interlock_checks": 1, "interlock_blocks": 2}, "interlock_blocks cannot exceed"),
    ],
)
def test_codac_runtime_evidence_builder_argument_guards(kwargs: dict[str, Any], match: str) -> None:
    """Check codac runtime evidence builder argument guards."""
    iface = _make_interface()
    base: dict[str, Any] = {
        "controller_id": "nsc-v0.19.2",
        "observed_cycle_us": [410.0, 420.0],
        "interlock_checks": 2,
        "interlock_blocks": 1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        codac_runtime_evidence(iface, **base)


def test_codac_runtime_evidence_rejects_backpressure_for_facility_claim() -> None:
    """Check codac runtime evidence rejects backpressure for facility claim."""
    iface = _make_interface()
    with pytest.raises(ValueError, match="backpressure events cannot support"):
        codac_runtime_evidence(
            iface,
            controller_id="nsc-v0.19.2",
            observed_cycle_us=[410.0, 420.0],
            interlock_checks=2,
            interlock_blocks=1,
            backpressure_events=3,
            facility_claim_allowed=True,
        )
