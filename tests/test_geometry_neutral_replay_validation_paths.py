# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometry-neutral replay validator, evidence and CLI branches

"""Validator-guard, evidence-admission and CLI branches of geometry-neutral replay.

Drives the scalar/UUID/sha256 coercion guards, the schema loader rejections, the
AER admission metadata consistency rules, the v1.1 shot-phase-log and manifest
digest guards, the tamper-evident evidence payload admission (including the
device-claim escalation), the markdown AER section, and the argparse CLI entry
point in both strict pass and strict fail dispositions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

import scpn_control.scpn.geometry_neutral_replay as gnr
from scpn_control.scpn.geometry_neutral_replay import (
    GEOMETRY_NEUTRAL_REPLAY_BOUNDED,
    GEOMETRY_NEUTRAL_REPLAY_QUALIFIED,
    SCHEMA_VERSION_V1_1,
    attach_aer_admission_metadata,
    build_aer_admission_metadata,
    generate_geometry_neutral_report,
    generate_report,
    geometry_neutral_replay_evidence,
    load_geometry_neutral_replay_evidence,
    load_geometry_neutral_replay_report,
    load_replay_schema,
    main,
    render_markdown,
    save_geometry_neutral_replay_evidence,
    validate_report,
)
from scpn_control.scpn.observation import AERControlObservation, SpikeBuffer, SpikeEvent

Report = dict[str, Any]


# ── fixtures ──────────────────────────────────────────────────────────


def _v1_1_report() -> Report:
    report = generate_report(steps=8, seed=20260604)
    bench = report["geometry_neutral_replay"]
    bench["schema_version"] = SCHEMA_VERSION_V1_1
    bench["pulse_id"] = "123e4567-e89b-12d3-a456-426614174000"
    bench["shot_phase_log"] = [
        {"t": 0.0, "state": "precharge", "reason": "bank voltage admitted"},
        {"t": 0.0005, "state": "trigger", "reason": "ignitron trigger issued"},
    ]
    return cast(Report, report)


def _aer_admission_metadata() -> Report:
    buffer = SpikeBuffer(capacity=8)
    buffer.extend(
        [
            SpikeEvent(neuron_id=0, timestamp_ns=10),
            SpikeEvent(neuron_id=1, timestamp_ns=30),
            SpikeEvent(neuron_id=1, timestamp_ns=50),
        ]
    )
    observation = AERControlObservation(
        timestamp_ns=100,
        spike_stream=buffer,
        decode_window_ns=100,
        decode_strategy="rate",
        n_features=4,
        require_monotonic=True,
    )
    return cast(
        Report,
        build_aer_admission_metadata(
            admission_report=observation.admission_report(),
            decode_strategy="rate",
            decode_window_ns=100,
            n_features=4,
            feature_normalisation="unit",
            require_monotonic=True,
            feature_vector=observation.to_features(),
        ),
    )


def _reseal_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = gnr._payload_sha256(payload)
    return payload


# ── scalar / identifier coercion guards ───────────────────────────────


def test_require_bool_rejects_non_boolean() -> None:
    with pytest.raises(ValueError, match="flag must be boolean"):
        gnr._require_bool("flag", 1)


@pytest.mark.parametrize("value", [True, "not-a-number"])
def test_require_finite_nonnegative_rejects_bool_and_non_numeric(value: Any) -> None:
    with pytest.raises(ValueError, match="x must be finite and non-negative"):
        gnr._require_finite_nonnegative("x", value)


@pytest.mark.parametrize("value", [True, "nan-text", float("inf")])
def test_require_finite_rejects_bool_non_numeric_and_infinite(value: Any) -> None:
    with pytest.raises(ValueError, match="x must be finite"):
        gnr._require_finite("x", value)


def test_require_optional_sha256_passes_none_and_rejects_malformed() -> None:
    assert gnr._require_optional_sha256("d", None) is None
    with pytest.raises(ValueError, match="d must be a SHA-256 hex digest"):
        gnr._require_optional_sha256("d", "short")


def test_require_nonnegative_int_rejects_negative() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        gnr._require_nonnegative_int("n", -1)


def test_require_positive_int_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        gnr._require_positive_int("n", 0)


def test_require_uuid_text_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="id must be a UUID string"):
        gnr._require_uuid_text("id", 42)


# ── schema loader ─────────────────────────────────────────────────────


def test_load_replay_schema_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="unsupported geometry-neutral replay schema version"):
        load_replay_schema("scpn-control.geometry-neutral-replay.v9")


def test_load_replay_schema_rejects_non_object_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gnr.json, "loads", lambda *args, **kwargs: ["not", "an", "object"])
    with pytest.raises(ValueError, match="replay schema document must be a JSON object"):
        load_replay_schema(gnr.SCHEMA_VERSION)


def test_register_v1_1_schema_rejects_id_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gnr, "load_replay_schema", lambda version: {"$id": "wrong-id"})
    with pytest.raises(ValueError, match="v1.1 replay schema id mismatch"):
        gnr.register_v1_1_schema()


# ── build_aer_admission_metadata feature-vector guards ────────────────


def test_build_aer_admission_rejects_feature_vector_shape_mismatch() -> None:
    buffer = SpikeBuffer(capacity=8)
    buffer.extend([SpikeEvent(neuron_id=0, timestamp_ns=10)])
    observation = AERControlObservation(
        timestamp_ns=20, spike_stream=buffer, decode_window_ns=100, decode_strategy="rate", n_features=4
    )
    with pytest.raises(ValueError, match="feature_vector must match n_features"):
        build_aer_admission_metadata(
            admission_report=observation.admission_report(),
            decode_strategy="rate",
            decode_window_ns=100,
            n_features=4,
            feature_vector=[1.0, 2.0],
        )


def test_build_aer_admission_rejects_non_finite_feature_vector() -> None:
    buffer = SpikeBuffer(capacity=8)
    buffer.extend([SpikeEvent(neuron_id=0, timestamp_ns=10)])
    observation = AERControlObservation(
        timestamp_ns=20, spike_stream=buffer, decode_window_ns=100, decode_strategy="rate", n_features=2
    )
    with pytest.raises(ValueError, match="feature_vector must be finite"):
        build_aer_admission_metadata(
            admission_report=observation.admission_report(),
            decode_strategy="rate",
            decode_window_ns=100,
            n_features=2,
            feature_vector=[1.0, float("inf")],
        )


# ── generate_report contract and public alias ─────────────────────────


def test_generate_report_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="steps must be >= 4"):
        generate_report(steps=3)


def test_generate_geometry_neutral_report_alias_matches_core() -> None:
    report = generate_geometry_neutral_report(steps=6, seed=7)
    validate_report(report)
    assert report["geometry_neutral_replay"]["schema_version"] == gnr.SCHEMA_VERSION


# ── AER admission metadata consistency rules ──────────────────────────


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.__setitem__("schema_version", "bad"), "schema_version is unsupported"),
        (
            lambda m: m.__setitem__("out_of_order_event_count", 2),
            "monotonic_input conflicts with out_of_order_event_count",
        ),
        (
            lambda m: (
                m.__setitem__("monotonic_input", False),
                m.__setitem__("require_monotonic", False),
                m.__setitem__("out_of_order_event_count", 0),
            ),
            "must record out_of_order_event_count",
        ),
        (lambda m: m.__setitem__("decode_strategy", "frequency"), "decode_strategy must be one of"),
        (lambda m: m.__setitem__("feature_normalisation", "softmax"), "feature_normalisation must be one of"),
        (lambda m: m.__setitem__("feature_count", 99), "feature_count must match n_features"),
        (lambda m: m.pop("feature_count"), "feature_vector_sha256 requires feature_count"),
    ],
)
def test_aer_admission_metadata_rejects_inconsistent_fields(mutate: Any, match: str) -> None:
    metadata = dict(_aer_admission_metadata())
    mutate(metadata)
    with pytest.raises(ValueError, match=match):
        gnr._validate_aer_admission_metadata(metadata)


# ── v1.1 shot-phase-log and manifest digest guards ────────────────────


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: b.__setitem__("shot_phase_log", "not-a-list"), "shot_phase_log must be a list"),
        (lambda b: b.__setitem__("shot_phase_log", [42]), r"shot_phase_log\[0\] must be an object"),
        (lambda b: b["shot_phase_log"][0].__setitem__("t", -1.0), r"shot_phase_log\[0\].t must be non-negative"),
        (
            lambda b: b["shot_phase_log"][0].__setitem__("state", "  "),
            r"shot_phase_log\[0\].state must be non-empty text",
        ),
        (
            lambda b: b["shot_phase_log"][0].__setitem__("reason", ""),
            r"shot_phase_log\[0\].reason must be non-empty text",
        ),
    ],
)
def test_v1_1_extensions_reject_malformed_shot_phase_log(mutate: Any, match: str) -> None:
    report = _v1_1_report()
    mutate(report["geometry_neutral_replay"])
    with pytest.raises(ValueError, match=match):
        validate_report(report)


def test_v1_1_manifest_digest_without_aer_metadata_is_rejected() -> None:
    report = _v1_1_report()
    report["geometry_neutral_replay"]["manifest"]["aer_admission_digest"] = "a" * 64
    with pytest.raises(ValueError, match="manifest aer admission digest requires aer_admission metadata"):
        validate_report(report)


# ── core report contract guards ───────────────────────────────────────


def test_validate_report_reports_missing_required_key() -> None:
    report = generate_report(steps=6)
    del report["geometry_neutral_replay"]["metrics"]
    with pytest.raises(ValueError, match="missing geometry_neutral_replay.metrics"):
        validate_report(report)


def test_validate_report_rejects_manifest_schema_mismatch() -> None:
    report = generate_report(steps=6)
    report["geometry_neutral_replay"]["manifest"]["schema_version"] = "bad"
    with pytest.raises(ValueError, match="unexpected manifest schema_version"):
        validate_report(report)


def test_load_report_rejects_non_object_document(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="replay report must be a JSON object"):
        load_geometry_neutral_replay_report(path)


# ── tamper-evident evidence payload admission ─────────────────────────


def _bounded_evidence_payload() -> dict[str, Any]:
    report = generate_report(steps=6)
    return asdict(geometry_neutral_replay_evidence(report))


def test_evidence_is_built_with_utc_timestamp_when_unspecified() -> None:
    evidence = geometry_neutral_replay_evidence(generate_report(steps=6))
    assert evidence.generated_utc.endswith("Z")
    assert evidence.claim_status == GEOMETRY_NEUTRAL_REPLAY_BOUNDED


@pytest.mark.parametrize(
    ("field", "value", "match", "reseal"),
    [
        ("schema_version", "bad", "schema_version is unsupported", False),
        ("payload_sha256", "abc", "payload_sha256 must be a SHA-256 hex digest", False),
        ("generated_utc", "2026-06-20T00:00:00", "generated_utc must be a UTC timestamp", True),
        ("replay_schema_version", "bad", "replay_schema_version is unsupported", True),
        ("scenario_digest", "xyz", "scenario_digest must be a SHA-256 hex digest", True),
        (
            "measured_or_benchmark_artefact_sha256",
            "xyz",
            "measured_or_benchmark_artefact_sha256 must be a SHA-256 hex digest",
            True,
        ),
        ("claim_status", GEOMETRY_NEUTRAL_REPLAY_QUALIFIED, "claim_status does not match device claim state", True),
    ],
)
def test_evidence_payload_rejects_single_field_tampering(field: str, value: Any, match: str, reseal: bool) -> None:
    payload = _bounded_evidence_payload()
    payload[field] = value
    if reseal:
        _reseal_evidence(payload)
    with pytest.raises(ValueError, match=match):
        gnr._validate_geometry_neutral_replay_evidence_payload(payload, require_device_claim=False)


def test_evidence_device_claim_requires_deterministic_threshold_pass() -> None:
    payload = _bounded_evidence_payload()
    payload["device_claim_allowed"] = True
    payload["claim_status"] = GEOMETRY_NEUTRAL_REPLAY_QUALIFIED
    payload["measured_or_benchmark_artefact_sha256"] = "a" * 64
    payload["magnetic_configuration_reference"] = "measured W7-X reduced-order reference"
    payload["deterministic"] = False
    _reseal_evidence(payload)
    with pytest.raises(ValueError, match="require deterministic threshold-passing replay evidence"):
        gnr._validate_geometry_neutral_replay_evidence_payload(payload, require_device_claim=False)


def test_evidence_device_claim_admits_non_synthetic_reference() -> None:
    """A deterministic, threshold-passing device claim with a non-synthetic reference is admitted.

    Exercises the successful fall-through past the synthetic-provenance guard (branch 888->890).
    """
    payload = _bounded_evidence_payload()
    payload["device_claim_allowed"] = True
    payload["claim_status"] = GEOMETRY_NEUTRAL_REPLAY_QUALIFIED
    payload["measured_or_benchmark_artefact_sha256"] = "a" * 64
    payload["magnetic_configuration_reference"] = "measured W7-X reduced-order reference"
    _reseal_evidence(payload)
    evidence = gnr._validate_geometry_neutral_replay_evidence_payload(payload, require_device_claim=False)
    assert evidence.device_claim_allowed is True


def test_v1_1_extensions_accept_frc_diagnostics_without_optional_keys() -> None:
    """An frc_diagnostics mapping present but missing every optional key validates cleanly.

    Exercises the absent-key fall-throughs for s_parameter_at_burn / mrti_peak_amplitude_m /
    tilt_growth_rate_s_inv (branches 700->705, 705->710, 710->716).
    """
    gnr._validate_v1_1_extensions({"frc_diagnostics": {}, "manifest": {}})


def test_build_aer_admission_metadata_omits_feature_digest_without_vector() -> None:
    """Without a feature_vector the metadata carries no feature digest (branch 279->287)."""
    buffer = SpikeBuffer(capacity=8)
    buffer.extend([SpikeEvent(neuron_id=0, timestamp_ns=10), SpikeEvent(neuron_id=1, timestamp_ns=30)])
    observation = AERControlObservation(
        timestamp_ns=100,
        spike_stream=buffer,
        decode_window_ns=100,
        decode_strategy="rate",
        n_features=4,
        require_monotonic=True,
    )
    metadata = build_aer_admission_metadata(
        admission_report=observation.admission_report(),
        decode_strategy="rate",
        decode_window_ns=100,
        n_features=4,
    )
    assert "feature_vector_sha256" not in metadata


def test_assert_claim_admissible_rejects_foreign_object() -> None:
    from scpn_control.scpn.geometry_neutral_replay import assert_geometry_neutral_replay_claim_admissible

    with pytest.raises(ValueError, match="evidence must be GeometryNeutralReplayEvidence"):
        assert_geometry_neutral_replay_claim_admissible(cast(Any, {"not": "evidence"}))


def test_save_evidence_rejects_foreign_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence must be GeometryNeutralReplayEvidence"):
        save_geometry_neutral_replay_evidence(cast(Any, {"not": "evidence"}), tmp_path / "e.json")


def test_load_evidence_rejects_non_object_document(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="replay evidence must be a JSON object"):
        load_geometry_neutral_replay_evidence(path)


def test_evidence_round_trips_through_disk(tmp_path: Path) -> None:
    evidence = geometry_neutral_replay_evidence(generate_report(steps=6))
    path = tmp_path / "evidence.json"
    save_geometry_neutral_replay_evidence(evidence, path)
    loaded = load_geometry_neutral_replay_evidence(path)
    assert loaded == evidence


# ── markdown AER section ──────────────────────────────────────────────


def test_render_markdown_includes_aer_admission_section() -> None:
    report = attach_aer_admission_metadata(_v1_1_report(), _aer_admission_metadata())
    markdown = render_markdown(report)
    assert "## AER Admission" in markdown
    assert "Decode strategy: `rate`" in markdown
    assert "Feature count: `4`" in markdown


# ── argparse CLI entry point ──────────────────────────────────────────


def test_cli_writes_artifacts_and_returns_success(tmp_path: Path) -> None:
    out_json = tmp_path / "replay.json"
    out_md = tmp_path / "replay.md"
    code = main(
        ["--steps", "8", "--seed", "11", "--output-json", str(out_json), "--output-md", str(out_md), "--strict"]
    )
    assert code == 0
    assert out_json.is_file()
    assert "Geometry-Neutral Stellarator Replay" in out_md.read_text(encoding="utf-8")


def test_cli_strict_mode_fails_on_unmet_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failing = deepcopy(generate_report(steps=6))
    failing["geometry_neutral_replay"]["passes_thresholds"] = False
    monkeypatch.setattr(gnr, "generate_report", lambda **kwargs: failing)
    code = main(["--output-json", str(tmp_path / "r.json"), "--output-md", str(tmp_path / "r.md"), "--strict"])
    assert code == 2
