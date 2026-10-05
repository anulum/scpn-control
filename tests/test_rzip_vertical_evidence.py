# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP report domain, consistency and public writer regressions.

"""Exercise sealed reports from the real rigid vertical model through public APIs."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import pytest

from validation import rzip_vertical_evidence as evidence
from validation import validate_rzip_vertical_stability as rzip


@pytest.fixture(scope="module")
def result() -> rzip.RzipValidationResult:
    """Run the actual no-wall and resistive-wall validation once per module."""
    return rzip.validate_rzip_vertical_stability()


def reseal(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the published JSON seal so domain tests cannot fail on stale hashes."""
    payload["payload_sha256"] = ""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def test_actual_passing_and_failing_results(result: rzip.RzipValidationResult) -> None:
    """Both real model verdicts survive sealed serialization and JSON decoding."""
    for observed in (result, rzip.validate_rzip_vertical_stability(exact_tol=1e-30)):
        payload = evidence.build_evidence(observed, target_id="rigid-reference")
        assert evidence.validate_evidence_payload(payload) is observed.passed
        assert rzip.validate_evidence_payload(json.loads(json.dumps(payload))) is observed.passed
        assert all(payload[name] == getattr(observed, name) for name in ("passed", "exact_tol", "marginal_tol"))
    assert result.passed is True
    assert observed.passed is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("passed",), "false", "must be a boolean"),
        (("passed",), 1, "must be a boolean"),
        (("passed",), False, "individual checks"),
        (("growth_passed",), False, "reported metrics"),
        (("frequency_passed",), 1, "must be a boolean"),
        (("generated_utc",), 3, "UTC timestamp"),
        (("generated_utc",), "not-a-time", "UTC timestamp"),
        (("generated_utc",), "2026-10-03T09:00:00", "UTC timestamp"),
        (("generated_utc",), "2026-10-03T09:00:00+01:00", "UTC timestamp"),
        (("target_id",), " ", "target_id"),
        (("target_id",), 1, "target_id"),
        (("config",), [], "declared fields"),
        (("config",), {}, "declared fields"),
        (("config", "r0"), True, "finite number"),
        (("config", "r0"), "1.7", "finite number"),
        (("config", "r0"), 10**400, "finite number"),
        (("config", "r0"), 0, "positive"),
        (("config", "a"), 1.7, "tokamak ordering"),
        (("unstable_indices",), [], "nonempty array"),
        (("unstable_indices",), "-1", "nonempty array"),
        (("unstable_indices",), [1], "declared sign"),
        (("unstable_indices",), [False], "finite number"),
        (("stable_indices",), [0], "declared sign"),
        (("exact_tol",), 0, "positive"),
        (("marginal_tol",), -1, "positive"),
        (("max_growth_rel_error",), -1, "non-negative"),
        (("max_growth_rel_error",), 1, "reported metrics"),
        (("max_frequency_rel_error",), 1, "reported metrics"),
        (("max_growth_time_rel_error",), 1, "reported metrics"),
        (("marginal_growth_rate",), 1, "reported metrics"),
        (("scaling",), [], "nonempty array"),
        (("scaling",), [{}], "three declared laws"),
        (("scaling", 0), [], "declared fields"),
        (("scaling", 0, "name"), 1, "known and unique"),
        (("scaling", 0, "name"), "unknown", "known and unique"),
        (("scaling", 1, "name"), "current_linear", "known and unique"),
        (("scaling", 0, "measured_ratio"), 0, "positive"),
        (("scaling", 0, "expected_ratio"), 3, "declared law"),
        (("scaling", 0, "rel_error"), 1, "declared law"),
        (("max_scaling_rel_error",), 1, "scaling checks"),
        (("wall",), {}, "declared fields"),
        (("wall", "no_wall_growth_rate"), 0, "positive"),
        (("wall", "wall_slows_growth"), 1, "must be a boolean"),
        (("wall", "wall_slows_growth"), False, "wall verdicts"),
        (("wall", "with_wall_finite"), False, "wall verdicts"),
    ],
)
def test_resealed_malformed_or_inconsistent_report(
    result: rzip.RzipValidationResult, path: tuple[str | int, ...], value: object, message: str
) -> None:
    """A matching seal cannot admit coerced domains, wrong laws or false verdicts."""
    payload = evidence.build_evidence(result, target_id="invalid-report")
    parent: Any = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        evidence.validate_evidence_payload(reseal(payload))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_report_requires_finite_json(result: rzip.RzipValidationResult, value: float) -> None:
    """A resealed nonfinite metric receives an authored refusal before comparison."""
    payload = evidence.build_evidence(result, target_id="nonfinite")
    payload["max_growth_rel_error"] = value
    with pytest.raises(ValueError, match="must be finite"):
        evidence.validate_evidence_payload(reseal(payload))


def test_schema_seal_and_exact_root_fields(result: rzip.RzipValidationResult) -> None:
    """Wrong roots, schemas, stale seals and undeclared fields cannot be admitted."""
    payload = evidence.build_evidence(result, target_id="schema")
    roots: tuple[object, ...] = ([], {"schema_version": "unknown"})
    for root in roots:
        with pytest.raises(ValueError, match="unsupported"):
            evidence.validate_evidence_payload(cast(Mapping[str, Any], root))
    for seal in (None, "short", "X" * 64):
        changed = dict(payload, payload_sha256=seal)
        with pytest.raises(ValueError, match="hex digest"):
            evidence.validate_evidence_payload(changed)
    with pytest.raises(ValueError, match="does not match"):
        evidence.validate_evidence_payload(dict(payload, target_id="changed"))
    with pytest.raises(ValueError, match="declared fields"):
        evidence.validate_evidence_payload(reseal(dict(payload, unknown=1)))
    with pytest.raises(ValueError, match="JSON-serializable"):
        evidence.validate_evidence_payload(dict(payload, unknown=object()))


def test_consistent_failed_checks_are_reports_not_admission(result: rzip.RzipValidationResult) -> None:
    """The decoder returns False for correctly declared numerical or wall failures."""
    payload = evidence.build_evidence(result, target_id="failed-reference")
    payload.update(max_growth_rel_error=1, growth_passed=False, passed=False)
    assert evidence.validate_evidence_payload(reseal(payload)) is False
    payload = evidence.build_evidence(result, target_id="wall-failure")
    payload["wall"]["with_wall_growth_rate"] = payload["wall"]["no_wall_growth_rate"]
    payload["wall"]["wall_slows_growth"] = False
    payload.update(wall_passed=False, passed=False)
    assert evidence.validate_evidence_payload(reseal(payload)) is False


def test_builder_rejects_false_flags_and_detaches_data(result: rzip.RzipValidationResult) -> None:
    """Serialization validates dataclass verdicts without mutating result fields."""
    for label in ("", " ", cast(str, 1)):
        with pytest.raises(ValueError, match="target_id"):
            evidence.build_evidence(result, target_id=label)
    with pytest.raises(ValueError, match="individual checks"):
        evidence.build_evidence(replace(result, passed=False), target_id="false-verdict")
    payload = evidence.build_evidence(result, target_id="detached")
    payload["config"]["r0"] = 2
    payload["unstable_indices"][0] = -3
    assert result.config.r0 == 1.7 and result.unstable_indices[0] == -2.5


def test_writer_roundtrip_overwrite_and_failed_status(result: rzip.RzipValidationResult, tmp_path: Path) -> None:
    """The direct writer preserves the checked payload and replaces both reports."""
    path = tmp_path / "rzip.json"
    payload = evidence.build_evidence(result, target_id="write-roundtrip")
    untouched = copy.deepcopy(payload)
    evidence.write_report(payload, path)
    assert json.loads(path.read_text()) == payload and payload == untouched
    assert "**pass**" in path.with_suffix(".md").read_text()
    failed = evidence.build_evidence(rzip.validate_rzip_vertical_stability(exact_tol=1e-30), target_id="failed")
    evidence.write_report(failed, path)
    assert json.loads(path.read_text()) == failed
    assert "**fail**" in path.with_suffix(".md").read_text()
    before = (path.read_bytes(), path.with_suffix(".md").read_bytes())
    with pytest.raises(ValueError, match="does not match"):
        evidence.write_report(dict(payload, target_id="stale-seal"), path)
    assert before == (path.read_bytes(), path.with_suffix(".md").read_bytes())


def test_writer_native_filesystem_refusals(result: rzip.RzipValidationResult, tmp_path: Path) -> None:
    """Absent parents and an unwritable Markdown sibling retain direct IO semantics."""
    payload = evidence.build_evidence(result, target_id="filesystem")
    absent = tmp_path / "absent" / "rzip.json"
    with pytest.raises(FileNotFoundError):
        evidence.write_report(payload, absent)
    assert not absent.parent.exists()
    partial = tmp_path / "partial.json"
    partial.with_suffix(".md").mkdir()
    with pytest.raises(IsADirectoryError):
        evidence.write_report(payload, partial)
    assert json.loads(partial.read_text()) == payload
    assert partial.with_suffix(".md").is_dir()


@pytest.mark.parametrize("name", ["exact_tol", "marginal_tol"])
@pytest.mark.parametrize("value", [True, "1e-9", 0, -1, float("nan"), float("inf"), 10**400])
def test_public_numerical_tolerance_refusals(name: str, value: Any) -> None:
    """Production validation rejects invalid tolerances before evaluating the model."""
    with pytest.raises(ValueError, match=name):
        rzip.validate_rzip_vertical_stability(**{name: value})


def test_public_analytic_domains() -> None:
    """Both analytic references reject nonfinite, coerced and wrong-sign indices."""
    config = rzip.default_config()
    for value in (True, "1", float("nan"), float("inf"), 10**400):
        with pytest.raises(ValueError):
            rzip.analytic_no_wall_growth_rate(config, cast(float, value))
        with pytest.raises(ValueError):
            rzip.analytic_no_wall_frequency(config, cast(float, value))
    with pytest.raises(ValueError, match="positive"):
        rzip.analytic_no_wall_frequency(config, 0)
    with pytest.raises(ValueError, match="negative"):
        rzip.analytic_no_wall_growth_rate(config, 0)
    with pytest.raises(ValueError, match="VerticalConfig"):
        rzip.validate_rzip_vertical_stability(config=cast(rzip.VerticalConfig, {}))


@pytest.mark.parametrize("operation", [rzip.no_wall_growth_time_consistency, rzip.wall_stabilisation])
@pytest.mark.parametrize("value", [0, 1, True, "-1", float("nan"), float("inf")])
def test_unstable_mode_operations_require_negative_indices(
    operation: Callable[[rzip.VerticalConfig, float], object], value: Any
) -> None:
    """Wall slowing and exponential growth time refuse stable or malformed indices."""
    with pytest.raises(ValueError, match="destabilising index"):
        operation(rzip.default_config(), value)


def test_cli_verdicts_and_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Real CLI execution emits both model verdicts and the requested sealed report."""
    assert rzip.main([]) == 0
    assert "Status: pass" in capsys.readouterr().out
    assert rzip.main(["--exact-tol", "1e-30"]) == 1
    assert "Status: fail" in capsys.readouterr().out
    path = tmp_path / "cli.json"
    assert rzip.main(["--json-out", "--marginal-tol", "1e-7", "--report", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(path.read_text())
    assert payload["marginal_tol"] == 1e-7
    assert evidence.validate_evidence_payload(payload) is True


@pytest.mark.parametrize("flag", ["--exact-tol", "--marginal-tol"])
@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_cli_refuses_invalid_tolerances(flag: str, value: str, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI maps numerical-domain refusals to argparse diagnostics and exit 2."""
    with pytest.raises(SystemExit) as caught:
        rzip.main([flag, value])
    assert caught.value.code == 2
    assert flag.removeprefix("--").replace("-", "_") in capsys.readouterr().err
