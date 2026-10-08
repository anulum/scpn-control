# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density report contract tests
"""Exercise density producer/consumer report refusals through public APIs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from validation import validate_density_control as module


def report() -> dict[str, object]:
    """Use the real default particle-balance calculation and producer."""
    return dict(module.build_evidence(module.validate_density_control(), target_id="actual-density"))


def seal(payload: dict[str, object]) -> dict[str, object]:
    """Apply the report's declared content-hash encoding to a negative case."""
    payload["payload_sha256"] = ""
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return payload


@pytest.mark.parametrize("strict", [False, True])
def test_actual_positive_and_negative_reports_round_trip(strict: bool) -> None:
    """Check genuine arithmetic agreement and a stricter precision failure."""
    result = module.validate_density_control(exact_tol=1e-30 if strict else 1e-9)
    assert result.passed is (not strict)
    payload = module.build_evidence(result, target_id="actual-round-trip")
    assert module.validate_evidence_payload(payload) is result.passed


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", "false"),
        ("greenwald_limit_rel_error", 1.0),
        ("nbi_conservation_rel_error", float("nan")),
        ("sources_passed", False),
        ("passed", False),
        ("exact_tol", 0.0),
        ("invariance_tol", -1.0),
        ("target_id", " "),
        ("target_id", 1),
        ("generated_utc", 1),
        ("generated_utc", "bad"),
        ("generated_utc", "2026-10-08T00:00:00"),
        ("generated_utc", "2026-10-08T00:00:00+01:00"),
        ("runtime_source_sha256", {}),
        ("runtime_source_sha256", {"not-source": "wrong"}),
    ],
)
def test_valid_hash_does_not_admit_bad_declarations(field: str, value: object) -> None:
    """Refuse malformed or contradictory fields even with a valid hash."""
    payload = report()
    payload[field] = value
    with pytest.raises(ValueError):
        module.validate_evidence_payload(seal(payload))


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", -1.0, 0.0, 2.0])
def test_recycling_config_checks_scalar_domain(value: object) -> None:
    """Reject coerced, nonfinite and out-of-range recycling coefficients."""
    constructor: Callable[..., module.DensityConfig] = module.DensityConfig
    raw = report()["config"]
    assert isinstance(raw, dict)
    values: dict[str, object] = dict(raw)
    values["recycling_coeff"] = value
    with pytest.raises(ValueError):
        constructor(**values)


@pytest.mark.parametrize("kind", ["missing", "extra", "legacy", "bad-hash", "mismatch", "non-json"])
def test_complete_v3_shape_and_digest(kind: str) -> None:
    """Require the complete upgraded checker and source-label contract."""
    payload = report()
    if kind == "missing":
        del payload["config"]
    elif kind == "extra":
        payload["unknown"] = 1
    elif kind == "legacy":
        payload["schema_version"] = "scpn-control.density-control-validation.v2"
    elif kind == "bad-hash":
        payload["payload_sha256"] = "z" * 64
    elif kind == "mismatch":
        payload["target_id"] = "changed"
    else:
        payload["unknown"] = object()
    if kind in {"missing", "extra", "legacy"}:
        seal(payload)
    with pytest.raises(ValueError):
        module.validate_evidence_payload(payload)


@pytest.mark.parametrize("value", [None, [], "text"])
def test_runtime_nonobject_roots(value: object) -> None:
    """Refuse nonobject roots at the public consumer boundary."""
    validator: Callable[..., bool] = module.validate_evidence_payload
    with pytest.raises(ValueError):
        validator(value)


@pytest.mark.parametrize("field,value", [("n_rho", True), ("recycling_coeff", True), ("minor_radius_m", 8.0)])
def test_nested_config_is_actually_validated(field: str, value: object) -> None:
    """Invoke the real configuration checks for decoded reports."""
    payload = report()
    raw = payload["config"]
    assert isinstance(raw, dict)
    payload["config"] = {**raw, field: value}
    with pytest.raises(ValueError):
        module.validate_evidence_payload(seal(payload))


@pytest.mark.parametrize("value", [None, [], [{"name": "only"}]])
def test_complete_scaling_array(value: object) -> None:
    """Refuse missing or partial declared Greenwald scaling laws."""
    payload = report()
    payload["scaling"] = value
    with pytest.raises(ValueError):
        module.validate_evidence_payload(seal(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "unknown"),
        ("name", 1),
        ("expected_ratio", 3.0),
        ("measured_ratio", 3.0),
        ("rel_error", True),
        ("rel_error", float("inf")),
        ("rel_error", -1.0),
        ("rel_error", 10**400),
    ],
)
def test_scaling_identity_numbers_and_arithmetic(field: str, value: object) -> None:
    """Reject wrong law identity, finite domains and derived errors."""
    payload = report()
    raw = payload["scaling"]
    assert isinstance(raw, list)
    changed = copy.deepcopy(raw)
    changed[0][field] = value
    payload["scaling"] = changed
    with pytest.raises(ValueError):
        module.validate_evidence_payload(seal(payload))


def test_duplicate_law_and_wrong_maximum() -> None:
    """Check array identity and aggregate scaling rather than cardinality only."""
    payload = report()
    raw = payload["scaling"]
    assert isinstance(raw, list)
    payload["scaling"] = [raw[0], raw[0]]
    with pytest.raises(ValueError, match="known and unique"):
        module.validate_evidence_payload(seal(payload))
    payload = report()
    payload["max_scaling_rel_error"] = 0.5
    with pytest.raises(ValueError, match="scaling maximum"):
        module.validate_evidence_payload(seal(payload))


def test_runtime_source_labels_require_all_declared_digests() -> None:
    """Require every checker/model source label to have a digest declaration."""
    payload = report()
    raw = payload["runtime_source_sha256"]
    assert isinstance(raw, dict)
    payload["runtime_source_sha256"] = {name: "invalid" for name in raw}
    with pytest.raises(ValueError, match="Runtime source digests"):
        module.validate_evidence_payload(seal(payload))


def test_builder_refuses_inconsistent_result() -> None:
    """Refuse a producer record with a passing flag and failed metric."""
    result = module.validate_density_control()
    with pytest.raises(ValueError, match="stage verdict"):
        module.build_evidence(replace(result, greenwald_limit_rel_error=1.0), target_id="bad-result")


@pytest.mark.parametrize("field,value", [("major_radius_m", "6.2"), ("major_radius_m", 10**400)])
def test_configuration_number_refusal_is_deliberate(field: str, value: object) -> None:
    """Refuse invalid scalar conversion through the actual public constructor."""
    raw = report()["config"]
    assert isinstance(raw, dict)
    constructor: Callable[..., module.DensityConfig] = module.DensityConfig
    with pytest.raises(ValueError):
        constructor(**{**raw, field: value})


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "invalid"])
def test_cli_positive_tolerance_refusal(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Reject invalid thresholds with an authored CLI argument message."""
    with pytest.raises(SystemExit) as outcome:
        module.main(["--exact-tol", value])
    assert outcome.value.code == 2
    assert "Tolerance must be a finite positive number" in capsys.readouterr().err


def test_cli_report_failure_preserves_nonregular_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse a real directory destination without a native traceback."""
    destination = tmp_path / "occupied"
    destination.mkdir()
    assert module.main(["--report", str(destination)]) == 2
    assert capsys.readouterr().err.strip() == "Density report could not be published"
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("strict", [False, True])
def test_native_cli_positive_and_negative_report(tmp_path: Path, strict: bool) -> None:
    """Exercise real native JSON/Markdown production and public decoding."""
    root = Path(module.__file__).resolve().parents[1]
    destination = tmp_path / "density.json"
    args = [sys.executable, "-m", "validation.validate_density_control", "--json-out", "--report", str(destination)]
    if strict:
        args += ["--exact-tol", "1e-30"]
    env = {
        **os.environ,
        "PYTHONPATH": str(root) + os.pathsep + str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == int(strict), result.stderr
    stored = json.loads(destination.read_text())
    assert json.loads(result.stdout) == stored
    assert module.validate_evidence_payload(stored) is (not strict)
    assert destination.with_suffix(".md").is_file()
