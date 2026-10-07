# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real RZIP calibration and benchmark admission contracts.
"""Exercise checked observations and real model/CLI/IO through public entry points."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_control.control import rzip_model as model
from scpn_control.core.vessel_model import VesselElement, VesselModel

# Windows reports a directory opened as a file as a permission error.
DIRECTORY_AS_FILE_ERROR: type[OSError] = PermissionError if os.name == "nt" else IsADirectoryError


@pytest.fixture(scope="module")
def plant() -> model.RZIPModel:
    """Construct the actual two-loop unstable reference plant."""
    return model.RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=VesselModel(
            [
                VesselElement(R=2.0, Z=0.5, resistance=1e-3, cross_section=0.1, inductance=1e-5),
                VesselElement(R=2.0, Z=-0.5, resistance=1e-3, cross_section=0.1, inductance=1e-5),
            ]
        ),
    )


@pytest.fixture(scope="module")
def declared(plant: model.RZIPModel) -> model.RZIPCalibrationEvidence:
    """Capture a matching, self-declared external comparison without authenticating it."""
    return model.rzip_calibration_evidence(
        plant,
        source="external_code_benchmark",
        source_id="local-contract-probe",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=plant.vertical_growth_rate(),
        growth_rate_relative_tolerance=0.02,
    )


def _reseal(evidence: model.RZIPCalibrationEvidence) -> model.RZIPCalibrationEvidence:
    """Independently hash the historical unsigned ASCII JSON fields."""
    payload = asdict(evidence)
    payload.pop("evidence_payload_sha256")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return replace(evidence, evidence_payload_sha256=digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("schema_version", "1"),
        ("source", "unknown"),
        ("source", []),
        ("source_id", " "),
        ("source_id", 3),
        ("model_id", ""),
        ("model_id", {}),
        ("vertical_inertia_kg", 0.0),
        ("vertical_inertia_kg", True),
        ("vertical_inertia_kg", 10**500),
        ("wall_time_constant_s", math.inf),
        ("wall_time_constant_s", -1.0),
        ("growth_rate_relative_tolerance", 0.0),
        ("growth_rate_relative_tolerance", False),
        ("growth_rate_relative_tolerance", "0.2"),
        ("growth_rate_s_inv", -1.0),
        ("growth_rate_s_inv", math.nan),
        ("growth_rate_s_inv", math.inf),
        ("growth_rate_s_inv", True),
        ("growth_rate_s_inv", None),
        ("growth_time_ms", math.nan),
        ("growth_time_ms", math.inf),
        ("growth_time_ms", -1.0),
        ("growth_time_ms", 1.0),
        ("growth_time_ms", "1.0"),
        ("reference_growth_rate_s_inv", None),
        ("reference_growth_rate_s_inv", 0.0),
        ("reference_growth_rate_s_inv", True),
        ("reference_growth_rate_s_inv", "1.0"),
        ("growth_rate_relative_error", -1.0),
        ("growth_rate_relative_error", math.nan),
        ("growth_rate_relative_error", math.inf),
        ("growth_rate_relative_error", True),
        ("growth_rate_relative_error", None),
        ("growth_rate_relative_error", 0.5),
        ("facility_claim_allowed", "false"),
        ("facility_claim_allowed", 1),
        ("facility_claim_allowed", False),
        ("claim_status", "facility validated"),
    ],
)
def test_resealed_malformed_observation_refuses_before_io(
    declared: model.RZIPCalibrationEvidence,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    """A self-seal cannot admit contradictory metrics, nonliteral flags or widened status."""
    corrupt = _reseal(replace(declared, **{field: value}))
    output = tmp_path / "missing-parent" / "report.json"
    with pytest.raises(ValueError):
        model.save_rzip_calibration_evidence(corrupt, output)
    assert not output.parent.exists()
    with pytest.raises(ValueError):
        model.assert_rzip_facility_claim_admissible(corrupt)


@pytest.mark.parametrize(
    "source",
    [
        "local_regression_reference",
        "external_code_benchmark",
        "documented_public_reference",
        "measured_discharge",
    ],
)
@pytest.mark.parametrize("factor", [None, 1.01, 1.5])
def test_actual_declared_source_states(
    plant: model.RZIPModel,
    tmp_path: Path,
    source: str,
    factor: float | None,
) -> None:
    """Missing/mismatching/local sources remain writable coherent non-admissions."""
    evidence = model.rzip_calibration_evidence(
        plant,
        source=source,
        source_id=" stated source ",
        model_id=" bounded plant ",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=None if factor is None else factor * plant.vertical_growth_rate(),
        growth_rate_relative_tolerance=0.02,
    )
    assert evidence.source_id == "stated source" and evidence.model_id == "bounded plant"
    destination = tmp_path / "nested" / "report.json"
    model.save_rzip_calibration_evidence(evidence, destination)
    assert json.loads(destination.read_text()) == asdict(evidence)
    admitted = source != "local_regression_reference" and factor == 1.01
    assert evidence.facility_claim_allowed is admitted
    if admitted:
        assert model.assert_rzip_facility_claim_admissible(evidence) is evidence
    else:
        with pytest.raises(ValueError):
            model.assert_rzip_facility_claim_admissible(evidence)


@pytest.mark.parametrize("source", ["local_regression_reference", "external_code_benchmark"])
@pytest.mark.parametrize("reference", [None, 1.0])
def test_stable_growth_time_is_extended_real_non_admission(
    tmp_path: Path, source: str, reference: float | None
) -> None:
    """Zero growth preserves positive infinite time without an external admission flag."""
    plant = model.RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=VesselModel([]))
    evidence = model.rzip_calibration_evidence(
        plant,
        source=source,
        source_id="stable-local-observation",
        wall_time_constant_s=0.01,
        reference_growth_rate_s_inv=reference,
        growth_rate_relative_tolerance=2.0,
    )
    assert evidence.growth_rate_s_inv == 0 and evidence.growth_time_ms == math.inf
    assert evidence.facility_claim_allowed is False
    out = tmp_path / "stable.json"
    model.save_rzip_calibration_evidence(evidence, out)
    assert json.loads(out.read_text())["growth_time_ms"] == math.inf
    with pytest.raises(ValueError):
        model.assert_rzip_facility_claim_admissible(evidence)


def test_digest_and_reference_replacement_refuse(declared: model.RZIPCalibrationEvidence, tmp_path: Path) -> None:
    """Content changes cannot evade the seal or computed comparison check."""
    stale = replace(declared, source_id="changed")
    with pytest.raises(ValueError, match="payload digest"):
        model.save_rzip_calibration_evidence(stale, tmp_path / "stale.json")
    bad_reference = _reseal(replace(declared, reference_growth_rate_s_inv=2 * declared.growth_rate_s_inv))
    with pytest.raises(ValueError, match="comparison error"):
        model.assert_rzip_facility_claim_admissible(bad_reference)
    with pytest.raises(ValueError, match="must be RZIPCalibrationEvidence"):
        model.save_rzip_calibration_evidence(cast(Any, None), tmp_path / "bad.json")


@dataclass(frozen=True)
class ExtendedEvidence(model.RZIPCalibrationEvidence):
    """A real dataclass extension whose extra field is outside the v1 wire contract."""

    extra_field: int = 1


def test_unrepresentable_growth_time_remains_non_admitted(
    declared: model.RZIPCalibrationEvidence, tmp_path: Path
) -> None:
    """A coherent positive rate with an infinite floating-point time cannot admit a facility flag."""
    extended = _reseal(
        replace(
            declared,
            growth_rate_s_inv=1e-309,
            growth_time_ms=math.inf,
            reference_growth_rate_s_inv=1e-309,
            facility_claim_allowed=False,
            claim_status="external RZIP reference admission requires finite positive model growth",
        )
    )
    model.save_rzip_calibration_evidence(extended, tmp_path / "extended.json")
    with pytest.raises(ValueError, match="not admissible"):
        model.assert_rzip_facility_claim_admissible(extended)


def test_extended_dataclass_is_not_silently_serialised(declared: model.RZIPCalibrationEvidence, tmp_path: Path) -> None:
    """Unsupported additional wire fields refuse through the public writer."""
    with pytest.raises(ValueError, match="fields"):
        model.save_rzip_calibration_evidence(ExtendedEvidence(**asdict(declared)), tmp_path / "extra.json")


def test_writer_replaces_after_validation_and_propagates_io(
    declared: model.RZIPCalibrationEvidence, tmp_path: Path
) -> None:
    """Existing output is replaced; a real directory cannot act as a JSON file."""
    out = tmp_path / "report.json"
    out.write_text("old")
    model.save_rzip_calibration_evidence(declared, out)
    assert json.loads(out.read_text()) == asdict(declared)
    with pytest.raises(DIRECTORY_AS_FILE_ERROR):
        model.save_rzip_calibration_evidence(declared, tmp_path)


@pytest.mark.parametrize("value", [None, "not-a-number", 10**500])
def test_bad_builder_scale_conversion_is_authored(plant: model.RZIPModel, value: Any) -> None:
    """Actual scale conversion failures raise a named ValueError before producing evidence."""
    with pytest.raises(ValueError, match="wall_time_constant_s must be finite and positive"):
        model.rzip_calibration_evidence(
            plant, source="local_regression_reference", source_id="real-case", wall_time_constant_s=value
        )
