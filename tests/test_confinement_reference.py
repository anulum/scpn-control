# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Confinement reference contract tests.

"""Exercise hash-bound confinement comparisons through the public reference adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation.multi_machine_validation import (
    ConfinementReference,
)


@pytest.fixture
def confinement_reference() -> ConfinementReference:
    """Bind the actual shipped calibration CSV and coefficients, without fabricated outputs."""
    root = Path(__file__).resolve().parents[1] / "validation/reference_data/itpa"
    csv_path = root / "hmode_confinement.csv"
    coeff = root / "ipb98y2_coefficients.json"
    return ConfinementReference(
        csv_path,
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        coeff,
        hashlib.sha256(coeff.read_bytes()).hexdigest(),
        0,
        "ITER",
        "design",
        "derived_calibration",
        0.1,
    )


def test_confinement_actual_reference_comparison(confinement_reference: ConfinementReference) -> None:
    """Real model results differ by operating point and may fail the declared comparison."""
    first = json.loads(confinement_reference.evaluate())
    second = json.loads(dataclasses.replace(confinement_reference, row_index=1, machine="JET", shot="92436").evaluate())
    assert first["predicted_tau_E_s"] == pytest.approx(3.6060077194126654, rel=1e-12)
    assert second["predicted_tau_E_s"] == pytest.approx(0.5201485510391346, rel=1e-12)
    assert first["comparison_pass"] is True
    assert second["comparison_pass"] is False
    assert first["inputs"]["Ploss_MW"] == 87.0
    assert first["reference_tau_E_s"] == 3.7
    assert first["source_row"]["shot"] == "design"
    assert first["reference_sha256"] == confinement_reference.csv_sha256
    for record in (first, second):
        assert record["actionable"] is record["facility_validated"] is record["source_class_verified"] is False
        assert record["training_domain"] == "not_assessed"


@pytest.mark.parametrize("which", ["csv", "coefficients"])
def test_confinement_refuses_changed_bound_bytes(
    confinement_reference: ConfinementReference, tmp_path: Path, which: str
) -> None:
    """A content change cannot keep the original reference/coefficient admission digest."""
    original = confinement_reference.csv_path if which == "csv" else confinement_reference.coefficients_path
    changed = tmp_path / original.name
    changed.write_bytes(original.read_bytes() + b"\n")
    case = (
        dataclasses.replace(confinement_reference, csv_path=changed)
        if which == "csv"
        else dataclasses.replace(confinement_reference, coefficients_path=changed)
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        case.evaluate()


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf"), True, np.bool_(True), np.bool_(False)])
def test_confinement_refuses_invalid_tolerance(confinement_reference: ConfinementReference, tolerance: Any) -> None:
    """A caller cannot manufacture success using an unbounded/nonfinite tolerance."""
    with pytest.raises(ValueError, match="tolerance"):
        dataclasses.replace(confinement_reference, relative_tolerance=tolerance).evaluate()


def test_confinement_refuses_wrong_operating_point(confinement_reference: ConfinementReference) -> None:
    """Selecting another row requires updating its explicit machine/shot identity."""
    with pytest.raises(ValueError, match="identity mismatch"):
        dataclasses.replace(confinement_reference, row_index=1).evaluate()


@pytest.mark.parametrize(
    "defect",
    [
        "negative_power",
        "nonfinite_reference",
        "duplicate_header",
        "extra_field",
        "missing_field",
        "missing_header",
        "blank_source",
        "comparison_overflow",
    ],
)
def test_confinement_valid_digest_does_not_admit_malformed_data(
    confinement_reference: ConfinementReference, tmp_path: Path, defect: str
) -> None:
    """Hash agreement establishes byte identity, while the reader still checks content."""
    lines = confinement_reference.csv_path.read_text().splitlines()
    fields = lines[0].split(",")
    values = lines[1].split(",")
    if defect == "negative_power":
        values[fields.index("Ploss_MW")] = "-87"
    elif defect == "nonfinite_reference":
        values[fields.index("tau_E_s")] = "nan"
    elif defect == "duplicate_header":
        fields[-1] = fields[0]
    elif defect == "extra_field":
        values.append("unexpected")
    elif defect == "missing_field":
        values.pop()
    elif defect == "missing_header":
        fields[-1] = "unrecognized"
    elif defect == "blank_source":
        values[fields.index("source")] = " "
    else:
        values[fields.index("tau_E_s")] = "1e-320"
    path = tmp_path / "reference.csv"
    path.write_text(",".join(fields) + "\n" + ",".join(values) + "\n")
    case = dataclasses.replace(
        confinement_reference, csv_path=path, csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        case.evaluate()


@pytest.mark.parametrize(
    "mutation", ["duplicate_root", "duplicate_exponent", "nonfinite_metadata", "overflow_metadata"]
)
def test_confinement_rejects_ambiguous_coefficients(
    confinement_reference: ConfinementReference, tmp_path: Path, mutation: str
) -> None:
    """Matching bytes must still describe a unique finite JSON coefficient object."""
    original = confinement_reference.coefficients_path.read_text()
    if mutation == "duplicate_root":
        changed = original.replace('"C": 0.0562', '"C": 99.0, "C": 0.0562', 1)
    elif mutation == "duplicate_exponent":
        changed = original.replace('"Ip_MA": 0.93', '"Ip_MA": 99.0, "Ip_MA": 0.93', 1)
    elif mutation == "nonfinite_metadata":
        changed = original.replace('"total_data_points": 5920', '"total_data_points": NaN', 1)
    else:
        changed = original.replace('"total_data_points": 5920', '"total_data_points": 1e9999', 1)
    assert changed != original
    path = tmp_path / "coefficients.json"
    path.write_text(changed)
    case = dataclasses.replace(
        confinement_reference,
        coefficients_path=path,
        coefficients_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="coefficient JSON"):
        case.evaluate()


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source_class": "held_out_empirical"}, "classification"),
        ({"row_index": True}, "integer"),
        ({"row_index": -1}, "integer"),
        ({"row_index": 1000000}, "outside"),
        ({"csv_sha256": "not-a-digest"}, "SHA256"),
        ({"machine": " "}, "identity"),
    ],
)
def test_confinement_rejects_invalid_reference_selection(
    confinement_reference: ConfinementReference, changes: dict[str, Any], message: str
) -> None:
    """Refuse unsupported evidence categories and malformed explicit reference selectors."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(confinement_reference, **changes).evaluate()
