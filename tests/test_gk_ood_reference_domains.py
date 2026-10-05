# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD finite and probability domain behaviour

"""Exercise numeric/metric declarations through public persisted JSON without fitting covariance or predictions."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_gk_ood_reference_contracts import declaration, inspect


@pytest.mark.parametrize("parent", ["training_distribution", "thresholds", "acceptance", "mahalanobis_metric"])
@pytest.mark.parametrize("value", [None, [], "object", True])
def test_nested_object_domains(tmp_path: Path, parent: str, value: object) -> None:
    """Wrong nested object shapes are field refusals and cannot set metadata acceptance."""
    report = inspect(tmp_path, declaration(**{parent: value}))
    assert report["status"] == "fail" and report["campaign_artifacts"] == 0
    assert parent in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("value", [None, [], True, 0, -1, 1.5])
def test_positive_integer_sample_count(tmp_path: Path, value: object) -> None:
    """Sample count is a positive nonboolean integer declaration, never a requested allocation."""
    payload = declaration()
    payload["training_distribution"]["sample_count"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and "sample_count" in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("value", [None, [], True, "", "   "])
def test_dataset_identity(tmp_path: Path, value: object) -> None:
    """Training distribution needs a nonblank dataset identity but fetches no named dataset."""
    payload = declaration()
    payload["training_distribution"]["dataset_id"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and "dataset_id" in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("field", ["mean", "std"])
@pytest.mark.parametrize("value", [None, {}, [], [1] * 9, [1] * 11, [True] * 10, ["1"] * 10, [10**400] * 10])
def test_finite_ten_element_distribution(tmp_path: Path, field: str, value: object) -> None:
    """Exactly ten representable nonboolean numbers are required for both declared distribution arrays."""
    payload = declaration()
    payload["training_distribution"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("std", [[-1.0] * 10, [0.0] * 9 + [-5e-324]])
def test_negative_standard_deviation_refused(tmp_path: Path, std: list[float]) -> None:
    """Finite negative standard deviations cannot describe a distribution even when all other fields pass."""
    payload = declaration()
    payload["training_distribution"]["std"] = std
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["campaign_artifacts"] == 0
    assert any(error["error"] == "standard deviations must be non-negative" for error in report["errors"])


def test_signed_means_zero_std_and_uncapped_metadata_count(tmp_path: Path) -> None:
    """Signed means, zero/subnormal deviations and uncapped integer metadata counts retain valid descriptive semantics."""
    payload = declaration()
    payload["training_distribution"].update(mean=[-1.0] * 10, std=[0.0, -0.0, 5e-324] + [1.0] * 7, sample_count=10**400)
    assert inspect(tmp_path, payload)["status"] == "pass"


@pytest.mark.parametrize("field", ["mahalanobis", "soft_sigma", "ensemble_disagreement"])
@pytest.mark.parametrize("value", [None, [], True, False, "1", 10**400, 0, -1, -5e-324])
def test_finite_positive_thresholds(tmp_path: Path, field: str, value: object) -> None:
    """Every detector threshold requires a finite representable positive nonboolean declaration."""
    payload = declaration()
    payload["thresholds"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize(
    "field",
    [
        "false_positive_rate",
        "false_negative_rate",
        "max_false_positive_rate",
        "max_false_negative_rate",
        "ood_recall",
        "min_ood_recall",
    ],
)
@pytest.mark.parametrize("value", [None, [], True, False, "1", 10**400, -0.01, 1.01])
def test_finite_probability_domains(tmp_path: Path, field: str, value: object) -> None:
    """All observed/maximum/minimum rate declarations require representable nonboolean values in inclusive [0, 1]."""
    payload = declaration()
    payload["acceptance"][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and field in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("value", [None, [], True, "", "   "])
def test_metric_method_identity(tmp_path: Path, value: object) -> None:
    """Covariance method must be nonblank author metadata; this does not establish its execution."""
    payload = declaration()
    payload["mahalanobis_metric"]["calibration_method"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and "calibration_method" in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("value", [None, [], True, "", "a" * 63, "a" * 65, "g" * 64, "+" + "a" * 63, "٠" * 64])
def test_covariance_digest_domain(tmp_path: Path, value: object) -> None:
    """Covariance digest requires exactly64 ASCII hex characters without retrieving matrix bytes."""
    payload = declaration()
    payload["mahalanobis_metric"]["covariance_inverse_sha256"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and "covariance_inverse_sha256" in {error["field"] for error in report["errors"]}


@pytest.mark.parametrize("value", [None, False, 1, "true", []])
def test_literal_metric_label(tmp_path: Path, value: object) -> None:
    """Positive-definite metadata requires literal true, with no matrix definiteness computation inferred."""
    payload = declaration()
    payload["mahalanobis_metric"]["positive_definite"] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and "positive_definite" in {error["field"] for error in report["errors"]}


def test_metric_feature_order_and_uppercase_digest(tmp_path: Path) -> None:
    """Metric feature order is exact and original uppercase ASCII digest declarations remain admitted."""
    payload = declaration()
    payload["mahalanobis_metric"]["covariance_inverse_sha256"] = "A" * 64
    assert inspect(tmp_path, payload)["status"] == "pass"
    payload["mahalanobis_metric"]["feature_order"] = payload["feature_schema"][::-1]
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and "feature_order" in {error["field"] for error in report["errors"]}
