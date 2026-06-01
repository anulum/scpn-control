# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium training tests

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA, FEATURE_NAMES
from validation.plan_neural_equilibrium_training_campaign import REPORT_SCHEMA as PLAN_SCHEMA
from validation.train_mast_efm_neural_equilibrium import (
    RESULT_TEMPLATES_SCHEMA,
    TRAINING_SCHEMA,
    TrainingInputs,
    build_result_templates,
    build_training_report,
    validate_result_templates,
    validate_training_report,
    write_result_templates,
    write_report,
)


def _write_payloads(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    dataset = tmp_path / "mast_efm_supervised_dataset.npz"
    n = 6
    features = np.column_stack([np.linspace(1.0 + col, 2.0 + col, n) for col in range(len(FEATURE_NAMES))])
    split = np.array(["train", "train", "train", "validation", "test", "test"])
    z, r = 3, 4
    base = np.arange(n * z * r, dtype=np.float64).reshape(n, z, r) / 10.0
    psirz_mask = np.ones_like(base, dtype=bool)
    base[0, 0, 0] = np.nan
    psirz_mask[0, 0, 0] = False
    lcfs_r = np.tile(np.linspace(0.4, 0.8, 5), (n, 1))
    lcfs_z = np.tile(np.linspace(-0.2, 0.2, 5), (n, 1))
    lcfs_mask = np.ones((n, 5), dtype=bool)
    lcfs_mask[0, 4] = False
    lcfs_r[0, 4] = np.nan
    lcfs_z[0, 4] = np.nan
    np.savez_compressed(
        dataset,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        split=split,
        shot_id=np.array([1, 1, 1, 2, 3, 3]),
        time_s=np.linspace(0.0, 0.5, n),
        r_grid_m=np.linspace(0.0, 1.0, r),
        z_grid_m=np.linspace(-1.0, 1.0, z),
        psirz_Wb_per_rad=base,
        psirz_valid_mask=psirz_mask,
        psi_axis_Wb_per_rad=np.linspace(0.0, 0.1, n),
        psi_boundary_Wb_per_rad=np.linspace(1.0, 1.1, n),
        pprime_Pa_per_Wb_rad=np.column_stack([np.linspace(1.0, 2.0, n), np.linspace(2.0, 3.0, n)]),
        pprime_valid_mask=np.ones((n, 2), dtype=bool),
        q_profile=np.column_stack([np.linspace(2.0, 3.0, n), np.linspace(3.0, 4.0, n)]),
        q_profile_valid_mask=np.ones((n, 2), dtype=bool),
        lcfs_r_m=lcfs_r,
        lcfs_z_m=lcfs_z,
        lcfs_valid_mask=lcfs_mask,
        lcfs_point_count=np.array([4, 5, 5, 5, 5, 5]),
        magnetic_axis_r_m=np.linspace(0.6, 0.7, n),
        magnetic_axis_z_m=np.linspace(-0.01, 0.01, n),
    )
    import hashlib

    sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    dataset_report = tmp_path / "dataset.json"
    dataset_report.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA,
                "status": "blocked",
                "dataset_sha256": sha,
                "reference_dataset_id": "mast-efm-test",
                "split_counts": {"train": 3, "validation": 1, "test": 2},
                "fallback_features": [],
            }
        ),
        encoding="utf-8",
    )
    campaign_plan = tmp_path / "plan.json"
    campaign_plan.write_text(json.dumps({"schema_version": PLAN_SCHEMA, "status": "prepared"}), encoding="utf-8")
    feature_provenance = tmp_path / "feature_provenance.json"
    feature_provenance.write_text(
        json.dumps(
            {
                "schema_version": "scpn-control.mast-efm-feature-provenance-audit.v1",
                "reference_dataset_id": "mast-efm-test",
                "blocked_features": [],
                "payload_sha256": "b" * 64,
                "feature_status": {
                    "Ip_MA": {"status": "resolved"},
                    "Bt_T": {"status": "resolved"},
                    "ffprime_scale": {"status": "resolved"},
                },
            }
        ),
        encoding="utf-8",
    )
    original_source = tmp_path / "original_source.json"
    original_source.write_text(
        json.dumps(
            {
                "schema_version": "scpn-control.mast-efm-original-feature-source-audit.v1",
                "reference_dataset_id": "mast-efm-test",
                "status": "source_ready",
                "can_rebuild_dataset_now": True,
                "blocked_features": [],
                "payload_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "weights.npz"
    return dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source


def test_training_report_default_is_dry_run_and_does_not_write_weights(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)

    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    assert report["schema_version"] == TRAINING_SCHEMA
    assert report["status"] == "prepared"
    assert report["execution_mode"] == "dry_run"
    assert report["dataset_exists_on_this_host"] is True
    assert report["dataset_metadata"]["split_counts"] == {"train": 3, "validation": 1, "test": 2}
    assert report["fallback_features"] == []
    assert "ML350 is storage-only" in report["execution_host_policy"]
    assert report["pre_run_admission"]["source_provenance"]["status"] == "pass"
    assert report["pre_run_admission"]["compute_execution"]["status"] == "fail"
    assert any("compute host kind" in item for item in report["pre_run_admission"]["errors"])
    assert all("fallback" not in item for item in report["blocked_before_admission"])
    assert any("workstation or external cloud" in item for item in report["blocked_before_admission"])
    assert report["holdout_metrics"] is None
    assert validate_training_report(report) is report
    assert not weights.exists()


def test_training_report_execute_writes_weights_and_holdout_metrics(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)

    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
            compute_host_kind="workstation",
            compute_host_label="workstation-fixture",
            execute=True,
            max_flux_components=2,
        )
    )

    assert report["status"] == "executed"
    assert report["execution_mode"] == "execute"
    assert report["pre_run_admission"]["status"] == "pass"
    assert report["weights_sha256"]
    assert report["holdout_metrics"]["validation"]["psi_rmse_Wb_per_rad"] is not None
    assert report["holdout_metrics"]["test"]["magnetic_axis_rmse_m"] is not None
    assert validate_training_report(report, require_executed=True) is report
    with np.load(weights, allow_pickle=False) as payload:
        assert payload["flux_components"].shape == (2, 12)
        assert payload["axis_regression"].shape[1] == 2


def test_training_execute_refuses_storage_output_and_unadmitted_host(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, _, feature_provenance, original_source = _write_payloads(tmp_path)

    with pytest.raises(ValueError, match="compute host kind"):
        build_training_report(
            TrainingInputs(
                dataset_report=dataset_report,
                campaign_plan=campaign_plan,
                dataset_path=dataset,
                weights_out=Path("/mnt/data_sas/DATASETS/SCPN-CONTROL/models/weights.npz"),
                feature_provenance_report=feature_provenance,
                original_source_report=original_source,
                execute=True,
            )
        )

    with pytest.raises(ValueError, match="weights_out must not be under ML350 SAS storage"):
        build_training_report(
            TrainingInputs(
                dataset_report=dataset_report,
                campaign_plan=campaign_plan,
                dataset_path=dataset,
                weights_out=Path("/mnt/data_sas/DATASETS/SCPN-CONTROL/models/weights.npz"),
                feature_provenance_report=feature_provenance,
                original_source_report=original_source,
                compute_host_kind="external_cloud",
                compute_host_label="cloud-fixture",
                execute=True,
            )
        )


def test_training_execute_refuses_failed_source_provenance(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    feature_payload = json.loads(feature_provenance.read_text(encoding="utf-8"))
    feature_payload["blocked_features"] = ["Ip_MA"]
    feature_provenance.write_text(json.dumps(feature_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feature provenance report still has blocked features"):
        build_training_report(
            TrainingInputs(
                dataset_report=dataset_report,
                campaign_plan=campaign_plan,
                dataset_path=dataset,
                weights_out=weights,
                feature_provenance_report=feature_provenance,
                original_source_report=original_source,
                compute_host_kind="workstation",
                compute_host_label="workstation-fixture",
                execute=True,
            )
        )


def test_result_templates_bind_training_report_and_required_outputs(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    templates = build_result_templates(report)
    write_result_templates(templates, tmp_path / "templates.json", tmp_path / "templates.md")

    assert templates["schema_version"] == RESULT_TEMPLATES_SCHEMA
    assert templates["expected_dataset_sha256"] == report["dataset_sha256"]
    assert "psi_rmse_Wb_per_rad" in templates["holdout_metrics"]["required_metrics"]
    assert "p99_ms" in templates["latency_metrics"]["required_fields"]
    assert "strict_reference_report_sha256" in templates["admission_certificate"]["required_fields"]
    assert validate_result_templates(templates, training_report=report) is templates
    markdown = (tmp_path / "templates.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Result Templates" in markdown


def test_training_report_validation_rejects_digest_and_policy_tampering(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    tampered = dict(report)
    tampered["blocked_before_admission"] = []
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_training_report(tampered)

    tampered = dict(report)
    tampered["execution_host_policy"] = "training may run anywhere"
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_training_report(tampered)


def test_result_templates_validation_rejects_report_binding_drift(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )
    templates = build_result_templates(report)

    drifted = dict(templates)
    drifted["training_report_payload_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_result_templates(drifted, training_report=report)

    drifted = json.loads(json.dumps(templates))
    drifted["holdout_metrics"]["required_metrics"].remove("q_profile_rmse")
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_result_templates(drifted)


def test_write_report_records_execute_command_and_admission_boundary(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    write_report(report, tmp_path / "report.json", tmp_path / "report.md")

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Training Launch" in markdown
    assert "--execute" in markdown
    assert "not predictive EFIT/P-EFIT admission evidence" in markdown
    assert "ML350 is storage-only" in markdown
