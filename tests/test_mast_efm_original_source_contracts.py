# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Complete original MAST source contracts and consumers

"""Exercise complete original declarations, file custody, CLI and actual trainer admission."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mast_efm_zarr_fixtures import write_zarr
from test_mast_efm_original_feature_source_audit import _load_source, _write_dataset_report

from validation.audit_mast_efm_feature_provenance import build_audit
from validation.audit_mast_efm_feature_provenance import write_report as write_feature_report
from validation.audit_mast_efm_original_feature_sources import (
    build_original_feature_source_audit,
    main,
    parse_args,
    validate_original_audit_report,
    write_report,
)
from validation.mast_efm_original_source_inputs import inspect_original_sources, load_zarr_candidate_metadata
from validation.mast_efm_original_source_policy import classify_feature_sources
from validation.mast_efm_original_source_reporting import validate_original_audit_bindings
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.plan_neural_equilibrium_training_campaign import CampaignInputs, build_plan
from validation.train_mast_efm_neural_equilibrium import TrainingInputs, build_training_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("field", ["attrs", "dtype_type", "dtype_invalid"])
def test_source_policy_refuses_unsupported_preferred_descriptors(ready_audit: dict[str, Any], field: str) -> None:
    """Preferred names with malformed attributes or dtype declarations cannot select a feature source."""
    variables = copy.deepcopy(ready_audit["shots"][0]["source_variables"])
    descriptor = variables["plasma_current_x"]
    if field == "attrs":
        descriptor["attrs"] = []
    else:
        descriptor["dtype"] = 1 if field == "dtype_type" else "unsupported-observation-dtype"
    result = classify_feature_sources(variables)
    assert result["Ip_MA"]["status"] == "blocked"
    assert result["Ip_MA"]["selected_source"] is None
    assert result["Bt_T"]["selected_source"] == "bphi_rmag"


def test_candidate_metadata_reads_real_store_and_refuses_malformed_descriptors(tmp_path: Path) -> None:
    """Descriptor inspection reads the actual consolidated file and refuses malformed candidate objects."""
    store = write_zarr(tmp_path / "original.zarr")
    observed = load_zarr_candidate_metadata(store)
    assert observed["plasma_current_x"]["units"] == "A"
    path = store / ".zmetadata"
    metadata = json.loads(path.read_text())
    metadata["metadata"]["plasma_current_x/.zattrs"] = []
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="invalid original candidate metadata"):
        load_zarr_candidate_metadata(store)
    with pytest.raises(FileNotFoundError, match="consolidated Zarr metadata is missing"):
        load_zarr_candidate_metadata(tmp_path / "absent.zarr")


@pytest.mark.parametrize(
    "case",
    [
        "missing_time",
        "time_shape",
        "time_dtype",
        "unordered_time",
        "unknown_time",
        "missing_array",
        "mask_dtype",
        "shot_dtype",
        "target_dtype",
    ],
)
def test_original_inspection_refuses_actual_changed_reference_observations(
    ready_audit: dict[str, Any], tmp_path: Path, case: str
) -> None:
    """Even a correct new reference SHA cannot admit missing, reordered or wrongly typed observed arrays."""
    storage = tmp_path / "storage"
    shutil.copytree(ready_audit["storage_root"], storage)
    dataset = json.loads(Path(ready_audit["dataset_report"]).read_text())
    selected = dataset["shots"][0]
    reference = storage / selected["reference_path"]
    with np.load(reference, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if case == "missing_time":
        del arrays["time_s"]
    elif case == "time_shape":
        arrays["time_s"] = arrays["time_s"][:, None]
    elif case == "time_dtype":
        arrays["time_s"] = arrays["time_s"].astype(str)
    elif case == "unordered_time":
        arrays["time_s"] = arrays["time_s"][::-1]
    elif case == "unknown_time":
        arrays["time_s"] = arrays["time_s"] + 1
    elif case == "missing_array":
        del arrays["psi_axis_Wb_per_rad"]
    elif case == "mask_dtype":
        arrays["pprime_valid_mask"] = arrays["pprime_valid_mask"].astype(int)
    elif case == "shot_dtype":
        arrays["shot_id"] = arrays["shot_id"].astype(float)
    else:
        arrays["psi_axis_Wb_per_rad"] = arrays["psi_axis_Wb_per_rad"].astype(str)
    np.savez_compressed(reference, **arrays)
    selected["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    observed = inspect_original_sources(dataset, storage)
    assert observed[0]["conversion_check"]["status"] == "blocked"
    assert observed[0]["conversion_check"]["errors"]
    assert observed[0]["conversion_check"]["matched_reference_count"] == 0
    assert all(shot["conversion_check"]["status"] == "pass" for shot in observed[1:])


@pytest.fixture(scope="module")
def ready_audit(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build immutable shared engineering observations through real conversion and producer APIs."""
    report = tmp_path_factory.mktemp("original-source-contracts") / "dataset.json"
    storage = _write_dataset_report(report)
    audit = build_original_feature_source_audit(report, storage)
    assert audit["status"] == "source_ready"
    return audit


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "scpn-control.mast-efm-original-feature-source-audit.v1"),
        (("converted_feature_audit",), None),
        (("dataset_report",), "other.json"),
        (("storage_root",), "other-root"),
        (("reference_dataset_id",), "other-campaign"),
        (("fallback_features",), []),
        (("shots",), None),
        (("shot_count",), 1),
        (("shots", 0), "invalid shot"),
        (("shots", 0, "shot_id"), True),
        (("shots", 0, "reference_path"), "converted/other.npz"),
        (("shots", 0, "reference_sha256"), "0" * 64),
        (("shots", 0, "equilibria_count"), True),
        (("shots", 0, "zarr_path"), "mast/level1/shot_30420/efm.zarr"),
        (("shots", 0, "source_variables"), []),
        (("shots", 0, "source_variables"), {"unknown_source": {}}),
        (("shots", 0, "source_variables"), {"plasma_current_x": 1}),
        (("shots", 0, "feature_status"), {}),
        (("shots", 0, "source_snapshot"), None),
        (("shots", 0, "source_snapshot", "files"), []),
        (("shots", 0, "source_snapshot", "files", 0), 5),
        (("shots", 0, "source_snapshot", "files", 0, "path"), "../outside"),
        (("shots", 0, "source_snapshot", "files", 0, "path"), "/absolute"),
        (("shots", 0, "source_snapshot", "files", 0, "path"), "C:/absolute"),
        (("shots", 0, "source_snapshot", "files", 0, "path"), "a\\b"),
        (("shots", 0, "source_snapshot", "files", 0, "path"), "a//b"),
        (("shots", 0, "source_snapshot", "files", 0, "sha256"), "invalid"),
        (("shots", 0, "source_snapshot", "files", 0, "size_bytes"), True),
        (("shots", 0, "source_snapshot", "files", 0, "size_bytes"), -1),
        (("shots", 0, "source_snapshot", "metadata_sha256"), "0" * 64),
        (("shots", 0, "source_snapshot", "snapshot_sha256"), "0" * 64),
        (("shots", 0, "conversion_check"), None),
        (("shots", 0, "conversion_check", "status"), "other"),
        (("shots", 0, "conversion_check", "errors"), "not a list"),
        (("shots", 0, "conversion_check", "errors"), [""]),
        (("shots", 0, "conversion_check", "observed_time_count"), True),
        (("shots", 0, "conversion_check", "observed_time_count"), None),
        (("shots", 0, "conversion_check", "observed_time_count"), 3),
        (("shots", 0, "conversion_check", "reference_arrays_match"), 1),
        (("shots", 0, "conversion_check", "matched_reference_count"), True),
        (("shots", 0, "conversion_check", "matched_reference_count"), 0),
        (("feature_status",), {}),
        (("blocked_features",), ["Ip_MA"]),
        (("conversion_blocked_shots",), [30419]),
        (("can_rebuild_dataset_now",), False),
        (("status",), "blocked"),
        (("next_processing_steps",), None),
        (("next_processing_steps",), []),
        (("next_processing_steps",), [""]),
    ],
)
def test_resealed_original_declarations_refuse_incomplete_or_conflicting_evidence(
    ready_audit: dict[str, Any], path: tuple[str | int, ...], value: Any
) -> None:
    """A newly consistent outer digest cannot turn conflicting source/reference declarations into readiness."""
    audit = copy.deepcopy(ready_audit)
    selected: Any = audit
    for key in path[:-1]:
        selected = selected[key]
    selected[path[-1]] = value
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    with pytest.raises(ValueError):
        validate_original_audit_report(audit)


def test_original_report_requires_current_self_digest_and_sorted_unique_manifest(ready_audit: dict[str, Any]) -> None:
    """Stale outer bytes and reordered per-file declarations refuse independently."""
    audit = copy.deepcopy(ready_audit)
    audit["next_processing_steps"].append("new step")
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_original_audit_report(audit)
    audit = copy.deepcopy(ready_audit)
    audit["shots"][0]["source_snapshot"]["files"].reverse()
    audit["payload_sha256"] = canonical_campaign_digest(audit)
    with pytest.raises(ValueError, match="sorted/distinct"):
        validate_original_audit_report(audit)


@pytest.mark.parametrize("field", ["dataset_sha256", "dataset_path", "candidate_report", "reference_dataset_id"])
def test_original_source_bindings_consume_the_actual_selected_dataset(ready_audit: dict[str, Any], field: str) -> None:
    """Complete original declarations must bind the selected producer identity, locators and exact tensor digest."""
    dataset = json.loads(Path(ready_audit["dataset_report"]).read_text())
    raw_sha = hashlib.sha256(Path(ready_audit["dataset_report"]).read_bytes()).hexdigest()
    assert validate_original_audit_bindings(ready_audit, dataset, dataset_report_sha256=raw_sha) is ready_audit
    dataset[field] = "0" * 64 if field == "dataset_sha256" else "other"
    dataset["payload_sha256"] = canonical_campaign_digest(dataset)
    with pytest.raises(ValueError):
        validate_original_audit_bindings(ready_audit, dataset, dataset_report_sha256=raw_sha)


@pytest.mark.parametrize(
    "kind",
    [
        "dataset",
        "candidate",
        "reference",
        "source",
        "inside_source",
        "same_outputs",
        "symlink",
        "hardlink",
        "second_write",
    ],
)
def test_original_writer_protects_selected_inputs_and_reports_sequential_io(
    ready_audit: dict[str, Any], tmp_path: Path, kind: str
) -> None:
    """Actual output paths cannot overwrite selected input custody; second-file failure does not imply rollback."""
    storage = Path(ready_audit["storage_root"])
    shot = ready_audit["shots"][0]
    source = storage / shot["zarr_path"]
    converted = ready_audit["converted_feature_audit"]
    protected = {
        "dataset": Path(ready_audit["dataset_report"]),
        "candidate": storage / converted["candidate_report"],
        "reference": storage / shot["reference_path"],
        "source": source / ".zmetadata",
        "inside_source": source / "new-output.json",
    }
    json_out, markdown_out = tmp_path / "audit.json", tmp_path / "audit.md"
    if kind in protected:
        json_out = protected[kind]
    elif kind == "same_outputs":
        markdown_out = json_out
    elif kind == "symlink":
        json_out.symlink_to(source / ".zmetadata")
    elif kind == "hardlink":
        os.link(source / ".zmetadata", json_out)
    else:
        markdown_out.mkdir()
    before = json_out.read_bytes() if json_out.is_file() else None
    with pytest.raises(ValueError, match="cannot write original-source audit"):
        write_report(ready_audit, json_out, markdown_out)
    if kind == "second_write":
        assert json.loads(json_out.read_text()) == ready_audit
    elif before is None:
        assert not json_out.exists()
    else:
        assert json_out.read_bytes() == before


@pytest.mark.parametrize("kind", ["missing_time", "no_convergence", "vacuum_alias", "unrelated_target"])
def test_actual_original_observation_failures_block_the_real_trainer(tmp_path: Path, kind: str) -> None:
    """All four former metadata-only false admissions refuse on actual changed original stores before weights."""
    report = tmp_path / "dataset.json"
    storage = _write_dataset_report(report)
    store = storage / "mast/level1/shot_30419/efm.zarr"
    ds = _load_source(store)
    if kind == "missing_time":
        ds = ds.drop_vars("time")
    elif kind == "no_convergence":
        ds["cnvrgd_times"].values[:] = 0
    elif kind == "vacuum_alias":
        ds = ds.rename({"bphi_rmag": "bvac_rmag"})
    else:
        ds["pprime"].values[:] += 0.125
    write_zarr(store, ds)
    original = build_original_feature_source_audit(report, storage)
    assert original["status"] == "blocked" and original["conversion_blocked_shots"] == [30419]
    assert original["shots"][0]["conversion_check"]["errors"]
    assert all(shot["conversion_check"]["status"] == "pass" for shot in original["shots"][1:])
    feature = tmp_path / "feature.json"
    original_path = tmp_path / "original.json"
    write_feature_report(build_audit(report, storage), feature, feature.with_suffix(".md"))
    write_report(original, original_path, original_path.with_suffix(".md"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(build_plan(CampaignInputs(report, storage, ROOT / "validation/reference_data/qlknn"))))
    inputs = TrainingInputs(
        report,
        plan,
        storage / "processed/dataset.npz",
        tmp_path / "UNWRITTEN-weights.npz",
        feature,
        original_path,
        compute_host_kind="workstation",
    )
    assert build_training_report(inputs)["pre_run_admission"]["source_provenance"]["status"] == "fail"
    assert not inputs.weights_out.exists()


def test_original_audit_matches_a_selected_subset_of_actual_source_times(tmp_path: Path) -> None:
    """A bounded public conversion remains a valid exact subset of a longer observed original store."""
    report = tmp_path / "dataset.json"
    storage = _write_dataset_report(report, max_times=2)
    audit = build_original_feature_source_audit(report, storage)
    assert audit["status"] == "source_ready"
    assert all(shot["conversion_check"]["observed_time_count"] == 4 for shot in audit["shots"])
    assert all(shot["conversion_check"]["matched_reference_count"] == 2 for shot in audit["shots"])


def test_actual_original_source_cli_defaults_help_usage_and_cold_execution(
    ready_audit: dict[str, Any], tmp_path: Path
) -> None:
    """Public parsing and a cold standalone script emit a full current report with authored operational exits."""
    defaults = parse_args([])
    assert defaults.dataset_report.name == "mast_efm_neural_equilibrium_dataset.json"
    for option, code in [("--help", 0), ("--unsupported", 2)]:
        with pytest.raises(SystemExit) as error:
            main([option])
        assert error.value.code == code
    args = [
        "--dataset-report",
        ready_audit["dataset_report"],
        "--storage-root",
        ready_audit["storage_root"],
        "--json-out",
        str(tmp_path / "actual.json"),
        "--report-out",
        str(tmp_path / "actual.md"),
    ]
    assert main(args) == 0
    assert json.loads((tmp_path / "actual.json").read_text())["status"] == "source_ready"
    assert "Local conversion equivalence" in (tmp_path / "actual.md").read_text()
    cold = subprocess.run(
        [sys.executable, str(ROOT / "validation/audit_mast_efm_original_feature_sources.py"), *args],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        capture_output=True,
        text=True,
    )
    assert cold.returncode == 0 and "Traceback" not in cold.stderr
    args[args.index("--dataset-report") + 1] = str(tmp_path / "absent.json")
    assert main(args) == 1
    assert json.loads((tmp_path / "actual.json").read_text())["status"] == "source_ready"
