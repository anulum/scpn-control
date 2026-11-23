# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium campaign tests
"""Publication of MAST EFM campaign evidence from verified converter and evaluator reports.

The candidate and evaluation fixtures are signed with the producers' own digest
functions, so these tests also bind the publisher to the converter's and the
evaluator's signing conventions.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from validation.evaluate_mast_efm_neural_equilibrium import EVALUATION_SCHEMA
from validation.evaluate_mast_efm_neural_equilibrium import sha256_json as sign_evaluation
from validation.mast_efm_reference_contracts import CANDIDATE_SCHEMA, json_sha256
from validation.publish_mast_efm_neural_equilibrium_campaign import (
    CAMPAIGN_SCHEMA,
    CampaignInput,
    build_campaign_report,
    finite_metric,
    main,
    write_campaign_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON fixture, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _signed_candidate(shots: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a converter candidate report signed as the converter signs it."""
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA,
        "payload_sha256": None,
        "reference_dataset_id": "mast-efm-test",
        "reference_equilibria_count": 2 * len(shots),
        "shots": shots,
    }
    candidate["payload_sha256"] = json_sha256(candidate)
    return candidate


def _signed_evaluation(reference_path: Path, reference_sha256: str | None, **metrics: Any) -> dict[str, Any]:
    """Return an evaluator report for one reference bundle, signed as the evaluator signs it."""
    evaluation: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA,
        "reference_path": str(reference_path),
        "prediction_path": str(reference_path.with_name(reference_path.stem + "_prediction.npz")),
        "reference_artifact_sha256": reference_sha256,
        "prediction_artifact_sha256": "c" * 64,
        "weights_sha256": "d" * 64,
        "reference_equilibria_count": 2,
        "grid_shape": [65, 129],
        "admission_ready": False,
        "strict_artifact_emitted": False,
        "feature_mapping_notes": {"Ip_MA": "fallback: synthetic-domain centre", "R_axis_m": "source: axis"},
        "metrics": {
            "psi_rmse_Wb_per_rad": 1.5,
            "magnetic_axis_rmse_m": 0.7,
            "boundary_mean_distance_m": 0.4,
            "boundary_p95_distance_m": 1.2,
            "pressure_rmse_Pa": None,
            "q_profile_rmse": None,
            **metrics,
        },
    }
    evaluation["payload_sha256"] = sign_evaluation(evaluation)
    return evaluation


def _campaign(tmp_path: Path, shot_ids: tuple[int, ...] = (30419,)) -> tuple[Path, list[Path], Path]:
    """Write a signed candidate and one signed evaluation per shot; return their paths and the storage root."""
    storage_root = tmp_path / "storage"
    folder = storage_root / "converted/neural_equilibrium_reference"
    references = {shot: folder / f"mast_efm_shot_{shot}_reference.npz" for shot in shot_ids}
    candidate = folder / "candidate.json"
    _write_json(
        candidate, _signed_candidate([{"output_path": str(path), "sha256": "b" * 64} for path in references.values()])
    )
    evaluations = []
    for shot, reference in references.items():
        path = folder / f"evaluation_{shot}.json"
        _write_json(path, _signed_evaluation(reference, "b" * 64))
        evaluations.append(path)
    return candidate, evaluations, storage_root


def test_build_campaign_report_aggregates_metrics_and_keeps_claim_blocked(tmp_path: Path) -> None:
    """Verified inputs publish aggregate flux and geometry metrics; the claim stays blocked."""
    candidate, evaluations, storage_root = _campaign(tmp_path)

    report = build_campaign_report(
        CampaignInput(candidate_report=candidate, evaluation_reports=tuple(evaluations), storage_root=storage_root)
    )

    assert report["schema_version"] == CAMPAIGN_SCHEMA
    assert report["status"] == "blocked"
    assert report["admission_ready"] is False
    assert report["evaluated_reference_equilibria_count"] == 2
    assert report["aggregate_metrics"]["psi_rmse_Wb_per_rad_mean"] == 1.5
    assert report["aggregate_metrics"]["pressure_rmse_Pa_mean"] is None
    assert (
        report["shots"][0]["reference_path"]
        == "converted/neural_equilibrium_reference/mast_efm_shot_30419_reference.npz"
    )
    assert report["shots"][0]["reference_sha256"] == "b" * 64
    assert report["fallback_features"] == ["Ip_MA"]
    assert len(report["payload_sha256"]) == 64


def test_write_campaign_report_emits_markdown_boundary(tmp_path: Path) -> None:
    """The Markdown report carries the admission boundary and the remaining fallback features."""
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "blocked",
        "reference_dataset_id": "mast-efm-test",
        "shot_ids": [30419],
        "evaluated_reference_equilibria_count": 2,
        "candidate_payload_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "aggregate_metrics": {
            "psi_rmse_Wb_per_rad_mean": 1.5,
            "psi_rmse_Wb_per_rad_max": 1.5,
            "magnetic_axis_rmse_m_mean": 0.7,
            "boundary_mean_distance_m_mean": 0.4,
            "boundary_p95_distance_m_mean": 1.2,
            "pressure_rmse_Pa_mean": None,
            "q_profile_rmse_mean": None,
        },
        "shots": [
            {
                "shot_id": 30419,
                "reference_equilibria_count": 2,
                "metrics": {
                    "psi_rmse_Wb_per_rad": 1.5,
                    "magnetic_axis_rmse_m": 0.7,
                    "boundary_mean_distance_m": 0.4,
                    "boundary_p95_distance_m": 1.2,
                },
            }
        ],
        "blocked_reason": "predictive claims remain blocked",
        "fallback_features": ["Ip_MA"],
        "next_processing_steps": ["train full-output model"],
    }

    write_campaign_report(report, tmp_path / "campaign.json", tmp_path / "campaign.md")

    markdown = (tmp_path / "campaign.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Campaign Evidence" in markdown
    assert "predictive claims remain blocked" in markdown
    assert "`Ip_MA`" in markdown


@pytest.mark.parametrize("defect", ["schema", "digest"])
def test_a_foreign_or_stale_candidate_is_refused(tmp_path: Path, defect: str) -> None:
    """A candidate that is not the converter's report, or whose digest is stale, is refused."""
    candidate, evaluations, storage_root = _campaign(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if defect == "schema":
        payload["schema_version"] = "another.schema.v1"
        message = "is not a"
    else:
        payload["reference_dataset_id"] = "edited-after-signing"
        message = "payload_sha256 does not match"
    _write_json(candidate, payload)
    with pytest.raises(ValueError, match=message):
        build_campaign_report(CampaignInput(candidate, tuple(evaluations), storage_root))


@pytest.mark.parametrize("defect", ["schema", "digest"])
def test_a_foreign_or_stale_evaluation_is_refused(tmp_path: Path, defect: str) -> None:
    """An evaluation that is not the evaluator's report, or whose digest is stale, is refused."""
    candidate, evaluations, storage_root = _campaign(tmp_path)
    payload = json.loads(evaluations[0].read_text(encoding="utf-8"))
    if defect == "schema":
        payload["schema_version"] = "another.schema.v1"
        message = "is not a"
    else:
        payload["metrics"]["psi_rmse_Wb_per_rad"] = 0.1
        message = "payload_sha256 does not match"
    _write_json(evaluations[0], payload)
    with pytest.raises(ValueError, match=message):
        build_campaign_report(CampaignInput(candidate, tuple(evaluations), storage_root))


def test_a_reference_outside_the_candidate_is_refused(tmp_path: Path) -> None:
    """An evaluation of a bundle the candidate does not list cannot be published."""
    candidate, _, storage_root = _campaign(tmp_path)
    stray = tmp_path / "elsewhere" / "mast_efm_shot_30419_reference.npz"
    evaluation = tmp_path / "stray_evaluation.json"
    _write_json(evaluation, _signed_evaluation(stray, "b" * 64))
    with pytest.raises(ValueError, match="is not a shot of the candidate report"):
        build_campaign_report(CampaignInput(candidate, (evaluation,), storage_root))


@pytest.mark.parametrize("candidate_sha256", ["e" * 64, None])
def test_a_reference_digest_other_than_the_candidates_is_refused(tmp_path: Path, candidate_sha256: str | None) -> None:
    """The evaluated bundle must carry the candidate's string SHA-256, not another or a missing one."""
    storage_root = tmp_path / "storage"
    reference = storage_root / "mast_efm_shot_30419_reference.npz"
    candidate = storage_root / "candidate.json"
    _write_json(candidate, _signed_candidate([{"output_path": str(reference), "sha256": candidate_sha256}]))
    evaluation = storage_root / "evaluation.json"
    _write_json(evaluation, _signed_evaluation(reference, "b" * 64 if candidate_sha256 else None))
    with pytest.raises(ValueError, match="does not have the candidate's SHA-256"):
        build_campaign_report(CampaignInput(candidate, (evaluation,), storage_root))


def test_a_shot_evaluated_twice_is_refused(tmp_path: Path) -> None:
    """Two evaluations of the same shot would double-count it, so the second is refused."""
    candidate, evaluations, storage_root = _campaign(tmp_path)
    with pytest.raises(ValueError, match="shot 30419 is evaluated more than once"):
        build_campaign_report(CampaignInput(candidate, (evaluations[0], evaluations[0]), storage_root))


def test_no_evaluation_report_is_refused(tmp_path: Path) -> None:
    """A campaign needs at least one evaluation report."""
    candidate, _, storage_root = _campaign(tmp_path)
    with pytest.raises(ValueError, match="at least one evaluation report is required"):
        build_campaign_report(CampaignInput(candidate, (), storage_root))


@pytest.mark.parametrize("value", [True, "1.5"])
def test_a_metric_that_is_not_a_number_is_refused(tmp_path: Path, value: object) -> None:
    """A boolean or string metric is refused instead of being coerced into a number."""
    candidate, _, storage_root = _campaign(tmp_path)
    reference = Path(json.loads(candidate.read_text(encoding="utf-8"))["shots"][0]["output_path"])
    evaluation = tmp_path / "evaluation.json"
    _write_json(evaluation, _signed_evaluation(reference, "b" * 64, magnetic_axis_rmse_m=value))
    with pytest.raises(ValueError, match="metric magnetic_axis_rmse_m is not a finite number"):
        build_campaign_report(CampaignInput(candidate, (evaluation,), storage_root))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finite_metric_refuses_nonfinite_values(value: float) -> None:
    """The evaluator cannot sign a nonfinite metric; read directly, one is refused as well."""
    with pytest.raises(ValueError, match="metric psi is not a finite number"):
        finite_metric({"metrics": {"psi": value}}, "psi")


def test_missing_metrics_aggregate_to_null(tmp_path: Path) -> None:
    """When no shot reports a metric, its aggregates are null rather than zero."""
    candidate, _, storage_root = _campaign(tmp_path)
    reference = Path(json.loads(candidate.read_text(encoding="utf-8"))["shots"][0]["output_path"])
    evaluation = tmp_path / "evaluation.json"
    _write_json(
        evaluation,
        _signed_evaluation(
            reference,
            "b" * 64,
            psi_rmse_Wb_per_rad=None,
            magnetic_axis_rmse_m=None,
            boundary_mean_distance_m=None,
            boundary_p95_distance_m=None,
        ),
    )
    report = build_campaign_report(CampaignInput(candidate, (evaluation,), storage_root))
    assert report["aggregate_metrics"]["psi_rmse_Wb_per_rad_mean"] is None
    assert report["aggregate_metrics"]["boundary_p95_distance_m_max"] is None


def test_a_reference_name_without_a_shot_number_is_refused(tmp_path: Path) -> None:
    """The shot id comes from the reference file name; a name without digits is refused."""
    candidate, _, storage_root = _campaign(tmp_path)
    evaluation = tmp_path / "evaluation.json"
    _write_json(evaluation, _signed_evaluation(tmp_path / "reference.npz", "b" * 64))
    with pytest.raises(ValueError, match="cannot infer shot id"):
        build_campaign_report(CampaignInput(candidate, (evaluation,), storage_root))


def test_command_publishes_both_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command writes JSON and Markdown for verified inputs and reports the shot count."""
    candidate, evaluations, storage_root = _campaign(tmp_path, (30419, 30420))
    json_out, markdown_out = tmp_path / "out/campaign.json", tmp_path / "out/campaign.md"
    argv = ["--candidate-report", str(candidate), "--storage-root", str(storage_root)]
    for path in evaluations:
        argv += ["--evaluation-report", str(path)]
    assert main([*argv, "--json-out", str(json_out), "--report-out", str(markdown_out)]) == 0
    assert "status=blocked shots=2" in capsys.readouterr().out
    assert json.loads(json_out.read_text(encoding="utf-8"))["shot_ids"] == [30419, 30420]
    assert "MAST EFM Neural-Equilibrium Campaign Evidence" in markdown_out.read_text(encoding="utf-8")


def test_command_refuses_an_output_over_an_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An output path that would replace the candidate is refused before anything is written."""
    candidate, evaluations, storage_root = _campaign(tmp_path)
    before = candidate.read_bytes()
    argv = [
        "--candidate-report",
        str(candidate),
        "--evaluation-report",
        str(evaluations[0]),
        "--storage-root",
        str(storage_root),
        "--json-out",
        str(candidate),
        "--report-out",
        str(tmp_path / "campaign.md"),
    ]
    assert main(argv) == 1
    assert "MAST EFM campaign publication refused" in capsys.readouterr().err
    assert candidate.read_bytes() == before
    assert not (tmp_path / "campaign.md").exists()


def test_command_refuses_a_stale_evaluation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A refused input makes the command exit 1 with the reason and no outputs."""
    candidate, evaluations, storage_root = _campaign(tmp_path)
    payload = json.loads(evaluations[0].read_text(encoding="utf-8"))
    payload["weights_sha256"] = "f" * 64
    _write_json(evaluations[0], payload)
    json_out = tmp_path / "campaign.json"
    argv = ["--candidate-report", str(candidate), "--evaluation-report", str(evaluations[0])]
    argv += ["--storage-root", str(storage_root), "--json-out", str(json_out), "--report-out", str(tmp_path / "c.md")]
    assert main(argv) == 1
    assert "payload_sha256 does not match" in capsys.readouterr().err
    assert not json_out.exists()


def test_script_entry_point_reaches_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the file as a script hands its arguments to ``main``; ``--help`` exits cleanly."""
    script = ROOT / "validation" / "publish_mast_efm_neural_equilibrium_campaign.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    # Without the checkout on the path, the script must put it there itself to import its validators.
    without_checkout = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
    monkeypatch.setattr(sys, "path", without_checkout)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert sys.path[0] == str(ROOT)
    assert "--evaluation-report" in capsys.readouterr().out


def test_a_candidate_that_is_not_a_json_object_is_refused(tmp_path: Path) -> None:
    """A candidate file holding a JSON array is refused before any verification."""
    candidate, evaluations, storage_root = _campaign(tmp_path)
    candidate.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        build_campaign_report(CampaignInput(candidate, tuple(evaluations), storage_root))


def test_paths_outside_the_storage_root_are_published_as_given(tmp_path: Path) -> None:
    """Reference and prediction paths that are not below the storage root keep their full form."""
    candidate, evaluations, _ = _campaign(tmp_path)
    report = build_campaign_report(CampaignInput(candidate, tuple(evaluations), tmp_path / "another-root"))
    shot = report["shots"][0]
    assert shot["reference_path"] == (tmp_path / "storage/converted/neural_equilibrium_reference").as_posix() + (
        "/mast_efm_shot_30419_reference.npz"
    )
    assert shot["prediction_path"].endswith("mast_efm_shot_30419_reference_prediction.npz")
    assert Path(shot["prediction_path"]).is_absolute()
