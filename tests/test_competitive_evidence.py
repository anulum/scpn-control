# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Competitive evidence public-contract tests.

"""Exercise competitive-evidence admission through public files and CLI."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from tools import check_competitive_evidence

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_competitive_evidence.py"
MANIFEST = ROOT / "docs/_data/competitive_evidence.json"
PAGE = ROOT / "docs/competitive_analysis.md"


def _payload() -> dict[str, Any]:
    return check_competitive_evidence.load_manifest(MANIFEST)


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "competitive_evidence.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_project(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_live_competitive_evidence_passes_public_cli() -> None:
    """The committed source registry and public page pass the real CLI."""
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--manifest", str(MANIFEST), "--page", str(PAGE), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert completed.returncode == 0, result["errors"]
    assert result["passed"] is True
    assert result["system_count"] == 10
    assert result["quantitative_comparison_count"] == 0


def test_live_manifest_uses_exact_sources_and_no_numeric_ranking() -> None:
    """Every release-backed system binds a tag SHA and the numeric set is empty."""
    payload = _payload()
    systems = payload["systems"]
    source_bound_systems = [
        system for system in systems if system["artifact_kind"] in {"release", "repository-snapshot"}
    ]
    assert len(source_bound_systems) == 8
    assert all(len(system["commit_sha"]) == 40 for system in source_bound_systems)
    assert all("latest" not in url for system in systems for url in system["source_urls"])
    assert payload["quantitative_comparisons"] == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda payload: payload.update(schema="wrong"), "schema must be"),
        (lambda payload: payload.update(as_of="not-a-date"), "as_of must be an ISO date"),
        (
            lambda payload: payload.update(as_of=(date.today() + timedelta(days=1)).isoformat()),
            "as_of cannot be in the future",
        ),
        (
            lambda payload: payload.update(as_of=(date.today() - timedelta(days=121)).isoformat()),
            "older than 120 days",
        ),
        (lambda payload: payload.update(project_version="v0.23"), "project_version"),
        (lambda payload: payload.update(comparison_policy=[]), "comparison_policy must be an object"),
        (
            lambda payload: payload["comparison_policy"].update(absence_state="no"),
            "absence_state",
        ),
        (
            lambda payload: payload["comparison_policy"].update(scope="short"),
            "scope requires a specific statement",
        ),
        (lambda payload: payload.update(systems=[]), "at least eight"),
        (lambda payload: payload["systems"].append("bad"), "must be an object"),
        (
            lambda payload: payload["systems"][1].update(id=payload["systems"][0]["id"]),
            "duplicate system id",
        ),
        (lambda payload: payload["systems"][1].update(id="Bad ID"), "stable lowercase"),
        (lambda payload: payload["systems"][1].pop("primary_role"), "fields must exactly"),
        (lambda payload: payload["systems"][1].update(name=""), ".name is required"),
        (lambda payload: payload["systems"][1].update(artifact_kind="archive"), "unsupported"),
        (lambda payload: payload["systems"][1].update(assessed_version=""), "assessed_version"),
        (lambda payload: payload["systems"][1].update(artifact_date=4), "artifact_date"),
        (lambda payload: payload["systems"][1].update(commit_sha="abc"), "bind source evidence"),
        (lambda payload: payload["systems"][8].update(commit_sha="0" * 40), "null for paper"),
        (lambda payload: payload["systems"][1].update(primary_role="short"), "primary_role"),
        (lambda payload: payload["systems"][1].update(evidence_boundary="short"), "evidence_boundary"),
        (lambda payload: payload["systems"][1].update(documented_strengths=[]), "documented_strengths"),
        (lambda payload: payload["systems"][1].update(source_urls=[]), "source_urls requires"),
        (lambda payload: payload["systems"][1].update(source_urls=["http://invalid"]), "use HTTPS"),
        (
            lambda payload: payload["systems"][1].update(source_urls=["https://example.org/v1.4.3"]),
            "primary-source host",
        ),
        (
            lambda payload: payload["systems"][1].update(source_urls=["https://github.com/example/latest"]),
            "bind the assessed release",
        ),
        (lambda payload: payload.update(quantitative_comparisons={}), "must be a list"),
        (lambda payload: payload["quantitative_comparisons"].append("bad"), "must be an object"),
        (lambda payload: payload["quantitative_comparisons"].append({"system_a": "a"}), "matched protocol"),
        (
            lambda payload: payload["quantitative_comparisons"].append(
                {field: "" for field in check_competitive_evidence.COMPARISON_FIELDS}
            ),
            "cannot contain empty",
        ),
    ],
)
def test_manifest_contract_fails_closed(
    mutation: Callable[[dict[str, Any]], object],
    expected: str,
) -> None:
    """Malformed, stale, vague, or unmatched evidence cannot be admitted."""
    payload = copy.deepcopy(_payload())
    mutation(payload)
    errors = check_competitive_evidence.validate_manifest(payload)
    assert any(expected in error for error in errors)


def test_manifest_loader_rejects_duplicate_keys_and_non_object(tmp_path: Path) -> None:
    """Ambiguous JSON and non-object roots fail before semantic validation."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        check_competitive_evidence.load_manifest(duplicate)
    with pytest.raises(ValueError, match="must be a JSON object"):
        check_competitive_evidence.load_manifest(_write_manifest(tmp_path, []))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[project]\nname = "other"\nversion = "1.0.0"\n', "identify scpn-control"),
        ('[project]\nname = "scpn-control"\nversion = "rolling"\n', "exact major.minor.patch"),
    ],
)
def test_project_version_loader_fails_closed(tmp_path: Path, text: str, message: str) -> None:
    """Competitive evidence cannot detach from canonical package identity."""
    with pytest.raises(ValueError, match=message):
        check_competitive_evidence.load_project_version(_write_project(tmp_path, text))


def test_audit_rejects_manifest_project_version_drift(tmp_path: Path) -> None:
    """A version bump requires refreshing the assessed SCPN evidence snapshot."""
    project = _write_project(tmp_path, '[project]\nname = "scpn-control"\nversion = "0.24.0"\n')
    result = check_competitive_evidence.audit(MANIFEST, PAGE, project)
    assert result["project_version"] == "0.24.0"
    assert "manifest project_version must match pyproject.toml" in result["errors"]


def test_complete_matched_protocol_shape_is_admitted() -> None:
    """A quantitative row is structurally admissible only when every protocol field is present."""
    payload = _payload()
    payload["quantitative_comparisons"] = [
        {field: f"declared {field}" for field in check_competitive_evidence.COMPARISON_FIELDS}
    ]
    assert check_competitive_evidence.validate_manifest(payload) == []


def test_page_validation_defers_malformed_types_to_manifest_validation() -> None:
    """Rendering validation remains bounded when semantic validation has already found bad types."""
    payload = _payload()
    payload["systems"] = "invalid"
    assert check_competitive_evidence.validate_page("", payload)

    payload = _payload()
    payload["systems"].append("invalid")
    payload["systems"][0]["name"] = 4
    payload["systems"][0]["assessed_version"] = 4
    payload["systems"][0]["source_urls"] = "invalid"
    assert check_competitive_evidence.validate_page(PAGE.read_text(encoding="utf-8"), payload) == []


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("", "evidence date"),
        (PAGE.read_text(encoding="utf-8").replace("v1.4.3", "version-missing"), "assessed_version for torax"),
        (
            PAGE.read_text(encoding="utf-8").replace(
                "https://github.com/google-deepmind/torax/releases/tag/v1.4.3", "source-missing", 1
            ),
            "missing primary source",
        ),
        (PAGE.read_text(encoding="utf-8") + "\nSCPN is superior.\n", "ranking language"),
        (PAGE.read_text(encoding="utf-8") + "\nCONTROL-AUD-009\n", "private planning marker"),
        (
            PAGE.read_text(encoding="utf-8").replace(
                "quantitative comparison set is currently empty", "numbers unavailable"
            ),
            "must disclose",
        ),
    ],
)
def test_public_page_fails_closed_on_drift(page: str, expected: str) -> None:
    """The rendering cannot omit sources, hide the empty set, or use ranking language."""
    errors = check_competitive_evidence.validate_page(page, _payload())
    assert any(expected in error for error in errors)


def test_cli_reports_findings_and_parse_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public entry point has stable pass, finding, JSON, and parse-error modes."""
    assert check_competitive_evidence.main(["--manifest", str(MANIFEST), "--page", str(PAGE)]) == 0
    assert capsys.readouterr().out == "Competitive evidence passed\n"

    assert check_competitive_evidence.main(["--manifest", str(MANIFEST), "--page", str(PAGE), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True

    bad_page = tmp_path / "competitive_analysis.md"
    bad_page.write_text("missing", encoding="utf-8")
    assert check_competitive_evidence.main(["--manifest", str(MANIFEST), "--page", str(bad_page)]) == 1
    assert "competitive evidence:" in capsys.readouterr().err

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert check_competitive_evidence.main(["--manifest", str(invalid), "--page", str(PAGE)]) == 2
    assert "competitive evidence error:" in capsys.readouterr().err


def test_help_is_side_effect_free(tmp_path: Path) -> None:
    """CLI help performs no audit and creates no local artifact."""
    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "competitive evidence registry" in completed.stdout
    assert tuple(tmp_path.iterdir()) == before
