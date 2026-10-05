# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Shared reference declaration location policy tests

"""Exercise public lexical policies and their real persisted density/registered-CLI consumer.

URI and executable strings are declarations. None of these tests authenticates
an external binary, reference, model run or facility result.
"""

from __future__ import annotations

import doctest
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from test_density_reference_validation import _valid_density_reference_artifact

from scpn_control.cli import main as root_cli
from validation import reference_uri
from validation.validate_density_reference import validate_density_reference


@pytest.mark.parametrize(
    "value,finding",
    [
        (None, "must be a non-empty URI"),
        ([], "must be a non-empty URI"),
        ("", "must be a non-empty URI"),
        ("  ", "must be a non-empty URI"),
        ("relative.json", "must include an explicit URI scheme"),
        ("http://example.org/data", "scheme must be file, https, s3, or gs"),
        ("https://[", "must be a syntactically valid URI"),
        ("https://example.org", "must identify a stable remote artifact path"),
        ("https:///data", "must identify a stable remote artifact path"),
        ("s3://bucket/a/../data", "must identify a stable remote artifact path"),
        ("file://host/validation/reports/data", "file URI must not include a host"),
        ("file:///etc/passwd", "file URI must be under /validation/reports or /validation/reference_data"),
        ("file:///validation/reports/../data", "must not contain parent traversal"),
        ("https://example.org/a\x00b", "must not contain control characters"),
        ("https://example.org/a\nb", "must not contain control characters"),
        ("\thttps://example.org/data", "must not contain control characters"),
        ("https://example.org/a\x7fb", "must not contain control characters"),
    ],
)
def test_artifact_declaration_refusals_are_authored(value: object, finding: str) -> None:
    """Malformed/container/control/path declarations return precise selected-field findings without exceptions."""
    assert reference_uri.reference_artifact_uri_error(value, "selected") == f"selected {finding}"


@pytest.mark.parametrize(
    "value",
    [
        "file:///validation/reports/campaign.json",
        "file:///validation/reference_data/campaign.npz",
        "https://example.org/reference.npz",
        "s3://bucket/reference.npz",
        "gs://bucket/reference.npz",
        "  https://example.org/reference.npz  ",
        "https://example.org/a/%2e%2e/reference.npz",
        "https://user@example.org/reference.npz?version=1#manifest",
        "file:///validation/reports/",
    ],
)
def test_artifact_policy_acceptance_is_lexical_only(value: str) -> None:
    """Policy-conforming strings require no file/network read; encoded parents and directory prefixes are not authenticated."""
    assert reference_uri.reference_artifact_uri_error(value, "reference") is None


@pytest.mark.parametrize(
    "value,finding",
    [
        (None, "must be a non-empty absolute executable path"),
        (False, "must be a non-empty absolute executable path"),
        (" ", "must be a non-empty absolute executable path"),
        ("//[", "must be a syntactically valid absolute executable path"),
        ("file:///opt/solver", "must be an absolute filesystem path, not a URI"),
        ("//host/opt/solver", "must be an absolute filesystem path, not a URI"),
        ("solver", "must be an absolute filesystem path"),
        ("C:/solver.exe", "must be an absolute filesystem path, not a URI"),
        ("/opt/a/../solver", "must not contain parent traversal"),
        ("/opt/", "must identify an executable file path"),
        ("/opt/.", "must identify an executable file path"),
        ("/tmp/solver", "must not point to mutable or system-control paths"),
        ("/home/operator/solver", "must be under an admitted deployment or facility executable root"),
        ("/opt/sol\x00ver", "must not contain control characters"),
        ("/opt/sol\nver", "must not contain control characters"),
        ("/opt/solver\n", "must not contain control characters"),
        ("/opt/sol\x7fver", "must not contain control characters"),
    ],
)
def test_executable_declaration_refusals_are_authored(value: object, finding: str) -> None:
    """Invalid provenance location strings, including parser failures and final dot components, return selected findings."""
    assert reference_uri.external_executable_path_error(value, "selected") == f"selected {finding}"


@pytest.mark.parametrize(
    "prefix",
    [
        "/opt/",
        "/usr/local/",
        "/usr/bin/",
        "/bin/",
        "/nix/store/",
        "/validation/external_bins/",
        "/facility/",
        "/mnt/facility/",
        "/gpfs/",
        "/lustre/",
    ],
)
def test_admitted_executable_roots_do_not_require_a_binary(prefix: str) -> None:
    """Every existing admitted root accepts a declared leaf without executing or locating that binary."""
    assert reference_uri.external_executable_path_error(prefix + "declared-solver") is None


def test_surrounding_spaces_are_not_control_characters() -> None:
    """Ordinary surrounding spaces remain compatible with the declared path normalization policy."""
    assert reference_uri.external_executable_path_error("  /opt/declared-solver  ") is None


@pytest.mark.parametrize("uri", ["https://[", "https://example.org/a\x00b", "https://example.org/a\nb"])
def test_density_reader_and_registered_cli_preserve_uri_findings(tmp_path: Path, uri: str) -> None:
    """Actual persisted declaration and root CLI both reject malformed reference locations at the owning field."""
    payload = _valid_density_reference_artifact()
    payload.update(source="external_integrated_modelling", external_code="ASTRA", reference_artifact_uri=uri)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    report = validate_density_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert {finding["field"] for finding in report["errors"]} == {"reference_artifact_uri"}
    assert report["errors"][0]["error"] == reference_uri.reference_artifact_uri_error(uri, "reference_artifact_uri")
    result = CliRunner().invoke(
        root_cli,
        ["validate-density-reference", "--artifact-root", str(path), "--require-reference-artifacts", "--json-out"],
    )
    assert result.exit_code == 1 and json.loads(result.output) == report
    assert path.read_bytes() == before


def test_native_public_examples_execute() -> None:
    """Execute three real public refusal examples instead of replacing them with import-only checks."""
    result = doctest.testmod(reference_uri, raise_on_error=True)
    assert result.attempted == 3 and result.failed == 0
