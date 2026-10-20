# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact Python dependency compatibility contracts.
"""Exercise native lockfiles, metadata HTTP and the real compatibility CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dependency_metadata_server import metadata_service, release_metadata

from tools.preflight import run_hash_pin_check, run_python_compat_check
from tools.python_dependency_compatibility import main, python_compatibility_errors
from tools.python_dependency_metadata import DependencyMetadataError

ROOT = Path(__file__).resolve().parents[2]


def _lock(tmp_path: Path, content: str = "example-distribution==1.0\n") -> Path:
    """Write one real exact-pin input to the public compatibility surface."""
    path = tmp_path / "requirements.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("constraint", [">=3.11", ">=3.10,<3.14", "", None])
def test_compatibility_admits_each_applicable_declared_python_floor(tmp_path: Path, constraint: str | None) -> None:
    """Actual release metadata covers every target, with a single fetch per release."""
    path = _lock(tmp_path, "# locked\n\nExample_Distribution[extra]==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n")
    with metadata_service(release_metadata(constraint=constraint)) as (url, requests):
        assert python_compatibility_errors([path, path], metadata_base_url=url) == []
    assert requests == ["/pypi/example-distribution/1.0/json"]


@pytest.mark.parametrize(
    "constraint,rejected",
    [
        ("<3.12", ["3.12.0", "3.13.0"]),
        (">=3.12", ["3.11.0"]),
        ("!=3.12.0", ["3.12.0"]),
        (">=3.11.1", ["3.11.0"]),
        ("~=3.11.0", ["3.12.0", "3.13.0"]),
    ],
)
def test_compatibility_rejects_upper_bounds_exclusions_and_patch_floors(
    tmp_path: Path, constraint: str, rejected: list[str]
) -> None:
    """PEP440 interpretation catches incompatibilities the old lower-bound regex lost."""
    path = _lock(tmp_path)
    with metadata_service(release_metadata(constraint=constraint)) as (url, _requests):
        errors = python_compatibility_errors([path], metadata_base_url=url)
    assert errors == [
        f"example-distribution==1.0 requires Python {constraint}; target {target} is incompatible"
        for target in rejected
    ]


def test_compatibility_evaluates_real_python_markers(tmp_path: Path) -> None:
    """Marker-excluded pins are not fetched, while applicable targets remain checked."""
    path = _lock(
        tmp_path, 'example-distribution==1.0; python_version >= "3.12"\nabsent==2.0; python_full_version < "3.0"\n'
    )
    with metadata_service(release_metadata(constraint=">=3.12")) as (url, requests):
        assert python_compatibility_errors([path], metadata_base_url=url) == []
    assert len(requests) == 1


@pytest.mark.parametrize(
    "content",
    [
        "",
        "# no pins\n",
        "example>=1",
        "example==1.*",
        "--index-url https://pypi.org/simple",
        "example @ https://example.org/a.whl",
        "not a requirement",
        'example==1.0; sys_platform == "win32"',
        "example==bad",
    ],
)
def test_compatibility_refuses_empty_or_unqualified_input_syntax(tmp_path: Path, content: str) -> None:
    """Unsupported declarations cannot vanish from a successful measurement."""
    with pytest.raises(DependencyMetadataError):
        python_compatibility_errors([_lock(tmp_path, content)])


def test_compatibility_refuses_actual_missing_directory_and_invalid_utf8_locks(tmp_path: Path) -> None:
    """Every supplied lock must be a readable UTF8 file with nonempty exact pins."""
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    for path in [tmp_path / "absent", tmp_path, invalid]:
        with pytest.raises(DependencyMetadataError, match="required compatibility lock"):
            python_compatibility_errors([path])


def test_compatibility_refuses_invalid_metadata_constraint(tmp_path: Path) -> None:
    """Invalid requires_python syntax refuses rather than declaring no restriction."""
    with (
        metadata_service(release_metadata(constraint="not-python")) as (url, _requests),
        pytest.raises(DependencyMetadataError, match="invalid Python constraint"),
    ):
        python_compatibility_errors([_lock(tmp_path)], metadata_base_url=url)


@pytest.mark.parametrize(
    "targets", [[], ["3.11", "3.11.0"], ["bad"], ["1!3.13"], ["3.13.post1"], ["3.13+local"], ["3.13.0.1"]]
)
def test_compatibility_refuses_empty_duplicate_or_invalid_python_targets(tmp_path: Path, targets: list[str]) -> None:
    """An unqualified target set is not an empty passing comparison."""
    with pytest.raises(DependencyMetadataError, match="targets|versions"):
        python_compatibility_errors([_lock(tmp_path)], targets)


def test_compatibility_preserves_prerelease_markers_and_checks_each_release(tmp_path: Path) -> None:
    """Physical per-release responses distinguish prerelease and final markers."""
    path = _lock(
        tmp_path,
        'example-distribution==1.0; python_full_version < "3.13.0rc2"\n'
        'final-distribution==2.0; python_full_version >= "3.13.0"\n',
    )
    payloads = {
        "/pypi/example-distribution/1.0/json": release_metadata(constraint="<3.13.0rc2"),
        "/pypi/final-distribution/2.0/json": release_metadata("final-distribution", "2.0", ">=3.13.0"),
    }
    with metadata_service(payloads) as (url, requests):
        assert python_compatibility_errors([path], ["3.13rc1", "3.13"], metadata_base_url=url) == []
    assert requests == list(payloads)


def test_compatibility_refuses_no_required_locks() -> None:
    """An empty lock inventory cannot clear the compatibility gate."""
    with pytest.raises(DependencyMetadataError, match="locks"):
        python_compatibility_errors([])


def test_compatibility_refuses_an_entirely_excluded_release_inventory(tmp_path: Path) -> None:
    """An empty applicable cohort cannot yield a vacuous compatibility verdict."""
    path = _lock(tmp_path, 'example-distribution==1.0; python_version < "3.0"\n')
    with pytest.raises(DependencyMetadataError, match="No locked release applies"):
        python_compatibility_errors([path])


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_compatibility_validates_timeout_before_release_selection(tmp_path: Path, timeout: float) -> None:
    """Invalid socket bounds refuse even when all Python markers exclude pins."""
    path = _lock(tmp_path, 'example-distribution==1.0; python_version < "3.0"\n')
    with pytest.raises(DependencyMetadataError, match="timeout"):
        python_compatibility_errors([path], timeout_s=timeout)


@pytest.mark.parametrize("constraint,exit_code", [(">=3.11", 0), (">=3.12", 1), ("invalid", 2)])
def test_actual_compatibility_cli_propagates_complete_verdict(tmp_path: Path, constraint: str, exit_code: int) -> None:
    """The physical command uses real file/HTTP inputs and fixed process exits."""
    path = _lock(tmp_path)
    with metadata_service(release_metadata(constraint=constraint)) as (url, requests):
        command = [sys.executable]
        coverage_rc = os.environ.get("SCPN_PYTHON_COMPATIBILITY_COVERAGE_RC")
        if coverage_rc:
            command += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
        command += [
            str(ROOT / "tools/python_dependency_compatibility.py"),
            "--requirements",
            str(path),
            "--metadata-base-url",
            url,
            "--python-version",
            "3.11",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == exit_code and completed.stderr == ""
    assert len(requests) == 1
    assert "Traceback" not in completed.stdout


def test_compatibility_main_refuses_absent_inputs_without_other_preflight_gates(tmp_path: Path) -> None:
    """Standalone refusal never launches the repository's aggregate preflight."""
    assert main(["--requirements", str(tmp_path / "missing")]) == 2


@pytest.mark.parametrize("constraint,admitted", [(">=3.11", True), ("<3.12", False), ("invalid", False)])
def test_preflight_uses_the_same_real_metadata_and_all_python_targets(
    tmp_path: Path, constraint: str, admitted: bool
) -> None:
    """The public preflight gate cannot drop upper bounds or failed inspection."""
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    for name in ["ci-deps", "ci-test"]:
        (requirements / (name + ".txt")).write_text("example-distribution==1.0\n", encoding="utf-8")
    with metadata_service(release_metadata(constraint=constraint)) as (url, requests):
        assert run_python_compat_check(tmp_path, metadata_base_url=url) is admitted
    assert len(requests) == 1


def test_preflight_refuses_missing_common_and_hash_lockfiles(tmp_path: Path) -> None:
    """Absent declared inputs fail both native gates without running installers."""
    assert run_python_compat_check(tmp_path) is False
    assert run_hash_pin_check(tmp_path) is False
