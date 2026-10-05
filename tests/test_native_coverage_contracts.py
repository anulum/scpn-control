# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native coverage declaration API and CLI contracts.

"""Exercise the coverage guard through actual files, public API and child CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.ci_workflow_inventory import read_ci_workflow_source
from tools.native_coverage_matrix import (
    DEFAULT_DOCS,
    DEFAULT_PYPROJECT,
    REPO_ROOT,
    NativeCoverageFinding,
    NativeCoverageMatrix,
    main,
    validate_native_coverage_matrix,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    """Retain real workflow bodies with their monolithic dependency edges."""
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        read_ci_workflow_source().replace(
            "  native-coverage-combine:\n",
            "  native-coverage-combine:\n    needs: [python-tests, rust-python-interop]\n",
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "pyproject.toml"
    shutil.copyfile(DEFAULT_PYPROJECT, metadata)
    docs = tuple(tmp_path / source.name for source in DEFAULT_DOCS)
    for source, target in zip(DEFAULT_DOCS, docs, strict=True):
        shutil.copyfile(source, target)
    assert validate_native_coverage_matrix(workflow, metadata, docs).passed
    return workflow, metadata, docs


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("python -m coverage report --fail-under=100", 'echo "python -m coverage report --fail-under=100"'),
        ("python -m coverage report --fail-under=100", "python -m coverage report --fail-under=100 || true"),
        ("python -m coverage report --fail-under=100", "python -m coverage report --fail-under=0"),
        ("python -m coverage report --fail-under=100", "python -m coverage report --fail-under=100; true"),
        ("python -m coverage report --fail-under=100", "exit 0\n          python -m coverage report --fail-under=100"),
        ("python -m coverage report --fail-under=100", "set +e\n          python -m coverage report --fail-under=100"),
        (
            "python -m coverage report --fail-under=100",
            "if false; then\n          python -m coverage report --fail-under=100\n          fi",
        ),
        (
            "python -m coverage report --fail-under=100",
            "cat <<EOF\n          python -m coverage report --fail-under=100\n          EOF",
        ),
        ("python -m coverage report --fail-under=100", "python -m coverage report --fail-under=100\\"),
        ("python -m coverage report --fail-under=100", "echo 'unfinished"),
        ("python -m coverage report --fail-under=100", "cat <<"),
        ("python -m coverage report --fail-under=100", "cat <<EOF\n          absent-delimiter"),
        ("--cov-fail-under=100", "--cov-fail-under=100 --cov-fail-under=0"),
        ("--cov-fail-under=100", "--cov-fail-under=100 --no-cov"),
        ("--cov-fail-under=100", "--cov-fail-under=100 --collect-only"),
        ("--cov-fail-under=100", "--cov-fail-under=100 --co"),
        ("--cov=scpn_control", "--cov=another_package"),
        ("--source=scpn_control", "--source=another_package"),
        ("tests/test_boris_pyo3_bridge.py", "missing.py"),
        ("COVERAGE_FILE=.coverage.rust", "COVERAGE_FILE=.coverage.unrelated"),
        ("include-hidden-files: true", "include-hidden-files: false"),
        ("include-hidden-files: true", "include-hidden-files: yes"),
        (
            "cp .coverage artifacts/coverage/python/.coverage.python",
            "cp absent artifacts/coverage/python/.coverage.python",
        ),
        (
            "mv .coverage.rust artifacts/coverage/rust/.coverage.rust",
            "mv absent artifacts/coverage/rust/.coverage.rust",
        ),
        ("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", "actions/download-artifact@v7"),
        ("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", "actions/upload-artifact@v7"),
        ("path: artifacts/coverage/rust", "path: artifacts/coverage/unrelated"),
        ("coverage-data-rust", "coverage-data-unrelated"),
        ("coverage-report-combined", "coverage-report-unrelated"),
        ("coverage combine --keep", "coverage combine"),
        ("coverage xml -o coverage.xml", "coverage xml -o elsewhere.xml"),
        ("run: python tools/native_coverage_matrix.py", "run: echo python tools/native_coverage_matrix.py"),
        ("needs: [python-tests, rust-python-interop]", "needs: [python-tests, python-tests]"),
        ("  python-tests:\n", "  python-tests:\n    if: false\n"),
        ("  python-tests:\n", "  python-tests:\n    continue-on-error: true\n"),
        ("  rust-python-interop:\n", "  rust-python-interop:\n    if: false\n"),
        ("  native-coverage-combine:\n", "  native-coverage-combine:\n    continue-on-error: true\n"),
        ("      - name: Combine coverage data\n", "      - name: Combine coverage data\n        if: false\n"),
        (
            "      - name: Combine coverage data\n",
            "      - name: Combine coverage data\n        working-directory: elsewhere\n",
        ),
        ("      - name: Combine coverage data\n", "      - name: Combine coverage data\n        shell: pwsh\n"),
        (
            "      - name: Combine coverage data\n",
            "      - name: Combine coverage data\n        env:\n          COVERAGE_FILE: absent\n",
        ),
        ("jobs:\n", "env:\n  COVERAGE_FILE: absent\njobs:\n"),
        ("jobs:\n", "defaults:\n  run:\n    working-directory: elsewhere\njobs:\n"),
    ],
)
def test_public_guard_refuses_ineffective_declarations(tmp_path: Path, old: str, new: str) -> None:
    """Coverage evidence must be collected, saved and consumed by enabled steps."""
    workflow, metadata, docs = _inputs(tmp_path)
    original = workflow.read_text(encoding="utf-8")
    assert old in original
    workflow.write_text(original.replace(old, new), encoding="utf-8")
    assert not validate_native_coverage_matrix(workflow, metadata, docs).passed


@pytest.mark.parametrize(
    "text",
    [
        "# jobs: only comments\n",
        "jobs: {}\n",
        "jobs: [invalid]\n",
        "jobs: {}\njobs: {}\n",
        "jobs: {broken\n",
        "!!python/object/apply:os.system ['touch forbidden']\n",
        "!!map []\n",
        "jobs:\n  ? [unhashable]\n  : {}\n",
        "[not-a-workflow]\n",
    ],
)
def test_public_guard_authors_malformed_yaml_findings(tmp_path: Path, text: str) -> None:
    """Malformed, duplicate and unsafe YAML never executes or exposes exception text."""
    workflow, metadata, docs = _inputs(tmp_path)
    workflow.write_text(text, encoding="utf-8")
    matrix = validate_native_coverage_matrix(workflow, metadata, docs)
    assert matrix.findings[0].check == "workflow input"
    assert matrix.findings[0].detail == "Workflow source or ownership declarations are unreadable or malformed."
    assert not (REPO_ROOT / "forbidden").exists()


@pytest.mark.parametrize("value", ["0 # fail_under = 100", "true", '"100"', "99.999", "100.0"])
def test_public_guard_reads_the_numeric_toml_threshold(tmp_path: Path, value: str) -> None:
    """Only a numeric 100 threshold passes, regardless of comments and spelling."""
    workflow, metadata, docs = _inputs(tmp_path)
    metadata.write_text(f"[tool.coverage.report]\nfail_under = {value}\n", encoding="utf-8")
    matrix = validate_native_coverage_matrix(workflow, metadata, docs)
    assert matrix.passed is (value == "100.0")


@pytest.mark.parametrize("owner", ["workflow", "threshold", "documentation"])
@pytest.mark.parametrize("mode", ["missing", "utf8", "directory"])
def test_public_guard_authors_unreadable_input_findings(tmp_path: Path, owner: str, mode: str) -> None:
    """IO and decoding failures become stable authored input categories."""
    workflow, metadata, docs = _inputs(tmp_path)
    target = {"workflow": workflow, "threshold": metadata, "documentation": docs[0]}[owner]
    if mode == "missing":
        target.unlink()
    elif mode == "utf8":
        target.write_bytes(b"\xff")
    else:
        target.unlink()
        target.mkdir()
    matrix = validate_native_coverage_matrix(workflow, metadata, docs)
    assert f"{owner} input" in {finding.check for finding in matrix.findings}
    assert not matrix.passed


def test_public_guard_handles_empty_docs_invalid_toml_and_repository_paths(tmp_path: Path) -> None:
    """Missing prose, malformed metadata and canonical directory inputs fail safely."""
    workflow, metadata, _docs = _inputs(tmp_path)
    assert [f.check for f in validate_native_coverage_matrix(workflow, metadata, ()).findings] == ["public docs"]
    metadata.write_text("[not valid", encoding="utf-8")
    assert "threshold input" in {f.check for f in validate_native_coverage_matrix(workflow, metadata).findings}
    assert validate_native_coverage_matrix(REPO_ROOT).findings[0].path == "."


def test_public_guard_preserves_safe_equivalent_command_and_yaml_forms(tmp_path: Path) -> None:
    """Direct Python-module commands, YAML aliases and literal true conditions work."""
    workflow, metadata, docs = _inputs(tmp_path)
    text = workflow.read_text(encoding="utf-8")
    text = text.replace("pytest -p hypothesis", "python3 -m pytest -p hypothesis")
    text = text.replace("python -m coverage report --fail-under=100", "coverage report --fail-under 100")
    text = text.replace("run: python tools/native_coverage_matrix.py", "run: python -m tools.native_coverage_matrix")
    text = text.replace("  native-coverage-combine:\n", "  native-coverage-combine:\n    <<: &enabled {if: true}\n")
    text = text.replace("  rust-python-interop:\n", "  rust-python-interop:\n    if: true\n")
    workflow.write_text(text, encoding="utf-8")
    assert validate_native_coverage_matrix(workflow, metadata, docs).passed


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize(
    "arguments,expected", [([], 0), (["--json"], 0), (["--help"], 0), (["--unknown"], 2), (["--docs"], 1)]
)
def test_real_module_and_script_cli(module: bool, arguments: list[str], expected: int) -> None:
    """Both source-tree entry points retain success, findings and argparse exits."""
    target = ["-m", "tools.native_coverage_matrix"] if module else ["tools/native_coverage_matrix.py"]
    result = subprocess.run(
        [sys.executable, *target, *arguments],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == expected
    assert "Traceback" not in result.stderr
    if arguments == ["--json"]:
        assert json.loads(result.stdout)["passed"] is True


def test_json_cli_is_read_only_and_returns_authored_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Selected-input CLI failure leaves every input byte unchanged."""
    workflow, metadata, docs = _inputs(tmp_path)
    workflow.write_text("# comments only\n", encoding="utf-8")
    paths = (workflow, metadata, *docs)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert (
        main(["--workflow", str(workflow), "--pyproject", str(metadata), "--docs", *(str(p) for p in docs), "--json"])
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["findings"][0]["check"] == "workflow input"
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_public_result_serialization_is_fresh_and_does_not_authenticate() -> None:
    """Result construction records caller findings without sealing or running inputs."""
    matrix = NativeCoverageMatrix((NativeCoverageFinding("selected", "declared", "authored"),))
    first, second = matrix.to_jsonable(), matrix.to_jsonable()
    assert first == second and first is not second
    assert first["findings"] is not second["findings"]
    assert matrix.passed is False


def _distributed_inputs(tmp_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    """Copy the physical declared owners so canonical API code reads real files."""
    tools = tmp_path / "tools"
    tools.mkdir()
    policy = tools / "ci_workflow_policy.json"
    shutil.copyfile(REPO_ROOT / "tools/ci_workflow_policy.json", policy)
    workflows = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    for name in ("ci.yml", "ci-python-quality.yml", "ci-native-polyglot.yml", "ci-native-coverage.yml"):
        shutil.copyfile(REPO_ROOT / ".github/workflows" / name, workflows / name)
    metadata = tmp_path / "pyproject.toml"
    shutil.copyfile(DEFAULT_PYPROJECT, metadata)
    docs = tuple(tmp_path / source.name for source in DEFAULT_DOCS)
    for source, target in zip(DEFAULT_DOCS, docs, strict=True):
        shutil.copyfile(source, target)
    assert validate_native_coverage_matrix(policy, metadata, docs).passed
    return policy, metadata, docs


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "coordinator",
        "categories",
        "duplicate",
        "missing",
        "id",
        "owner",
        "workflow",
        "graph",
        "needs",
        "json",
        "duplicate-key",
        "nonfinite",
        "coordinator_escape",
        "coordinator_suffix",
    ],
)
def test_distributed_policy_is_bound_to_physical_owners(tmp_path: Path, change: str) -> None:
    """Selected JSON policies cannot silently change ownership or dependency edges."""
    policy, metadata, docs = _distributed_inputs(tmp_path)
    data = cast(dict[str, object], json.loads(policy.read_text(encoding="utf-8")))
    categories = cast(list[dict[str, object]], data["categories"])
    category = next(c for c in categories if c["id"] == "native-coverage")
    if change in {"version", "coordinator", "categories"}:
        data[{"version": "schema_version", "coordinator": "coordinator", "categories": "categories"}[change]] = False
    elif change == "duplicate":
        categories.append(category.copy())
    elif change == "missing":
        categories.remove(category)
    elif change == "id":
        category["id"] = 13
    elif change == "owner":
        category["jobs"] = []
    elif change == "workflow":
        category["workflow"] = "../elsewhere.yml"
    elif change == "graph":
        data["dependency_graph"] = {}
    elif change == "needs":
        category["caller_needs"] = ["python-quality"]
    elif change == "coordinator_escape":
        data["coordinator"] = "README.md"
    elif change == "coordinator_suffix":
        data["coordinator"] = ".github/workflows/ci.txt"
    text = json.dumps(data)
    if change == "json":
        text = "{ malformed"
    elif change == "duplicate-key":
        text = text[:-1] + ', "categories": []}'
    elif change == "nonfinite":
        text = text[:-1] + ', "invalid": NaN}'
    policy.write_text(text, encoding="utf-8")
    assert not validate_native_coverage_matrix(policy, metadata, docs).passed


@pytest.mark.parametrize(
    "owner,old,new",
    [
        ("ci.yml", "  python-quality:\n", "  python-quality:\n    if: false\n"),
        ("ci.yml", "  native-polyglot:\n", "  native-polyglot:\n    continue-on-error: true\n"),
        ("ci.yml", "  native-coverage:\n", "  native-coverage:\n    if: false\n"),
        ("ci.yml", "uses: ./.github/workflows/ci-native-coverage.yml", "uses: ./elsewhere.yml"),
        ("ci-python-quality.yml", "jobs:\n", "env:\n  COVERAGE_FILE: elsewhere\njobs:\n"),
        (
            "ci-native-coverage.yml",
            "  CARGO_TERM_COLOR: always",
            "  COVERAGE_FILE: elsewhere\n  CARGO_TERM_COLOR: always",
        ),
        ("ci-native-coverage.yml", "jobs:\n", "defaults:\n  run:\n    shell: pwsh\njobs:\n"),
        ("ci-native-polyglot.yml", "tests/test_boris_pyo3_bridge.py", "tests/test_boris_pyo3_bridge.py --collect-only"),
        ("ci-python-quality.yml", "--cov=scpn_control", "--cov=scpn_control --no-cov"),
        ("ci-python-quality.yml", '"3.12"', '"3.17"'),
        ("ci-python-quality.yml", 'python-version: ["3.11", "3.12", "3.13"]', 'python-version: "3.12"'),
        ("ci-native-polyglot.yml", "--source=scpn_control -m pytest", "--source=scpn_control -m unittest"),
        (
            "ci-python-quality.yml",
            "        os:",
            "        exclude: [{os: ubuntu-latest, python-version: '3.12'}]\n        os:",
        ),
        (
            "ci-native-coverage.yml",
            "python -m coverage report --fail-under=100",
            "export COVERAGE_FILE=elsewhere\n          python -m coverage report --fail-under=100",
        ),
        (
            "ci-native-coverage.yml",
            "python -m coverage report --fail-under=100",
            "cd elsewhere\n          python -m coverage report --fail-under=100",
        ),
        (
            "ci-native-coverage.yml",
            "python -m coverage report --fail-under=100",
            ". other-script\n          python -m coverage report --fail-under=100",
        ),
        (
            "ci-native-coverage.yml",
            "python -m coverage report --fail-under=100",
            "COVERAGE_FILE=elsewhere\n          python -m coverage report --fail-under=100",
        ),
    ],
)
def test_distributed_guard_reads_inherited_state_and_coordinator(
    tmp_path: Path, owner: str, old: str, new: str
) -> None:
    """Reusable environment and caller failures survive physical-source inspection."""
    policy, metadata, docs = _distributed_inputs(tmp_path)
    path = tmp_path / ".github/workflows" / owner
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    assert not validate_native_coverage_matrix(policy, metadata, docs).passed
