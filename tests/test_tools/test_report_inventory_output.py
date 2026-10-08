# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real inventory publication and recovery tests.
"""Exercise source protection and paired output recovery through public calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from report_lifecycle_fixtures import AUDIT_AS_OF, copy_corpus, run_cli

from tools.report_inventory_output import write_inventory_outputs
from tools.validation_report_freshness import ROOT, build_validation_report_freshness_matrix


@pytest.mark.parametrize("target", ["report", "registry", "refresh", "new-report", "new-refresh"])
@pytest.mark.parametrize("option", ["--output-json", "--output-md"])
def test_cli_refuses_input_destinations(tmp_path: Path, target: str, option: str) -> None:
    """Both output formats preserve maintained input bytes and reserved namespaces."""
    corpus = copy_corpus(tmp_path / "corpus")
    paths = {
        "report": corpus.reports / "gk_interface_artifacts.json",
        "registry": corpus.registry,
        "refresh": next((corpus.root / "validation/report_refreshes").rglob("*.json")),
        "new-report": corpus.reports / "new.json",
        "new-refresh": corpus.root / "validation/report_refreshes/new.json",
    }
    before = corpus.snapshot()
    result = run_cli(corpus, [option, str(paths[target])])
    assert result.returncode == 1
    assert "Inventory outputs must" in result.stderr
    assert not result.stdout
    assert corpus.snapshot() == before


@pytest.mark.parametrize("kind", ["existing", "absent", "hardlink"])
@pytest.mark.parametrize("option", ["--output-json", "--output-md"])
def test_cli_preserves_selected_repository_claim_ledger(tmp_path: Path, kind: str, option: str) -> None:
    """Both formats preserve the selected corpus's ledger, including aliases and reserved paths."""
    corpus = copy_corpus(tmp_path / "corpus")
    ledger = corpus.root / "validation/public_claim_ledger.json"
    output = ledger
    if kind != "absent":
        ledger.write_bytes((ROOT / "validation/public_claim_ledger.json").read_bytes())
    if kind == "hardlink":
        output = corpus.root / "ledger-alias.json"
        output.hardlink_to(ledger)
    before = corpus.snapshot()
    result = run_cli(corpus, [option, str(output)])
    assert result.returncode == 1
    assert "must not replace source evidence or its registry" in result.stderr
    assert not result.stdout
    assert corpus.snapshot() == before
    if kind == "absent":
        assert not ledger.exists()
    else:
        assert output.read_bytes() == (ROOT / "validation/public_claim_ledger.json").read_bytes()


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("option", ["--output-json", "--output-md"])
def test_cli_preserves_command_repository_claim_ledger(tmp_path: Path, existing: bool, option: str) -> None:
    """A cold source-identical command protects its own ledger when reading a separate corpus."""
    corpus = copy_corpus(tmp_path / "corpus")
    command_root = tmp_path / "command"
    tools = command_root / "tools"
    tools.mkdir(parents=True)
    source_names = [
        "__init__.py",
        "validation_report_freshness.py",
        "report_lifecycle_types.py",
        "report_lifecycle_values.py",
        "report_lifecycle_registry.py",
        "report_lifecycle_admission.py",
        "report_inventory_types.py",
        "report_inventory_output.py",
        "inventory_file_output.py",
    ]
    for name in source_names:
        shutil.copy2(ROOT / "tools" / name, tools / name)
    ledger = command_root / "validation/public_claim_ledger.json"
    if existing:
        ledger.parent.mkdir()
        ledger.write_bytes((ROOT / "validation/public_claim_ledger.json").read_bytes())
    before = corpus.snapshot()
    result = subprocess.run(
        [sys.executable, str(tools / "validation_report_freshness.py"), *corpus.arguments(), option, str(ledger)],
        cwd=corpus.root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 1
    assert "must not replace source evidence or its registry" in result.stderr
    assert not result.stdout
    assert corpus.snapshot() == before
    assert all((tools / name).read_bytes() == (ROOT / "tools" / name).read_bytes() for name in source_names)
    if existing:
        assert ledger.read_bytes() == (ROOT / "validation/public_claim_ledger.json").read_bytes()
    else:
        assert not ledger.exists()
        assert not ledger.parent.exists()


@pytest.mark.parametrize("kind", ["hardlink", "symlink", "directory"])
def test_cli_refuses_filesystem_aliases(tmp_path: Path, kind: str) -> None:
    """Filesystem aliases and directory outputs cannot replace a report."""
    corpus = copy_corpus(tmp_path / "corpus")
    output = corpus.root / "outside.json"
    source = corpus.reports / "gk_interface_artifacts.json"
    if kind == "hardlink":
        output.hardlink_to(source)
    elif kind == "symlink":
        output.symlink_to(source)
    else:
        output.mkdir()
    before = corpus.snapshot()
    result = run_cli(corpus, ["--output-json", str(output)])
    assert result.returncode == 1
    assert corpus.snapshot() == before
    assert output.is_symlink() if kind == "symlink" else output.exists()


@pytest.mark.parametrize("kind", ["same", "hardlink", "resolved"])
def test_cli_refuses_overlapping_outputs(tmp_path: Path, kind: str) -> None:
    """Distinct option spellings must still identify two separate output files."""
    corpus = copy_corpus(tmp_path / "corpus")
    first = corpus.root / "inventory.json"
    first.write_bytes(b"old inventory\n")
    second = first
    if kind == "hardlink":
        second = corpus.root / "inventory.md"
        second.hardlink_to(first)
    elif kind == "resolved":
        directory = corpus.root / "directory"
        directory.mkdir()
        second = directory / "../inventory.json"
    before = corpus.snapshot()
    result = run_cli(corpus, ["--output-json", str(first), "--output-md", str(second)])
    assert result.returncode == 1
    assert "must use distinct files" in result.stderr
    assert first.read_bytes() == second.read_bytes() == b"old inventory\n"
    assert corpus.snapshot() == before


def test_public_writer_replaces_complete_pair_without_changing_inputs(tmp_path: Path) -> None:
    """The public writer serialises the validated matrix and preserves predecessors' mode."""
    corpus = copy_corpus(tmp_path / "corpus")
    matrix = build_validation_report_freshness_matrix(
        corpus.reports,
        registry_path=corpus.registry,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    output_json = corpus.root / "output/nested/inventory.json"
    output_md = corpus.root / "output/nested/inventory.md"
    output_json.parent.mkdir(parents=True)
    output_json.write_bytes(b"old JSON")
    output_md.write_bytes(b"old Markdown")
    output_json.chmod(0o600)
    mode = output_json.stat().st_mode
    before = corpus.snapshot()
    write_inventory_outputs(matrix, json_path=output_json, markdown_path=output_md, registry_path=corpus.registry)
    assert json.loads(output_json.read_bytes()) == matrix.to_dict()
    assert output_md.read_text(encoding="utf-8") == matrix.to_markdown()
    assert output_json.stat().st_mode == mode
    assert corpus.snapshot() == before
    assert sorted(path.name for path in output_json.parent.iterdir()) == ["inventory.json", "inventory.md"]


_AUDITED_PUBLICATION = r"""
import json
import sys
from datetime import datetime
from pathlib import Path
from tools.report_inventory_output import InventoryOutputError, write_inventory_outputs
from tools.validation_report_freshness import build_validation_report_freshness_matrix

root, mode, existing = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "yes"
first, second = root / "inventory.json", root / "inventory.md"
matrix = build_validation_report_freshness_matrix(
    root / "validation/reports", registry_path=root / "validation/report_lifecycle_registry.json",
    as_of=datetime.fromisoformat("2026-09-05T13:00:01+00:00"), max_age_days=21,
)
def permission(event: str, arguments: tuple[object, ...]) -> None:
    if event == "open":
        path = str(arguments[0])
        if mode in {"stage", "backup"} and ".inventory.md." in path and path.endswith("." + ("tmp" if mode == "stage" else "backup")):
            raise PermissionError("private runtime details")
    if event == "os.rename":
        source, destination = str(arguments[0]), str(arguments[1])
        if mode == "alias" and destination == str(first) and source.endswith(".tmp"):
            second.hardlink_to(Path(source))
        if destination == str(second):
            if mode == "occupant":
                first.write_bytes(b"another writer")
            raise PermissionError("private runtime details")
        if mode == "recovery" and source.endswith(".backup"):
            raise PermissionError("private recovery details")
sys.addaudithook(permission)
try:
    write_inventory_outputs(matrix, json_path=first, markdown_path=second,
                            registry_path=root / "validation/report_lifecycle_registry.json")
except InventoryOutputError as error:
    print(json.dumps({"kind": "refusal", "message": str(error),
                      "retained": [str(path) for path in error.recovery_paths]}))
except OSError:
    print(json.dumps({"kind": "native", "retained": []}))
else:
    raise AssertionError("publication unexpectedly succeeded")
"""


@pytest.mark.parametrize("mode", ["stage", "backup", "replace", "recovery", "occupant"])
def test_public_writer_recovers_or_retains_real_predecessors(tmp_path: Path, mode: str) -> None:
    """Runtime audit permissions exercise actual staging, replacement and recovery files."""
    corpus = copy_corpus(tmp_path / "corpus")
    first, second = corpus.root / "inventory.json", corpus.root / "inventory.md"
    first.write_bytes(b"old JSON")
    second.write_bytes(b"old Markdown")
    before = corpus.snapshot()
    result = subprocess.run(
        [sys.executable, "-c", _AUDITED_PUBLICATION, str(corpus.root), mode, "yes"],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert corpus.snapshot() == before
    assert second.read_bytes() == b"old Markdown"
    if mode in {"recovery", "occupant"}:
        assert receipt["kind"] == "refusal"
        assert receipt["message"] == "Inventory output recovery is incomplete; retained files require inspection"
        retained = [Path(path) for path in receipt["retained"]]
        assert retained and all(path.is_file() for path in retained)
        assert any(path.read_bytes() == b"old JSON" for path in retained)
        if mode == "occupant":
            assert first.read_bytes() == b"another writer"
        else:
            assert json.loads(first.read_bytes())["summary"]["report_count"] == 128
    else:
        assert receipt["kind"] == "native"
        assert first.read_bytes() == b"old JSON"
        assert not list(corpus.root.glob(".inventory.*"))


def test_failed_pair_removes_new_first_output(tmp_path: Path) -> None:
    """A failed second replacement removes the newly created first output."""
    corpus = copy_corpus(tmp_path / "corpus")
    before = corpus.snapshot()
    result = subprocess.run(
        [sys.executable, "-c", _AUDITED_PUBLICATION, str(corpus.root), "replace", "no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["kind"] == "native"
    assert not (corpus.root / "inventory.json").exists()
    assert not (corpus.root / "inventory.md").exists()
    assert not list(corpus.root.glob(".inventory.*"))
    assert corpus.snapshot() == before


def test_cli_native_output_failure_uses_fixed_message(tmp_path: Path) -> None:
    """An actual blocked parent refuses publication without native path/error text."""
    corpus = copy_corpus(tmp_path / "corpus")
    blocked = corpus.root / "private-parent"
    blocked.write_bytes(b"parent is a file")
    output_md = corpus.root / "inventory.md"
    output_md.write_bytes(b"old Markdown")
    before = corpus.snapshot()
    result = run_cli(corpus, ["--output-json", str(blocked / "inventory.json"), "--output-md", str(output_md)])
    assert result.returncode == 1
    assert result.stderr == "Validation report freshness outputs could not be published\n"
    assert blocked.read_bytes() == b"parent is a file"
    assert output_md.read_bytes() == b"old Markdown"
    assert corpus.snapshot() == before


@pytest.mark.parametrize("existing", [True, False])
def test_symlinked_corpus_protects_refresh_tree_actually_read(tmp_path: Path, existing: bool) -> None:
    """A linked report corpus protects its lexical refresh tree rather than another repository's tree."""
    corpus = copy_corpus(tmp_path / "repository")
    link = tmp_path / "link/validation"
    link.mkdir(parents=True)
    (link / "reports").symlink_to(corpus.reports, target_is_directory=True)
    shutil.copytree(corpus.root / "validation/report_refreshes", link / "report_refreshes")
    control = run_cli(corpus, ["--reports-root", str(link / "reports")])
    assert control.returncode == 0, control.stderr
    output = link / "report_refreshes/2026-08-27" / ("aer_admission.json" if existing else "new.json")
    before = {path: path.read_bytes() for path in (link / "report_refreshes").rglob("*.json")}
    result = run_cli(corpus, ["--reports-root", str(link / "reports"), "--output-json", str(output)])
    assert result.returncode == 1
    assert "outside report and refresh namespaces" in result.stderr
    assert {path: path.read_bytes() for path in (link / "report_refreshes").rglob("*.json")} == before
    assert output.exists() == existing


def test_case_variant_new_output_names_follow_native_path_identity(tmp_path: Path) -> None:
    """Refuse case-equivalent names on the actual volume while retaining distinct names."""
    corpus = copy_corpus(tmp_path / "corpus")
    probe = corpus.root / "CaseProbe"
    probe.write_bytes(b"filesystem identity probe")
    case_insensitive = (corpus.root / "caseprobe").exists()
    probe.unlink()
    first = corpus.root / "Inventory.json"
    second = corpus.root / "inventory.JSON"
    assert not first.exists() and not second.exists()
    result = run_cli(corpus, ["--output-json", str(first), "--output-md", str(second)])
    if case_insensitive:
        assert result.returncode == 1
        assert "must use distinct files" in result.stderr
        assert not first.exists() and not second.exists()
    else:
        assert result.returncode == 0
        assert json.loads(first.read_bytes())["summary"]["report_count"] == 128
        assert second.read_bytes().startswith(b"# SCPN Control")


def test_public_writer_rechecks_alias_after_first_new_publication(tmp_path: Path) -> None:
    """A real inode alias appearing with publication cannot receive the second format."""
    corpus = copy_corpus(tmp_path / "corpus")
    before = corpus.snapshot()
    result = subprocess.run(
        [sys.executable, "-c", _AUDITED_PUBLICATION, str(corpus.root), "alias", "no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["kind"] == "refusal"
    assert receipt["message"] == "Inventory outputs must use distinct files"
    assert not (corpus.root / "inventory.json").exists()
    assert json.loads((corpus.root / "inventory.md").read_bytes())["summary"]["report_count"] == 128
    assert corpus.snapshot() == before


def test_missing_refresh_namespace_is_reserved_before_parent_creation(tmp_path: Path) -> None:
    """A real blocked report subset reserves its absent refresh namespace without creating it."""
    corpus = copy_corpus(tmp_path / "corpus")
    registry = cast(dict[str, object], json.loads(corpus.registry.read_bytes()))
    records = cast(list[dict[str, object]], registry["reports"])
    name = "validation/reports/gk_interface_artifacts.json"
    record = next(item for item in records if item["path"] == name)
    for path in corpus.reports.rglob("*.json"):
        if path != corpus.root / name:
            path.unlink()
    shutil.rmtree(corpus.root / "validation/report_refreshes")
    registry["reports"] = [record]
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 1,
        "historical_only": 0,
        "rerunnable_local": 0,
    }
    corpus.registry.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    before = corpus.snapshot()
    result = run_cli(corpus, ["--output-json", str(corpus.root / "validation/report_refreshes/new.json")])
    assert result.returncode == 1
    assert "outside report and refresh namespaces" in result.stderr
    assert not (corpus.root / "validation/report_refreshes").exists()
    assert corpus.snapshot() == before
