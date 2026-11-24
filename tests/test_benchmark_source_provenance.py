# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests of source-bound benchmark provenance.
"""Exercise source snapshots against real Git repositories and recorded runs."""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scpn_control.benchmark_records import RUN_SCHEMA, BenchmarkOutput, BenchmarkRun, load_verified_latest
from scpn_control.benchmark_source_provenance import (
    SOURCE_DIGEST_ALGORITHM,
    capture_source_snapshot,
    read_index_blobs,
    runtime_identity,
)


def _git(repository: Path, *arguments: str) -> None:
    """Run Git in ``repository`` with no user or system configuration and no hooks."""
    environment = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", HOME=str(repository))
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=t", "-c", "user.email=t@example.org", *arguments],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
    )


def _repository(tmp_path: Path) -> Path:
    """Return a committed repository holding two declared sources and one undeclared file."""
    repository = tmp_path / "checkout"
    (repository / "src/scpn_control").mkdir(parents=True)
    (repository / "src/scpn_control/model.py").write_bytes(b"VALUE = 1\n")
    (repository / "pyproject.toml").write_bytes(b"[project]\nname = 'x'\n")
    (repository / "README.md").write_bytes(b"not a declared source\n")
    _git(repository, "init", "-q")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "base")
    return repository


def test_a_committed_checkout_is_clean_and_digests_only_declared_sources(tmp_path: Path) -> None:
    """Only the two declared files count; the undeclared README never enters the digest."""
    repository = _repository(tmp_path)
    snapshot = capture_source_snapshot(repository)

    expected = hashlib.sha256()
    for relative, content in (
        ("pyproject.toml", b"[project]\nname = 'x'\n"),
        ("src/scpn_control/model.py", b"VALUE = 1\n"),
    ):
        expected.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
    assert snapshot.tree_sha256 == expected.hexdigest()
    assert (snapshot.file_count, snapshot.worktree_vs_index) == (2, "clean")
    (repository / "README.md").write_bytes(b"changed\n")
    assert capture_source_snapshot(repository) == snapshot
    assert snapshot.to_record()["algorithm"] == SOURCE_DIGEST_ALGORITHM


def test_modified_untracked_and_deleted_sources_are_dirty_and_distinguishable(tmp_path: Path) -> None:
    """Each kind of change makes the checkout dirty and changes the content digest."""
    repository = _repository(tmp_path)
    clean = capture_source_snapshot(repository)

    (repository / "src/scpn_control/model.py").write_bytes(b"VALUE = 2\n")
    modified = capture_source_snapshot(repository)
    assert (modified.worktree_vs_index, modified.modified, modified.untracked, modified.deleted) == ("dirty", 1, 0, 0)
    assert modified.tree_sha256 != clean.tree_sha256

    (repository / "src/scpn_control/model.py").write_bytes(b"VALUE = 1\n")
    (repository / "src/scpn_control/extra.py").write_bytes(b"EXTRA = 1\n")
    untracked = capture_source_snapshot(repository)
    assert (untracked.worktree_vs_index, untracked.untracked, untracked.file_count) == ("dirty", 1, 3)

    (repository / "src/scpn_control/extra.py").unlink()
    (repository / "pyproject.toml").unlink()
    deleted = capture_source_snapshot(repository)
    assert (deleted.worktree_vs_index, deleted.deleted, deleted.file_count) == ("dirty", 1, 1)


def test_line_endings_and_binary_content_are_compared_as_git_stages_them(tmp_path: Path) -> None:
    """A CRLF checkout of an LF blob is clean; binary content is compared unchanged."""
    repository = _repository(tmp_path)
    (repository / "src/scpn_control/blob.bin").write_bytes(b"\0\r\n\x01")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "binary")
    (repository / "src/scpn_control/model.py").write_bytes(b"VALUE = 1\r\n")
    assert capture_source_snapshot(repository).worktree_vs_index == "clean"

    (repository / "src/scpn_control/blob.bin").write_bytes(b"\0\n\x01")
    assert capture_source_snapshot(repository).modified == 1


def test_version_four_and_conflicted_indexes(tmp_path: Path) -> None:
    """A prefix-compressed v4 index reads alike; an unmerged entry makes the checkout dirty."""
    repository = _repository(tmp_path)
    _git(repository, "update-index", "--index-version", "4")
    blobs = read_index_blobs(repository / ".git/index")
    assert set(blobs) == {"README.md", "pyproject.toml", "src/scpn_control/model.py"}
    assert capture_source_snapshot(repository).worktree_vs_index == "clean"

    blob = blobs["src/scpn_control/model.py"]
    _git(repository, "update-index", "--index-version", "2")
    _git(repository, "update-index", "--force-remove", "src/scpn_control/model.py")
    info = "".join(f"100644 {blob} {stage}\tsrc/scpn_control/model.py\n" for stage in (1, 2, 3))
    subprocess.run(
        ["git", "update-index", "--index-info"], cwd=repository, input=info.encode(), check=True, capture_output=True
    )
    assert read_index_blobs(repository / ".git/index")["src/scpn_control/model.py"] is None
    assert capture_source_snapshot(repository).worktree_vs_index == "dirty"


def test_internal_plans_secrets_and_caches_are_never_read(tmp_path: Path) -> None:
    """Excluded names under a declared root leave the digest and the counts untouched."""
    repository = _repository(tmp_path)
    before = capture_source_snapshot(repository)
    (repository / "tools").mkdir()
    for relative in ("tools/.env", "tools/api.key", "tools/credentials.json", "tools/secret_plan.md"):
        (repository / relative).write_bytes(b"do not read\n")
    (repository / "src/scpn_control/__pycache__").mkdir()
    (repository / "src/scpn_control/__pycache__/model.cpython-312.pyc").write_bytes(b"\0bytecode")
    (repository / "src/scpn_control/model.pyc").write_bytes(b"\0bytecode")
    assert capture_source_snapshot(repository) == before


def test_a_missing_or_unreadable_index_is_reported_as_unknown(tmp_path: Path) -> None:
    """No Git directory, or an index of another format, gives ``unknown`` with a digest."""
    plain = tmp_path / "plain"
    (plain / "src/scpn_control").mkdir(parents=True)
    (plain / "src/scpn_control/model.py").write_bytes(b"VALUE = 1\n")
    assert capture_source_snapshot(plain).worktree_vs_index == "unknown"

    repository = _repository(tmp_path)
    (repository / ".git/index").write_bytes(b"NOTANINDEX")
    snapshot = capture_source_snapshot(repository)
    assert (snapshot.worktree_vs_index, snapshot.file_count) == ("unknown", 2)
    with pytest.raises(ValueError, match="not a Git index"):
        read_index_blobs(repository / ".git/index")


def test_runtime_identity_hashes_a_located_native_module_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module file found first on the path is hashed by content and never executed.

    An installed or already imported native extension is set aside for the test, so the
    lookup resolves the prepended file through the ordinary import-spec search.
    """
    monkeypatch.delitem(sys.modules, "scpn_control_rs", raising=False)
    module = tmp_path / "scpn_control_rs.py"
    module.write_bytes(b"raise SystemExit('imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    identity = runtime_identity()
    assert identity["native_extension"] == {
        "module": "scpn_control_rs",
        "status": "present",
        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }
    assert isinstance(identity["python_executable_sha256"], str) and len(identity["python_executable_sha256"]) == 64


def _run(repository: Path, campaign_id: str) -> tuple[BenchmarkRun, Path]:
    """Begin a recorded run over one output inside ``repository``."""
    output = repository / "results/report.json"
    output.parent.mkdir(exist_ok=True)
    run = BenchmarkRun.begin(
        repository_root=repository,
        records_root=repository / "records",
        family="provenance",
        outputs=[BenchmarkOutput("report", output)],
        command=["python", "benchmarks/x.py"],
        campaign_id=campaign_id,
    )
    return run, output


def test_a_clean_run_is_reproducible_and_verifies_as_latest(tmp_path: Path) -> None:
    """A clean checkout with no drift seals a v2 reproducible record the latest loader accepts."""
    repository = _repository(tmp_path)
    run, output = _run(repository, "clean-run")
    output.write_text("{}", encoding="utf-8")
    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))

    assert (manifest["schema_version"], manifest["status"], manifest["reproducible"]) == (RUN_SCHEMA, "succeeded", True)
    assert manifest["source_provenance"]["drift"] is False
    assert manifest["source_provenance"]["start"] == manifest["source_provenance"]["end"]
    _, selected = load_verified_latest(repository / "records", "provenance")
    assert selected["campaign_id"] == "clean-run"


def test_source_drift_during_a_run_taints_it_and_never_advances_latest(tmp_path: Path) -> None:
    """A declared source edited mid-run seals a tainted, irreproducible record and restores outputs."""
    repository = _repository(tmp_path)
    run, output = _run(repository, "drift-run")
    output.write_text("{}", encoding="utf-8")
    (repository / "src/scpn_control/model.py").write_bytes(b"VALUE = 3\n")
    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))

    assert (manifest["status"], manifest["reproducible"], manifest["source_provenance"]["drift"]) == (
        "tainted",
        False,
        True,
    )
    assert manifest["source_provenance"]["start"]["tree_sha256"] != manifest["source_provenance"]["end"]["tree_sha256"]
    assert not (repository / "records/latest/provenance.json").exists()
    assert (run.run_directory / "failed-output/0").read_text(encoding="utf-8") == "{}"


def test_a_dirty_start_succeeds_but_is_not_reproducible(tmp_path: Path) -> None:
    """An edited but stable checkout is recorded as such and never claimed reproducible."""
    repository = _repository(tmp_path)
    (repository / "src/scpn_control/model.py").write_bytes(b"VALUE = 9\n")
    run, output = _run(repository, "dirty-run")
    output.write_text("{}", encoding="utf-8")
    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))

    assert (manifest["status"], manifest["reproducible"]) == ("succeeded", False)
    assert manifest["source_provenance"]["start"]["worktree_vs_index"] == "dirty"


def test_a_deleted_tracked_secret_or_bytecode_file_is_not_counted(tmp_path: Path) -> None:
    """Tracked files the snapshot never reads cannot count as deleted declared sources."""
    repository = _repository(tmp_path)
    (repository / "tools").mkdir()
    (repository / "tools/api.key").write_bytes(b"tracked by mistake\n")
    (repository / "src/scpn_control/old.pyc").write_bytes(b"\0bytecode")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "tracked excluded files")
    (repository / "tools/api.key").unlink()
    (repository / "src/scpn_control/old.pyc").unlink()
    snapshot = capture_source_snapshot(repository)
    assert (snapshot.worktree_vs_index, snapshot.deleted) == ("clean", 0)


def test_a_linked_worktree_resolves_its_gitdir_file(tmp_path: Path) -> None:
    """A ``.git`` file pointing at a worktree index is followed; an unreadable one is unknown."""
    repository = _repository(tmp_path)
    linked = tmp_path / "linked"
    _git(repository, "worktree", "add", "-q", str(linked))
    assert (linked / ".git").is_file()
    assert capture_source_snapshot(linked).worktree_vs_index == "clean"

    # Git for Windows marks the .git file hidden, and Windows refuses to open a
    # hidden file for truncating writes; replace the file instead.
    (linked / ".git").unlink()
    (linked / ".git").write_text("not a gitdir pointer\n", encoding="utf-8")
    assert capture_source_snapshot(linked).worktree_vs_index == "unknown"


def test_a_version_four_index_decodes_a_multi_byte_prefix_strip(tmp_path: Path) -> None:
    """A path following a long one strips more than 127 bytes, a multi-byte offset varint."""
    repository = _repository(tmp_path)
    long_name = "src/scpn_control/" + "a" * 150 + ".py"
    (repository / long_name).write_bytes(b"LONG = 1\n")
    (repository / "src/scpn_control/b.py").write_bytes(b"B = 1\n")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "long path")
    _git(repository, "update-index", "--index-version", "4")
    blobs = read_index_blobs(repository / ".git/index")
    assert {long_name, "src/scpn_control/b.py"} <= set(blobs)
    assert capture_source_snapshot(repository).worktree_vs_index == "clean"


def test_a_version_three_index_with_extended_flags_is_read(tmp_path: Path) -> None:
    """An intent-to-add entry carries the extended flag word of a version 3 index."""
    repository = _repository(tmp_path)
    (repository / "src/scpn_control/planned.py").write_bytes(b"PLANNED = 1\n")
    _git(repository, "add", "-N", "src/scpn_control/planned.py")
    _git(repository, "update-index", "--index-version", "3")
    blobs = read_index_blobs(repository / ".git/index")
    assert "src/scpn_control/planned.py" in blobs and "src/scpn_control/model.py" in blobs
    assert capture_source_snapshot(repository).modified == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"DIRC\x00\x00\x00\x05\x00\x00\x00\x00", "unsupported Git index version 5"),
        (b"DIRC\x00\x00\x00\x02\x00\x00\x00\x01" + b"\x00" * 10, "truncated Git index entry"),
    ],
)
def test_malformed_indexes_are_refused(tmp_path: Path, payload: bytes, message: str) -> None:
    """Another version or a cut-off entry is refused rather than misread."""
    index = tmp_path / "index"
    index.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        read_index_blobs(index)


def _resign(repository: Path, family: str, change: dict[str, object | None]) -> None:
    """Rewrite the latest run manifest with ``change`` and re-sign its payload and pointer."""
    from scpn_control.benchmark_record_integrity import _json_bytes, _sha256_bytes

    latest_path = repository / "records/latest" / f"{family}.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest_path = repository / "records" / latest["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("payload_sha256")
    for key, value in change.items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    manifest["payload_sha256"] = _sha256_bytes(_json_bytes(manifest))
    manifest_bytes = _json_bytes(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(manifest_bytes)
    latest["manifest_sha256"] = _sha256_bytes(manifest_bytes)
    latest_path.write_bytes(_json_bytes(latest))


def _sealed_clean_run(repository: Path) -> dict[str, object]:
    """Seal one clean reproducible run and return its manifest."""
    run, output = _run(repository, "verified-run")
    output.write_text("{}", encoding="utf-8")
    manifest: dict[str, object] = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))
    return manifest


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: {"source_provenance": None}, "source provenance is missing"),
        (lambda m: {"source_provenance": {**m["source_provenance"], "end": None}}, "needs start and end"),
        (
            lambda m: {"source_provenance": {**m["source_provenance"], "end": {"tree_sha256": "0" * 64}}},
            "malformed or differ",
        ),
        (lambda m: {"source_provenance": {**m["source_provenance"], "drift": True}}, "records source drift"),
        (lambda m: {"runtime": None}, "reproducibility or runtime identity"),
    ],
)
def test_the_latest_loader_refuses_inconsistent_v2_provenance(tmp_path: Path, mutate: object, message: str) -> None:
    """Even a correctly re-signed v2 manifest is refused when its provenance block is inconsistent."""
    repository = _repository(tmp_path)
    manifest = _sealed_clean_run(repository)
    assert callable(mutate)
    _resign(repository, "provenance", mutate(manifest))
    with pytest.raises(ValueError, match=message):
        load_verified_latest(repository / "records", "provenance")


def test_a_historical_v1_manifest_stays_readable(tmp_path: Path) -> None:
    """Records sealed before source provenance existed are verified as they were written."""
    repository = _repository(tmp_path)
    _sealed_clean_run(repository)
    _resign(
        repository,
        "provenance",
        {
            "schema_version": "scpn-control.benchmark-run.v1",
            "source_provenance": None,
            "runtime": None,
            "reproducible": None,
        },
    )
    _, selected = load_verified_latest(repository / "records", "provenance")
    assert selected["schema_version"] == "scpn-control.benchmark-run.v1"


def test_an_unreadable_native_module_file_is_recorded_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already imported extension whose file has since been removed is reported, not guessed.

    ``find_spec`` returns the spec of a module present in ``sys.modules``; its origin file
    is gone, as after an upgrade that replaced the installed extension.
    """
    module = types.ModuleType("scpn_control_rs")
    module.__spec__ = importlib.machinery.ModuleSpec("scpn_control_rs", None, origin=str(tmp_path / "removed.so"))
    monkeypatch.setitem(sys.modules, "scpn_control_rs", module)

    native = runtime_identity()["native_extension"]
    assert native == {"module": "scpn_control_rs", "status": "present", "sha256": "unavailable"}
