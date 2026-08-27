# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Immutable benchmark record tests
"""Exercise immutable benchmark custody through its production API."""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

import pytest

import scpn_control.benchmark_records as records_module
from scpn_control.benchmark_records import (
    LATEST_SCHEMA,
    RUN_SCHEMA,
    BenchmarkOutput,
    BenchmarkRun,
    load_verified_latest,
    new_campaign_id,
    redact_command,
    require_recorded_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _begin(records: Path, output: Path, campaign_id: str) -> BenchmarkRun:
    return BenchmarkRun.begin(
        repository_root=REPO_ROOT,
        records_root=records,
        family="controller-latency",
        outputs=[BenchmarkOutput("report", output)],
        command=["python", "benchmarks/controller_latency.py", "--steps", "5", "--warmup", "1"],
        campaign_id=campaign_id,
        measurement={"repeat_definition": "one process invocation"},
    )


def test_two_runs_preserve_legacy_and_both_immutable_reports(tmp_path: Path) -> None:
    """Two runs retain old bytes, both results, and a verified latest pointer."""
    output = tmp_path / "latest" / "controller_latency.json"
    output.parent.mkdir()
    output.write_text('{"generation":0}\n', encoding="utf-8")
    records = tmp_path / "records"

    first = _begin(records, output, "20260828T010000.000000Z-first")
    output.write_text('{"generation":1}\n', encoding="utf-8")
    first_manifest_path = first.finish(exit_code=0)

    second = _begin(records, output, "20260828T010001.000000Z-second")
    output.write_text('{"generation":2}\n', encoding="utf-8")
    second_manifest_path = second.finish(exit_code=0)

    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["schema_version"] == RUN_SCHEMA
    assert first_manifest["status"] == "succeeded"
    assert second_manifest["status"] == "succeeded"
    assert first_manifest["artifacts"][0]["sha256"] != second_manifest["artifacts"][0]["sha256"]
    assert len(first_manifest["legacy_inputs"]) == 1
    assert len(second_manifest["legacy_inputs"]) == 1
    assert Path(first_manifest["legacy_inputs"][0]["archived_path"]).read_text(encoding="utf-8") == '{"generation":0}\n'

    latest, selected = load_verified_latest(records, "controller-latency")
    assert latest["schema_version"] == LATEST_SCHEMA
    assert latest["campaign_id"] == second.campaign_id
    assert selected["campaign_id"] == second.campaign_id
    assert len(list((records / "runs" / "controller-latency").iterdir())) == 2


def test_campaign_collision_fails_before_output_changes(tmp_path: Path) -> None:
    """A reused campaign identifier fails before a producer can overwrite output."""
    output = tmp_path / "report.json"
    output.write_text("original", encoding="utf-8")
    records = tmp_path / "records"
    _begin(records, output, "fixed-campaign")

    with pytest.raises(FileExistsError):
        _begin(records, output, "fixed-campaign")
    assert output.read_text(encoding="utf-8") == "original"


def test_failed_run_is_preserved_without_advancing_latest(tmp_path: Path) -> None:
    """Failed output remains inspectable without becoming the selected run."""
    output = tmp_path / "report.json"
    output.write_text("first", encoding="utf-8")
    records = tmp_path / "records"
    first = _begin(records, output, "successful-run")
    first.finish(exit_code=0)

    failed = _begin(records, output, "failed-run")
    output.write_text("partial", encoding="utf-8")
    failed_manifest = json.loads(failed.finish(exit_code=9).read_text(encoding="utf-8"))
    latest, _ = load_verified_latest(records, "controller-latency")

    assert failed_manifest["status"] == "failed"
    assert failed_manifest["artifacts"][0]["sha256"]
    assert latest["campaign_id"] == "successful-run"


def test_latest_loader_rejects_manifest_tampering(tmp_path: Path) -> None:
    """A latest index cannot admit a manifest whose bytes changed."""
    output = tmp_path / "report.json"
    output.write_text("result", encoding="utf-8")
    records = tmp_path / "records"
    run = _begin(records, output, "tamper-target")
    manifest = run.finish(exit_code=0)
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_verified_latest(records, "controller-latency")


def test_command_redaction_covers_inline_and_following_secrets() -> None:
    """Recorded commands redact both supported credential argument forms."""
    command = redact_command(["tool", "--token", "alpha", "--api-key=beta", "--steps", "5"])
    assert command == ["tool", "--token", "[REDACTED_SECRET]", "--api-key=[REDACTED_SECRET]", "--steps", "5"]


def test_generated_campaign_identifier_is_valid_and_unique() -> None:
    """Generated campaign identifiers are distinct and filesystem-safe."""
    first = new_campaign_id()
    second = new_campaign_id()
    assert first != second
    assert first.endswith(tuple("0123456789abcdef"))


def test_persistent_output_requires_recorded_campaign(tmp_path: Path) -> None:
    """A producer cannot directly replace repository benchmark evidence."""
    repository = tmp_path / "repository"
    output = repository / "validation" / "reports" / "result.json"
    with pytest.raises(RuntimeError, match="run_recorded_benchmark"):
        require_recorded_campaign(output, repository_root=repository)


def test_temporary_output_does_not_require_campaign(tmp_path: Path) -> None:
    """Scratch output outside persistent evidence roots remains available."""
    repository = tmp_path / "repository"
    output = tmp_path / "scratch" / "result.json"
    assert require_recorded_campaign(output, repository_root=repository) is None


def test_directory_artifact_is_copied_and_digest_bound(tmp_path: Path) -> None:
    """A multi-file benchmark output directory is retained as one artifact."""
    output = tmp_path / "parity-artifacts"
    output.mkdir()
    (output / "cpu.json").write_text("cpu", encoding="utf-8")
    records = tmp_path / "records"
    run = BenchmarkRun.begin(
        repository_root=REPO_ROOT,
        records_root=records,
        family="jax-parity",
        outputs=[BenchmarkOutput("parity-artifacts", output)],
        command=["python", "validation/benchmark_jax_gk_parity.py"],
        campaign_id="directory-run",
    )
    (output / "gpu.json").write_text("gpu", encoding="utf-8")

    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))
    immutable = Path(manifest["artifacts"][0]["immutable_path"])
    assert manifest["artifacts"][0]["kind"] == "directory"
    assert {path.name for path in immutable.iterdir()} == {"cpu.json", "gpu.json"}


def test_invalid_identifiers_and_duplicate_roles_fail_before_execution(tmp_path: Path) -> None:
    """Malformed identifiers and ambiguous output roles are rejected."""
    with pytest.raises(ValueError, match="output role"):
        BenchmarkOutput("bad role", tmp_path / "report.json")
    with pytest.raises(ValueError, match="benchmark family"):
        BenchmarkRun.begin(
            repository_root=REPO_ROOT,
            records_root=tmp_path / "records",
            family="bad family",
            outputs=[],
            command=["producer"],
        )
    with pytest.raises(ValueError, match="roles must be unique"):
        BenchmarkRun.begin(
            repository_root=REPO_ROOT,
            records_root=tmp_path / "records",
            family="duplicate-role",
            outputs=[BenchmarkOutput("report", tmp_path / "a"), BenchmarkOutput("report", tmp_path / "b")],
            command=["producer"],
            campaign_id="duplicate-role-run",
        )


def test_campaign_guard_accepts_valid_id_and_rejects_malformed_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent producers receive only validated wrapper campaign IDs."""
    repository = tmp_path / "repository"
    output = repository / "artifacts" / "report.json"
    monkeypatch.setenv(records_module.CAMPAIGN_ENV, "valid-campaign")
    assert require_recorded_campaign(output, repository_root=repository) == "valid-campaign"
    monkeypatch.setenv(records_module.CAMPAIGN_ENV, "invalid campaign")
    with pytest.raises(ValueError, match="campaign id"):
        require_recorded_campaign(output, repository_root=repository)


def test_directory_artifacts_reject_symlinks(tmp_path: Path) -> None:
    """Directory digests cannot hide mutable content behind a symlink."""
    output = tmp_path / "artifacts"
    output.mkdir()
    target = tmp_path / "target.json"
    target.write_text("target", encoding="utf-8")
    (output / "linked.json").symlink_to(target)

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        BenchmarkRun.begin(
            repository_root=REPO_ROOT,
            records_root=tmp_path / "records",
            family="symlink-rejection",
            outputs=[BenchmarkOutput("directory", output)],
            command=["producer"],
            campaign_id="symlink-run",
        )


def test_missing_output_fails_run_and_finalisation_is_one_shot(tmp_path: Path) -> None:
    """Missing declared outputs fail the run and a manifest cannot be resealed."""
    output = tmp_path / "missing.json"
    run = _begin(tmp_path / "records", output, "missing-output")
    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["missing_output_roles"] == ["report"]
    with pytest.raises(FileExistsError, match="already finalised"):
        run.finish(exit_code=0)


def test_legacy_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    """An existing legacy object with the wrong bytes cannot be admitted."""
    output = tmp_path / "report.json"
    output.write_text("expected", encoding="utf-8")
    digest = records_module._sha256_path(output)
    records = tmp_path / "records"
    legacy = records / "legacy" / digest / "report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="legacy benchmark digest mismatch"):
        _begin(records, output, "legacy-mismatch")


def test_latest_loader_rejects_schema_escape_and_inadmissible_manifest(tmp_path: Path) -> None:
    """Latest selection fails closed on metadata, path, and run-status drift."""
    output = tmp_path / "report.json"
    output.write_text("result", encoding="utf-8")
    records = tmp_path / "records"
    run = _begin(records, output, "latest-validation")
    manifest_path = run.finish(exit_code=0)
    latest_path = records / "latest" / "controller-latency.json"
    original_latest = latest_path.read_bytes()

    latest = json.loads(original_latest)
    latest["schema_version"] = "wrong"
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema or family"):
        load_verified_latest(records, "controller-latency")

    latest = json.loads(original_latest)
    latest["manifest_path"] = "../outside.json"
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_verified_latest(records, "controller-latency")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    latest = json.loads(original_latest)
    latest["manifest_sha256"] = records_module._sha256_bytes(manifest_bytes)
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(ValueError, match="inadmissible"):
        load_verified_latest(records, "controller-latency")


def test_host_provenance_fallbacks_are_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unavailable Git and host telemetry produce explicit fallback values."""
    assert records_module._git_commit(tmp_path) == "unknown"
    monkeypatch.setattr(records_module, "Path", lambda *_args: (_ for _ in ()).throw(OSError("no procfs")))
    monkeypatch.setattr(platform, "processor", lambda: "fallback-cpu")
    assert records_module._cpu_model() == "fallback-cpu"

    monkeypatch.setattr(records_module, "Path", Path)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "getloadavg", raising=False)
    context = records_module._host_context()
    assert context["cpu_affinity"] is None
    assert context["load_average"] is None


def test_git_commit_resolves_detached_loose_packed_and_worktree_refs(tmp_path: Path) -> None:
    """Commit provenance resolves common Git storage forms without a subprocess."""
    detached = tmp_path / "detached"
    (detached / ".git").mkdir(parents=True)
    detached_sha = "A" * 40
    (detached / ".git" / "HEAD").write_text(detached_sha, encoding="utf-8")
    assert records_module._git_commit(detached) == detached_sha.lower()

    loose = tmp_path / "loose"
    loose_ref = loose / ".git" / "refs" / "heads" / "main"
    loose_ref.parent.mkdir(parents=True)
    loose_sha = "b" * 40
    (loose / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    loose_ref.write_text(loose_sha, encoding="utf-8")
    assert records_module._git_commit(loose) == loose_sha

    packed = tmp_path / "packed"
    (packed / ".git").mkdir(parents=True)
    packed_sha = "c" * 40
    (packed / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (packed / ".git" / "packed-refs").write_text(f"{packed_sha} refs/heads/main\n", encoding="utf-8")
    assert records_module._git_commit(packed) == packed_sha

    worktree = tmp_path / "worktree"
    git_directory = tmp_path / "git-data" / "worktrees" / "tree"
    common_directory = tmp_path / "git-data"
    git_directory.mkdir(parents=True)
    worktree.mkdir()
    worktree_sha = "d" * 40
    (worktree / ".git").write_text(f"gitdir: {git_directory}\n", encoding="utf-8")
    (git_directory / "HEAD").write_text("ref: refs/heads/worktree\n", encoding="utf-8")
    (git_directory / "commondir").write_text("../..\n", encoding="utf-8")
    worktree_ref = common_directory / "refs" / "heads" / "worktree"
    worktree_ref.parent.mkdir(parents=True)
    worktree_ref.write_text(worktree_sha, encoding="utf-8")
    assert records_module._git_commit(worktree) == worktree_sha


@pytest.mark.parametrize(
    "git_marker, head, packed, expected",
    [
        ("not-a-gitdir", "", "", "unknown"),
        (None, "short", "", "unknown"),
        (None, "ref: refs/heads/missing", "", "unknown"),
        (None, "ref: refs/heads/main", "bad refs/heads/main", "unknown"),
        (None, "ref: refs/heads/main", "# no matching ref", "unknown"),
    ],
)
def test_git_commit_rejects_malformed_or_missing_refs(
    tmp_path: Path, git_marker: str | None, head: str, packed: str, expected: str
) -> None:
    """Malformed Git metadata never becomes a fabricated source commit."""
    repository = tmp_path / (git_marker or head.replace("/", "-").replace(" ", "-") or "case")
    repository.mkdir()
    if git_marker is not None:
        (repository / ".git").write_text(git_marker, encoding="utf-8")
    else:
        (repository / ".git").mkdir()
        (repository / ".git" / "HEAD").write_text(head, encoding="utf-8")
        if packed:
            (repository / ".git" / "packed-refs").write_text(packed, encoding="utf-8")
    assert records_module._git_commit(repository) == expected


def test_sha256_path_rejects_missing_path(tmp_path: Path) -> None:
    """Digesting a missing artifact fails rather than inventing a checksum."""
    with pytest.raises(FileNotFoundError):
        records_module._sha256_path(tmp_path / "missing")


def test_dependency_and_cpu_fallback_branches_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent locks and procfs model lines retain explicit deterministic values."""
    digest, included = records_module._dependency_lock_digest(tmp_path)
    assert len(digest) == 64
    assert included == []

    class _CpuInfoWithoutModel:
        def __init__(self, *_args: object) -> None:
            pass

        def read_text(self, **_kwargs: object) -> str:
            return "processor: 0\n"

    monkeypatch.setattr(records_module, "Path", _CpuInfoWithoutModel)
    monkeypatch.setattr(platform, "processor", lambda: "fallback-cpu")
    assert records_module._cpu_model() == "fallback-cpu"


def test_concurrent_legacy_copy_races_reuse_digest_identical_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent identical legacy archive wins safely for files and directories."""
    file_output = tmp_path / "result.json"
    file_output.write_text("file", encoding="utf-8")
    file_records = tmp_path / "file-records"
    original_file_copy = records_module._copy_file_exclusive

    def _raced_file_copy(source: Path, destination: Path) -> None:
        original_file_copy(source, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(records_module, "_copy_file_exclusive", _raced_file_copy)
    _begin(file_records, file_output, "file-race")

    directory_output = tmp_path / "directory"
    directory_output.mkdir()
    (directory_output / "result.json").write_text("directory", encoding="utf-8")
    directory_records = tmp_path / "directory-records"
    original_copytree = shutil.copytree

    def _raced_copytree(source: Path, destination: Path) -> str:
        original_copytree(source, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(shutil, "copytree", _raced_copytree)
    BenchmarkRun.begin(
        repository_root=REPO_ROOT,
        records_root=directory_records,
        family="directory-race",
        outputs=[BenchmarkOutput("directory", directory_output)],
        command=["producer"],
        campaign_id="directory-race",
    )
