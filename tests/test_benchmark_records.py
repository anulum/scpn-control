# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Immutable benchmark record tests
"""Exercise immutable benchmark custody through its production API."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from child_coverage import CHILD_COVERAGE_PRELUDE, child_environment

import scpn_control.benchmark_records as records_module
from scpn_control.benchmark_artifacts import sha256_path
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


@pytest.mark.parametrize("directory_first", [False, True])
def test_public_producer_seals_suffix_colliding_roles_and_retains_predecessors(
    tmp_path: Path, directory_first: bool
) -> None:
    """Real source-copy outputs with distinct valid roles seal in either declared order."""
    original_file = REPO_ROOT / "validation/reports/gk_species_reference.json"
    fresh_file = REPO_ROOT / "validation/reports/kinetic_efit_claims.json"
    original_tree = REPO_ROOT / "validation/reference_data/gk_species"
    fresh_tree = REPO_ROOT / "validation/reference_data/gk_geometry"
    output_file = tmp_path / "result.json"
    output_tree = tmp_path / "result-tree"
    shutil.copyfile(original_file, output_file)
    shutil.copytree(original_tree, output_tree)
    prior_digests = {"report": sha256_path(output_file), "report.json": sha256_path(output_tree)}
    declarations = [("report", output_file), ("report.json", output_tree)]
    if directory_first:
        declarations.reverse()
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/run_recorded_benchmark.py"),
        "--repository-root",
        str(tmp_path),
        "--records-root",
        "records",
        "--family",
        "role-storage",
        "--campaign-id",
        "directory-first" if directory_first else "file-first",
    ]
    for role, path in declarations:
        command.extend(["--artifact", f"{role}={path}"])
    command.extend(
        [
            "--",
            sys.executable,
            "-c",
            "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2]); shutil.copytree(sys.argv[3], sys.argv[4])",
            str(fresh_file),
            str(output_file),
            str(fresh_tree),
            str(output_tree),
        ]
    )
    result = subprocess.run(
        command,
        env=child_environment(source_root=REPO_ROOT / "src"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    latest, manifest = load_verified_latest(tmp_path / "records", "role-storage")
    run_root = tmp_path / "records" / latest["manifest_path"]
    run_root = run_root.parent
    artifacts = {entry["role"]: entry for entry in manifest["artifacts"]}
    assert manifest["status"] == "succeeded" and manifest["exit_code"] == 0
    assert set(artifacts) == {"report", "report.json"}
    assert artifacts["report"]["immutable_path_in_run"] != artifacts["report.json"]["immutable_path_in_run"]
    assert sha256_path(run_root / artifacts["report"]["immutable_path_in_run"]) == sha256_path(fresh_file)
    assert sha256_path(run_root / artifacts["report.json"]["immutable_path_in_run"]) == sha256_path(fresh_tree)
    assert sha256_path(output_file) == sha256_path(fresh_file)
    assert sha256_path(output_tree) == sha256_path(fresh_tree)
    for ordinal, (role, _) in enumerate(declarations):
        assert sha256_path(run_root / "prior-output" / str(ordinal)) == prior_digests[role]
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64",
    reason="Linux x86_64 kernel sandbox is required for native sysinfo denial",
)
def test_recording_survives_kernel_refusal_of_optional_load_metadata(tmp_path: Path) -> None:
    """A Linux process sandbox can deny load sampling without losing a result."""
    script = """
import ctypes
import errno
import json
import os
import sys
from pathlib import Path
from scpn_control.benchmark_records import BenchmarkOutput, BenchmarkRun, load_verified_latest
root = Path(sys.argv[1])
class Filter(ctypes.Structure):
    "A native Linux sock_filter instruction."
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]
class Program(ctypes.Structure):
    "A native Linux sock_fprog bound to its retained instruction array."
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]
# x86_64 sysinfo is syscall 99: load number, deny sysinfo with EACCES, allow others.
filters = (Filter * 4)(Filter(0x20, 0, 0, 0), Filter(0x15, 0, 1, 99),
                       Filter(0x06, 0, 0, 0x00050000 | errno.EACCES), Filter(0x06, 0, 0, 0x7fff0000))
program = Program(4, filters)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
    raise OSError(ctypes.get_errno(), "could not install owned process sandbox")
try:
    os.getloadavg()
except OSError:
    pass
else:
    raise AssertionError("kernel did not refuse the actual load query")
output = root / "report.json"
run = BenchmarkRun.begin(repository_root=root, records_root=root / "records",
    family="host-metadata", outputs=[BenchmarkOutput("report", output)],
    command=["producer"], campaign_id="denied-load")
output.write_text("complete recorded output")
manifest = json.loads(run.finish(exit_code=0).read_text())
assert manifest["host_start"]["load_average"] is None
assert manifest["host_end"]["load_average"] is None
assert manifest["status"] == "succeeded"
assert load_verified_latest(root / "records", "host-metadata")[0]["campaign_id"] == "denied-load"
"""
    result = subprocess.run(
        [sys.executable, "-c", CHILD_COVERAGE_PRELUDE + script, str(tmp_path)],
        env=child_environment(source_root=REPO_ROOT / "src"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _begin(records: Path, output: Path, campaign_id: str) -> BenchmarkRun:
    return BenchmarkRun.begin(
        repository_root=records.parent,
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
    assert first_manifest["legacy_inputs"][0]["digest_algorithm"] == "sha256"
    assert (tmp_path / first_manifest["legacy_inputs"][0]["archived_path"]).read_text(
        encoding="utf-8"
    ) == '{"generation":0}\n'

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
    first = _begin(records, output, "fixed-campaign")

    with pytest.raises(FileExistsError):
        _begin(records, output, "fixed-campaign")
    assert not output.exists()
    first.finish(exit_code=1)
    assert output.read_text(encoding="utf-8") == "original"


def test_failed_run_is_preserved_without_advancing_latest(tmp_path: Path) -> None:
    """Failed output remains inspectable without becoming the selected run."""
    output = tmp_path / "report.json"
    output.write_text("first", encoding="utf-8")
    records = tmp_path / "records"
    first = _begin(records, output, "successful-run")
    output.write_text("first", encoding="utf-8")
    first.finish(exit_code=0)

    failed = _begin(records, output, "failed-run")
    output.write_text("partial", encoding="utf-8")
    failed_manifest = json.loads(failed.finish(exit_code=9).read_text(encoding="utf-8"))
    latest, _ = load_verified_latest(records, "controller-latency")

    assert failed_manifest["status"] == "failed"
    assert failed_manifest["artifacts"][0]["sha256"]
    assert latest["campaign_id"] == "successful-run"
    assert output.read_text(encoding="utf-8") == "first"
    assert (failed.run_directory / "failed-output/0").read_text(encoding="utf-8") == "partial"


def test_latest_loader_rejects_manifest_tampering(tmp_path: Path) -> None:
    """A latest index cannot admit a manifest whose bytes changed."""
    output = tmp_path / "report.json"
    output.write_text("result", encoding="utf-8")
    records = tmp_path / "records"
    run = _begin(records, output, "tamper-target")
    output.write_text("result", encoding="utf-8")
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
        repository_root=tmp_path,
        records_root=records,
        family="jax-parity",
        outputs=[BenchmarkOutput("parity-artifacts", output)],
        command=["python", "validation/benchmark_jax_gk_parity.py"],
        campaign_id="directory-run",
    )
    output.mkdir()
    (output / "gpu.json").write_text("gpu", encoding="utf-8")

    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))
    immutable = tmp_path / manifest["artifacts"][0]["immutable_path"]
    assert manifest["artifacts"][0]["kind"] == "directory"
    assert {path.name for path in immutable.iterdir()} == {"gpu.json"}
    archived = tmp_path / manifest["legacy_inputs"][0]["archived_path"]
    assert manifest["legacy_inputs"][0]["digest_algorithm"] == "sha256-directory-tree.v2"
    assert (archived / "cpu.json").read_text() == "cpu"


def test_invalid_identifiers_and_duplicate_roles_fail_before_execution(tmp_path: Path) -> None:
    """Malformed identifiers and ambiguous output roles are rejected."""
    with pytest.raises(ValueError, match="output role"):
        BenchmarkOutput("bad role", tmp_path / "report.json")
    with pytest.raises(ValueError, match="benchmark family"):
        BenchmarkRun.begin(
            repository_root=tmp_path,
            records_root=tmp_path / "records",
            family="bad family",
            outputs=[],
            command=["producer"],
        )
    with pytest.raises(ValueError, match="roles must be unique"):
        BenchmarkRun.begin(
            repository_root=tmp_path,
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
    output = tmp_path / "artifact-output"
    output.mkdir()
    target = tmp_path / "target.json"
    target.write_text("target", encoding="utf-8")
    (output / "linked.json").symlink_to(target)

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        BenchmarkRun.begin(
            repository_root=tmp_path,
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
    digest = sha256_path(output)
    records = tmp_path / "records"
    legacy = records / "legacy/sha256" / digest / "artifact"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="legacy benchmark kind or digest mismatch"):
        _begin(records, output, "legacy-mismatch")


def test_latest_loader_rejects_schema_escape_and_inadmissible_manifest(tmp_path: Path) -> None:
    """Latest selection fails closed on metadata, path, and run-status drift."""
    output = tmp_path / "report.json"
    output.write_text("result", encoding="utf-8")
    records = tmp_path / "records"
    run = _begin(records, output, "latest-validation")
    output.write_text("result", encoding="utf-8")
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
    latest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
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
    case_name = git_marker or head.replace("/", "-").replace(" ", "-").replace(":", "-") or "case"
    repository = tmp_path / case_name
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
        sha256_path(tmp_path / "missing")


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
        repository_root=tmp_path,
        records_root=directory_records,
        family="directory-race",
        outputs=[BenchmarkOutput("directory", directory_output)],
        command=["producer"],
        campaign_id="directory-race",
    )


def test_noop_run_cannot_promote_preexisting_report(tmp_path: Path) -> None:
    """A zero exit code must not certify bytes left by an earlier invocation."""
    output = tmp_path / "report.json"
    output.write_text('{"result":1}\n', encoding="utf-8")
    records = tmp_path / "records"
    run = _begin(records, output, "noop-stale")
    manifest = json.loads(run.finish(exit_code=0).read_text())
    assert manifest["status"] == "failed"
    assert not (records / "latest/controller-latency.json").exists()
    archived = tmp_path / manifest["legacy_inputs"][0]["archived_path"]
    assert archived.read_text() == '{"result":1}\n'


def test_mixed_new_and_stale_outputs_do_not_advance_latest(tmp_path: Path) -> None:
    """Every declared output needs current-invocation custody, not just one."""
    report = tmp_path / "report.json"
    summary = tmp_path / "summary.md"
    report.write_text("old report", encoding="utf-8")
    summary.write_text("old summary", encoding="utf-8")
    records = tmp_path / "records"
    run = BenchmarkRun.begin(
        repository_root=tmp_path,
        records_root=records,
        family="mixed-output",
        outputs=[BenchmarkOutput("report", report), BenchmarkOutput("summary", summary)],
        command=["documented-test-producer"],
        campaign_id="mixed-stale",
    )
    report.write_text("new report", encoding="utf-8")
    manifest = json.loads(run.finish(exit_code=0).read_text())
    assert manifest["status"] == "failed"
    assert not (records / "latest/mixed-output.json").exists()
    assert report.read_text(encoding="utf-8") == "old report"
    assert summary.read_text(encoding="utf-8") == "old summary"
    assert (run.run_directory / "failed-output/0").read_text(encoding="utf-8") == "new report"


def test_identical_new_report_is_fresh_despite_unchanged_digest(tmp_path: Path) -> None:
    """Deterministic rewrites are valid; byte inequality is not freshness."""
    output = tmp_path / "report.json"
    output.write_text("deterministic result", encoding="utf-8")
    records = tmp_path / "records"
    run = _begin(records, output, "identical-new")
    output.write_text("deterministic result", encoding="utf-8")
    manifest = json.loads(run.finish(exit_code=0).read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["legacy_inputs"][0]["sha256"] == manifest["artifacts"][0]["sha256"]
    assert load_verified_latest(records, "controller-latency")[0]["campaign_id"] == run.campaign_id


@pytest.mark.parametrize("kind", ["file", "directory", "directory-empty-file"])
def test_empty_recreated_output_cannot_replace_prior_result(tmp_path: Path, kind: str) -> None:
    """Empty fresh destinations fail without losing prior or failed-run evidence."""
    output = tmp_path / "result"
    if kind == "file":
        output.write_bytes(b"prior")
    else:
        output.mkdir()
        (output / "report").write_bytes(b"prior")
    run = _begin(tmp_path / "records", output, "empty-result")
    if kind == "file":
        output.write_bytes(b"")
    else:
        output.mkdir()
        if kind == "directory-empty-file":
            (output / "report").write_bytes(b"")
    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["empty_output_roles"] == ["report"]
    assert not (run.records_root / "latest/controller-latency.json").exists()
    assert (output if kind == "file" else output / "report").read_bytes() == b"prior"
    assert (run.run_directory / "failed-output/0").exists()
    assert manifest["artifacts"][0]["size_bytes"] == 0


def test_failed_new_destination_is_retained_outside_live_outputs(tmp_path: Path) -> None:
    """A partial result with no predecessor is quarantined rather than left live."""
    output = tmp_path / "report.json"
    run = _begin(tmp_path / "records", output, "new-partial")
    output.write_bytes(b"partial")
    manifest = json.loads(run.finish(exit_code=7).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and not output.exists()
    assert (run.run_directory / "failed-output/0").read_bytes() == b"partial"
    assert (run.run_directory / manifest["artifacts"][0]["immutable_path_in_run"]).read_bytes() == b"partial"


@pytest.mark.parametrize("encoded", ["false", "true", "0.0", '"0"', "null"])
def test_invalid_untyped_exit_status_cannot_replace_valid_latest(tmp_path: Path, encoded: str) -> None:
    """Statuses decoded by an untyped integrator must not publish inadmissible manifests."""
    output = tmp_path / "report.json"
    records = tmp_path / "records"
    previous = _begin(records, output, "previous")
    output.write_bytes(b"previous")
    previous.finish(exit_code=0)
    run = _begin(records, output, "invalid-status")
    output.write_bytes(b"new")
    with pytest.raises(ValueError, match="exit code must be an integer"):
        run.finish(exit_code=json.loads(encoded))
    assert not (run.run_directory / "manifest.json").exists()
    assert load_verified_latest(records, run.family)[0]["campaign_id"] == "previous"
    run.finish(exit_code=3)
    assert output.read_bytes() == b"previous"


def test_refused_reservation_does_not_consume_campaign_identifier(tmp_path: Path) -> None:
    """An overlapping output refusal can be retried once its actual owner releases."""
    from scpn_control.benchmark_output_lease import BenchmarkOutputLease

    output = tmp_path / "report.json"
    records = tmp_path / "records"
    holder = BenchmarkOutputLease.acquire(tmp_path, [output], tmp_path / "other-run")
    try:
        with pytest.raises(RuntimeError, match="already reserved"):
            _begin(records, output, "retry-same-id")
        assert not (records / "runs/controller-latency/retry-same-id").exists()
    finally:
        holder.release()
    run = _begin(records, output, "retry-same-id")
    output.write_bytes(b"complete")
    assert json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))["status"] == "succeeded"


def test_dependency_locks_and_external_output_paths_are_recorded(tmp_path: Path) -> None:
    """Preserve declared dependency bytes and absolute external output custody."""
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    output = tmp_path / "external.json"
    run = BenchmarkRun.begin(
        repository_root=repository,
        records_root=repository / "records",
        family="dependency-custody",
        outputs=[BenchmarkOutput("report", output)],
        command=["producer"],
        campaign_id="dependency-run",
    )
    output.write_text("result", encoding="utf-8")
    manifest = json.loads(run.finish(exit_code=0).read_text())
    assert manifest["dependency_locks"] == ["uv.lock"]
    assert len(manifest["dependency_lock_sha256"]) == 64
    assert manifest["artifacts"][0]["source_path"] == str(output)


def test_symlink_destination_is_refused_before_displacement(tmp_path: Path) -> None:
    """Reject aliased output ownership without touching the original target."""
    target = tmp_path / "target.json"
    target.write_text("original", encoding="utf-8")
    output = tmp_path / "output.json"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="cannot be symlinks"):
        _begin(tmp_path / "records", output, "symlink-output")
    assert output.is_symlink() and target.read_text() == "original"


def test_custody_directory_cannot_be_declared_as_producer_output(tmp_path: Path) -> None:
    """A producer cannot displace its own immutable records directory."""
    with pytest.raises(ValueError, match="overlap the records root"):
        _begin(tmp_path / "records", tmp_path / "records/output", "custody-overlap")


def test_symlink_created_during_invocation_keeps_reservation_for_recovery(tmp_path: Path) -> None:
    """An unsealable result cannot release ownership or advance latest."""
    output = tmp_path / "output.json"
    run = _begin(tmp_path / "records", output, "unsafe-output")
    target = tmp_path / "target.json"
    target.write_text("unowned data", encoding="utf-8")
    output.symlink_to(target)
    with pytest.raises(ValueError, match="cannot be symlinks"):
        run.finish(exit_code=0)
    assert run.output_lease.marker.exists()
    assert not (tmp_path / "records/latest/controller-latency.json").exists()
    output.unlink()
    assert json.loads(run.finish(exit_code=1).read_text())["status"] == "failed"
    assert not run.output_lease.marker.exists()
    assert target.read_text() == "unowned data"


@pytest.mark.parametrize(
    "failure",
    ["changed-failed-output", "occupied-quarantine", "occupied-rollback", "missing-rollback", "occupied-restore"],
)
def test_recovery_conflicts_retain_every_cohort_and_ownership(tmp_path: Path, failure: str) -> None:
    """Real runtime observers and occupied paths cannot discard recovery bytes."""
    script = """
import json
import sys
from pathlib import Path
from scpn_control.benchmark_records import BenchmarkOutput, BenchmarkRun
root = Path(sys.argv[1])
failure = sys.argv[2]
first, second = root / "first.json", root / "second.json"
first.write_text("first original")
second.write_text("second original")
records = root / "records"
run_directory = records / "runs/recovery/conflict"
observed = []
def observe(event, args):
    "Interrupt a real displacement or modify real bytes at manifest reading."
    if observed:
        return
    if failure == "occupied-rollback" and event == "shutil.move" and args[0] == str(second):
        observed.append(event)
        first.write_text("unowned occupant")
        raise PermissionError("runtime observer refused second displacement")
    if failure == "missing-rollback" and event == "shutil.move" and args[0] == str(second):
        observed.append(event)
        second.unlink()
        raise PermissionError("runtime observer removed second displacement source")
    if failure == "occupied-restore" and event == "shutil.move" and args[0] == str(second) and (run_directory / "manifest.json").exists():
        observed.append(event)
        first.write_text("unowned restore occupant")
    if failure == "changed-failed-output" and event == "open" and args[0] == str(run_directory / "manifest.json") and args[1] == "r":
        observed.append(event)
        first.write_text("changed after sealing")
sys.addaudithook(observe)
try:
    run = BenchmarkRun.begin(repository_root=root, records_root=records,
        family="recovery", outputs=[BenchmarkOutput("first", first), BenchmarkOutput("second", second)],
        command=["producer"], campaign_id="conflict")
    first.write_text("failed producer bytes")
    if failure == "occupied-restore":
        second.write_text("second failed producer bytes")
    if failure == "occupied-quarantine":
        destination = run_directory / "failed-output/0"
        destination.parent.mkdir()
        destination.write_text("unowned quarantine occupant")
    run.finish(exit_code=1)
except RuntimeError as error:
    expected = {"changed-failed-output": "changed before recovery", "occupied-quarantine": "recovery destination is occupied", "occupied-rollback": "recovery destination is occupied", "missing-rollback": "cannot locate original output", "occupied-restore": "recovery destination is occupied"}
    assert expected[failure] in str(error), str(error)
else:
    raise AssertionError("recovery conflict was accepted")
assert (run_directory / "prior-output/0").read_text() == "first original"
assert len(list((root / "artifacts/benchmarks/output-leases").glob("*.json"))) == 1
assert not (records / "latest/recovery.json").exists()
if failure in ("occupied-rollback", "missing-rollback"):
    if failure == "occupied-rollback":
        assert first.read_text() == "unowned occupant"
        assert second.read_text() == "second original"
    else:
        assert not first.exists() and not second.exists()
        assert any(p.read_text() == "second original" for p in (records / "legacy/sha256").rglob("artifact"))
    assert not (run_directory / "manifest.json").exists()
elif failure == "occupied-restore":
    assert first.read_text() == "unowned restore occupant"
    assert not second.exists()
    assert (run_directory / "failed-output/0").read_text() == "failed producer bytes"
    assert (run_directory / "failed-output/1").read_text() == "second failed producer bytes"
    assert (run_directory / "prior-output/1").read_text() == "second original"
else:
    manifest = json.loads((run_directory / "manifest.json").read_text())
    first_artifact = next(entry for entry in manifest["artifacts"] if entry["role"] == "first")
    assert (run_directory / first_artifact["immutable_path_in_run"]).read_text() == "failed producer bytes"
    assert (run_directory / "prior-output/1").read_text() == "second original"
    assert first.read_text() == ("changed after sealing" if failure == "changed-failed-output" else "failed producer bytes")
    if failure == "occupied-quarantine":
        assert destination.read_text() == "unowned quarantine occupant"
"""
    result = subprocess.run(
        [sys.executable, "-c", CHILD_COVERAGE_PRELUDE + script, str(tmp_path), failure],
        env=child_environment(source_root=REPO_ROOT / "src"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@contextmanager
def _deny_output_removal(path: Path) -> Iterator[None]:
    """Keep reads available while the OS refuses source deletion and rename."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        # GENERIC_READ; share read/write but not delete; OPEN_EXISTING.
        handle = create(str(path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not close(handle):
                raise ctypes.WinError(ctypes.get_last_error())
    else:
        mode = path.parent.stat().st_mode
        path.parent.chmod(0o555)
        try:
            yield
        finally:
            path.parent.chmod(mode)


def test_failed_second_displacement_restores_first_output(tmp_path: Path) -> None:
    """An actual filesystem denial must roll back earlier successful moves."""
    first = tmp_path / "first.json"
    second = tmp_path / "protected/second.json"
    second.parent.mkdir()
    first.write_text("first original", encoding="utf-8")
    second.write_text("second original", encoding="utf-8")
    records = tmp_path / "records"
    outputs = [BenchmarkOutput("first", first), BenchmarkOutput("second", second)]
    with _deny_output_removal(second), pytest.raises(PermissionError):
        BenchmarkRun.begin(
            repository_root=tmp_path,
            records_root=records,
            family="rollback",
            outputs=outputs,
            command=["producer"],
            campaign_id="failed-displacement",
        )
    assert first.read_text() == "first original"
    assert second.read_text() == "second original"
    run_directory = records / "runs/rollback/failed-displacement"
    assert not (run_directory / "prior-output/0").exists()
    assert not (run_directory / "manifest.json").exists()
    assert not (records / "latest/rollback.json").exists()
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))
    # The released namespaces must remain usable by a subsequent real run.
    retry = BenchmarkRun.begin(
        repository_root=tmp_path,
        records_root=records,
        family="rollback",
        outputs=outputs,
        command=["producer"],
        campaign_id="retry-displacement",
    )
    first.write_text("new first", encoding="utf-8")
    second.write_text("new second", encoding="utf-8")
    assert json.loads(retry.finish(exit_code=0).read_text())["status"] == "succeeded"


def test_observed_snapshot_creation_detects_changed_source(tmp_path: Path) -> None:
    """A runtime file observer mutating output cannot replace verified latest."""
    script = """
import json
import sys
from pathlib import Path
from scpn_control.benchmark_records import BenchmarkOutput, BenchmarkRun, load_verified_latest
root = Path(sys.argv[1])
output = root / "report.json"
def begin(campaign):
    "Reserve a public campaign for the filesystem-observer test."
    return BenchmarkRun.begin(repository_root=root, records_root=root / "records",
        family="source-stability", outputs=[BenchmarkOutput("report", output)],
        command=["producer"], campaign_id=campaign)
first = begin("stable")
output.write_text("stable original")
first.finish(exit_code=0)
run = begin("unstable")
output.write_text("fresh producer result")
snapshot = run.run_directory / "artifacts/0.json"
observed = []
def observe_creation(event, args):
    "Change real source bytes when immutable snapshot creation is observed."
    if event == "open" and args[0] == str(snapshot) and not observed:
        observed.append(event)
        output.write_text("changed by a filesystem observer")
sys.addaudithook(observe_creation)
try:
    run.finish(exit_code=0)
except RuntimeError as error:
    assert "changed while being sealed" in str(error)
else:
    raise AssertionError("mutated source was sealed successfully")
assert observed == ["open"]
assert snapshot.read_text() == "fresh producer result"
assert output.read_text() == "changed by a filesystem observer"
assert run.output_lease.marker.exists()
assert not (run.run_directory / "manifest.json").exists()
assert load_verified_latest(root / "records", "source-stability")[0]["campaign_id"] == "stable"
"""
    result = subprocess.run(
        [sys.executable, "-c", CHILD_COVERAGE_PRELUDE + script, str(tmp_path)],
        env=child_environment(source_root=REPO_ROOT / "src"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("recreate", [False, True])
def test_relative_output_stays_bound_when_working_directory_changes(tmp_path: Path, recreate: bool) -> None:
    """Changing cwd cannot redirect a reserved output or its recovery path."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "result.json").write_text("original first", encoding="utf-8")
    (second / "result.json").write_text("unowned second", encoding="utf-8")
    previous = Path.cwd()
    try:
        os.chdir(first)
        run = BenchmarkRun.begin(
            repository_root=Path("."),
            records_root=Path("records"),
            family="cwd-bound",
            outputs=[BenchmarkOutput("report", Path("result.json"))],
            command=["producer"],
            campaign_id="cwd-change",
        )
        if recreate:
            (first / "result.json").write_text("fresh first", encoding="utf-8")
        os.chdir(second)
        manifest = json.loads(run.finish(exit_code=0).read_text())
    finally:
        os.chdir(previous)
    assert manifest["status"] == ("succeeded" if recreate else "failed")
    assert (second / "result.json").read_text() == "unowned second"
    assert (first / "result.json").read_text() == ("fresh first" if recreate else "original first")
    if recreate:
        artifact = first / manifest["artifacts"][0]["immutable_path"]
        assert artifact.read_text() == "fresh first"
    else:
        assert not (first / "records/latest/cwd-bound.json").exists()
    assert not run.output_lease.marker.exists()


@pytest.mark.parametrize("directory_first", [False, True])
@pytest.mark.parametrize("case_distinct", [False, True])
@pytest.mark.parametrize("producer_fails", [False, True])
def test_public_producer_preserves_equal_digest_legacy_kinds_and_role_names(
    tmp_path: Path, directory_first: bool, case_distinct: bool, producer_fails: bool
) -> None:
    """Retain actual tree fingerprint bytes and trees through success or recovery."""
    original_tree = REPO_ROOT / "validation/reference_data/gk_geometry"
    fresh_tree = REPO_ROOT / "validation/reference_data/gk_species"
    fresh_file = REPO_ROOT / "validation/reports/kinetic_efit_claims.json"
    metadata = bytearray(b"scpn-control.directory-tree.v2\0")
    for entry in sorted(original_tree.rglob("*"), key=lambda item: item.relative_to(original_tree).as_posix()):
        kind = "file" if entry.is_file() else "directory"
        metadata.extend(kind.encode("ascii") + b"\0")
        metadata.extend(entry.relative_to(original_tree).as_posix().encode("utf-8") + b"\0")
        if entry.is_file():
            metadata.extend(bytes.fromhex(sha256_path(entry)))
        metadata.extend(b"\0")
    output_file, output_tree = tmp_path / "result.bin", tmp_path / "result-tree"
    output_file.write_bytes(metadata)
    shutil.copytree(original_tree, output_tree)
    digest = sha256_path(output_file)
    assert digest == sha256_path(output_tree)
    directory_role = "REPORT" if case_distinct else "report.bin"
    declarations = [("report", output_file), (directory_role, output_tree)]
    if directory_first:
        declarations.reverse()
    campaign = "directory-first" if directory_first else "file-first"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/run_recorded_benchmark.py"),
        "--repository-root",
        str(tmp_path),
        "--records-root",
        "records",
        "--family",
        "legacy-kinds",
        "--campaign-id",
        campaign,
    ]
    for role, path in declarations:
        command.extend(["--artifact", f"{role}={path}"])
    command.extend(
        [
            "--",
            sys.executable,
            "-c",
            "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2]); "
            "shutil.copytree(sys.argv[3],sys.argv[4]); sys.exit(int(sys.argv[5]))",
            str(fresh_file),
            str(output_file),
            str(fresh_tree),
            str(output_tree),
            "1" if producer_fails else "0",
        ]
    )
    result = subprocess.run(
        command, env=child_environment(source_root=REPO_ROOT / "src"), capture_output=True, text=True, timeout=30
    )
    assert result.returncode == (1 if producer_fails else 0), result.stdout + result.stderr
    run_root = tmp_path / "records/runs/legacy-kinds" / campaign
    if producer_fails:
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert not (tmp_path / "records/latest/legacy-kinds.json").exists()
        assert output_file.read_bytes() == metadata
        assert output_tree.is_dir() and sha256_path(output_tree) == digest
    else:
        _, manifest = load_verified_latest(tmp_path / "records", "legacy-kinds")
        assert manifest["status"] == "succeeded"
        assert output_file.read_bytes() == fresh_file.read_bytes()
        assert sha256_path(output_tree) == sha256_path(fresh_tree)
    legacy = {entry["role"]: entry for entry in manifest["legacy_inputs"]}
    assert legacy["report"]["archived_path"] != legacy[directory_role]["archived_path"]
    for role, expected_kind in [("report", "file"), (directory_role, "directory")]:
        entry = legacy[role]
        archive = tmp_path / entry["archived_path"]
        assert archive.is_file() if expected_kind == "file" else archive.is_dir()
        assert sha256_path(archive) == entry["sha256"] == digest
        assert entry["digest_algorithm"] == ("sha256" if expected_kind == "file" else "sha256-directory-tree.v2")
    invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
    retained = {entry["role"]: Path(entry["prior_output"]) for entry in invocation["outputs"]}
    assert retained["report"] != retained[directory_role]
    if not producer_fails:
        assert retained["report"].is_file() and retained["report"].read_bytes() == metadata
        assert retained[directory_role].is_dir() and sha256_path(retained[directory_role]) == digest
    else:
        for ordinal, (role, _) in enumerate(declarations):
            failed_output = run_root / "failed-output" / str(ordinal)
            assert failed_output.is_file() if role == "report" else failed_output.is_dir()
            assert sha256_path(failed_output) == sha256_path(fresh_file if role == "report" else fresh_tree)
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))


@pytest.mark.parametrize("directory_output", [False, True])
@pytest.mark.parametrize("matching_kind", [False, True])
def test_archive_reuse_checks_equal_digest_filesystem_kind(
    tmp_path: Path, directory_output: bool, matching_kind: bool
) -> None:
    """Reuse actual equal-digest inputs only with their declared filesystem kind."""
    tree = REPO_ROOT / "validation/reference_data/gk_geometry"
    metadata = bytearray(b"scpn-control.directory-tree.v2\0")
    for entry in sorted(tree.rglob("*"), key=lambda item: item.relative_to(tree).as_posix()):
        metadata.extend(("file" if entry.is_file() else "directory").encode("ascii") + b"\0")
        metadata.extend(entry.relative_to(tree).as_posix().encode("utf-8") + b"\0")
        if entry.is_file():
            metadata.extend(bytes.fromhex(sha256_path(entry)))
        metadata.extend(b"\0")
    output = tmp_path / "output"
    if directory_output:
        shutil.copytree(tree, output)
    else:
        output.write_bytes(metadata)
    digest = sha256_path(output)
    algorithm = "sha256-directory-tree.v2" if directory_output else "sha256"
    archive = tmp_path / "records/legacy" / algorithm / digest / "artifact"
    archive.parent.mkdir(parents=True)
    if directory_output == matching_kind:
        shutil.copytree(tree, archive)
    else:
        archive.write_bytes(metadata)
    assert sha256_path(archive) == digest
    if matching_kind:
        run = BenchmarkRun.begin(
            repository_root=tmp_path,
            records_root=tmp_path / "records",
            family="reuse-kind",
            outputs=[BenchmarkOutput("report", output)],
            command=["producer"],
            campaign_id="reuse-kind",
        )
        assert not output.exists()
        assert json.loads(run.finish(exit_code=1).read_text(encoding="utf-8"))["status"] == "failed"
    else:
        with pytest.raises(RuntimeError, match="legacy benchmark kind or digest mismatch"):
            BenchmarkRun.begin(
                repository_root=tmp_path,
                records_root=tmp_path / "records",
                family="wrong-kind",
                outputs=[BenchmarkOutput("report", output)],
                command=["producer"],
                campaign_id="wrong-kind",
            )
    assert output.is_dir() == directory_output
    assert archive.is_dir() == (directory_output == matching_kind)
    assert sha256_path(output) == sha256_path(archive) == digest
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))


def test_archive_reuse_refuses_a_digest_identical_symlink(tmp_path: Path) -> None:
    """A real symlink cannot stand in for a preserved regular-file archive."""
    output = tmp_path / "result.json"
    source = REPO_ROOT / "validation/reports/kinetic_efit_claims.json"
    shutil.copyfile(source, output)
    digest = sha256_path(output)
    archive = tmp_path / "records/legacy/sha256" / digest / "artifact"
    archive.parent.mkdir(parents=True)
    archive.symlink_to(output)
    with pytest.raises(RuntimeError, match="legacy benchmark kind or digest mismatch"):
        BenchmarkRun.begin(
            repository_root=tmp_path,
            records_root=tmp_path / "records",
            family="symlink-archive",
            outputs=[BenchmarkOutput("report", output)],
            command=["producer"],
            campaign_id="symlink-archive",
        )
    assert archive.is_symlink() and archive.resolve() == output.resolve()
    assert output.read_bytes() == source.read_bytes()
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))
