# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — libFuzzer campaign orchestrator and triage-gate tests
"""Exercise fuzz campaign records, filesystem evidence and CLI admission."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

import tools.run_fuzz_campaign as rfc
from tools.run_fuzz_campaign import (
    ARTEFACT_PREFIXES,
    FUZZ_TARGETS,
    SCHEMA_VERSION,
    TargetRun,
    _markdown,
    _run,
    _seed_corpus,
    assemble_report,
    build_all,
    collect_crash_artifacts,
    main,
    parse_libfuzzer_stats,
    run_target,
    seed_corpus_manifest,
    sha256_file,
    toolchain_metadata,
    triage,
)

LIBFUZZER_STDOUT = """\
#2097152 pulse  cov: 412 ft: 1337 corp: 88/9000b
###### End of recommended dictionary. ######
Done 7271771 runs in 41 second(s)
stat::number_of_executed_units: 7271771
stat::average_exec_per_sec:     177360
stat::new_units_added:          12832
stat::slowest_unit_time_sec:    0
stat::peak_rss_mb:              646
"""


def _clean_run(name: str = "config_json") -> TargetRun:
    """Construct a submitted positive run record for evidence-validation tests."""
    return TargetRun(
        name=name,
        surface=FUZZ_TARGETS[name],
        duration_s=41.0,
        exit_code=0,
        executed_units=7_271_771,
        average_exec_per_sec=177_360,
        peak_rss_mb=646,
        artefacts=[],
    )


# ── seed hashing ──────────────────────────────────────────────────────


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """Hash actual seed bytes and compare with an independent hashlib digest."""
    blob = b"vmec_like_v1\nnfp=1\n"
    seed = tmp_path / "seed.txt"
    seed.write_bytes(blob)
    assert sha256_file(seed) == hashlib.sha256(blob).hexdigest()


def test_seed_manifest_hashes_every_seed_and_is_order_independent(tmp_path: Path) -> None:
    """Bind both real seed files to sorted names, bytes and aggregate digest."""
    target_dir = tmp_path / "config_json"
    target_dir.mkdir()
    (target_dir / "b.json").write_bytes(b"{}")
    (target_dir / "a.json").write_bytes(b"[]")
    manifest = seed_corpus_manifest(tmp_path, ["config_json"])
    entry = manifest["config_json"]
    assert entry["seed_count"] == 2
    assert entry["total_bytes"] == 4
    assert set(entry["files"]) == {"a.json", "b.json"}
    # The aggregate is the digest of sorted name:hash lines, so it is stable
    # regardless of directory iteration order.
    expected_src = "\n".join(
        f"{name}:{hashlib.sha256(data).hexdigest()}"
        for name, data in sorted({"a.json": b"[]", "b.json": b"{}"}.items())
    )
    assert entry["aggregate_sha256"] == hashlib.sha256(expected_src.encode()).hexdigest()


def test_seed_manifest_mutation_changes_aggregate(tmp_path: Path) -> None:
    """Change an actual seed file and observe a changed manifest digest."""
    target_dir = tmp_path / "bout_stability"
    target_dir.mkdir()
    seed = target_dir / "nominal.txt"
    seed.write_bytes(b"n=1\n")
    before = seed_corpus_manifest(tmp_path, ["bout_stability"])["bout_stability"]["aggregate_sha256"]
    seed.write_bytes(b"n=2\n")
    after = seed_corpus_manifest(tmp_path, ["bout_stability"])["bout_stability"]["aggregate_sha256"]
    assert before != after


def test_seed_manifest_handles_missing_target_directory(tmp_path: Path) -> None:
    """Represent an absent corpus with zero files and the empty-input digest."""
    manifest = seed_corpus_manifest(tmp_path, ["kuramoto_kernel"])
    entry = manifest["kuramoto_kernel"]
    assert entry["seed_count"] == 0
    assert entry["files"] == {}
    # Empty corpus still yields a deterministic aggregate digest.
    assert entry["aggregate_sha256"] == hashlib.sha256(b"").hexdigest()


# ── libFuzzer stats parsing ───────────────────────────────────────────


def test_parse_libfuzzer_stats_extracts_summary() -> None:
    """Parse all supported counters from the submitted native summary text."""
    stats = parse_libfuzzer_stats(LIBFUZZER_STDOUT)
    assert stats["executed_units"] == 7_271_771
    assert stats["average_exec_per_sec"] == 177_360
    assert stats["peak_rss_mb"] == 646


def test_parse_libfuzzer_stats_defaults_to_zero_without_summary() -> None:
    """Represent absent native counters as zero for later triage refusal."""
    stats = parse_libfuzzer_stats("no stats here\nsegfault maybe\n")
    assert stats == {"executed_units": 0, "average_exec_per_sec": 0, "peak_rss_mb": 0}


def test_parse_libfuzzer_stats_tolerates_malformed_numbers() -> None:
    """Represent a malformed native RSS token as zero without a parser exception."""
    stats = parse_libfuzzer_stats("stat::peak_rss_mb: not_a_number\n")
    assert stats["peak_rss_mb"] == 0


# ── crash-artefact collection ─────────────────────────────────────────


@pytest.mark.parametrize("prefix", ARTEFACT_PREFIXES)
def test_collect_crash_artifacts_flags_every_reproducer_prefix(tmp_path: Path, prefix: str) -> None:
    """Find each supported reproducer prefix in an actual target directory."""
    target_dir = tmp_path / "capacitor_bank"
    target_dir.mkdir()
    (target_dir / f"{prefix}deadbeef").write_bytes(b"\x00")
    found = collect_crash_artifacts(tmp_path, "capacitor_bank")
    assert found == [f"{prefix}deadbeef"]


def test_collect_crash_artifacts_ignores_non_reproducer_files(tmp_path: Path) -> None:
    """Exclude ordinary corpus and note files from the reproducer list."""
    target_dir = tmp_path / "capacitor_bank"
    target_dir.mkdir()
    (target_dir / "README").write_bytes(b"notes")
    (target_dir / "corpus-input").write_bytes(b"\x01")
    assert collect_crash_artifacts(tmp_path, "capacitor_bank") == []


def test_collect_crash_artifacts_missing_dir_is_empty(tmp_path: Path) -> None:
    """Return no reproducer names for an absent target directory."""
    assert collect_crash_artifacts(tmp_path, "vmec_import") == []


# ── TargetRun semantics ───────────────────────────────────────────────


def test_target_run_clean_is_not_crashed() -> None:
    """Distinguish a zero native status with no artefacts from a process failure."""
    assert _clean_run().crashed is False


def test_target_run_nonzero_exit_is_crashed() -> None:
    """Classify a submitted nonzero native status as a process failure."""
    run = TargetRun("vmec_import", FUZZ_TARGETS["vmec_import"], 1.0, 77, 10, 5, 12)
    assert run.crashed is True


def test_target_run_artefacts_imply_crashed_even_on_zero_exit() -> None:
    """Classify recorded reproducers as failure even with a zero process status."""
    run = TargetRun(
        "capacitor_bank",
        FUZZ_TARGETS["capacitor_bank"],
        1.0,
        0,
        10,
        5,
        12,
        artefacts=["crash-abcd"],
    )
    assert run.crashed is True


# ── fail-closed triage ────────────────────────────────────────────────


def test_triage_admits_complete_clean_campaign() -> None:
    """Validate one positive submitted record for each registered target."""
    runs = [_clean_run(name) for name in FUZZ_TARGETS]
    passed, failures = triage(runs, list(FUZZ_TARGETS))
    assert passed is True
    assert failures == []


def test_triage_fails_closed_on_missing_target_evidence() -> None:
    """Refuse a requested target whose submitted campaign has no corresponding run."""
    runs = [_clean_run("config_json")]
    passed, failures = triage(runs, ["config_json", "vmec_import"])
    assert passed is False
    assert any("vmec_import" in f and "missing-evidence" in f for f in failures)


def test_triage_fails_on_nonzero_exit() -> None:
    """Retain the native failure status in the campaign triage reasons."""
    crashed = TargetRun("bout_stability", FUZZ_TARGETS["bout_stability"], 2.0, 1, 9, 4, 11)
    passed, failures = triage([crashed], ["bout_stability"])
    assert passed is False
    assert any("non-zero libFuzzer exit code 1" in f for f in failures)


def test_triage_fails_on_reproducer_artefacts() -> None:
    """Retain every observed reproducer filename in the triage reasons."""
    crashed = TargetRun(
        "kuramoto_kernel",
        FUZZ_TARGETS["kuramoto_kernel"],
        2.0,
        0,
        9,
        4,
        11,
        artefacts=["crash-1234", "timeout-5678"],
    )
    passed, failures = triage([crashed], ["kuramoto_kernel"])
    assert passed is False
    assert any("crash-1234" in f and "timeout-5678" in f for f in failures)


# ── report assembly ───────────────────────────────────────────────────


def _assemble(runs: list[TargetRun], requested: list[str]) -> dict[str, Any]:
    """Submit target records and fixture metadata to the public report builder."""
    return assemble_report(
        runs=runs,
        requested=requested,
        seeds={"config_json": {"seed_count": 1, "aggregate_sha256": "0" * 64}},
        toolchain={"rustc": "rustc 1.98.0-nightly", "cargo_fuzz": "cargo-fuzz 0.13.1"},
        target_triple="x86_64-unknown-linux-gnu",
        sanitizer="AddressSanitizer (cargo-fuzz default)",
        max_total_time_s=300,
        evidence_class="nightly_regression",
        generated_utc="2026-06-15T20:00:00Z",
    )


def test_assemble_report_carries_provenance_and_clean_verdict() -> None:
    """Carry supplied provenance and the triage result without production admission."""
    report = _assemble([_clean_run("config_json")], ["config_json"])
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["production_claim_allowed"] is False
    assert report["toolchain"]["cargo_fuzz"] == "cargo-fuzz 0.13.1"
    assert report["target_triple"] == "x86_64-unknown-linux-gnu"
    assert report["sanitizer"].startswith("AddressSanitizer")
    assert report["triage"]["passed"] is True
    assert report["targets"][0]["name"] == "config_json"


def test_assemble_report_embeds_triage_failures() -> None:
    """Include missing-target evidence failures in the submitted campaign report."""
    report = _assemble([_clean_run("config_json")], ["config_json", "vmec_import"])
    assert report["triage"]["passed"] is False
    assert report["triage"]["failures"]


def test_assemble_report_payload_digest_is_deterministic_and_binding() -> None:
    """Recompute the compact sorted JSON digest independently from the report."""
    report = _assemble([_clean_run("config_json")], ["config_json"])
    digest = report.pop("payload_sha256")
    recomputed = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == recomputed


def test_assemble_report_digest_changes_when_a_run_changes() -> None:
    """Bind the submitted native outcome to a changed report digest."""
    clean = _assemble([_clean_run("config_json")], ["config_json"])["payload_sha256"]
    crashed = TargetRun("config_json", FUZZ_TARGETS["config_json"], 1.0, 1, 9, 4, 11)
    dirty = _assemble([crashed], ["config_json"])["payload_sha256"]
    assert clean != dirty


def test_fuzz_targets_cover_the_documented_surface_classes() -> None:
    # Every required untrusted/high-volume surface class has a target.
    """Retain the declared parser, numeric, vector and FFI target descriptions."""
    surfaces = " ".join(FUZZ_TARGETS.values())
    assert "parser" in surfaces
    assert "numeric adapter" in surfaces
    assert "vector kernel" in surfaces
    assert "FFI" in surfaces


# ── subprocess + orchestration coverage ───────────────────────────────


def _completed(cmd: list[str], *, stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    """Construct the retained subprocess fixture result with an explicit status."""
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)


def test_run_passes_command_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the retained subprocess fixture receives the original argument vector."""
    captured: dict[str, object] = {}

    def fake(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Capture the retained test command and return its explicit subprocess fixture."""
        captured["cmd"] = cmd
        return _completed(cmd, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake)
    result = _run(["echo", "hi"])
    assert result.stdout == "ok"
    assert captured["cmd"] == ["echo", "hi"]


def test_toolchain_metadata_records_versions_and_triple(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the retained version-command fixtures populate their report fields."""

    def fake_run(cmd: list[str], cwd: object = None) -> subprocess.CompletedProcess[str]:
        """Return the retained command-specific version-observation fixture."""
        text = " ".join(cmd)
        if "-vV" in text:
            return _completed(cmd, stdout="rustc 1.98.0\nhost: x86_64-unknown-linux-gnu\n")
        if "fuzz" in text:
            return _completed(cmd, stdout="cargo-fuzz 0.13.1\n")
        return _completed(cmd, stdout="rustc 1.98.0-nightly\n")

    monkeypatch.setattr(rfc, "_run", fake_run)
    toolchain, triple = toolchain_metadata()
    assert toolchain["cargo_fuzz"] == "cargo-fuzz 0.13.1"
    assert toolchain["rustc"] == "rustc 1.98.0-nightly"
    assert triple == "x86_64-unknown-linux-gnu"


def test_toolchain_metadata_defaults_triple_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse missing host metadata in the retained command-observation fixture."""
    monkeypatch.setattr(rfc, "_run", lambda cmd, cwd=None: _completed(cmd, stdout="no host line"))
    with pytest.raises(RuntimeError, match="host metadata is missing"):
        toolchain_metadata()


def test_seed_corpus_copies_tracked_seeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Copy actual seed bytes and preserve the existing destination on a second call."""
    seeds = tmp_path / "seeds"
    corpus = tmp_path / "corpus"
    (seeds / "config_json").mkdir(parents=True)
    (seeds / "config_json" / "a.json").write_bytes(b"{}")
    monkeypatch.setattr(rfc, "SEEDS_ROOT", seeds)
    monkeypatch.setattr(rfc, "CORPUS_ROOT", corpus)
    _seed_corpus("config_json")
    assert (corpus / "config_json" / "a.json").read_bytes() == b"{}"
    # Re-running must not fail when the seed already exists.
    _seed_corpus("config_json")


def test_run_target_builds_a_clean_target_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Check the retained runner fixture records its supplied native summary."""
    monkeypatch.setattr(rfc, "_seed_corpus", lambda target: None)
    monkeypatch.setattr(rfc, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(
        rfc,
        "_run",
        lambda cmd, cwd=None: _completed(cmd, stdout=LIBFUZZER_STDOUT),
    )
    run = run_target("config_json", 5, 4096)
    assert run.name == "config_json"
    assert run.exit_code == 0
    assert run.executed_units == 7_271_771
    assert run.crashed is False


def test_run_target_flags_crash_from_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Combine the retained native failure fixture with an actual reproducer file."""
    artifacts = tmp_path / "artifacts" / "config_json"
    artifacts.mkdir(parents=True)
    (artifacts / "crash-dead").write_bytes(b"\x00")
    monkeypatch.setattr(rfc, "_seed_corpus", lambda target: None)
    monkeypatch.setattr(rfc, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(rfc, "_run", lambda cmd, cwd=None: _completed(cmd, stdout="", rc=77))
    run = run_target("config_json", 5, 4096)
    assert run.crashed is True
    assert run.artefacts == ["crash-dead"]


def test_build_all_returns_process_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate the retained build subprocess fixture status."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(["x"], rc=0))
    assert build_all() == 0


def test_markdown_renders_table_and_failures() -> None:
    """Render submitted target outcomes and their triage reasons in Markdown."""
    clean = assemble_report(
        runs=[_clean_run("config_json")],
        requested=["config_json"],
        seeds={"config_json": {"seed_count": 1, "aggregate_sha256": "0" * 64}},
        toolchain={"rustc": "rustc 1.98.0", "cargo_fuzz": "cargo-fuzz 0.13.1"},
        target_triple="x86_64-unknown-linux-gnu",
        sanitizer="AddressSanitizer",
        max_total_time_s=5,
        evidence_class="nightly_regression",
        generated_utc="2026-06-16T00:00:00Z",
    )
    body = _markdown(clean)
    assert "libFuzzer Campaign Evidence" in body
    assert "`config_json`" in body

    failing = assemble_report(
        runs=[TargetRun("config_json", FUZZ_TARGETS["config_json"], 1.0, 1, 9, 4, 11)],
        requested=["config_json", "vmec_import"],
        seeds={},
        toolchain={"rustc": "rustc", "cargo_fuzz": "cargo-fuzz"},
        target_triple="x86_64-unknown-linux-gnu",
        sanitizer="AddressSanitizer",
        max_total_time_s=5,
        evidence_class="nightly_regression",
        generated_utc="2026-06-16T00:00:00Z",
    )
    failing_body = _markdown(failing)
    assert "Triage failures" in failing_body


def _patch_main(monkeypatch: pytest.MonkeyPatch, *, run: TargetRun, build_rc: int = 0) -> None:
    """Configure the retained orchestration fixtures for one submitted target."""
    monkeypatch.setattr(rfc, "build_all", lambda: build_rc)
    monkeypatch.setattr(rfc, "toolchain_metadata", lambda: ({"rustc": "rustc", "cargo_fuzz": "cargo-fuzz"}, "triple"))
    monkeypatch.setattr(rfc, "run_target", lambda target, t, rss: run)


def test_main_build_only_skips_fuzzing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the existing build-only orchestration contract under its fixtures."""
    calls: dict[str, bool] = {"ran": False}
    monkeypatch.setattr(rfc, "build_all", lambda: 0)
    monkeypatch.setattr(rfc, "run_target", lambda *a, **k: calls.__setitem__("ran", True))
    assert main(["--build-only"]) == 0
    assert calls["ran"] is False


def test_main_returns_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate the existing native build-failure fixture status."""
    _patch_main(monkeypatch, run=_clean_run(), build_rc=3)
    assert main(["--targets", "config_json"]) == 3


def test_main_passes_and_writes_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Write actual report files from the retained submitted clean-run fixtures."""
    _patch_main(monkeypatch, run=_clean_run("config_json"))
    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    rc = main(["--targets", "config_json", "--json-out", str(json_out), "--markdown-out", str(md_out)])
    assert rc == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["triage"]["passed"] is True
    assert md_out.read_text(encoding="utf-8")


def test_main_prints_report_without_output_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the existing report fixture when no output destinations are supplied."""
    _patch_main(monkeypatch, run=_clean_run("config_json"))
    rc = main(["--targets", "config_json"])
    assert rc == 0
    assert SCHEMA_VERSION in capsys.readouterr().out


def test_main_fails_on_triage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the retained campaign fixture with a nonzero target process status."""
    crashed = TargetRun("config_json", FUZZ_TARGETS["config_json"], 1.0, 1, 9, 4, 11)
    _patch_main(monkeypatch, run=crashed)
    assert main(["--targets", "config_json"]) == 1


def test_main_rejects_unknown_target() -> None:
    """Refuse an unregistered CLI target before building or running a child."""
    with pytest.raises(SystemExit):
        main(["--targets", "no_such_target"])


@pytest.mark.parametrize(
    ("requested", "names", "reason"),
    [
        ([], [], "empty-campaign"),
        (["config_json", "config_json"], ["config_json"], "duplicate requested"),
        (["config_json"], ["config_json", "config_json"], "duplicate target run"),
        (["config_json"], ["config_json", "vmec_import"], "unrequested target"),
        (["unknown"], [], "unknown requested target"),
    ],
)
def test_triage_refuses_invalid_campaign_selection(requested: list[str], names: list[str], reason: str) -> None:
    """Refuse vacuous, duplicate, unknown and extra submitted target evidence."""
    passed, failures = triage([_clean_run(name) for name in names], requested)
    assert passed is False
    assert any(reason in failure for failure in failures)


@pytest.mark.parametrize("count", [0, -1, True])
def test_triage_refuses_missing_or_invalid_executed_units(count: int) -> None:
    """Require a positive nonboolean execution count even for a zero native exit."""
    run = replace(_clean_run(), executed_units=count)
    passed, failures = triage([run], ["config_json"])
    assert passed is False
    assert any("positive executed-unit" in failure for failure in failures)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf"), True, 10**1000, "absent", None])
def test_triage_refuses_unusable_native_duration(duration: object) -> None:
    """Refuse nonpositive, nonfinite, boolean and unrepresentable elapsed seconds."""
    record = asdict(_clean_run())
    record["duration_s"] = duration
    run = TargetRun(**json.loads(json.dumps(record)))
    passed, failures = triage([run], ["config_json"])
    assert passed is False
    assert any("duration must be positive and finite" in failure for failure in failures)


@pytest.mark.parametrize("field", ["average_exec_per_sec", "peak_rss_mb"])
def test_triage_refuses_negative_summary_counters(field: str) -> None:
    """Reject signed negative native rates or RSS without admitting a clean exit."""
    run = (
        replace(_clean_run(), average_exec_per_sec=-1)
        if field == "average_exec_per_sec"
        else replace(_clean_run(), peak_rss_mb=-1)
    )
    passed, failures = triage([run], ["config_json"])
    assert passed is False
    assert any("invalid nonnegative run statistics" in failure for failure in failures)


def test_report_refuses_empty_campaign_and_mismatched_surface() -> None:
    """Keep report verdicts false for no requested work and a wrong surface label."""
    assert _assemble([], [])["triage"]["passed"] is False
    run = replace(_clean_run(), surface=FUZZ_TARGETS["vmec_import"])
    report = _assemble([run], ["config_json"])
    assert report["triage"]["passed"] is False
    assert any("mismatched target surface" in failure for failure in report["triage"]["failures"])
    assert report["production_claim_allowed"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--targets"],
        ["--targets", "config_json", "config_json"],
        ["--max-total-time", "0"],
        ["--max-total-time", "-1"],
        ["--rss-limit-mb", "0"],
        ["--rss-limit-mb", "-1"],
        ["--build-only", "--targets"],
    ],
)
def test_fuzz_cli_refuses_invalid_campaign_before_native_build(argv: list[str], tmp_path: Path) -> None:
    """Execute real CLI argument refusals before any native build is invoked."""
    result = subprocess.run(
        [sys.executable, str(Path(rfc.__file__).resolve()), *argv],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "fuzz build failed" not in result.stderr
    assert "fuzz targets built" not in result.stderr


@pytest.mark.parametrize(
    ("target", "seconds", "rss", "reason"),
    [
        ("unknown", 1, 1024, "unknown fuzz target"),
        ("config_json", 0, 1024, "max_total_time_s"),
        ("config_json", 1, 0, "rss_limit_mb"),
        ("config_json", True, 1024, "max_total_time_s"),
    ],
)
def test_run_target_refuses_invalid_resources_before_corpus_work(
    target: str, seconds: int, rss: int, reason: str
) -> None:
    """Validate the public target runner without creating a corpus or native child."""
    with pytest.raises(ValueError, match=reason):
        run_target(target, seconds, rss)


def test_toolchain_metadata_refuses_unavailable_native_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject an actual missing executable without requiring Rust in Python CI."""
    native_bin = tmp_path / "empty-bin"
    native_bin.mkdir()
    monkeypatch.setenv("PATH", str(native_bin))
    with pytest.raises(OSError):
        toolchain_metadata()


def test_fuzz_cli_refuses_unavailable_native_commands_before_build(tmp_path: Path) -> None:
    """Refuse the actual CLI when its native command directory is empty."""
    native_bin = tmp_path / "empty-bin"
    native_bin.mkdir()
    environment = dict(os.environ, PATH=str(native_bin), RUSTUP_AUTO_INSTALL="0")
    result = subprocess.run(
        [sys.executable, str(Path(rfc.__file__).resolve()), "--targets", "config_json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "fuzz toolchain metadata failed\n"
