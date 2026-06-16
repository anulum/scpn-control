# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — libFuzzer campaign orchestrator and triage-gate tests

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import subprocess

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
    blob = b"vmec_like_v1\nnfp=1\n"
    seed = tmp_path / "seed.txt"
    seed.write_bytes(blob)
    assert sha256_file(seed) == hashlib.sha256(blob).hexdigest()


def test_seed_manifest_hashes_every_seed_and_is_order_independent(tmp_path: Path) -> None:
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
    target_dir = tmp_path / "bout_stability"
    target_dir.mkdir()
    seed = target_dir / "nominal.txt"
    seed.write_bytes(b"n=1\n")
    before = seed_corpus_manifest(tmp_path, ["bout_stability"])["bout_stability"]["aggregate_sha256"]
    seed.write_bytes(b"n=2\n")
    after = seed_corpus_manifest(tmp_path, ["bout_stability"])["bout_stability"]["aggregate_sha256"]
    assert before != after


def test_seed_manifest_handles_missing_target_directory(tmp_path: Path) -> None:
    manifest = seed_corpus_manifest(tmp_path, ["kuramoto_kernel"])
    entry = manifest["kuramoto_kernel"]
    assert entry["seed_count"] == 0
    assert entry["files"] == {}
    # Empty corpus still yields a deterministic aggregate digest.
    assert entry["aggregate_sha256"] == hashlib.sha256(b"").hexdigest()


# ── libFuzzer stats parsing ───────────────────────────────────────────


def test_parse_libfuzzer_stats_extracts_summary() -> None:
    stats = parse_libfuzzer_stats(LIBFUZZER_STDOUT)
    assert stats["executed_units"] == 7_271_771
    assert stats["average_exec_per_sec"] == 177_360
    assert stats["peak_rss_mb"] == 646


def test_parse_libfuzzer_stats_defaults_to_zero_without_summary() -> None:
    stats = parse_libfuzzer_stats("no stats here\nsegfault maybe\n")
    assert stats == {"executed_units": 0, "average_exec_per_sec": 0, "peak_rss_mb": 0}


def test_parse_libfuzzer_stats_tolerates_malformed_numbers() -> None:
    stats = parse_libfuzzer_stats("stat::peak_rss_mb: not_a_number\n")
    assert stats["peak_rss_mb"] == 0


# ── crash-artefact collection ─────────────────────────────────────────


@pytest.mark.parametrize("prefix", ARTEFACT_PREFIXES)
def test_collect_crash_artifacts_flags_every_reproducer_prefix(tmp_path: Path, prefix: str) -> None:
    target_dir = tmp_path / "capacitor_bank"
    target_dir.mkdir()
    (target_dir / f"{prefix}deadbeef").write_bytes(b"\x00")
    found = collect_crash_artifacts(tmp_path, "capacitor_bank")
    assert found == [f"{prefix}deadbeef"]


def test_collect_crash_artifacts_ignores_non_reproducer_files(tmp_path: Path) -> None:
    target_dir = tmp_path / "capacitor_bank"
    target_dir.mkdir()
    (target_dir / "README").write_bytes(b"notes")
    (target_dir / "corpus-input").write_bytes(b"\x01")
    assert collect_crash_artifacts(tmp_path, "capacitor_bank") == []


def test_collect_crash_artifacts_missing_dir_is_empty(tmp_path: Path) -> None:
    assert collect_crash_artifacts(tmp_path, "vmec_import") == []


# ── TargetRun semantics ───────────────────────────────────────────────


def test_target_run_clean_is_not_crashed() -> None:
    assert _clean_run().crashed is False


def test_target_run_nonzero_exit_is_crashed() -> None:
    run = TargetRun("vmec_import", FUZZ_TARGETS["vmec_import"], 1.0, 77, 10, 5, 12)
    assert run.crashed is True


def test_target_run_artefacts_imply_crashed_even_on_zero_exit() -> None:
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
    runs = [_clean_run(name) for name in FUZZ_TARGETS]
    passed, failures = triage(runs, list(FUZZ_TARGETS))
    assert passed is True
    assert failures == []


def test_triage_fails_closed_on_missing_target_evidence() -> None:
    runs = [_clean_run("config_json")]
    passed, failures = triage(runs, ["config_json", "vmec_import"])
    assert passed is False
    assert any("vmec_import" in f and "missing-evidence" in f for f in failures)


def test_triage_fails_on_nonzero_exit() -> None:
    crashed = TargetRun("bout_stability", FUZZ_TARGETS["bout_stability"], 2.0, 1, 9, 4, 11)
    passed, failures = triage([crashed], ["bout_stability"])
    assert passed is False
    assert any("non-zero libFuzzer exit code 1" in f for f in failures)


def test_triage_fails_on_reproducer_artefacts() -> None:
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


def _assemble(runs, requested) -> dict:
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
    report = _assemble([_clean_run("config_json")], ["config_json"])
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["production_claim_allowed"] is False
    assert report["toolchain"]["cargo_fuzz"] == "cargo-fuzz 0.13.1"
    assert report["target_triple"] == "x86_64-unknown-linux-gnu"
    assert report["sanitizer"].startswith("AddressSanitizer")
    assert report["triage"]["passed"] is True
    assert report["targets"][0]["name"] == "config_json"


def test_assemble_report_embeds_triage_failures() -> None:
    report = _assemble([_clean_run("config_json")], ["config_json", "vmec_import"])
    assert report["triage"]["passed"] is False
    assert report["triage"]["failures"]


def test_assemble_report_payload_digest_is_deterministic_and_binding() -> None:
    report = _assemble([_clean_run("config_json")], ["config_json"])
    digest = report.pop("payload_sha256")
    recomputed = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == recomputed


def test_assemble_report_digest_changes_when_a_run_changes() -> None:
    clean = _assemble([_clean_run("config_json")], ["config_json"])["payload_sha256"]
    crashed = TargetRun("config_json", FUZZ_TARGETS["config_json"], 1.0, 1, 9, 4, 11)
    dirty = _assemble([crashed], ["config_json"])["payload_sha256"]
    assert clean != dirty


def test_fuzz_targets_cover_the_documented_surface_classes() -> None:
    # Every required untrusted/high-volume surface class has a target.
    surfaces = " ".join(FUZZ_TARGETS.values())
    assert "parser" in surfaces
    assert "numeric adapter" in surfaces
    assert "vector kernel" in surfaces
    assert "FFI" in surfaces


# ── subprocess + orchestration coverage ───────────────────────────────


def _completed(cmd: list[str], *, stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)


def test_run_passes_command_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return _completed(cmd, stdout="ok")

    monkeypatch.setattr(rfc.subprocess, "run", fake)
    result = _run(["echo", "hi"])
    assert result.stdout == "ok"
    assert captured["cmd"] == ["echo", "hi"]


def test_toolchain_metadata_records_versions_and_triple(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], cwd: object = None) -> subprocess.CompletedProcess[str]:
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
    monkeypatch.setattr(rfc, "_run", lambda cmd, cwd=None: _completed(cmd, stdout="no host line"))
    _toolchain, triple = toolchain_metadata()
    assert triple == "unknown"


def test_seed_corpus_copies_tracked_seeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(rfc.subprocess, "run", lambda *a, **k: _completed(["x"], rc=0))
    assert build_all() == 0


def test_markdown_renders_table_and_failures() -> None:
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
    monkeypatch.setattr(rfc, "build_all", lambda: build_rc)
    monkeypatch.setattr(rfc, "toolchain_metadata", lambda: ({"rustc": "rustc", "cargo_fuzz": "cargo-fuzz"}, "triple"))
    monkeypatch.setattr(rfc, "run_target", lambda target, t, rss: run)


def test_main_build_only_skips_fuzzing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, bool] = {"ran": False}
    monkeypatch.setattr(rfc, "build_all", lambda: 0)
    monkeypatch.setattr(rfc, "run_target", lambda *a, **k: calls.__setitem__("ran", True))
    assert main(["--build-only"]) == 0
    assert calls["ran"] is False


def test_main_returns_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rfc, "build_all", lambda: 3)
    assert main(["--targets", "config_json"]) == 3


def test_main_passes_and_writes_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    _patch_main(monkeypatch, run=_clean_run("config_json"))
    rc = main(["--targets", "config_json"])
    assert rc == 0
    assert SCHEMA_VERSION in capsys.readouterr().out


def test_main_fails_on_triage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    crashed = TargetRun("config_json", FUZZ_TARGETS["config_json"], 1.0, 1, 9, 4, 11)
    _patch_main(monkeypatch, run=crashed)
    assert main(["--targets", "config_json"]) == 1


def test_main_rejects_unknown_target() -> None:
    with pytest.raises(SystemExit):
        main(["--targets", "no_such_target"])
