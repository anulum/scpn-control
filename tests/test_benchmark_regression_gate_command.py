# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark gate input custody and command errors

"""Exercise real benchmark gate commands on copies of historical declared metrics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import benchmark_regression_gate as owner

ROOT = Path(__file__).resolve().parents[1]


def _stamp(report: dict[str, Any]) -> None:
    """Stamp a copied declared record; this creates no new timing evidence."""
    report.pop("payload_sha256", None)
    report["payload_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Copy a historical baseline/policy and label a same-metrics replay report."""
    baseline = folder / "baseline.json"
    baseline.write_bytes((ROOT / "benchmarks/baselines/capacitor_bank.json").read_bytes())
    declared = json.loads(baseline.read_text())
    report = {
        "schema_version": owner.REPORT_SCHEMA,
        "generated_utc": declared["measured_utc"],
        "evidence_class": "historical-baseline-replay-for-software-test",
        "production_claim_allowed": False,
        "provenance": declared["provenance"],
        "benchmarks": declared["benchmarks"],
    }
    _stamp(report)
    selected = folder / "report.json"
    selected.write_text(json.dumps(report) + "\n", encoding="utf-8")
    policy = folder / "thresholds.toml"
    policy.write_bytes((ROOT / "benchmarks/regression_thresholds.toml").read_bytes())
    return selected, baseline, policy


def _argv(inputs: tuple[Path, Path, Path], output: Path | None = None) -> list[str]:
    """Select actual file-based CLI inputs and an optional verdict destination."""
    report, baseline, policy = inputs
    args = ["--report", str(report), "--baseline", str(baseline), "--thresholds", str(policy)]
    if output is not None:
        args.extend(["--json-out", str(output)])
    return args


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the actual script outside its source tree, with no campaign privilege."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.pop("SCPN_BENCHMARK_CAMPAIGN_ID", None)
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/benchmark_regression_gate.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _alias(source: Path, kind: str) -> Path:
    """Select direct, resolved, symlink, hard-link or parent-link identity."""
    if kind == "direct":
        return source
    if kind == "normalized":
        return source.parent / ".." / source.parent.name / source.name
    if kind == "parent-link":
        parent = source.parent.with_name(source.parent.name + "_linked")
        parent.symlink_to(source.parent, target_is_directory=True)
        return parent / source.name
    target = source.with_name(source.name + "." + kind)
    if kind == "symlink":
        target.symlink_to(source)
    else:
        target.hardlink_to(source)
    return target


@pytest.mark.parametrize("role", [0, 1, 2], ids=["report", "baseline", "policy"])
@pytest.mark.parametrize("kind", ["direct", "normalized", "symlink", "hardlink", "parent-link"])
def test_real_command_preserves_each_selected_input_alias(tmp_path: Path, role: int, kind: str) -> None:
    """Refuse verdict destinations that would overwrite selected evidence or policy."""
    inputs = _inputs(tmp_path)
    before = {p: p.read_bytes() for p in inputs}
    assert _run(_argv(inputs), tmp_path).returncode == 0
    result = _run(_argv(inputs, _alias(inputs[role], kind)), tmp_path)
    assert result.returncode == 1
    assert "cannot write JSON verdict" in result.stderr
    assert "Traceback" not in result.stderr
    assert before == {p: p.read_bytes() for p in inputs}


@pytest.mark.parametrize("role", [0, 1], ids=["report", "baseline"])
@pytest.mark.parametrize("bad", [b'{"schema_version":', b"\xff", b"[]"])
def test_real_command_refuses_unreadable_metric_documents(tmp_path: Path, role: int, bad: bytes) -> None:
    """Map JSON syntax, UTF-8 and object-shape failures to a diagnostic exit."""
    inputs = _inputs(tmp_path)
    inputs[role].write_bytes(bad)
    output = tmp_path / "verdict.json"
    result = _run(_argv(inputs, output) + ["--evidence-only"], tmp_path)
    assert result.returncode == 1
    assert "cannot read JSON metric documents" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
    assert inputs[role].read_bytes() == bad


@pytest.mark.parametrize("failure", ["directory", "parent-file"])
def test_real_command_refuses_output_io_failures(tmp_path: Path, failure: str) -> None:
    """Report actual output-file and parent-directory errors even in evidence-only mode."""
    inputs = _inputs(tmp_path)
    target = tmp_path / "output"
    if failure == "directory":
        target.mkdir()
        output = target
    else:
        target.write_text("preserve", encoding="utf-8")
        output = target / "verdict.json"
    before = {p: p.read_bytes() for p in inputs}
    result = _run(_argv(inputs, output) + ["--evidence-only"], tmp_path)
    assert result.returncode == 1
    assert "cannot write JSON verdict" in result.stderr
    assert "Traceback" not in result.stderr
    assert before == {p: p.read_bytes() for p in inputs}
    if failure == "parent-file":
        assert target.read_text() == "preserve"


def test_public_command_refuses_invalid_output_spelling(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse a NUL path through the Python CLI API before reading or writing files."""
    inputs = _inputs(tmp_path)
    assert owner.main(_argv(inputs) + ["--json-out", "\x00"]) == 1
    assert "output custody error" in capsys.readouterr().err


def test_real_command_refuses_symlink_loop_output(tmp_path: Path) -> None:
    """Handle a real unresolvable output path with the original inputs preserved."""
    inputs = _inputs(tmp_path)
    output = tmp_path / "loop"
    output.symlink_to(output)
    result = _run(_argv(inputs, output), tmp_path)
    assert result.returncode == 1
    assert "output custody error" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.is_symlink()


def test_real_command_keeps_persistent_campaign_gate(tmp_path: Path) -> None:
    """Refuse a source-tree evidence destination without inventing campaign authorization."""
    inputs = _inputs(tmp_path)
    output = ROOT / "artifacts" / ("benchmark-command-refusal-" + tmp_path.name + ".json")
    assert not output.exists()
    result = _run(_argv(inputs, output), tmp_path)
    assert result.returncode == 1
    assert "output custody error" in result.stderr
    assert "requires tools/run_recorded_benchmark.py" in result.stderr
    assert not output.exists()


def test_real_command_writes_unrelated_existing_output(tmp_path: Path) -> None:
    """Replace a permitted destination with the same actual public gate verdict."""
    inputs = _inputs(tmp_path)
    output = tmp_path / "verdict.json"
    output.write_text("previous report", encoding="utf-8")
    before = {p: p.read_bytes() for p in inputs}
    result = _run(_argv(inputs, output), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "benchmark gate passed" in result.stdout
    expected = owner.gate(
        json.loads(inputs[0].read_text()),
        json.loads(inputs[1].read_text()),
        owner.load_thresholds_file(inputs[2]),
        generated_utc="",
    )
    assert output.read_bytes() == (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
    assert before == {p: p.read_bytes() for p in inputs}


def test_real_command_refuses_nonfinite_ratio_serialisation(tmp_path: Path) -> None:
    """Keep a declared extreme-ratio rejection out of nonstandard JSON output."""
    inputs = _inputs(tmp_path)
    report = json.loads(inputs[0].read_text())
    baseline = json.loads(inputs[1].read_text())
    baseline["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] = 1e-308
    report["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] = 1e308
    baseline["baseline_sha256"] = owner.canonical_metrics_digest(baseline["benchmarks"])
    _stamp(report)
    inputs[0].write_text(json.dumps(report), encoding="utf-8")
    inputs[1].write_text(json.dumps(baseline), encoding="utf-8")
    output = tmp_path / "verdict.json"
    result = _run(_argv(inputs, output) + ["--evidence-only"], tmp_path)
    assert result.returncode == 1
    assert "cannot write JSON verdict" in result.stderr
    assert not output.exists()
