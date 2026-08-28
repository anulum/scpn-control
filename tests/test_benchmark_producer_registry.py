# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark producer registry tests
"""Exercise the complete benchmark producer custody inventory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import tools.check_benchmark_producers as producer_audit
from tools.check_benchmark_producers import (
    REGISTRY,
    _documentation_command_findings,
    _python_calls_recorded_guard,
    audit_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "check_benchmark_producers.py"


def test_live_registry_classifies_every_discovered_producer() -> None:
    """The canonical registry covers every current Python and Rust producer."""
    assert audit_registry() == []


def test_python_guard_audit_requires_a_call_not_only_an_import() -> None:
    """An unused guard import cannot satisfy persistent-output custody."""
    imported_only = "from scpn_control.benchmark_records import require_recorded_campaign\n"
    called = imported_only + "require_recorded_campaign(output, repository_root=root)\n"

    assert not _python_calls_recorded_guard(imported_only)
    assert _python_calls_recorded_guard(called)


def test_public_documentation_audit_rejects_direct_persistent_producer(tmp_path: Path) -> None:
    """Published replay commands cannot bypass the recorded runner."""
    (tmp_path / "README.md").write_text(
        "```bash\npython validation/benchmark_transport.py\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()

    findings = _documentation_command_findings(tmp_path, {"validation/benchmark_transport.py"})

    assert findings == [
        "public benchmark command bypasses recorded runner: README.md:2: validation/benchmark_transport.py"
    ]


def test_cli_reports_the_live_inventory_count() -> None:
    """The real command-line audit admits the canonical repository state."""
    completed = subprocess.run(
        [sys.executable, str(TOOL)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "producers classified" in completed.stdout


def test_imported_cli_reports_live_inventory_count(capsys: pytest.CaptureFixture[str]) -> None:
    """The imported command surface returns a successful live verdict."""
    assert producer_audit.main([]) == 0
    assert "64 producers classified" in capsys.readouterr().out


def test_registry_fails_when_one_real_producer_is_unclassified(tmp_path: Path) -> None:
    """Inventory drift cannot silently omit a newly or accidentally removed path."""
    source = REGISTRY.read_text(encoding="utf-8")
    removed = '  "benchmarks/controller_latency.py",\n'
    assert removed in source
    incomplete = tmp_path / "producer_registry.toml"
    incomplete.write_text(source.replace(removed, ""), encoding="utf-8")

    findings = audit_registry(incomplete)
    assert "unclassified benchmark producer: benchmarks/controller_latency.py" in findings


def test_documentation_audit_tolerates_absent_public_documents(tmp_path: Path) -> None:
    """A minimal repository without public Markdown has no command findings."""
    assert _documentation_command_findings(tmp_path, {"validation/benchmark_transport.py"}) == []


def test_registry_reports_every_custody_class_failure(tmp_path: Path) -> None:
    """Schema, inventory, guard, stream, scratch, and infrastructure drift are explicit."""
    sources = {
        "benchmarks/bench_bad.py": "from guard import require_recorded_campaign\n",
        "benchmarks/bench_unclassified.py": "print('unclassified')\n",
        "scripts/udp_fault_tolerance_benchmark.py": "print('not append only')\n",
        "tools/gk_convergence_benchmark.py": "print('not scratch')\n",
        "tools/promote_benchmark_baseline.py": "print('custody')\n",
        "tools/benchmark_full_stack.py": "print('stdout')\n",
    }
    for relative_path, source in sources.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "python benchmarks/bench_bad.py\n",
        encoding="utf-8",
    )
    registry = tmp_path / "benchmarks" / "producer_registry.toml"
    registry.write_text(
        """
schema_version = "wrong"
recorded_guard = ["benchmarks/bench_bad.py", "benchmarks/bench_missing.py"]
append_stream = ["scripts/udp_fault_tolerance_benchmark.py"]
temporary_scratch = ["tools/gk_convergence_benchmark.py"]
stdout_or_build_product = ["tools/benchmark_full_stack.py"]
custody_infrastructure = ["tools/promote_benchmark_baseline.py"]
""".lstrip(),
        encoding="utf-8",
    )

    findings = audit_registry(registry, tmp_path)

    assert f"schema_version must be {producer_audit.SCHEMA}" in findings
    assert "unclassified benchmark producer: benchmarks/bench_unclassified.py" in findings
    assert "registry path is not a discovered benchmark producer: benchmarks/bench_missing.py" in findings
    assert "recorded producer lacks campaign guard: benchmarks/bench_bad.py" in findings
    assert "append-stream producer does not visibly append: scripts/udp_fault_tolerance_benchmark.py" in findings
    assert "temporary producer has no explicit scratch destination: tools/gk_convergence_benchmark.py" in findings
    assert "custody infrastructure lacks benchmark contract text: tools/promote_benchmark_baseline.py" in findings
    assert any("public benchmark command bypasses recorded runner" in finding for finding in findings)


def test_registry_rejects_invalid_category_arrays_and_duplicates(tmp_path: Path) -> None:
    """Every category is an array and one producer has exactly one owner."""
    producer = tmp_path / "benchmarks" / "bench_duplicate.py"
    producer.parent.mkdir(parents=True)
    producer.write_text("require_recorded_campaign(output, repository_root=root)\n", encoding="utf-8")
    registry = tmp_path / "registry.toml"
    registry.write_text(
        """
schema_version = "scpn-control.benchmark-producer-registry.v1"
recorded_guard = ["benchmarks/bench_duplicate.py"]
append_stream = "invalid"
temporary_scratch = []
stdout_or_build_product = ["benchmarks/bench_duplicate.py"]
custody_infrastructure = []
""".lstrip(),
        encoding="utf-8",
    )

    findings = audit_registry(registry, tmp_path)

    assert "append_stream must be an array of paths" in findings
    assert "benchmarks/bench_duplicate.py appears in both recorded_guard and stdout_or_build_product" in findings


def test_registry_cli_reports_audit_error_and_findings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI converts parser/read failures and findings into nonzero verdicts."""

    def _raise(*_args: object, **_kwargs: object) -> list[str]:
        raise OSError("unreadable")

    monkeypatch.setattr(producer_audit, "audit_registry", _raise)
    assert producer_audit.main([]) == 1
    assert "unreadable" in capsys.readouterr().err

    monkeypatch.setattr(producer_audit, "audit_registry", lambda *_args, **_kwargs: ["finding"])
    assert producer_audit.main([]) == 1
    assert "finding" in capsys.readouterr().err
