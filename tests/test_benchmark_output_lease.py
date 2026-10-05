# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark output reservation tests
"""Exercise disjoint and conflicting output reservations across processes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.benchmark_output_lease import BenchmarkOutputLease


def test_disjoint_campaigns_coexist_and_released_paths_can_be_reused(tmp_path: Path) -> None:
    """Independent destinations remain concurrent while identical paths exclude."""
    first = BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "one"], tmp_path / "run-one")
    second = BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "two"], tmp_path / "run-two")
    with pytest.raises(RuntimeError, match="already reserved"):
        BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "one/child"], tmp_path / "run-three")
    first.release()
    third = BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "one/child"], tmp_path / "run-three")
    second.release()
    third.release()
    assert not list((tmp_path / "artifacts/benchmarks/output-leases").glob("*.json"))


@pytest.mark.parametrize("paths", [[], ["same", "same"], ["parent", "parent/child"], ["artifacts"]])
def test_invalid_output_namespaces_are_refused(tmp_path: Path, paths: list[str]) -> None:
    """Reject empty, overlapping and registry-containing output namespaces."""
    with pytest.raises(ValueError):
        BenchmarkOutputLease.acquire(tmp_path, [tmp_path / name for name in paths], tmp_path / "run")


def test_other_process_cannot_claim_parent_of_reserved_output(tmp_path: Path) -> None:
    """A real child process observes the same reservation registry."""
    lease = BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "reports/item.json"], tmp_path / "run")
    script = """
import sys
from pathlib import Path
from scpn_control.benchmark_output_lease import BenchmarkOutputLease
root = Path(sys.argv[1])
try:
    BenchmarkOutputLease.acquire(root, [root / "reports"], root / "other-records/run")
except RuntimeError as error:
    assert "already reserved" in str(error)
else:
    raise AssertionError("cross-process overlapping output accepted")
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)], env=env, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
    finally:
        lease.release()


@pytest.mark.parametrize("record", ['{"outputs":null}', '{"outputs":[]}', '{"outputs":["relative"]}', "[]"])
def test_unrecognised_reservation_metadata_fails_closed(tmp_path: Path, record: str) -> None:
    """Malformed reservation metadata cannot silently release another owner."""
    root = tmp_path / "artifacts/benchmarks/output-leases"
    root.mkdir(parents=True)
    (root / "interrupted.json").write_text(record)
    with pytest.raises(RuntimeError, match="invalid benchmark reservation"):
        BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "report"], tmp_path / "run")


@pytest.mark.parametrize("occupied", [False, True])
def test_failed_displacement_recovery_retains_cross_process_reservation(tmp_path: Path, occupied: bool) -> None:
    """A real Python audit policy denial cannot reopen unrecovered outputs."""
    script = """
import sys
from pathlib import Path
from scpn_control.benchmark_records import BenchmarkOutput, BenchmarkRun
root = Path(sys.argv[1])
first, second = root / "first.json", root / "second.json"
occupied = sys.argv[2] == "True"
first.write_text("first original")
second.write_text("second original")
def deny_moves(event, args):
    "Apply a real runtime audit policy denying selected file moves."
    if event == "shutil.move":
        source = Path(args[0])
        if source == second:
            if occupied:
                first.write_text("new occupant")
            raise PermissionError("audit policy denies displacement")
        if source.parent.name == "prior-output":
            raise PermissionError("audit policy denies recovery")
sys.addaudithook(deny_moves)
try:
    BenchmarkRun.begin(repository_root=root, records_root=root / "records",
        family="recovery", outputs=[BenchmarkOutput("first", first), BenchmarkOutput("second", second)],
        command=["producer"], campaign_id="denied-recovery")
except (PermissionError, RuntimeError) as error:
    expected = "recovery destination is occupied" if occupied else "audit policy denies recovery"
    assert expected in str(error), str(error)
else:
    raise AssertionError("audit denial was not enforced")
assert first.read_text() == "new occupant" if occupied else not first.exists()
assert second.read_text() == "second original"
assert (root / "records/runs/recovery/denied-recovery/prior-output/first").read_text() == "first original"
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(occupied)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(RuntimeError, match="already reserved"):
        BenchmarkOutputLease.acquire(tmp_path, [tmp_path / "first.json"], tmp_path / "new-run")
    assert not (tmp_path / "records/latest/recovery.json").exists()
