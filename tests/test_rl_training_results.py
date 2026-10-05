# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual RL public workflow tests
"""Exercise real JSON candidate-schema/selection boundaries without training claims."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.benchmark_records import CAMPAIGN_ENV
from tools.rl_training_results import (
    check_fresh_output,
    load_benchmark,
    preflight_directory,
    select_best,
    validate_stats,
)

ROOT = Path(__file__).resolve().parents[1]


def _metrics(reward: float, seed: int) -> dict[str, object]:
    """Create explicitly synthetic schema input, without weights or trained provenance."""
    return {
        "seed": seed,
        "ppo": {"mean_reward": reward, "std_reward": 0.0, "mean_length": 2.0, "disruption_rate": 0.0, "n_episodes": 1},
    }


def test_real_json_selection_below_old_sentinel_and_equal_reward(tmp_path: Path) -> None:
    """Real file selection handles finite rewards below-99999 and keeps first ties."""
    for seed, reward in ((42, -120000.0), (123, -110000.0), (456, -110000.0)):
        (tmp_path / f"ppo_tokamak_seed{seed}.metrics.json").write_text(json.dumps(_metrics(reward, seed)))
    assert select_best(tmp_path) == (123, -110000.0)
    process = subprocess.run(
        [sys.executable, str(ROOT / "tools/rl_training_results.py"), "select", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 0 and process.stdout == "123\n" and process.stderr == ""
    for seeds in ((), (42, 42), (True,), (-1,), (2**32,)):
        with pytest.raises(ValueError):
            select_best(tmp_path, seeds)
    for identity in (None, 42.0, True, 123):
        payload = _metrics(0, 42)
        payload["seed"] = identity
        (tmp_path / "ppo_tokamak_seed42.metrics.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="training seed"):
            select_best(tmp_path)


def test_actual_retained_legacy_metrics_do_not_authenticate_seed() -> None:
    """The actual old seed-labelled metrics lack the new explicit seed identity."""
    with pytest.raises(ValueError, match="training seed"):
        select_best(ROOT / "weights")
    stats = load_benchmark(ROOT / "benchmarks/rl_vs_classical.json")
    assert set(stats) == {"PPO", "PID", "MPC"} and all(x["n_episodes"] == 50 for x in stats.values())


def test_actual_metric_shapes_ranges_and_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Public schema and file checks reject malformed inputs without altered outcomes."""
    good = _metrics(-1, 42)["ppo"]
    assert isinstance(good, dict)
    assert validate_stats(good)["mean_reward"] == -1
    malformed: object
    for malformed in (None, [], {}, True):
        with pytest.raises(ValueError):
            validate_stats(malformed)
    changes = (
        ("n_episodes", 0),
        ("n_episodes", True),
        ("mean_reward", True),
        ("mean_reward", float("inf")),
        ("mean_reward", 10**400),
        ("std_reward", -1),
        ("mean_length", 0),
        ("disruption_rate", 1.1),
    )
    for key, value in changes:
        raw = copy.deepcopy(good)
        raw[key] = value
        with pytest.raises(ValueError):
            validate_stats(raw)
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"PPO": good, "PID": good, "MPC": {**good, "n_episodes": 2}}))
    with pytest.raises(ValueError, match="counts must agree"):
        load_benchmark(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="must be an object"):
        load_benchmark(path)
    with pytest.raises(FileExistsError):
        check_fresh_output(path)
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        check_fresh_output(link)
    with pytest.raises(NotADirectoryError):
        check_fresh_output(path / "new.json")
    with pytest.raises(NotADirectoryError):
        preflight_directory(path)
    with pytest.raises(NotADirectoryError):
        preflight_directory(link)
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    with pytest.raises(RuntimeError):
        preflight_directory(tmp_path / "fresh")
    monkeypatch.setenv(CAMPAIGN_ENV, "schema-preflight-input")
    preflight_directory(tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()
