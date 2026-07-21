# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Symbolic-lane evidence tests
"""Symbolic-lane evidence tests.

Verify the symbolic-transition lane is at parity with the neuromorphic readout on
OUT-OF-SAMPLE seeds, load-bearing, and NOT overstated as superior.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MOD = ROOT / "validation" / "scpn_symbolic_lane_evidence.py"
_SPEC = importlib.util.spec_from_file_location("scpn_symbolic_lane_evidence", _MOD)
assert _SPEC and _SPEC.loader
evidence = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evidence
_SPEC.loader.exec_module(evidence)


def test_symbolic_lane_is_at_parity_out_of_sample_and_load_bearing() -> None:
    """Parity holds on an out-of-sample seed, PID is beaten, and the lane is load-bearing."""
    # Reduced but real: one out-of-sample seed (the committed report uses 7/123/999).
    report = evidence.evaluate_evidence(tuning_seed=42, unseen_seeds=(7,), step_counts=(240,))
    r = report["ratio_symbolic_over_neuromorphic_unseen"]
    # Parity judged on OUT-OF-SAMPLE seeds only (the tuning seed 42 is reported separately
    # under in_sample), so the parity claim is not contaminated by the tuning point.
    assert report["parity_holds"] is True
    assert 0.85 <= r["min"] <= r["max"] <= 1.15
    assert 0.9 <= r["mean"] <= 1.1
    assert report["tuning_seed"] == 42
    assert 42 not in report["unseen_seeds"]
    assert all(row["seed"] != 42 for row in report["robustness_unseen"])
    # PID measured on the SAME per-seed profile (apples-to-apples), so the beat is real.
    assert report["symbolic_vs_pid"]["symbolic_better_than_pid"] is True
    assert report["symbolic_vs_pid"]["ratio_symbolic_over_pid_mean"] < 1.0
    # Load-bearing: removing the drain saturates a_* and the symbolic controller diverges.
    assert report["load_bearing"]["diverges_without_drain"] is True
    assert report["load_bearing"]["undrained_over_reference"] > 3.0
    assert "NOT a superiority claim" in report["claim"]


def test_report_payload_digest_is_self_consistent() -> None:
    """The report's payload_sha256 matches a recomputation over its own payload."""
    report = evidence.generate_report(unseen_seeds=(7,), step_counts=(240,))
    stored = report["payload_sha256"]
    assert isinstance(stored, str) and len(stored) == 64
    payload = dict(report)
    payload["payload_sha256"] = None
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == stored


def test_symbolic_reads_transition_outputs_but_neuromorphic_does_not() -> None:
    """The symbolic readout reads transition-output places; the neuromorphic one does not."""
    # The structural distinction the claim rests on: the symbolic controller's readout
    # reads transition-output places (load-bearing); the neuromorphic baseline does not.
    assert evidence._build_variant("symbolic").readout_reads_transition_outputs is True
    assert evidence._build_variant("neuromorphic").readout_reads_transition_outputs is False


def test_symbolic_transitions_fire_at_runtime_but_neuromorphic_is_bypassed() -> None:
    """A DIRECT runtime firing trace (not just structural overlap): symbolic fires, neuromorphic does not."""
    # The undrained-divergence + structural-overlap evidence is complemented by a direct
    # trace: the symbolic transitions actually fire during the run, while the neuromorphic
    # baseline's transitions (delay 10000) fire nothing over a short run.
    assert evidence._transition_firing_activity("symbolic", 7, 120) > 0.0
    assert evidence._transition_firing_activity("neuromorphic", 7, 120) == 0.0


def test_report_records_environment_provenance() -> None:
    """The report records the resolved backend + dependency identity the RMSE values bind to."""
    report = evidence.evaluate_evidence(unseen_seeds=(7,), step_counts=(240,))
    env = report["environment"]
    assert env["runtime_backend"] in {"rust", "numpy"}
    assert isinstance(env["sc_neurocore_present"], bool)
    assert env["python_implementation"] == "CPython"
    assert env["python_version"] and env["numpy_version"]
    # The firing trace is recorded alongside the divergence evidence.
    lb = report["load_bearing"]
    assert lb["transitions_fire_at_runtime"] is True
    assert lb["symbolic_transition_firing_activity"] > 0.0
    assert lb["neuromorphic_transition_firing_activity"] == 0.0


def test_regeneration_reproduces_the_robust_claims() -> None:
    """A fresh run reproduces the parity / better-than-PID / load-bearing CONCLUSIONS.

    The exact RMSE digest is environment-specific (recorded in the environment block); the
    CONCLUSIONS are robust across the numpy-float and sc_neurocore paths, so the hermetic
    check binds to the claims, not to a brittle cross-environment digest.
    """
    report = evidence.evaluate_evidence(tuning_seed=42, unseen_seeds=(7,), step_counts=(240,))
    assert report["parity_holds"] is True
    assert report["symbolic_vs_pid"]["symbolic_better_than_pid"] is True
    assert report["load_bearing"]["diverges_without_drain"] is True
    assert report["load_bearing"]["transitions_fire_at_runtime"] is True


def test_evaluate_evidence_rejects_empty_unseen_seeds() -> None:
    """An empty unseen cohort would make ``parity_holds`` vacuously True — reject it."""
    with pytest.raises(ValueError, match="unseen_seeds must be non-empty"):
        evidence.evaluate_evidence(unseen_seeds=())


def test_evaluate_evidence_rejects_tuning_seed_in_unseen() -> None:
    """The tuning seed must be disjoint from the unseen cohort (no out-of-sample contamination)."""
    with pytest.raises(ValueError, match="must be disjoint from unseen_seeds"):
        evidence.evaluate_evidence(tuning_seed=42, unseen_seeds=(42, 7))


def test_evaluate_evidence_rejects_empty_step_counts() -> None:
    """A run needs at least one step count."""
    with pytest.raises(ValueError, match="step_counts must be non-empty"):
        evidence.evaluate_evidence(unseen_seeds=(7,), step_counts=())


def test_build_variant_rejects_unknown_variant() -> None:
    """An unknown variant name must fail loudly, not silently build the undrained lane."""
    with pytest.raises(ValueError, match="unknown variant"):
        evidence._build_variant("bogus")


def test_main_regenerates_a_self_consistent_report(tmp_path: Path) -> None:
    """``main`` writes a digest-self-consistent report whose robust claims hold (exact-regen check)."""
    out = tmp_path / "evidence.json"
    assert evidence.main(["--output-json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["parity_holds"] is True
    assert data["load_bearing"]["transitions_fire_at_runtime"] is True
    assert data["environment"]["runtime_backend"] in {"rust", "numpy"}
    stored = data["payload_sha256"]
    payload = dict(data)
    payload["payload_sha256"] = None
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == stored


def test_committed_report_exists_and_is_self_consistent() -> None:
    """The committed evidence report is present, honest, and digest-self-consistent."""
    path = ROOT / "validation" / "reports" / "scpn_symbolic_lane_evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "scpn-control.symbolic-lane-evidence.v1"
    assert data["parity_holds"] is True
    assert data["load_bearing"]["diverges_without_drain"] is True
    assert data["load_bearing"]["transitions_fire_at_runtime"] is True
    assert 42 not in data["unseen_seeds"]
    assert data["symbolic_vs_pid"]["symbolic_better_than_pid"] is True
    # The committed artifact is self-describing about the environment its RMSE values bind to.
    assert data["environment"]["runtime_backend"] in {"rust", "numpy"}
    assert data["environment"]["numpy_version"]
    stored = data["payload_sha256"]
    payload = dict(data)
    payload["payload_sha256"] = None
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == stored
