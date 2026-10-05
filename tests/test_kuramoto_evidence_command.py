# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto evidence real producer commands.

"""Observe bounded Python and actual native parity through the producer entry."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from campaign_command_observation import ROOT, campaign_command, observe_command

from scpn_control.phase.kuramoto import (
    KURAMOTO_RUNTIME_EVIDENCE_BOUNDED,
    KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED,
    assert_kuramoto_runtime_claim_admissible,
    kuramoto_runtime_evidence,
    load_kuramoto_runtime_evidence,
)

MODULE = "validation.benchmark_kuramoto_runtime_evidence"


@pytest.mark.parametrize("entry", ["script", "imported-main"])
@pytest.mark.parametrize("mode", ["external", "mean_field"])
def test_actual_seeded_report_matches_public_reference(tmp_path: Path, entry: str, mode: str) -> None:
    """CLI flags bind the same deterministic vectors/step as the public core.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary report and actual command custody.
    entry, mode : str
        Script/imported entry and actual driver-resolution modes.
    """
    output = tmp_path / "report.json"
    args = [
        "--output-json",
        str(output),
        "--oscillators",
        "8",
        "--deployment-target-oscillators",
        "8",
        "--seed",
        "17",
        "--K",
        "1.7",
        "--alpha",
        "0.37",
        "--zeta",
        "0.5",
        "--psi-driver",
        "0.3",
        "--psi-mode",
        mode,
    ]
    result = observe_command(tmp_path, campaign_command(MODULE, entry) + args, include_repository=entry != "script")
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = load_kuramoto_runtime_evidence(output)
    rng = np.random.default_rng(17)
    reference = kuramoto_runtime_evidence(
        rng.uniform(-np.pi, np.pi, 8).astype(np.float64),
        rng.normal(0, 0.3, 8).astype(np.float64),
        dt=1e-3,
        K=1.7,
        alpha=0.37,
        zeta=0.5,
        psi_driver=0.3,
        psi_mode=mode,
        deployment_target_oscillators=8,
    )
    actual = dataclasses.asdict(evidence)
    expected = dataclasses.asdict(reference)
    for field in ["generated_utc", "payload_sha256"]:
        actual.pop(field)
        expected.pop(field)
    assert actual == expected
    assert evidence.claim_status == KURAMOTO_RUNTIME_EVIDENCE_BOUNDED and not evidence.deployment_claim_allowed
    assert evidence.timestep_refinement_passed
    with pytest.raises(ValueError, match="deployment"):
        assert_kuramoto_runtime_claim_admissible(evidence)


@pytest.mark.parametrize("mode", ["external", "mean_field"])
def test_actual_existing_native_claim_and_target_refusal(tmp_path: Path, mode: str) -> None:
    """The real existing Rust image passes parity while an uncovered target refuses.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Actual native report and command observations.
    mode : str
        External and mean-field driver parity are both exercised.
    """
    output = tmp_path / "native.json"
    command = campaign_command(MODULE, "script", native=True)
    args = [
        "--output-json",
        str(output),
        "--oscillators",
        "8",
        "--deployment-target-oscillators",
        "8",
        "--psi-mode",
        mode,
        "--alpha",
        ".37",
        "--psi-driver",
        ".3",
        "--deployment-claim",
    ]
    result = observe_command(tmp_path, command + args)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = load_kuramoto_runtime_evidence(output, require_deployment_claim=True)
    assert evidence.claim_status == KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED
    assert evidence.rust_available and evidence.rust_parity_checked and evidence.parity_passed
    assert assert_kuramoto_runtime_claim_admissible(evidence) == evidence
    untouched = output.read_bytes()
    bad = args.copy()
    bad[bad.index("--deployment-target-oscillators") + 1] = "9"
    refused = observe_command(tmp_path, command + bad)
    assert refused.returncode == 1 and "oscillator" in refused.stderr
    assert output.read_bytes() == untouched


@pytest.mark.parametrize(
    "args,exit_code,diagnostic",
    [
        (["--help"], 0, "--deployment-claim"),
        ([], 2, "--output-json"),
        (["--oscillators", "0"], 1, "oscillators must be positive"),
        (["--dt", "nan"], 1, "dt must be positive and finite"),
    ],
)
def test_actual_parser_and_input_refusals(tmp_path: Path, args: list[str], exit_code: int, diagnostic: str) -> None:
    """Help, parser and producer errors retain their actual exit/no-write semantics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output absent on every refusal.
    args : list[str]
        Real help, syntax or invalid-domain flags.
    exit_code, diagnostic : int and str
        Documented status and observable diagnostic.
    """
    output = tmp_path / "invalid.json"
    tokens = args if args in (["--help"], []) else ["--output-json", str(output)] + args
    result = observe_command(tmp_path, campaign_command(MODULE, "script") + tokens)
    assert result.returncode == exit_code and diagnostic in result.stdout + result.stderr
    assert not output.exists()


def test_actual_persistent_output_refuses_without_custody(tmp_path: Path) -> None:
    """The producer refuses an unrecorded persistent path before creating any file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Actual process observation storage.
    """
    output = ROOT / "artifacts/a13-kuramoto-unrecorded-refusal.json"
    assert not output.exists()
    result = observe_command(tmp_path, campaign_command(MODULE, "script") + ["--output-json", str(output)])
    assert result.returncode == 1 and "requires tools/run_recorded_benchmark.py" in result.stderr
    assert not output.exists()
