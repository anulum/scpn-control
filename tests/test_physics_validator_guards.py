# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Input guards, evidence refusals and CLI outputs of the physics validators.
"""The physics validators refuse bad inputs and bad evidence with ValueError and report what they ran.

The numeric guards are copied into each validator. They must turn every
unusable value into ValueError, including an integer too large for a float,
which ``float()`` reports as OverflowError. The evidence readers refuse
unsupported schemas, malformed or mismatched digests and, for the disruption
sequence, an opened production claim or a failed run. The command-line
entries write the reports they are asked for and print either the sealed
payload or a summary.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from validation import validate_disruption_sequence as disruption
from validation import validate_kuramoto_synchronisation as kuramoto
from validation import validate_marfe_onset as marfe
from validation import validate_vmec_lite_geometry as vmec

VALIDATORS = (
    "validate_burn_control",
    "validate_density_control",
    "validate_elm_peeling_ballooning",
    "validate_grad_shafranov_solovev",
    "validate_guiding_centre_conservation",
    "validate_halo_current",
    "validate_kuramoto_synchronisation",
    "validate_marfe_onset",
    "validate_momentum_transport",
    "validate_ntm_island_dynamics",
    "validate_rwm_feedback",
    "validate_sawtooth_kadomtsev",
    "validate_transport_diffusion",
    "validate_vmec_lite_geometry",
)
TOO_LARGE_FOR_FLOAT = 10**400


def _module(name: str) -> ModuleType:
    """Import a validator as a package module."""
    return importlib.import_module(f"validation.{name}")


@pytest.mark.parametrize("name", VALIDATORS)
def test_finite_guard_refuses_every_unusable_value(name: str) -> None:
    """Overflowing integers, booleans, text and non-finite floats are refused; finite numbers pass as floats."""
    finite = _module(name)._finite_float
    for value, message in (
        (TOO_LARGE_FOR_FLOAT, "x must be finite"),
        (True, "x must be a finite number"),
        ("1.0", "x must be a finite number"),
        (math.inf, "x must be finite"),
        (math.nan, "x must be finite"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            finite("x", value)
    assert finite("x", 3) == 3.0
    assert isinstance(finite("x", 3), float)


@pytest.mark.parametrize("name", VALIDATORS)
def test_sign_and_integer_guards_refuse_out_of_range_values(name: str) -> None:
    """Positive guards refuse zero, non-negative guards refuse negatives, integer guards refuse bools and zero."""
    module = _module(name)
    with pytest.raises(ValueError, match="^x must be positive$"):
        module._positive_float("x", 0.0)
    assert module._positive_float("x", 2) == 2.0
    if hasattr(module, "_nonnegative_float"):
        with pytest.raises(ValueError, match="^x must be non-negative$"):
            module._nonnegative_float("x", -1.0)
        assert module._nonnegative_float("x", 0) == 0.0
    if hasattr(module, "_positive_int"):
        for value in (True, 0, 1.5):
            with pytest.raises(ValueError):
                module._positive_int("n", value)
        assert module._positive_int("n", 64) == 64


def test_disruption_positive_guard_refuses_overflow_and_nonpositive_values() -> None:
    """The disruption validator's own positive guard refuses overflow, booleans, infinity and zero."""
    for value, message in (
        (TOO_LARGE_FOR_FLOAT, "must be finite"),
        (True, "must be finite"),
        (math.inf, "must be finite"),
        (0.0, "must be positive"),
    ):
        with pytest.raises(ValueError, match=f"^x {message}$"):
            disruption._positive_float("x", value)
    assert disruption._positive_float("x", 2) == 2.0


@pytest.fixture(scope="module")
def disruption_result() -> disruption.DisruptionSequenceValidationResult:
    """Run the deterministic disruption-sequence validation once."""
    return disruption.validate_disruption_sequence()


def test_disruption_refuses_a_spi_target_below_the_pre_disruption_density() -> None:
    """The SPI density target must exceed the pre-disruption density before anything is run."""
    with pytest.raises(ValueError, match="must exceed the pre-disruption density"):
        disruption.validate_disruption_sequence(spi_density_target_20=disruption.default_config().ne_pre_20)


def test_disruption_evidence_needs_a_target_id(
    disruption_result: disruption.DisruptionSequenceValidationResult,
) -> None:
    """A blank target identifier is refused."""
    with pytest.raises(ValueError, match="target_id must be a non-empty string"):
        disruption.build_evidence(disruption_result, target_id="  ")


def _resealed(payload: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Return ``payload`` with ``changes`` applied and its disruption digest recomputed."""
    changed = {**payload, **changes}
    changed["payload_sha256"] = disruption._payload_sha256(changed)
    return changed


def test_disruption_evidence_reader_refuses_each_inadmissible_payload(
    disruption_result: disruption.DisruptionSequenceValidationResult,
) -> None:
    """Schema, digest form, digest match, the closed production claim and a pass are all required."""
    payload = disruption.build_evidence(disruption_result, target_id="guard-test")
    assert disruption.validate_evidence_payload(payload) is True
    for bad, message in (
        ({**payload, "schema_version": "other"}, "unsupported disruption-sequence evidence schema_version"),
        ({**payload, "payload_sha256": "abc"}, "payload_sha256 must be a SHA-256 hex digest"),
        ({**payload, "payload_sha256": "0" * 64}, "payload_sha256 does not match canonical payload"),
        (_resealed(payload, production_claim_allowed=True), "production_claim_allowed must remain false"),
        (_resealed(payload, passed=False), "evidence payload must be passing"),
    ):
        with pytest.raises(ValueError, match=message):
            disruption.validate_evidence_payload(bad)


def test_disruption_command_writes_both_reports_and_prints_the_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command writes the sealed JSON and the Markdown summary, then prints the payload or a summary."""
    json_path = tmp_path / "out" / "disruption.json"
    markdown_path = tmp_path / "out" / "disruption.md"
    assert disruption.main(["--output-json", str(json_path), "--output-md", str(markdown_path), "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert printed["payload_sha256"] == written["payload_sha256"]
    assert disruption.validate_evidence_payload(written) is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Disruption Sequence Bounded Validation" in markdown
    assert f"- Payload SHA-256: `{written['payload_sha256']}`" in markdown
    assert written["claim_boundary"] in markdown
    assert disruption.main([]) == 0
    summary = capsys.readouterr().out
    assert "Status: pass" in summary and "production claim allowed: false" in summary


def test_kuramoto_refuses_a_branch_coupling_at_or_below_critical() -> None:
    """Branch couplings must lie above the critical coupling."""
    with pytest.raises(ValueError, match="ratio > 1"):
        kuramoto.validate_synchronisation(
            n_oscillators=64, n_steps=20, n_average=10, branch_couplings_over_critical=(1.0,)
        )


def test_kuramoto_evidence_reader_refuses_schema_and_digest_problems() -> None:
    """An unsupported schema, a malformed digest and a mismatched digest are refused."""
    for bad, message in (
        ({"schema_version": "other"}, "unsupported kuramoto synchronisation evidence schema_version"),
        ({"schema_version": kuramoto.KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION, "payload_sha256": "abc"}, "SHA-256"),
        (
            {"schema_version": kuramoto.KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION, "payload_sha256": "0" * 64},
            "does not match payload",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            kuramoto.validate_evidence_payload(bad)


@pytest.mark.parametrize("json_out", [True, False])
def test_kuramoto_command_reports_a_small_run(json_out: bool, capsys: pytest.CaptureFixture[str]) -> None:
    """A small command-line run prints the sealed payload or a per-branch summary and exits by its verdict."""
    argv = ["--n-oscillators", "256", "--n-steps", "400", "--target-id", "guard-test"]
    code = kuramoto.main([*argv, "--json-out"] if json_out else argv)
    out = capsys.readouterr().out
    if json_out:
        payload = json.loads(out)
        assert payload["target_id"] == "guard-test"
        assert code == (0 if payload["passed"] else 1)
    else:
        assert out.startswith("Kuramoto-Sakaguchi synchronisation validation")
        assert out.count("K/Kc=") == 3
        assert f"Status: {'pass' if code == 0 else 'fail'}" in out


def test_marfe_config_refuses_a_blank_impurity() -> None:
    """The impurity species must be named."""
    with pytest.raises(ValueError, match="impurity must be a non-empty string"):
        replace(marfe.default_config(), impurity="  ")


def test_marfe_scan_and_fraction_guards() -> None:
    """Impurity fractions above one, empty scans and non-increasing scans are refused."""
    with pytest.raises(ValueError, match=r"impurity_fraction must lie in \(0, 1\]"):
        marfe._impurity_fraction(1.5)
    with pytest.raises(ValueError, match="scan must be non-empty"):
        marfe._ordered_positive_scan("scan", ())
    with pytest.raises(ValueError, match="scan must be strictly increasing"):
        marfe._ordered_positive_scan("scan", (2.0, 2.0))
    assert marfe._ordered_positive_scan("scan", (1, 2.5)) == (1.0, 2.5)


@pytest.mark.parametrize(("module", "title"), [(marfe, "MARFE"), (vmec, "VMEC-lite")])
def test_report_command_writes_the_sealed_pair_and_prints(
    module: Any, title: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --report the command writes sealed JSON and its Markdown twin; it prints JSON or a summary."""
    report = tmp_path / "evidence.json"
    assert module.main(["--report", str(report), "--json-out", "--target-id", "guard-test"]) == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out)["payload_sha256"] == written["payload_sha256"]
    assert module.validate_evidence_payload(written) is True
    assert report.with_suffix(".md").read_text(encoding="utf-8").strip()
    assert module.main([]) == 0
    summary = capsys.readouterr().out
    assert title in summary.splitlines()[0]
    assert summary.rstrip().endswith("Status: pass")


def test_vmec_case_refuses_a_triangularity_or_radius_that_breaks_the_boundary() -> None:
    """Triangularity must lie in (-1, 1) and the major radius must keep the sampled boundary at R > 0."""
    case = vmec.default_case()
    with pytest.raises(ValueError, match=r"triangularity must lie in \(-1, 1\)"):
        replace(case, triangularity=1.0)
    with pytest.raises(ValueError, match="keep the sampled boundary positive"):
        replace(case, major_radius_m=case.minor_radius_m * (1.0 + abs(case.triangularity)))


def test_vmec_evidence_reader_refuses_a_malformed_digest() -> None:
    """A digest that is not 64 lower-case hexadecimal characters is refused."""
    with pytest.raises(ValueError, match="payload_sha256 must be a SHA-256 hex digest"):
        vmec.validate_evidence_payload(
            {"schema_version": vmec.VMEC_LITE_GEOMETRY_SCHEMA_VERSION, "payload_sha256": "ABC"}
        )
