# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lean Verification Validator Branch Tests
"""Negative-path coverage for the Lean formal-verification payload validators.

Builds one canonical payload and mutates a single field per test to drive each
rejection branch, plus direct unit tests for the identifier and string-list
helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scpn_control.scpn.lean_verification import (
    LEAN_LEGACY_CONTRACT_SOURCE_PATHS,
    LEAN_REQUIRED_CONTRACT_MODULE_REFERENCES,
    LEAN_REQUIRED_CONTRACT_MODULE_PREFIXES,
    LEAN_REQUIRED_CONTRACT_SAFETY_CASE_IDS,
    LeanFormalVerificationError,
    LeanFormalVerificationReport,
    _is_sha256_hex,
    build_lean_formal_report_payload,
    is_lean_module_reference,
    is_lean_module_name,
    is_python_module_name,
    is_safe_relative_path,
    is_lean_theorem_name,
    is_safety_case_id,
    load_lean_formal_report,
    validate_bounded_proof_assumptions,
    validate_lean_formal_report_payload,
    validate_non_empty_string_list,
    validate_required_contract_evidence_links,
    validate_required_contract_theorem_coverage,
    validate_lean_module_reference_list,
    validate_safe_relative_path_list,
)


def _report() -> LeanFormalVerificationReport:
    return LeanFormalVerificationReport(
        status="pass",
        solver="Lean 4.13.0",
        lean_version="4.13.0",
        checked_specs=["pid.actuator_saturation", "snn.marking_bounds"],
        artifact_sha256="a" * 64,
        proof_source_sha256="b" * 64,
        lakefile_sha256="c" * 64,
        theorem_names=[
            "ScpnControl.PID.actuatorSaturationPreserved",
            "ScpnControl.SNN.markingBoundsPreserved",
        ],
        theorem_modules=["ScpnControl.PID", "ScpnControl.SNN"],
        proved_contracts=["pid.actuator_saturation", "snn.marking_bounds"],
        module_paths=[
            "scpn_control.control.pid_controller",
            "scpn_control.scpn.controller",
        ],
        safety_case_ids=["SC-PID-ACTUATOR-SATURATION", "SC-SNN-MARKING-BOUNDS"],
        claim_boundary="bounded Lean proof over exported controller envelope",
        proof_assumptions=[
            "bounded actuator command interval from exported artifact readout limits",
            "bounded SNN marking interval [0, 1] from compiled artifact topology",
        ],
    )


def _payload() -> dict[str, Any]:
    return build_lean_formal_report_payload(_report())


class TestSha256HexHelper:
    def test_rejects_wrong_length(self) -> None:
        assert _is_sha256_hex("abc") is False

    def test_rejects_non_hexadecimal_of_correct_length(self) -> None:
        assert _is_sha256_hex("z" * 64) is False

    def test_accepts_valid_digest(self) -> None:
        assert _is_sha256_hex("a" * 64) is True


class TestIdentifierHelpers:
    def test_module_name_rejects_non_alpha_first_segment_char(self) -> None:
        assert is_lean_module_name("1Bad.Name") is False

    def test_module_name_rejects_empty_dotted_segment(self) -> None:
        assert is_lean_module_name("ScpnControl..PID") is False

    def test_theorem_name_allows_prime_suffix(self) -> None:
        assert is_lean_theorem_name("ScpnControl.PID.theorem'") is True

    def test_module_name_forbids_prime_suffix(self) -> None:
        assert is_lean_module_name("ScpnControl.PID'") is False

    def test_safety_case_id_rejects_empty(self) -> None:
        assert is_safety_case_id("") is False

    def test_python_module_name_accepts_installed_package_module(self) -> None:
        assert is_python_module_name("scpn_control.scpn.controller") is True

    def test_python_module_name_rejects_keyword_segment(self) -> None:
        assert is_python_module_name("scpn_control.class.controller") is False

    def test_safe_relative_path_accepts_legacy_source_path(self) -> None:
        assert is_safe_relative_path("src/scpn_control/scpn/controller.py") is True

    def test_lean_module_reference_accepts_installed_and_legacy_forms(self) -> None:
        assert is_lean_module_reference("scpn_control.scpn.controller") is True
        assert is_lean_module_reference("src/scpn_control/scpn/controller.py") is True


class TestStringListValidators:
    def test_rejects_empty_list(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="non-empty list"):
            validate_non_empty_string_list([], "checked_specs")

    def test_rejects_non_string_element(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="non-empty strings"):
            validate_non_empty_string_list([123], "checked_specs")

    def test_rejects_duplicate_element(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="duplicates"):
            validate_non_empty_string_list(["x", "x"], "checked_specs")

    def test_safe_relative_path_rejects_traversal(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="safe relative paths"):
            validate_safe_relative_path_list(["../escape.py"], "module_paths")

    def test_safe_relative_path_rejects_absolute_path(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="safe relative paths"):
            validate_safe_relative_path_list(["/etc/passwd.py"], "module_paths")

    def test_safe_relative_path_accepts_legacy_source_path(self) -> None:
        assert validate_safe_relative_path_list(
            ["src/scpn_control/scpn/controller.py"],
            "module_paths",
        ) == ["src/scpn_control/scpn/controller.py"]

    def test_module_reference_rejects_absolute_path(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="safe relative paths or importable module names"):
            validate_lean_module_reference_list(["/etc/passwd.py"], "module_paths")


class TestBoundedProofAssumptions:
    def test_rejects_certification_claim(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="must not claim certification"):
            validate_bounded_proof_assumptions(["bounded yet certified envelope"])


class TestContractCoverageHelpers:
    def test_theorem_coverage_skips_contracts_not_proved(self) -> None:
        # Only the SNN contract is proved; the PID required entry must be skipped
        # (the `continue` branch) rather than demanding PID theorems.
        validate_required_contract_theorem_coverage(
            proved_contracts=["snn.marking_bounds"],
            theorem_names=["ScpnControl.SNN.markingBoundsPreserved"],
            theorem_modules=["ScpnControl.SNN"],
        )

    def test_evidence_links_reject_unsupported_module_reference(self) -> None:
        contract = "pid.actuator_saturation"
        with pytest.raises(LeanFormalVerificationError, match="unsupported references"):
            validate_required_contract_evidence_links(
                proved_contracts=[contract],
                theorem_names=["ScpnControl.PID.actuatorSaturationPreserved"],
                theorem_modules=[LEAN_REQUIRED_CONTRACT_MODULE_PREFIXES[contract]],
                module_paths=[LEAN_REQUIRED_CONTRACT_MODULE_REFERENCES[contract], "scpn_control.extra.unsupported"],
                safety_case_ids=[LEAN_REQUIRED_CONTRACT_SAFETY_CASE_IDS[contract]],
            )

    def test_evidence_links_accept_legacy_source_path(self) -> None:
        contract = "pid.actuator_saturation"
        validate_required_contract_evidence_links(
            proved_contracts=[contract],
            theorem_names=["ScpnControl.PID.actuatorSaturationPreserved"],
            theorem_modules=[LEAN_REQUIRED_CONTRACT_MODULE_PREFIXES[contract]],
            module_paths=[LEAN_LEGACY_CONTRACT_SOURCE_PATHS[contract]],
            safety_case_ids=[LEAN_REQUIRED_CONTRACT_SAFETY_CASE_IDS[contract]],
        )

    def test_evidence_links_reject_unsupported_safety_case_id(self) -> None:
        contract = "pid.actuator_saturation"
        with pytest.raises(LeanFormalVerificationError, match="unsupported IDs"):
            validate_required_contract_evidence_links(
                proved_contracts=[contract],
                theorem_names=["ScpnControl.PID.actuatorSaturationPreserved"],
                theorem_modules=[LEAN_REQUIRED_CONTRACT_MODULE_PREFIXES[contract]],
                module_paths=[LEAN_REQUIRED_CONTRACT_MODULE_REFERENCES[contract]],
                safety_case_ids=[LEAN_REQUIRED_CONTRACT_SAFETY_CASE_IDS[contract], "SC-EXTRA-UNSUPPORTED"],
            )


class TestPayloadRejection:
    def test_rejects_non_object(self) -> None:
        with pytest.raises(LeanFormalVerificationError, match="must be an object"):
            validate_lean_formal_report_payload(["not", "a", "dict"])

    def test_rejects_wrong_schema_version(self) -> None:
        payload = _payload()
        payload["schema_version"] = "scpn-control.lean-formal-report.v0"
        with pytest.raises(LeanFormalVerificationError, match="schema_version is invalid"):
            validate_lean_formal_report_payload(payload)

    def test_rejects_wrong_backend(self) -> None:
        payload = _payload()
        payload["backend"] = "z3"
        with pytest.raises(LeanFormalVerificationError, match="backend must be lean4"):
            validate_lean_formal_report_payload(payload)

    def test_rejects_invalid_status(self) -> None:
        payload = _payload()
        payload["status"] = "unknown"
        with pytest.raises(LeanFormalVerificationError, match="status is invalid"):
            validate_lean_formal_report_payload(payload)

    def test_rejects_non_string_solver(self) -> None:
        payload = _payload()
        payload["solver"] = 1234
        with pytest.raises(LeanFormalVerificationError, match="solver is invalid"):
            validate_lean_formal_report_payload(payload)

    def test_rejects_malformed_artifact_digest(self) -> None:
        payload = _payload()
        payload["artifact_sha256"] = "not-a-sha"
        with pytest.raises(LeanFormalVerificationError, match="artifact_sha256 is invalid"):
            validate_lean_formal_report_payload(payload)

    def test_rejects_malformed_assumption_digest(self) -> None:
        payload = _payload()
        payload["assumption_sha256"] = "short"
        with pytest.raises(LeanFormalVerificationError, match="assumption_sha256 is invalid"):
            validate_lean_formal_report_payload(payload)

    def test_rejects_malformed_payload_digest(self) -> None:
        payload = _payload()
        payload["payload_sha256"] = "tampered"
        with pytest.raises(LeanFormalVerificationError, match="payload_sha256 is invalid"):
            validate_lean_formal_report_payload(payload)


class TestLoadRejectsNonObject:
    def test_load_rejects_top_level_array(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(LeanFormalVerificationError, match="must be a JSON object"):
            load_lean_formal_report(path)
