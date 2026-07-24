# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact contract validation and safety-critical admit

"""Runtime validation for SCPN controller artifacts (``.scpnctl.json``).

This leaf owns :class:`ArtifactValidationError`, structural contract checks on
:class:`~scpn_control.scpn.artifact_model.Artifact`, formal-verification evidence
admission, and safety-critical admit
(:func:`validate_safety_critical_artifact`). Load/save, JSON schema, and compact
codec remain on :mod:`scpn_control.scpn.artifact` (CTL-G07 R4-S2).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from scpn_control.scpn.artifact_model import (
    FORMAL_VERIFICATION_BACKENDS,
    Artifact,
    FormalVerificationEvidence,
)
from scpn_control.scpn.lean_verification import (
    LEAN_REQUIRED_PROVED_CONTRACTS,
    LeanFormalVerificationError,
    compute_assumption_sha256,
    is_lean_module_name,
    is_lean_theorem_name,
    is_safety_case_id,
    load_lean_formal_report,
    validate_bounded_proof_assumptions,
    validate_lean_module_reference_list,
    validate_lean_solver_version_binding,
    validate_required_contract_evidence_links,
    validate_required_contract_theorem_coverage,
)


class ArtifactValidationError(ValueError):
    """Raised when an artifact fails lightweight validation."""


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_non_empty_string_list(
    value: object,
    field_name: str,
    *,
    validator: Callable[[str], bool] | None = None,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ArtifactValidationError(f"formal_verification.{field_name} must be a non-empty list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise ArtifactValidationError(f"formal_verification.{field_name} must contain non-empty strings")
        if validator is not None and not validator(item):
            raise ArtifactValidationError(f"formal_verification.{field_name} contains invalid identifier")
        if item in seen:
            raise ArtifactValidationError(f"formal_verification.{field_name} must not contain duplicates")
        seen.add(item)
        result.append(item)
    return result


def _validate_lean4_formal_verification(evidence: FormalVerificationEvidence) -> None:
    if not isinstance(evidence.lean_version, str) or not evidence.lean_version:
        raise ArtifactValidationError("formal_verification.lean_version must be a non-empty string for lean4")
    try:
        validate_lean_solver_version_binding(solver=evidence.solver, lean_version=evidence.lean_version)
    except LeanFormalVerificationError as exc:
        raise ArtifactValidationError(f"formal_verification.{exc}") from exc
    if not isinstance(evidence.lakefile_sha256, str) or not _is_sha256_hex(evidence.lakefile_sha256):
        raise ArtifactValidationError("formal_verification.lakefile_sha256 must be a SHA-256 hex digest for lean4")
    if not isinstance(evidence.proof_source_sha256, str) or not _is_sha256_hex(evidence.proof_source_sha256):
        raise ArtifactValidationError("formal_verification.proof_source_sha256 must be a SHA-256 hex digest for lean4")
    theorem_names = _validate_non_empty_string_list(
        evidence.theorem_names,
        "theorem_names",
        validator=is_lean_theorem_name,
    )
    theorem_modules = _validate_non_empty_string_list(
        evidence.theorem_modules,
        "theorem_modules",
        validator=is_lean_module_name,
    )
    proved_contracts = _validate_non_empty_string_list(evidence.proved_contracts, "proved_contracts")
    try:
        module_paths = validate_lean_module_reference_list(evidence.module_paths, "module_paths")
    except LeanFormalVerificationError as exc:
        raise ArtifactValidationError(f"formal_verification.{exc}") from exc
    safety_case_ids = _validate_non_empty_string_list(
        evidence.safety_case_ids,
        "safety_case_ids",
        validator=is_safety_case_id,
    )
    try:
        proof_assumptions = validate_bounded_proof_assumptions(evidence.proof_assumptions)
    except LeanFormalVerificationError as exc:
        raise ArtifactValidationError(f"formal_verification.{exc}") from exc
    if not isinstance(evidence.assumption_sha256, str) or not _is_sha256_hex(evidence.assumption_sha256):
        raise ArtifactValidationError("formal_verification.assumption_sha256 must be a SHA-256 hex digest for lean4")
    if evidence.assumption_sha256.lower() != compute_assumption_sha256(proof_assumptions):
        raise ArtifactValidationError(
            "formal_verification.assumption_sha256 does not match formal_verification.proof_assumptions"
        )
    missing_contracts = sorted(LEAN_REQUIRED_PROVED_CONTRACTS.difference(proved_contracts))
    if missing_contracts:
        raise ArtifactValidationError(
            "formal_verification.proved_contracts missing required lean4 contracts: " + ", ".join(missing_contracts)
        )
    unsupported_contracts = sorted(set(proved_contracts).difference(LEAN_REQUIRED_PROVED_CONTRACTS))
    if unsupported_contracts:
        raise ArtifactValidationError(
            "formal_verification.proved_contracts contains unsupported lean4 contracts: "
            + ", ".join(unsupported_contracts)
        )
    missing_specs = sorted(set(proved_contracts).difference(evidence.checked_specs))
    if missing_specs:
        raise ArtifactValidationError(
            "formal_verification.checked_specs must include every lean4 proved_contract: " + ", ".join(missing_specs)
        )
    if len(theorem_modules) > len(theorem_names):
        raise ArtifactValidationError("formal_verification.theorem_modules cannot exceed theorem_names")
    try:
        validate_required_contract_theorem_coverage(
            proved_contracts=proved_contracts,
            theorem_names=theorem_names,
            theorem_modules=theorem_modules,
        )
    except LeanFormalVerificationError as exc:
        raise ArtifactValidationError(f"formal_verification.{exc}") from exc
    try:
        validate_required_contract_evidence_links(
            proved_contracts=proved_contracts,
            theorem_names=theorem_names,
            theorem_modules=theorem_modules,
            module_paths=module_paths,
            safety_case_ids=safety_case_ids,
        )
    except LeanFormalVerificationError as exc:
        raise ArtifactValidationError(f"formal_verification.{exc}") from exc


def _validate_formal_verification(evidence: FormalVerificationEvidence) -> None:
    if not isinstance(evidence.required, bool):
        raise ArtifactValidationError("formal_verification.required must be a boolean")
    if evidence.status not in {"pass", "fail", "blocked"}:
        raise ArtifactValidationError("formal_verification.status must be 'pass', 'fail', or 'blocked'")
    if not isinstance(evidence.backend, str) or not evidence.backend:
        raise ArtifactValidationError("formal_verification.backend must be a non-empty string")
    if evidence.backend not in FORMAL_VERIFICATION_BACKENDS:
        raise ArtifactValidationError(
            f"formal_verification.backend must be one of {', '.join(sorted(FORMAL_VERIFICATION_BACKENDS))}"
        )
    if not isinstance(evidence.solver, str) or not evidence.solver:
        raise ArtifactValidationError("formal_verification.solver must be a non-empty string")
    if isinstance(evidence.max_depth, bool) or not isinstance(evidence.max_depth, int) or evidence.max_depth < 0:
        raise ArtifactValidationError("formal_verification.max_depth must be an integer >= 0")
    if not isinstance(evidence.checked_specs, list) or not evidence.checked_specs:
        raise ArtifactValidationError("formal_verification.checked_specs must be a non-empty list")
    for spec in evidence.checked_specs:
        if not isinstance(spec, str) or not spec:
            raise ArtifactValidationError("formal_verification.checked_specs must contain non-empty strings")
    if not isinstance(evidence.artifact_sha256, str) or not _is_sha256_hex(evidence.artifact_sha256):
        raise ArtifactValidationError("formal_verification.artifact_sha256 must be a SHA-256 hex digest")
    if not isinstance(evidence.report_sha256, str) or not _is_sha256_hex(evidence.report_sha256):
        raise ArtifactValidationError("formal_verification.report_sha256 must be a SHA-256 hex digest")
    if not isinstance(evidence.claim_boundary, str) or not evidence.claim_boundary:
        raise ArtifactValidationError("formal_verification.claim_boundary must be a non-empty string")
    boundary = evidence.claim_boundary.lower()
    if "bounded" not in boundary or "unbounded" in boundary:
        raise ArtifactValidationError("formal_verification.claim_boundary must state a bounded proof boundary")
    if evidence.report_uri is not None and (not isinstance(evidence.report_uri, str) or not evidence.report_uri):
        raise ArtifactValidationError("formal_verification.report_uri must be a non-empty string when supplied")
    if evidence.report_uri is not None:
        _formal_report_relative_path(evidence)
    if evidence.generated_utc is not None and (
        not isinstance(evidence.generated_utc, str) or not evidence.generated_utc
    ):
        raise ArtifactValidationError("formal_verification.generated_utc must be a non-empty string when supplied")
    if evidence.counterexample_path is not None:
        if not isinstance(evidence.counterexample_path, list) or not evidence.counterexample_path:
            raise ArtifactValidationError("formal_verification.counterexample_path must be a non-empty list")
        for transition in evidence.counterexample_path:
            if not isinstance(transition, str) or not transition:
                raise ArtifactValidationError("formal_verification.counterexample_path must contain transition names")
    if evidence.counterexample_property is not None and (
        not isinstance(evidence.counterexample_property, str) or not evidence.counterexample_property
    ):
        raise ArtifactValidationError("formal_verification.counterexample_property must be non-empty when supplied")
    if evidence.status == "fail" and (evidence.counterexample_path is None or evidence.counterexample_property is None):
        raise ArtifactValidationError(
            "formal_verification failed proof evidence must include counterexample path and property"
        )
    if evidence.status == "pass" and (
        evidence.counterexample_path is not None or evidence.counterexample_property is not None
    ):
        raise ArtifactValidationError("formal_verification passing evidence must not include a counterexample")
    if evidence.backend == "lean4":
        _validate_lean4_formal_verification(evidence)


def _formal_report_relative_path(evidence: FormalVerificationEvidence) -> Path | None:
    uri = evidence.report_uri
    if uri is None:
        return None
    if "\\" in uri or "://" in uri or uri.startswith(("file:", "/", "~")):
        raise ArtifactValidationError("formal_verification.report_uri must be a safe relative repository path")
    rel = PurePosixPath(uri)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ArtifactValidationError("formal_verification.report_uri must be a safe relative repository path")
    return Path(*rel.parts)


def _verify_formal_report_digest(
    evidence: FormalVerificationEvidence,
    formal_report_root: str | Path | None,
) -> None:
    if formal_report_root is None:
        return
    rel = _formal_report_relative_path(evidence)
    if rel is None:
        raise ArtifactValidationError("safety-critical artifact requires formal_verification.report_uri")
    root = Path(formal_report_root).resolve()
    report_path = (root / rel).resolve()
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise ArtifactValidationError("formal_verification.report_uri escapes formal_report_root") from exc
    if not report_path.is_file():
        raise ArtifactValidationError("formal_verification.report_uri does not resolve to a report file")
    report_bytes = report_path.read_bytes()
    actual = hashlib.sha256(report_bytes).hexdigest()
    if actual != evidence.report_sha256.lower():
        raise ArtifactValidationError("formal_verification.report_sha256 does not match report file")
    if evidence.backend == "z3":
        from scpn_control.scpn.z3_formal_report import load_z3_formal_report

        try:
            report_payload = load_z3_formal_report(report_path)
        except ValueError as exc:
            raise ArtifactValidationError(
                "formal_verification.report_uri must reference a valid Z3 report: " + str(exc)
            ) from exc
        if report_payload["status"] != evidence.status:
            raise ArtifactValidationError("formal_verification.status does not match Z3 report")
        if report_payload["max_depth"] != evidence.max_depth:
            raise ArtifactValidationError("formal_verification.max_depth does not match Z3 report")
        if report_payload["checked_specs"] != evidence.checked_specs:
            raise ArtifactValidationError("formal_verification.checked_specs does not match Z3 report")
        if report_payload["solver"] != evidence.solver:
            raise ArtifactValidationError("formal_verification.solver does not match Z3 report")
    if evidence.backend == "lean4":
        try:
            report_payload = load_lean_formal_report(report_path)
        except LeanFormalVerificationError as exc:
            raise ArtifactValidationError(
                "formal_verification.report_uri must reference a valid Lean 4 report: " + str(exc)
            ) from exc
        expected = {
            "status": evidence.status,
            "solver": evidence.solver,
            "lean_version": evidence.lean_version,
            "checked_specs": evidence.checked_specs,
            "artifact_sha256": evidence.artifact_sha256.lower(),
            "proof_source_sha256": evidence.proof_source_sha256.lower()
            if evidence.proof_source_sha256 is not None
            else None,
            "lakefile_sha256": evidence.lakefile_sha256.lower() if evidence.lakefile_sha256 is not None else None,
            "theorem_names": evidence.theorem_names,
            "theorem_modules": evidence.theorem_modules,
            "proved_contracts": evidence.proved_contracts,
            "module_paths": evidence.module_paths,
            "safety_case_ids": evidence.safety_case_ids,
            "claim_boundary": evidence.claim_boundary,
            "proof_assumptions": evidence.proof_assumptions,
            "assumption_sha256": evidence.assumption_sha256.lower() if evidence.assumption_sha256 is not None else None,
        }
        for key, expected_value in expected.items():
            actual_value = report_payload[key]
            if key.endswith("_sha256") and isinstance(actual_value, str):
                actual_value = actual_value.lower()
            if actual_value != expected_value:
                raise ArtifactValidationError(f"formal_verification.{key} does not match Lean 4 report")


def validate_safety_critical_artifact(
    artifact: Artifact,
    formal_report_root: str | Path | None = None,
) -> None:
    """Fail closed unless a controller artifact carries passing bounded-proof evidence."""
    # Lazy import: payload hash lives on the owner (load/save/codec surface).
    from scpn_control.scpn.artifact import compute_artifact_payload_sha256

    if artifact.formal_verification is None:
        raise ArtifactValidationError("safety-critical artifact requires formal_verification evidence")
    _validate_formal_verification(artifact.formal_verification)
    if not artifact.formal_verification.required:
        raise ArtifactValidationError("safety-critical artifact formal_verification.required must be true")
    if artifact.formal_verification.status != "pass":
        raise ArtifactValidationError("safety-critical artifact requires passing formal_verification evidence")
    if artifact.formal_verification.report_uri is None:
        raise ArtifactValidationError("safety-critical artifact requires formal_verification.report_uri")
    actual_artifact_hash = compute_artifact_payload_sha256(artifact)
    if actual_artifact_hash != artifact.formal_verification.artifact_sha256.lower():
        raise ArtifactValidationError("formal_verification.artifact_sha256 does not match artifact payload")
    _verify_formal_report_digest(artifact.formal_verification, formal_report_root)


def _validate(artifact: Artifact) -> None:
    """Lightweight checks: required fields, ranges, shape consistency."""
    meta = artifact.meta
    if artifact.formal_verification is not None:
        _validate_formal_verification(artifact.formal_verification)

    if meta.firing_mode not in ("binary", "fractional"):
        raise ArtifactValidationError(f"firing_mode must be 'binary' or 'fractional', got '{meta.firing_mode}'")
    if isinstance(meta.firing_margin, bool) or not isinstance(meta.firing_margin, (int, float)):
        raise ArtifactValidationError("firing_margin must be finite and >= 0")
    if not math.isfinite(meta.firing_margin) or meta.firing_margin < 0.0:
        raise ArtifactValidationError("firing_margin must be finite and >= 0")

    if isinstance(meta.fixed_point.data_width, bool) or not isinstance(meta.fixed_point.data_width, int):
        raise ArtifactValidationError("fixed_point.data_width must be an integer >= 1")
    if meta.fixed_point.data_width < 1:
        raise ArtifactValidationError("fixed_point.data_width must be >= 1")
    if isinstance(meta.fixed_point.fraction_bits, bool) or not isinstance(meta.fixed_point.fraction_bits, int):
        raise ArtifactValidationError("fixed_point.fraction_bits must be an integer >= 0")
    if meta.fixed_point.fraction_bits < 0:
        raise ArtifactValidationError("fixed_point.fraction_bits must be >= 0")
    if meta.fixed_point.fraction_bits >= meta.fixed_point.data_width:
        raise ArtifactValidationError("fixed_point.fraction_bits must be < fixed_point.data_width")
    if not isinstance(meta.fixed_point.signed, bool):
        raise ArtifactValidationError("fixed_point.signed must be a boolean")

    if isinstance(meta.stream_length, bool) or not isinstance(meta.stream_length, int):
        raise ArtifactValidationError("stream_length must be an integer >= 1")
    if meta.stream_length < 1:
        raise ArtifactValidationError("stream_length must be >= 1")

    if isinstance(meta.dt_control_s, bool) or not isinstance(meta.dt_control_s, (int, float)):
        raise ArtifactValidationError("dt_control_s must be finite and > 0")
    if not math.isfinite(meta.dt_control_s):
        raise ArtifactValidationError("dt_control_s must be finite and > 0")
    if meta.dt_control_s <= 0:
        raise ArtifactValidationError("dt_control_s must be > 0")

    # Weight ranges
    for val in artifact.weights.w_in.data:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ArtifactValidationError("w_in weights must be finite numeric values in [0, 1]")
        if not (0.0 <= val <= 1.0):
            raise ArtifactValidationError(
                f"w_in weight {val} outside [0, 1]; controller artifacts do not encode inhibitor arcs"
            )
    for val in artifact.weights.w_out.data:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ArtifactValidationError("w_out weights must be finite numeric values in [0, 1]")
        if not (0.0 <= val <= 1.0):
            raise ArtifactValidationError(f"w_out weight {val} outside [0, 1]")

    # Threshold ranges
    for t in artifact.topology.transitions:
        if isinstance(t.threshold, bool) or not isinstance(t.threshold, (int, float)):
            raise ArtifactValidationError(f"threshold {t.threshold} for '{t.name}' must be finite and in [0, 1]")
        if not math.isfinite(t.threshold):
            raise ArtifactValidationError(f"threshold {t.threshold} for '{t.name}' must be finite and in [0, 1]")
        if not (0.0 <= t.threshold <= 1.0):
            raise ArtifactValidationError(f"threshold {t.threshold} for '{t.name}' outside [0, 1]")
        if t.margin is not None:
            if isinstance(t.margin, bool) or not isinstance(t.margin, (int, float)):
                raise ArtifactValidationError(f"margin {t.margin} for '{t.name}' must be finite and >= 0")
            if not math.isfinite(t.margin) or t.margin < 0.0:
                raise ArtifactValidationError(f"margin {t.margin} for '{t.name}' must be finite and >= 0")
        if isinstance(t.delay_ticks, bool) or not isinstance(t.delay_ticks, int):
            raise ArtifactValidationError(f"delay_ticks {t.delay_ticks} for '{t.name}' must be an integer >= 0")
        if t.delay_ticks < 0:
            raise ArtifactValidationError(f"delay_ticks {t.delay_ticks} for '{t.name}' must be >= 0")

    # Shape consistency
    nP = artifact.nP
    nT = artifact.nT
    expected_w_in = nT * nP
    expected_w_out = nP * nT

    if len(artifact.weights.w_in.data) != expected_w_in:
        raise ArtifactValidationError(f"w_in data length {len(artifact.weights.w_in.data)} != nT*nP={expected_w_in}")
    if len(artifact.weights.w_out.data) != expected_w_out:
        raise ArtifactValidationError(f"w_out data length {len(artifact.weights.w_out.data)} != nP*nT={expected_w_out}")

    # Marking length
    if len(artifact.initial_state.marking) != nP:
        raise ArtifactValidationError(f"marking length {len(artifact.initial_state.marking)} != nP={nP}")
    for val in artifact.initial_state.marking:
        if not (0.0 <= val <= 1.0):
            raise ArtifactValidationError(f"initial marking {val} outside [0, 1]")
    for inj in artifact.initial_state.place_injections:
        if isinstance(inj.place_id, bool) or not isinstance(inj.place_id, int):
            raise ArtifactValidationError("place_injections.place_id must be an integer")
        if inj.place_id < 0 or inj.place_id >= nP:
            raise ArtifactValidationError(f"place_injections.place_id {inj.place_id} out of bounds for nP={nP}")
        if not isinstance(inj.source, str) or not inj.source:
            raise ArtifactValidationError("place_injections.source must be a non-empty string")
        if isinstance(inj.scale, bool) or not isinstance(inj.scale, (int, float)) or not math.isfinite(inj.scale):
            raise ArtifactValidationError("place_injections.scale must be finite numeric")
        if isinstance(inj.offset, bool) or not isinstance(inj.offset, (int, float)) or not math.isfinite(inj.offset):
            raise ArtifactValidationError("place_injections.offset must be finite numeric")
        if not isinstance(inj.clamp_0_1, bool):
            raise ArtifactValidationError("place_injections.clamp_0_1 must be a boolean")

    # Readout consistency
    n_actions = len(artifact.readout.actions)
    for action in artifact.readout.actions:
        if isinstance(action.id, bool) or not isinstance(action.id, int) or action.id < 0:
            raise ArtifactValidationError("readout.actions.id must be an integer >= 0")
        if not isinstance(action.name, str) or not action.name:
            raise ArtifactValidationError("readout.actions.name must be a non-empty string")
        if isinstance(action.pos_place, bool) or not isinstance(action.pos_place, int):
            raise ArtifactValidationError("readout.actions.pos_place must be an integer")
        if isinstance(action.neg_place, bool) or not isinstance(action.neg_place, int):
            raise ArtifactValidationError("readout.actions.neg_place must be an integer")
        if action.pos_place < 0 or action.pos_place >= nP:
            raise ArtifactValidationError(f"readout.actions.pos_place {action.pos_place} out of bounds for nP={nP}")
        if action.neg_place < 0 or action.neg_place >= nP:
            raise ArtifactValidationError(f"readout.actions.neg_place {action.neg_place} out of bounds for nP={nP}")
    if len(artifact.readout.gains) != n_actions:
        raise ArtifactValidationError("readout.gains length must equal number of actions")
    if len(artifact.readout.abs_max) != n_actions:
        raise ArtifactValidationError("readout.abs_max length must equal number of actions")
    if len(artifact.readout.slew_per_s) != n_actions:
        raise ArtifactValidationError("readout.slew_per_s length must equal number of actions")
    for val in artifact.readout.gains:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ArtifactValidationError("readout.gains must contain finite numeric values")
    for val in artifact.readout.abs_max:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
            raise ArtifactValidationError("readout.abs_max must contain finite numeric values >= 0")
    for val in artifact.readout.slew_per_s:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
            raise ArtifactValidationError("readout.slew_per_s must contain finite numeric values >= 0")


def validate_artifact(artifact: Artifact) -> None:
    """Validate a controller artifact before runtime use.

    Parameters
    ----------
    artifact
        Parsed or directly constructed artifact to validate.

    Raises
    ------
    ArtifactValidationError
        If metadata, topology, weights, readout, initial state, or formal
        evidence fields violate the controller artifact contract.
    """
    _validate(artifact)
