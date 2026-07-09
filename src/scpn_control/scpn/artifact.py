# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact admission and validation.
"""
SCPN Controller Artifact (``.scpnctl.json``) loader / saver.

Defines the ``Artifact`` dataclass that mirrors the JSON schema sections
(meta, topology, weights, readout, initial_state) and provides lightweight
validation on direct construction, load, save, and controller admission.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import zlib
from collections.abc import Callable
from dataclasses import MISSING, InitVar, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

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

ARTIFACT_SCHEMA_VERSION = "1.0.0"
MAX_PACKED_WORDS = 10_000_000
MAX_DECOMPRESSED_BYTES = MAX_PACKED_WORDS * 8
MAX_COMPRESSED_BYTES = 50_000_000
FORMAL_VERIFICATION_BACKENDS = {"explicit-state", "lean4", "z3"}
SHA256_HEX_PATTERN = "^[0-9a-fA-F]{64}$"
SAFE_RELATIVE_PATH_PATTERN = r"^(?!/|~|file:|.*://)(?!.*(?:^|/)\.\.(?:/|$))(?!.*\\).+"


# ── Sub-structures ──────────────────────────────────────────────────────────


@dataclass
class FixedPoint:
    """Fixed-point number format of the packed weights.

    Attributes
    ----------
    data_width
        Total word width in bits.
    fraction_bits
        Number of fractional bits.
    signed
        Whether the format is signed (two's complement).
    """

    data_width: int
    fraction_bits: int
    signed: bool


@dataclass
class SeedPolicy:
    """Deterministic random-seed policy for reproducible firing.

    Attributes
    ----------
    id
        Seed-policy identifier.
    hash_fn
        Name of the hash function used to derive seeds.
    rng_family
        Random-number-generator family.
    """

    id: str
    hash_fn: str
    rng_family: str


@dataclass
class CompilerInfo:
    """Provenance of the compiler that produced the artifact.

    Attributes
    ----------
    name
        Compiler name.
    version
        Compiler version string.
    git_sha
        Git commit SHA of the compiler build.
    """

    name: str
    version: str
    git_sha: str


@dataclass
class ArtifactMeta:
    """Metadata header of a controller artifact.

    Attributes
    ----------
    artifact_version
        Artifact schema version.
    name
        Human-readable controller name.
    dt_control_s
        Control time step in seconds.
    stream_length
        Number of stochastic stream words per weight.
    fixed_point
        Fixed-point format of the packed weights.
    firing_mode
        Transition firing mode.
    firing_margin
        Default fractional firing margin used when a transition has no
        transition-specific margin.
    seed_policy
        Deterministic seed policy.
    created_utc
        Creation timestamp in UTC ISO-8601.
    compiler
        Compiler provenance.
    notes
        Optional free-text notes.
    """

    artifact_version: str
    name: str
    dt_control_s: float
    stream_length: int
    fixed_point: FixedPoint
    firing_mode: str
    seed_policy: SeedPolicy
    created_utc: str
    compiler: CompilerInfo
    firing_margin: float = 0.05
    notes: str | None = None


@dataclass
class PlaceSpec:
    """A Petri-net place.

    Attributes
    ----------
    id
        Place index.
    name
        Place name.
    """

    id: int
    name: str


@dataclass
class TransitionSpec:
    """A Petri-net transition with its firing threshold.

    Attributes
    ----------
    id
        Transition index.
    name
        Transition name.
    threshold
        Firing threshold on the weighted input.
    margin
        Optional hysteresis margin around the threshold.
    delay_ticks
        Firing delay in control ticks.
    """

    id: int
    name: str
    threshold: float
    margin: float | None = None
    delay_ticks: int = 0


@dataclass
class Topology:
    """Petri-net topology: places and transitions.

    Attributes
    ----------
    places
        The network places.
    transitions
        The network transitions.
    """

    places: List[PlaceSpec]
    transitions: List[TransitionSpec]


@dataclass
class WeightMatrix:
    """Dense row-major weight matrix.

    Attributes
    ----------
    shape
        ``[rows, cols]`` of the matrix.
    data
        Row-major matrix entries. Controller artifacts require non-negative
        weights; inhibitor arcs are not encoded in the dense artifact matrix.
    """

    shape: List[int]  # [rows, cols]
    data: List[float]  # row-major


@dataclass
class PackedWeights:
    """Bit-packed stochastic weight tensor.

    Attributes
    ----------
    shape
        ``[rows, cols, words]`` of the packed tensor.
    data_u64
        Packed 64-bit words.
    """

    shape: List[int]  # [rows, cols, words]
    data_u64: List[int]


@dataclass
class PackedWeightsGroup:
    """Packed input and optional output weight streams.

    Attributes
    ----------
    words_per_stream
        Number of 64-bit words per stochastic stream.
    w_in_packed
        Packed input weights.
    w_out_packed
        Optional packed output weights.
    """

    words_per_stream: int
    w_in_packed: PackedWeights
    w_out_packed: PackedWeights | None = None


@dataclass
class Weights:
    """Controller weights in dense and optional packed form.

    Attributes
    ----------
    w_in
        Dense input weight matrix.
    w_out
        Dense output weight matrix.
    packed
        Optional bit-packed stochastic weights.
    """

    w_in: WeightMatrix
    w_out: WeightMatrix
    packed: PackedWeightsGroup | None = None


@dataclass
class ActionReadout:
    """Differential readout mapping two places to one action.

    Attributes
    ----------
    id
        Action index.
    name
        Action name.
    pos_place
        Place contributing the positive term.
    neg_place
        Place contributing the negative term.
    """

    id: int
    name: str
    pos_place: int
    neg_place: int


@dataclass
class Readout:
    """Action readout layer with per-action scaling and limits.

    Attributes
    ----------
    actions
        The differential action readouts.
    gains
        Per-action output gain.
    abs_max
        Per-action absolute output limit.
    slew_per_s
        Per-action slew-rate limit in units per second.
    """

    actions: List[ActionReadout]
    gains: List[float]
    abs_max: List[float]
    slew_per_s: List[float]


@dataclass
class PlaceInjection:
    """Mapping of an external signal into a place marking.

    Attributes
    ----------
    place_id
        Target place index.
    source
        Name of the external input signal.
    scale
        Multiplicative scale applied to the source.
    offset
        Additive offset applied after scaling.
    clamp_0_1
        Whether the result is clamped to [0, 1].
    """

    place_id: int
    source: str
    scale: float
    offset: float
    clamp_0_1: bool


@dataclass
class InitialState:
    """Initial marking and external input injections.

    Attributes
    ----------
    marking
        Initial token marking per place.
    place_injections
        External-signal injections into places.
    """

    marking: List[float]
    place_injections: List[PlaceInjection]


@dataclass
class FormalVerificationEvidence:
    """Hashable bounded-proof admission evidence for controller artifacts."""

    required: bool
    status: str
    backend: str
    solver: str
    max_depth: int
    checked_specs: List[str]
    artifact_sha256: str
    report_sha256: str
    claim_boundary: str
    report_uri: str | None = None
    generated_utc: str | None = None
    counterexample_path: List[str] | None = None
    counterexample_property: str | None = None
    lean_version: str | None = None
    lakefile_sha256: str | None = None
    proof_source_sha256: str | None = None
    theorem_names: List[str] | None = None
    theorem_modules: List[str] | None = None
    proved_contracts: List[str] | None = None
    module_paths: List[str] | None = None
    safety_case_ids: List[str] | None = None
    proof_assumptions: List[str] | None = None
    assumption_sha256: str | None = None


FORMAL_VERIFICATION_ALLOWED_FIELDS = frozenset(FormalVerificationEvidence.__dataclass_fields__)


def _required_dataclass_field_names(dataclass_type: Any) -> tuple[str, ...]:
    """Return dataclass field names that have no default value."""

    return tuple(
        field.name for field in fields(dataclass_type) if field.default is MISSING and field.default_factory is MISSING
    )


ARTIFACT_META_REQUIRED_FIELDS = tuple(field.name for field in fields(ArtifactMeta) if field.name != "notes")
FIXED_POINT_REQUIRED_FIELDS = _required_dataclass_field_names(FixedPoint)
SEED_POLICY_REQUIRED_FIELDS = _required_dataclass_field_names(SeedPolicy)
COMPILER_REQUIRED_FIELDS = _required_dataclass_field_names(CompilerInfo)
PLACE_SPEC_REQUIRED_FIELDS = _required_dataclass_field_names(PlaceSpec)
TRANSITION_SPEC_REQUIRED_FIELDS = tuple(field.name for field in fields(TransitionSpec) if field.name != "margin")
WEIGHT_MATRIX_REQUIRED_FIELDS = _required_dataclass_field_names(WeightMatrix)
PACKED_WEIGHT_REQUIRED_RAW_FIELDS = ("shape", "data_u64")
PACKED_WEIGHT_REQUIRED_COMPACT_FIELDS = ("shape", "encoding", "count", "data_u64_b64_zlib")
PACKED_WEIGHTS_GROUP_REQUIRED_FIELDS = ("words_per_stream", "w_in_packed")
ACTION_READOUT_REQUIRED_FIELDS = _required_dataclass_field_names(ActionReadout)
READOUT_REQUIRED_FIELDS = _required_dataclass_field_names(Readout)
PLACE_INJECTION_REQUIRED_FIELDS = _required_dataclass_field_names(PlaceInjection)
INITIAL_STATE_REQUIRED_FIELDS = _required_dataclass_field_names(InitialState)
FORMAL_VERIFICATION_REQUIRED_FIELDS = _required_dataclass_field_names(FormalVerificationEvidence)


# ── Artifact ────────────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """Full SCPN controller artifact (``.scpnctl.json``)."""

    meta: ArtifactMeta
    topology: Topology
    weights: Weights
    readout: Readout
    initial_state: InitialState
    formal_verification: FormalVerificationEvidence | None = None
    validate_on_init: InitVar[bool] = True

    def __post_init__(self, validate_on_init: bool) -> None:
        """Validate direct artifact construction unless explicitly disabled."""
        if validate_on_init:
            validate_artifact(self)

    @property
    def nP(self) -> int:
        """Number of places in the topology."""
        return len(self.topology.places)

    @property
    def nT(self) -> int:
        """Number of transitions in the topology."""
        return len(self.topology.transitions)


ARTIFACT_PAYLOAD_REQUIRED_SECTIONS = tuple(
    field.name for field in fields(Artifact) if field.name != "formal_verification"
)


# ── Validation ──────────────────────────────────────────────────────────────


class ArtifactValidationError(ValueError):
    """Raised when an artifact fails lightweight validation."""


def _parse_formal_verification(raw: Any) -> FormalVerificationEvidence | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ArtifactValidationError("formal_verification must be an object")
    unsupported_fields = sorted(set(raw).difference(FORMAL_VERIFICATION_ALLOWED_FIELDS))
    if unsupported_fields:
        raise ArtifactValidationError(
            "formal_verification contains unsupported fields: " + ", ".join(unsupported_fields)
        )
    return FormalVerificationEvidence(
        required=raw["required"],
        status=raw["status"],
        backend=raw["backend"],
        solver=raw["solver"],
        max_depth=raw["max_depth"],
        checked_specs=raw["checked_specs"],
        artifact_sha256=raw["artifact_sha256"],
        report_sha256=raw["report_sha256"],
        claim_boundary=raw["claim_boundary"],
        report_uri=raw.get("report_uri"),
        generated_utc=raw.get("generated_utc"),
        counterexample_path=raw.get("counterexample_path"),
        counterexample_property=raw.get("counterexample_property"),
        lean_version=raw.get("lean_version"),
        lakefile_sha256=raw.get("lakefile_sha256"),
        proof_source_sha256=raw.get("proof_source_sha256"),
        theorem_names=raw.get("theorem_names"),
        theorem_modules=raw.get("theorem_modules"),
        proved_contracts=raw.get("proved_contracts"),
        module_paths=raw.get("module_paths"),
        safety_case_ids=raw.get("safety_case_ids"),
        proof_assumptions=raw.get("proof_assumptions"),
        assumption_sha256=raw.get("assumption_sha256"),
    )


def _formal_verification_dict(evidence: FormalVerificationEvidence) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "required": evidence.required,
        "status": evidence.status,
        "backend": evidence.backend,
        "solver": evidence.solver,
        "max_depth": evidence.max_depth,
        "checked_specs": evidence.checked_specs,
        "artifact_sha256": evidence.artifact_sha256,
        "report_sha256": evidence.report_sha256,
        "claim_boundary": evidence.claim_boundary,
    }
    if evidence.report_uri is not None:
        obj["report_uri"] = evidence.report_uri
    if evidence.generated_utc is not None:
        obj["generated_utc"] = evidence.generated_utc
    if evidence.counterexample_path is not None:
        obj["counterexample_path"] = evidence.counterexample_path
    if evidence.counterexample_property is not None:
        obj["counterexample_property"] = evidence.counterexample_property
    if evidence.lean_version is not None:
        obj["lean_version"] = evidence.lean_version
    if evidence.lakefile_sha256 is not None:
        obj["lakefile_sha256"] = evidence.lakefile_sha256
    if evidence.proof_source_sha256 is not None:
        obj["proof_source_sha256"] = evidence.proof_source_sha256
    if evidence.theorem_names is not None:
        obj["theorem_names"] = evidence.theorem_names
    if evidence.theorem_modules is not None:
        obj["theorem_modules"] = evidence.theorem_modules
    if evidence.proved_contracts is not None:
        obj["proved_contracts"] = evidence.proved_contracts
    if evidence.module_paths is not None:
        obj["module_paths"] = evidence.module_paths
    if evidence.safety_case_ids is not None:
        obj["safety_case_ids"] = evidence.safety_case_ids
    if evidence.proof_assumptions is not None:
        obj["proof_assumptions"] = evidence.proof_assumptions
    if evidence.assumption_sha256 is not None:
        obj["assumption_sha256"] = evidence.assumption_sha256
    return obj


def _artifact_payload_dict(
    artifact: Artifact,
    compact_packed: bool = False,
) -> Dict[str, Any]:
    """Return canonical artifact payload sections excluding proof evidence."""

    def _weight_matrix_dict(wm: WeightMatrix) -> Dict[str, Any]:
        return {"shape": wm.shape, "data": wm.data}

    packed_dict: Dict[str, Any] | None = None
    if artifact.weights.packed is not None:
        pg = artifact.weights.packed
        if compact_packed:
            pw_in_d = {"shape": pg.w_in_packed.shape}
            pw_in_d.update(_encode_u64_compact(pg.w_in_packed.data_u64))
        else:
            pw_in_d = {"shape": pg.w_in_packed.shape, "data_u64": pg.w_in_packed.data_u64}
        pw_out_d = None
        if pg.w_out_packed is not None:
            if compact_packed:
                pw_out_d = {"shape": pg.w_out_packed.shape}
                pw_out_d.update(_encode_u64_compact(pg.w_out_packed.data_u64))
            else:
                pw_out_d = {
                    "shape": pg.w_out_packed.shape,
                    "data_u64": pg.w_out_packed.data_u64,
                }
        packed_dict = {
            "words_per_stream": pg.words_per_stream,
            "w_in_packed": pw_in_d,
        }
        if pw_out_d is not None:
            packed_dict["w_out_packed"] = pw_out_d

    obj: Dict[str, Any] = {
        "meta": {
            "artifact_version": artifact.meta.artifact_version,
            "name": artifact.meta.name,
            "dt_control_s": artifact.meta.dt_control_s,
            "stream_length": artifact.meta.stream_length,
            "fixed_point": {
                "data_width": artifact.meta.fixed_point.data_width,
                "fraction_bits": artifact.meta.fixed_point.fraction_bits,
                "signed": artifact.meta.fixed_point.signed,
            },
            "firing_mode": artifact.meta.firing_mode,
            "firing_margin": artifact.meta.firing_margin,
            "seed_policy": {
                "id": artifact.meta.seed_policy.id,
                "hash_fn": artifact.meta.seed_policy.hash_fn,
                "rng_family": artifact.meta.seed_policy.rng_family,
            },
            "created_utc": artifact.meta.created_utc,
            "compiler": {
                "name": artifact.meta.compiler.name,
                "version": artifact.meta.compiler.version,
                "git_sha": artifact.meta.compiler.git_sha,
            },
        },
        "topology": {
            "places": [{"id": p.id, "name": p.name} for p in artifact.topology.places],
            "transitions": [
                {
                    "id": t.id,
                    "name": t.name,
                    "threshold": t.threshold,
                    **({"margin": t.margin} if t.margin is not None else {}),
                    "delay_ticks": int(t.delay_ticks),
                }
                for t in artifact.topology.transitions
            ],
        },
        "weights": {
            "w_in": _weight_matrix_dict(artifact.weights.w_in),
            "w_out": _weight_matrix_dict(artifact.weights.w_out),
        },
        "readout": {
            "actions": [
                {
                    "id": a.id,
                    "name": a.name,
                    "pos_place": a.pos_place,
                    "neg_place": a.neg_place,
                }
                for a in artifact.readout.actions
            ],
            "gains": {"per_action": artifact.readout.gains},
            "limits": {
                "per_action_abs_max": artifact.readout.abs_max,
                "slew_per_s": artifact.readout.slew_per_s,
            },
        },
        "initial_state": {
            "marking": artifact.initial_state.marking,
            "place_injections": [
                {
                    "place_id": inj.place_id,
                    "source": inj.source,
                    "scale": inj.scale,
                    "offset": inj.offset,
                    "clamp_0_1": inj.clamp_0_1,
                }
                for inj in artifact.initial_state.place_injections
            ],
        },
    }

    if artifact.meta.notes is not None:
        obj["meta"]["notes"] = artifact.meta.notes

    if packed_dict is not None:
        obj["weights"]["packed"] = packed_dict

    return obj


def compute_artifact_payload_sha256(artifact: Artifact) -> str:
    """Compute a canonical SHA-256 digest over artifact payload sections."""
    payload = json.dumps(
        _artifact_payload_dict(artifact),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ArtifactValidationError(f"formal_verification.{field_name} must be a non-empty list")
    result: List[str] = []
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
        from scpn_control.scpn.z3_model_checking import load_z3_formal_report

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


def _encode_u64_compact(data_u64: List[int]) -> Dict[str, Any]:
    """Encode uint64 list as zlib-compressed base64 little-endian payload."""
    raw = bytearray()
    for value in data_u64:
        raw.extend(int(value & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False))
    compressed = zlib.compress(bytes(raw), level=9)
    payload = base64.b64encode(compressed).decode("ascii")
    return {
        "encoding": "u64-le-zlib-base64",
        "count": len(data_u64),
        "data_u64_b64_zlib": payload,
    }


def _decode_u64_compact(encoded: Dict[str, Any]) -> List[int]:
    """Decode compact uint64 payload generated by ``_encode_u64_compact``."""
    if encoded.get("encoding") != "u64-le-zlib-base64":
        raise ArtifactValidationError(f"Unsupported packed encoding: {encoded.get('encoding')}")

    payload = encoded.get("data_u64_b64_zlib")
    if not isinstance(payload, str):
        raise ArtifactValidationError("Missing compact packed payload string.")

    count_val = encoded.get("count")
    if isinstance(count_val, int) and (count_val < 0 or count_val > MAX_PACKED_WORDS):
        raise ArtifactValidationError(f"Packed count {count_val} exceeds limit {MAX_PACKED_WORDS}.")

    try:
        comp = base64.b64decode(payload.encode("ascii"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ArtifactValidationError(f"Invalid base64 payload: {exc}") from exc

    if len(comp) > MAX_COMPRESSED_BYTES:
        raise ArtifactValidationError(f"Compressed payload too large: {len(comp)} bytes.")

    try:
        decomp = zlib.decompressobj()
        raw = decomp.decompress(comp, MAX_DECOMPRESSED_BYTES + 1)
        if decomp.unconsumed_tail:
            raise ArtifactValidationError("Decompressed packed payload exceeds configured limit.")
        raw += decomp.flush()
    except ArtifactValidationError:
        raise
    except (zlib.error, ValueError, OSError) as exc:
        raise ArtifactValidationError(f"Invalid compact packed payload: {exc}") from exc

    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise ArtifactValidationError("Decompressed packed payload exceeds configured limit.")

    if len(raw) % 8 != 0:
        raise ArtifactValidationError(f"Compact packed payload byte-length {len(raw)} is not divisible by 8.")

    available = len(raw) // 8
    if isinstance(count_val, int):
        count = count_val
    elif count_val is None:
        count = available
    else:
        raise ArtifactValidationError(f"Invalid compact packed count type: {type(count_val).__name__}.")
    if count < 0 or count > available:
        raise ArtifactValidationError(f"Invalid compact packed count {count}; available words={available}.")

    return [int.from_bytes(raw[i * 8 : (i + 1) * 8], "little", signed=False) for i in range(count)]


def encode_u64_compact(data_u64: List[int]) -> Dict[str, Any]:
    """Public compact codec helper for deterministic uint64 payload encoding."""
    return _encode_u64_compact(list(map(int, data_u64)))


def decode_u64_compact(encoded: Dict[str, Any]) -> List[int]:
    """Public compact codec helper for deterministic uint64 payload decoding."""
    return _decode_u64_compact(encoded)


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


# ── Load / Save ─────────────────────────────────────────────────────────────


def _parse_dense_weight_data(raw: object, field_name: str) -> List[float]:
    """Parse one dense artifact weight array without silent type coercion."""

    if not isinstance(raw, list):
        raise ArtifactValidationError(f"{field_name}.data must be a list")

    parsed: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ArtifactValidationError(f"{field_name} weights must be finite numeric values in [0, 1]")
        parsed.append(float(value))
    return parsed


def load_artifact(
    path: str | Path,
    require_formal_verification: bool = False,
    formal_report_root: str | Path | None = None,
) -> Artifact:
    """Parse a ``.scpnctl.json`` file into an ``Artifact`` dataclass."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Meta
    m = obj["meta"]
    meta = ArtifactMeta(
        artifact_version=m["artifact_version"],
        name=m["name"],
        dt_control_s=m["dt_control_s"],
        stream_length=m["stream_length"],
        fixed_point=FixedPoint(
            data_width=m["fixed_point"]["data_width"],
            fraction_bits=m["fixed_point"]["fraction_bits"],
            signed=m["fixed_point"]["signed"],
        ),
        firing_mode=m["firing_mode"],
        firing_margin=m.get("firing_margin", 0.05),
        seed_policy=SeedPolicy(
            id=m["seed_policy"]["id"],
            hash_fn=m["seed_policy"]["hash_fn"],
            rng_family=m["seed_policy"]["rng_family"],
        ),
        created_utc=m["created_utc"],
        compiler=CompilerInfo(
            name=m["compiler"]["name"],
            version=m["compiler"]["version"],
            git_sha=m["compiler"]["git_sha"],
        ),
        notes=m.get("notes"),
    )

    # Topology
    places = [PlaceSpec(id=p["id"], name=p["name"]) for p in obj["topology"]["places"]]
    transitions = [
        TransitionSpec(
            id=t["id"],
            name=t["name"],
            threshold=t["threshold"],
            margin=t.get("margin"),
            delay_ticks=t.get("delay_ticks", 0),
        )
        for t in obj["topology"]["transitions"]
    ]
    topology = Topology(places=places, transitions=transitions)

    # Weights
    w_in = WeightMatrix(
        shape=obj["weights"]["w_in"]["shape"],
        data=_parse_dense_weight_data(obj["weights"]["w_in"]["data"], "w_in"),
    )
    w_out = WeightMatrix(
        shape=obj["weights"]["w_out"]["shape"],
        data=_parse_dense_weight_data(obj["weights"]["w_out"]["data"], "w_out"),
    )
    packed = None
    if "packed" in obj["weights"]:
        pw = obj["weights"]["packed"]
        w_in_obj = pw["w_in_packed"]
        if "data_u64" in w_in_obj:
            w_in_data = list(map(int, w_in_obj["data_u64"]))
        else:
            w_in_data = _decode_u64_compact(w_in_obj)
        pw_in = PackedWeights(
            shape=w_in_obj["shape"],
            data_u64=w_in_data,
        )
        pw_out = None
        if "w_out_packed" in pw:
            w_out_obj = pw["w_out_packed"]
            if "data_u64" in w_out_obj:
                w_out_data = list(map(int, w_out_obj["data_u64"]))
            else:
                w_out_data = _decode_u64_compact(w_out_obj)
            pw_out = PackedWeights(
                shape=w_out_obj["shape"],
                data_u64=w_out_data,
            )
        packed = PackedWeightsGroup(
            words_per_stream=int(pw["words_per_stream"]),
            w_in_packed=pw_in,
            w_out_packed=pw_out,
        )
    weights = Weights(w_in=w_in, w_out=w_out, packed=packed)

    # Readout
    actions = [
        ActionReadout(
            id=a["id"],
            name=a["name"],
            pos_place=a["pos_place"],
            neg_place=a["neg_place"],
        )
        for a in obj["readout"]["actions"]
    ]
    readout = Readout(
        actions=actions,
        gains=obj["readout"]["gains"]["per_action"],
        abs_max=obj["readout"]["limits"]["per_action_abs_max"],
        slew_per_s=obj["readout"]["limits"]["slew_per_s"],
    )

    # Initial state
    injections = [
        PlaceInjection(
            place_id=inj["place_id"],
            source=inj["source"],
            scale=inj["scale"],
            offset=inj["offset"],
            clamp_0_1=inj["clamp_0_1"],
        )
        for inj in obj["initial_state"]["place_injections"]
    ]
    initial_state = InitialState(
        marking=list(map(float, obj["initial_state"]["marking"])),
        place_injections=injections,
    )

    artifact = Artifact(
        meta=meta,
        topology=topology,
        weights=weights,
        readout=readout,
        initial_state=initial_state,
        formal_verification=_parse_formal_verification(obj.get("formal_verification")),
        validate_on_init=False,
    )
    validate_artifact(artifact)
    if require_formal_verification:
        validate_safety_critical_artifact(artifact, formal_report_root=formal_report_root)
    return artifact


def _object_schema(
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a closed JSON object schema with deterministic required order."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


def _array_schema(
    items: dict[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Return a JSON array schema with optional cardinality bounds."""

    schema: dict[str, Any] = {"type": "array", "items": items}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _non_empty_string_schema(**extras: Any) -> dict[str, Any]:
    """Return a string schema that rejects empty values."""

    schema: dict[str, Any] = {"type": "string", "minLength": 1}
    schema.update(extras)
    return schema


def _non_empty_string_array_schema(**extras: Any) -> dict[str, Any]:
    """Return a non-empty array schema for string-list validation fields."""

    return _array_schema(_non_empty_string_schema(**extras), min_items=1)


def _packed_weight_schema() -> dict[str, Any]:
    """Return the raw-or-compact packed-weight payload schema."""

    shape_schema = _array_schema({"type": "integer", "minimum": 0}, min_items=3, max_items=3)
    raw_schema = _object_schema(
        {
            "shape": shape_schema,
            "data_u64": _array_schema({"type": "integer", "minimum": 0, "maximum": 2**64 - 1}),
        },
        PACKED_WEIGHT_REQUIRED_RAW_FIELDS,
    )
    compact_schema = _object_schema(
        {
            "shape": shape_schema,
            "encoding": {"type": "string", "const": "u64-le-zlib-base64"},
            "count": {"type": "integer", "minimum": 0, "maximum": MAX_PACKED_WORDS},
            "data_u64_b64_zlib": _non_empty_string_schema(),
        },
        PACKED_WEIGHT_REQUIRED_COMPACT_FIELDS,
    )
    return {"oneOf": [raw_schema, compact_schema]}


def _formal_verification_schema() -> dict[str, Any]:
    """Return the proof-manifest schema derived from evidence dataclass fields."""

    properties: dict[str, Any] = {
        "required": {"type": "boolean"},
        "status": {"type": "string", "enum": ["pass", "fail", "blocked"]},
        "backend": {"type": "string", "enum": sorted(FORMAL_VERIFICATION_BACKENDS)},
        "solver": _non_empty_string_schema(),
        "max_depth": {"type": "integer", "minimum": 0},
        "checked_specs": _non_empty_string_array_schema(),
        "artifact_sha256": _non_empty_string_schema(pattern=SHA256_HEX_PATTERN),
        "report_sha256": _non_empty_string_schema(pattern=SHA256_HEX_PATTERN),
        "claim_boundary": _non_empty_string_schema(),
        "report_uri": _non_empty_string_schema(pattern=SAFE_RELATIVE_PATH_PATTERN),
        "generated_utc": _non_empty_string_schema(),
        "counterexample_path": _non_empty_string_array_schema(),
        "counterexample_property": _non_empty_string_schema(),
        "lean_version": _non_empty_string_schema(),
        "lakefile_sha256": _non_empty_string_schema(pattern=SHA256_HEX_PATTERN),
        "proof_source_sha256": _non_empty_string_schema(pattern=SHA256_HEX_PATTERN),
        "theorem_names": _non_empty_string_array_schema(),
        "theorem_modules": _non_empty_string_array_schema(),
        "proved_contracts": _non_empty_string_array_schema(),
        "module_paths": _non_empty_string_array_schema(pattern=SAFE_RELATIVE_PATH_PATTERN),
        "safety_case_ids": _non_empty_string_array_schema(),
        "proof_assumptions": _non_empty_string_array_schema(),
        "assumption_sha256": _non_empty_string_schema(pattern=SHA256_HEX_PATTERN),
    }
    if set(properties) != FORMAL_VERIFICATION_ALLOWED_FIELDS:
        missing = sorted(FORMAL_VERIFICATION_ALLOWED_FIELDS.difference(properties))
        extra = sorted(set(properties).difference(FORMAL_VERIFICATION_ALLOWED_FIELDS))
        raise RuntimeError(f"formal_verification schema drift: missing={missing}, extra={extra}")
    return _object_schema(properties, FORMAL_VERIFICATION_REQUIRED_FIELDS)


def get_artifact_json_schema() -> Dict[str, Any]:
    """Return the JSON schema for current ``.scpnctl.json`` artifact payloads.

    Returns
    -------
    dict[str, Any]
        Draft-07 JSON schema generated from the same dataclass fields,
        serializer sections, packed-weight codec variants, and formal-proof
        evidence field set admitted by ``load_artifact()`` and
        ``save_artifact()``.
    """

    non_negative_unit_interval = {"type": "number", "minimum": 0, "maximum": 1}
    non_negative_number = {"type": "number", "minimum": 0}
    fixed_point_schema = _object_schema(
        {
            "data_width": {"type": "integer", "minimum": 1},
            "fraction_bits": {"type": "integer", "minimum": 0},
            "signed": {"type": "boolean"},
        },
        FIXED_POINT_REQUIRED_FIELDS,
    )
    seed_policy_schema = _object_schema(
        {
            "id": _non_empty_string_schema(),
            "hash_fn": _non_empty_string_schema(),
            "rng_family": _non_empty_string_schema(),
        },
        SEED_POLICY_REQUIRED_FIELDS,
    )
    compiler_schema = _object_schema(
        {
            "name": _non_empty_string_schema(),
            "version": _non_empty_string_schema(),
            "git_sha": _non_empty_string_schema(),
        },
        COMPILER_REQUIRED_FIELDS,
    )
    place_schema = _object_schema(
        {"id": {"type": "integer", "minimum": 0}, "name": _non_empty_string_schema()},
        PLACE_SPEC_REQUIRED_FIELDS,
    )
    transition_schema = _object_schema(
        {
            "id": {"type": "integer", "minimum": 0},
            "name": _non_empty_string_schema(),
            "threshold": non_negative_unit_interval,
            "margin": non_negative_number,
            "delay_ticks": {"type": "integer", "minimum": 0},
        },
        TRANSITION_SPEC_REQUIRED_FIELDS,
    )
    weight_matrix_schema = _object_schema(
        {
            "shape": _array_schema({"type": "integer", "minimum": 0}, min_items=2, max_items=2),
            "data": _array_schema(non_negative_unit_interval),
        },
        WEIGHT_MATRIX_REQUIRED_FIELDS,
    )
    packed_group_schema = _object_schema(
        {
            "words_per_stream": {"type": "integer", "minimum": 1},
            "w_in_packed": {"$ref": "#/definitions/packed_weight"},
            "w_out_packed": {"$ref": "#/definitions/packed_weight"},
        },
        PACKED_WEIGHTS_GROUP_REQUIRED_FIELDS,
    )
    action_schema = _object_schema(
        {
            "id": {"type": "integer", "minimum": 0},
            "name": _non_empty_string_schema(),
            "pos_place": {"type": "integer", "minimum": 0},
            "neg_place": {"type": "integer", "minimum": 0},
        },
        ACTION_READOUT_REQUIRED_FIELDS,
    )
    place_injection_schema = _object_schema(
        {
            "place_id": {"type": "integer", "minimum": 0},
            "source": _non_empty_string_schema(),
            "scale": {"type": "number"},
            "offset": {"type": "number"},
            "clamp_0_1": {"type": "boolean"},
        },
        PLACE_INJECTION_REQUIRED_FIELDS,
    )
    formal_verification_schema = _formal_verification_schema()
    schema: dict[str, Any] = _object_schema(
        {
            "meta": _object_schema(
                {
                    "artifact_version": {"type": "string", "const": ARTIFACT_SCHEMA_VERSION},
                    "name": _non_empty_string_schema(),
                    "dt_control_s": {"type": "number", "exclusiveMinimum": 0},
                    "stream_length": {"type": "integer", "minimum": 1},
                    "fixed_point": fixed_point_schema,
                    "firing_mode": {"type": "string", "enum": ["binary", "fractional"]},
                    "firing_margin": non_negative_number,
                    "seed_policy": seed_policy_schema,
                    "created_utc": _non_empty_string_schema(),
                    "compiler": compiler_schema,
                    "notes": {"type": "string"},
                },
                ARTIFACT_META_REQUIRED_FIELDS,
            ),
            "topology": _object_schema(
                {
                    "places": _array_schema(place_schema),
                    "transitions": _array_schema(transition_schema),
                },
                ("places", "transitions"),
            ),
            "weights": _object_schema(
                {
                    "w_in": {"$ref": "#/definitions/weight_matrix"},
                    "w_out": {"$ref": "#/definitions/weight_matrix"},
                    "packed": packed_group_schema,
                },
                ("w_in", "w_out"),
            ),
            "readout": _object_schema(
                {
                    "actions": _array_schema(action_schema),
                    "gains": _object_schema(
                        {"per_action": _array_schema({"type": "number"})},
                        ("per_action",),
                    ),
                    "limits": _object_schema(
                        {
                            "per_action_abs_max": _array_schema(non_negative_number),
                            "slew_per_s": _array_schema(non_negative_number),
                        },
                        ("per_action_abs_max", "slew_per_s"),
                    ),
                },
                ("actions", "gains", "limits"),
            ),
            "initial_state": _object_schema(
                {
                    "marking": _array_schema(non_negative_unit_interval),
                    "place_injections": _array_schema(place_injection_schema),
                },
                INITIAL_STATE_REQUIRED_FIELDS,
            ),
            "formal_verification": formal_verification_schema,
        },
        ARTIFACT_PAYLOAD_REQUIRED_SECTIONS,
    )
    schema.update(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "SCPN Controller Artifact",
            "definitions": {
                "weight_matrix": weight_matrix_schema,
                "packed_weight": _packed_weight_schema(),
                "formal_verification": formal_verification_schema,
            },
        }
    )
    return schema


def save_artifact(
    artifact: Artifact,
    path: str | Path,
    compact_packed: bool = False,
) -> None:
    """Serialize an ``Artifact`` to indented JSON."""
    validate_artifact(artifact)
    obj = _artifact_payload_dict(artifact, compact_packed=compact_packed)

    if artifact.formal_verification is not None:
        _validate_formal_verification(artifact.formal_verification)
        obj["formal_verification"] = _formal_verification_dict(artifact.formal_verification)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
