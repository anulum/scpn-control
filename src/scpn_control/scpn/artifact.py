# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact
"""
SCPN Controller Artifact (``.scpnctl.json``) loader / saver.

Defines the ``Artifact`` dataclass that mirrors the JSON schema sections
(meta, topology, weights, readout, initial_state) and provides lightweight
validation on load.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

ARTIFACT_SCHEMA_VERSION = "1.0.0"
MAX_PACKED_WORDS = 10_000_000
MAX_DECOMPRESSED_BYTES = MAX_PACKED_WORDS * 8
MAX_COMPRESSED_BYTES = 50_000_000
FORMAL_VERIFICATION_BACKENDS = {"explicit-state", "z3"}


# ── Sub-structures ──────────────────────────────────────────────────────────


@dataclass
class FixedPoint:
    data_width: int
    fraction_bits: int
    signed: bool


@dataclass
class SeedPolicy:
    id: str
    hash_fn: str
    rng_family: str


@dataclass
class CompilerInfo:
    name: str
    version: str
    git_sha: str


@dataclass
class ArtifactMeta:
    artifact_version: str
    name: str
    dt_control_s: float
    stream_length: int
    fixed_point: FixedPoint
    firing_mode: str
    seed_policy: SeedPolicy
    created_utc: str
    compiler: CompilerInfo
    notes: str | None = None


@dataclass
class PlaceSpec:
    id: int
    name: str


@dataclass
class TransitionSpec:
    id: int
    name: str
    threshold: float
    margin: float | None = None
    delay_ticks: int = 0


@dataclass
class Topology:
    places: List[PlaceSpec]
    transitions: List[TransitionSpec]


@dataclass
class WeightMatrix:
    shape: List[int]  # [rows, cols]
    data: List[float]  # row-major


@dataclass
class PackedWeights:
    shape: List[int]  # [rows, cols, words]
    data_u64: List[int]


@dataclass
class PackedWeightsGroup:
    words_per_stream: int
    w_in_packed: PackedWeights
    w_out_packed: PackedWeights | None = None


@dataclass
class Weights:
    w_in: WeightMatrix
    w_out: WeightMatrix
    packed: PackedWeightsGroup | None = None


@dataclass
class ActionReadout:
    id: int
    name: str
    pos_place: int
    neg_place: int


@dataclass
class Readout:
    actions: List[ActionReadout]
    gains: List[float]
    abs_max: List[float]
    slew_per_s: List[float]


@dataclass
class PlaceInjection:
    place_id: int
    source: str
    scale: float
    offset: float
    clamp_0_1: bool


@dataclass
class InitialState:
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

    @property
    def nP(self) -> int:
        return len(self.topology.places)

    @property
    def nT(self) -> int:
        return len(self.topology.transitions)


# ── Validation ──────────────────────────────────────────────────────────────


class ArtifactValidationError(ValueError):
    """Raised when an artifact fails lightweight validation."""


def _parse_formal_verification(raw: Any) -> FormalVerificationEvidence | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ArtifactValidationError("formal_verification must be an object")
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
    actual = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if actual != evidence.report_sha256.lower():
        raise ArtifactValidationError("formal_verification.report_sha256 does not match report file")


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
        if not (-1.0 <= val <= 1.0):
            raise ArtifactValidationError(f"w_in weight {val} outside [-1, 1]")
    for val in artifact.weights.w_out.data:
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


# ── Load / Save ─────────────────────────────────────────────────────────────


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
        data=list(map(float, obj["weights"]["w_in"]["data"])),
    )
    w_out = WeightMatrix(
        shape=obj["weights"]["w_out"]["shape"],
        data=list(map(float, obj["weights"]["w_out"]["data"])),
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
    )
    _validate(artifact)
    if require_formal_verification:
        validate_safety_critical_artifact(artifact, formal_report_root=formal_report_root)
    return artifact


def get_artifact_json_schema() -> Dict[str, Any]:
    """Return the formal JSON schema for .scpnctl.json artifacts.

    Used by downstream tools to validate SNN compilation results before
    deployment to the sc-neurocore FPGA backend.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SCPN Controller Artifact",
        "type": "object",
        "required": ["meta", "topology", "weights", "readout", "initial_state"],
        "properties": {
            "meta": {
                "type": "object",
                "required": ["artifact_version", "name", "stream_length"],
                "properties": {
                    "artifact_version": {"type": "string"},
                    "name": {"type": "string"},
                    "dt_control_s": {"type": "number"},
                    "stream_length": {"type": "integer"},
                    "fixed_point": {
                        "type": "object",
                        "properties": {
                            "data_width": {"type": "integer"},
                            "fraction_bits": {"type": "integer"},
                            "signed": {"type": "boolean"},
                        },
                    },
                },
            },
            "topology": {
                "type": "object",
                "properties": {
                    "places": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "transitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "threshold": {"type": "number"},
                            },
                        },
                    },
                },
            },
            "weights": {
                "type": "object",
                "properties": {
                    "w_in": {"$ref": "#/definitions/weight_matrix"},
                    "w_out": {"$ref": "#/definitions/weight_matrix"},
                    "packed": {
                        "type": "object",
                        "properties": {
                            "shape": {"type": "array", "items": {"type": "integer"}},
                            "data_b64": {"type": "string"},
                        },
                    },
                },
            },
            "readout": {
                "type": "object",
                "properties": {
                    "actions": {"type": "array"},
                    "gains": {"type": "object"},
                    "limits": {"type": "object"},
                },
            },
            "initial_state": {
                "type": "object",
                "properties": {
                    "marking": {"type": "array", "items": {"type": "number"}},
                    "place_injections": {"type": "array"},
                },
            },
            "formal_verification": {
                "type": "object",
                "required": [
                    "required",
                    "status",
                    "backend",
                    "solver",
                    "max_depth",
                    "checked_specs",
                    "artifact_sha256",
                    "report_sha256",
                    "claim_boundary",
                ],
                "properties": {
                    "required": {"type": "boolean"},
                    "status": {"type": "string", "enum": ["pass", "fail", "blocked"]},
                    "backend": {"type": "string"},
                    "solver": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 0},
                    "checked_specs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "artifact_sha256": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
                    "report_sha256": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
                    "claim_boundary": {"type": "string"},
                    "report_uri": {"type": "string"},
                    "generated_utc": {"type": "string"},
                    "counterexample_path": {"type": "array", "items": {"type": "string"}},
                    "counterexample_property": {"type": "string"},
                },
            },
        },
        "definitions": {
            "weight_matrix": {
                "type": "object",
                "properties": {
                    "shape": {"type": "array", "items": {"type": "integer"}},
                    "data": {"type": "array", "items": {"type": "number"}},
                },
            }
        },
    }


def save_artifact(
    artifact: Artifact,
    path: str | Path,
    compact_packed: bool = False,
) -> None:
    """Serialize an ``Artifact`` to indented JSON."""
    obj = _artifact_payload_dict(artifact, compact_packed=compact_packed)

    if artifact.formal_verification is not None:
        _validate_formal_verification(artifact.formal_verification)
        obj["formal_verification"] = _formal_verification_dict(artifact.formal_verification)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
