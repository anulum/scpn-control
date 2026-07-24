# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact load / save / schema / codec product surface.
"""
SCPN Controller Artifact (``.scpnctl.json``) loader / saver.

Re-exports the topology model and validation leaf, and owns load/save, JSON
schema emission, compact packed-weight codec, and payload hashing. Structural
validation and safety-critical admit live in
:mod:`scpn_control.scpn.artifact_validate` (CTL-G07 R4-S2).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import zlib
from pathlib import Path
from typing import Any, Dict, List

from scpn_control.scpn.artifact_model import (
    ACTION_READOUT_REQUIRED_FIELDS,
    ARTIFACT_META_REQUIRED_FIELDS,
    ARTIFACT_PAYLOAD_REQUIRED_SECTIONS,
    ARTIFACT_SCHEMA_VERSION,
    COMPILER_REQUIRED_FIELDS,
    FIXED_POINT_REQUIRED_FIELDS,
    FORMAL_VERIFICATION_ALLOWED_FIELDS,
    FORMAL_VERIFICATION_BACKENDS,
    FORMAL_VERIFICATION_REQUIRED_FIELDS,
    INITIAL_STATE_REQUIRED_FIELDS,
    MAX_COMPRESSED_BYTES,
    MAX_DECOMPRESSED_BYTES,
    MAX_PACKED_WORDS,
    PACKED_WEIGHT_REQUIRED_COMPACT_FIELDS,
    PACKED_WEIGHT_REQUIRED_RAW_FIELDS,
    PACKED_WEIGHTS_GROUP_REQUIRED_FIELDS,
    PLACE_INJECTION_REQUIRED_FIELDS,
    PLACE_SPEC_REQUIRED_FIELDS,
    SAFE_RELATIVE_PATH_PATTERN,
    SEED_POLICY_REQUIRED_FIELDS,
    SHA256_HEX_PATTERN,
    TRANSITION_SPEC_REQUIRED_FIELDS,
    WEIGHT_MATRIX_REQUIRED_FIELDS,
    ActionReadout,
    Artifact,
    ArtifactMeta,
    CompilerInfo,
    FixedPoint,
    FormalVerificationEvidence,
    InitialState,
    PackedWeights,
    PackedWeightsGroup,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
)
from scpn_control.scpn.artifact_validate import (
    ArtifactValidationError as ArtifactValidationError,
)
from scpn_control.scpn.artifact_validate import (
    _formal_report_relative_path as _formal_report_relative_path,
)
from scpn_control.scpn.artifact_validate import (
    _validate as _validate,
)
from scpn_control.scpn.artifact_validate import (
    _validate_formal_verification as _validate_formal_verification,
)
from scpn_control.scpn.artifact_validate import (
    _verify_formal_report_digest as _verify_formal_report_digest,
)
from scpn_control.scpn.artifact_validate import (
    validate_artifact as validate_artifact,
)
from scpn_control.scpn.artifact_validate import (
    validate_safety_critical_artifact as validate_safety_critical_artifact,
)

# ── Validation (re-exported from artifact_validate; CTL-G07 R4-S2) ──────────


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
