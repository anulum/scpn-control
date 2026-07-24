# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact JSON schema product surface.
"""
SCPN Controller Artifact (``.scpnctl.json``) product surface.

Re-exports topology model, validation, compact codec, and load/save IO leaves.
Owns JSON schema emission for ``.scpnctl.json``. CTL-G07 R4-S4.
"""

from __future__ import annotations

from typing import Any, Dict

from scpn_control.scpn.artifact_codec import (
    _decode_u64_compact as _decode_u64_compact,
)
from scpn_control.scpn.artifact_codec import (
    _encode_u64_compact as _encode_u64_compact,
)
from scpn_control.scpn.artifact_codec import (
    decode_u64_compact as decode_u64_compact,
)
from scpn_control.scpn.artifact_codec import (
    encode_u64_compact as encode_u64_compact,
)
from scpn_control.scpn.artifact_io import (
    _artifact_payload_dict as _artifact_payload_dict,
)
from scpn_control.scpn.artifact_io import (
    _formal_verification_dict as _formal_verification_dict,
)
from scpn_control.scpn.artifact_io import (
    _parse_dense_weight_data as _parse_dense_weight_data,
)
from scpn_control.scpn.artifact_io import (
    _parse_formal_verification as _parse_formal_verification,
)
from scpn_control.scpn.artifact_io import (
    compute_artifact_payload_sha256 as compute_artifact_payload_sha256,
)
from scpn_control.scpn.artifact_io import (
    load_artifact as load_artifact,
)
from scpn_control.scpn.artifact_io import (
    save_artifact as save_artifact,
)
from scpn_control.scpn.artifact_model import (
    ACTION_READOUT_REQUIRED_FIELDS as ACTION_READOUT_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    ARTIFACT_META_REQUIRED_FIELDS as ARTIFACT_META_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    ARTIFACT_PAYLOAD_REQUIRED_SECTIONS as ARTIFACT_PAYLOAD_REQUIRED_SECTIONS,
)
from scpn_control.scpn.artifact_model import (
    ARTIFACT_SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
)
from scpn_control.scpn.artifact_model import (
    COMPILER_REQUIRED_FIELDS as COMPILER_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    FIXED_POINT_REQUIRED_FIELDS as FIXED_POINT_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    FORMAL_VERIFICATION_ALLOWED_FIELDS as FORMAL_VERIFICATION_ALLOWED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    FORMAL_VERIFICATION_BACKENDS as FORMAL_VERIFICATION_BACKENDS,
)
from scpn_control.scpn.artifact_model import (
    FORMAL_VERIFICATION_REQUIRED_FIELDS as FORMAL_VERIFICATION_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    INITIAL_STATE_REQUIRED_FIELDS as INITIAL_STATE_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    MAX_COMPRESSED_BYTES as MAX_COMPRESSED_BYTES,
)
from scpn_control.scpn.artifact_model import (
    MAX_DECOMPRESSED_BYTES as MAX_DECOMPRESSED_BYTES,
)
from scpn_control.scpn.artifact_model import (
    MAX_PACKED_WORDS as MAX_PACKED_WORDS,
)
from scpn_control.scpn.artifact_model import (
    PACKED_WEIGHT_REQUIRED_COMPACT_FIELDS as PACKED_WEIGHT_REQUIRED_COMPACT_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    PACKED_WEIGHT_REQUIRED_RAW_FIELDS as PACKED_WEIGHT_REQUIRED_RAW_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    PACKED_WEIGHTS_GROUP_REQUIRED_FIELDS as PACKED_WEIGHTS_GROUP_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    PLACE_INJECTION_REQUIRED_FIELDS as PLACE_INJECTION_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    PLACE_SPEC_REQUIRED_FIELDS as PLACE_SPEC_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    SAFE_RELATIVE_PATH_PATTERN as SAFE_RELATIVE_PATH_PATTERN,
)
from scpn_control.scpn.artifact_model import (
    SEED_POLICY_REQUIRED_FIELDS as SEED_POLICY_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    SHA256_HEX_PATTERN as SHA256_HEX_PATTERN,
)
from scpn_control.scpn.artifact_model import (
    TRANSITION_SPEC_REQUIRED_FIELDS as TRANSITION_SPEC_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    WEIGHT_MATRIX_REQUIRED_FIELDS as WEIGHT_MATRIX_REQUIRED_FIELDS,
)
from scpn_control.scpn.artifact_model import (
    ActionReadout as ActionReadout,
)
from scpn_control.scpn.artifact_model import (
    Artifact as Artifact,
)
from scpn_control.scpn.artifact_model import (
    ArtifactMeta as ArtifactMeta,
)
from scpn_control.scpn.artifact_model import (
    CompilerInfo as CompilerInfo,
)
from scpn_control.scpn.artifact_model import (
    FixedPoint as FixedPoint,
)
from scpn_control.scpn.artifact_model import (
    FormalVerificationEvidence as FormalVerificationEvidence,
)
from scpn_control.scpn.artifact_model import (
    InitialState as InitialState,
)
from scpn_control.scpn.artifact_model import (
    PackedWeights as PackedWeights,
)
from scpn_control.scpn.artifact_model import (
    PackedWeightsGroup as PackedWeightsGroup,
)
from scpn_control.scpn.artifact_model import (
    PlaceInjection as PlaceInjection,
)
from scpn_control.scpn.artifact_model import (
    PlaceSpec as PlaceSpec,
)
from scpn_control.scpn.artifact_model import (
    Readout as Readout,
)
from scpn_control.scpn.artifact_model import (
    SeedPolicy as SeedPolicy,
)
from scpn_control.scpn.artifact_model import (
    Topology as Topology,
)
from scpn_control.scpn.artifact_model import (
    TransitionSpec as TransitionSpec,
)
from scpn_control.scpn.artifact_model import (
    WeightMatrix as WeightMatrix,
)
from scpn_control.scpn.artifact_model import (
    Weights as Weights,
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

# ── Re-exports: model / validate (R4-S1/S2) / codec (R4-S3) / io (R4-S4) ────


# ── JSON schema (owner; CTL-G07 R4-S5 next) ─────────────────────────────────
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
