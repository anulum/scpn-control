# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact JSON Schema emission

"""Draft-07 JSON Schema for SCPN controller artifacts (``.scpnctl.json``).

This leaf owns :func:`get_artifact_json_schema` and the closed object/array
schema helpers used to emit it. Load/save, validation, codec, and model leaves
remain separate (CTL-G07 R4-S5).
"""

from __future__ import annotations

from typing import Any

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
)


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


def get_artifact_json_schema() -> dict[str, Any]:
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
