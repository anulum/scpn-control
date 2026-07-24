# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact product surface re-exports
"""SCPN Controller Artifact (``.scpnctl.json``) product surface.

Re-exports topology model, validation, compact codec, load/save IO, and JSON
schema leaves so callers keep a stable ``scpn_control.scpn.artifact`` import
path (CTL-G07 R4-S1–S5).
"""

from __future__ import annotations

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
from scpn_control.scpn.artifact_schema import (
    _array_schema as _array_schema,
)
from scpn_control.scpn.artifact_schema import (
    _formal_verification_schema as _formal_verification_schema,
)
from scpn_control.scpn.artifact_schema import (
    _non_empty_string_array_schema as _non_empty_string_array_schema,
)
from scpn_control.scpn.artifact_schema import (
    _non_empty_string_schema as _non_empty_string_schema,
)
from scpn_control.scpn.artifact_schema import (
    _object_schema as _object_schema,
)
from scpn_control.scpn.artifact_schema import (
    _packed_weight_schema as _packed_weight_schema,
)
from scpn_control.scpn.artifact_schema import (
    get_artifact_json_schema as get_artifact_json_schema,
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
