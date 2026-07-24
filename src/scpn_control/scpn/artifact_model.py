# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact topology / payload model dataclasses

"""Pure dataclass model for SCPN controller artifacts (``.scpnctl.json``).

This leaf owns the topology/weight/readout/meta/formal-evidence dataclasses and
the ``Artifact`` container. Validation, load/save, schema, and compact codec
remain on :mod:`scpn_control.scpn.artifact` (CTL-G07 R4-S2+).
"""

from __future__ import annotations

from dataclasses import MISSING, InitVar, dataclass, fields
from typing import Any, List

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
            # Lazy import: validation lives on the owner module (R4-S2+).
            from scpn_control.scpn.artifact import validate_artifact

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


