#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Versioned FAIR-MAST disruption signal-binding contract
"""Bind canonical disruption channels to exact FAIR-MAST Level-2 arrays.

The contract is deliberately separate from extraction and resampling. It names
the source group, array, dimensions, units, timebase, sign convention, transform,
missing-data rule, provenance, and uncertainty for every canonical channel.
Unresolved source semantics remain explicit blockers and can never fall through
to a guessed array, implicit unit conversion, or zero-filled channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from validation.mast_source_artifact_reader import VerifiedSourceArtifact
from validation.mast_source_object_manifest import canonical_json_sha256

MAST_SIGNAL_BINDING_SCHEMA = "scpn-control.mast-signal-binding-spec.v1.0.0"
MAST_SIGNAL_BINDING_VERSION = "1.0.0"
MAST_LEVEL2_SOURCE_SCHEMA = "FAIR-MAST Level-2 Zarr v3"
MAST_INGESTION_COMMIT_URL = "https://github.com/ukaea/fair-mast-ingestion/tree/862f08d7d91930b988d674e7ec67f3a03aacafac"
MAST_LEVEL2_DATA_URL = "https://mastapp.site/level2-data.html"

CANONICAL_DISRUPTION_CHANNELS: tuple[str, ...] = (
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
)

BindingStatus = Literal["bound", "blocked"]


class SignalBindingSpecError(ValueError):
    """Raised when a MAST signal-binding specification is inconsistent."""


@dataclass(frozen=True)
class SignalBinding:
    """One canonical-channel binding or explicit source-semantic blocker.

    Parameters
    ----------
    channel:
        Canonical CONTROL channel name.
    status:
        ``bound`` only when source semantics support the complete mapping;
        otherwise ``blocked``.
    source_key:
        Exact ``<group>.<array>`` source key, or the best source candidate for a
        blocked mapping.
    source_dimensions:
        Exact dimensions declared by the live source metadata.
    source_units:
        Source unit string exactly as declared by FAIR-MAST.
    output_units:
        Canonical CONTROL output unit string.
    timebase_key:
        Exact source time coordinate for the signal.
    timebase_dimensions:
        Exact dimensions declared by the source time coordinate.
    timebase_units:
        Source time-coordinate unit string exactly as declared by FAIR-MAST.
    sign_convention:
        Source-to-output sign contract; never inferred from values.
    validity_interval:
        Rule defining samples eligible for later extraction.
    missing_data_rule:
        Required non-finite/missing treatment.
    transform:
        Named source-to-output transform.
    transform_parameters:
        Deterministic transform parameters as sorted key/value pairs.
    citation:
        Primary source for the binding metadata.
    uncertainty:
        Source uncertainty state; absence is explicit.
    blocker:
        Stable reason code for a blocked binding, else ``None``.
    """

    channel: str
    status: BindingStatus
    source_key: str | None
    source_dimensions: tuple[str, ...]
    source_units: str | None
    output_units: str
    timebase_key: str | None
    timebase_dimensions: tuple[str, ...]
    timebase_units: str | None
    sign_convention: str
    validity_interval: str
    missing_data_rule: str
    transform: str
    transform_parameters: tuple[tuple[str, str], ...]
    citation: str
    uncertainty: str
    blocker: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete bound mappings and unqualified blocked mappings."""
        if not self.channel or not self.output_units or not self.citation:
            raise SignalBindingSpecError("channel, output_units, and citation must be non-empty")
        if self.status == "bound":
            if (
                self.source_key is None
                or self.timebase_key is None
                or not self.source_dimensions
                or not self.timebase_dimensions
            ):
                raise SignalBindingSpecError(
                    f"bound channel {self.channel!r} requires source key, dimensions, and timebase metadata"
                )
            if self.blocker is not None:
                raise SignalBindingSpecError(f"bound channel {self.channel!r} cannot carry a blocker")
        elif self.status == "blocked":
            if self.blocker is None or not self.blocker:
                raise SignalBindingSpecError(f"blocked channel {self.channel!r} requires a reason code")
        else:
            raise SignalBindingSpecError(f"unsupported binding status {self.status!r}")
        if tuple(sorted(self.transform_parameters)) != self.transform_parameters:
            raise SignalBindingSpecError(f"channel {self.channel!r} transform parameters must be sorted")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-ready representation of the binding."""
        return {
            "channel": self.channel,
            "status": self.status,
            "source_key": self.source_key,
            "source_dimensions": list(self.source_dimensions),
            "source_units": self.source_units,
            "output_units": self.output_units,
            "timebase_key": self.timebase_key,
            "timebase_dimensions": list(self.timebase_dimensions),
            "timebase_units": self.timebase_units,
            "sign_convention": self.sign_convention,
            "validity_interval": self.validity_interval,
            "missing_data_rule": self.missing_data_rule,
            "transform": self.transform,
            "transform_parameters": dict(self.transform_parameters),
            "citation": self.citation,
            "uncertainty": self.uncertainty,
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class MastSignalBindingSpec:
    """Versioned FAIR-MAST binding set for the CONTROL disruption channels."""

    schema_version: str
    version: str
    machine: str
    source_schema: str
    ingestion_commit_url: str
    bindings: tuple[SignalBinding, ...]

    def __post_init__(self) -> None:
        """Require one binding record for every canonical channel."""
        if self.schema_version != MAST_SIGNAL_BINDING_SCHEMA:
            raise SignalBindingSpecError(f"schema_version must equal {MAST_SIGNAL_BINDING_SCHEMA!r}")
        if self.version != MAST_SIGNAL_BINDING_VERSION or self.machine != "MAST":
            raise SignalBindingSpecError("version and machine must identify the supported MAST contract")
        channels = tuple(binding.channel for binding in self.bindings)
        if channels != CANONICAL_DISRUPTION_CHANNELS:
            raise SignalBindingSpecError("bindings must cover canonical channels once and in canonical order")

    def to_dict(self) -> dict[str, object]:
        """Return a self-digested JSON-ready binding specification."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "version": self.version,
            "machine": self.machine,
            "source_schema": self.source_schema,
            "ingestion_commit_url": self.ingestion_commit_url,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "payload_sha256": None,
        }
        payload["payload_sha256"] = canonical_json_sha256(payload)
        return payload


def _binding(
    channel: str,
    *,
    source_key: str,
    source_dimensions: tuple[str, ...],
    source_units: str,
    output_units: str,
    timebase_key: str,
    timebase_dimensions: tuple[str, ...] = ("time",),
    timebase_units: str = "s",
    sign_convention: str,
    transform: str,
    transform_parameters: tuple[tuple[str, str], ...] = (),
) -> SignalBinding:
    return SignalBinding(
        channel=channel,
        status="bound",
        source_key=source_key,
        source_dimensions=source_dimensions,
        source_units=source_units,
        output_units=output_units,
        timebase_key=timebase_key,
        timebase_dimensions=timebase_dimensions,
        timebase_units=timebase_units,
        sign_convention=sign_convention,
        validity_interval="finite source value at a finite source time; L2F-11 defines aligned edges",
        missing_data_rule="emit validity mask and reason; never zero-fill or fabricate",
        transform=transform,
        transform_parameters=transform_parameters,
        citation=MAST_INGESTION_COMMIT_URL,
        uncertainty="not supplied by FAIR-MAST array metadata; preserve as unknown",
    )


def _blocked_binding(
    channel: str,
    *,
    source_key: str | None,
    source_dimensions: tuple[str, ...],
    source_units: str | None,
    output_units: str,
    timebase_key: str | None,
    timebase_dimensions: tuple[str, ...],
    timebase_units: str | None,
    transform: str,
    blocker: str,
) -> SignalBinding:
    return SignalBinding(
        channel=channel,
        status="blocked",
        source_key=source_key,
        source_dimensions=source_dimensions,
        source_units=source_units,
        output_units=output_units,
        timebase_key=timebase_key,
        timebase_dimensions=timebase_dimensions,
        timebase_units=timebase_units,
        sign_convention="unresolved; no sign may be inferred from source values",
        validity_interval="unresolved binding is never eligible for extraction",
        missing_data_rule="emit blocked reason; never substitute another channel or zero-fill",
        transform=transform,
        transform_parameters=(),
        citation=MAST_INGESTION_COMMIT_URL,
        uncertainty="unresolved with the source-semantic blocker",
        blocker=blocker,
    )


def mast_level2_signal_binding_spec() -> MastSignalBindingSpec:
    """Return the versioned MAST Level-2 disruption signal-binding contract."""
    return MastSignalBindingSpec(
        schema_version=MAST_SIGNAL_BINDING_SCHEMA,
        version=MAST_SIGNAL_BINDING_VERSION,
        machine="MAST",
        source_schema=MAST_LEVEL2_SOURCE_SCHEMA,
        ingestion_commit_url=MAST_INGESTION_COMMIT_URL,
        bindings=(
            _binding(
                "time_s",
                source_key="summary.time",
                source_dimensions=("time",),
                source_units="s",
                output_units="s",
                timebase_key="summary.time",
                sign_convention="increasing elapsed shot time",
                transform="identity_reference_timebase",
            ),
            _binding(
                "Ip_MA",
                source_key="summary.ip",
                source_dimensions=("time",),
                source_units="A",
                output_units="MA",
                timebase_key="summary.time",
                sign_convention="positive means anti-clockwise when viewed from above",
                transform="scale",
                transform_parameters=(("factor", "1e-6"),),
            ),
            _blocked_binding(
                "BT_T",
                source_key="equilibrium.bvac_rmag",
                source_dimensions=("time",),
                source_units="T",
                output_units="T",
                timebase_key="equilibrium.time",
                timebase_dimensions=("time",),
                timebase_units="s",
                transform="identity_candidate_only",
                blocker="bvac_rmag_semantics_not_yet_approved_as_canonical_bt",
            ),
            _blocked_binding(
                "beta_N",
                source_key="equilibrium.beta_tor_normal",
                source_dimensions=("time",),
                source_units="T",
                output_units="1",
                timebase_key="equilibrium.time",
                timebase_dimensions=("time",),
                timebase_units="s",
                transform="identity_candidate_only",
                blocker="source_units_conflict_with_dimensionless_definition",
            ),
            _binding(
                "q95",
                source_key="equilibrium.q95",
                source_dimensions=("time",),
                source_units="",
                output_units="1",
                timebase_key="equilibrium.time",
                sign_convention="positive only when toroidal current and magnetic field are in the same direction",
                transform="identity",
            ),
            _binding(
                "ne_1e19",
                source_key="summary.line_average_n_e",
                source_dimensions=("time",),
                source_units="1 / m ** 3",
                output_units="1e19 m^-3",
                timebase_key="summary.time",
                sign_convention="non-negative line-averaged electron density",
                transform="scale",
                transform_parameters=(("factor", "1e-19"),),
            ),
            _blocked_binding(
                "n1_amp",
                source_key="magnetics.b_field_tor_probe_saddle_field",
                source_dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
                source_units="T",
                output_units="T",
                timebase_key="magnetics.time_saddle",
                timebase_dimensions=("time_saddle",),
                timebase_units="s",
                transform="toroidal_mode_decomposition_unresolved",
                blocker="saddle_geometry_angle_units_and_modal_reduction_unresolved",
            ),
            _blocked_binding(
                "n2_amp",
                source_key="magnetics.b_field_tor_probe_saddle_field",
                source_dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
                source_units="T",
                output_units="T",
                timebase_key="magnetics.time_saddle",
                timebase_dimensions=("time_saddle",),
                timebase_units="s",
                transform="toroidal_mode_decomposition_unresolved",
                blocker="saddle_geometry_angle_units_and_modal_reduction_unresolved",
            ),
            _blocked_binding(
                "locked_mode_amp",
                source_key="magnetics.b_field_tor_probe_saddle_field",
                source_dimensions=("b_field_tor_probe_saddle_field_channel", "time_saddle"),
                source_units="T",
                output_units="T",
                timebase_key="magnetics.time_saddle",
                timebase_dimensions=("time_saddle",),
                timebase_units="s",
                transform="locked_mode_reduction_unresolved",
                blocker="saddle_geometry_angle_units_and_modal_reduction_unresolved",
            ),
            _blocked_binding(
                "dBdt_gauss_per_s",
                source_key="magnetics.b_field_tor_probe_cc_field",
                source_dimensions=("b_field_tor_probe_cc_channel", "time_mirnov"),
                source_units="T",
                output_units="G/s",
                timebase_key="magnetics.time_mirnov",
                timebase_dimensions=("time_mirnov",),
                timebase_units="s",
                transform="probe_reduction_and_scale_unresolved",
                blocker="source_units_and_label_conflict",
            ),
            _binding(
                "vertical_position_m",
                source_key="equilibrium.magnetic_axis_z",
                source_dimensions=("time",),
                source_units="m",
                output_units="m",
                timebase_key="equilibrium.time",
                sign_convention="source magnetic-axis vertical coordinate",
                transform="identity",
            ),
        ),
    )


def assess_artifact_signal_bindings(
    artifact: VerifiedSourceArtifact,
    *,
    spec: MastSignalBindingSpec | None = None,
) -> dict[str, object]:
    """Assess one verified source artefact against the versioned binding contract.

    This function verifies only source membership and metadata needed by L2F-10.
    It performs no interpolation, filtering, modal reduction, unit conversion, or
    missing-value imputation.

    Parameters
    ----------
    artifact:
        Provenance-verified source artefact from the production reader.
    spec:
        Binding specification; defaults to the supported MAST Level-2 contract.

    Returns
    -------
    dict[str, object]
        Digest-bound fail-closed assessment. Extraction is admissible only when
        all canonical bindings and source metadata are resolved.
    """
    active_spec = spec if spec is not None else mast_level2_signal_binding_spec()
    results: list[dict[str, object]] = []
    for binding in active_spec.bindings:
        results.append(_assess_binding(artifact, binding))
    ready_count = sum(result["status"] == "source_metadata_verified" for result in results)
    blocked_count = len(results) - ready_count
    report: dict[str, object] = {
        "schema_version": MAST_SIGNAL_BINDING_SCHEMA,
        "spec_sha256": active_spec.to_dict()["payload_sha256"],
        "shot_id": artifact.shot_id,
        "artifact_sha256": artifact.artifact_sha256,
        "source_uri": artifact.source_uri,
        "binding_contract_complete": True,
        "channel_extraction_admissible": blocked_count == 0,
        "n_canonical_channels": len(results),
        "n_source_metadata_verified": ready_count,
        "n_blocked": blocked_count,
        "bindings": results,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _assess_binding(artifact: VerifiedSourceArtifact, binding: SignalBinding) -> dict[str, object]:
    result: dict[str, object] = {
        "channel": binding.channel,
        "binding_status": binding.status,
        "source_key": binding.source_key,
        "timebase_key": binding.timebase_key,
        "status": "blocked",
        "reason_code": binding.blocker,
    }
    if binding.status == "blocked":
        result["status"] = "binding_blocked"
        return result
    source_key = cast(str, binding.source_key)
    timebase_key = cast(str, binding.timebase_key)
    missing = [key for key in (source_key, timebase_key) if key not in artifact.arrays]
    if missing:
        result["status"] = "source_member_absent"
        result["reason_code"] = "required_source_member_absent"
        result["missing_source_keys"] = missing
        return result
    source_metadata = artifact.metadata[source_key]
    time_metadata = artifact.metadata[timebase_key]
    if (
        source_metadata.get("metadata_status") != "source_xarray"
        or time_metadata.get("metadata_status") != "source_xarray"
    ):
        result["status"] = "source_metadata_unverified"
        result["reason_code"] = "manifest_does_not_attest_source_xarray_metadata"
        return result
    actual_dimensions = tuple(source_metadata.get("dimensions") or ())
    if actual_dimensions != binding.source_dimensions or source_metadata.get("units") != binding.source_units:
        result["status"] = "source_metadata_mismatch"
        result["reason_code"] = "source_dimensions_or_units_mismatch"
        result["actual_dimensions"] = list(actual_dimensions)
        result["actual_units"] = source_metadata.get("units")
        return result
    actual_timebase_dimensions = tuple(time_metadata.get("dimensions") or ())
    if (
        actual_timebase_dimensions != binding.timebase_dimensions
        or time_metadata.get("units") != binding.timebase_units
    ):
        result["status"] = "timebase_metadata_mismatch"
        result["reason_code"] = "timebase_dimensions_or_units_mismatch"
        result["actual_timebase_dimensions"] = list(actual_timebase_dimensions)
        result["actual_timebase_units"] = time_metadata.get("units")
        return result
    source = np.asarray(artifact.arrays[source_key])
    timebase = np.asarray(artifact.arrays[timebase_key])
    if source.ndim != 1 or timebase.ndim != 1 or source.shape != timebase.shape:
        result["status"] = "source_shape_mismatch"
        result["reason_code"] = "bound_scalar_signal_must_match_one_dimensional_timebase"
        result["source_shape"] = list(source.shape)
        result["timebase_shape"] = list(timebase.shape)
        return result
    result["status"] = "source_metadata_verified"
    result["reason_code"] = None
    result["source_shape"] = list(source.shape)
    result["finite_samples"] = int(np.count_nonzero(np.isfinite(source) & np.isfinite(timebase)))
    result["total_samples"] = int(source.size)
    return result
