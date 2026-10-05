# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM original feature-source audit
"""Classify preferred original feature metadata without granting runtime readiness."""

from __future__ import annotations

from typing import Any

import numpy as np

from validation.neural_equilibrium_dataset_contracts import FALLBACK_FEATURES

AUDIT_SCHEMA = "scpn-control.mast-efm-original-feature-source-audit.v2"

FEATURE_SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "Ip_MA": {
        "candidates": ("plasma_current_x", "plasma_current_c", "plasma_current_rz"),
        "preferred": "plasma_current_x",
        "required_units": "A",
        "required_dims": ("time",),
        "required_transform": "A_to_MA",
        "source_kind": "measured_total_plasma_current",
        "resolved_status": "source_found_requires_rebuild",
        "resolution": "measured total plasma current is available in original public EFM metadata",
    },
    "Bt_T": {
        "candidates": ("bphi_rmag", "bphi_rgeom", "bvac_rmag", "bvac_rgeom", "bvac_val"),
        "preferred": "bphi_rmag",
        "required_units": "T",
        "required_dims": ("time",),
        "required_transform": "identity_T",
        "source_kind": "total_toroidal_field_at_magnetic_axis",
        "resolved_status": "source_found_requires_rebuild",
        "resolution": "total toroidal field at the magnetic axis is available in original public EFM metadata",
    },
    "ffprime_scale": {
        "candidates": ("ffprime", "ffprime_coefs", "fpsi_c"),
        "preferred": "ffprime",
        "required_units": "T-rad",
        "required_dims": ("time", "psi_norm"),
        "required_transform": "profile_rms_to_campaign_median_normalised_scalar",
        "source_kind": "ffprime_profile",
        "resolved_status": "source_found_requires_rebuild",
        "resolution": (
            "FF-prime profile is available; the dataset policy uses per-time-slice RMS magnitude, "
            "campaign-median normalisation, and [0.25, 4.0] clipping"
        ),
    },
}


def variables_from_metadata(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Summarize present candidate declarations; absent names never acquire synthetic descriptors."""
    result: dict[str, dict[str, Any]] = {}
    names = sorted({name for policy in FEATURE_SOURCE_POLICY.values() for name in policy["candidates"]})
    for name in names:
        array = metadata.get(name + "/.zarray")
        if array is None:
            continue
        attrs = metadata.get(name + "/.zattrs", {})
        if not isinstance(array, dict) or not isinstance(attrs, dict):
            raise ValueError(f"invalid original candidate metadata: {name}")
        result[name] = {
            "attrs": dict(attrs),
            "chunks": array.get("chunks"),
            "shape": array.get("shape"),
            "dtype": array.get("dtype"),
            "dims": attrs.get("_ARRAY_DIMENSIONS", attrs.get("dims", [])),
            **{field: attrs.get(field) for field in ("description", "mds_name", "quality", "uda_name", "units")},
        }
    return result


def _supported(variable: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Require real nonempty shaped data with the exact supported unit/dimension labels."""
    attrs = variable.get("attrs", {})
    if not isinstance(attrs, dict):
        return False
    units = variable.get("units", attrs.get("units"))
    dims = variable.get("dims", attrs.get("_ARRAY_DIMENSIONS", attrs.get("dims", [])))
    shape = variable.get("shape")
    if (
        units != policy["required_units"]
        or not isinstance(dims, list)
        or any(not isinstance(dim, str) for dim in dims)
        or len(dims) != len(policy["required_dims"])
        or set(dims) != set(policy["required_dims"])
        or not isinstance(shape, list)
        or len(shape) != len(dims)
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        return False
    if not isinstance(variable.get("dtype"), str):
        return False
    try:
        dtype = np.dtype(variable["dtype"])
    except (TypeError, ValueError):
        return False
    return dtype.kind in "fiu"


def classify_feature_sources(variables: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Classify only converter-supported preferred channels; alternatives remain policy inventory.

    Supported metadata does not prove chunk availability, conversion success,
    reference equivalence or measurement authenticity. Transposed labelled FF
    time axes are supported by the actual converter and retained here.

    >>> classify_feature_sources({})["Bt_T"]["status"]
    'blocked'
    """
    result: dict[str, dict[str, Any]] = {}
    for feature in FALLBACK_FEATURES:
        policy = FEATURE_SOURCE_POLICY[feature]
        names = list(policy["candidates"])
        present = [name for name in names if name in variables]
        preferred = policy["preferred"]
        source = variables.get(preferred)
        selected = isinstance(source, dict) and _supported(source, policy)
        status = (
            "source_found_requires_rebuild"
            if selected
            else ("source_found_requires_policy" if any(name != preferred for name in present) else "blocked")
        )
        entry: dict[str, Any] = {
            "candidate_sources": names,
            "present_sources": present,
            "required_dims": list(policy["required_dims"]),
            "required_transform": policy["required_transform"],
            "required_units": policy["required_units"],
            "source_kind": policy["source_kind"],
            "selected_source": preferred if selected else None,
            "status": status,
            "resolution": policy["resolution"]
            if selected
            else "preferred converter-supported source metadata is unavailable",
        }
        if isinstance(source, dict) and selected:
            entry["selected_source_metadata"] = {
                field: source.get(field, source.get("attrs", {}).get(field))
                for field in ("description", "dims", "dtype", "mds_name", "quality", "shape", "uda_name", "units")
            }
            # Legacy summaries may carry dimension/unit descriptors only in attrs.
            entry["selected_source_metadata"]["dims"] = source.get(
                "dims", source.get("attrs", {}).get("_ARRAY_DIMENSIONS", source.get("attrs", {}).get("dims", []))
            )
        result[feature] = entry
    return result


def aggregate_feature_status(shots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate every selected shot, keeping unavailable and alternative-only channels blocked."""
    result: dict[str, dict[str, Any]] = {}
    for feature in FALLBACK_FEATURES:
        entries = [shot["feature_status"][feature] for shot in shots]
        statuses = {entry["status"] for entry in entries}
        status = (
            "blocked"
            if "blocked" in statuses
            else (
                "source_found_requires_policy"
                if "source_found_requires_policy" in statuses
                else "source_found_requires_rebuild"
            )
        )
        exemplar = entries[0]
        selected = sorted({entry["selected_source"] for entry in entries if entry["selected_source"]})
        result[feature] = {
            **{
                key: exemplar[key]
                for key in ("candidate_sources", "required_dims", "required_transform", "required_units", "source_kind")
            },
            "present_sources": sorted({name for entry in entries for name in entry["present_sources"]}),
            "selected_source": selected[0] if len(selected) == 1 else None,
            "selected_sources": selected,
            "status": status,
            "resolution": exemplar["resolution"]
            if len({entry["resolution"] for entry in entries}) == 1
            else "source metadata differs across shots",
        }
    return result
