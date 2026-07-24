# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact load / save and payload hashing

"""Load/save and canonical payload hashing for SCPN controller artifacts.

This leaf owns ``.scpnctl.json`` parse/serialise, formal-verification field
parse/emit helpers, and :func:`compute_artifact_payload_sha256`. JSON schema
emission remains on :mod:`scpn_control.scpn.artifact` (CTL-G07 R4-S4).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scpn_control.scpn.artifact_codec import _decode_u64_compact, _encode_u64_compact
from scpn_control.scpn.artifact_model import (
    FORMAL_VERIFICATION_ALLOWED_FIELDS,
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
    ArtifactValidationError,
    _validate_formal_verification,
    validate_artifact,
    validate_safety_critical_artifact,
)


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


def _formal_verification_dict(evidence: FormalVerificationEvidence) -> dict[str, Any]:
    obj: dict[str, Any] = {
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
) -> dict[str, Any]:
    """Return canonical artifact payload sections excluding proof evidence."""

    def _weight_matrix_dict(wm: WeightMatrix) -> dict[str, Any]:
        return {"shape": wm.shape, "data": wm.data}

    packed_dict: dict[str, Any] | None = None
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

    obj: dict[str, Any] = {
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


def _parse_dense_weight_data(raw: object, field_name: str) -> list[float]:
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
