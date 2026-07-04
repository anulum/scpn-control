# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sealed safety-certificate claim artefact
"""Float-free sealed-claim artefact for the SCPN Studio Hub transparency log.

The Hub seals evidence artefacts (``studio.publication-seal.v1``) into an
append-only RFC-6962 Merkle transparency log and verifies them in-browser with
an RFC-8785 JCS canonicaliser that rejects non-integer JSON numbers. This
module emits CONTROL's safety-certificate claim in that constrained form:
every numeric value is an integer within the exact-interoperability range or
an exact decimal string — JSON floats are rejected fail-closed before a byte
is written.

Unlike :mod:`scpn_control.studio.evidence`, this module deliberately has no
``scpn_studio_platform`` dependency: the artefact is plain JSON handed to the
Hub keeper, so it must be producible from a checkout without the optional
studio SDK installed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

SEALED_SAFETY_CLAIM_SCHEMA: Final[str] = "scpn-control.sealed-safety-claim.v1"
_STUDIO_ID: Final[str] = "scpn-control"

# RFC 8785 §3.2.2.3 serialises numbers as ES6 doubles; integers outside the
# 2**53 - 1 magnitude cannot round-trip exactly, so the guard rejects them.
_JCS_MAX_SAFE_INT: Final[int] = 2**53 - 1


def assert_jcs_safe(value: object, *, path: str = "$") -> None:
    """Reject any value the Hub's RFC-8785 JCS verifier cannot seal.

    Parameters
    ----------
    value
        JSON-compatible tree to inspect (mappings, sequences, scalars).
    path
        JSONPath-style location prefix used in error messages.

    Raises
    ------
    ValueError
        If the tree contains a ``float`` (including ``bool``-adjacent NumPy
        scalars coerced to float), an integer beyond ±(2**53 − 1), or a
        non-JSON type. Exact decimal values must be carried as strings.
    """
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, float):
        raise ValueError(
            f"{path}: JSON floats cannot be sealed (RFC-8785 JCS verifier rejects them); encode the exact decimal as a string"
        )
    if isinstance(value, int):
        if abs(value) > _JCS_MAX_SAFE_INT:
            raise ValueError(f"{path}: integer magnitude exceeds 2**53-1 and cannot round-trip through an ES6 number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object keys must be strings, got {type(key).__name__}")
            assert_jcs_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_jcs_safe(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path}: type {type(value).__name__} is not JSON-sealable")


def build_safety_certificate_sealed_claim(
    certificate: Mapping[str, Any],
    *,
    live_topology_sha256: str,
    checker: str,
    checker_version: str,
    studio_version: str,
    claim_id: str,
    issued_utc: str,
) -> dict[str, Any]:
    """Build the float-free sealed-claim payload from an issued certificate.

    Parameters
    ----------
    certificate
        A certificate from
        :func:`scpn_control.scpn.issue_runtime_safety_certificate` — its
        ``binding.petri_topology_sha256`` is the proven subject,
        ``formal_certificate_sha256`` the proof artifact digest,
        ``checked_specs`` the proven obligations, and ``payload_sha256`` the
        certificate digest this claim cites for provenance.
    live_topology_sha256
        SHA-256 of the live controller's Petri-net topology. A mismatch with
        the proven topology degrades ``claim_status`` to ``validation-gap``
        and ``admission`` to ``rejected`` — the claim never presents a voided
        proof as established.
    checker, checker_version
        Proof engine and its exact version; certificates do not stamp these,
        so the caller that ran the proof supplies them.
    studio_version
        Version of the CONTROL studio producing the claim.
    claim_id
        Stable capability/claim identifier under which the Hub records the
        sealed artefact.
    issued_utc
        ISO-8601 UTC timestamp of claim emission (caller-supplied so the
        payload stays deterministic for a given certificate).

    Returns
    -------
    dict[str, Any]
        JCS-safe payload (verified by :func:`assert_jcs_safe` before return).

    Raises
    ------
    KeyError
        If the certificate lacks binding, proof, digest, or spec fields.
    ValueError
        If any identity field is empty or the payload fails the JCS guard.
    """
    for name, text in (
        ("live_topology_sha256", live_topology_sha256),
        ("checker", checker),
        ("checker_version", checker_version),
        ("studio_version", studio_version),
        ("claim_id", claim_id),
        ("issued_utc", issued_utc),
    ):
        if not text.strip():
            raise ValueError(f"{name} must be non-empty")

    binding = certificate["binding"]
    checked_specs = [str(spec) for spec in certificate["checked_specs"]]
    theorem_id = " ; ".join(checked_specs) if checked_specs else str(certificate["scope"])
    formal = certificate["formal_certificate"]
    non_vacuous = bool(formal.get("non_vacuous", False)) if isinstance(formal, Mapping) else False
    proven_topology = str(binding["petri_topology_sha256"])
    topology_matches = proven_topology == live_topology_sha256

    payload: dict[str, Any] = {
        "schema": SEALED_SAFETY_CLAIM_SCHEMA,
        "studio": _STUDIO_ID,
        "studio_version": studio_version,
        "claim_id": claim_id,
        "issued_utc": issued_utc,
        "claim": {
            "statement": (
                "Machine-checked safety/liveness obligations hold for the compiled"
                " StochasticPetriNet topology cited by subject_topology_sha256;"
                " the proof binds to that exact topology and is void on drift."
            ),
            "theorem_id": theorem_id,
            "checked_specs": checked_specs,
            "checker": checker,
            "checker_version": checker_version,
            "non_vacuous": non_vacuous,
            "topology_matches": topology_matches,
            "claim_status": "reference-validated" if topology_matches else "validation-gap",
            "admission": "admitted" if topology_matches else "rejected",
        },
        "provenance": {
            "certificate_sha256": str(certificate["payload_sha256"]),
            "proof_sha256": str(certificate["formal_certificate_sha256"]),
            "subject_topology_sha256": proven_topology,
            "live_topology_sha256": live_topology_sha256,
        },
    }
    assert_jcs_safe(payload)
    return payload


def render_sealed_claim_json(payload: Mapping[str, Any]) -> str:
    """Serialise a sealed-claim payload to deterministic UTF-8 JSON.

    Parameters
    ----------
    payload
        A JCS-safe payload (re-verified here fail-closed).

    Returns
    -------
    str
        Sorted-key, compact-separator JSON with a trailing newline — byte-stable
        for a given payload so the sha256 the keeper receives is reproducible.
    """
    assert_jcs_safe(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def write_sealed_claim(payload: Mapping[str, Any], path: Path) -> str:
    """Write the sealed-claim artefact and return its SHA-256 hex digest.

    Parameters
    ----------
    payload
        A JCS-safe payload (verified during rendering).
    path
        Destination file path; parent directories are created.

    Returns
    -------
    str
        SHA-256 hex digest of the written bytes — quote it when pinging the
        Hub keeper alongside the artefact path.
    """
    rendered = render_sealed_claim_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rendered.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
