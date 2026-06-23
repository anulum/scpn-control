# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio capability manifest
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""The CONTROL studio's capability manifest (schema A) on the platform contract.

Authors CONTROL's :class:`scpn_studio_platform.manifest.CapabilityManifest`: the
verbs it advertises, the evidence schemas they emit, the federated UI panel the Hub
loads, and the content-addressed digest of that declared surface. The digest is
computed with the platform's :func:`scpn_studio_platform.manifest.content_digest`
over the canonical JSON of the declared verbs and evidence schemas, so it is
reproducible across checkouts and independent of git state — the cross-repo
``capability_manifest`` drift gotcha does not apply.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

from scpn_studio_platform.manifest import (
    CapabilityManifest,
    TransportProfile,
    UiModule,
    content_digest,
)

from .verbs import CONTROL_VERBS, STUDIO_ID, evidence_schemas


def _resolve_studio_version() -> str:
    """Return the installed ``scpn-control`` version, or a local sentinel.

    Returns
    -------
    str
        The distribution version when ``scpn-control`` is installed, otherwise
        ``"0+unknown"`` so the manifest never carries a fabricated version.
    """
    try:
        return version("scpn-control")
    except PackageNotFoundError:  # pragma: no cover - only in a non-installed tree
        return "0+unknown"


STUDIO_VERSION = _resolve_studio_version()
"""The CONTROL studio version this manifest stamps (the installed package version)."""

PLATFORM_SDK_RANGE = ">=0.2,<0.3"
"""The platform SDK SemVer range the studio builds on (matches the ``studio`` extra)."""

PROTOCOL_VERSION = "1"
"""The SYNAPSE wire protocol version the studio pins."""

UI_PANEL = UiModule(
    remote_entry="scpn_control_studio/remoteEntry.js",
    exposes=("./ControlStudioPanel",),
)
"""The federated UI panel the Hub loads for the CONTROL studio."""


def declared_surface() -> dict[str, bytes]:
    """Return the content-addressable declared surface of the CONTROL studio.

    The surface is the canonical JSON of each advertised verb plus the evidence
    schema list, keyed by a stable logical path. Hashing file *content* (not git
    state) is what makes the digest reproducible across checkouts.

    Returns
    -------
    dict[str, bytes]
        Mapping of logical path to canonical-JSON bytes, suitable for
        :func:`scpn_studio_platform.manifest.content_digest`.
    """
    surface: dict[str, bytes] = {
        f"verb/{verb.name}": json.dumps(verb.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        for verb in CONTROL_VERBS
    }
    surface["evidence/schemas"] = json.dumps(list(evidence_schemas()), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return surface


def build_manifest(*, studio_version: str = STUDIO_VERSION) -> CapabilityManifest:
    """Build the CONTROL studio's capability manifest.

    Parameters
    ----------
    studio_version
        The studio version to stamp; defaults to :data:`STUDIO_VERSION` (the
        installed ``scpn-control`` version).

    Returns
    -------
    CapabilityManifest
        The schema-A manifest, with a content digest over :func:`declared_surface`.
    """
    return CapabilityManifest(
        studio=STUDIO_ID,
        studio_version=studio_version,
        platform_sdk=PLATFORM_SDK_RANGE,
        content_digest=content_digest(declared_surface()),
        protocol_version=PROTOCOL_VERSION,
        transport_profile=TransportProfile.LOCAL_FIRST,
        verbs=CONTROL_VERBS,
        evidence_types=evidence_schemas(),
        ui_module=UI_PANEL,
    )
