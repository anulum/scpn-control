# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST replay archive snapshot and structural validation.
"""Decode immutable NPZ snapshots without pickle or physical admission."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from validation.mast_replay_contracts._inputs import MEASURED_CHANNELS, shot_identifiers, time_axis
from validation.mast_source_object_manifest import array_value_sha256, canonical_json_sha256

REPLAY_MEMBER_DIGEST_KIND = "canonical-channel-values-sha256-v1"


class ReplayArchiveError(ValueError):
    """Deliberately authored archive refusal suitable for a public response."""

    response_safe = True


def read_replay_archive_bytes(
    raw: bytes,
    *,
    path_name: str,
    expected_shot_ids: Sequence[int] | None = None,
    integral_float_compatibility: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode one NPZ snapshot into fresh shot vectors and byte/value bindings.

    Exact unique member names, sorted positive identities, finite floating
    aligned vectors and strictly increasing nonempty times are required.
    Integer-valued legacy float identities are optionally accepted by the
    dataset reader; producer inspection requires integer dtype. No pickle,
    provenance authentication, resource limit or filesystem snapshot is added.
    Native read/decode errors become fixed authored ``ReplayArchiveError``.
    """
    if not path_name:
        raise ReplayArchiveError("replay archive path_name must be non-empty")
    shots: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ReplayArchiveError("replay archive member names must be unique")
            if "shot_ids" not in archive.files:
                raise ReplayArchiveError("replay archive must contain shot_ids")
            try:
                ids = shot_identifiers(archive["shot_ids"], integral_float_compatibility=integral_float_compatibility)
            except ValueError:
                raise ReplayArchiveError(
                    "replay archive shot_ids must be a one-dimensional integer vector; "
                    "identities must be unique, positive, and sorted exact finite signed-int64 values"
                ) from None
            if expected_shot_ids is not None and ids != list(expected_shot_ids):
                raise ReplayArchiveError("replay archive shot_ids do not match the producer inventory")
            expected = {"shot_ids"} | {f"{shot_id}:{name}" for shot_id in ids for name in MEASURED_CHANNELS}
            if set(archive.files) != expected:
                raise ReplayArchiveError("replay archive member inventory does not match its shot/channel schema")
            for shot_id in ids:
                channels = {}
                digests = []
                count: int | None = None
                for name in MEASURED_CHANNELS:
                    value = np.asarray(archive[f"{shot_id}:{name}"])
                    if value.ndim != 1 or value.dtype.kind != "f" or not bool(np.all(np.isfinite(value))):
                        raise ReplayArchiveError(
                            f"replay archive shot {shot_id} channel {name} must be a finite float vector"
                        )
                    if count is None:
                        count = int(value.size)
                    elif value.size != count:
                        raise ReplayArchiveError(f"replay archive shot {shot_id} channel lengths differ")
                    channels[name] = value.copy()
                    digests.append({"name": name, "value_sha256": array_value_sha256(value)})
                if count is None or count <= 0:
                    raise ReplayArchiveError(f"replay archive shot {shot_id} must contain samples")
                try:
                    time_axis(channels["time_s"], name="time_s")
                except ValueError:
                    raise ReplayArchiveError("replay archive time_s must be strictly increasing") from None
                shots.append({"shot_id": shot_id, "channels": channels})
                members.append(
                    {
                        "shot_id": shot_id,
                        "n_samples": count,
                        "sha256": canonical_json_sha256({"shot_id": shot_id, "channels": digests}),
                    }
                )
    except ReplayArchiveError:
        raise
    except (OSError, KeyError, TypeError, ValueError, EOFError, BadZipFile):
        raise ReplayArchiveError("cannot validate replay archive bytes") from None
    binding = {
        "path": path_name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "shot_count": len(members),
        "member_digest_kind": REPLAY_MEMBER_DIGEST_KIND,
        "shot_members": members,
    }
    return shots, binding


def inspect_replay_archive_bytes(
    raw: bytes, *, path_name: str, expected_shot_ids: Sequence[int] | None = None
) -> dict[str, Any]:
    """Return strict integer-schema byte/value bindings for one NPZ snapshot."""
    return read_replay_archive_bytes(raw, path_name=path_name, expected_shot_ids=expected_shot_ids)[1]


def inspect_replay_archive(path: Path, *, expected_shot_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Read once and inspect a file; authored refusals replace native I/O text.

    Direct/resolved/symbolic path observations are sequential and may race.
    The report hashes the bytes actually read and authenticates no producer.
    """
    if not path.is_file():
        raise ReplayArchiveError("replay archive does not exist")
    try:
        raw = path.read_bytes()
    except OSError:
        raise ReplayArchiveError("cannot read replay archive") from None
    return inspect_replay_archive_bytes(raw, path_name=path.name, expected_shot_ids=expected_shot_ids)
