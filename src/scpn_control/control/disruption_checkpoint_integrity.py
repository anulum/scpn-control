# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption checkpoint byte integrity

"""SHA-256 pin parsing and stable checkpoint-byte snapshots."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import BinaryIO, Iterator


class DisruptionCheckpointIntegrityError(RuntimeError):
    """Raised when a disruption-model checkpoint fails its weights-hash check.

    Raised when a pinned digest mismatches, when a sidecar is malformed, or when
    ``require_pin`` is set but no digest is available. It is a hard, fail-closed
    error and is *not* downgraded to the heuristic fallback (unlike a corrupt or
    unreadable file). The strong "never load unverified weights" guarantee holds
    only when a digest is pinned or ``require_pin=True``; an unpinned load is
    RCE-safe (``weights_only=True``) and records the digest for provenance, but
    does not verify the weights against a known-good reference.
    """


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(text: str) -> bool:
    """Return whether every character in ``text`` is a hexadecimal digit."""
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _expected_checkpoint_digest(path: Path, explicit: str | None) -> str | None:
    """Resolve the expected checkpoint digest from an explicit value or sidecar.

    Precedence: an ``explicit`` digest wins; otherwise a ``<checkpoint>.sha256``
    sidecar file (first whitespace-delimited token, as written by ``sha256sum``)
    is used if present. Returns ``None`` when neither pins a digest.
    """
    if explicit is not None:
        token = explicit.strip().split()[0] if explicit.strip() else ""
        if len(token) != 64 or not _is_hex(token):
            raise ValueError("expected_sha256 must be a 64-character hex SHA-256 digest")
        return token.lower()
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        return None
    raw = sidecar.read_text(encoding="utf-8").strip()
    token = raw.split()[0] if raw else ""
    if len(token) != 64 or not _is_hex(token):
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint sidecar {sidecar.name} does not contain a valid SHA-256 digest"
        )
    return token.lower()


def verify_checkpoint_integrity(
    path: Path,
    expected_sha256: str | None = None,
    *,
    require_pin: bool = False,
) -> str:
    """Return a checkpoint's SHA-256, enforcing an expected digest when pinned.

    When ``expected_sha256`` (or a ``<checkpoint>.sha256`` sidecar) pins a digest,
    a mismatch raises :class:`DisruptionCheckpointIntegrityError`. With nothing
    pinned the digest is returned for provenance without gating the load — unless
    ``require_pin`` is set, in which case an unpinned load is itself a fail-closed
    :class:`DisruptionCheckpointIntegrityError` (the strong "never load unverified
    weights" posture for safety-critical use).
    """
    expected = _expected_checkpoint_digest(path, expected_sha256)
    if expected is None and require_pin:
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint {path.name} has no pinned SHA-256 digest but require_pin is set"
        )
    actual = _sha256_file(path)
    if expected is not None and actual.lower() != expected:
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint {path.name} SHA-256 {actual} does not match the pinned digest {expected}"
        )
    return actual


@contextmanager
def verified_checkpoint_snapshot(
    path: Path,
    expected_sha256: str | None = None,
    *,
    require_pin: bool = False,
) -> Iterator[tuple[BinaryIO, str]]:
    """Yield immutable copied bytes and their verified digest for one load.

    The source path is opened once. Bytes are copied in bounded chunks into an
    unlinked temporary file while the digest is computed. The caller deserialises
    that same copy, so a path replacement after verification cannot change the
    weights associated with the returned digest.
    """
    expected = _expected_checkpoint_digest(path, expected_sha256)
    if expected is None and require_pin:
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint {path.name} has no pinned SHA-256 digest but require_pin is set"
        )
    digest = hashlib.sha256()
    with path.open("rb") as source, TemporaryFile(mode="w+b") as snapshot:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
            snapshot.write(chunk)
        actual = digest.hexdigest()
        if expected is not None and actual != expected:
            raise DisruptionCheckpointIntegrityError(
                f"checkpoint {path.name} SHA-256 {actual} does not match the pinned digest {expected}"
            )
        snapshot.seek(0)
        yield snapshot, actual
