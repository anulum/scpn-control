# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Checkpoint Integrity Tests
"""Torch-free coverage of the disruption-model checkpoint weights-hash gate.

Exercises the fail-closed integrity helpers that guard ``load_or_train_predictor``
before it deserialises a checkpoint: explicit-digest and ``.sha256`` sidecar
resolution, matching/mismatching verification, and malformed-input rejection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scpn_control.control.disruption_predictor import (
    DisruptionCheckpointIntegrityError,
    _expected_checkpoint_digest,
    _is_hex,
    _sha256_file,
    verify_checkpoint_integrity,
)

_PAYLOAD = b"trained-disruption-weights-blob"


def _write_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "disruption_model.pth"
    path.write_bytes(_PAYLOAD)
    return path, hashlib.sha256(_PAYLOAD).hexdigest()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    assert _sha256_file(path) == digest


def test_is_hex_accepts_and_rejects() -> None:
    assert _is_hex("0" * 64)
    assert _is_hex("abcDEF123")
    assert not _is_hex("z" * 64)
    assert not _is_hex("")


def test_verify_returns_digest_when_nothing_is_pinned(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    assert verify_checkpoint_integrity(path) == digest


def test_verify_accepts_matching_explicit_digest(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    assert verify_checkpoint_integrity(path, digest) == digest


def test_verify_is_case_insensitive_on_the_pinned_digest(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    assert verify_checkpoint_integrity(path, digest.upper()) == digest


def test_verify_accepts_sha256sum_formatted_digest(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    assert verify_checkpoint_integrity(path, f"{digest}  disruption_model.pth") == digest


def test_verify_rejects_mismatching_explicit_digest(tmp_path: Path) -> None:
    path, _ = _write_checkpoint(tmp_path)
    with pytest.raises(DisruptionCheckpointIntegrityError, match="does not match the pinned digest"):
        verify_checkpoint_integrity(path, "0" * 64)


def test_verify_rejects_malformed_explicit_digest(tmp_path: Path) -> None:
    path, _ = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="64-character hex SHA-256 digest"):
        verify_checkpoint_integrity(path, "deadbeef")
    with pytest.raises(ValueError, match="64-character hex SHA-256 digest"):
        verify_checkpoint_integrity(path, "   ")


def test_sidecar_digest_is_used_when_present(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    path.with_name(path.name + ".sha256").write_text(digest, encoding="utf-8")
    assert verify_checkpoint_integrity(path) == digest


def test_sidecar_sha256sum_format_is_parsed(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  disruption_model.pth\n", encoding="utf-8")
    assert verify_checkpoint_integrity(path) == digest


def test_mismatching_sidecar_raises(tmp_path: Path) -> None:
    path, _ = _write_checkpoint(tmp_path)
    path.with_name(path.name + ".sha256").write_text("1" * 64, encoding="utf-8")
    with pytest.raises(DisruptionCheckpointIntegrityError, match="does not match the pinned digest"):
        verify_checkpoint_integrity(path)


def test_malformed_sidecar_raises(tmp_path: Path) -> None:
    path, _ = _write_checkpoint(tmp_path)
    path.with_name(path.name + ".sha256").write_text("not-a-digest", encoding="utf-8")
    with pytest.raises(DisruptionCheckpointIntegrityError, match="valid SHA-256 digest"):
        verify_checkpoint_integrity(path)


def test_empty_sidecar_raises(tmp_path: Path) -> None:
    path, _ = _write_checkpoint(tmp_path)
    path.with_name(path.name + ".sha256").write_text("   \n", encoding="utf-8")
    with pytest.raises(DisruptionCheckpointIntegrityError, match="valid SHA-256 digest"):
        verify_checkpoint_integrity(path)


def test_explicit_digest_takes_precedence_over_sidecar(tmp_path: Path) -> None:
    path, digest = _write_checkpoint(tmp_path)
    # A wrong sidecar must be ignored when an explicit (correct) digest is given.
    path.with_name(path.name + ".sha256").write_text("2" * 64, encoding="utf-8")
    assert verify_checkpoint_integrity(path, digest) == digest


def test_expected_digest_resolver_returns_none_without_pin(tmp_path: Path) -> None:
    path, _ = _write_checkpoint(tmp_path)
    assert _expected_checkpoint_digest(path, None) is None
