# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for disruption checkpoint leaf

"""Drive production checkpoint integrity and train/load leaf contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

import scpn_control.control.disruption_checkpoint as leaf
import scpn_control.control.disruption_predictor as owner

_PAYLOAD = b"trained-disruption-weights-blob-leaf"


def _write_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "disruption_model.pth"
    path.write_bytes(_PAYLOAD)
    return path, hashlib.sha256(_PAYLOAD).hexdigest()


def test_owner_checkpoint_symbols_bind_to_leaf() -> None:
    """Owner re-exports are the production checkpoint leaf objects."""
    assert owner.verify_checkpoint_integrity is leaf.verify_checkpoint_integrity
    assert owner.train_predictor is leaf.train_predictor
    assert owner.load_or_train_predictor is leaf.load_or_train_predictor
    assert owner.DisruptionCheckpointIntegrityError is leaf.DisruptionCheckpointIntegrityError
    assert owner.default_model_path is leaf.default_model_path
    assert owner._sha256_file is leaf._sha256_file
    assert owner._expected_checkpoint_digest is leaf._expected_checkpoint_digest
    assert owner._is_hex is leaf._is_hex
    assert owner.DEFAULT_MODEL_FILENAME == leaf.DEFAULT_MODEL_FILENAME


def test_sha256_and_verify_paths(tmp_path: Path) -> None:
    """Digest helpers and pin gate match hashlib and reject mismatches."""
    path, digest = _write_checkpoint(tmp_path)
    assert leaf._sha256_file(path) == digest
    assert leaf.verify_checkpoint_integrity(path) == digest
    assert leaf.verify_checkpoint_integrity(path, digest) == digest
    assert leaf.verify_checkpoint_integrity(path, digest.upper()) == digest
    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="does not match"):
        leaf.verify_checkpoint_integrity(path, "0" * 64)
    with pytest.raises(ValueError, match="64-character hex"):
        leaf.verify_checkpoint_integrity(path, "deadbeef")
    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="no pinned"):
        leaf.verify_checkpoint_integrity(path, require_pin=True)


def test_sidecar_and_is_hex(tmp_path: Path) -> None:
    """Sidecar resolution and hex validation are fail-closed."""
    path, digest = _write_checkpoint(tmp_path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  disruption_model.pth\n", encoding="utf-8")
    assert leaf.verify_checkpoint_integrity(path) == digest
    assert leaf.verify_checkpoint_integrity(path, require_pin=True) == digest

    bad = tmp_path / "bad.pth"
    bad.write_bytes(_PAYLOAD)
    bad_sidecar = bad.with_name(bad.name + ".sha256")
    bad_sidecar.write_text("not-a-digest\n", encoding="utf-8")
    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="valid SHA-256"):
        leaf.verify_checkpoint_integrity(bad)

    assert leaf._is_hex("0" * 64)
    assert leaf._is_hex("abcDEF123")
    assert not leaf._is_hex("z" * 64)
    assert not leaf._is_hex("")
    assert leaf._expected_checkpoint_digest(path, None) == digest
    empty = tmp_path / "empty.pth"
    empty.write_bytes(b"x")
    assert leaf._expected_checkpoint_digest(empty, None) is None


def test_default_model_path_under_repo() -> None:
    """Default model path resolves under the repo artifacts directory."""
    path = leaf.default_model_path()
    assert path.name == leaf.DEFAULT_MODEL_FILENAME
    assert path.parent.name == "artifacts"
    assert leaf._repo_root().is_dir()


def test_load_or_train_fallback_and_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch-missing and train-failure fallbacks stay fail-closed without opt-in."""
    monkeypatch.setattr(leaf, "torch", None)
    with pytest.raises(RuntimeError, match="Torch is required"):
        leaf.load_or_train_predictor(model_path=tmp_path / "m.pt", allow_fallback=False)
    model, info = leaf.load_or_train_predictor(
        model_path=tmp_path / "m.pt",
        allow_fallback=True,
        allow_legacy_fallback=True,
    )
    assert model is None
    assert info["fallback"] is True
    assert info["reason"] == "torch_unavailable"

    with pytest.raises(ValueError, match="allow_legacy_fallback=True"):
        leaf.load_or_train_predictor(allow_fallback=True, allow_legacy_fallback=False)

    # Restore a truthy torch sentinel so the train path is reached, then fail train.
    monkeypatch.setattr(leaf, "torch", object())
    with patch.object(leaf, "train_predictor", side_effect=RuntimeError("mock train fail")):
        model2, info2 = leaf.load_or_train_predictor(
            model_path=tmp_path / "missing.pt",
            force_retrain=True,
            train_if_missing=True,
            allow_fallback=True,
            allow_legacy_fallback=True,
        )
    assert model2 is None
    assert info2["fallback"] is True
    assert "train_failed" in info2["reason"]


def test_load_or_train_missing_checkpoint_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing checkpoint without train raises or falls back when opted in."""
    monkeypatch.setattr(leaf, "torch", object())
    missing = tmp_path / "absent.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        leaf.load_or_train_predictor(
            model_path=missing,
            train_if_missing=False,
            force_retrain=False,
            allow_fallback=False,
        )
    model, info = leaf.load_or_train_predictor(
        model_path=missing,
        train_if_missing=False,
        force_retrain=False,
        allow_fallback=True,
        allow_legacy_fallback=True,
    )
    assert model is None
    assert info["reason"] == "checkpoint_missing"
