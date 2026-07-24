# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for disruption checkpoint leaf

"""Real production paths for checkpoint integrity and train/load (CTL-G07 R7-S4).

Integrity tests use real file bytes and the public ``verify_checkpoint_integrity``
gate. Train/load tests require torch on the interpreter path and exercise the
full production chain: synthetic train → weights file → pin → load → owner
``predict_disruption_risk_safe``. Torch-absent cases only exercise the genuine
optional-dependency contract (module attribute ``torch is None``), not fakes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import scpn_control.control.disruption_checkpoint as leaf
import scpn_control.control.disruption_predictor as owner

_PAYLOAD = b"trained-disruption-weights-blob-leaf"
torch = pytest.importorskip("torch")


def _write_bytes(tmp_path: Path, name: str = "disruption_model.pth") -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(_PAYLOAD)
    return path, hashlib.sha256(_PAYLOAD).hexdigest()


def _small_train(tmp_path: Path, *, name: str = "real_model.pth") -> tuple[Path, object, dict]:
    """Train a tiny real transformer checkpoint through the production API."""
    path = tmp_path / name
    model, info = leaf.train_predictor(
        seq_len=16,
        n_shots=8,
        epochs=1,
        model_path=path,
        seed=0,
        save_plot=False,
    )
    assert path.is_file()
    assert path.stat().st_size > 0
    assert info["seq_len"] == 16
    assert info["facility_roc_validated"] is False
    assert "claim_boundary" in info
    return path, model, info


def test_owner_checkpoint_symbols_bind_to_leaf() -> None:
    """Owner re-exports are the production checkpoint leaf objects."""
    assert owner.verify_checkpoint_integrity is leaf.verify_checkpoint_integrity
    assert owner.train_predictor is leaf.train_predictor
    assert owner.load_or_train_predictor is leaf.load_or_train_predictor
    assert owner.DisruptionCheckpointIntegrityError is leaf.DisruptionCheckpointIntegrityError
    assert owner.default_model_path is leaf.default_model_path
    assert owner.DEFAULT_MODEL_FILENAME == leaf.DEFAULT_MODEL_FILENAME


def test_verify_checkpoint_integrity_real_bytes_and_fail_closed(tmp_path: Path) -> None:
    """Public integrity gate matches hashlib and rejects pin / sidecar faults."""
    path, digest = _write_bytes(tmp_path)
    assert leaf.verify_checkpoint_integrity(path) == digest
    assert leaf.verify_checkpoint_integrity(path, digest) == digest
    assert leaf.verify_checkpoint_integrity(path, digest.upper()) == digest
    assert leaf.verify_checkpoint_integrity(path, f"{digest}  disruption_model.pth") == digest

    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="does not match"):
        leaf.verify_checkpoint_integrity(path, "0" * 64)
    with pytest.raises(ValueError, match="64-character hex"):
        leaf.verify_checkpoint_integrity(path, "deadbeef")
    # 64 characters that are not hex digits exercises _is_hex's ValueError path.
    with pytest.raises(ValueError, match="64-character hex"):
        leaf.verify_checkpoint_integrity(path, "g" * 64)
    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="no pinned"):
        leaf.verify_checkpoint_integrity(path, require_pin=True)

    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  disruption_model.pth\n", encoding="utf-8")
    assert leaf.verify_checkpoint_integrity(path, require_pin=True) == digest

    bad = tmp_path / "bad.pth"
    bad.write_bytes(_PAYLOAD)
    bad.with_name(bad.name + ".sha256").write_text("not-a-digest\n", encoding="utf-8")
    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="valid SHA-256"):
        leaf.verify_checkpoint_integrity(bad)

    # Empty explicit pin after strip is rejected the same way as a short token.
    with pytest.raises(ValueError, match="64-character hex"):
        leaf.verify_checkpoint_integrity(path, "   ")


def test_default_model_path_under_repo() -> None:
    """Default model path resolves under the real repository artifacts tree."""
    path = leaf.default_model_path()
    assert path.name == leaf.DEFAULT_MODEL_FILENAME
    assert path.parent.name == "artifacts"
    assert leaf._repo_root().is_dir()
    assert (leaf._repo_root() / "src" / "scpn_control").is_dir()


def test_train_load_pin_and_safe_predict_e2e(tmp_path: Path) -> None:
    """Real train → SHA pin → load → owner safe-predict checkpoint mode."""
    path, trained, train_info = _small_train(tmp_path)
    digest = leaf.verify_checkpoint_integrity(path)
    assert len(digest) == 64

    loaded, load_info = leaf.load_or_train_predictor(
        model_path=path,
        seq_len=16,
        train_if_missing=False,
        force_retrain=False,
        expected_sha256=digest,
        require_pin=True,
    )
    assert loaded is not None
    assert load_info["fallback"] is False
    assert load_info["weights_sha256"] == digest
    assert load_info["seq_len"] == 16
    assert load_info["facility_roc_validated"] is False

    # Mismatched pin is fail-closed and never downgraded by allow_fallback.
    with pytest.raises(leaf.DisruptionCheckpointIntegrityError, match="does not match"):
        leaf.load_or_train_predictor(
            model_path=path,
            train_if_missing=False,
            expected_sha256="0" * 64,
            allow_fallback=True,
            allow_legacy_fallback=True,
        )

    signal = np.linspace(0.1, 1.5, 32, dtype=float)
    risk, meta = owner.predict_disruption_risk_safe(
        signal,
        model_path=path,
        seq_len=16,
        train_if_missing=False,
        allow_legacy_fallback=False,
    )
    assert 0.0 <= float(risk) <= 1.0
    assert meta["mode"] == "checkpoint"
    assert meta["risk_source"] == "transformer_mc_dropout"
    assert meta["probabilistic_output"] is True
    assert meta["weights_sha256"] == digest
    assert train_info["model_path"] == str(path)
    # Trained model is a real torch Module surface.
    assert hasattr(trained, "eval")
    assert hasattr(loaded, "eval")


def test_load_or_train_missing_checkpoint_real_torch(tmp_path: Path) -> None:
    """Missing weights with real torch: raise hard or fall back when opted in."""
    missing = tmp_path / "absent.pth"
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
    assert info["fallback"] is True
    assert info["reason"] == "checkpoint_missing"


def test_load_or_train_corrupt_checkpoint_real_torch(tmp_path: Path) -> None:
    """Corrupt weights bytes: load fails closed without fallback; fallback when opted in."""
    import pickle

    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"not-a-torch-checkpoint")
    digest = leaf.verify_checkpoint_integrity(path)

    with pytest.raises((RuntimeError, ValueError, OSError, KeyError, pickle.UnpicklingError)):
        leaf.load_or_train_predictor(
            model_path=path,
            train_if_missing=False,
            expected_sha256=digest,
            allow_fallback=False,
        )

    model, info = leaf.load_or_train_predictor(
        model_path=path,
        train_if_missing=False,
        expected_sha256=digest,
        allow_fallback=True,
        allow_legacy_fallback=True,
    )
    assert model is None
    assert info["fallback"] is True
    assert "checkpoint_load_failed" in info["reason"]


def test_load_or_train_bare_state_dict_checkpoint(tmp_path: Path) -> None:
    """Legacy bare state_dict checkpoints still load through the production path."""
    path, model, _info = _small_train(tmp_path, name="wrapped.pth")
    bare = tmp_path / "bare_state_dict.pth"
    torch.save(model.state_dict(), bare)
    loaded, info = leaf.load_or_train_predictor(
        model_path=bare,
        seq_len=16,
        train_if_missing=False,
        expected_sha256=leaf.verify_checkpoint_integrity(bare),
        require_pin=True,
    )
    assert loaded is not None
    assert info["seq_len"] == 16
    assert info["fallback"] is False


def test_load_or_train_force_retrain_succeeds(tmp_path: Path) -> None:
    """force_retrain re-runs real train_predictor even when a file already exists."""
    path, _old, _info = _small_train(tmp_path, name="old.pth")
    model, info = leaf.load_or_train_predictor(
        model_path=path,
        force_retrain=True,
        train_if_missing=True,
        train_kwargs={"seq_len": 16, "n_shots": 8, "epochs": 1, "save_plot": False, "seed": 9},
    )
    assert model is not None
    assert info["trained"] is True
    assert info["fallback"] is False
    assert path.is_file()


def test_train_multi_epoch_log_branch_and_optional_plot(tmp_path: Path) -> None:
    """Multi-epoch train exercises non-log epochs; plot path is real when MPL is present."""
    path = tmp_path / "multi_epoch.pth"
    # epochs=3 → middle epoch skips the every-10-steps log branch (242→234).
    model, info = leaf.train_predictor(
        seq_len=16,
        n_shots=8,
        epochs=3,
        model_path=path,
        seed=5,
        save_plot=True,
    )
    assert model is not None
    assert path.is_file()
    assert info["epochs"] == 3


def test_train_failure_via_unwritable_path_real_torch(tmp_path: Path) -> None:
    """Force a real train I/O failure (directory-as-file) through production load_or_train."""
    blocked = tmp_path / "not_a_file_path"
    blocked.mkdir()
    model, info = leaf.load_or_train_predictor(
        model_path=blocked,
        force_retrain=True,
        train_if_missing=True,
        allow_fallback=True,
        allow_legacy_fallback=True,
        train_kwargs={"seq_len": 16, "n_shots": 8, "epochs": 1, "save_plot": False, "seed": 1},
    )
    assert model is None
    assert info["fallback"] is True
    assert "train_failed" in info["reason"]

    # Same I/O fault without fallback must raise, not soft-fail (covers re-raise arm).
    with pytest.raises((RuntimeError, ValueError, OSError, TypeError)):
        leaf.load_or_train_predictor(
            model_path=blocked,
            force_retrain=True,
            train_if_missing=True,
            allow_fallback=False,
            train_kwargs={"seq_len": 16, "n_shots": 8, "epochs": 1, "save_plot": False, "seed": 2},
        )


def test_optional_torch_absent_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Honest optional-dependency contract when leaf.torch is None (not a fake runtime)."""
    monkeypatch.setattr(leaf, "torch", None)
    monkeypatch.setattr(leaf, "optim", None)
    monkeypatch.setattr(leaf, "nn", None)

    with pytest.raises(RuntimeError, match="Torch is required"):
        leaf.train_predictor(model_path=tmp_path / "x.pth", n_shots=8, epochs=1, save_plot=False)
    with pytest.raises(RuntimeError, match="Torch is required"):
        leaf.load_or_train_predictor(model_path=tmp_path / "x.pth", allow_fallback=False)
    with pytest.raises(ValueError, match="allow_legacy_fallback=True"):
        leaf.load_or_train_predictor(allow_fallback=True, allow_legacy_fallback=False)

    model, info = leaf.load_or_train_predictor(
        model_path=tmp_path / "x.pth",
        allow_fallback=True,
        allow_legacy_fallback=True,
    )
    assert model is None
    assert info["reason"] == "torch_unavailable"
    assert info["fallback"] is True
