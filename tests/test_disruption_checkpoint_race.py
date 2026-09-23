# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption checkpoint byte-binding test.

"""Exercise a path swap while the public checkpoint loader uses real Torch."""

import hashlib
from pathlib import Path
from typing import Any

import pytest

import scpn_control.control.disruption_checkpoint as checkpoint
from scpn_control.control.disruption_predictor import DisruptionTransformer

torch = pytest.importorskip("torch")


def test_loaded_weights_match_the_digest_when_path_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rename between hashing and Torch load cannot substitute other weights."""
    torch.manual_seed(11)
    original_model = DisruptionTransformer(seq_len=16)
    original_path = tmp_path / "pinned.pth"
    torch.save({"state_dict": original_model.state_dict(), "seq_len": 16}, original_path)
    expected_digest = hashlib.sha256(original_path.read_bytes()).hexdigest()

    torch.manual_seed(29)
    other_model = DisruptionTransformer(seq_len=16)
    replacement_path = tmp_path / "replacement.pth"
    torch.save({"state_dict": other_model.state_dict(), "seq_len": 16}, replacement_path)
    assert original_path.read_bytes() != replacement_path.read_bytes()

    actual_load = torch.load
    swapped = False

    def swap_then_load(source: Any, *args: Any, **kwargs: Any) -> Any:
        """Replace the pathname immediately before real Torch deserialisation."""
        nonlocal swapped
        replacement_path.replace(original_path)
        swapped = True
        return actual_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", swap_then_load)
    loaded, info = checkpoint.load_or_train_predictor(
        model_path=original_path,
        seq_len=16,
        train_if_missing=False,
        expected_sha256=expected_digest,
        require_pin=True,
    )

    assert swapped
    assert info["weights_sha256"] == expected_digest
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() != expected_digest
    for key, expected_tensor in original_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected_tensor)
