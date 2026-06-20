# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Checkpoint Validation Branch Tests
"""Branch coverage for the checkpoint payload and finite-tree validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scpn_control.core.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    _validate_finite_tree,
    _validate_payload,
    save_checkpoint,
)


def _valid_payload() -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "episode": 0,
        "solver_state": {},
        "metrics": {},
    }


class TestFiniteTree:
    def test_rejects_non_string_dict_key(self) -> None:
        with pytest.raises(ValueError, match="keys must be strings"):
            _validate_finite_tree("solver_state", {123: 1.0})


class TestPayloadValidation:
    def test_rejects_non_object(self) -> None:
        with pytest.raises(ValueError, match="payload must be an object"):
            _validate_payload(["not", "a", "dict"])

    def test_rejects_wrong_schema_version(self) -> None:
        payload = _valid_payload()
        payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema_version must be"):
            _validate_payload(payload)

    def test_rejects_non_dict_solver_state(self) -> None:
        payload = _valid_payload()
        payload["solver_state"] = []
        with pytest.raises(ValueError, match="solver_state must be an object"):
            _validate_payload(payload)

    def test_rejects_non_dict_metrics(self) -> None:
        payload = _valid_payload()
        payload["metrics"] = []
        with pytest.raises(ValueError, match="metrics must be an object"):
            _validate_payload(payload)


class TestSaveCheckpointGuards:
    def test_rejects_non_dict_solver_state(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="solver_state must be an object"):
            save_checkpoint(tmp_path / "ckpt.json", solver_state=[], episode=0, metrics={})  # type: ignore[arg-type]

    def test_rejects_non_dict_metrics(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="metrics must be an object"):
            save_checkpoint(tmp_path / "ckpt.json", solver_state={}, episode=0, metrics=[])  # type: ignore[arg-type]
