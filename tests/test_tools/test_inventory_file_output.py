# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Guarded file publisher public contract tests
"""Exercise byte publication using the complete maintained planning matrix."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from tools.evidence_gap_matrix import ROOT, build_evidence_gap_matrix
from tools.inventory_file_output import InventoryOutputError, publish_guarded_outputs
from tools.report_inventory_output import InventoryOutputError as LegacyInventoryOutputError


def test_generic_writer_publishes_real_matrix_without_a_freshness_adapter(tmp_path: Path) -> None:
    """Complete planning payloads publish directly while preserving consumed registry bytes."""
    registry = tmp_path / "registry.json"
    registry.write_bytes((ROOT / "validation/physics_traceability.json").read_bytes())
    before = registry.read_bytes()
    matrix = build_evidence_gap_matrix(registry)
    first, second = tmp_path / "nested/matrix.json", tmp_path / "nested/matrix.md"
    publish_guarded_outputs(
        (
            (first, (json.dumps(matrix.to_dict(), sort_keys=True) + "\n").encode()),
            (second, matrix.to_markdown().encode()),
        ),
        protected_files=(registry,),
    )
    assert json.loads(first.read_bytes()) == matrix.to_dict()
    assert second.read_text() == matrix.to_markdown()
    assert registry.read_bytes() == before
    assert sorted(path.name for path in first.parent.iterdir()) == ["matrix.json", "matrix.md"]
    publish_guarded_outputs((), protected_files=(registry,))
    assert registry.read_bytes() == before


def test_guarded_refusal_retains_legacy_exception_type_and_pickle_address(tmp_path: Path) -> None:
    """Actual input-alias refusal remains catchable and serialisable by old consumers."""
    registry = tmp_path / "registry.json"
    registry.write_bytes((ROOT / "validation/physics_traceability.json").read_bytes())
    before = registry.read_bytes()
    with pytest.raises(LegacyInventoryOutputError) as captured:
        publish_guarded_outputs(((registry, b"replacement refused"),), protected_files=(registry,))
    assert LegacyInventoryOutputError is InventoryOutputError
    assert type(captured.value).__module__ == "tools.report_inventory_output"
    restored = pickle.loads(pickle.dumps(captured.value))
    assert type(restored) is LegacyInventoryOutputError
    assert str(restored) == str(captured.value)
    assert restored.recovery_paths == ()
    assert registry.read_bytes() == before


@pytest.mark.parametrize("kind", ["outside", "occupied"])
def test_public_exclusive_names_refuse_invalid_membership_or_existing_files(tmp_path: Path, kind: str) -> None:
    """Exclusive declaration checks precede modification of a mutable predecessor."""
    mutable, receipt = tmp_path / "mutable.json", tmp_path / "receipt.json"
    mutable.write_bytes(b"existing mutable bytes")
    exclusive = receipt
    if kind == "outside":
        exclusive = tmp_path / "unrequested.json"
    else:
        receipt.write_bytes(b"other writer receipt")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(InventoryOutputError):
        publish_guarded_outputs(
            ((mutable, b"new mutable"), (receipt, b"new receipt")), protected_files=(), exclusive_files=(exclusive,)
        )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
