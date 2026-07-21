# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Typed NPZ archive writer tests.
"""Tests for the typed NPZ writer used by control artefact exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control._npz import NpzSizeError, load_npz_capped, save_npz_arrays


def test_save_npz_arrays_writes_numpy_loadable_archive(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "weights.npz"
    save_npz_arrays(
        path,
        {
            "allow_pickle": np.array([0.0]),
            "gain": np.array([5.0]),
            "weights": np.arange(6.0).reshape(2, 3),
        },
    )

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["allow_pickle"], np.array([0.0]))
        np.testing.assert_array_equal(archive["gain"], np.array([5.0]))
        np.testing.assert_array_equal(archive["weights"], np.arange(6.0).reshape(2, 3))


@pytest.mark.parametrize("name", ["", "../weights", "nested/value", r"nested\\value"])
def test_save_npz_arrays_rejects_path_like_member_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid NPZ array name"):
        save_npz_arrays(tmp_path / "bad.npz", {name: np.array([1.0])})


def test_save_npz_arrays_rejects_object_arrays_without_pickle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Object arrays cannot be saved"):
        save_npz_arrays(tmp_path / "object.npz", {"items": np.array([object()], dtype=object)})


def test_load_npz_capped_loads_a_valid_archive(tmp_path: Path) -> None:
    path = tmp_path / "shot.npz"
    save_npz_arrays(path, {"ip": np.array([15e6]), "psi": np.arange(6.0).reshape(2, 3)})
    with load_npz_capped(path) as data:
        np.testing.assert_array_equal(data["psi"], np.arange(6.0).reshape(2, 3))
        assert set(data.keys()) == {"ip", "psi"}


def test_load_npz_capped_rejects_archive_over_the_decompressed_cap(tmp_path: Path) -> None:
    # A decompression bomb declares (or expands to) far more than its packed size.
    # Auditing the central-directory sizes refuses it before np.load allocates.
    path = tmp_path / "big.npz"
    save_npz_arrays(path, {"payload": np.zeros(1024, dtype=np.float64)})  # 8192 bytes declared
    with pytest.raises(NpzSizeError, match="exceeds cap"):
        load_npz_capped(path, max_decompressed_bytes=1024)
