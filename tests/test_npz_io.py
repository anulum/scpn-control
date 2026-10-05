# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Typed NPZ archive writer tests.
"""Tests for the typed NPZ writer used by control artefact exports."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from scpn_control._npz import NpzSizeError, load_npz_capped, save_npz_arrays


def test_save_npz_arrays_writes_numpy_loadable_archive(tmp_path: Path) -> None:
    """Round-trip named arrays, including an ordinary allow_pickle member."""
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


@pytest.mark.parametrize("name", ["", ".", "..", "../weights", "nested/value", r"nested\\value"])
def test_save_npz_arrays_rejects_path_like_member_names(tmp_path: Path, name: str) -> None:
    """Refuse archive member names that are empty or denote paths."""
    with pytest.raises(ValueError, match="invalid NPZ array name"):
        save_npz_arrays(tmp_path / "bad.npz", {name: np.array([1.0])})


@pytest.mark.parametrize("compressed", [False, True])
def test_save_npz_arrays_rejects_object_arrays_without_pickle(tmp_path: Path, compressed: bool) -> None:
    """Keep the default object-array refusal for both compression modes."""
    with pytest.raises(ValueError, match="Object arrays cannot be saved"):
        save_npz_arrays(tmp_path / "object.npz", {"items": np.array([object()], dtype=object)}, compressed=compressed)


@pytest.mark.parametrize("compressed", [False, True])
def test_save_npz_arrays_preserves_native_numpy_archive_bytes(tmp_path: Path, compressed: bool) -> None:
    """Match NumPy's complete numeric archive bytes and compression metadata."""
    arrays = {
        "samples": np.arange(12, dtype=np.float64).reshape(3, 4),
        "shot_ids": np.asarray([123, 124], dtype=np.int64),
        "labels": np.asarray([True, False]),
        "metadata": np.asarray('{"authority":"ip_proxy"}'),
    }
    native = np.savez_compressed if compressed else np.savez
    expected = tmp_path / "native.npz"
    actual = tmp_path / "typed.npz"
    native(
        expected,
        samples=arrays["samples"],
        shot_ids=arrays["shot_ids"],
        labels=arrays["labels"],
        metadata=arrays["metadata"],
    )
    save_npz_arrays(actual, arrays, compressed=compressed, allow_pickle=True)
    assert actual.read_bytes() == expected.read_bytes()
    with zipfile.ZipFile(actual) as archive:
        assert {member.compress_type for member in archive.infolist()} == {
            zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        }
    with np.load(actual, allow_pickle=False) as archive:
        assert archive.files == list(arrays)
        for name, value in arrays.items():
            np.testing.assert_array_equal(archive[name], value)
            assert archive[name].dtype == value.dtype


@pytest.mark.parametrize("compressed", [False, True])
def test_save_npz_arrays_explicit_pickle_matches_native_archive(tmp_path: Path, compressed: bool) -> None:
    """Preserve native object-array export when the caller explicitly permits it."""
    values = np.asarray([{"shot_id": 123}], dtype=object)
    native = np.savez_compressed if compressed else np.savez
    expected = tmp_path / "native-object.npz"
    actual = tmp_path / "typed-object.npz"
    native(expected, values=values)
    save_npz_arrays(actual, {"values": values}, compressed=compressed, allow_pickle=True)
    assert actual.read_bytes() == expected.read_bytes()
    with np.load(actual, allow_pickle=True) as archive:
        assert archive["values"].tolist() == values.tolist()


def test_load_npz_capped_loads_a_valid_archive(tmp_path: Path) -> None:
    """Load the complete array inventory from an archive within the size cap."""
    path = tmp_path / "shot.npz"
    save_npz_arrays(path, {"ip": np.array([15e6]), "psi": np.arange(6.0).reshape(2, 3)})
    with load_npz_capped(path) as data:
        np.testing.assert_array_equal(data["psi"], np.arange(6.0).reshape(2, 3))
        assert set(data.keys()) == {"ip", "psi"}


def test_load_npz_capped_rejects_archive_over_the_decompressed_cap(tmp_path: Path) -> None:
    """Refuse declared expanded bytes before loading array members."""
    # A decompression bomb declares (or expands to) far more than its packed size.
    # Auditing the central-directory sizes refuses it before np.load allocates.
    path = tmp_path / "big.npz"
    save_npz_arrays(path, {"payload": np.zeros(1024, dtype=np.float64)})  # 8192 bytes declared
    with pytest.raises(NpzSizeError, match="exceeds cap"):
        load_npz_capped(path, max_decompressed_bytes=1024)
