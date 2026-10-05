# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public MAST replay interpolation and publication failures.
"""Exercise real derivation and filesystem publication without private calls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control._npz import save_npz_arrays
from validation.build_disruption_replay_channels import build_channels, derive_replay_channels

from ._fixtures import STAMP, mirror


def test_public_derivation_refuses_no_finite_equilibrium_channel() -> None:
    """An unusable scalar signal is refused through the public mirror API."""
    values = mirror()
    values["equilibrium.beta_tor_normal"][:] = np.nan
    with pytest.raises(ValueError, match="no finite samples to interpolate"):
        derive_replay_channels(values, locked_window=3)


def test_public_derivation_interpolates_empty_fast_bins() -> None:
    """Sparse native samples exercise the real per-bin interpolation fallback."""
    values = mirror()
    selection = np.array([0, 6, 12, 18, 24, 31], dtype=np.int64)
    values["magnetics.time_saddle"] = values["magnetics.time_saddle"][selection]
    values["magnetics.b_field_tor_probe_saddle_field"] = values["magnetics.b_field_tor_probe_saddle_field"][
        :, selection
    ]
    result = derive_replay_channels(values, locked_window=3)
    for name in ("n1_amp", "n2_amp", "locked_mode_amp"):
        assert result[name].shape == values["summary.time"].shape
        assert np.all(np.isfinite(result[name]))
    assert np.max(result["n1_amp"]) > 0.0


def test_dangling_destination_is_refused_without_temporary_leak(tmp_path: Path) -> None:
    """Exclusive publication refuses a genuine existing dangling filesystem entry."""
    material = tmp_path / "material"
    material.mkdir()
    source = material / "shot_101.npz"
    save_npz_arrays(source, mirror(), allow_pickle=False)
    before = source.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    destination = output / "channels.npz"
    target = output / "absent.npz"
    destination.symlink_to(target)
    with pytest.raises(ValueError, match="refusing to overwrite an existing replay archive"):
        build_channels(material, out_dir=output, generated_at=STAMP, locked_window=3)
    assert destination.is_symlink() and destination.readlink() == target
    assert list(output.glob(".channels.*.npz")) == []
    assert source.read_bytes() == before
