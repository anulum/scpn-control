# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary acceptance fixture construction.
"""Fixture construction of the free-boundary acceptance campaign on the real kernel.

The template samples its shape targets from the real ``FusionKernel``; a
fixture without target points has nothing to sample and is refused. The
tracking-configuration merge skips empty parts and returns ``None`` when no
part declares anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation import free_boundary_acceptance_presets as presets


def test_template_samples_one_flux_target_per_declared_point(tmp_path: Path) -> None:
    """The real two-outer-step solve yields one finite flux target per configured target point."""
    cfg = presets._build_tracking_template(tmp_path)
    points = cfg["free_boundary"]["target_flux_points"]
    values = cfg["free_boundary"]["target_flux_values"]
    assert len(values) == len(points) == 2
    assert all(isinstance(value, float) and value == value for value in values)
    assert (tmp_path / "template.json").is_file()


def test_template_refuses_a_fixture_without_target_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without target flux points the real coilset has nothing to sample, so the template is refused."""
    base = presets._base_tracking_config()
    del base["free_boundary"]["target_flux_points"]
    monkeypatch.setattr(presets, "_base_tracking_config", lambda: base)
    with pytest.raises(ValueError, match="^coilset produced no target flux points to sample$"):
        presets._build_tracking_template(tmp_path)


def test_tracking_merge_skips_empty_parts_and_lets_later_keys_win() -> None:
    """None and empty parts contribute nothing; a later part replaces an earlier key."""
    merged = presets._merge_tracking_cfg(None, {"a": 1, "b": 2}, {}, {"b": 3})
    assert merged == {"a": 1, "b": 3}


def test_tracking_merge_of_only_empty_parts_is_none() -> None:
    """With no declared field the merge returns None, so the fixture carries no tracking section."""
    assert presets._merge_tracking_cfg(None, {}) is None
    assert presets._merge_tracking_cfg() is None
