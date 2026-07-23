# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for fusion kernel configuration leaf

"""Drive production fusion-kernel config parse/dump on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scpn_control.core.fusion_kernel as owner
import scpn_control.core.fusion_kernel_config as config_leaf
from scpn_control.core.fusion_kernel import FusionKernel


def _valid_raw_config() -> dict[str, object]:
    return {
        "reactor_name": "Test-Reactor",
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -2.0, "Z_max": 2.0},
        "grid_resolution": [12, 12],
        "physics": {"plasma_current_target": 1.0},
        "solver": {"boundary_variant": "fixed"},
        "coils": [],
    }


def test_public_config_symbols_bind_to_leaf() -> None:
    """Owner re-exports are the production config leaf objects."""
    assert owner.DimensionsConfig is config_leaf.DimensionsConfig
    assert owner.PhysicsConfig is config_leaf.PhysicsConfig
    assert owner.SolverConfig is config_leaf.SolverConfig
    assert owner.CoilConfig is config_leaf.CoilConfig
    assert owner.FusionKernelConfig is config_leaf.FusionKernelConfig
    assert owner._parse_fusion_kernel_config is config_leaf._parse_fusion_kernel_config
    assert owner._fusion_kernel_config_dump is config_leaf._fusion_kernel_config_dump


def test_parse_and_dump_round_trip() -> None:
    """Valid config parse yields a model; dump preserves reactor geometry."""
    model = config_leaf.parse_fusion_kernel_config(_valid_raw_config())
    assert model.reactor_name == "Test-Reactor"
    assert model.solver.boundary_variant == "fixed_boundary"
    dumped = config_leaf.fusion_kernel_config_dump(model)
    assert dumped["grid_resolution"] == [12, 12]
    assert dumped["dimensions"]["R_min"] == pytest.approx(1.0)
    assert dumped["solver"]["boundary_variant"] == "fixed_boundary"


def test_parse_rejects_non_object_root() -> None:
    """Fail closed when the config root is not a JSON object."""
    with pytest.raises(ValueError, match="JSON object"):
        config_leaf.parse_fusion_kernel_config([1, 2, 3])


def test_dimensions_reject_nonphysical_bounds() -> None:
    """Fail closed on non-finite or unordered dimension bounds."""
    with pytest.raises(ValueError, match="finite"):
        config_leaf.DimensionsConfig(R_min=float("inf"), R_max=3.0, Z_min=-1.0, Z_max=1.0)
    with pytest.raises(ValueError, match="R_min"):
        config_leaf.DimensionsConfig(R_min=3.0, R_max=1.0, Z_min=-1.0, Z_max=1.0)


def test_owner_load_config_uses_leaf_and_keeps_real_kernel(tmp_path: Path) -> None:
    """FusionKernel remains a real CONTROL surface that loads via the config leaf."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_valid_raw_config()), encoding="utf-8")
    kernel = FusionKernel(path)
    assert isinstance(kernel, FusionKernel)
    assert isinstance(kernel.config_model, config_leaf.FusionKernelConfig)
    assert kernel.config_model is not None
    assert kernel.cfg["reactor_name"] == "Test-Reactor"
    assert kernel.NR == 12
    assert kernel.NZ == 12


def test_owner_load_config_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Duplicate JSON keys fail closed through the production load path."""
    path = tmp_path / "dup.json"
    path.write_text(
        '{"reactor_name":"A","reactor_name":"B","dimensions":{"R_min":1,"R_max":2,"Z_min":-1,"Z_max":1},'
        '"grid_resolution":[8,8],"physics":{"plasma_current_target":1.0}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        FusionKernel(path)


def test_normalize_boundary_variant_accepts_aliases() -> None:
    """Boundary variant aliases normalise to the canonical free/fixed labels."""
    assert config_leaf.normalize_boundary_variant("free") == "free_boundary"
    assert config_leaf.normalize_boundary_variant("fixed-boundary") == "fixed_boundary"
    with pytest.raises(ValueError, match="boundary_variant"):
        config_leaf.normalize_boundary_variant("toroidal")
