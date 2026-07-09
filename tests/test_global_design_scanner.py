# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GlobalDesignExplorer tests.
"""Tests for the bounded global design scanner."""

from __future__ import annotations

import pytest

from scpn_control.core.global_design_scanner import DesignScannerConfig, GlobalDesignExplorer


def test_evaluate_design_returns_disruption_contract_metrics() -> None:
    """The scanner returns the metrics consumed by disruption episodes."""

    explorer = GlobalDesignExplorer("contract-test")
    metrics = explorer.evaluate_design(r_maj=1.4, b_t=10.0, ip=6.0)

    assert metrics["Q"] > 0.0
    assert metrics["P_fusion_MW"] > 0.0
    assert metrics["neutron_wall_load_MW_m2"] > 0.0
    assert metrics["cost_index"] > 0.0
    assert metrics["minor_radius_m"] > 0.0
    assert metrics["volume_m3"] > 0.0
    assert metrics["greenwald_fraction"] == pytest.approx(0.82)
    assert 0.0 <= metrics["beta_p_proxy"] <= 10.0


def test_mapping_configuration_changes_auxiliary_power() -> None:
    """Scalar configuration mappings alter the bounded design estimate."""

    baseline = GlobalDesignExplorer().evaluate_design(r_maj=1.45, b_t=11.0, ip=6.5)
    configured = GlobalDesignExplorer({"auxiliary_power_mw": 120.0}).evaluate_design(
        r_maj=1.45,
        b_t=11.0,
        ip=6.5,
    )

    assert configured["Q"] < baseline["Q"]


def test_from_mapping_rejects_nonpositive_values() -> None:
    """Configuration values must be positive finite scalars."""

    with pytest.raises(ValueError, match="aspect_ratio must be > 0"):
        DesignScannerConfig.from_mapping({"aspect_ratio": 0.0})


def test_evaluate_design_rejects_nonpositive_inputs() -> None:
    """Design coordinates must stay in positive physical domains."""

    explorer = GlobalDesignExplorer()
    with pytest.raises(ValueError, match="b_t must be > 0"):
        explorer.evaluate_design(r_maj=1.4, b_t=0.0, ip=6.0)
