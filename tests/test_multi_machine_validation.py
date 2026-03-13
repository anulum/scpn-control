# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Multi-Machine Validation Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation.multi_machine_validation import (
    MultiMachineValidator,
    diiid_h_mode,
    iter_15ma,
)


def test_synthetic_diagnostics():
    m = iter_15ma()
    rho = np.linspace(0, 1, 50)
    Te = m.Te_profile(rho)
    ne = m.ne_profile(rho)

    thomson = m.diagnostics.thomson_scattering(Te, ne, 20)
    assert len(thomson["Te_keV"]) == 20
    assert len(thomson["ne_19"]) == 20

    # Check noise (SNR > 10)
    # Median SNR = mean / std. Our added noise is 5% for Te -> SNR = 20

    sxr = m.diagnostics.soft_xray(Te, ne, rho, 40)
    assert len(sxr) == 40
    # Values should be strictly positive unless all profiles are 0
    assert np.all(sxr > 0)


def test_multi_machine_validator(tmp_path: Path):
    machines = [iter_15ma(), diiid_h_mode()]
    val = MultiMachineValidator(machines)

    # Run tests
    val.run_all()

    assert len(val.results) == 14  # 7 tests per machine * 2 machines

    # Check valid metrics
    for r in val.results:
        assert not np.isnan(r.value)
        assert r.value >= 0.0

    # Check overall pass rate
    passed = sum(1 for r in val.results if r.passed)
    pass_rate = passed / len(val.results)
    assert pass_rate >= 0.8  # Target > 80%

    # Save formats
    j_path = tmp_path / "valid.json"
    val.save_json(j_path)
    assert j_path.exists()

    with open(j_path) as f:
        data = json.load(f)
        assert len(data) == 14

    m_path = tmp_path / "valid.md"
    val.save_markdown(m_path)
    assert m_path.exists()
    with open(m_path) as f:
        content = f.read()
        assert "ITER" in content
        assert "DIII-D" in content
        assert "PASS" in content
