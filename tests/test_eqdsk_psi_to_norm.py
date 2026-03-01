# ──────────────────────────────────────────────────────────────────────
# SCPN Control — eqdsk psi_to_norm and to_config edge tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for GEqdsk.psi_to_norm (line 103) and to_config."""

from __future__ import annotations

import numpy as np

from scpn_control.core.eqdsk import GEqdsk


def _make_eqdsk() -> GEqdsk:
    nw, nh = 5, 5
    return GEqdsk(
        description="test",
        nw=nw,
        nh=nh,
        rdim=2.0,
        zdim=4.0,
        rleft=5.2,
        rcentr=6.2,
        bcentr=5.3,
        rmaxis=6.2,
        zmaxis=0.0,
        simag=-1.0,
        sibry=1.0,
        current=15e6,
        psirz=np.zeros((nh, nw)),
        fpol=np.ones(nw),
        pres=np.zeros(nw),
        ffprime=np.zeros(nw),
        pprime=np.zeros(nw),
        qpsi=np.ones(nw) * 2.0,
        zmid=0.0,
    )


class TestPsiToNorm:
    def test_psi_to_norm_range(self):
        """psi_to_norm maps simag→0 and sibry→1 (line 103)."""
        eq = _make_eqdsk()
        psi = np.array([eq.simag, 0.0, eq.sibry])
        result = eq.psi_to_norm(psi)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_psi_to_norm_array(self):
        eq = _make_eqdsk()
        psi = np.linspace(eq.simag, eq.sibry, 20)
        result = eq.psi_to_norm(psi)
        assert result.shape == (20,)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(result[-1], 1.0, atol=1e-14)


class TestToConfig:
    def test_to_config_structure(self):
        eq = _make_eqdsk()
        cfg = eq.to_config("test_reactor")
        assert cfg["reactor_name"] == "test_reactor"
        assert "dimensions" in cfg
        assert "physics" in cfg
        assert "solver" in cfg
        assert cfg["grid_resolution"] == [5, 5]
