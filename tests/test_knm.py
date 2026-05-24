# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Knm
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Knm Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.phase.knm import KnmSpec, build_knm_paper27


class TestKnmSpecValidation:
    def test_rejects_nonfinite_coupling_matrix(self):
        K = np.eye(3)
        K[0, 1] = np.nan

        with pytest.raises(ValueError, match="K"):
            KnmSpec(K=K)

    def test_rejects_negative_coupling_matrix(self):
        K = np.eye(3)
        K[1, 2] = -1e-6

        with pytest.raises(ValueError, match="K"):
            KnmSpec(K=K)

    def test_rejects_nonfinite_alpha(self):
        alpha = np.zeros((3, 3))
        alpha[0, 0] = np.inf

        with pytest.raises(ValueError, match="alpha"):
            KnmSpec(K=np.eye(3), alpha=alpha)

    def test_rejects_negative_zeta(self):
        zeta = np.array([0.0, -1e-6, 0.1])

        with pytest.raises(ValueError, match="zeta"):
            KnmSpec(K=np.eye(3), zeta=zeta)


class TestPaper27BuilderValidation:
    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"L": 0}, "L"),
            ({"K_base": -0.1}, "K_base"),
            ({"K_alpha": -0.1}, "K_alpha"),
            ({"zeta_uniform": -1e-6}, "zeta_uniform"),
        ],
    )
    def test_build_knm_paper27_rejects_unphysical_inputs(self, kwargs, field):
        with pytest.raises(ValueError, match=field):
            build_knm_paper27(**kwargs)


class TestPaper27BuilderInvariants:
    def test_build_knm_paper27_returns_symmetric_nonnegative_matrix(self):
        spec = build_knm_paper27(L=8)
        K = np.asarray(spec.K)

        np.testing.assert_allclose(K, K.T, atol=1e-14)
        assert np.all(np.isfinite(K))
        assert np.all(K >= 0.0)

    def test_build_knm_paper27_preserves_anchor_contracts(self):
        spec = build_knm_paper27(L=5)
        K = np.asarray(spec.K)

        assert K[0, 1] == pytest.approx(0.302)
        assert K[1, 2] == pytest.approx(0.201)
        assert K[2, 3] == pytest.approx(0.252)
        assert K[3, 4] == pytest.approx(0.154)
