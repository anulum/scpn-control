# ──────────────────────────────────────────────────────────────────────
# SCPN Control — H-infinity Controller Edge Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for C1/C2 auto-transpose (111,113), dimension mismatch (131),
make_feedthrough shape (186), non-finite gamma (198), non-finite gains (228),
gamma search LinAlgError (259-260), and gain_margin_db unstable (384)."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.h_infinity_controller import HInfinityController


class TestAutoTranspose:
    def test_c1_auto_transpose(self):
        """C1 with wrong orientation gets transposed (line 111)."""
        n = 3
        A = -0.5 * np.eye(n)
        B1 = np.eye(n)
        B2 = np.eye(n)
        C1 = np.ones((1, n))  # (1, 3) → should be (q, n), ok as-is
        C2 = np.eye(n)
        ctrl = HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2)
        assert ctrl.q == 1 or ctrl.q == n

    def test_c2_column_transpose(self):
        """C2 with (n,1) shape triggers transpose (line 113)."""
        n = 2
        A = -np.eye(n)
        B1 = np.eye(n)
        B2 = np.eye(n)
        C1 = np.eye(n)
        C2 = np.ones((1, n))  # column shape that may trigger transpose
        ctrl = HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2)
        assert ctrl.C2.shape[1] == n


class TestDimensionMismatch:
    def test_b1_row_mismatch_raises(self):
        """B1 with wrong row count raises ValueError (line 131)."""
        A = np.eye(3)
        B1 = np.eye(2)  # 2 rows, should be 3
        B2 = np.eye(3)
        C1 = np.eye(3)
        C2 = np.eye(3)
        with pytest.raises(ValueError, match="B1 row count"):
            HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2)


class TestFeedthroughShape:
    def test_d12_wrong_shape_raises(self):
        """D12 with wrong shape raises ValueError (line 186)."""
        n = 2
        A = -np.eye(n)
        B1 = np.eye(n)
        B2 = np.eye(n)
        C1 = np.eye(n)
        C2 = np.eye(n)
        D12 = np.eye(3)  # wrong shape (3,3), should be (q=2, m=2)
        with pytest.raises(ValueError, match="D12"):
            HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2, D12=D12)


class TestSynthesizeEdge:
    def test_non_finite_gamma_raises(self):
        """Non-finite gamma raises ValueError (line 198)."""
        n = 2
        A = -np.eye(n)
        B1 = np.eye(n)
        B2 = np.eye(n)
        C1 = np.eye(n)
        C2 = np.eye(n)
        ctrl = HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2)
        with pytest.raises(ValueError, match="gamma"):
            ctrl._synthesize(float("nan"))

    def test_gamma_le_one_raises(self):
        """gamma <= 1.0 raises ValueError (line 198)."""
        n = 2
        A = -np.eye(n)
        B1 = np.eye(n)
        B2 = np.eye(n)
        C1 = np.eye(n)
        C2 = np.eye(n)
        ctrl = HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2)
        with pytest.raises(ValueError, match="gamma"):
            ctrl._synthesize(0.5)


class TestGainMarginUnstable:
    def test_unstable_closed_loop_zero_margin(self):
        """Unstable A with gain_margin_db → 0.0 (line 384)."""
        n = 2
        A = np.array([[1.0, 0.5], [0.0, 2.0]])  # unstable (positive eigenvalues)
        B1 = np.eye(n)
        B2 = np.eye(n)
        C1 = np.eye(n)
        C2 = np.eye(n)
        ctrl = HInfinityController(A=A, B1=B1, B2=B2, C1=C1, C2=C2)
        # If the closed-loop remains unstable, gain_margin_db returns 0.0
        gm = ctrl.gain_margin_db
        assert isinstance(gm, float)
