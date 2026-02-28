# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Compiler Internal Function Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for _encode_weight_matrix_packed and _resolve_git_sha fallback."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from scpn_control.scpn.compiler import (
    _encode_weight_matrix_packed,
    _resolve_git_sha,
)


class TestResolveGitSha:
    def test_returns_seven_char_string(self):
        sha = _resolve_git_sha()
        assert len(sha) == 7

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SCPN_GIT_SHA", "abc1234")
        assert _resolve_git_sha() == "abc1234"

    def test_subprocess_failure_returns_fallback(self, monkeypatch):
        for key in ("SCPN_GIT_SHA", "GITHUB_SHA", "CI_COMMIT_SHA"):
            monkeypatch.delenv(key, raising=False)

        def _bad_run(*args, **kwargs):
            raise OSError("git not found")

        monkeypatch.setattr(subprocess, "run", _bad_run)
        assert _resolve_git_sha() == "0000000"

    def test_subprocess_empty_output_returns_fallback(self, monkeypatch):
        for key in ("SCPN_GIT_SHA", "GITHUB_SHA", "CI_COMMIT_SHA"):
            monkeypatch.delenv(key, raising=False)

        class _FakeResult:
            stdout = "  \n"

        def _empty_run(*args, **kwargs):
            return _FakeResult()

        monkeypatch.setattr(subprocess, "run", _empty_run)
        assert _resolve_git_sha() == "0000000"


class TestEncodeWeightMatrixPacked:
    def test_output_shape(self):
        W = np.array([[0.5, 0.8], [0.2, 0.9]])
        packed = _encode_weight_matrix_packed(W, bitstream_length=128, seed=42)
        n_words = int(np.ceil(128 / 64))
        assert packed.shape == (2, 2, n_words)
        assert packed.dtype == np.uint64

    def test_deterministic(self):
        W = np.random.default_rng(0).random((3, 4))
        p1 = _encode_weight_matrix_packed(W, bitstream_length=64, seed=7)
        p2 = _encode_weight_matrix_packed(W, bitstream_length=64, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_zero_weight_gives_zero_bits(self):
        W = np.zeros((2, 2))
        packed = _encode_weight_matrix_packed(W, bitstream_length=64, seed=0)
        assert np.all(packed == 0)

    def test_one_weight_gives_all_ones(self):
        W = np.ones((1, 1))
        packed = _encode_weight_matrix_packed(W, bitstream_length=64, seed=0)
        assert packed[0, 0, 0] == np.uint64(0xFFFFFFFFFFFFFFFF)

    def test_clipping_out_of_range(self):
        W = np.array([[2.0, -1.0]])
        p = _encode_weight_matrix_packed(W, bitstream_length=64, seed=0)
        assert p.shape == (1, 2, 1)
        assert p[0, 0, 0] == np.uint64(0xFFFFFFFFFFFFFFFF)
        assert p[0, 1, 0] == np.uint64(0)

    def test_non_multiple_of_64_bitstream(self):
        W = np.array([[0.5]])
        packed = _encode_weight_matrix_packed(W, bitstream_length=100, seed=0)
        n_words = int(np.ceil(100 / 64))
        assert packed.shape == (1, 1, n_words)

    def test_statistical_fidelity(self):
        """Mean bit density ~ weight probability within tolerance."""
        rng = np.random.default_rng(42)
        W = np.array([[0.3]])
        packed = _encode_weight_matrix_packed(W, bitstream_length=10000, seed=42)
        total_ones = 0
        for word_idx in range(packed.shape[2]):
            total_ones += bin(int(packed[0, 0, word_idx])).count("1")
        observed_rate = total_ones / (packed.shape[2] * 64)
        assert abs(observed_rate - 0.3) < 0.05
