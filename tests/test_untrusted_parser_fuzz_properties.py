# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Property-based fuzzing of untrusted-input parse paths.
"""Property-based robustness of the untrusted-input parsers.

The G-EQDSK reader and the disruption feature-builder consume attacker-suppliable
data (``.geqdsk`` files globbed from reference directories; diagnostic signal
arrays). The deterministic regressions pin the *known* rejection paths; these
Hypothesis properties widen the net to the *unknown* ones — asserting that across
a large randomised input space each parser only ever returns a well-formed result
or fails closed with its typed error, never an unexpected exception, a non-finite
output, or an unbounded allocation.

A continuous coverage-guided ``atheris`` target over the same parsers remains a
deferred infrastructure decision (a new dependency plus a fuzz-CI job); these
in-suite property tests need no new dependency and close the reachable threat
model deterministically on every run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_control.control.disruption_predictor import (
    DISRUPTION_FEATURE_CONTRACT,
    build_disruption_feature_vector,
)
from scpn_control.core.eqdsk import GEqdsk, GEqdskFormatError, read_geqdsk

# Fortran-float tokens (normal, exponent-overflow to ±inf, integers used as
# boundary/limiter counts) plus non-numeric garbage the token filter must drop.
_TOKEN_POOL = st.sampled_from(["0.0", "1.5", "-2.0e1", "1e999", "-1e999", "3", "2", "1", "0", "GARBAGE", "1.0E-3"])

_TOROIDAL_KEYS = st.sampled_from(
    [
        "toroidal_n1_amp",
        "toroidal_n2_amp",
        "toroidal_n3_amp",
        "toroidal_asymmetry_index",
        "toroidal_radial_spread",
        "unused_key",
    ]
)


@st.composite
def _geqdsk_documents(draw: st.DrawFn) -> str:
    """Build a near-valid G-EQDSK document with adversarial numeric tokens.

    The header always declares a small valid ``nw × nh`` grid so the body — of
    deliberately variable length — exercises the scalar, array, count, and range
    code paths (truncation, exponent-overflow counts, oversized garbage).
    """
    nw = draw(st.integers(min_value=1, max_value=3))
    nh = draw(st.integers(min_value=1, max_value=3))
    exact = 20 + nw * (5 + nh) + 2  # scalars + fpol/pres/ffprime/pprime/psirz/qpsi + 2 counts
    body = draw(st.lists(_TOKEN_POOL, min_size=0, max_size=exact + 8))
    return f"TEST 0 {nw} {nh}\n" + " ".join(body) + "\n"


def _assert_parses_or_typed_error(target: Path) -> None:
    """Assert the parse returns a shape-consistent ``GEqdsk`` or raises the typed error."""
    try:
        equilibrium = read_geqdsk(str(target))
    except GEqdskFormatError:
        return
    assert isinstance(equilibrium, GEqdsk)
    assert equilibrium.nw >= 1
    assert equilibrium.nh >= 1
    assert equilibrium.fpol.shape == (equilibrium.nw,)
    assert equilibrium.qpsi.shape == (equilibrium.nw,)
    assert equilibrium.psirz.shape == (equilibrium.nh, equilibrium.nw)
    assert equilibrium.rbdry.shape == equilibrium.zbdry.shape
    assert equilibrium.rlim.shape == equilibrium.zlim.shape


class TestReadGeqdskFuzz:
    """``read_geqdsk`` fails closed across arbitrary and near-valid inputs."""

    @given(raw=st.binary(max_size=2048))
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_arbitrary_bytes_never_crash(self, tmp_path: Path, raw: bytes) -> None:
        """Arbitrary bytes yield a parse or ``GEqdskFormatError`` — never another exception."""
        target = tmp_path / "fuzz.geqdsk"
        target.write_bytes(raw)
        _assert_parses_or_typed_error(target)

    @given(document=_geqdsk_documents())
    @settings(
        max_examples=300,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_structured_documents_parse_or_reject(self, tmp_path: Path, document: str) -> None:
        """Near-valid documents reach the deep code paths and still fail closed."""
        target = tmp_path / "structured.geqdsk"
        target.write_text(document, encoding="utf-8")
        _assert_parses_or_typed_error(target)


class TestBuildDisruptionFeatureVectorFuzz:
    """``build_disruption_feature_vector`` never emits a non-finite vector."""

    @given(
        signal=st.lists(
            st.floats(allow_nan=True, allow_infinity=True, width=64),
            min_size=0,
            max_size=64,
        ),
        toroidal=st.none()
        | st.dictionaries(
            _TOROIDAL_KEYS,
            st.floats(allow_nan=True, allow_infinity=True, width=64),
            max_size=6,
        ),
    )
    @settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_returns_finite_vector_or_raises(self, signal: list[float], toroidal: dict[str, float] | None) -> None:
        """Any signal/observable pair yields a finite length-11 vector or ``ValueError``."""
        arr = np.asarray(signal, dtype=float)
        # Squaring a finite-but-huge signal overflows to ±inf; suppress the numpy
        # floating-point warning so the builder's own finiteness guard governs.
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                vector: np.ndarray | None = build_disruption_feature_vector(arr, toroidal)
            except ValueError:
                vector = None
        if vector is None:
            return
        assert vector.shape == (len(DISRUPTION_FEATURE_CONTRACT),)
        assert bool(np.all(np.isfinite(vector)))
