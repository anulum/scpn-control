# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FUSION-owned IDA equilibrium adapter tests
"""Behaviour and fail-closed tests for the CONTROL IDA equilibrium facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import scpn_control.core.ida_equilibrium_adapter as adapter
from scpn_control.core.ida_equilibrium_adapter import (
    IDAEquilibriumAdapterError,
    IDAEquilibriumRequest,
    ida_equilibrium_vjp,
    solve_ida_equilibrium,
)


class FakeBackend:
    """Deterministic stand-in that exercises the adapter boundary, not solver physics."""

    version = "4.0.0"
    jax_version = "0.6.2"
    jaxlib_version = "0.6.2"
    solver_file_sha256 = "a" * 64
    profile_basis_file_sha256 = "b" * 64
    x64_enabled = True

    def __init__(self) -> None:
        self.design_calls: list[tuple[int, int, int]] = []
        self.solve_calls: list[dict[str, object]] = []
        self.output_mode = "valid"
        self.gradient_mode = "valid"

    def asarray(self, value: np.ndarray) -> object:
        """Return a float64 stand-in for a JAX array."""
        return np.asarray(value, dtype=np.float64)

    def design_matrix(self, query: np.ndarray, n_coeff: int, degree: int) -> object:
        """Return a deterministic fixed profile-design matrix."""
        self.design_calls.append((query.size, n_coeff, degree))
        matrix = np.zeros((query.size, n_coeff), dtype=np.float64)
        matrix[:, 0] = 1.0
        return matrix

    def evaluate_profile(self, coefficients: object, design_matrix: object) -> object:
        """Apply the fixed profile basis to compact coefficients."""
        return np.asarray(design_matrix) @ np.asarray(coefficients)

    def solve(
        self,
        coil_current: object,
        pprime_values: object,
        ffprime_values: object,
        r_grid: object,
        z_grid: object,
        coil_r: object,
        coil_z: object,
        psin_knots: object,
        n_picard: int,
        picard_omega: float,
    ) -> object:
        """Return a deterministic flux field or the selected malformed field."""
        r = np.asarray(r_grid)
        z = np.asarray(z_grid)
        self.solve_calls.append(
            {
                "coil_current": np.asarray(coil_current),
                "pprime_values": np.asarray(pprime_values),
                "ffprime_values": np.asarray(ffprime_values),
                "coil_r": np.asarray(coil_r),
                "coil_z": np.asarray(coil_z),
                "psin_knots": np.asarray(psin_knots),
                "n_picard": n_picard,
                "picard_omega": picard_omega,
            }
        )
        if self.output_mode == "shape":
            return np.zeros((z.size, r.size - 1))
        if self.output_mode == "nan":
            return np.full((z.size, r.size), np.nan)
        if self.output_mode == "object":
            return object()
        return np.add.outer(z, r) + 1.0e-12 * (np.sum(coil_current) + np.sum(pprime_values) + np.sum(ffprime_values))

    def vjp(
        self,
        function: Callable[[object, object, object], object],
        coil_current: object,
        pprime_coefficients: object,
        ffprime_coefficients: object,
    ) -> tuple[object, Callable[[object], tuple[object, object, object]]]:
        """Return the model value and deterministic three-input pullback."""
        value = function(coil_current, pprime_coefficients, ffprime_coefficients)

        def pullback(cotangent: object) -> tuple[object, object, object]:
            scale = float(np.sum(cotangent))
            if self.gradient_mode == "shape":
                return np.ones(1), np.ones_like(pprime_coefficients), np.ones_like(ffprime_coefficients)
            if self.gradient_mode == "nan":
                return (
                    np.full_like(coil_current, np.nan),
                    np.ones_like(pprime_coefficients),
                    np.ones_like(ffprime_coefficients),
                )
            return (
                np.full_like(coil_current, scale),
                np.full_like(pprime_coefficients, 2.0 * scale),
                np.full_like(ffprime_coefficients, 3.0 * scale),
            )

        return value, pullback


def _request(**overrides: object) -> IDAEquilibriumRequest:
    values: dict[str, object] = {
        "r_grid_m": np.linspace(1.0, 2.0, 49),
        "z_grid_m": np.linspace(-1.0, 1.0, 49),
        "coil_r_m": np.array([0.8, 0.9, 2.1, 2.2]),
        "coil_z_m": np.array([-0.8, 0.8, -0.8, 0.8]),
        "coil_current_a": np.array([1.0e5, -1.0e5, 8.0e4, -8.0e4]),
        "psin_knots": np.linspace(0.0, 1.0, 9),
        "pprime_coefficients_pa_per_wb": np.linspace(-2.0e4, 0.0, 6),
        "ffprime_coefficients_t2_m2_per_wb": np.linspace(0.8, 0.1, 7),
        "profile_degree": 3,
        "n_picard": 40,
        "picard_omega": 0.6,
    }
    values.update(overrides)
    return IDAEquilibriumRequest(**values)


def test_request_normalises_and_freezes_caller_arrays() -> None:
    """Copy mutable caller arrays into read-only float64 storage."""
    source = np.linspace(1.0, 2.0, 49)
    request = _request(r_grid_m=source)
    source[0] = 99.0

    assert request.r_grid_m[0] == 1.0
    assert request.r_grid_m.dtype == np.float64
    assert request.r_grid_m.flags.c_contiguous
    assert request.r_grid_m.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        request.r_grid_m[0] = 2.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"r_grid_m": object()}, "coercible"),
        ({"r_grid_m": np.zeros((2, 2))}, "one-dimensional"),
        ({"r_grid_m": np.array([1.0, np.nan])}, "finite"),
        ({"profile_degree": True}, "profile_degree must be an integer"),
        ({"profile_degree": 1.5}, "profile_degree must be an integer"),
        ({"profile_degree": 0}, "between 1 and 5"),
        ({"profile_degree": 6}, "between 1 and 5"),
        ({"n_picard": False}, "n_picard must be an integer"),
        ({"n_picard": 1.5}, "n_picard must be an integer"),
        ({"n_picard": 0}, "between 1 and 256"),
        ({"n_picard": 257}, "between 1 and 256"),
        ({"picard_omega": np.nan}, "finite and in"),
        ({"picard_omega": True}, "finite and in"),
        ({"picard_omega": "bad"}, "finite and in"),
        ({"picard_omega": 0.0}, "finite and in"),
        ({"picard_omega": 1.1}, "finite and in"),
        ({"r_grid_m": np.linspace(1.0, 2.0, 48)}, "between 49 and 257"),
        ({"z_grid_m": np.linspace(-1.0, 1.0, 258)}, "between 49 and 257"),
        ({"r_grid_m": np.linspace(2.0, 1.0, 49)}, "strictly increasing"),
        ({"z_grid_m": np.linspace(-1.0, 1.0, 49) ** 3}, "uniformly spaced"),
        ({"r_grid_m": np.linspace(-1.0, 1.0, 49)}, "strictly positive"),
        ({"coil_z_m": np.array([0.0])}, "identical lengths"),
        (
            {
                "coil_r_m": np.ones(65),
                "coil_z_m": np.zeros(65),
                "coil_current_a": np.zeros(65),
            },
            "at most 64",
        ),
        ({"coil_r_m": np.array([-1.0, 1.0, 2.0, 3.0])}, "strictly positive"),
        ({"psin_knots": np.array([0.0, 0.5, 0.5, 1.0])}, "strictly increasing"),
        ({"psin_knots": np.linspace(0.1, 1.0, 9)}, "span exactly"),
        ({"pprime_coefficients_pa_per_wb": np.ones(3)}, "between 4 and 64"),
        ({"ffprime_coefficients_t2_m2_per_wb": np.ones(65)}, "between 4 and 64"),
    ],
)
def test_request_rejects_malformed_or_unbounded_inputs(overrides: dict[str, object], message: str) -> None:
    """Reject every malformed, non-finite, or resource-unbounded request class."""
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


def test_solve_binds_upstream_source_bytes_and_false_claim_boundary() -> None:
    """Bind solver output to request and upstream digests without claim promotion."""
    request = _request()
    backend = FakeBackend()

    result = adapter._solve_with_backend(request, backend)

    assert result.psi_wb.shape == (49, 49)
    assert result.psi_wb.flags.writeable is False
    assert backend.design_calls == [(9, 6, 3), (9, 7, 3)]
    assert backend.solve_calls[0]["n_picard"] == 40
    assert backend.solve_calls[0]["picard_omega"] == 0.6
    assert result.evidence.schema_version == adapter.IDA_ADAPTER_SCHEMA
    assert result.evidence.status == adapter.IDA_ADAPTER_STATUS
    assert result.evidence.upstream_version == "4.0.0"
    assert result.evidence.jax_version == "0.6.2"
    assert result.evidence.jaxlib_version == "0.6.2"
    assert result.evidence.solver_file_sha256 == "a" * 64
    assert result.evidence.profile_basis_file_sha256 == "b" * 64
    assert result.evidence.psi_sha256 == adapter._array_digest(result.psi_wb)
    assert result.evidence.request_sha256 == adapter._request_digest(request)
    assert dict(result.evidence.claim_boundary) == {field: False for field in adapter._CLAIM_FIELDS}
    assert result.evidence.blockers == adapter._BLOCKERS
    evidence_dict = result.evidence.as_dict()
    assert evidence_dict["grid_shape"] == [49, 49]
    assert evidence_dict["claim_boundary"] == dict(result.evidence.claim_boundary)


def test_request_digest_is_semantic_and_deterministic() -> None:
    """Make request and array digests stable while detecting parameter drift."""
    request = _request()
    equal = _request(r_grid_m=request.r_grid_m.copy())
    changed = replace(request, n_picard=41)

    assert adapter._request_digest(request) == adapter._request_digest(equal)
    assert adapter._request_digest(request) != adapter._request_digest(changed)
    assert adapter._array_digest(request.r_grid_m) == adapter._array_digest(request.r_grid_m.copy())


@pytest.mark.parametrize(("mode", "message"), [("shape", "shape"), ("nan", "non-finite"), ("object", "not a float64")])
def test_solve_rejects_invalid_upstream_fields(mode: str, message: str) -> None:
    """Reject wrong-shape, non-finite, and non-array solver output."""
    backend = FakeBackend()
    backend.output_mode = mode
    with pytest.raises(IDAEquilibriumAdapterError, match=message):
        adapter._solve_with_backend(_request(), backend)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda backend: setattr(backend, "version", "3.9.11"), "requires scpn-fusion 4.x"),
        (lambda backend: setattr(backend, "version", "release"), "semantic numeric"),
        (lambda backend: setattr(backend, "version", "4"), "semantic numeric"),
        (lambda backend: setattr(backend, "x64_enabled", False), "requires JAX FP64"),
        (lambda backend: setattr(backend, "jax_version", ""), "runtime versions"),
        (lambda backend: setattr(backend, "solver_file_sha256", "bad"), "solver_file_sha256"),
        (lambda backend: setattr(backend, "profile_basis_file_sha256", "G" * 64), "profile_basis_file_sha256"),
    ],
)
def test_backend_admission_rejects_version_precision_and_provenance(
    mutation: Callable[[FakeBackend], None], message: str
) -> None:
    """Reject incompatible versions, non-FP64 execution, and invalid source hashes."""
    backend = FakeBackend()
    mutation(backend)
    with pytest.raises(IDAEquilibriumAdapterError, match=message):
        adapter._solve_with_backend(_request(), backend)


def test_vjp_returns_exactly_shaped_read_only_gradients() -> None:
    """Preserve the three differentiated input shapes and cotangent digest."""
    request = _request()
    cotangent = np.ones((49, 49), dtype=np.float64)
    backend = FakeBackend()

    result = adapter._vjp_with_backend(request, cotangent, backend)

    assert result.equilibrium.psi_wb.shape == cotangent.shape
    assert result.coil_current_gradient.shape == request.coil_current_a.shape
    assert result.pprime_coefficient_gradient.shape == request.pprime_coefficients_pa_per_wb.shape
    assert result.ffprime_coefficient_gradient.shape == request.ffprime_coefficients_t2_m2_per_wb.shape
    assert np.all(result.coil_current_gradient == cotangent.size)
    assert np.all(result.pprime_coefficient_gradient == 2.0 * cotangent.size)
    assert np.all(result.ffprime_coefficient_gradient == 3.0 * cotangent.size)
    assert result.coil_current_gradient.flags.writeable is False
    assert result.cotangent_sha256 == adapter._array_digest(cotangent)


@pytest.mark.parametrize(
    ("cotangent", "message"),
    [
        (np.ones((48, 49)), "shape"),
        (np.full((49, 49), np.inf), "non-finite"),
        (object(), "not a float64"),
    ],
)
def test_vjp_rejects_invalid_cotangent(cotangent: object, message: str) -> None:
    """Reject cotangents that cannot represent the returned flux field."""
    with pytest.raises(IDAEquilibriumAdapterError, match=message):
        adapter._vjp_with_backend(_request(), cotangent, FakeBackend())


@pytest.mark.parametrize(("mode", "message"), [("shape", "shape"), ("nan", "non-finite")])
def test_vjp_rejects_invalid_upstream_gradients(mode: str, message: str) -> None:
    """Reject malformed or non-finite upstream gradient vectors."""
    backend = FakeBackend()
    backend.gradient_mode = mode
    with pytest.raises(IDAEquilibriumAdapterError, match=message):
        adapter._vjp_with_backend(_request(), np.ones((49, 49)), backend)


def test_vjp_rejects_wrong_gradient_arity() -> None:
    """Require one VJP output for each declared differentiated input."""
    backend = FakeBackend()

    def bad_vjp(
        function: Callable[[object, object, object], object],
        coil_current: object,
        pprime_coefficients: object,
        ffprime_coefficients: object,
    ) -> tuple[object, Callable[[object], tuple[object, object, object]]]:
        value = function(coil_current, pprime_coefficients, ffprime_coefficients)

        def pullback(_cotangent: object) -> tuple[object, object, object]:
            return cast(tuple[object, object, object], (np.ones(1), np.ones(1)))

        return value, pullback

    backend.vjp = bad_vjp
    with pytest.raises(IDAEquilibriumAdapterError, match="three differentiated inputs"):
        adapter._vjp_with_backend(_request(), np.ones((49, 49)), backend)


def test_public_functions_load_the_runtime_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route both public calls through the admitted runtime backend loader."""
    backend = FakeBackend()
    monkeypatch.setattr(adapter, "_LoadedBackend", lambda: backend)

    solved = solve_ida_equilibrium(_request())
    differentiated = ida_equilibrium_vjp(_request(), np.ones((49, 49)))

    assert solved.evidence.upstream_version == "4.0.0"
    assert differentiated.equilibrium.evidence.upstream_version == "4.0.0"


def test_source_file_digest_guards_missing_unreadable_and_empty_sources(tmp_path: Path) -> None:
    """Hash exact readable source bytes and reject absent provenance."""
    with pytest.raises(IDAEquilibriumAdapterError, match="does not expose"):
        adapter._source_file_sha256(object(), label="fixture")
    with pytest.raises(IDAEquilibriumAdapterError, match="cannot read"):
        adapter._source_file_sha256(SimpleNamespace(__file__=str(tmp_path / "missing.py")), label="fixture")
    empty = tmp_path / "empty.py"
    empty.write_bytes(b"")
    with pytest.raises(IDAEquilibriumAdapterError, match="empty"):
        adapter._source_file_sha256(SimpleNamespace(__file__=str(empty)), label="fixture")
    source = tmp_path / "module.py"
    source.write_bytes(b"value = 1\n")
    assert adapter._source_file_sha256(SimpleNamespace(__file__=str(source)), label="fixture") == (
        "585c93666fcb046b7b264d3fa73202aa2a38254ae82a4b3ba19e873c2d5a9886"
    )


def test_loaded_backend_binds_dynamic_modules_and_calls_exact_solver_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind dynamic modules and preserve the exact upstream solver arguments."""
    solver_source = tmp_path / "solver.py"
    basis_source = tmp_path / "basis.py"
    solver_source.write_text("solver = True\n", encoding="utf-8")
    basis_source.write_text("basis = True\n", encoding="utf-8")
    calls: list[tuple[object, ...]] = []

    def design(query: object, n_coeff: int, degree: int) -> np.ndarray:
        return np.ones((np.asarray(query).size, n_coeff)) * degree

    def evaluate(coefficients: object, matrix: object) -> np.ndarray:
        return np.asarray(matrix) @ np.asarray(coefficients)

    def solve(*args: object) -> np.ndarray:
        calls.append(args)
        return np.zeros((49, 49))

    def vjp(
        function: Callable[[object, object, object], object], *primals: object
    ) -> tuple[object, Callable[[object], tuple[object, object, object]]]:
        value = function(*primals)
        return value, lambda _cotangent: cast(tuple[object, object, object], primals)

    modules: dict[str, object] = {
        "scpn_fusion": SimpleNamespace(__version__="4.0.0"),
        "scpn_fusion.core.jax_free_boundary_gs_implicit": SimpleNamespace(
            __file__=str(solver_source), solve_free_boundary_gs_implicit=solve
        ),
        "scpn_fusion.core.jax_profile_basis": SimpleNamespace(
            __file__=str(basis_source), bspline_design_matrix=design, evaluate_profile=evaluate
        ),
        "jax": SimpleNamespace(config=SimpleNamespace(x64_enabled=True), vjp=vjp),
        "jaxlib": SimpleNamespace(__version__="0.6.2"),
        "jax.numpy": SimpleNamespace(asarray=np.asarray),
    }
    cast(SimpleNamespace, modules["jax"]).__version__ = "0.6.2"
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: modules[name])

    backend = adapter._LoadedBackend()
    request = _request()
    result = adapter._solve_with_backend(request, backend)

    assert backend.x64_enabled is True
    assert result.psi_wb.shape == (49, 49)
    assert len(calls) == 1
    assert calls[0][-4:] == (40, 0.6, 4.0e-7 * np.pi, True)
    value, pullback = backend.vjp(lambda a, _b, _c: a, np.ones(1), np.ones(2), np.ones(3))
    assert np.asarray(value).shape == (1,)
    assert len(pullback(np.ones(1))) == 3


def test_loaded_backend_fails_closed_on_missing_or_wrong_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an absent dependency and an incompatible upstream major version."""

    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(IDAEquilibriumAdapterError, match="requires scpn-fusion 4.x with JAX"):
        adapter._LoadedBackend()

    modules = {
        "scpn_fusion": SimpleNamespace(__version__="3.9.11"),
        "scpn_fusion.core.jax_free_boundary_gs_implicit": SimpleNamespace(),
        "scpn_fusion.core.jax_profile_basis": SimpleNamespace(),
        "jax": SimpleNamespace(config=SimpleNamespace(x64_enabled=True)),
        "jaxlib": SimpleNamespace(__version__="0.6.2"),
        "jax.numpy": SimpleNamespace(),
    }
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: modules[name])
    with pytest.raises(IDAEquilibriumAdapterError, match="requires scpn-fusion 4.x"):
        adapter._LoadedBackend()
