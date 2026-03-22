# ──────────────────────────────────────────────────────────────────────
# SCPN Control — External GK Solver Tests (GENE, GS2, CGYRO, QuaLiKiz)
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

from scpn_control.core.gk_interface import GKLocalParams


@pytest.fixture
def cbc_params():
    return GKLocalParams(
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        rho=0.5,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )


# ── GENE ──


def test_gene_input_generation(cbc_params):
    from scpn_control.core.gk_gene import generate_gene_input

    text = generate_gene_input(cbc_params)
    assert "&geometry" in text
    assert "q0 = 1.400000" in text
    assert "omt = 6.900000" in text


def test_gene_parse_empty(tmp_path):
    from scpn_control.core.gk_gene import parse_gene_output

    result = parse_gene_output(tmp_path)
    assert not result.converged


def test_gene_parse_nrg(tmp_path):
    from scpn_control.core.gk_gene import parse_gene_output

    nrg = tmp_path / "nrg_0001"
    nrg.write_text("1.0 0.2 -0.5 0.0\n2.0 0.25 -0.6 0.0\n")
    result = parse_gene_output(tmp_path)
    assert result.converged
    assert result.gamma[0] == pytest.approx(0.25)


def test_gene_solver_unavailable(cbc_params, tmp_path):
    from scpn_control.core.gk_gene import GENESolver

    solver = GENESolver(binary="nonexistent_gene_xyz", work_dir=tmp_path)
    assert not solver.is_available()
    result = solver.run_from_params(cbc_params)
    assert not result.converged


# ── GS2 ──


def test_gs2_input_generation(cbc_params):
    from scpn_control.core.gk_gs2 import generate_gs2_input

    text = generate_gs2_input(cbc_params)
    assert "qinp = 1.400000" in text
    assert "tprim = 6.900000" in text


def test_gs2_parse_empty(tmp_path):
    from scpn_control.core.gk_gs2 import parse_gs2_output

    result = parse_gs2_output(tmp_path)
    assert not result.converged


def test_gs2_parse_omega(tmp_path):
    from scpn_control.core.gk_gs2 import parse_gs2_output

    omega = tmp_path / "gs2.omega"
    omega.write_text("0.3 0.15 -0.4\n")
    result = parse_gs2_output(tmp_path)
    assert result.converged
    assert result.dominant_mode == "ITG"


def test_gs2_solver_unavailable(cbc_params, tmp_path):
    from scpn_control.core.gk_gs2 import GS2Solver

    solver = GS2Solver(binary="nonexistent_gs2_xyz", work_dir=tmp_path)
    assert not solver.is_available()
    result = solver.run_from_params(cbc_params)
    assert not result.converged


# ── CGYRO ──


def test_cgyro_input_generation(cbc_params):
    from scpn_control.core.gk_cgyro import generate_cgyro_input

    text = generate_cgyro_input(cbc_params)
    assert "Q=1.400000" in text
    assert "DLNTDR_1=6.900000" in text


def test_cgyro_parse_empty(tmp_path):
    from scpn_control.core.gk_cgyro import parse_cgyro_output

    result = parse_cgyro_output(tmp_path)
    assert not result.converged


def test_cgyro_parse_freq(tmp_path):
    from scpn_control.core.gk_cgyro import parse_cgyro_output

    freq = tmp_path / "out.cgyro.freq"
    freq.write_text("0.18 -0.35\n")
    result = parse_cgyro_output(tmp_path)
    assert result.converged
    assert result.dominant_mode == "ITG"


def test_cgyro_solver_unavailable(cbc_params, tmp_path):
    from scpn_control.core.gk_cgyro import CGYROSolver

    solver = CGYROSolver(binary="nonexistent_cgyro_xyz", work_dir=tmp_path)
    assert not solver.is_available()
    result = solver.run_from_params(cbc_params)
    assert not result.converged


# ── QuaLiKiz ──


def test_qualikiz_unavailable(cbc_params, tmp_path):
    from scpn_control.core.gk_qualikiz import QuaLiKizSolver

    solver = QuaLiKizSolver(work_dir=tmp_path)
    # qualikiz_tools not installed → not available
    result = solver.run_from_params(cbc_params)
    assert not result.converged


# ── Extended coverage tests ──


def test_cgyro_parse_error(tmp_path):
    """Cover gk_cgyro.py lines 78-79: corrupt freq file triggers warning."""
    from scpn_control.core.gk_cgyro import parse_cgyro_output

    freq = tmp_path / "out.cgyro.freq"
    freq.write_text("not_a_number garbage\n")
    result = parse_cgyro_output(tmp_path)
    assert not result.converged


def test_cgyro_run_subprocess_failure(cbc_params, tmp_path):
    """Cover gk_cgyro.py lines 103-113: subprocess exception path."""
    from unittest.mock import patch

    from scpn_control.core.gk_cgyro import CGYROSolver

    solver = CGYROSolver(work_dir=tmp_path)
    solver.prepare_input(cbc_params)
    with patch("shutil.which", return_value="/usr/bin/cgyro"):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = solver.run(tmp_path)
    assert not result.converged


def test_gene_parse_single_row(tmp_path):
    """Cover gk_gene.py lines 114, 128-130: single-row nrg and reshape."""
    from scpn_control.core.gk_gene import parse_gene_output

    nrg = tmp_path / "nrg_0001"
    nrg.write_text("1.0 0.3 -0.4 0.0\n")
    result = parse_gene_output(tmp_path)
    assert result.converged
    assert result.chi_i == pytest.approx(0.3)


def test_gene_parse_corrupt(tmp_path):
    """Cover gk_gene.py lines 128-130: ValueError on corrupt file."""
    from scpn_control.core.gk_gene import parse_gene_output

    nrg = tmp_path / "nrg_0001"
    nrg.write_text("not numeric\n")
    result = parse_gene_output(tmp_path)
    assert not result.converged


def test_gene_run_subprocess_failure(cbc_params, tmp_path):
    """Cover gk_gene.py lines 152-162: subprocess failure path."""
    from unittest.mock import patch

    from scpn_control.core.gk_gene import GENESolver

    solver = GENESolver(work_dir=tmp_path)
    solver.prepare_input(cbc_params)
    with patch("shutil.which", return_value="/usr/bin/gene"):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = solver.run(tmp_path)
    assert not result.converged


def test_gs2_parse_error(tmp_path):
    """Cover gk_gs2.py lines 100-101: corrupt omega file."""
    from scpn_control.core.gk_gs2 import parse_gs2_output

    omega = tmp_path / "gs2.omega"
    omega.write_text("not numeric data\n")
    result = parse_gs2_output(tmp_path)
    assert not result.converged


def test_gs2_run_subprocess_failure(cbc_params, tmp_path):
    """Cover gk_gs2.py lines 125-135: subprocess failure path."""
    from unittest.mock import patch

    from scpn_control.core.gk_gs2 import GS2Solver

    solver = GS2Solver(work_dir=tmp_path)
    solver.prepare_input(cbc_params)
    with patch("shutil.which", return_value="/usr/bin/gs2"), patch("subprocess.run", side_effect=FileNotFoundError):
        result = solver.run(tmp_path)
    assert not result.converged


def test_qualikiz_python_api_path(cbc_params, tmp_path):
    """Cover gk_qualikiz.py lines 33-45, 64-67, 83: Python API and is_available."""
    import sys
    from types import ModuleType
    from unittest.mock import patch

    from scpn_control.core.gk_qualikiz import QuaLiKizSolver, _try_qualikiz_python

    # Create mock qualikiz_tools
    mock_mod = ModuleType("qualikiz_tools")

    def fake_run(**kwargs):
        return {"chi_i": 1.5, "chi_e": 1.2, "D_e": 0.3}

    mock_mod.run = fake_run

    with patch.dict(sys.modules, {"qualikiz_tools": mock_mod}):
        result = _try_qualikiz_python(cbc_params)
        assert result is not None
        assert result.chi_i == pytest.approx(1.5)
        assert result.converged

        solver = QuaLiKizSolver(work_dir=tmp_path)
        assert solver.is_available()
        solver.prepare_input(cbc_params)
        out = solver.run(tmp_path)
        assert out.converged
