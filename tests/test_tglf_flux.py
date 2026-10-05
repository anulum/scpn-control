# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Standalone signed TGLF flux tests.

"""Read actual GACODE artifacts and optionally exercise the installed provider."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.tglf_flux import TGLFFluxError, TGLFFluxSolver, read_tglf_fluxes
from scpn_control.core.tglf_units import TGLFReferenceUnits, physical_tglf_flux
from scpn_control.core.transport_flux import TransportFaceFlux, advance_face_flux

_FIXTURE = Path(__file__).parent / "data/tglf/default"


def test_actual_provider_files_preserve_signed_species_fluxes() -> None:
    """The real two-species artifact retains inward particle flux and both modes."""
    provenance = json.loads((_FIXTURE / "provenance.json").read_text())
    for name, digest in provenance["files_sha256"].items():
        assert hashlib.sha256((_FIXTURE / name).read_bytes()).hexdigest() == digest
    result = read_tglf_fluxes(_FIXTURE)
    assert result.particle_flux_gb == pytest.approx((-1.9938544745721216, -1.9938544745721214), rel=1e-14)
    assert result.energy_flux_gb == pytest.approx((16.209564881227283, 38.411116503174036), rel=1e-14)
    assert result.exchange_gb == pytest.approx((6.661053580155404, -6.661053580155402), rel=1e-14)
    assert result.momentum_flux_gb == pytest.approx((3.3011294788465045e-14, -1.4269512891640234e-11), rel=1e-14, abs=0)
    assert len(result.k_y) == len(result.growth_rate) == len(result.frequency) == 21
    assert len(result.growth_rate[0]) == 2
    assert result.growth_rate[0] == pytest.approx((0.11840068382431693, 0.049100463988464113))
    assert result.frequency[0] == pytest.approx((-0.023983126894722744, 0.047803319911465414))


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("out.tglf.grid", "2.5\n16\n"),
        ("out.tglf.gbflux", "1 2 3\n"),
        ("out.tglf.gbflux", "nan 1 2 3 4 5 6 7\n"),
        ("out.tglf.ky_spectrum", "index limits: nky\n2\n0.1\n"),
        ("out.tglf.ky_spectrum", "index limits: nky\n2\n0.2\n0.1\n"),
        ("out.tglf.eigenvalue_spectrum", "header\nheader\n1 2 3\n"),
    ],
)
def test_corrupt_provider_artifact_is_rejected(tmp_path: Path, name: str, content: str) -> None:
    """Corrupt a captured real output member and require structural refusal."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    (tmp_path / name).write_text(content)
    with pytest.raises(TGLFFluxError):
        read_tglf_fluxes(tmp_path)


def test_missing_provider_member_is_rejected(tmp_path: Path) -> None:
    """A partial real output set cannot be represented as successful zero flux."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    (tmp_path / "out.tglf.gbflux").unlink()
    with pytest.raises(TGLFFluxError):
        read_tglf_fluxes(tmp_path)


def test_missing_launcher_creates_no_execution(tmp_path: Path) -> None:
    """Executable discovery fails before an execution directory is created."""
    solver = TGLFFluxSolver(tmp_path / "runs", binary=str(tmp_path / "missing"))
    with pytest.raises(TGLFFluxError, match="unavailable"):
        solver.run(_FIXTURE / "input.tglf")
    assert not solver.work_dir.exists()


@pytest.fixture
def real_solver(tmp_path: Path) -> TGLFFluxSolver:
    """Opt in to an actual installed GACODE; no simulated provider substitution."""
    binary = os.environ.get("SCPN_TGLF_BINARY")
    if not binary:
        pytest.skip("SCPN_TGLF_BINARY must name an actual configured GACODE launcher")
    environment = json.loads(os.environ.get("SCPN_TGLF_ENV_JSON", "{}"))
    return TGLFFluxSolver(tmp_path / "runs", binary=binary, environment=environment)


def test_real_provider_execution_preserves_signed_flux(real_solver: TGLFFluxSolver) -> None:
    """Execute the real default twice and retain distinct, input-hashed evidence."""
    first = real_solver.run(_FIXTURE / "input.tglf", timeout_s=30)
    second = real_solver.run(_FIXTURE / "input.tglf", timeout_s=30)
    assert first.run_dir != second.run_dir
    assert first.particle_flux_gb[0] < 0
    assert first.particle_flux_gb == pytest.approx(first.particle_flux_gb[::-1], rel=1e-12)
    assert second.energy_flux_gb == pytest.approx(first.energy_flux_gb, rel=1e-8)
    receipt = json.loads((second.run_dir / "execution.json").read_text())
    assert receipt["returncode"] == 0 and not receipt["timed_out"]
    assert receipt["output_validated"]
    assert receipt["input_sha256"] == hashlib.sha256((_FIXTURE / "input.tglf").read_bytes()).hexdigest()
    for name, digest in receipt["output_sha256"].items():
        assert hashlib.sha256((second.run_dir / name).read_bytes()).hexdigest() == digest


def test_real_bad_input_cannot_reuse_prior_success(real_solver: TGLFFluxSolver, tmp_path: Path) -> None:
    """A real parser failure following success must not admit preceding flux files."""
    good = real_solver.run(_FIXTURE / "input.tglf", timeout_s=30)
    invalid = tmp_path / "invalid.tglf"
    invalid.write_text("NOT_A_TGLF_KEY=1\n")
    with pytest.raises(TGLFFluxError, match="execution failed"):
        real_solver.run(invalid, timeout_s=30)
    directories = list(real_solver.work_dir.iterdir())
    assert len(directories) == 2
    failed = next(path for path in directories if path != good.run_dir)
    assert not (failed / "out.tglf.gbflux").exists()
    assert json.loads((failed / "execution.json").read_text())["returncode"] != 0


def test_real_provider_timeout_is_recorded(real_solver: TGLFFluxSolver) -> None:
    """A real launcher receives a bounded lifetime and leaves an explicit timeout receipt."""
    with pytest.raises(TGLFFluxError, match="timeout=True"):
        real_solver.run(_FIXTURE / "input.tglf", timeout_s=1e-6)
    (directory,) = real_solver.work_dir.iterdir()
    receipt = json.loads((directory / "execution.json").read_text())
    assert receipt["timed_out"] and receipt["returncode"] != 0


@pytest.mark.parametrize(
    "member",
    [
        "out.tglf.gbflux",
        "input.tglf",
        "out.tglf.sum_flux_spectrum",
        "input.tglf.gen",
        "out.tglf.scalar_saturation_parameters",
    ],
)
def test_retained_real_run_change_is_rejected(real_solver: TGLFFluxSolver, member: str) -> None:
    """A successful execution cannot lend its receipt to subsequently changed retained bytes."""
    result = real_solver.run(_FIXTURE / "input.tglf")
    assert read_tglf_fluxes(result.run_dir) == result
    path = result.run_dir / member
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(TGLFFluxError, match="retained artifacts"):
        read_tglf_fluxes(result.run_dir)
    assert result.particle_flux_gb[0] < 0


def _instrumented_launcher(real_solver: TGLFFluxSolver, tmp_path: Path, mode: str) -> Path:
    # This wrapper executes the actual installed GACODE; it only controls the
    # surrounding process lifetime and custody faults under test.
    """Wrap the real GACODE executable with controlled process-lifetime or launcher-custody faults."""
    launcher = tmp_path / "instrumented_tglf"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([{real_solver.binary!r}, '-e', '.'])\n"
        "Path('provider.pid').write_text(str(child.pid))\n"
        f"mode = {mode!r}\n"
        "if mode == 'cancel':\n"
        "    Path('ready').write_text(str(os.getpid()))\n"
        "    child.wait()\n"
        "    time.sleep(60)\n"
        "if mode == 'escape':\n"
        "    escaped = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'], start_new_session=True)\n"
        "    Path('escaped.pid').write_text(str(escaped.pid))\n"
        "code = child.wait()\n"
        "if mode in ('receipt-write', 'receipt-cancel'):\n"
        "    Path('execution.json').mkdir()\n"
        "    Path('provider-returncode').write_text(str(code))\n"
        "    Path('wrapper.pid').write_text(str(os.getpid()))\n"
        "    if mode == 'receipt-cancel':\n"
        "        Path('ready').write_text(str(os.getpid()))\n"
        "        time.sleep(60)\n"
        "if mode == 'drift':\n"
        "    path = Path(__file__)\n"
        "    path.write_text(path.read_text() + '# launcher changed\\n')\n"
        "sys.exit(code)\n"
    )
    launcher.chmod(0o700)
    return launcher


def test_real_launcher_identity_drift_is_rejected(real_solver: TGLFFluxSolver, tmp_path: Path) -> None:
    """A wrapper changing its own bytes after real GACODE execution cannot attest pre-launch identity."""
    launcher = _instrumented_launcher(real_solver, tmp_path, "drift")
    before = hashlib.sha256(launcher.read_bytes()).hexdigest()
    real_solver.binary = str(launcher)
    with pytest.raises(TGLFFluxError, match="launcher changed"):
        real_solver.run(_FIXTURE / "input.tglf")
    (directory,) = real_solver.work_dir.iterdir()
    receipt = json.loads((directory / "execution.json").read_text())
    assert receipt["launcher_sha256"] == before
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() != before
    assert not receipt["launcher_unchanged"] and not receipt["output_validated"]


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_actual_wait_interruption_reaps_launcher(
    real_solver: TGLFFluxSolver, tmp_path: Path, exception: type[BaseException]
) -> None:
    """A real signal interrupts the public call while its real-provider launcher is still alive."""
    real_solver.binary = str(_instrumented_launcher(real_solver, tmp_path, "cancel"))
    stop = threading.Event()
    sent = threading.Event()

    def interrupt(signum: int, frame: object) -> None:
        """Raise the selected cancellation failure in the main thread during the real provider wait."""
        raise exception("external TGLF interruption")

    def signal_when_ready() -> None:
        """Signal only after the real launcher publishes readiness, stopping if the test exits first."""
        deadline = time.monotonic() + 10
        while not stop.wait(0.01) and time.monotonic() < deadline:
            if list(real_solver.work_dir.glob("tglf-*/ready")):
                sent.set()
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    previous = signal.signal(signal.SIGUSR1, interrupt)
    sender = threading.Thread(target=signal_when_ready)
    sender.start()
    started = time.monotonic()
    try:
        with pytest.raises(exception, match="external TGLF interruption"):
            real_solver.run(_FIXTURE / "input.tglf", timeout_s=20)
    finally:
        stop.set()
        sender.join(timeout=2)
        signal.signal(signal.SIGUSR1, previous)
    assert sent.is_set() and not sender.is_alive()
    assert time.monotonic() - started < 10
    (directory,) = real_solver.work_dir.iterdir()
    receipt = json.loads((directory / "execution.json").read_text())
    assert receipt["failure_type"] == exception.__name__
    assert receipt["returncode"] == -signal.SIGKILL
    assert not receipt["cleanup_errors"] and not receipt["output_validated"]
    with pytest.raises(ChildProcessError):
        os.waitpid(receipt["pid"], os.WNOHANG)


def test_escaped_descendant_log_handles_do_not_block_return(real_solver: TGLFFluxSolver, tmp_path: Path) -> None:
    """An escaped process retains inherited log handles without holding the solver call open."""
    real_solver.binary = str(_instrumented_launcher(real_solver, tmp_path, "escape"))
    started = time.monotonic()
    try:
        result = real_solver.run(_FIXTURE / "input.tglf", timeout_s=2)
        assert time.monotonic() - started < 10
        assert result.particle_flux_gb[0] < 0
        # Escaped sessions are explicitly outside process-group containment.
        escaped = int((result.run_dir / "escaped.pid").read_text())
        os.kill(escaped, 0)
    finally:
        for pidfile in real_solver.work_dir.glob("tglf-*/escaped.pid"):
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    "member", ["out.tglf.sum_flux_spectrum", "input.tglf.gen", "out.tglf.scalar_saturation_parameters"]
)
def test_coarse_archive_without_precision_evidence_is_rejected(tmp_path: Path, member: str) -> None:
    """Missing full-precision or resolved-input evidence cannot silently select coarse moments."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    (tmp_path / member).unlink()
    with pytest.raises(TGLFFluxError, match="capture"):
        read_tglf_fluxes(tmp_path)


@pytest.mark.parametrize("fault", ["row", "block", "duplicate", "order", "columns", "nan", "overflow", "disagree"])
def test_weighted_spectrum_corruption_is_rejected(tmp_path: Path, fault: str) -> None:
    """Refuse damaged real spectra, including finite but summary-inconsistent moments."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "out.tglf.sum_flux_spectrum"
    lines = path.read_text().splitlines()
    if fault == "row":
        del lines[2]
    elif fault == "block":
        del lines[23:]
    elif fault == "duplicate":
        lines[23] = lines[0]
    elif fault == "order":
        lines = lines[23:] + lines[:23]
    elif fault == "columns":
        lines[1] = lines[1].replace("toroidal stress,parallel stress", "parallel stress,toroidal stress")
    elif fault == "nan":
        lines[2] = "nan 1 2 3 4"
    elif fault == "overflow":
        lines[2] = lines[3] = "1e308 1e308 1e308 1e308 1e308"
    else:
        row = lines[2].split()
        row[0] = str(float(row[0]) + 0.001)
        lines[2] = " ".join(row)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TGLFFluxError, match="spectrum|disagrees"):
        read_tglf_fluxes(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("GYRO  UNITS", "OTHER  UNITS"),
        (".false.  USE_BPER", ".true.  USE_BPER"),
        (".false.  USE_BPAR", ".true.  USE_BPAR"),
        ("2  NS", "3  NS"),
        ("2  NMODES", "1  NMODES"),
        (".false.  ADIABATIC_ELEC", ".true.  ADIABATIC_ELEC"),
        (".true.  IFLUX", ".false.  IFLUX"),
        (".false.  USE_BPER", "0  USE_BPER"),
        ("2  NS", "2  NS\n2  NS"),
        ("GYRO  UNITS", "GYRO"),
    ],
)
def test_resolved_input_mismatch_is_rejected(tmp_path: Path, old: str, new: str) -> None:
    """Resolved species, fields, mode and normalization contracts constrain the captured output set."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "input.tglf.gen"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    with pytest.raises(TGLFFluxError, match="configuration|spectrum"):
        read_tglf_fluxes(tmp_path)


def test_real_three_species_flux_enters_conservative_step(real_solver: TGLFFluxSolver, tmp_path: Path) -> None:
    """A prescribed Miller charge-two case conserves charge without repairing the rounded provider flux."""
    deck = {
        "UNITS": "GYRO",
        "USE_TRANSPORT_MODEL": ".true.",
        "GEOMETRY_FLAG": 1,
        "NS": 3,
        "MASS_1": 0.00027230851111300849,
        "MASS_2": 1,
        "MASS_3": 2,
        "ZS_1": -1,
        "ZS_2": 1,
        "ZS_3": 2,
        "AS_1": 1,
        "AS_2": 0.938,
        "AS_3": 0.031,
        "TAUS_1": 1,
        "TAUS_2": 1.5,
        "TAUS_3": 1.5,
        "RLNS_1": 1,
        "RLNS_2": 1.0330490405117272,
        "RLNS_3": 0.5,
        "RLTS_1": 3,
        "RLTS_2": 2,
        "RLTS_3": 2,
        "BETAE": 0.010066772686255245,
        "XNUE": 0.220973591883244,
        "ZEFF": 1.062,
        "DEBYE": 0.014551423090695239,
        "RMIN_LOC": 0.5,
        "RMAJ_LOC": 3,
        "Q_LOC": 2,
        "Q_PRIME_LOC": 16,
        "P_PRIME_LOC": -0.013432248355297499,
        "KAPPA_LOC": 1.7,
        "DELTA_LOC": 0.3,
        "NKY": 12,
        "KY": 0.3,
        "NMODES": 2,
        "SAT_RULE": 0,
        "USE_BPER": ".false.",
        "USE_BPAR": ".false.",
    }
    path = tmp_path / "miller.tglf"
    path.write_text("".join(f"{key}={value}\n" for key, value in deck.items()))
    raw = real_solver.run(path)
    e, main, trace = raw.particle_flux_gb
    assert e == pytest.approx(main + 2 * trace, rel=1e-12, abs=0)
    coarse = np.loadtxt(raw.run_dir / "out.tglf.gbflux")[:3]
    assert abs(coarse[0] - coarse[1] - 2 * coarse[2]) / abs(coarse[0]) > 1e-6
    physical = physical_tglf_flux(raw, TGLFReferenceUnits(5e19, 2.0, 2.0, 2 * 1.67262192369e-27, 2.0))
    particle = np.zeros((4, 4))
    particle[[0, 1, 3]] = np.asarray(physical.particle_m2_s)[:, None]
    heat = np.repeat([[physical.energy_w_m2[0]], [sum(physical.energy_w_m2[1:])]], 4, axis=1)
    density = np.repeat([[5.0], [4.69], [0.0], [0.155]], 5, axis=1)
    temperature = np.repeat([[2.0], [3.0]], 5, axis=1)
    original_density, original_temperature = density.copy(), temperature.copy()
    result = advance_face_flux(
        rho=np.linspace(0, 1, 5),
        major_radius_m=6.0,
        minor_radius_m=2.0,
        density=density,
        temperature=temperature,
        impurity_density=np.zeros(5),
        flux=TransportFaceFlux(particle, heat),
        dt=1e-6,
    )
    np.testing.assert_allclose(result.density[0], result.density[1] + 2 * result.density[3], rtol=1e-12, atol=0)
    assert result.balance.particle_relative_error < 1e-12
    assert result.balance.energy_relative_error < 1e-12
    np.testing.assert_array_equal(density, original_density)
    np.testing.assert_array_equal(temperature, original_temperature)
    assert not np.array_equal(result.density[:, 1:-1], density[:, 1:-1])
    (raw.run_dir / "conservative_probe.json").write_text(
        json.dumps(
            {
                "particle_gb": raw.particle_flux_gb,
                "charge_residual_gb": e - main - 2 * trace,
                "particle_relative_error": result.balance.particle_relative_error,
                "energy_relative_error": result.balance.energy_relative_error,
                "boundary": "Prescribed local charge-two mass/reference=2 case, common ion temperature; repeated face flux, not radial self-consistency or machine calibration",
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("use_bpar", [False, True])
def test_real_electromagnetic_fields_are_complete(
    real_solver: TGLFFluxSolver,
    tmp_path: Path,
    use_bpar: bool,
) -> None:
    """Integrate actual two/three-field output and refuse omission even if remaining blocks are complete."""
    path = tmp_path / "electromagnetic.tglf"
    path.write_text(f"USE_BPER=.TRUE.\nUSE_BPAR={'.true.' if use_bpar else '.false.'}\nBETAE=0.01\n")
    raw = real_solver.run(path)
    assert np.all(np.isfinite(raw.energy_flux_gb)) and any(raw.energy_flux_gb)
    assert read_tglf_fluxes(raw.run_dir) == raw
    archive = tmp_path / "archive"
    shutil.copytree(raw.run_dir, archive)
    (archive / "execution.json").unlink()
    spectrum = archive / "out.tglf.sum_flux_spectrum"
    lines = spectrum.read_text().splitlines()
    block_size = len(raw.k_y) + 2
    del lines[block_size : 2 * block_size]
    spectrum.write_text("\n".join(lines) + "\n")
    with pytest.raises(TGLFFluxError, match="spectrum"):
        read_tglf_fluxes(archive)


@pytest.mark.parametrize("offset, admitted", [(0.49, True), (-0.49, True), (0.51, False), (-0.51, False)])
@pytest.mark.parametrize("summary", ["1.0000E+00", "-1.0000E+00", "1.0000E-300", "-1.0000E-300"])
def test_printed_rounding_interval_is_enforced(
    tmp_path: Path,
    offset: float,
    admitted: bool,
    summary: str,
) -> None:
    """Controlled artifact mutations bracket the signed five-digit decimal interval, including tiny values."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    gbflux = tmp_path / "out.tglf.gbflux"
    coarse = gbflux.read_text().split()
    coarse[0] = summary
    gbflux.write_text(" ".join(coarse) + "\n")
    target = float(summary) + offset * abs(float(summary)) * 1e-4
    path = tmp_path / "out.tglf.sum_flux_spectrum"
    lines = path.read_text().splitlines()
    for index in range(2, 23):
        row = lines[index].split()
        row[0] = repr(target if index == 2 else 0.0)
        lines[index] = " ".join(row)
    path.write_text("\n".join(lines) + "\n")
    if admitted:
        assert read_tglf_fluxes(tmp_path).particle_flux_gb[0] == target
    else:
        with pytest.raises(TGLFFluxError, match="disagrees"):
            read_tglf_fluxes(tmp_path)


@pytest.mark.parametrize("fault", ["units", "rule", "missing", "duplicate"])
def test_effective_saturation_mismatch_is_rejected(tmp_path: Path, fault: str) -> None:
    """Generated inputs cannot override conflicting or incomplete effective provider settings."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "out.tglf.scalar_saturation_parameters"
    lines = path.read_text().splitlines()
    if fault == "units":
        lines[2] = "UNITS = CGYRO"
    elif fault == "rule":
        lines[1] = "SAT_RULE = 1"
    elif fault == "missing":
        del lines[2]
    else:
        lines.append("UNITS = GYRO")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TGLFFluxError, match="configuration"):
        read_tglf_fluxes(tmp_path)


@pytest.mark.parametrize("saturation_rule", [2, 3])
def test_real_saturation_preset_cannot_masquerade_as_gyro(
    real_solver: TGLFFluxSolver,
    tmp_path: Path,
    saturation_rule: int,
) -> None:
    """Actual SAT2/3 startup rewrites requested GYRO to CGYRO; the GYRO-only boundary refuses it."""
    path = tmp_path / "preset.tglf"
    path.write_text(f"UNITS=GYRO\nSAT_RULE={saturation_rule}\n")
    with pytest.raises(TGLFFluxError, match="configuration"):
        real_solver.run(path)
    (run,) = real_solver.work_dir.iterdir()
    generated = (run / "input.tglf.gen").read_text()
    assert "GYRO  UNITS" in generated
    effective = (run / "out.tglf.scalar_saturation_parameters").read_text()
    assert "CGYRO" in effective
    receipt = json.loads((run / "execution.json").read_text())
    assert receipt["returncode"] == 0 and not receipt["output_validated"]
    assert "out.tglf.scalar_saturation_parameters" in receipt["output_sha256"]


def test_real_sat1_gyro_contract_remains_admissible(real_solver: TGLFFluxSolver, tmp_path: Path) -> None:
    """A real SAT1 run retaining GYRO settings is not refused merely for selecting a different fit."""
    path = tmp_path / "sat1.tglf"
    path.write_text("UNITS=GYRO\nSAT_RULE=1\n")
    result = real_solver.run(path)
    assert np.all(np.isfinite(result.energy_flux_gb))
    assert read_tglf_fluxes(result.run_dir) == result


@pytest.mark.parametrize("validated", [1, "false", [True], {"accepted": True}])
def test_receipt_validation_flag_requires_literal_true(real_solver: TGLFFluxSolver, validated: object) -> None:
    """Truthy JSON values must not masquerade as a validated actual execution receipt."""
    result = real_solver.run(_FIXTURE / "input.tglf")
    path = result.run_dir / "execution.json"
    receipt = json.loads(path.read_text())
    receipt["output_validated"] = validated
    path.write_text(json.dumps(receipt))
    before = {member.name: member.read_bytes() for member in result.run_dir.iterdir() if member.is_file()}
    with pytest.raises(TGLFFluxError, match="validated receipt"):
        read_tglf_fluxes(result.run_dir)
    assert {member.name: member.read_bytes() for member in result.run_dir.iterdir() if member.is_file()} == before


@pytest.mark.parametrize("fault", ["malformed_json", "array", "null", "empty_object", "missing_input"])
def test_unverifiable_actual_execution_receipt_is_rejected(real_solver: TGLFFluxSolver, fault: str) -> None:
    """Malformed receipt structure and a missing input cannot certify retained output bytes."""
    result = real_solver.run(_FIXTURE / "input.tglf")
    path = result.run_dir / "execution.json"
    if fault == "missing_input":
        (result.run_dir / "input.tglf").unlink()
    else:
        path.write_text({"malformed_json": "{", "array": "[]", "null": "null", "empty_object": "{}"}[fault])
    before = {member.name: member.read_bytes() for member in result.run_dir.iterdir() if member.is_file()}
    with pytest.raises(TGLFFluxError, match="Cannot verify TGLF execution receipt"):
        read_tglf_fluxes(result.run_dir)
    assert {member.name: member.read_bytes() for member in result.run_dir.iterdir() if member.is_file()} == before


@pytest.mark.parametrize("columns", [4, 6])
def test_weighted_spectrum_requires_exactly_five_moments(tmp_path: Path, columns: int) -> None:
    """Missing or extra columns in a real captured spectral row cannot change its moment convention."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "out.tglf.sum_flux_spectrum"
    rows = path.read_text().splitlines()
    row = rows[2].split()
    rows[2] = " ".join(row[:4] if columns == 4 else row + ["0.0"])
    path.write_text("\n".join(rows) + "\n")
    before = path.read_bytes()
    with pytest.raises(TGLFFluxError, match="weighted flux spectrum is inconsistent"):
        read_tglf_fluxes(tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["numeric_comment", "extra_precision"])
def test_coarse_moments_require_the_actual_writer_format(tmp_path: Path, fault: str) -> None:
    """Numeric-loader permissiveness cannot admit comments or a changed summary precision."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "out.tglf.gbflux"
    if fault == "numeric_comment":
        path.write_text(path.read_text() + "# 0\n")
    else:
        tokens = path.read_text().split()
        tokens[0] = format(float(tokens[0]), ".5e")
        path.write_text(" ".join(tokens) + "\n")
    before = path.read_bytes()
    expected = "only the expected numeric tokens" if fault == "numeric_comment" else "expected 1pe11.4 precision"
    with pytest.raises(TGLFFluxError, match=expected):
        read_tglf_fluxes(tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("timeout", [0.0, -1.0, np.nan, np.inf, True, False, np.bool_(True), np.bool_(False)])
def test_invalid_timeout_precedes_execution_directory_creation(tmp_path: Path, timeout: float) -> None:
    """Invalid lifetime bounds are rejected before launcher discovery or filesystem writes."""
    solver = TGLFFluxSolver(tmp_path / "runs", binary=str(tmp_path / "absent-launcher"))
    with pytest.raises(ValueError, match="timeout_s must be finite and positive"):
        solver.run(_FIXTURE / "input.tglf", timeout_s=timeout)
    assert not solver.work_dir.exists()


def test_filesystem_change_during_capture_is_rejected(real_solver: TGLFFluxSolver) -> None:
    """A FIFO rendezvous changes retained grid bytes after capture, before final validation.

    An actual completed provider run supplies the artifacts and receipt. The last
    captured member temporarily becomes a FIFO; a real writer process synchronizes
    with the public reader without patching its capture or parser helpers.
    """
    result = real_solver.run(_FIXTURE / "input.tglf")
    directory = result.run_dir
    before = {member.name: member.read_bytes() for member in directory.iterdir() if member.is_file()}
    scalar = directory / "out.tglf.scalar_saturation_parameters"
    replacement = directory / "scalar-replacement"
    replacement.write_bytes(before[scalar.name])
    scalar.unlink()
    os.mkfifo(scalar)
    ready = directory / "writer-ready"
    writer = (
        "from pathlib import Path\nimport os, sys\n"
        "directory = Path(sys.argv[1])\n"
        "scalar = directory / 'out.tglf.scalar_saturation_parameters'\n"
        "replacement = directory / 'scalar-replacement'\n"
        "ready = directory / 'writer-ready'\n"
        "ready.write_text('ready')\n"
        "with scalar.open('wb') as stream:\n"
        "    stream.write(replacement.read_bytes())\n"
        "    grid = directory / 'out.tglf.grid'\n"
        "    grid.write_bytes(grid.read_bytes() + b'\\n')\n"
        "    os.replace(replacement, scalar)\n"
        "    ready.unlink()\n"
    )
    with subprocess.Popen(
        [sys.executable, "-c", writer, str(directory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        deadline = time.monotonic() + 10
        try:
            while not ready.exists():
                assert process.poll() is None, "filesystem writer exited before rendezvous"
                assert time.monotonic() < deadline, "filesystem writer did not become ready"
                threading.Event().wait(0.01)
            with pytest.raises(TGLFFluxError, match="TGLF retained output changed during validation"):
                read_tglf_fluxes(directory)
            stdout, stderr = process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
    assert process.returncode == 0, stderr
    assert not stdout and not stderr
    expected = dict(before)
    expected["out.tglf.grid"] += b"\n"
    assert {member.name: member.read_bytes() for member in directory.iterdir() if member.is_file()} == expected


def test_actual_exec_failure_retains_unvalidated_receipt(real_solver: TGLFFluxSolver, tmp_path: Path) -> None:
    """An existing real-provider wrapper with a missing interpreter records an OS launch failure."""
    launcher = _instrumented_launcher(real_solver, tmp_path, "launch")
    _, body = launcher.read_text().split("\n", 1)
    missing_interpreter = Path("/scpn-missing-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16])
    assert not missing_interpreter.exists()
    launcher.write_text(f"#!{missing_interpreter}\n{body}")
    real_solver.binary = str(launcher)
    with pytest.raises(FileNotFoundError):
        real_solver.run(_FIXTURE / "input.tglf")
    (directory,) = real_solver.work_dir.iterdir()
    receipt = json.loads((directory / "execution.json").read_text())
    assert receipt["failure_type"] == "FileNotFoundError"
    assert receipt["pid"] is None and receipt["returncode"] is None
    assert receipt["output_validated"] is False
    assert receipt["output_sha256"] == {} and receipt["cleanup_errors"] == []
    assert receipt["launcher_sha256"] == hashlib.sha256(launcher.read_bytes()).hexdigest()
    assert (directory / "input.tglf").read_bytes() == (_FIXTURE / "input.tglf").read_bytes()
    assert not (directory / "provider.pid").exists()
    with pytest.raises(TGLFFluxError, match="Cannot capture standalone TGLF output"):
        read_tglf_fluxes(directory)


@pytest.mark.parametrize("interrupted", [False, True])
def test_receipt_write_failure_preserves_original_execution_outcome(
    real_solver: TGLFFluxSolver, tmp_path: Path, interrupted: bool
) -> None:
    """Real filesystem refusal propagates alone or annotates the original cancellation error.

    The wrapper first completes the actual provider, then places a directory at
    the receipt destination. A real signal supplies the cancellation variant;
    neither filesystem writes nor process wait/cleanup methods are mocked.
    """
    mode = "receipt-cancel" if interrupted else "receipt-write"
    real_solver.binary = str(_instrumented_launcher(real_solver, tmp_path, mode))
    stop = threading.Event()
    sent = threading.Event()

    def interrupt(signum: int, frame: object) -> None:
        """Raise an identifiable original failure during the live wrapper wait."""
        raise RuntimeError("original receipt cancellation")

    def signal_when_ready() -> None:
        """Wait for actual provider completion and receipt obstruction before signalling."""
        deadline = time.monotonic() + 10
        while not stop.wait(0.01) and time.monotonic() < deadline:
            if list(real_solver.work_dir.glob("tglf-*/ready")):
                sent.set()
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    previous = signal.signal(signal.SIGUSR1, interrupt)
    sender = threading.Thread(target=signal_when_ready)
    if interrupted:
        sender.start()
    try:
        if interrupted:
            with pytest.raises(RuntimeError, match="original receipt cancellation") as caught:
                real_solver.run(_FIXTURE / "input.tglf", timeout_s=20)
            assert any("Cannot write TGLF failure receipt" in note for note in caught.value.__notes__)
            assert sent.is_set()
        else:
            with pytest.raises(IsADirectoryError):
                real_solver.run(_FIXTURE / "input.tglf", timeout_s=20)
    finally:
        stop.set()
        if interrupted:
            sender.join(timeout=2)
        signal.signal(signal.SIGUSR1, previous)
    assert not sender.is_alive()
    (directory,) = real_solver.work_dir.iterdir()
    assert (directory / "provider-returncode").read_text() == "0"
    assert (directory / "execution.json").is_dir()
    assert (directory / "out.tglf.sum_flux_spectrum").stat().st_size > 0
    assert (directory / "input.tglf").read_bytes() == (_FIXTURE / "input.tglf").read_bytes()
    with pytest.raises(ChildProcessError):
        os.waitpid(int((directory / "wrapper.pid").read_text()), os.WNOHANG)
    with pytest.raises(TGLFFluxError, match="Cannot verify TGLF execution receipt"):
        read_tglf_fluxes(directory)
