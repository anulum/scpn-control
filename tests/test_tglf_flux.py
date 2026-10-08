# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Standalone signed TGLF flux reader tests.

"""Read actual retained GACODE artifacts; no provider is needed.

The artifacts are the captured output of a real provider run. Tests that need
an installed provider are in ``tests/test_tglf_launcher.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scpn_control.core.tglf_flux import TGLFFluxError, capture_tglf_outputs, read_tglf_fluxes

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


def _retained_run(tmp_path: Path) -> Path:
    """Copy the captured provider run and give it a receipt in the launcher's form.

    The artifact bytes are those of the real captured run. The receipt is
    written here from those bytes with the three fields the reader checks; the
    receipts that the launcher itself writes are read back in
    ``tests/test_tglf_launcher.py`` wherever a provider is configured.
    """
    directory = tmp_path / "retained"
    shutil.copytree(_FIXTURE, directory)
    receipt = {
        "input_sha256": hashlib.sha256((directory / "input.tglf").read_bytes()).hexdigest(),
        "output_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in capture_tglf_outputs(directory).items()
        },
        "output_validated": True,
    }
    (directory / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return directory


def _files(directory: Path) -> dict[str, bytes]:
    """Return every regular file of a run directory with its bytes."""
    return {member.name: member.read_bytes() for member in directory.iterdir() if member.is_file()}


def test_retained_run_with_matching_receipt_reads_the_same_fluxes(tmp_path: Path) -> None:
    """A receipt that matches the retained bytes changes nothing in the decoded result."""
    directory = _retained_run(tmp_path)
    before = _files(directory)
    result = read_tglf_fluxes(directory)
    expected = read_tglf_fluxes(_FIXTURE)
    assert result.run_dir == directory
    assert result.particle_flux_gb == expected.particle_flux_gb
    assert result.energy_flux_gb == expected.energy_flux_gb
    assert result.momentum_flux_gb == expected.momentum_flux_gb
    assert result.exchange_gb == expected.exchange_gb
    assert result.growth_rate == expected.growth_rate
    assert result.particle_flux_gb[0] < 0
    assert _files(directory) == before


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
def test_retained_run_change_is_rejected(tmp_path: Path, member: str) -> None:
    """A receipt cannot be lent to retained bytes that changed after it was written."""
    directory = _retained_run(tmp_path)
    path = directory / member
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(TGLFFluxError, match="retained artifacts"):
        read_tglf_fluxes(directory)


@pytest.mark.parametrize("validated", [1, "false", [True], {"accepted": True}, False, None])
def test_retained_receipt_validation_flag_requires_literal_true(tmp_path: Path, validated: object) -> None:
    """Truthy or absent-like JSON values do not stand for a validated receipt."""
    directory = _retained_run(tmp_path)
    path = directory / "execution.json"
    receipt = json.loads(path.read_text())
    receipt["output_validated"] = validated
    path.write_text(json.dumps(receipt))
    before = _files(directory)
    with pytest.raises(TGLFFluxError, match="validated receipt"):
        read_tglf_fluxes(directory)
    assert _files(directory) == before


@pytest.mark.parametrize("fault", ["malformed_json", "array", "null", "empty_object", "missing_input"])
def test_unverifiable_retained_receipt_is_rejected(tmp_path: Path, fault: str) -> None:
    """A malformed receipt or a missing input cannot certify retained output bytes."""
    directory = _retained_run(tmp_path)
    if fault == "missing_input":
        (directory / "input.tglf").unlink()
    else:
        (directory / "execution.json").write_text(
            {"malformed_json": "{", "array": "[]", "null": "null", "empty_object": "{}"}[fault]
        )
    before = _files(directory)
    with pytest.raises(TGLFFluxError, match="Cannot verify TGLF execution receipt"):
        read_tglf_fluxes(directory)
    assert _files(directory) == before


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="the rendezvous needs a POSIX FIFO, which Windows does not provide"
)
def test_retained_file_change_during_capture_is_rejected(tmp_path: Path) -> None:
    """A FIFO rendezvous changes retained grid bytes after capture, before the final check.

    The last captured member temporarily becomes a FIFO. A real writer process
    synchronises with the public reader, so neither the capture nor the parser
    is replaced.
    """
    directory = _retained_run(tmp_path)
    before = _files(directory)
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
    assert _files(directory) == expected


@pytest.mark.parametrize("member", ["input.tglf", "execution.json"])
@pytest.mark.parametrize("operation", ["replace", "remove"])
def test_retained_admission_change_during_parsing_is_rejected(tmp_path: Path, member: str, operation: str) -> None:
    """Refuse input or receipt changes after admission, using the real parser.

    A profile event at the public parsing call replaces or removes the actual
    retained file. Neither the reader nor parser is replaced, and the original
    provider outputs remain unchanged. The prior profile is restored on every
    exit so the rendezvous works without a platform-specific filesystem API.
    """
    directory = _retained_run(tmp_path)
    outputs = capture_tglf_outputs(directory)
    changed = False

    def change_admission(frame: object, event: str, argument: object) -> None:
        """Mutate the retained admission file when the real public parser starts."""
        nonlocal changed
        code = getattr(frame, "f_code", None)
        if (
            not changed
            and event == "call"
            and getattr(code, "co_name", None) == "parse_captured_tglf_fluxes"
            and getattr(code, "co_filename", None) == read_tglf_fluxes.__code__.co_filename
        ):
            path = directory / member
            if operation == "replace":
                replacement = directory / f"{member}.replacement"
                replacement.write_bytes(path.read_bytes() + b"\n")
                replacement.replace(path)
            else:
                path.unlink()
            changed = True

    previous_profile = sys.getprofile()
    try:
        sys.setprofile(change_admission)
        with pytest.raises(TGLFFluxError, match="retained admission"):
            read_tglf_fluxes(directory)
    finally:
        sys.setprofile(previous_profile)
    assert changed
    assert capture_tglf_outputs(directory) == outputs


@pytest.mark.parametrize("token", ["0e999999999999999999999999999", "-0e999999999999999999999999999"])
def test_provider_numeric_exponent_outside_decimal_range_is_rejected(tmp_path: Path, token: str) -> None:
    """Refuse signed zero tokens whose exponent exceeds the decimal decoder's range.

    Binary64 accepts these spellings as zero, but the exact decimal comparison
    cannot represent their exponents. The actual captured-output reader must
    produce its authored parse refusal rather than admit an unverified token.
    """
    directory = _retained_run(tmp_path)
    (directory / "execution.json").unlink()
    flux_path = directory / "out.tglf.gbflux"
    tokens = flux_path.read_text(encoding="utf-8").split()
    tokens[0] = token
    flux_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")
    with pytest.raises(TGLFFluxError, match="Cannot parse standalone TGLF output"):
        read_tglf_fluxes(directory)


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "1D-400", "-1D-400"])
def test_provider_nonzero_token_underflow_is_rejected(tmp_path: Path, token: str) -> None:
    """Preserve the documented refusal of signed nonzero tokens that round to zero."""
    directory = _retained_run(tmp_path)
    (directory / "execution.json").unlink()
    flux_path = directory / "out.tglf.gbflux"
    tokens = flux_path.read_text(encoding="utf-8").split()
    tokens[0] = token
    flux_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")
    with pytest.raises(TGLFFluxError, match="Cannot parse standalone TGLF output"):
        read_tglf_fluxes(directory)
