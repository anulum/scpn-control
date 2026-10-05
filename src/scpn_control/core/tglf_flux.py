# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Signed standalone TGLF flux execution.

"""Execute GACODE decks and retain normalized signed fluxes without a diffusion closure."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class _CapturedTextLoader(Protocol):
    """The native callable-converter loadtxt surface used for captured float64 text.

    NumPy 1.26 accepts this converter at runtime, but its stubs list only a
    converter mapping. Keep the actual narrow keyword contract at this boundary.
    """

    def __call__(
        self,
        fname: io.BytesIO,
        *,
        ndmin: Literal[1, 2],
        converters: Callable[[str | bytes], float],
        skiprows: int = 0,
    ) -> NDArray[np.float64]:
        """Read captured byte lines with the supplied converter and native dimensionality rules."""
        ...


_OUTPUTS = (
    "out.tglf.grid",
    "out.tglf.gbflux",
    "out.tglf.ky_spectrum",
    "out.tglf.eigenvalue_spectrum",
    "out.tglf.sum_flux_spectrum",
    "input.tglf.gen",
    "out.tglf.scalar_saturation_parameters",
)


class TGLFFluxError(RuntimeError):
    """A standalone execution or its normalized flux files are invalid."""


@dataclass(frozen=True)
class TGLFFluxResult:
    """Immutable, signed GACODE output in its input deck's reference units.

    Attributes
    ----------
    run_dir : Path
        Directory containing the input and output evidence.
    particle_flux_gb, energy_flux_gb, momentum_flux_gb, exchange_gb : tuple
        One value per species, electron first then ions. These are normalized
        fluxes integrated from full-precision, already weighted spectral rows,
        not diffusivities; negative values are retained. Momentum is toroidal
        stress, not the separate parallel-stress column. Reference
        density, temperature, length, mass, magnetic field and flux coordinate
        must be supplied separately before conversion to physical units.
    k_y : tuple
        Positive increasing normalized perpendicular wavenumbers.
    growth_rate, frequency : tuple of tuples
        Rows follow k_y; columns follow the provider's mode order. Both use
        the provider's normalized frequency convention. No mode relabelling,
        clipping, physical calibration or solver-convergence claim is made.
    """

    run_dir: Path
    particle_flux_gb: tuple[float, ...]
    energy_flux_gb: tuple[float, ...]
    momentum_flux_gb: tuple[float, ...]
    exchange_gb: tuple[float, ...]
    k_y: tuple[float, ...]
    growth_rate: tuple[tuple[float, ...], ...]
    frequency: tuple[tuple[float, ...], ...]


def read_tglf_fluxes(run_dir: Path) -> TGLFFluxResult:
    """Read a complete standalone GACODE flux/spectrum output set.

    Parameters
    ----------
    run_dir : Path
        Directory with grid, gbflux, ky_spectrum, eigenvalue_spectrum,
        sum_flux_spectrum, input.tglf.gen and scalar_saturation_parameters.
        Effective SAT_RULE/UNITS must match the requested GYRO contract.
        Files with execution.json require a literal JSON true validation flag and
        must match its validated output/input hashes.
        Without a receipt, this parser checks consistency, not freshness.

    Returns
    -------
    TGLFFluxResult
        Signed immutable output integrated once across weighted ky increments
        and configured fields. Coarse gbflux is checked within its printed
        half-unit rounding interval, never used as the returned moments.
        No flux-to-diffusivity conversion or ambipolar correction is applied.

    Raises
    ------
    TGLFFluxError
        Missing, nonfinite or structurally inconsistent output. The contract
        follows GACODE tglf.f90 and tglf_inout.f90 standalone writers.
    """
    captured = _capture_outputs(run_dir)
    receipt_path = run_dir / "execution.json"
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
            expected = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
            if (
                receipt["output_validated"] is not True
                or receipt["output_sha256"] != expected
                or receipt["input_sha256"] != hashlib.sha256((run_dir / "input.tglf").read_bytes()).hexdigest()
            ):
                raise TGLFFluxError("TGLF retained artifacts do not match the validated receipt")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TGLFFluxError("Cannot verify TGLF execution receipt") from exc
    result = _parse_outputs(run_dir, captured)
    _check_retained_outputs(run_dir, captured)
    return result


def _capture_outputs(run_dir: Path) -> dict[str, bytes]:
    """Read each required provider artifact once into the byte snapshot used for parsing and hashing."""
    try:
        return {name: (run_dir / name).read_bytes() for name in _OUTPUTS}
    except OSError as exc:
        raise TGLFFluxError(f"Cannot capture standalone TGLF output in {run_dir}") from exc


def _check_retained_outputs(run_dir: Path, captured: Mapping[str, bytes]) -> None:
    """Refuse evidence that changed between capture and the final retained-file check."""
    if _capture_outputs(run_dir) != captured:
        raise TGLFFluxError("TGLF retained output changed during validation")


def _finite_float(token: bytes | str) -> float:
    """Decode a numeric provider token without losing nonzero decimals below the binary64 range."""
    text = token.decode("ascii") if isinstance(token, bytes) else token
    text = text.replace("D", "E").replace("d", "e")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite TGLF numeric token")
    if value == 0:
        try:
            if Decimal(text) != 0:
                raise ValueError("nonzero TGLF numeric token underflows to zero")
        except InvalidOperation as exc:
            raise ValueError("TGLF numeric token is outside the decimal range") from exc
    return value


def _parse_outputs(run_dir: Path, captured: Mapping[str, bytes]) -> TGLFFluxResult:
    """Decode only captured bytes and validate species counts, signed moments and paired spectral modes."""
    load_numeric = cast(_CapturedTextLoader, np.loadtxt)
    try:
        grid = load_numeric(io.BytesIO(captured[_OUTPUTS[0]]), ndmin=1, converters=_finite_float)
        flux = load_numeric(io.BytesIO(captured[_OUTPUTS[1]]), ndmin=1, converters=_finite_float)
        nky = int(captured[_OUTPUTS[2]].splitlines()[1])
        ky = load_numeric(io.BytesIO(captured[_OUTPUTS[2]]), skiprows=2, ndmin=1, converters=_finite_float)
        spectrum = load_numeric(io.BytesIO(captured[_OUTPUTS[3]]), skiprows=2, ndmin=2, converters=_finite_float)
    except (ValueError, IndexError) as exc:
        raise TGLFFluxError(f"Cannot parse standalone TGLF output in {run_dir}") from exc

    if (
        grid.shape != (2,)
        or not np.all(np.isfinite(grid))
        or np.any(grid != np.floor(grid))
        or grid[0] < 2
        or grid[1] <= 0
    ):
        raise TGLFFluxError("TGLF grid must contain integral species and spatial counts")
    species = int(grid[0])
    if flux.shape != (4 * species,) or not np.all(np.isfinite(flux)):
        raise TGLFFluxError("TGLF gbflux must contain four finite groups for every species")
    if (
        nky <= 0
        or ky.shape != (nky,)
        or not np.all(np.isfinite(ky))
        or np.any(ky <= 0)
        or np.any(np.diff(ky) <= 0)
        or spectrum.shape[0] != nky
        or spectrum.shape[1] < 2
        or spectrum.shape[1] % 2
        or not np.all(np.isfinite(spectrum))
    ):
        raise TGLFFluxError("TGLF ky and paired growth/frequency spectra are inconsistent")
    fields = _resolved_fields(
        captured["input.tglf.gen"],
        captured["out.tglf.scalar_saturation_parameters"],
        species,
        int(grid[1]),
        spectrum.shape[1] // 2,
    )
    groups = _integrated_moments(captured["out.tglf.sum_flux_spectrum"], species, fields, nky)
    _check_rounded_moments(captured["out.tglf.gbflux"], groups)
    return TGLFFluxResult(
        run_dir=run_dir,
        particle_flux_gb=tuple(float(x) for x in groups[0]),
        energy_flux_gb=tuple(float(x) for x in groups[1]),
        momentum_flux_gb=tuple(float(x) for x in groups[2]),
        exchange_gb=tuple(float(x) for x in groups[3]),
        k_y=tuple(float(x) for x in ky),
        growth_rate=tuple(tuple(float(x) for x in row) for row in spectrum[:, ::2]),
        frequency=tuple(tuple(float(x) for x in row) for row in spectrum[:, 1::2]),
    )


def _resolved_fields(data: bytes, effective_data: bytes, species: int, spatial: int, modes: int) -> int:
    """Check requested/effective kinetic GYRO settings and return the complete configured field count."""
    try:
        rows = [line.split() for line in data.decode("ascii").splitlines() if line.strip()]
        if any(len(row) != 2 for row in rows):
            raise ValueError("expected value/key pairs")
        params = {row[1]: row[0] for row in rows}
        if len(params) != len(rows):
            raise ValueError("duplicate resolved input key")
        expected = {"UNITS": "GYRO", "USE_TRANSPORT_MODEL": ".true.", "IFLUX": ".true.", "ADIABATIC_ELEC": ".false."}
        if any(params[key].lower() != value.lower() for key, value in expected.items()):
            raise ValueError("requires kinetic-electron GYRO transport fluxes")
        effective_rows: list[tuple[str, str]] = []
        for line in effective_data.decode("ascii").splitlines():
            match = re.fullmatch(r"\s*(SAT_RULE|UNITS)\s*=\s*(\S+)\s*", line)
            if match:
                effective_rows.append((match.group(1), match.group(2)))
        effective = dict(effective_rows)
        if len(effective_rows) != 2 or set(effective) != {"SAT_RULE", "UNITS"}:
            raise ValueError("missing or duplicate effective saturation settings")
        if effective["UNITS"] != "GYRO" or int(effective["SAT_RULE"]) != int(params["SAT_RULE"]):
            raise ValueError("provider saturation presets changed the declared GYRO contract")
        if (int(params["NS"]), int(params["NXGRID"]), int(params["NMODES"])) != (species, spatial, modes):
            raise ValueError("resolved dimensions disagree with output")
        if any(params[key].lower() not in {".true.", ".false."} for key in ("USE_BPER", "USE_BPAR")):
            raise ValueError("invalid resolved field flag")
        return 3 if params["USE_BPAR"].lower() == ".true." else 2 if params["USE_BPER"].lower() == ".true." else 1
    except (ValueError, KeyError, UnicodeError) as exc:
        raise TGLFFluxError("TGLF resolved flux configuration is inconsistent") from exc


def _integrated_moments(data: bytes, species: int, fields: int, nky: int) -> NDArray[np.float64]:
    """Sum the writer's weighted ky increments once, including every configured species/field block."""
    try:
        lines = [line.strip() for line in data.decode("ascii").splitlines() if line.strip()]
        if len(lines) != species * fields * (nky + 2):
            raise ValueError("incomplete or extra spectral blocks")
        values = np.empty((species, fields, nky, 5), dtype=np.float64)
        offset = 0
        for sp in range(species):
            for field in range(fields):
                header = re.fullmatch(r"species\s*=\s*(\d+)\s+field\s*=\s*(\d+)", lines[offset])
                if header is None or tuple(map(int, header.groups())) != (sp + 1, field + 1):
                    raise ValueError("missing, duplicate or reordered species/field")
                if lines[offset + 1] != "particle flux,energy flux,toroidal stress,parallel stress,exchange":
                    raise ValueError("unknown spectral column convention")
                for ky in range(nky):
                    row = lines[offset + 2 + ky].split()
                    if len(row) != 5:
                        raise ValueError("expected five spectral moments")
                    values[sp, field, ky] = [_finite_float(token) for token in row]
                offset += nky + 2
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite spectral moment")
        integrated = np.array(
            [
                [math.fsum(float(x) for x in values[sp, :, :, moment].flat) for sp in range(species)]
                for moment in (0, 1, 2, 4)
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(integrated)):
            raise ValueError("nonfinite integrated moment")
        return integrated
    except (ValueError, OverflowError, UnicodeError) as exc:
        raise TGLFFluxError("TGLF weighted flux spectrum is inconsistent") from exc


def _check_rounded_moments(data: bytes, groups: NDArray[np.float64]) -> None:
    """Require each integrated moment to lie in its 1pe11.4 summary token's decimal rounding interval."""
    tokens = data.split()
    if len(tokens) != groups.size:
        raise TGLFFluxError("TGLF gbflux must contain only the expected numeric tokens")
    for token, value in zip(tokens, groups.flat, strict=True):
        if re.fullmatch(rb"[+-]?\d\.\d{4}[Ee][+-]\d{2,3}", token) is None:
            raise TGLFFluxError("TGLF gbflux does not use the expected 1pe11.4 precision")
        rounded = Decimal(token.decode("ascii"))
        exponent = int(token.lower().split(b"e")[1]) - 4
        half_quantum = Decimal((0, (5,), exponent - 1))
        if abs(Fraction.from_float(float(value)) - Fraction(rounded)) > Fraction(half_quantum):
            raise TGLFFluxError("TGLF integrated flux disagrees with rounded gbflux")


class TGLFFluxSolver:
    """Run an existing GACODE input deck in a fresh directory on POSIX.

    Parameters
    ----------
    work_dir : Path
        Caller-owned parent for persistent per-execution evidence directories.
    binary : str
        GACODE standalone launcher name or absolute path.
    environment : mapping or None
        Child-only overrides for the existing process environment, including
        GACODE_ROOT and dependency paths if needed. Values are not logged.

    Notes
    -----
    Input is the provider's key=value deck, not GKLocalParams or a namelist.
    The provider owns defaults and input validation. This class deliberately
    returns normalized signed fluxes rather than the coefficient-only GKOutput.
    Runtime coupling must specify reference units and a conservation contract.
    """

    def __init__(
        self,
        work_dir: Path,
        binary: str = "tglf",
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Store evidence paths and copy child environment overrides without launching a provider."""
        self.work_dir = work_dir
        self.binary = binary
        self.environment = dict(environment or {})

    def run(self, input_deck: Path, *, timeout_s: float = 30.0) -> TGLFFluxResult:
        """Execute a copied input deck and retain logs and a hashed execution receipt.

        Parameters
        ----------
        input_deck : Path
            Existing standalone input.tglf-format file. Its bytes are copied;
            neighboring outputs are never imported.
        timeout_s : float
            Positive finite child-process timeout in seconds; booleans are
            rejected. On timeout the
            owned POSIX process group is killed and its leader reaped. Cleanup
            is also attempted after success, exceptions and cancellation, with
            a separate two-second leader-reaping limit. Descendants that create
            another session/group are outside that termination boundary.

        Returns
        -------
        TGLFFluxResult
            Validated raw signed output from this execution directory.

        Raises
        ------
        ValueError
            Timeout is boolean or is not positive and finite.
        TGLFFluxError
            Unsupported OS, unavailable executable, execution failure, timeout
            or incomplete/changed output. The failed run directory and logs remain.
            Parsing and output hashes use the same captured bytes; retained
            files are checked before publishing success. Later edits are
            detected by read_tglf_fluxes, not prevented by this API.
            Launcher identity is checked before/after execution; it does not
            authenticate the launcher's dependent executables.
        BaseException
            Original wait/cancellation failure, after bounded cleanup. Cleanup
            failures are recorded and attached as exception notes.
        OSError
            Input or evidence directory cannot be read or written.
        """
        if isinstance(timeout_s, (bool, np.bool_)) or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if os.name != "posix":
            raise TGLFFluxError("Standalone TGLF process-group execution requires POSIX")
        env = dict(os.environ)
        env.update(self.environment)
        binary = shutil.which(self.binary, path=env.get("PATH"))
        if binary is None:
            raise TGLFFluxError(f"TGLF launcher is unavailable: {self.binary}")
        binary = str(Path(binary).resolve())
        deck = input_deck.read_bytes()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="tglf-", dir=self.work_dir)).resolve()
        (run_dir / "input.tglf").write_bytes(deck)
        argv = [binary, "-e", "."]
        launcher_bytes = Path(binary).read_bytes()
        failure: BaseException | None = None
        cleanup_errors: list[str] = []
        process: subprocess.Popen[bytes] | None = None
        # Regular files cannot leave communicate() waiting for pipes retained by
        # descendants that have escaped the owned POSIX process group.
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    argv, cwd=run_dir, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                )
                process.wait(timeout=timeout_s)
            except BaseException as exc:
                failure = exc
            finally:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except BaseException as exc:
                        cleanup_errors.append(f"process-group cleanup: {type(exc).__name__}: {exc}")
                        if failure is None:
                            failure = exc
                    try:
                        process.wait(timeout=2.0)
                    except BaseException as exc:
                        cleanup_errors.append(f"leader reaping: {type(exc).__name__}: {exc}")
                        if failure is None:
                            failure = exc
        timed_out = isinstance(failure, subprocess.TimeoutExpired)
        receipt: dict[str, object] = {
            "argv": argv,
            "input_sha256": hashlib.sha256(deck).hexdigest(),
            "launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
            "launcher_unchanged": False,
            "output_validated": False,
            "pid": None if process is None else process.pid,
            "returncode": None if process is None else process.returncode,
            "timed_out": timed_out,
            "failure_type": None if failure is None else type(failure).__name__,
            "cleanup_errors": cleanup_errors,
            "output_sha256": {},
        }
        try:
            (run_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        except OSError as exc:
            if failure is None:
                raise
            failure.add_note(f"Cannot write TGLF failure receipt: {exc}")
        if failure is not None:
            for error in cleanup_errors:
                failure.add_note(error)
            if timed_out:
                raise TGLFFluxError(f"TGLF execution failed (timeout=True) in {run_dir}") from failure
            raise failure
        if process is None or process.returncode != 0:
            raise TGLFFluxError(f"TGLF execution failed (exit={receipt['returncode']}) in {run_dir}")
        captured = _capture_outputs(run_dir)
        receipt["output_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
        (run_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result = _parse_outputs(run_dir, captured)
        _check_retained_outputs(run_dir, captured)
        if (run_dir / "input.tglf").read_bytes() != deck or Path(binary).read_bytes() != launcher_bytes:
            raise TGLFFluxError("TGLF input or launcher changed during execution")
        receipt["launcher_unchanged"] = True
        receipt["output_validated"] = True
        (run_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return result
