# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Signed standalone TGLF flux output reader.

"""Read retained GACODE output as normalised signed fluxes without a diffusion closure.

The launcher that executes the provider lives outside this package, in
``validation/tglf_launcher.py``: it needs an installed GACODE and cannot be
exercised where none exists. This module only reads what a run retained.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
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
        One value per species, electron first then ions. These are normalised
        fluxes integrated from full-precision, already weighted spectral rows,
        not diffusivities; negative values are retained. Momentum is toroidal
        stress, not the separate parallel-stress column. Reference
        density, temperature, length, mass, magnetic field and flux coordinate
        must be supplied separately before conversion to physical units.
    k_y : tuple
        Positive increasing normalised perpendicular wavenumbers.
    growth_rate, frequency : tuple of tuples
        Rows follow k_y; columns follow the provider's mode order. Both use
        the provider's normalised frequency convention. No mode relabelling,
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
        Missing, nonfinite or structurally inconsistent output, or retained
        input/receipt bytes changed or became unreadable during parsing. The
        contract follows GACODE tglf.f90 and tglf_inout.f90 standalone writers.

    Notes
    -----
    Receipt, input and output checks are sequential snapshots, not a filesystem
    lock. They cannot prevent replacement after the last check or after return.
    """
    captured = capture_tglf_outputs(run_dir)
    receipt_path = run_dir / "execution.json"
    retained_admission: dict[Path, bytes] = {}
    if receipt_path.exists():
        try:
            input_path = run_dir / "input.tglf"
            retained_admission = {
                receipt_path: receipt_path.read_bytes(),
                input_path: input_path.read_bytes(),
            }
            receipt = json.loads(retained_admission[receipt_path].decode("utf-8"))
            expected = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
            if (
                receipt["output_validated"] is not True
                or receipt["output_sha256"] != expected
                or receipt["input_sha256"] != hashlib.sha256(retained_admission[input_path]).hexdigest()
            ):
                raise TGLFFluxError("TGLF retained artifacts do not match the validated receipt")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TGLFFluxError("Cannot verify TGLF execution receipt") from exc
    result = parse_captured_tglf_fluxes(run_dir, captured)
    for path, admitted_bytes in retained_admission.items():
        try:
            if path.read_bytes() != admitted_bytes:
                raise TGLFFluxError("TGLF retained admission changed during validation")
        except OSError as exc:
            raise TGLFFluxError("Cannot recheck TGLF retained admission") from exc
    return result


def capture_tglf_outputs(run_dir: Path) -> dict[str, bytes]:
    """Read each required provider artifact once into one byte snapshot.

    Parameters
    ----------
    run_dir : Path
        Directory that holds the complete standalone output set.

    Returns
    -------
    dict of str to bytes
        Member name to its bytes, in the fixed member order. Parsing and
        hashing both use this snapshot, so they describe the same bytes.

    Raises
    ------
    TGLFFluxError
        A member is missing or cannot be read.
    """
    try:
        return {name: (run_dir / name).read_bytes() for name in _OUTPUTS}
    except OSError as exc:
        raise TGLFFluxError(f"Cannot capture standalone TGLF output in {run_dir}") from exc


def parse_captured_tglf_fluxes(run_dir: Path, captured: Mapping[str, bytes]) -> TGLFFluxResult:
    """Parse one byte snapshot and require the retained files to still equal it.

    Parameters
    ----------
    run_dir : Path
        Directory the snapshot was captured from.
    captured : mapping of str to bytes
        Snapshot returned by ``capture_tglf_outputs`` for ``run_dir``.

    Returns
    -------
    TGLFFluxResult
        Signed immutable output decoded from the snapshot only.

    Raises
    ------
    TGLFFluxError
        The snapshot is structurally inconsistent, or a retained file changed
        between the capture and the end of parsing.
    """
    result = _parse_outputs(run_dir, captured)
    if capture_tglf_outputs(run_dir) != captured:
        raise TGLFFluxError("TGLF retained output changed during validation")
    return result


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
        # Every entry comes from ``_finite_float``, and ``math.fsum`` of finite
        # terms either returns a finite sum or raises ``OverflowError``.
        return np.array(
            [
                [math.fsum(float(x) for x in values[sp, :, :, moment].flat) for sp in range(species)]
                for moment in (0, 1, 2, 4)
            ],
            dtype=np.float64,
        )
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
