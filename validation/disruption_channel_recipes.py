#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption channel derivation recipes
"""Pure numerical recipes that derive replay channels from level2 magnetics.

These functions turn raw FAIR-MAST level2 signal arrays into the derived
``run_real_shot_replay`` channels the feature-source audit marks ``derived``:
toroidal ``n``-mode amplitudes and the locked-mode envelope from a toroidal
saddle array, ``dB/dt`` from a poloidal probe, EFIT ``q95`` by flux
interpolation, and the vacuum toroidal field from the TF-coil current. They are
deterministic and NumPy-only, so the physics is unit-tested here independently of
the out-of-band Zarr acquisition that supplies the raw arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from validation.mast_replay_contracts._inputs import finite_array, positive_integer, time_axis

#: Vacuum permeability mu0 in T*m/A.
MU0: float = 4.0e-7 * float(np.pi)
#: Tesla-to-gauss conversion factor.
TESLA_TO_GAUSS: float = 1.0e4


def amperes_to_megamperes(current_a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale finite current values from A to MA without mutating their shape.

    Float64 conversion precedes division. Nonfinite values raise authored
    ``ValueError``; conversion failures propagate. NumPy determines the scalar
    versus array result for zero-dimensional input. No physical range is imposed.
    """
    return finite_array(current_a, name="current_a") / 1.0e6


def per_1e19(density_per_m3: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale finite density values from m^-3 to 10^19 m^-3 with their shape.

    Input bytes are preserved; conversion errors propagate and nonfinite values
    raise authored ``ValueError``. Negative values retain their sign. NumPy
    determines the scalar versus array result for zero-dimensional input.
    """
    return finite_array(density_per_m3, name="density_per_m3") / 1.0e19


def toroidal_harmonic(
    saddle_tesla: NDArray[np.float64], angles_rad: NDArray[np.float64], n: int
) -> NDArray[np.complex128]:
    """Return the complex toroidal harmonic ``n`` of a saddle-coil array.

    ``saddle_tesla`` has shape ``(n_samples, n_coils)`` and ``angles_rad`` gives
    the toroidal angle of each coil. The standard mode-number projection
    ``A_n(t) = (2/N) * sum_k b_k(t) * exp(-i n phi_k)`` is used, exact for an
    evenly spaced array while ``n`` stays below the array Nyquist number
    ``n_coils / 2``.

    Both arrays must be finite and are converted to float64 without mutation.
    ``n`` is a positive Python integer excluding bool. The fresh complex128
    ``(n_samples,)`` result is in tesla; empty sample rows are supported.
    Authored ``ValueError`` rejects dimensions, domains and Nyquist violations.
    Coil spacing/geometry is supplied by the caller, not validated or attested.
    """
    saddle = finite_array(saddle_tesla, name="saddle_tesla")
    angles = finite_array(angles_rad, name="angles_rad")
    if saddle.ndim != 2:
        raise ValueError("saddle_tesla must be 2-D (n_samples, n_coils).")
    if angles.ndim != 1 or angles.shape[0] != saddle.shape[1]:
        raise ValueError("angles_rad must be 1-D with one angle per coil.")
    n = positive_integer(n, name="harmonic n")
    n_coils = angles.shape[0]
    if n >= n_coils / 2.0:
        raise ValueError("harmonic n must stay below the array Nyquist number n_coils/2.")
    phasor: NDArray[np.complex128] = np.exp(-1j * float(n) * angles)
    return (2.0 / n_coils) * (saddle @ phasor)


def n_mode_amplitude(saddle_tesla: NDArray[np.float64], angles_rad: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Return a fresh float64 ``(n_samples,)`` harmonic magnitude in tesla.

    Inputs, authored domain errors and supplied-geometry limits are exactly
    those of ``toroidal_harmonic``; caller arrays remain unchanged.
    """
    return np.abs(toroidal_harmonic(saddle_tesla, angles_rad, n))


def locked_mode_envelope(
    saddle_tesla: NDArray[np.float64], angles_rad: NDArray[np.float64], *, window: int
) -> NDArray[np.float64]:
    """Return the non-rotating (locked) ``n=1`` amplitude envelope.

    The complex ``n=1`` phasor is boxcar-averaged over ``window`` samples: a
    rotating mode averages toward zero over a rotation period, so the surviving
    magnitude is the stationary, locked component. ``window`` should span roughly
    a mode rotation period and is tuned to the sampling rate at acquisition.

    ``window`` is a positive Python integer excluding bool and cannot exceed
    the sample count; even windows remain supported by this pure recipe. The
    fresh float64 trace is in tesla, with NumPy's ``same`` convolution's
    zero-padding at the ends. Invalid counts/arrays raise authored
    ``ValueError``. This candidate is not an attested stationary estimator.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive number of samples (integer, not bool).")
    phasor = toroidal_harmonic(saddle_tesla, angles_rad, 1)
    if window > phasor.shape[0]:
        raise ValueError("window must not exceed the number of samples.")
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    smoothed_real = np.convolve(phasor.real, kernel, mode="same")
    smoothed_imag = np.convolve(phasor.imag, kernel, mode="same")
    envelope: NDArray[np.float64] = np.abs(smoothed_real + 1j * smoothed_imag)
    return envelope


def dbdt_gauss_per_s(b_tesla: NDArray[np.float64], time_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Differentiate matching finite 1-D field/time inputs into a fresh G/s trace.

    ``b_tesla`` contains field in T and ``time_s`` seconds, with at least two
    strictly increasing samples. NumPy gradient uses its default first-order
    boundary differences and nonuniform interior formula. Authored
    ``ValueError`` rejects invalid shape, finite domain or chronology; no
    smoothing or source-quantity attestation is performed. Inputs are unchanged.
    """
    field = finite_array(b_tesla, name="b_tesla")
    time = finite_array(time_s, name="time_s")
    if field.ndim != 1 or field.shape != time.shape:
        raise ValueError("b_tesla and time_s must be matching 1-D arrays.")
    time_axis(time, name="time_s", minimum=2)
    result: NDArray[np.float64] = np.gradient(field, time) * TESLA_TO_GAUSS
    return result


def q_at_psi_norm(
    q_profile: NDArray[np.float64], psi_norm_grid: NDArray[np.float64], *, target: float = 0.95
) -> NDArray[np.float64]:
    """Interpolate the safety factor at a normalised flux surface (default psi_n=0.95).

    ``q_profile`` is ``(n_samples, n_psi)`` (a 1-D single profile is promoted to a
    single sample); ``psi_norm_grid`` is the ``(n_psi,)`` normalised-flux axis.

    Values, grid and scalar target must be finite. The nonempty grid is sorted
    and must have unique knots; supplied units/normalisation are not attested.
    Interpolation is linear with historical endpoint clamping outside its
    range. A fresh float64 ``(n_samples,)`` result is dimensionless; empty
    sample rows are supported. Invalid shapes/domains raise authored
    ``ValueError`` and conversion errors propagate. Caller arrays are unchanged.
    """
    profile = finite_array(q_profile, name="q_profile")
    psi = finite_array(psi_norm_grid, name="psi_norm_grid")
    if profile.ndim == 1:
        profile = profile[np.newaxis, :]
    if profile.ndim != 2:
        raise ValueError("q_profile must be 1-D or 2-D (n_samples, n_psi).")
    if psi.ndim != 1 or psi.shape[0] != profile.shape[1]:
        raise ValueError("psi_norm_grid must be 1-D with one value per profile column.")
    if psi.size == 0:
        raise ValueError("psi_norm_grid must contain at least one knot")
    target = float(target)
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    order = np.argsort(psi)
    psi_sorted = psi[order]
    if not bool(np.all(np.diff(psi_sorted) > 0.0)):
        raise ValueError("psi_norm_grid must have unique knots")
    return np.asarray(
        [float(np.interp(target, psi_sorted, row[order])) for row in profile],
        dtype=np.float64,
    )


def vacuum_toroidal_field(tf_current_a: NDArray[np.float64], r_geo_m: float, *, n_turns: int) -> NDArray[np.float64]:
    """Return the vacuum toroidal field ``B_phi = mu0 N I / (2 pi R)`` at the axis.

    The caller supplies ``n_turns`` and ``r_geo_m`` from the machine description
    and must verify their acquisition provenance; no geometry default is
    assumed here.

    Current in A must be finite, radius in metres finite and positive, and
    turns a positive Python integer excluding bool. Signed current preserves
    field sign. Output uses T with input shape (NumPy may return a scalar for
    zero-dimensional input). Authored ``ValueError`` rejects invalid domains;
    conversion and extreme arithmetic failures propagate. Inputs are unchanged
    and machine geometry/provenance remains the caller's responsibility.
    """
    current = finite_array(tf_current_a, name="tf_current_a")
    r_geo_m = float(r_geo_m)
    if not np.isfinite(r_geo_m) or r_geo_m <= 0.0:
        raise ValueError("r_geo_m must be positive and finite.")
    n_turns = positive_integer(n_turns, name="n_turns")
    return MU0 * float(n_turns) * current / (2.0 * float(np.pi) * float(r_geo_m))
