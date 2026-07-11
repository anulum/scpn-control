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

#: Vacuum permeability mu0 in T*m/A.
MU0: float = 4.0e-7 * float(np.pi)
#: Tesla-to-gauss conversion factor.
TESLA_TO_GAUSS: float = 1.0e4


def amperes_to_megamperes(current_a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a current trace from amperes to megamperes."""
    return np.asarray(current_a, dtype=np.float64) / 1.0e6


def per_1e19(density_per_m3: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale an electron density from m^-3 to units of 1e19 m^-3."""
    return np.asarray(density_per_m3, dtype=np.float64) / 1.0e19


def toroidal_harmonic(
    saddle_tesla: NDArray[np.float64], angles_rad: NDArray[np.float64], n: int
) -> NDArray[np.complex128]:
    """Return the complex toroidal harmonic ``n`` of a saddle-coil array.

    ``saddle_tesla`` has shape ``(n_samples, n_coils)`` and ``angles_rad`` gives
    the toroidal angle of each coil. The standard mode-number projection
    ``A_n(t) = (2/N) * sum_k b_k(t) * exp(-i n phi_k)`` is used, exact for an
    evenly spaced array while ``n`` stays below the array Nyquist number
    ``n_coils / 2``.
    """
    saddle = np.asarray(saddle_tesla, dtype=np.float64)
    angles = np.asarray(angles_rad, dtype=np.float64)
    if saddle.ndim != 2:
        raise ValueError("saddle_tesla must be 2-D (n_samples, n_coils).")
    if angles.ndim != 1 or angles.shape[0] != saddle.shape[1]:
        raise ValueError("angles_rad must be 1-D with one angle per coil.")
    if n < 1:
        raise ValueError("harmonic n must be a positive integer.")
    n_coils = angles.shape[0]
    if n >= n_coils / 2.0:
        raise ValueError("harmonic n must stay below the array Nyquist number n_coils/2.")
    phasor: NDArray[np.complex128] = np.exp(-1j * float(n) * angles)
    return (2.0 / n_coils) * (saddle @ phasor)


def n_mode_amplitude(saddle_tesla: NDArray[np.float64], angles_rad: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Return the magnitude of toroidal harmonic ``n`` as a real amplitude trace."""
    return np.abs(toroidal_harmonic(saddle_tesla, angles_rad, n))


def locked_mode_envelope(
    saddle_tesla: NDArray[np.float64], angles_rad: NDArray[np.float64], *, window: int
) -> NDArray[np.float64]:
    """Return the non-rotating (locked) ``n=1`` amplitude envelope.

    The complex ``n=1`` phasor is boxcar-averaged over ``window`` samples: a
    rotating mode averages toward zero over a rotation period, so the surviving
    magnitude is the stationary, locked component. ``window`` should span roughly
    a mode rotation period and is tuned to the sampling rate at acquisition.
    """
    if window < 1:
        raise ValueError("window must be a positive number of samples.")
    phasor = toroidal_harmonic(saddle_tesla, angles_rad, 1)
    if window > phasor.shape[0]:
        raise ValueError("window must not exceed the number of samples.")
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    smoothed_real = np.convolve(phasor.real, kernel, mode="same")
    smoothed_imag = np.convolve(phasor.imag, kernel, mode="same")
    return np.abs(smoothed_real + 1j * smoothed_imag)


def dbdt_gauss_per_s(b_tesla: NDArray[np.float64], time_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``dB/dt`` in gauss per second from a field trace in tesla."""
    field = np.asarray(b_tesla, dtype=np.float64)
    time = np.asarray(time_s, dtype=np.float64)
    if field.ndim != 1 or field.shape != time.shape:
        raise ValueError("b_tesla and time_s must be matching 1-D arrays.")
    result: NDArray[np.float64] = np.gradient(field, time) * TESLA_TO_GAUSS
    return result


def q_at_psi_norm(
    q_profile: NDArray[np.float64], psi_norm_grid: NDArray[np.float64], *, target: float = 0.95
) -> NDArray[np.float64]:
    """Interpolate the safety factor at a normalised flux surface (default psi_n=0.95).

    ``q_profile`` is ``(n_samples, n_psi)`` (a 1-D single profile is promoted to a
    single sample); ``psi_norm_grid`` is the ``(n_psi,)`` normalised-flux axis.
    """
    profile = np.asarray(q_profile, dtype=np.float64)
    psi = np.asarray(psi_norm_grid, dtype=np.float64)
    if profile.ndim == 1:
        profile = profile[np.newaxis, :]
    if profile.ndim != 2:
        raise ValueError("q_profile must be 1-D or 2-D (n_samples, n_psi).")
    if psi.ndim != 1 or psi.shape[0] != profile.shape[1]:
        raise ValueError("psi_norm_grid must be 1-D with one value per profile column.")
    order = np.argsort(psi)
    psi_sorted = psi[order]
    return np.asarray(
        [float(np.interp(target, psi_sorted, row[order])) for row in profile],
        dtype=np.float64,
    )


def vacuum_toroidal_field(tf_current_a: NDArray[np.float64], r_geo_m: float, *, n_turns: int) -> NDArray[np.float64]:
    """Return the vacuum toroidal field ``B_phi = mu0 N I / (2 pi R)`` at the axis.

    ``n_turns`` and ``r_geo_m`` are MAST machine constants confirmed from the
    machine description at acquisition; no default is assumed here so the recipe
    cannot silently fabricate a field magnitude.
    """
    current = np.asarray(tf_current_a, dtype=np.float64)
    if r_geo_m <= 0.0:
        raise ValueError("r_geo_m must be positive.")
    if n_turns <= 0:
        raise ValueError("n_turns must be positive.")
    return MU0 * float(n_turns) * current / (2.0 * float(np.pi) * float(r_geo_m))
