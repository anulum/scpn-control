# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Legacy diagnostic examples.

"""Generate legacy noisy diagnostic examples without physical calibration.

This defining owner retains NumPy global-RNG behavior and the documented ignored
inputs/proxy geometry. Samples do not establish measured diagnostic fidelity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class SyntheticDiagnosticSuite:
    """Generate legacy noisy examples, not calibrated diagnostic forward models.

    Methods use NumPy's global random state and do not validate input domains.
    Gaussian factors are unbounded and can produce negative samples. Geometry
    and calibration limitations are documented on each method. These samples
    cannot establish reconstructed equilibrium or measured instrument fidelity.
    """

    def thomson_scattering(
        self, Te: npt.NDArray[np.float64], ne: npt.NDArray[np.float64], n_channels: int = 20
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Sample temperature and density at evenly spaced integer indices.

        Te is in keV and ne in 1e19 m^-3; returned arrays retain those units.
        Indices are chosen from Te length and also applied to ne, so callers
        must supply compatible one-dimensional arrays. Channel counts larger
        than the profile can repeat indices. Independent multiplicative normal
        noise has standard deviations 5% and 3%; no spatial response, calibration
        or positivity clipping is modeled.
        """
        indices = np.linspace(0, len(Te) - 1, n_channels, dtype=int)

        # Add 5% noise to Te, 3% to ne
        Te_meas = Te[indices] * (1.0 + np.random.randn(n_channels) * 0.05)
        ne_meas = ne[indices] * (1.0 + np.random.randn(n_channels) * 0.03)

        return {"Te_keV": Te_meas, "ne_19": ne_meas}

    def ece_radiometer(
        self, Te: npt.NDArray[np.float64], B_profile: npt.NDArray[np.float64], n_channels: int = 32
    ) -> npt.NDArray[np.float64]:
        """Sample Te in keV with 2% multiplicative normal noise.

        Uses evenly spaced integer indices, potentially repeated. B_profile is
        currently ignored: no resonance-frequency mapping, optical-depth or
        radiative-transfer model is evaluated. The returned channels therefore
        provide noisy temperature examples rather than an ECE forward model.
        """
        indices = np.linspace(0, len(Te) - 1, n_channels, dtype=int)
        # ECE noise typically ~2%
        Te_meas: npt.NDArray[np.float64] = Te[indices] * (1.0 + np.random.randn(n_channels) * 0.02)
        return Te_meas

    def interferometer(
        self, ne: npt.NDArray[np.float64], rho: npt.NDArray[np.float64], a: float, n_chords: int = 8
    ) -> npt.NDArray[np.float64]:
        # Mock line integration
        """Multiply mean density by assumed circular chord lengths with 1% noise.

        ne is in 1e19 m^-3 and a in meters; outputs are in 1e19 m^-2.
        Chord impact parameters span 0 to 0.9 times a. rho is ignored and the
        arithmetic mean replaces profile integration, so equal means yield
        equal noiseless outputs regardless of radial structure. No instrument
        phase response or reconstruction is evaluated.
        """
        avg_n = np.mean(ne)
        path_lengths = 2.0 * a * np.sqrt(1.0 - np.linspace(0, 0.9, n_chords) ** 2)

        ideal_integrals = avg_n * path_lengths
        # 1% noise
        meas: npt.NDArray[np.float64] = ideal_integrals * (1.0 + np.random.randn(n_chords) * 0.01)
        return meas

    def bolometer(
        self, P_rad_profile: npt.NDArray[np.float64], rho: npt.NDArray[np.float64], a: float, n_chords: int = 16
    ) -> npt.NDArray[np.float64]:
        """Multiply mean radiation input by circular chord lengths with 10% noise.

        a is in meters; output units are input units times meters. Callers must
        specify what P_rad_profile represents; the method does not convert a
        volume power density into calibrated detector power. rho is ignored and
        no radial line integration, solid angle or spectral response is modeled.
        """
        avg_P = np.mean(P_rad_profile)
        path_lengths = 2.0 * a * np.sqrt(1.0 - np.linspace(0, 0.9, n_chords) ** 2)

        ideal_integrals = avg_P * path_lengths
        # 10% noise for bolometry
        meas: npt.NDArray[np.float64] = ideal_integrals * (1.0 + np.random.randn(n_chords) * 0.10)
        return meas

    def soft_xray(
        self,
        Te: npt.NDArray[np.float64],
        ne: npt.NDArray[np.float64],
        rho: npt.NDArray[np.float64],
        n_chords: int = 40,
    ) -> npt.NDArray[np.float64]:
        # Emissivity ~ n_e^2 * sqrt(Te)
        """Return noisy mean ne**2 * sqrt(Te) examples on nominal channels.

        Te is in keV and ne in 1e19 m^-3, but output is an uncalibrated proxy,
        not watts or detected photon counts. rho is ignored; all chord lengths
        are set to one. A 5% multiplicative normal factor is applied independently
        per channel. Negative temperatures propagate NumPy invalid-value behavior;
        no atomic emissivity, geometry or detector response is calculated.
        """
        emissivity = ne**2 * np.sqrt(Te)
        avg_emis = np.mean(emissivity)
        path_lengths = np.ones(n_chords)  # Simplified

        meas: npt.NDArray[np.float64] = avg_emis * path_lengths * (1.0 + np.random.randn(n_chords) * 0.05)
        return meas

    def magnetics(self, R0: float, a: float) -> dict[str, Any]:
        # Just mock standard sensors
        """Return random constants labeled as magnetic sensor examples.

        R0 and a are ignored. Twenty flux-loop values center on 1, thirty probe
        values on 0.5 and the scalar Ip on 1, with relative normal noise of
        0.1%, 0.5% and 1%. No physical units or calibration are established by
        these constants; Ip is unrelated to MachineConfig.Ip_MA. This method
        performs no field calculation and cannot support current reconstruction.
        """
        return {
            "flux_loops": np.ones(20) * 1.0 * (1.0 + np.random.randn(20) * 0.001),
            "b_probes": np.ones(30) * 0.5 * (1.0 + np.random.randn(30) * 0.005),
            "Ip": 1.0 * (1.0 + np.random.randn() * 0.01),
        }
