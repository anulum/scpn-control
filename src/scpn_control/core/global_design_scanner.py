# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Global Design Scanner
"""Bounded tokamak design scanner for disruption-contract episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from scpn_control.core._validators import require_positive_float
from scpn_control.core.fusion_kernel import dt_fusion_power_mw, neutron_wall_loading_mw_m2

_EV_J = 1.602176634e-19
_KEV_J = 1.0e3 * _EV_J
_MU0 = 4.0e-7 * np.pi
_DT_REFERENCE_REACTIVITY_M3_S = 1.1e-22
_DEFAULT_ASPECT_RATIO = 3.1
_DEFAULT_ELONGATION = 1.75
_DEFAULT_DENSITY_FRACTION = 0.82
_DEFAULT_CONFINEMENT_MULTIPLIER = 1.0
_DEFAULT_AUXILIARY_POWER_MW = 55.0


def _positive_config_value(config: Mapping[str, float], name: str, default: float) -> float:
    """Return a positive scalar configuration value."""
    return require_positive_float(name, config.get(name, default))


@dataclass(frozen=True, slots=True)
class DesignScannerConfig:
    """Configuration constants for ``GlobalDesignExplorer``.

    Parameters
    ----------
    aspect_ratio
        Major-radius to minor-radius ratio used to infer the plasma minor
        radius.
    elongation
        Plasma elongation for volume and neutron-wall-load estimates.
    density_fraction
        Fraction of the Greenwald density used for the bounded power estimate.
    confinement_multiplier
        Multiplier applied to the IPB98-like confinement-time proxy.
    auxiliary_power_mw
        External heating and current-drive power used in the Q proxy.
    """

    aspect_ratio: float = _DEFAULT_ASPECT_RATIO
    elongation: float = _DEFAULT_ELONGATION
    density_fraction: float = _DEFAULT_DENSITY_FRACTION
    confinement_multiplier: float = _DEFAULT_CONFINEMENT_MULTIPLIER
    auxiliary_power_mw: float = _DEFAULT_AUXILIARY_POWER_MW

    @classmethod
    def from_mapping(cls, config: Mapping[str, float]) -> "DesignScannerConfig":
        """Build a scanner configuration from scalar overrides."""
        return cls(
            aspect_ratio=_positive_config_value(config, "aspect_ratio", _DEFAULT_ASPECT_RATIO),
            elongation=_positive_config_value(config, "elongation", _DEFAULT_ELONGATION),
            density_fraction=_positive_config_value(config, "density_fraction", _DEFAULT_DENSITY_FRACTION),
            confinement_multiplier=_positive_config_value(
                config,
                "confinement_multiplier",
                _DEFAULT_CONFINEMENT_MULTIPLIER,
            ),
            auxiliary_power_mw=_positive_config_value(config, "auxiliary_power_mw", _DEFAULT_AUXILIARY_POWER_MW),
        )


class GlobalDesignExplorer:
    """Evaluate bounded tokamak design points for control-contract episodes.

    The explorer provides the small production surface required by
    ``run_disruption_episode``. It is a deterministic bounded model, not a
    replacement for a systems-code design scan or neutronics campaign.

    Parameters
    ----------
    config
        Preset name or scalar configuration mapping. Preset names currently
        select the default bounded model and are retained for compatibility
        with older callers that passed labels such as ``"dummy"``.
    """

    def __init__(self, config: str | Mapping[str, float] | None = None) -> None:
        if isinstance(config, Mapping):
            self._config = DesignScannerConfig.from_mapping(config)
            self.label = "mapping"
        else:
            self._config = DesignScannerConfig()
            self.label = "baseline" if config is None else str(config)

    def evaluate_design(self, r_maj: float, b_t: float, ip: float) -> dict[str, float]:
        """Evaluate one tokamak design point.

        Parameters
        ----------
        r_maj
            Major radius in metres.
        b_t
            Toroidal magnetic field in tesla.
        ip
            Plasma current in mega-amperes.

        Returns
        -------
        dict[str, float]
            Bounded design metrics: ``Q``, ``P_fusion_MW``,
            ``neutron_wall_load_MW_m2``, ``cost_index``, ``minor_radius_m``,
            ``volume_m3``, ``greenwald_fraction``, and ``beta_p_proxy``.
        """
        r_maj = require_positive_float("r_maj", r_maj)
        b_t = require_positive_float("b_t", b_t)
        ip = require_positive_float("ip", ip)

        minor_radius_m = r_maj / self._config.aspect_ratio
        volume_m3 = 2.0 * np.pi**2 * r_maj * minor_radius_m * minor_radius_m * self._config.elongation
        greenwald_density = ip * 1.0e20 / (np.pi * minor_radius_m * minor_radius_m)
        density = self._config.density_fraction * greenwald_density
        temperature_kev = self._temperature_proxy_kev(r_maj=r_maj, b_t=b_t, ip=ip)
        reactivity = self._reactivity_proxy(temperature_kev)
        fusion_power_mw = dt_fusion_power_mw(
            n_D_m3=0.5 * density,
            n_T_m3=0.5 * density,
            sigv_m3s=reactivity,
            V_m3=volume_m3,
        )
        stored_energy_mj = 1.5 * density * volume_m3 * temperature_kev * _KEV_J * 1.0e-6
        confinement_s = self._confinement_proxy_s(r_maj=r_maj, b_t=b_t, ip=ip, density=density)
        loss_power_mw = stored_energy_mj / max(confinement_s, 1.0e-6)
        q_value = fusion_power_mw / max(loss_power_mw + self._config.auxiliary_power_mw, 1.0e-6)
        beta_p_proxy = self._beta_p_proxy(ip=ip, minor_radius_m=minor_radius_m, stored_energy_mj=stored_energy_mj)
        neutron_load = neutron_wall_loading_mw_m2(
            P_fus_MW=fusion_power_mw,
            R0_m=r_maj,
            a_m=minor_radius_m,
            kappa=self._config.elongation,
        )
        cost_index = self._cost_index(r_maj=r_maj, b_t=b_t, ip=ip)
        return {
            "Q": float(q_value),
            "P_fusion_MW": float(fusion_power_mw),
            "neutron_wall_load_MW_m2": float(neutron_load),
            "cost_index": float(cost_index),
            "minor_radius_m": float(minor_radius_m),
            "volume_m3": float(volume_m3),
            "greenwald_fraction": float(self._config.density_fraction),
            "beta_p_proxy": float(beta_p_proxy),
        }

    @staticmethod
    def _temperature_proxy_kev(*, r_maj: float, b_t: float, ip: float) -> float:
        """Return a bounded ion-temperature proxy in keV."""
        raw = 3.5 + 0.55 * b_t + 0.28 * ip - 0.22 * r_maj
        return float(np.clip(raw, 3.0, 28.0))

    @staticmethod
    def _reactivity_proxy(temperature_kev: float) -> float:
        """Return a bounded DT reactivity proxy in ``m^3 s^-1``."""
        scaled = (temperature_kev / 12.0) ** 2
        rolloff = 1.0 + (temperature_kev / 42.0) ** 2
        return float(_DT_REFERENCE_REACTIVITY_M3_S * scaled / rolloff)

    def _confinement_proxy_s(self, *, r_maj: float, b_t: float, ip: float, density: float) -> float:
        """Return an IPB98-like energy-confinement proxy in seconds."""
        density_20 = max(density / 1.0e20, 1.0e-6)
        tau = 0.045 * (ip**0.93) * (b_t**0.15) * (density_20**0.41) * (r_maj**1.25)
        return float(self._config.confinement_multiplier * tau)

    @staticmethod
    def _beta_p_proxy(*, ip: float, minor_radius_m: float, stored_energy_mj: float) -> float:
        """Return a poloidal-beta proxy from stored energy and current."""
        poloidal_field_t = _MU0 * ip * 1.0e6 / max(2.0 * np.pi * minor_radius_m, 1.0e-9)
        pressure_proxy = stored_energy_mj * 1.0e6 / max(3.0 * np.pi * minor_radius_m**3, 1.0e-9)
        magnetic_pressure = poloidal_field_t * poloidal_field_t / max(2.0 * _MU0, 1.0e-12)
        return float(np.clip(pressure_proxy / max(magnetic_pressure, 1.0e-12), 0.0, 10.0))

    @staticmethod
    def _cost_index(*, r_maj: float, b_t: float, ip: float) -> float:
        """Return a relative design-cost proxy."""
        magnet_term = (b_t / 10.0) ** 2 * (r_maj / 1.4)
        current_term = 0.25 * (ip / 6.0) ** 1.4
        size_term = (r_maj / 1.4) ** 2.2
        return float(size_term + magnet_term + current_term)


__all__ = ["DesignScannerConfig", "GlobalDesignExplorer"]
