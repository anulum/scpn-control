# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated Transport Solver

"""Integrated radial heat, particle, neoclassical, bootstrap, and source transport solver."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np

HAS_MPL = find_spec("matplotlib") is not None
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    # Type-check against the concrete Python base class. At runtime the Rust-backed
    # wrapper (typed Any in _rust_compat) is preferred when available; it mirrors the
    # Python FusionKernel attribute interface, so the Python class is the correct
    # static base for TransportSolver and avoids subclassing Any.
    from scpn_control.core.fusion_kernel import FusionKernel
    from scpn_control.core.gk_interface import GKSolverBase
    from scpn_control.core.gk_tglf_native import TGLFNativeSolver
else:
    try:
        from scpn_control.core._rust_compat import FusionKernel
    except ImportError:
        from scpn_control.core.fusion_kernel import FusionKernel

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.adaptive_time_controller import AdaptiveTimeController as AdaptiveTimeController
from scpn_control.core.aux_heating import aux_heating_source_profiles as aux_heating_source_profiles
from scpn_control.core.momentum_transport import (
    MomentumTransportSolver,
)
from scpn_control.core.momentum_transport import (
    intrinsic_rotation_torque as intrinsic_rotation_torque,
)
from scpn_control.core.momentum_transport import (
    nbi_torque as nbi_torque,
)
from scpn_control.core.pedestal import PedestalProfile
from scpn_control.core.plasma_power_terms import bremsstrahlung_power_density as bremsstrahlung_power_density
from scpn_control.core.radial_diffusion import (
    build_cn_tridiag as build_cn_tridiag,
)
from scpn_control.core.radial_diffusion import (
    explicit_diffusion_rhs as explicit_diffusion_rhs,
)
from scpn_control.core.radial_diffusion import (
    thomas_solve as thomas_solve,
)
from scpn_control.core.runtime_sanitization import sanitize_with_fallback as sanitize_with_fallback
from scpn_control.core.species_evolution import SpeciesParticleBalance
from scpn_control.core.species_evolution import evolve_multi_ion_species as evolve_multi_ion_species
from scpn_control.core.transport_face_runtime import evolve_fluxes_impl
from scpn_control.core.transport_flux import (
    TransportFaceFlux,
    TransportFaceGeometry,
    TransportFluxBalance,
)
from scpn_control.core.transport_flux import (
    advance_face_flux as advance_face_flux,
)
from scpn_control.core.transport_geometry import (
    canonical_radial_grid,
    is_canonical_radial_grid,
)
from scpn_control.core.transport_geometry import (
    rho_volume_element as rho_volume_element,
)
from scpn_control.core.transport_model_selection import (
    _external_gk_transport_impl,
    _gyro_bohm_chi_impl,
    _legacy_bootstrap_current_approx_impl,
    _tglf_native_transport_impl,
    calculate_bootstrap_current_impl,
    chang_hinton_chi_profile_impl,
    inject_impurities_impl,
    set_neoclassical_impl,
    update_transport_model_impl,
)
from scpn_control.core.transport_neoclassical import (
    _finite_scalar as _finite_scalar,
)
from scpn_control.core.transport_neoclassical import (
    _load_gyro_bohm_coefficient as _load_gyro_bohm_coefficient,
)
from scpn_control.core.transport_neoclassical import (
    _normalised_radius as _normalised_radius,
)
from scpn_control.core.transport_neoclassical import (
    _profile_array as _profile_array,
)
from scpn_control.core.transport_neoclassical import (
    _validate_tokamak_geometry as _validate_tokamak_geometry,
)
from scpn_control.core.transport_neoclassical import (
    calculate_sauter_bootstrap_current_full as calculate_sauter_bootstrap_current_full,
)
from scpn_control.core.transport_neoclassical import (
    chang_hinton_chi_profile as chang_hinton_chi_profile,
)
from scpn_control.core.transport_orchestration import (
    _compute_confinement_time,
    _map_profiles_to_2d,
    _run_self_consistent,
    _run_to_steady_state,
)
from scpn_control.core.transport_species_runtime import (
    _compute_aux_heating_sources_impl,
    _evolve_species_impl,
    _rho_volume_element_impl,
    ion_density_impl,
)
from scpn_control.core.transport_state import (
    PhysicsError as PhysicsError,
)
from scpn_control.core.transport_state import (
    ThermalEnergyBalance,
    capture_evolution_state_impl,
    restore_evolution_state_impl,
)
from scpn_control.core.transport_thermal_runtime import _sanitize_runtime_state_impl, evolve_profiles_impl


class TransportSolver(FusionKernel):
    """
    1.5D Integrated Transport Code.

    Solves Heat and Particle diffusion equations on flux surfaces,
    coupled self-consistently with the 2D Grad-Shafranov equilibrium.

    When ``multi_ion=True``, the solver evolves separate D/T fuel densities,
    He-ash transport with pumping (configurable ``tau_He``), independent
    electron temperature Te, coronal-equilibrium tungsten radiation
    (Pütterich et al. 2010), and per-cell Bremsstrahlung.
    """

    _gk_solver: GKSolverBase
    _tglf_native_solver: TGLFNativeSolver
    Pressure_2D: FloatArray

    def __init__(
        self,
        config_path: str | Path,
        *,
        nr: int = 50,
        multi_ion: bool = False,
        transport_model: str = "gyro_bohm",
        external_gk_allow_gyrobohm_fallback: bool = False,
        tglf_native_allow_gyrobohm_fallback: bool = False,
        allow_constant_transport_fallback: bool = False,
        allow_simplified_bootstrap_fallback: bool = False,
        allow_legacy_approximations: bool = False,
    ) -> None:
        """Load equilibrium configuration and allocate radial profiles, species and step diagnostics.

        Legacy approximation flags require the global opt-in. Species arrays
        exist only in multi-ion mode; their densities use the electron profile
        units of 10^19 m^-3. No equilibrium or transport convergence is implied
        by initialisation.
        """
        super().__init__(config_path)
        dims = self.cfg["dimensions"]
        self.a = max(float(dims["R_max"] - dims["R_min"]) / 2.0, 1.0e-9)
        if int(nr) != nr or nr < 2:
            raise ValueError("nr must be an integer >= 2")
        if (
            external_gk_allow_gyrobohm_fallback
            or tglf_native_allow_gyrobohm_fallback
            or allow_constant_transport_fallback
            or allow_simplified_bootstrap_fallback
        ) and not allow_legacy_approximations:
            raise ValueError("legacy approximation flags require allow_legacy_approximations=True")
        self.transport_model = transport_model
        self.allow_legacy_approximations = allow_legacy_approximations
        self.external_gk_allow_gyrobohm_fallback = external_gk_allow_gyrobohm_fallback
        self.tglf_native_allow_gyrobohm_fallback = tglf_native_allow_gyrobohm_fallback
        self.allow_constant_transport_fallback = allow_constant_transport_fallback
        self.allow_simplified_bootstrap_fallback = allow_simplified_bootstrap_fallback
        self.external_profile_mode = True  # Tell Kernel to respect our calculated profiles
        self.nr = int(nr)  # Radial grid points (normalised radius rho)
        self.rho = np.linspace(0, 1, self.nr)
        self.drho = 1.0 / (self.nr - 1)

        self.multi_ion: bool = multi_ion

        # PROFILES (Evolving state variables)
        # Te = Electron Temp (keV), Ti = Ion Temp (keV), ne = Density (10^19 m-3)
        self.Te = 1.0 * (1 - self.rho**2)  # Initial guess
        self.Ti = 1.0 * (1 - self.rho**2)
        self.ne = 5.0 * (1 - self.rho**2) ** 0.5

        # Transport Coefficients (Anomalous Transport Models). Explicit FloatArray
        # annotations keep the any-dimensional shape type, so the general-shape
        # arrays returned by sanitisation and numpy reductions remain assignable to
        # these one-dimensional np.ones/np.zeros-initialised attributes.
        self.chi_e: FloatArray = np.ones(self.nr)  # Electron diffusivity
        self.chi_i: FloatArray = np.ones(self.nr)  # Ion diffusivity
        self.D_n: FloatArray = np.ones(self.nr)  # Particle diffusivity

        # Impurity Profile (Tungsten density)
        self.n_impurity: FloatArray = np.zeros(self.nr)

        # Neoclassical transport configuration (None = constant chi_base=0.5)
        self.neoclassical_params: dict[str, Any] | None = None

        # Conservation diagnostics (updated each evolve_profiles call)
        self._last_conservation_error: float = 0.0
        self._last_energy_balance: ThermalEnergyBalance | None = None
        self._last_flux_balance: TransportFluxBalance | None = None
        self._last_particle_balance_error: float = 0.0
        self._last_particle_balance: SpeciesParticleBalance | None = None
        self._last_helium_pumped: FloatArray = np.zeros(self.nr)

        # ── Multi-ion species (P1.1) ──
        # Densities in 10^19 m^-3 (same units as ne)
        self.n_D: AnyFloatArray | None
        self.n_T: AnyFloatArray | None
        self.n_He: AnyFloatArray | None
        if self.multi_ion:
            # Species densities inherit ne's floating precision and are reassigned
            # by general-shape numpy reductions during evolution, so AnyFloatArray
            # is the precise storage type here.
            self.n_D = 0.5 * self.ne.copy()  # Deuterium
            self.n_T = 0.5 * self.ne.copy()  # Tritium
            self.n_He = np.zeros(self.nr)  # He-4 ash
        else:
            self.n_D = None
            self.n_T = None
            self.n_He = None

        # He-ash pumping time (default 5 * tau_E, ITER design baseline)
        self.tau_He_factor: float = 5.0  # tau_He/tau_E ratio; ITER design basis, Reiter et al.

        # Particle diffusivity for species transport
        self.D_species: float | AnyFloatArray = 0.3  # m²/s, anomalous; Angioni et al., NF 47 (2007) 1326

        # Z_eff tracking (updated every evolve step in multi-ion mode)
        self._Z_eff: float = 1.5

        # Auxiliary-heating source model parameters
        self.aux_heating_profile_width: float = 0.1
        self.aux_heating_electron_fraction: float = 0.5

        # Last-step auxiliary-heating power-balance telemetry
        self._last_aux_heating_balance: dict[str, float] = {
            "target_total_MW": 0.0,
            "target_ion_MW": 0.0,
            "target_electron_MW": 0.0,
            "reconstructed_ion_MW": 0.0,
            "reconstructed_electron_MW": 0.0,
            "reconstructed_total_MW": 0.0,
        }

        # Numerical hardening telemetry (non-finite replacements per step)
        self._last_numerical_recovery_count: int = 0

        # Momentum transport (rotation profile). Explicit FloatArray annotation
        # keeps the any-dimensional shape type so the momentum solver's
        # reconstructed rotation profile (general shape) remains assignable.
        self.omega_phi: FloatArray = np.zeros(self.nr)
        self._momentum_solver: MomentumTransportSolver | None = None

    def _ensure_valid_radial_grid(self) -> int:
        """Restore the normalised radial grid if external mutation broke it."""
        if is_canonical_radial_grid(self.rho, self.nr, self.drho):
            return 0

        canonical_rho, canonical_drho = canonical_radial_grid(self.nr)
        if getattr(self, "rho", None) is not None and np.shape(self.rho) == canonical_rho.shape:
            self.rho[...] = canonical_rho
        else:
            self.rho = canonical_rho
        self.drho = canonical_drho

        if self._momentum_solver is not None:
            self._momentum_solver.rho = np.asarray(self.rho, dtype=float)
            self._momentum_solver.drho = self.drho

        return 1

    def set_neoclassical(
        self,
        R0: float,
        a: float,
        B0: float,
        A_ion: float = 2.0,
        Z_eff: float = 1.5,
        q0: float = 1.0,
        q_edge: float = 3.0,
    ) -> None:
        """Configure Chang-Hinton neoclassical transport model.

        When set, update_transport_model uses the Chang-Hinton formula instead
        of the constant chi_base = 0.5.
        """
        return set_neoclassical_impl(self, R0, a, B0, A_ion, Z_eff, q0, q_edge)

    def chang_hinton_chi_profile(self) -> FloatArray:
        """Backward-compatible Chang-Hinton profile helper.

        Older parity tests call this no-arg method on a partially-initialized
        transport object. Keep the method as a thin adapter over the module
        function so those tests remain stable.
        """
        return chang_hinton_chi_profile_impl(self)

    def inject_impurities(self, flux_from_wall_per_sec: float, dt: float) -> None:
        """
        Models impurity influx from PWI erosion.

        Simple diffusion model: Source at edge, diffuses inward.
        """
        return inject_impurities_impl(self, flux_from_wall_per_sec, dt)

    def _legacy_bootstrap_current_approx(self, R0: float, B_pol: AnyFloatArray) -> FloatArray:
        """Legacy approximate bootstrap-current closure (compatibility mode only)."""
        return _legacy_bootstrap_current_approx_impl(self, R0, B_pol)

    def calculate_bootstrap_current(self, R0: float, B_pol: AnyFloatArray) -> FloatArray:
        """Calculate bootstrap current using full Sauter closure by default.

        If neoclassical configuration is missing, this method fails closed unless
        ``allow_simplified_bootstrap_fallback=True`` was explicitly enabled.
        """
        return calculate_bootstrap_current_impl(self, R0, B_pol)

    def _gyro_bohm_chi(self) -> FloatArray:
        """Gyro-Bohm anomalous transport diffusivity [m^2/s].

        chi_gB = c_gB * rho_s^2 * c_s / (a * q * R)

        where rho_s = sqrt(T_i m_i) / (e B), c_s = sqrt(T_e / m_i).

        The calibration coefficient c_gB is loaded from
        ``validation/reference_data/itpa/gyro_bohm_coefficients.json``
        if available (calibrated against the ITPA H-mode confinement
        database by ``tools/calibrate_gyro_bohm.py``).  Falls back to
        the value in ``neoclassical_params['c_gB']`` if explicitly set,
        or to the module-level default (0.1) otherwise.
        """
        return _gyro_bohm_chi_impl(self)

    def _external_gk_transport(self, p: dict[str, Any]) -> FloatArray:
        """Run an external GK solver at each flux surface, return chi_i profile.

        Also updates self.chi_e and self.D_n directly. By default, fails closed
        if the external solver is unavailable, throws, or returns unconverged
        output. Legacy gyro-Bohm fallback can be explicitly enabled by setting
        ``external_gk_allow_gyrobohm_fallback=True`` on construction.
        """
        return _external_gk_transport_impl(self, p)

    def _tglf_native_transport(self, p: dict[str, Any]) -> FloatArray:
        """Run the native TGLF-equivalent solver at each flux surface.

        Uses SAT1 spectral saturation with E×B shear quench.  Also
        updates self.chi_e and self.D_n. By default, fails closed when
        native-GK transport is unconverged. Legacy gyro-Bohm fallback can
        be explicitly enabled with ``tglf_native_allow_gyrobohm_fallback=True``.
        """
        return _tglf_native_transport_impl(self, p)

    def update_transport_model(self, P_aux: float) -> None:
        """
        Gyro-Bohm + neoclassical transport model with EPED-like pedestal.

        When neoclassical params are set, uses:
        - Chang-Hinton neoclassical chi as additive floor
        - Gyro-Bohm anomalous transport (calibrated c_gB)
        - EPED-like pedestal model for H-mode boundary condition

        Gyrokinetic, native TGLF and external GK modes retain their separately
        computed electron diffusivity and D_n profile. Their ion diffusivity
        receives the Chang-Hinton contribution; the legacy critical-gradient
        estimate is not added on top of those turbulent channels. Other modes
        retain the shared ion/electron estimate and D_n = 0.1*chi_e closure.
        For native/external modes, an explicitly permitted cellwise gyro-Bohm
        fallback defines chi_gB = max(raw_gyro_bohm, 0.01) per non-core
        cell, then follows the three-channel composition: chi_i = chi_gB
        + chi_nc, chi_e = chi_gB, D_n = 0.1*chi_gB, with no additional
        critical-gradient term. Core/vacuum cells retain the driver floors
        (0.01 in each returned channel before ion chi_nc is added). Both the
        mode-specific fallback flag and allow_legacy_approximations are required.
        External execution errors enter that fallback decision; native solver
        exceptions propagate, while invalid/unconverged native results can
        fall back. Selecting a mode alone does not prove backend success.

        D_n is an electron-particle coefficient, possibly a fallback estimate;
        the separate D_species scalar or radial profile controls D/T/He.
        This method does not infer a
        multispecies particle or heat-convection closure from D_n.

        Fails closed when neoclassical parameters are not configured, unless
        ``allow_constant_transport_fallback=True`` is explicitly enabled.
        """
        return update_transport_model_impl(self, P_aux)

    def _sanitize_runtime_state(self) -> int:
        """Keep runtime profiles and coefficients finite during transport stepping."""
        return _sanitize_runtime_state_impl(self)

    def _rho_volume_element(self) -> FloatArray:
        """Toroidal volume element per radial cell [m^3]."""
        return _rho_volume_element_impl(self)

    @property
    def ion_density(self) -> FloatArray:
        """Return thermal ion number density in units of 10^19 m^-3.

        Multi-ion mode counts D, T, He and tungsten nuclei once each, assuming
        they share Ti. Electron charge multiplicity does not multiply ion heat
        capacity. Single-ion mode retains the ni = ne closure. The returned
        array is independent of the mutable species state.

        Raises
        ------
        ValueError
            If a multi-ion species profile is absent.
        """
        return ion_density_impl(self)

    def _compute_aux_heating_sources(self, P_aux_MW: float) -> tuple[FloatArray, FloatArray]:
        """Return ion/electron auxiliary-heating sources in keV/s.

        The source is power-normalised against the radial cell volumes to ensure
        that reconstructed injected power matches ``P_aux_MW`` by construction.
        Both outputs use electron density as reference capacity; the thermal
        caller converts the ion rate to the actual ion capacity.
        """
        return _compute_aux_heating_sources_impl(self, P_aux_MW)

    # ── Multi-ion helpers (P1.1) ────────────────────────────────────

    def _evolve_species(self, dt: float) -> tuple[FloatArray, FloatArray]:
        """Evolve D, T, He-ash densities for one time-step (explicit diffusion + sources).

        Uses internal sub-stepping to respect the CFL stability limit of the
        explicit diffusion scheme: dt_CFL = 0.4 * (a * drho)^2 / max(D_species).
        D_species may be a scalar or a finite nonnegative profile matching rho;
        all three species share its arithmetic face coefficients.

        Returns (S_He_source, P_rad_line):
          S_He_source — step-average He-ash production rate [10^19 m^-3 / s]
          P_rad_line  — line radiation power density from tungsten [W/m^3]
        """
        return _evolve_species_impl(self, dt)

    # ── Main evolution (Crank-Nicolson) ──────────────────────────────

    @property
    def energy_balance_error(self) -> float:
        """Relative energy conservation error from the last evolution step."""
        return float(self._last_conservation_error)

    @property
    def last_energy_balance(self) -> ThermalEnergyBalance | None:
        """Return an immutable snapshot from the most recent thermal assessment.

        Every evolve_profiles call invalidates the previous record first. A
        zero-time call or failure before assessment leaves None. Assessment is
        recorded before the conservation exception, so rejected steps remain
        inspectable. A caller retaining an older snapshot keeps its original
        values; later solver changes cannot rewrite them.
        """
        return self._last_energy_balance

    @property
    def last_particle_balance(self) -> SpeciesParticleBalance | None:
        """Return the immutable record of this attempt's completed species stage.

        Every evolve_profiles call resets it, including invalid and zero-time
        calls. A later thermal rejection retains an already completed species
        record. Subsequent thermal sanitisation or external profile mutation
        cannot rewrite it; it is not a final thermal-state inventory certificate.
        Single-ion mode has no species-stage record.
        """
        return self._last_particle_balance

    @property
    def particle_balance_error(self) -> float:
        """Relative particle conservation error from the last evolution step.

        Only meaningful in multi-ion mode. Returns 0.0 otherwise.
        """
        return float(self._last_particle_balance_error)

    @property
    def last_flux_balance(self) -> TransportFluxBalance | None:
        """Return the last dedicated face-flux step record, reset on every flux attempt.

        Other evolution methods do not rewrite this historical stage snapshot.
        It is not a certificate for subsequent profile changes.
        """
        return self._last_flux_balance

    def evolve_fluxes(
        self, dt: float, flux: TransportFaceFlux, *, geometry: TransportFaceGeometry | None = None
    ) -> TransportFluxBalance:
        """Advance multi-ion state with explicit signed physical face transport.

        Parameters
        ----------
        dt : float
            Finite nonnegative interval [s].
        flux : TransportFaceFlux
            Electron,D,T,He particle and electron/total-ion energy flux at
            nr-1 physical minor-radius faces. Electron flux must be ambipolar.
            The caller supplies species mapping and the total energy moment;
            no automatic TGLF sampling or diffusion/pinch inference is made.
        geometry : TransportFaceGeometry or None
            Explicit cell volumes and face dV/dr for this stage. None selects
            the existing circular-torus weights. Other transport operators are
            not automatically converted to the supplied geometry.

        Returns
        -------
        TransportFluxBalance
            Measured particle and thermal inventories plus reservoir exchange.

        Raises
        ------
        ValueError
            Not in multi-ion mode, invalid input or a nonphysical candidate.
            Profiles are unchanged on rejection; last_flux_balance is None.

        Notes
        -----
        Defaults to the cylindrical volumes of the thermal/species solver. The
        first and last faces exchange with core/edge reservoirs; the edge node
        is held fixed and the zero-volume axis copies adjacent ion profiles.
        This first-order frozen-flux stage supplies no diffusion, reactions,
        radiation, pumping or exchange. Compose those stages explicitly and
        avoid counting particle-associated energy or turbulent transport twice.
        """
        return evolve_fluxes_impl(self, dt, flux, geometry=geometry)

    #: Instance attributes a single :meth:`evolve_profiles` call may mutate.
    #: Richardson trials roll back exactly this set, so a new evolved quantity
    #: must be added here or it will leak between trials.
    EVOLUTION_STATE_FIELDS: ClassVar[tuple[str, ...]] = (
        "Ti",
        "Te",
        "ne",
        "n_D",
        "n_T",
        "n_He",
        "n_impurity",
        "omega_phi",
        "J_phi",
        "_Z_eff",
        "_last_energy_balance",
        "_last_particle_balance",
        "_last_particle_balance_error",
        "_last_conservation_error",
        "_last_helium_pumped",
        "_last_numerical_recovery_count",
        "_last_flux_balance",
        "_last_aux_heating_balance",
    )

    #: Snapshot key of the rotation profile the momentum sub-solver owns. The
    #: sub-solver advances its own ``omega_phi`` and the step only rebinds the
    #: outer attribute to the array it returned, so restoring the outer one
    #: alone leaves the next step starting from the rotation of the discarded
    #: trial.
    MOMENTUM_ROTATION_KEY: ClassVar[str] = "_momentum_solver.omega_phi"

    def capture_evolution_state(self) -> dict[str, Any]:
        """Copy every quantity a transport step may mutate.

        Arrays are copied, so the snapshot is independent of later in-place
        writes. Attributes absent on this instance are omitted rather than
        defaulted, and :meth:`restore_evolution_state` then leaves them alone.
        The momentum sub-solver's rotation is captured under
        :data:`MOMENTUM_ROTATION_KEY` when that sub-solver exists.

        Returns
        -------
        dict[str, Any]
            Snapshot consumable only by :meth:`restore_evolution_state`.
        """
        return capture_evolution_state_impl(self)

    def restore_evolution_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by :meth:`capture_evolution_state`.

        Arrays are copied back so the caller may restore the same snapshot more
        than once, which a Richardson trial sequence does.

        Parameters
        ----------
        state:
            Snapshot from :meth:`capture_evolution_state` on this instance.

        Raises
        ------
        KeyError
            If the snapshot carries a name outside
            :data:`EVOLUTION_STATE_FIELDS` and :data:`MOMENTUM_ROTATION_KEY`,
            which would mean it came from a different contract and cannot be
            trusted to be complete.
        RuntimeError
            If the snapshot holds a sub-solver rotation and this instance has
            no momentum sub-solver to give it back to.
        """
        return restore_evolution_state_impl(self, state)

    def evolve_profiles(
        self,
        dt: float,
        P_aux: float,
        enforce_conservation: bool = False,
        ped_te: PedestalProfile | None = None,
        ped_ti: PedestalProfile | None = None,
    ) -> tuple[float, float]:
        """Advance both temperatures with implicit diffusion and split exchange.

        For each channel, the conductive stage advances density-weighted storage:

            (n1*T1 - n0*T0)/dt = 0.5*[G(n0*chi, T0) + G(n1*chi, T1)] + Q/H

        Here G(k,T) = (1/a**2)/rho * d/drho(rho*k*dT/drho), with the
        supported cylindrical quadrature; n is in 10**19 m**-3, T in keV,
        chi in m**2/s, Q in W/m**3 and H = 1.5e19*e_keV_J. Subscript 0
        denotes the captured incoming state; n1 is the completed species-stage
        density and T1 is the pre-exchange CN solution. Multi-ion n_i counts
        D, T, He and impurity nuclei once each; n_e follows ion charge.
        Single-ion mode retains n_i = n_e. Completed channel densities must
        be strictly positive; old zero density contributes zero initial heat.

        chi is held fixed across this call, while conductive face weights use
        their respective old/new densities. In temperature-rate form, the RHS
        storage is (n0/n1)*T0 and Q/(H*n1) is in keV/s. Auxiliary power is
        volume-normalised on the completed densities. Radiation uses incoming
        temperatures and completed species/charge state. Helium pumping uses
        actual integrated removed counts and incoming temperatures, yielding
        step-average ion/electron heat sinks. These frozen source evaluations
        and split exchange limit combined temporal accuracy to first order.

        The pumping model removes mean thermal ion energy and that of two
        accompanying electrons per He, assuming thermalised ash and local
        quasineutral removal. It is not a velocity-selective pump, sheath,
        flowing-plasma enthalpy or fast-alpha model. No separate particle heat
        convection, pressure-work or fusion-product energy closure is supplied.
        Discrete heat conservation does not validate these missing physics.

        Internal exchange follows the conductive solve; pedestal overrides and
        sanitisation follow exchange. Boundary energy uses the old/new face
        fluxes and edge capacities. Admission describes the final thermal state
        and retains numerical recovery or unmodeled pedestal energy in its
        residual. Large timesteps may require recovery or fail admission.

        Parameters
        ----------
        dt : float
            Time step [s].
        P_aux : float
            Auxiliary heating power [MW].
        enforce_conservation : bool
            When True, raise :class:`PhysicsError` if the per-step energy
            conservation error of the final, postprocessed thermal state exceeds 1%.
        ped_te : PedestalProfile, optional
            Pedestal profile model for electron temperature.
        ped_ti : PedestalProfile, optional
            Pedestal profile model for ion temperature.
        """
        return evolve_profiles_impl(self, dt, P_aux, enforce_conservation, ped_te, ped_ti)

    def map_profiles_to_2d(self) -> None:
        """Project the 1D radial profiles back onto the 2D Grad-Shafranov grid, including neoclassical bootstrap current."""
        return _map_profiles_to_2d(self)

    # ── Confinement time ───────────────────────────────────────────────

    def compute_confinement_time(self, P_loss_MW: float) -> float:
        """Compute the energy confinement time from stored energy.

        τ_E = W_stored / P_loss, where W_stored = ∫ 3/2 (ni Ti + ne Te) dV
        and the volume element is estimated from the 1D radial profiles
        using cylindrical approximation.

        Parameters
        ----------
        P_loss_MW : float
            Total loss power [MW].  Must be > 0.

        Returns
        -------
        float
            Energy confinement time [s].
        """
        return _compute_confinement_time(self, P_loss_MW)

    # ── GS ↔ transport self-consistency loop ─────────────────────────

    def run_self_consistent(
        self,
        P_aux: float,
        n_inner: int = 100,
        n_outer: int = 10,
        dt: float = 0.01,
        psi_tol: float = 1e-3,
    ) -> dict[str, Any]:
        """Run self-consistent GS <-> transport iteration.

        This implements the standard integrated-modelling loop used by
        codes such as ASTRA and JINTRAC: evolve the 1D transport for
        *n_inner* steps, project profiles onto the 2D grid, re-solve the
        Grad-Shafranov equilibrium, and repeat until the poloidal-flux
        change drops below *psi_tol*.

        Algorithm
        ---------
        1. Run transport for *n_inner* steps (evolve Ti/Te/ne).
        2. Call :meth:`map_profiles_to_2d` to update ``J_phi`` on the 2D grid.
        3. Re-solve the Grad-Shafranov equilibrium with the updated source.
        4. Check psi convergence:
           ``||Psi_new - Psi_old|| / ||Psi_old|| < psi_tol``.
        5. Repeat until converged or *n_outer* iterations exhausted.

        Parameters
        ----------
        P_aux : float
            Auxiliary heating power [MW].
        n_inner : int
            Number of transport evolution steps per outer iteration.
        n_outer : int
            Maximum number of outer (GS re-solve) iterations.
        dt : float
            Transport time step [s].
        psi_tol : float
            Relative psi convergence tolerance.

        Returns
        -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_outer_converged": int, "psi_residuals": list[float],
            "Ti_profile": ndarray, "ne_profile": ndarray,
            "converged": bool}``

        Raises
        ------
        ValueError
            If either iteration count is not positive.
        """
        return _run_self_consistent(self, P_aux, n_inner, n_outer, dt, psi_tol)

    # ── Fast one-shot transport path ──────────────────────────────────

    def run_to_steady_state(
        self,
        P_aux: float,
        n_steps: int = 500,
        dt: float = 0.01,
        adaptive: bool = False,
        tol: float = 1e-3,
        self_consistent: bool = False,
        sc_n_inner: int = 100,
        sc_n_outer: int = 10,
        sc_psi_tol: float = 1e-3,
    ) -> dict[str, Any]:
        """Run transport evolution until approximate steady state.

        Parameters
        ----------
        P_aux : float
            Auxiliary heating power [MW].
        n_steps : int
            Number of evolution steps.
        dt : float
            Time step [s] (initial value when adaptive=True).
        adaptive : bool
            Use Richardson-extrapolation adaptive time stepping.
        tol : float
            Error tolerance for adaptive stepping.
        self_consistent : bool
            When True, delegate to :meth:`run_self_consistent` which
            iterates GS <-> transport to convergence.  The remaining
            ``sc_*`` parameters are forwarded.
        sc_n_inner : int
            Transport steps per outer GS iteration (self-consistent mode).
        sc_n_outer : int
            Maximum outer GS iterations (self-consistent mode).
        sc_psi_tol : float
            Relative psi convergence tolerance (self-consistent mode).

        Returns
        -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_steps": int, "Ti_profile": ndarray,
            "ne_profile": ndarray}``
            When adaptive=True, also includes ``dt_final``,
            ``dt_history``, ``error_history``.
            When self_consistent=True, returns the
            :meth:`run_self_consistent` dict instead.

        Raises
        ------
        ValueError
            If the selected mode has a nonpositive iteration count.
        """
        return _run_to_steady_state(
            self, P_aux, n_steps, dt, adaptive, tol, self_consistent, sc_n_inner, sc_n_outer, sc_psi_tol
        )


# Backward-compatible public alias used by parity and bridge tests.
IntegratedTransportSolver = TransportSolver
