<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Physics Traceability and Bounded Claims -->


# Physics Traceability and Bounded Claims

This report is generated from `validation/physics_traceability.json`.
It blocks full-fidelity public claims for entries whose evidence status is still open or bounded.

## Summary

- Status: pass
- Registry entries: 46
- Open fidelity gaps: 46
- Full-fidelity public claims blocked: 46
- Resolved module paths: 46
- Resolved evidence paths: 186
- Source marker coverage: 41/41

## Module Traceability Table

| Module | Equation or contract | References | Unit contract | Validation evidence | Status |
|--------|----------------------|------------|---------------|---------------------|--------|
| `validation/reference_data/diiid` | Replay claims require measured signal arrays, physical units, shot identifiers, retrieval timestamp, licence policy, immutable checksums, and source URI provenance. | DIII-D MDSplus facility data access contract; Repository real-data manifest schema 1.0 | Signal-specific SI or dimensionless units; arbitrary units are rejected for real-shot evidence. | validation/validate_data_manifests.py; src/scpn_control/core/real_data_manifest.py; strict acquisition-readiness gate via --require-real-acquisition | synthetic_only |
| `src/scpn_control/control/burn_controller.py` | Burn-control claims must declare Bosch-Hale DT reactivity, alpha-energy partition, Lawson triple product, burn-fraction approximation, alpha-heating integration volume, reactivity-exponent stability boundary, and PI auxiliary-heating limits. | Bosch and Hale 1992 DT reactivity; Lawson 1957 ignition criterion; Mitarai and Muraoka 1999 delayed alpha-heating feedback | SI joules, m^-3, seconds, keV, MW, m^3/s, metres, dimensionless Q, and Lawson triple product units m^-3 s keV. | tests/test_burn_controller.py alpha-heating, Lawson, burn-fraction, and control checks; tests/test_cov_100_pct.py burn-controller edge-path checks | bounded_model |
| `src/scpn_control/core/elm_model.py` | ELM model claims must declare pedestal crash trigger, crash depth, recovery timescale, peeling-ballooning boundary, ELM cycle state, RMP suppression threshold, and energy-particle-loss accounting. | peeling-ballooning ELM trigger references; resonant magnetic perturbation ELM suppression references | Time in seconds, energy in joules, density in m^-3, temperature in eV or keV, pressure in pascals, magnetic perturbation amplitude dimensionless or tesla with explicit convention. | tests/test_elm_model.py; tests/test_integrated_scenario.py | validation_gap |
| `src/scpn_control/core/eped_pedestal.py` | EPED claims must declare the pedestal-width scaling, peeling-ballooning pressure limit, bootstrap-current coupling, H-mode entry assumptions, global beta limits, and compatibility boundary to the transport solver. | Snyder et al. EPED pedestal model; Troyon beta-limit reference; Sauter bootstrap-current fit | Pedestal width in normalised flux or metres with convention metadata; pressure in pascals, temperature in eV or keV, density in m^-3, current in amperes, and beta dimensionless. | tests/test_eped_pedestal.py; tests/test_cross_module_physics.py; tests/test_transport_hmode_edge.py | validation_gap |
| `src/scpn_control/core/fusion_kernel.py` | Fusion-kernel claims must declare the Grad-Shafranov residual, finite-difference grid, coil Green functions, source-profile parameterisation, boundary conditions, nonlinear iteration controls, convergence criteria, and Rust/Python parity boundary. | Grad-Shafranov equilibrium equation; Green-function tokamak coil-response references; repository Rust/Python fusion-kernel parity contract | SI metres, webers per radian, tesla, amperes, pascals, source derivatives, grid spacings, and dimensionless convergence tolerances. | tests/test_fusion_kernel.py; tests/test_geqdsk_regression.py; tests/test_rust_python_parity.py | validation_gap |
| `src/scpn_control/core/jax_gk_solver.py` | JAX linear gyrokinetic claims must preserve the native response-matrix eigenvalue formulation, expose identical physical input contracts, and document any numerical-precision or backend divergence. | Repository native linear GK response-matrix contract; JAX numerical backend reproducibility guidance | Same growth-rate, frequency, geometry, and species units as native linear GK; backend dtype, tolerance, and normalisation must be explicit in validation artefacts. | tests/test_jax_gk_solver.py; validation/validate_jax_gk_parity.py strict persisted backend parity artifact gate; source-level native response-matrix parity contract | validation_gap |
| `src/scpn_control/phase/kuramoto.py` | Phase-runtime claims must declare the Kuramoto-Sakaguchi mean-field coupling, order-parameter calculation, exogenous global-driver injection, phase wrapping convention, Euler step, and optional Rust fast-path parity boundary. | Kuramoto and Sakaguchi phase oscillator model; Repository phase synchronisation runtime contract | Phases in radians, angular frequencies in radians per second, timestep in seconds, coupling gains dimensionless or radians per second by declared convention, order parameter dimensionless. | tests/test_phase_kuramoto.py; tests/test_phase_properties.py; tests/test_phase_properties_extended.py | bounded_model |
| `src/scpn_control/core/marfe.py` | MARFE claims must declare impurity radiation loss, temperature scan, condensation criterion, Greenwald comparison, power-balance inputs, impurity fraction, and detection thresholds for detached high-radiation states. | Stangeby 2000 scrape-off-layer and divertor references; Greenwald 2002 density-limit reference; radiation-condensation MARFE onset references | Temperature in eV, density in m^-3, power in MW or W with explicit conversion, plasma current in MA or A with convention metadata, impurity fraction dimensionless. | tests/test_marfe.py; tests/test_cross_module_physics.py | validation_gap |
| `src/scpn_control/core/gk_geometry.py` | Geometry claims must declare Miller-shape coordinates, Jacobian, magnetic-field approximation, toroidal-field vacuum assumption, safety-factor relation, metric coefficients, and local-equilibrium validity bounds. | Miller et al. 1998 local equilibrium; Cyclone Base Case local Miller geometry benchmark | Major radius, minor radius, local radius, and gradient lengths in metres; angles in radians; magnetic-field strength in tesla; q, shear, elongation, triangularity, and shaping derivatives dimensionless. | tests/test_gk_geometry.py; validation/validate_gk_geometry_reference.py immutable Miller reference cases; source-level local Miller geometry contract | validation_gap |
| `src/scpn_control/core/ntm_dynamics.py` | NTM dynamics claims must declare rational-surface search, modified Rutherford equation terms, bootstrap drive, polarisation and curvature terms, ECCD control coupling, seed-island assumptions, and controller-validity limits. | modified Rutherford equation NTM references; ECCD NTM control references; repository rational-surface and island-dynamics contract | Island width in metres, time in seconds, current in amperes, q dimensionless, rho dimensionless, ECCD power in MW or W with conversion metadata. | tests/test_ntm_dynamics.py; tests/test_cross_module_physics.py | validation_gap |
| `src/scpn_control/control/rzip_model.py` | Rigid-plasma vertical stability claims must declare the linearised state vector [Z, dZ/dt, circuit currents], mutual-inductance derivative, vertical field index, effective-mass assumption, and wall or active-coil circuit model. | Lazarus et al. 1990 rigid plasma vertical stability model; Wesson 2011 tokamak vertical stability and field-index references; Repository vessel circuit model contract | SI metres, seconds, amperes, henries, ohms, tesla, and dimensionless vertical field index; growth time is reported in milliseconds. | tests/test_rzip_model.py vertical growth, field-index, physical-parameter, and controller-measurement checks; tests/test_cov_100_pct.py RZIP fallback and singular-circuit checks; validation/validate_rzip_reference.py strict RZIP reference artifact gate | bounded_model |
| `src/scpn_control/scpn/fpga_export.py` | FPGA-export claims must declare LIF fixed-point quantisation, leak right-shift approximation, threshold scaling, signed weight saturation, generated HDL boundary, target family assumptions, and synthesis-tool responsibility. | Repository SCPN compiler and fixed-point export contract; Leaky integrate-and-fire digital implementation contract | Fixed-point values use declared bit width and fractional-bit scale; clock in MHz, timestep in seconds, FIFO depth and neuron counts dimensionless, weights and thresholds quantised by explicit integer scale. | tests/test_fpga_export.py; tests/test_scpn_compiler.py | bounded_model |
| `src/scpn_control/core/vmec_lite.py` | VMEC-lite claims must declare flux-surface parameterisation, rotational-transform profile, Fourier mode truncation, pressure/current assumptions, residual metric, and explicit exclusion from full 3D MHD equilibrium claims. | VMEC 3D equilibrium references; stellarator flux-surface Fourier parameterisation references | SI metres, tesla, pascals, amperes, webers, dimensionless rotational transform, and Fourier coefficients with declared length units. | tests/test_vmec_lite.py; validation/validate_vmec_reference.py strict VMEC reference artifact gate | validation_gap |
| `src/scpn_control/control/advanced_soc_fusion_learning.py` | SOC turbulence-learning claims must declare the sandpile lattice, critical-gradient threshold, predator-prey zonal-flow coupling, shear-suppression term, bounded substep relaxation, Q-learning state discretisation, action set, and random policy assumptions. | Diamond and Hahm 1995 SOC turbulence reference; Kim and Diamond 2003 zonal-flow predator-prey coupling; Biglari, Diamond and Terry 1990 shear-suppression reference | Dimensionless lattice gradients, flow amplitudes, shear, toppling counts, Q-table values, reward, and RNG-seeded action choices. | tests/test_advanced_soc.py SOC physics and learning checks; tests/test_advanced_soc_verbose_plot.py verbose and plotting-path checks; tests/test_visualization_paths.py SOC visualisation checks; validation/validate_soc_reference.py strict reference-artifact gate | bounded_model |
| `src/scpn_control/core/blob_transport.py` | Blob transport claims must declare interchange drive, sheath closure, radial velocity scaling, filament size, density and temperature perturbation assumptions, and scrape-off-layer validity bounds. | Stangeby 2000 scrape-off-layer transport references; blob-filament interchange transport scaling references | SI metres, seconds, m/s, density in m^-3, temperature in eV, magnetic field in tesla, and dimensionless normalised perturbations. | tests/test_blob_transport.py | validation_gap |
| `src/scpn_control/core` | Analytical approximations must state their source equation, retained terms, omitted terms, unit contract, and allowed parameter domain. | Cordey 1981 orbit-width estimate; Sauter et al. 1999 neoclassical fits; Stangeby 2000 SOL two-point model | Per-module SI or declared normalised units with explicit conversion boundaries. | docs/physics_methods.md simplification declarations; source grep for simplification and approximation markers | bounded_model |
| `src/scpn_control/control` | Control approximations must declare reduced-state dynamics, actuator assumptions, estimator assumptions, and facility-exclusion boundaries before any controller-readiness claim. | Tokamak control reduced-plant literature; Repository control safety and disruption contracts | Per-controller declared SI plasma, actuator, magnetic, force, and timing units. | src/scpn_control/control/gym_tokamak_env.py; src/scpn_control/control/free_boundary_tracking.py | bounded_model |
| `src/scpn_control` | Runtime approximations must state phase wrapping, fixed-point shift approximations, replay fixture provenance, and hardware export limits. | Kuramoto phase oscillator reference model; Repository fixed-point spiking export contract | Declared phase radians, tick timing, fixed-point scale factors, and replay geometry units. | src/scpn_control/phase/kuramoto.py; src/scpn_control/scpn/fpga_export.py; src/scpn_control/scpn/geometry_neutral_replay.py | bounded_model |
| `src/scpn_control/core/checkpoint.py` | Checkpoint claims must declare serialised solver state, episode counter, metrics schema, finite-value policy, version boundary, and failure handling for corrupt or missing checkpoint artefacts. | repository restart and replay provenance contract | Units are inherited from stored state and metrics; checkpoint metadata must preserve schema version, episode index, and provenance boundaries. | tests/test_checkpoint.py; tests/test_coverage_deep.py | bounded_model |
| `src/scpn_control/control/density_controller.py` | Density-control claims must declare the Greenwald limit, ITER operating margin, radial particle transport grid, gas-puff source, pellet source, NBI source, cryopump sink, recycling source, controller gains, and CFL-limited explicit transport step. | Greenwald 2002 density limit; ITER Physics Basis 1999 density operating margin; Milora 1995 neutral gas shielding pellet-ablation model | SI particles per second, metres, seconds, m^-3, m^2/s, m/s, pellet millimetres, beam keV, megawatts, and dimensionless Greenwald fraction. | tests/test_density_controller.py Greenwald, pellet, sink, controller, geometry, transport-profile, source-input, and timestep guard checks; validation/validate_density_reference.py strict density reference artifact gate | bounded_model |
| `src/scpn_control/control/free_boundary_tracking.py` | Free-boundary tracking claims must declare direct kernel-in-the-loop coil-response identification, bounded least-squares correction, actuator lag, slew limits, supervisor rejection, measurement bias, drift, latency, and observer compensation assumptions. | Grad-Shafranov free-boundary control references; Repository FusionKernel free-boundary objective contract; Repository deterministic free-boundary acceptance campaign | SI coil currents, metres, webers per radian, seconds, amperes per second, objective-space residuals, and dimensionless supervisor gains. | tests/test_free_boundary_tracking.py controller and disturbance-observer checks; tests/test_free_boundary_tracking_coverage.py coverage and edge-path checks; tests/test_free_boundary_tracking_acceptance.py deterministic acceptance campaign | bounded_model |
| `src/scpn_control/control/disruption_contracts.py` | Disruption-contract claims must declare synthetic disruption signal generation, toroidal-mode amplitudes, mitigation-cocktail coupling, impurity transport response, halo/runaway post-disruption response, TBR equivalence scaling, and RL action bias assumptions. | Pautasso et al. 2017 disruption current-quench constraints; Riccardo et al. 2010 halo-current rise-time references; Abdou et al. 2015 blanket neutronics calibration references | SI seconds, milliseconds, mega-amperes, megajoules, moles, megawatts, dimensionless risk, toroidal mode amplitudes, and tritium breeding ratio. | tests/test_disruption_contracts.py contract smoke checks; tests/test_disruption_contracts_pure.py pure physics-path checks; tests/test_disruption_edge_cases.py edge-case disruption checks; validation/validate_disruption_reference.py strict disruption reference artifact gate | bounded_model |
| `src/scpn_control/core/disruption_sequence.py` | Disruption-sequence claims must declare phase ordering, thermal-quench and current-quench timing, mitigation action coupling, runaway-electron beam phase, stochastic event boundaries, and replay provenance. | ITER disruption mitigation sequence references; repository disruption phase-state contract | Time in seconds or milliseconds with explicit convention, current in amperes or mega-amperes, energy in joules or megajoules, and dimensionless phase labels. | tests/test_disruption_sequence.py; tests/test_disruption_safe_api.py | validation_gap |
| `src/scpn_control/core/gk_interface.py` | Generated input decks and parsed outputs must round-trip through real TGLF, GENE, GS2, CGYRO, or QuaLiKiz executables. | TGLF Staebler et al. 2007; GENE Jenko et al. 2000; GS2 Kotschenreuther et al. 1995; CGYRO Candy et al. 2016; QuaLiKiz Bourdelle et al. 2007 | Code-specific flux, growth-rate, frequency, and geometry units converted into repository normalisation with explicit metadata. | docs/joss_paper.md external GK limitation; validation/validate_gk_interface_artifacts.py strict external interface artifact gate | external_dependency_blocked |
| `src/scpn_control/core/uncertainty.py` | Uncertainty claims must declare sampled variables, distributions, correlations, random seed, propagation chain, convergence criteria, sensitivity outputs, and finite-value rejection policy. | Monte Carlo uncertainty propagation references; repository fusion-performance uncertainty contract | Units inherit each propagated physical quantity; distribution parameters must preserve SI or declared normalised units and dimensionless uncertainty fractions. | tests/test_uncertainty.py; tests/test_full_chain_uq.py; tests/test_uncertainty_sigma_guard.py; validation/validate_uncertainty_reference.py strict UQ reference artifact gate | validation_gap |
| `src/scpn_control/scpn/geometry_neutral_replay.py` | Geometry-neutral replay claims must declare synthetic W7-X-like fixture provenance, field-line spread metric, actuator current bounds, latency model, stuck-actuator fault schedule, controller feature mapping, and replay acceptance thresholds. | Repository geometry-neutral control contract; W7-X-like reduced-order stellarator replay fixture | Field-line spread in radians, currents in amperes, timestep in seconds, latency in microseconds, effective ripple dimensionless, controller objectives and thresholds declared per replay manifest. | tests/test_geometry_neutral_replay.py; tests/test_geometry_neutral_contracts.py | bounded_model |
| `src/scpn_control/core/gk_ood_detector.py` | OOD detector claims must declare the feature vector, training distribution, distance metric, threshold calibration, uncertainty handling, and behavior outside the calibrated gyrokinetic operating envelope. | Repository gyrokinetic scheduler OOD contract; statistical process monitoring distribution-shift controls | Feature units inherit declared GK inputs and outputs; detector scores are dimensionless with explicit calibration metadata and threshold provenance. | tests/test_gk_ood_detector.py; tests/test_gk_hybrid_integration.py; validation/validate_gk_ood_calibration.py strict persisted campaign calibration gate | validation_gap |
| `src/scpn_control/core/gk_online_learner.py` | Online learner claims must declare training-window selection, validation loss, rollback policy, uncertainty or OOD gating, optimiser settings, and compatibility boundary with gyrokinetic scheduler inputs. | online learning stability and rollback control references; repository gyrokinetic hybrid learner contract | Inputs inherit GK feature units; losses and OOD scores dimensionless with explicit scaling and threshold metadata. | tests/test_gk_online_learner.py; tests/test_gk_hybrid_integration.py | validation_gap |
| `src/scpn_control/core/gk_species.py` | Species and collision claims must declare charge, mass, density, temperature, thermal speed, Larmor radius, gyroaverage Bessel approximation, diamagnetic frequency, pitch-angle scattering model, energy-diffusion simplification, and valid species parameter bounds. | Sugama et al. 2006 collision operator; gyrokinetic normalisation and species parameter contracts | Mass in kilograms, charge in coulombs, density in m^-3, temperature in electronvolts, velocity in m/s, gyrofrequency in rad/s, Larmor radius in metres, and collision rates in s^-1. | tests/test_gk_species.py; tests/test_gk_electromagnetic.py; validation/validate_gk_species_reference.py immutable species and collision reference cases | validation_gap |
| `src/scpn_control/control/halo_re_physics.py` | Post-disruption claims must declare the halo L/R circuit, contact fraction, toroidal peaking factor, current-quench waveform, Connor-Hastie primary generation, Rosenbluth-Putvinski avalanche generation, and ensemble uncertainty assumptions. | Fitzpatrick 2002 halo current and error-field interaction; Connor and Hastie 1975 runaway electron generation; Rosenbluth and Putvinski 1997 avalanche generation | SI amperes, mega-amperes, seconds, milliseconds, ohms, henries, metres, volts per metre, rates per second, and meganewtons per metre. | tests/test_halo_re_physics.py halo-current and runaway-electron regression checks; tests/test_halo_nonfinite_guards.py non-finite guard checks; tests/test_halo_validation_paths.py validation-path guard checks | bounded_model |
| `src/scpn_control/core/stability_mhd.py` | MHD stability claims must declare q-profile interpolation, magnetic shear, Mercier and Troyon criteria, beta-limit inputs, first-unstable-radius search, and the exclusion boundary for full ideal-MHD or resistive-MHD eigenmode claims. | Troyon beta-limit reference; Mercier ideal-MHD stability criterion; tokamak q-profile and magnetic-shear references | q and beta dimensionless, plasma current in MA or A with explicit convention, minor radius and major radius in metres, magnetic field in tesla, rho dimensionless. | tests/test_stability_mhd.py; tests/test_ballooning_solver.py; tests/test_cross_module_physics.py | validation_gap |
| `src/scpn_control/core/integrated_scenario.py` | Integrated scenario claims must declare coupling order, current diffusion, transport, pedestal, ELM, NTM, sawtooth, burn, and control interactions, timestep policy, state exchange units, and failure isolation boundaries. | integrated tokamak scenario modelling references; repository scenario-coupling contract | All coupled states must preserve declared SI or normalised units for current, density, temperature, pressure, q, beta, power, flux, and timing. | tests/test_integrated_scenario.py; tests/test_cross_module_physics.py | validation_gap |
| `src/scpn_control/core/gk_eigenvalue.py` | Native linear GK eigenvalue solve must match external TGLF, GENE, GS2, CGYRO, or QuaLiKiz growth rates and real frequencies for the same Miller geometry and species inputs. | Miller et al. 1998 local equilibrium; GENE and GACODE published input-output contracts | Growth rate and frequency normalised consistently to c_s/a or declared external-code convention. | ROADMAP.md local-dispersion overprediction note; docs/competitive_analysis.md linear GK quantitative accuracy status; validation/validate_gk_crosscode.py strict real-binary evidence gate | external_dependency_blocked |
| `src/scpn_control/core/momentum_transport.py` | Momentum transport claims must declare NBI torque, collisional damping, viscous momentum diffusion, rotation-profile boundary conditions, angular-momentum units, and the coupling boundary to the integrated transport solver. | tokamak toroidal momentum transport references; repository NBI torque and rotation-profile contract | SI newton metres, kg m^2/s^2, rad/s, metres, seconds, density in m^-3, and momentum diffusivity in m^2/s. | tests/test_momentum_transport.py; tests/test_momentum_integration.py | validation_gap |
| `src/scpn_control/core/neural_equilibrium.py` | Neural equilibrium predictions must be compared against identical P-EFIT or equivalent reference equilibria for psi, pressure, q, and boundary geometry. | EFIT/P-EFIT equilibrium reconstruction workflow; Repository neural equilibrium model contract | SI magnetic flux, pressure, metre-scale geometry, and dimensionless q-profile arrays on declared grids. | ROADMAP.md neural eq cross-validation future item; docs/joss_paper.md neural equilibrium limitation; validation/validate_neural_equilibrium_reference.py strict P-EFIT/reference artifact gate | external_dependency_blocked |
| `src/scpn_control/core/neural_transport.py` | Neural transport claims must declare input feature normalisation, QLKNN weight provenance, prediction targets, uncertainty output, out-of-domain handling, and cross-validation against reference transport cases. | QuaLiKiz neural network transport surrogate references; repository QLKNN weight and metric contract | Inputs and outputs use declared transport feature units, diffusivity in m^2/s or declared normalisation, fluxes in SI or gyro-Bohm units with conversion metadata. | tests/test_neural_transport.py; tests/test_neural_transport_core.py; tests/test_qlknn_transport.py; validation/validate_neural_transport_reference.py strict QuaLiKiz/reference artifact gate | external_dependency_blocked |
| `src/scpn_control/core/neural_turbulence.py` | Neural turbulence claims must declare QLKNNSurrogate inputs, feature scaling, turbulence target variables, fallback behaviour, uncertainty handling, and cross-validation boundary against gyrokinetic or quasilinear turbulence references. | QLKNN turbulence surrogate references; repository neural turbulence surrogate contract | Feature and target units follow the declared turbulence surrogate schema; diffusivities, growth rates, and fluxes require explicit SI or normalised-unit metadata. | tests/test_neural_turbulence.py; tests/test_cov_100_pct.py; validation/validate_neural_turbulence_reference.py strict GK-campaign/reference artifact gate | validation_gap |
| `src/scpn_control/core/gk_nonlinear.py` | Five-dimensional delta-f flux-tube Vlasov evolution with E cross B bracket, ballooning connection, kinetic electrons, and Sugama collision terms. | Dimits et al. 2000 Cyclone Base Case; Sugama et al. 2006 collision operator | Gyro-Bohm-normalised heat flux and chi_i with explicit R/L_Ti, rho_s, c_s, and reference gradient normalisation. | ROADMAP.md v0.18.0 open revalidation item; docs/joss_paper.md nonlinear GK validation limitation; validation/gk_nonlinear_cyclone.py saturation-evidence assessor | validation_gap |
| `src/scpn_control/core/orbit_following.py` | Orbit-following claims must declare guiding-centre equations, magnetic moment handling, banana-orbit width scaling, collision or loss assumptions, ensemble sampling, and valid magnetic-geometry bounds. | Cordey 1981 orbit-width estimate; tokamak guiding-centre orbit references | SI metres, seconds, tesla, electronvolts, kg, coulombs, pitch dimensionless, and ensemble probabilities dimensionless. | tests/test_orbit_following.py; tests/test_cov_100_pct.py; validation/validate_orbit_reference.py strict orbit reference artifact gate | validation_gap |
| `src/scpn_control/control/realtime_efit.py` | Real-time equilibrium reconstruction claims must declare the fixed-boundary Grad-Shafranov solve, polynomial p-prime and FF-prime source profiles, diagnostic response interpolation, Rogowski current reconstruction, and macroscopic-shape extraction assumptions. | Lao et al. 1985 EFIT equilibrium reconstruction; Strait et al. 2019 real-time EFIT workflow references; Repository GEQDSK and magnetic-diagnostic validation contracts | SI metres, webers per radian, tesla, amperes, pascals per weber, FF-prime units, and dimensionless q95, beta_pol, li, elongation, and triangularity. | tests/test_realtime_efit.py fixed-boundary solve and diagnostic reconstruction checks; tests/test_kinetic_efit.py kinetic EFIT integration checks; tests/test_validate_real_shots_equilibrium.py GEQDSK source residual checks | bounded_model |
| `src/scpn_control/core/gyrokinetic_transport.py` | Reduced gyrokinetic transport claims must declare the dispersion relation, instability branch classification, quasilinear heat-flux closure, critical-gradient threshold, saturation rule, and mapping from local GK outputs to transport coefficients. | Cyclone Base Case gyrokinetic benchmark; quasilinear gyrokinetic transport closure references | Growth rates and frequencies in declared normalised units, gradients dimensionless or m^-1 with explicit convention, heat flux in gyro-Bohm or SI units with conversion metadata. | tests/test_gyrokinetic_transport.py; tests/test_gk_benchmark_linear.py | validation_gap |
| `src/scpn_control/control/gym_tokamak_env.py` | 0D/1D control plants must declare reduced-state equations, conservation assumptions, and controller-validity bounds before being used for hardware or facility claims. | Wesson 2011 tokamak transport references; Repository Gymnasium plant and digital-twin contracts | Declared plasma current, temperature, density, confinement, and actuator units per state vector field. | README.md validation limitation; docs/use_cases.md synthetic disruption prediction status; tests/test_gym_tokamak_env.py reduced-order plant bounds and action-contract checks | bounded_model |
| `src/scpn_control/core/tearing_mode_coupling.py` | Sawtooth-to-NTM seeding claims must declare sawtooth crash trigger, seed-island generation, rational-surface coupling, phase and amplitude assumptions, NTM coupling path, and exclusion of full nonlinear MHD crash dynamics. | sawtooth crash and NTM seeding references; repository tearing-mode coupling contract | Time in seconds, island width in metres, q dimensionless, rho dimensionless, phase in radians, and amplitude units declared per coupling signal. | tests/test_tearing_mode_coupling.py; tests/test_integrated_scenario.py | validation_gap |
| `src/scpn_control/control/tokamak_digital_twin.py` | Digital-twin claims must declare the 2D poloidal grid, q-profile topology update, rational-surface island mask, finite-difference diffusion, optional gyrokinetic surrogate correction, radiation-loss scaling, actuator latency, RNG seed, and IDS export assumptions. | Repository digital-twin runtime contract; Wesson 2011 radiation-loss reference; Repository IMAS IDS export helper contract | Declared normalised grid indices, keV, m^-3, dimensionless q-profile values, actuator-lag summary units, RNG seed metadata, and IDS-compatible pulse history fields. | tests/test_tokamak_digital_twin.py deterministic twin checks; tests/test_digital_twin_physics.py physics-path checks; tests/test_digital_twin_ingest_runtime.py ingest runtime checks; validation/validate_digital_twin_reference.py strict reference-artifact gate | bounded_model |
| `src/scpn_control/core/integrated_transport_solver.py` | Integrated transport claims must declare the radial grid, heat and particle diffusion equations, neoclassical closure, bootstrap-current fit, source deposition terms, boundary conditions, timestep limits, and coupling contracts to pedestal and gyrokinetic closures. | Chang-Hinton neoclassical transport model; Sauter et al. 1999 bootstrap-current fits; repository integrated transport solver contract | SI metres, seconds, m^-3, eV, W/m^3, m^2/s, amperes per square metre, and dimensionless profile coordinates with explicit normalisation metadata. | tests/test_integrated_transport_solver.py; tests/test_transport_energy_conservation.py; tests/test_transport_neoclassical_guards.py | validation_gap |
| `src/scpn_control/control/volt_second_manager.py` | Volt-second management claims must declare inductive flux, resistive loop-voltage integration, Ejima startup flux, bootstrap-current correction, flat-top duration estimate, ramp and ramp-down decomposition, and flux-budget margin assumptions. | Wesson 2011 tokamak loop-voltage and flux-balance equations; Ejima et al. 1982 startup flux coefficient; ITER Physics Basis 1999 flat-top flux-budget references | SI webers, volt-seconds, henries, ohms, amperes, mega-amperes, seconds, metres, and dimensionless Ejima coefficient. | tests/test_volt_second_manager.py flux budget and scenario analysis checks | bounded_model |

## Components

### DIII-D experimental replay

- Fidelity status: `synthetic_only`
- Module path: `validation/reference_data/diiid`
- Full-fidelity public claim: blocked
- Covered source paths: 0
- Required actions:
  - Acquire facility-approved MDSplus or equivalent shot artefacts through a checked acquisition spec
  - Attach per-artefact checksums and validate the replay manifest with local checksum verification

### DT burn control and alpha-heating model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/burn_controller.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate burn operating points against an integrated transport solver or published burn-control benchmark before reactor-control claims
  - Replace the burn-fraction approximation with a calibrated slowing-down and ash-removal model before facility extrapolation

### ELM crash and RMP suppression approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/elm_model.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate ELM frequency, crash depth, and RMP suppression windows against measured or published H-mode cases
  - Persist pedestal pre-crash and post-crash profiles with every ELM validation artefact

### EPED pedestal and peeling-ballooning approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/eped_pedestal.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate pedestal height and width against published EPED benchmark points or measured pedestal databases
  - Record bootstrap-current, beta-limit, and shaping inputs for every pedestal validation artefact

### Grad-Shafranov fusion-kernel numerical contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/fusion_kernel.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate reconstructed equilibria against EFIT, GEQDSK, or published fixed-boundary benchmark cases
  - Preserve residual norms, convergence metadata, and Rust/Python parity evidence for every validation artefact

### JAX gyrokinetic numerical parity guard

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/jax_gk_solver.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Run backend parity over CBC, TEM, electromagnetic, and stable-mode cases with pinned tolerances
  - Persist CPU/GPU backend metadata with each parity artefact before claiming accelerator-equivalent physics

### Kuramoto-Sakaguchi phase synchronisation runtime

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/phase/kuramoto.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate synchronisation and stability metrics against published Kuramoto-Sakaguchi benchmark cases before broader phase-control claims
  - Persist Rust/Python parity evidence and timestep convergence checks for deployment-target oscillator counts

### MARFE radiation-condensation density-limit contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/marfe.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate onset temperatures and density limits against measured or published MARFE cases
  - Add impurity-specific radiation tables or documented provenance for each supported impurity species

### Miller geometry and field approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_geometry.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Compare metric coefficients and field factors against an independent Miller-geometry reference implementation
  - Store immutable reference cases covering circular, shaped, and high-shear local equilibria

### NTM island evolution and control approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/ntm_dynamics.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate island growth and suppression against measured or published NTM benchmark cases
  - Persist q-profile, rational-surface, seed-island, and ECCD alignment metadata with each validation case

### RZIP rigid vertical stability model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/rzip_model.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Replace the effective-mass heuristic with a documented plasma inertia or facility-calibrated parameter source before facility claims
  - Validate vertical growth rates against a reference RZIP, CREATE-L/NL, TSC, or measured vertical-displacement benchmark

### SCPN FPGA fixed-point export boundary

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/scpn/fpga_export.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Run generated HDL through real Vivado, Quartus, or Yosys synthesis before hardware readiness claims
  - Persist resource utilisation, timing closure, and bit-accurate simulator evidence for every supported target family

### VMEC-lite stellarator equilibrium approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/vmec_lite.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate surface geometry and rotational-transform outputs against VMEC or published stellarator benchmark equilibria
  - Persist Fourier truncation, pressure profile, current profile, and residual metadata for each validation case

### advanced SOC turbulence learning controller

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/advanced_soc_fusion_learning.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Replace sandpile turbulence proxy with gyrokinetic or measured turbulence evidence before transport-control claims
  - Validate Q-learning policy behaviour against replayed plasma-control objectives before control-readiness claims

### blob transport and scrape-off-layer approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/blob_transport.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate blob velocity and spreading against measured or published SOL filament cases
  - Persist filament size, collisionality, sheath, and magnetic-geometry metadata with each validation artefact

### bounded analytical approximations in physics modules

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core`
- Full-fidelity public claim: blocked
- Covered source paths: 0
- Required actions:
  - Keep this parent entry as a non-source guardrail for future core approximation markers
  - Require every new core approximation marker to land with a source-specific traceability entry and validation evidence

### bounded control plant approximations

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control`
- Full-fidelity public claim: blocked
- Covered source paths: 0
- Required actions:
  - Keep this parent entry as a non-source guardrail for future reduced-order control plant markers
  - Require every new control plant marker to land with a controller-specific traceability entry and replay or higher-fidelity evidence

### bounded phase and spiking runtime approximations

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control`
- Full-fidelity public claim: blocked
- Covered source paths: 0
- Required actions:
  - Keep this parent entry as a non-source guardrail for future phase, spiking, and replay runtime markers
  - Require every new runtime marker to land with a source-specific traceability entry plus hardware-target or replay-fixture evidence

### checkpoint state serialisation boundary contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/checkpoint.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Version checkpoint schemas before long-running production campaigns depend on replay compatibility
  - Add migration fixtures for every future checkpoint schema change

### density control and particle-source model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/density_controller.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Replace circular-geometry volume elements with equilibrium-derived flux-surface volumes before facility-density claims
  - Validate pellet deposition and recycling source profiles against measured or published fuelling cases

### direct free-boundary tracking controller

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/free_boundary_tracking.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Promote deterministic acceptance evidence to measured or validated free-boundary replay cases before facility-control claims
  - Attach actuator, latency, and sensor-calibration evidence for the target device before deployment claims

### disruption mitigation contract layer

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/disruption_contracts.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Replace synthetic disruption signals with measured labelled disruption windows before predictive claims
  - Validate mitigation-cocktail, halo, runaway, and TBR couplings against experimental or benchmark artefacts

### disruption sequence phase-ordering contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/disruption_sequence.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate phase timing and mitigation branches against labelled measured disruption windows
  - Persist shot identifiers, phase labels, timing tolerances, and mitigation metadata for sequence validation

### external gyrokinetic interfaces

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/gk_interface.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Run each interface against a real executable or documented public reference output
  - Promote parser fixtures from mock subprocesses to immutable external-code artefacts

### full-chain uncertainty quantification contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/uncertainty.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Calibrate uncertainty priors against measured or published scenario ensembles
  - Persist random seeds, distribution parameters, sample counts, convergence diagnostics, and sensitivity metrics with every UQ artefact

### geometry-neutral stellarator replay fixture

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/scpn/geometry_neutral_replay.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Replace synthetic W7-X-like replay inputs with measured or benchmark stellarator field-line artefacts before device-control claims
  - Persist replay manifests with magnetic-configuration provenance, actuator calibration, latency evidence, and acceptance thresholds

### gyrokinetic OOD detector distribution-bound contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_ood_detector.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Calibrate thresholds against real or published GK campaign ensembles
  - Add false-positive and false-negative acceptance criteria for deployment gating

### gyrokinetic online learner stability contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_online_learner.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate learner updates on immutable gyrokinetic campaign artefacts with held-out distribution-shift cases
  - Persist model weights, rollback decisions, and validation metrics for every online update

### gyrokinetic species and collision approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_species.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate species normalisation and collision coefficients against a published gyrokinetic reference case
  - Add multispecies edge-case fixtures for electron, main-ion, impurity, and extreme-temperature domains

### halo current and runaway electron disruption model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/halo_re_physics.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate halo current and runaway-current envelopes against measured disruption databases before mitigation claims
  - Attach wall-contact geometry and impurity-radiation evidence before extrapolating beyond the declared ensemble domain

### ideal-MHD stability metric approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/stability_mhd.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Compare stability metrics against an independent MHD stability code or published benchmark profiles
  - Store profile grids, interpolation choices, and unstable-region metadata with each stability validation artefact

### integrated scenario coupling contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/integrated_scenario.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate coupled scenario trajectories against measured discharges or published integrated-modelling benchmarks
  - Persist module-by-module state exchange, timestep, and convergence metadata with every scenario artefact

### linear gyrokinetic cross-code agreement

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/gk_eigenvalue.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Install or provide real external GK binaries
  - Run native-vs-external benchmark cases and store parser artefact evidence

### momentum transport and torque-balance approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/momentum_transport.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate torque deposition and rotation evolution against measured or published NBI momentum cases
  - Add acceptance artefacts for low-torque, high-torque, and sign-changing rotation profiles

### neural equilibrium cross-validation

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_equilibrium.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Acquire matched P-EFIT reference equilibria or an openly redistributable equivalent
  - Add error bounds for psi, pressure, q-profile, LCFS, and magnetic-axis predictions

### neural transport surrogate validation contract

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_transport.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Acquire or generate immutable reference QLKNN or transport benchmark cases for cross-validation
  - Record weight checksums, feature scaling, uncertainty metadata, and OOD decisions with every surrogate evaluation

### neural turbulence surrogate validation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/neural_turbulence.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate surrogate outputs against immutable gyrokinetic or QuaLiKiz turbulence benchmark cases
  - Persist feature scaling, model checksum, fallback mode, and error metrics for each validation case

### nonlinear gyrokinetic heat-flux saturation

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_nonlinear.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Run a long enough nonlinear CBC campaign to saturated chi_i
  - Compare saturated heat flux against published or real cross-code reference data

### orbit-following guiding-centre approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/orbit_following.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate orbit widths and loss fractions against an independent orbit-following code or published benchmark cases
  - Persist particle species, pitch, energy, geometry, and ensemble seed metadata with each validation artefact

### real-time EFIT-lite equilibrium reconstruction

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/realtime_efit.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate reconstructed psi, Ip, q95, beta_pol, and li against matched EFIT or P-EFIT equilibria before facility claims
  - Replace synthetic diagnostic-response evidence with measured flux-loop, B-probe, and Rogowski artefacts when facility data are available

### reduced gyrokinetic transport closure contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gyrokinetic_transport.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Compare predicted growth rates and transport coefficients with external GK or published quasilinear benchmark cases
  - Store branch classification and saturation metadata for every transport-closure validation case

### reduced-order plant and control environments

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/gym_tokamak_env.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Separate reduced-order training claims from facility-control claims in generated reports
  - Add plant-model validity bounds and cross-checks against higher-fidelity replay cases

### sawtooth-to-NTM seeding approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/tearing_mode_coupling.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate seed-island amplitudes and NTM triggering windows against measured or published sawtooth-triggered NTM cases
  - Persist pre-crash q-profile, crash amplitude, rational-surface, and coupling metadata with each validation case

### tokamak digital twin topology and diffusion model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/tokamak_digital_twin.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Replace normalised 2D diffusion dynamics with equilibrium and transport-calibrated state evolution before facility twin claims
  - Validate IDS export and actuator-lag summaries against replayed or measured discharge histories

### transport solver neoclassical and source-term approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/integrated_transport_solver.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate radial transport profiles against a measured discharge or published integrated-modelling benchmark
  - Persist source deposition, boundary condition, and convergence metadata with every transport validation artefact

### volt-second budget and flux-consumption manager

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/volt_second_manager.py`
- Full-fidelity public claim: blocked
- Covered source paths: 1
- Required actions:
  - Validate flux consumption against real loop-voltage and current traces before scenario-duration claims
  - Replace bootstrap-current proxy with neoclassical or transport-solver evidence before facility extrapolation
