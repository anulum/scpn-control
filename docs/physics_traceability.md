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
- Registry entries: 51
- Open fidelity gaps: 50
- Full-fidelity public claims blocked: 50
- Resolved module paths: 51
- Resolved evidence paths: 208
- External validation trackers: 8
- Source marker coverage: 33/33

## External Validation Collaboration Trackers

- External validation artefacts needed for full-fidelity SCPN-CONTROL claims: [#46](https://github.com/anulum/scpn-control/issues/46) — 0 open claim(s) — Parent tracker for external code, reference data, facility replay, benchmark, and hardware evidence requests.
- External gyrokinetic validation artefacts: [#47](https://github.com/anulum/scpn-control/issues/47) — 9 open claim(s) — TGLF, GENE, GS2, CGYRO, QuaLiKiz, nonlinear CBC, Miller geometry, species, JAX parity, OOD, and online-learning artefacts.
- Equilibrium and reconstruction reference artefacts: [#48](https://github.com/anulum/scpn-control/issues/48) — 5 open claim(s) — DIII-D or equivalent shots, EFIT, P-EFIT, GEQDSK, VMEC, and stellarator replay artefacts.
- Transport, edge, MHD, and scenario benchmark artefacts: [#49](https://github.com/anulum/scpn-control/issues/49) — 15 open claim(s) — Integrated transport, momentum, pedestal, ELM, MARFE, NTM, current drive, stability, tearing, SOL, orbit, UQ, and scenario benchmarks.
- Neural surrogate validation artefacts: [#50](https://github.com/anulum/scpn-control/issues/50) — 3 open claim(s) — QLKNN, QuaLiKiz, gyrokinetic, transport, turbulence, and equilibrium surrogate reference datasets and weight provenance.
- Plasma-control and facility replay artefacts: [#51](https://github.com/anulum/scpn-control/issues/51) — 11 open claim(s) — RZIP, RWM, free-boundary, density, digital twin, SOC learning, burn, volt-second, mu-synthesis, and reduced-plant replay evidence.
- Disruption, halo-current, and mitigation benchmark artefacts: [#52](https://github.com/anulum/scpn-control/issues/52) — 3 open claim(s) — Measured disruption databases, labelled disruption windows, halo/runaway envelopes, wall contact, impurity radiation, and mitigation metadata.
- Hardware, HDL, CODAC/EPICS, and runtime deployment evidence: [#53](https://github.com/anulum/scpn-control/issues/53) — 4 open claim(s) — Vivado, Quartus, Yosys, timing closure, simulator evidence, CODAC/EPICS timing, interlocks, backpressure, HIL replay, and runtime parity.

## Module Traceability Table

| Module | Equation or contract | References | Unit contract | Validation evidence | Status | Tracker |
|--------|----------------------|------------|---------------|---------------------|--------|---------|
| `validation/reference_data/diiid` | Replay claims require measured signal arrays, physical units, shot identifiers, retrieval timestamp, licence policy, immutable checksums, and source URI provenance. | DIII-D MDSplus facility data access contract; Repository real-data manifest schema 1.0 | Signal-specific SI or dimensionless units; arbitrary units are rejected for real-shot evidence. | validation/validate_data_manifests.py; src/scpn_control/core/real_data_manifest.py; tests/test_real_diiid_shots.py; tests/test_validate_data_manifests.py; strict acquisition-readiness gate via --require-real-acquisition | reference_validated |  |
| `src/scpn_control/control/burn_controller.py` | Burn-control claims must declare Bosch-Hale DT reactivity, alpha-energy partition, Lawson triple product, burn-fraction approximation, alpha-heating integration volume, tokamak major/minor-radius ordering, strictly ordered normalised alpha-heating profile grids, reactivity-exponent stability boundary, and PI auxiliary-heating limits. | Bosch and Hale 1992 DT reactivity; Lawson 1957 ignition criterion; Mitarai and Muraoka 1999 delayed alpha-heating feedback | SI joules, m^-3, seconds, keV, MW, m^3/s, metres, normalised profile radius, dimensionless Q, and Lawson triple product units m^-3 s keV. | tests/test_burn_controller.py alpha-heating, Lawson, burn-fraction, profile-domain, and control checks; tests/test_burn_controller.py burn-controller edge-path checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/elm_model.py` | ELM model claims must declare pedestal crash trigger, crash depth, recovery timescale, peeling-ballooning boundary, ELM cycle state, RMP suppression threshold, and energy-particle-loss accounting. | peeling-ballooning ELM trigger references; resonant magnetic perturbation ELM suppression references | Time in seconds, energy in joules, density in m^-3, temperature in eV or keV, pressure in pascals, magnetic perturbation amplitude dimensionless or tesla with explicit convention. | tests/test_elm_model.py; tests/test_integrated_scenario.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/eped_pedestal.py` | EPED claims must declare the pedestal-width scaling, collisionless width versus collisionality-narrowed effective-width ordering, peeling-ballooning pressure limit, bootstrap-current coupling, H-mode entry assumptions, global beta limits, and compatibility boundary to the transport solver. | Snyder et al. EPED pedestal model; Troyon beta-limit reference; Sauter bootstrap-current fit | Pedestal width in normalised flux or metres with convention metadata; pressure in pascals, temperature in eV or keV, density in m^-3, current in amperes, and beta dimensionless. | tests/test_eped_pedestal.py; tests/test_cross_module_physics.py; tests/test_transport_hmode_edge.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/fusion_kernel.py` | Fusion-kernel claims must declare the Grad-Shafranov residual, finite-difference grid, coil Green functions, source-profile parameterisation, boundary conditions, nonlinear iteration controls, convergence criteria, and Rust/Python parity boundary. | Grad-Shafranov equilibrium equation; Green-function tokamak coil-response references; repository Rust/Python fusion-kernel parity contract | SI metres, webers per radian, tesla, amperes, pascals, source derivatives, grid spacings, and dimensionless convergence tolerances. | tests/test_fusion_kernel.py; tests/test_geqdsk_regression.py; tests/test_rust_python_parity.py | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/jax_gk_solver.py` | JAX linear gyrokinetic claims must preserve the native response-matrix eigenvalue formulation, expose identical physical input contracts, document numerical-precision or backend divergence, and keep the stiffness-to-transport closure bounded as a controller-tuning surrogate. | Repository native linear GK response-matrix contract; JAX numerical backend reproducibility guidance | Same growth-rate, frequency, geometry, and species units as native linear GK; backend dtype, tolerance, and normalisation must be explicit in validation artefacts. | tests/test_jax_gk_solver.py response-matrix parity, stiffness, fail-closed, and bounded chi_i-profile closure checks; validation/validate_jax_gk_parity.py strict persisted backend parity artifact gate; source-level native response-matrix parity contract | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/phase/kuramoto.py` | Phase-runtime claims must declare the Kuramoto-Sakaguchi mean-field coupling, order-parameter calculation, exogenous global-driver injection, phase wrapping convention, Euler step, and optional Rust fast-path parity boundary. | Kuramoto and Sakaguchi phase oscillator model; Repository phase synchronisation runtime contract | Phases in radians, angular frequencies in radians per second, timestep in seconds, coupling gains dimensionless or radians per second by declared convention, order parameter dimensionless. | tests/test_phase_kuramoto.py; tests/test_phase_properties.py; tests/test_phase_properties_extended.py | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/marfe.py` | MARFE claims must declare impurity radiation loss, temperature scan, condensation criterion, Greenwald comparison, power-balance inputs, bounded impurity fraction, edge parallel connection-length scaling through q95 and R0, validated front-temperature state, and detection thresholds for detached high-radiation states. | Stangeby 2000 scrape-off-layer and divertor references; Greenwald 2002 density-limit reference; radiation-condensation MARFE onset references | Temperature in eV, density in m^-3, power in MW or W with explicit conversion, plasma current in MA or A with convention metadata, impurity fraction dimensionless on (0, 1], connection length in metres. | tests/test_marfe.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/gk_geometry.py` | Geometry claims must declare Miller-shape coordinates, derivative-resolved ballooning-angle grids, Jacobian, toroidal-field convention, safety-factor field-pitch relation, poloidal-field construction, b-dot-grad-theta metric, metric coefficients, and local-equilibrium validity bounds. | Miller et al. 1998 local equilibrium; Cyclone Base Case local Miller geometry benchmark | Major radius, minor radius, local radius, and gradient lengths in metres; angles in radians; magnetic-field strength in tesla; q, shear, elongation, triangularity, and shaping derivatives dimensionless. | tests/test_gk_geometry.py field-pitch, metric, curvature, shaping, and interface checks; validation/validate_gk_geometry_reference.py immutable Miller reference cases with b-dot-grad-theta samples; source-level local Miller geometry contract | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/ntm_dynamics.py` | NTM dynamics claims must declare rational-surface search, rational-surface containment inside the minor radius, tokamak major/minor-radius ordering, modified Rutherford equation terms, bootstrap drive, polarisation and curvature terms, ECCD control coupling, seed-island assumptions, and controller-validity limits. | modified Rutherford equation NTM references; ECCD NTM control references; repository rational-surface and island-dynamics contract | Island width in metres, time in seconds, current in amperes, q dimensionless, rho dimensionless, ECCD power in MW or W with conversion metadata. | tests/test_ntm_dynamics.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/rzip_model.py` | Rigid-plasma vertical stability claims must declare the linearised state vector [Z, dZ/dt, circuit currents], mutual-inductance derivative, vertical field index, tokamak major/minor-radius ordering, declared vertical inertia, wall-normalised feedback-gain threshold, and wall or active-coil circuit model. | Lazarus et al. 1990 rigid plasma vertical stability model; Wesson 2011 tokamak vertical stability and field-index references; Repository vessel circuit model contract | SI metres, seconds, amperes, henries, ohms, tesla, and dimensionless vertical field index; growth time is reported in milliseconds. | tests/test_rzip_model.py vertical growth, field-index, declared-inertia, feedback-gain, physical-parameter, and controller-measurement checks; tests/test_rzip_model.py RZIP fallback and singular-circuit checks; validation/validate_rzip_reference.py strict RZIP reference artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/scpn/fpga_export.py` | FPGA-export claims must declare LIF fixed-point quantisation, leak right-shift approximation, threshold scaling, signed weight saturation, generated HDL boundary, target family assumptions, and synthesis-tool responsibility. | Repository SCPN compiler and fixed-point export contract; Leaky integrate-and-fire digital implementation contract | Fixed-point values use declared bit width and fractional-bit scale; clock in MHz, timestep in seconds, FIFO depth and neuron counts dimensionless, weights and thresholds quantised by explicit integer scale. | tests/test_fpga_export.py; tests/test_scpn_compiler.py | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/vmec_lite.py` | VMEC-lite claims must declare flux-surface parameterisation, rotational-transform profile, Fourier mode truncation, positive R00 major-radius boundary state, pressure/current assumptions, residual metric, and explicit exclusion from full 3D MHD equilibrium claims. | VMEC 3D equilibrium references; stellarator flux-surface Fourier parameterisation references | SI metres, tesla, pascals, amperes, webers, dimensionless rotational transform, and Fourier coefficients with declared length units. | tests/test_vmec_lite.py; validation/validate_vmec_reference.py strict VMEC reference artifact gate | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/control/advanced_soc_fusion_learning.py` | SOC turbulence-learning claims must declare the sandpile lattice, critical-gradient threshold, predator-prey zonal-flow coupling, shear-suppression term, bounded substep relaxation, Q-learning state discretisation, action set, and random policy assumptions. | Diamond and Hahm 1995 SOC turbulence reference; Kim and Diamond 2003 zonal-flow predator-prey coupling; Biglari, Diamond and Terry 1990 shear-suppression reference | Dimensionless lattice gradients, flow amplitudes, shear, toppling counts, Q-table values, reward, and RNG-seeded action choices. | tests/test_advanced_soc.py SOC physics and learning checks; tests/test_advanced_soc_verbose_plot.py verbose and plotting-path checks; tests/test_visualization_paths.py SOC visualisation checks; validation/validate_soc_reference.py strict reference-artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/current_drive.py` | Current-drive claims must declare ECCD, LHCD, and NBI source powers, radial deposition centres and widths, grid-normalised deposition conservation, density and temperature normalisation, Fisch-Boozer or Fisch efficiency coefficients, Prater launch-angle factor, Stix slowing-down time and critical energy for NBI, and the absence of ray-tracing or Fokker-Planck facility validation. | Fisch and Boozer 1980 electron-cyclotron current-drive efficiency; Prater 2004 ECCD launch-angle efficiency scaling; Fisch 1978 lower-hybrid current-drive efficiency; Stix 1972 neutral-beam slowing-down and critical-energy formulae; Ehst and Karney 1991 neutral-beam current-drive model | Power in MW or W with explicit conversion, rho dimensionless, density in 10^19 m^-3, temperatures and beam energy in keV, current density in A/m^2, total current in amperes, efficiency coefficients in declared normalised A/W form. | tests/test_current_drive.py verifies ECCD, LHCD, and NBI finite-width deposition power conservation, efficiency scaling, slowing-down time, critical energy, source superposition, and total-current integration | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/blob_transport.py` | Blob transport claims must declare interchange drive, sheath closure, radial velocity scaling, filament size, density and temperature perturbation assumptions, non-empty strictly ordered separatrix-to-wall scrape-off-layer profile coordinates, positive ordered detector-event domains, and scrape-off-layer validity bounds. | Stangeby 2000 scrape-off-layer transport references; blob-filament interchange transport scaling references | SI metres, seconds, m/s, density in m^-3, temperature in eV, magnetic field in tesla, and dimensionless normalised perturbations. | tests/test_blob_transport.py velocity-regime, size-scaling, ensemble, flux, strict-profile-grid, wall-flux, and detector-domain checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core` | Analytical approximations must state their source equation, retained terms, omitted terms, unit contract, and allowed parameter domain. | Cordey 1981 orbit-width estimate; Sauter et al. 1999 neoclassical fits; Stangeby 2000 SOL two-point model | Per-module SI or declared normalised units with explicit conversion boundaries. | docs/physics_methods.md simplification declarations; source grep for simplification and approximation markers | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control` | Control approximations must declare reduced-state dynamics, actuator assumptions, estimator assumptions, and facility-exclusion boundaries before any controller-readiness claim. | Tokamak control reduced-plant literature; Repository control safety and disruption contracts | Per-controller declared SI plasma, actuator, magnetic, force, and timing units. | src/scpn_control/control/gym_tokamak_env.py; src/scpn_control/control/free_boundary_tracking.py | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control` | Runtime approximations must state phase wrapping, fixed-point shift approximations, replay fixture provenance, and hardware export limits. | Kuramoto phase oscillator reference model; Repository fixed-point spiking export contract | Declared phase radians, tick timing, fixed-point scale factors, and replay geometry units. | src/scpn_control/phase/kuramoto.py; src/scpn_control/scpn/fpga_export.py; src/scpn_control/scpn/geometry_neutral_replay.py | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/control/mu_synthesis.py` | Mu-analysis claims must declare the structured uncertainty blocks, positive block bounds, Riccati state-feedback K-step, bound-scaled static closed-loop DC robust-performance map, D-scaling upper-bound fit, finite controller state/timestep domains, and the exclusion of full frequency-dependent H-infinity D-K synthesis unless an external validated backend is wired. | Doyle 1982 structured singular value definition; Balas et al. 1993 mu-analysis and synthesis toolbox; Skogestad and Postlethwaite 2005 multivariable feedback control | State, control, output, and uncertainty units inherit the supplied state-space plant contract; mu, uncertainty bounds, and D-scalings are dimensionless. | tests/test_mu_synthesis.py verifies D-scaled upper-bound behaviour, uncertainty-bound scaling, finite-domain rejection, and closed-loop plant dependence; docs/learning/control_theory_primer.md states the bounded static analysis domain | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/checkpoint.py` | Checkpoint claims must declare serialised solver state, non-negative episode counter, object-valued metrics schema, finite-value policy, schema-version boundary, and fail-closed handling for corrupt or missing checkpoint artefacts. | repository restart and replay provenance contract | Units are inherited from stored state and metrics; checkpoint metadata must preserve schema version, episode index, and provenance boundaries. | tests/test_checkpoint.py round-trip, resume, finite-payload, schema-version, and corrupt-payload checks | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/control/density_controller.py` | Density-control claims must declare the Greenwald limit, ITER operating margin, radial particle transport grid, gas-puff source, pellet source, NBI source, cryopump sink, recycling source, controller gains, and CFL-limited explicit transport step. | Greenwald 2002 density limit; ITER Physics Basis 1999 density operating margin; Parks and Turnbull 1978 neutral gas shielding pellet-ablation model; Milora 1995 neutral gas shielding pellet-ablation model | SI particles per second, metres, seconds, m^-3, m^2/s, m/s, pellet millimetres, beam keV, megawatts, and dimensionless Greenwald fraction. | tests/test_density_controller.py Greenwald, NGS pellet trajectory deposition, sink, controller, geometry, transport-profile, source-input, and timestep guard checks; tests/test_pellet_injection.py Parks-Turnbull NGS ablation and pellet trajectory checks; validation/validate_density_reference.py strict density reference artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/differentiable_transport.py` | Differentiable transport claims must declare the four-channel profile order, cylindrical Crank-Nicolson diffusion step, uniform normalised radial grid, source-term convention, core zero-gradient boundary, edge Dirichlet boundary, transport-coefficient gradient target, neural-closure coefficient mapping, campaign provenance metadata, schema-versioned replay metadata, and optional Grad-Shafranov flux-map radial weighting contract. | repository JAX transport primitive contract; repository integrated transport solver contract | Profiles inherit channel units for electron temperature, ion temperature, electron density, and impurity density; chi uses m^2/s or the declared normalised diffusivity convention; sources use profile units per second; rho is dimensionless and uniformly spaced. | tests/test_differentiable_transport.py diffusion, boundary, fallback, optional JAX-gradient, neural-closure coefficient mapping, campaign metadata provenance, schema-versioned replay metadata, replay-drift guard, and equilibrium-weighted GS-flux checks; tests/test_nmpc_controller.py NMPC neural-closure tuning fails closed without JAX; tests/test_nmpc_controller.py NMPC transport tuning result carries campaign and closure provenance metadata | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/free_boundary_tracking.py` | Free-boundary tracking claims must declare direct kernel-in-the-loop coil-response identification, bounded least-squares correction, actuator lag, slew limits, supervisor rejection, measurement bias, drift, latency, and observer compensation assumptions. | Grad-Shafranov free-boundary control references; Repository FusionKernel free-boundary objective contract; Repository deterministic free-boundary acceptance campaign | SI coil currents, metres, webers per radian, seconds, amperes per second, objective-space residuals, and dimensionless supervisor gains. | tests/test_free_boundary_tracking.py controller and disturbance-observer checks; tests/test_free_boundary_tracking_variants.py actuator, observer, latency, and supervisor edge-path checks; tests/test_free_boundary_tracking_acceptance.py deterministic acceptance campaign | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/disruption_contracts.py` | Disruption-contract claims must declare synthetic disruption signal generation, toroidal-mode amplitudes, mitigation-cocktail coupling, impurity transport response, halo/runaway post-disruption response, TBR equivalence scaling, and RL action bias assumptions. | Pautasso et al. 2017 disruption current-quench constraints; Riccardo et al. 2010 halo-current rise-time references; Abdou et al. 2015 blanket neutronics calibration references | SI seconds, milliseconds, mega-amperes, megajoules, moles, megawatts, dimensionless risk, toroidal mode amplitudes, and tritium breeding ratio. | tests/test_disruption_contracts.py contract smoke checks; tests/test_disruption_contracts_pure.py pure physics-path checks; tests/test_disruption_edge_cases.py edge-case disruption checks; validation/validate_disruption_reference.py strict disruption reference artifact gate | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/disruption_sequence.py` | Disruption-sequence claims must declare phase ordering, finite positive configuration domains, Rechester-Rosenbluth thermal-quench transport, post-quench radiative-cooling exposure, current-quench timing, mitigation action coupling, runaway-electron beam phase, stochastic event boundaries, halo-force convention, and replay provenance. | ITER disruption mitigation sequence references; repository disruption phase-state contract | Time in seconds or milliseconds with explicit convention, current in amperes or mega-amperes, energy in joules or megajoules, magnetic field in tesla, geometry in metres, density in 10^20 m^-3, and dimensionless phase labels. | tests/test_disruption_sequence.py post-quench cooling, phase ordering, current quench, runaway, and halo-force checks; tests/test_disruption_safe_api.py | validation_gap | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/gk_interface.py` | Generated input decks and parsed outputs must round-trip through real TGLF, GENE, GS2, CGYRO, or QuaLiKiz executables. | TGLF Staebler et al. 2007; GENE Jenko et al. 2000; GS2 Kotschenreuther et al. 1995; CGYRO Candy et al. 2016; QuaLiKiz Bourdelle et al. 2007 | Code-specific flux, growth-rate, frequency, and geometry units converted into repository normalisation with explicit metadata. | docs/joss_paper.md external GK limitation; validation/validate_gk_interface_artifacts.py strict external interface artifact gate | external_dependency_blocked | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/uncertainty.py` | Uncertainty claims must declare sampled variables, distributions, correlations, random seed, propagation chain, convergence criteria, sensitivity outputs that preserve D-T composition and fuel dilution, and finite-value rejection policy. | Monte Carlo uncertainty propagation references; repository fusion-performance uncertainty contract | Units inherit each propagated physical quantity; distribution parameters must preserve SI or declared normalised units and dimensionless uncertainty fractions. | tests/test_uncertainty.py; tests/test_full_chain_uq.py; tests/test_uncertainty_sigma_guard.py; validation/validate_uncertainty_reference.py strict UQ reference artifact gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/scpn/geometry_neutral_replay.py` | Geometry-neutral replay claims must declare synthetic W7-X-like fixture provenance, field-line spread metric, actuator current bounds, latency model, stuck-actuator fault schedule, controller feature mapping, and replay acceptance thresholds. | Repository geometry-neutral control contract; W7-X-like reduced-order stellarator replay fixture | Field-line spread in radians, currents in amperes, timestep in seconds, latency in microseconds, effective ripple dimensionless, controller objectives and thresholds declared per replay manifest. | tests/test_geometry_neutral_replay.py; tests/test_geometry_neutral_contracts.py | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gk_ood_detector.py` | OOD detector claims must declare the feature vector, training distribution, distance metric, symmetric positive-definite inverse-covariance calibration, threshold calibration, uncertainty handling, non-negative transport ensemble predictions, and behavior outside the calibrated gyrokinetic operating envelope. | Repository gyrokinetic scheduler OOD contract; statistical process monitoring distribution-shift controls | Feature units inherit declared GK inputs and outputs; detector scores are dimensionless with explicit calibration metadata, Mahalanobis metric provenance, threshold provenance, and transport prediction channels in non-negative diffusivity units. | tests/test_gk_ood_detector.py; tests/test_gk_hybrid_integration.py; validation/validate_gk_ood_calibration.py strict persisted campaign calibration gate | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/gk_online_learner.py` | Online learner claims must declare training-window selection, non-empty train/validation split domains, nonnegative transport-coefficient targets, validation loss, rollback policy, uncertainty or OOD gating, optimiser settings, and compatibility boundary with gyrokinetic scheduler inputs. | online learning stability and rollback control references; repository gyrokinetic hybrid learner contract | Inputs inherit GK feature units; transport targets are nonnegative chi_e, chi_i, and D_e; losses and OOD scores are dimensionless with explicit scaling, learning rate, epoch count, generation limit, and threshold metadata. | tests/test_gk_online_learner.py; tests/test_gk_hybrid_integration.py | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/gk_species.py` | Species and collision claims must declare charge, mass, density, temperature, thermal speed, Larmor radius, gyroaverage Bessel evaluation, diamagnetic frequency, pitch-angle deflection coefficient, thermal energy-relaxation coefficient, electron-ion mass-ratio scaling, field-temperature dependence, valid species parameter bounds, valid velocity quadrature sizes, and strictly ordered finite lambda grids bounded by the local trapped-passing boundary. | Sugama et al. 2006 collision operator; gyrokinetic normalisation and species parameter contracts | Mass in kilograms, charge in coulombs, density in m^-3, temperature in electronvolts, velocity in m/s, gyrofrequency in rad/s, Larmor radius in metres, collision rates in s^-1, lambda on [0, 1] with lambda times B/B0 no greater than one, and positive magnetic-field ratio. | tests/test_gk_species.py; tests/test_gk_electromagnetic.py; validation/validate_gk_species_reference.py immutable species and collision reference cases with distinct deflection and energy-relaxation channels | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/halo_re_physics.py` | Post-disruption claims must declare the halo L/R circuit, contact fraction, toroidal peaking factor, current-quench waveform, Connor-Hastie primary generation, Rosenbluth-Putvinski avalanche generation, and ensemble uncertainty assumptions. | Fitzpatrick 2002 halo current and error-field interaction; Connor and Hastie 1975 runaway electron generation; Rosenbluth and Putvinski 1997 avalanche generation | SI amperes, mega-amperes, seconds, milliseconds, ohms, henries, metres, volts per metre, rates per second, and meganewtons per metre. | tests/test_halo_re_physics.py halo-current and runaway-electron regression checks; tests/test_halo_nonfinite_guards.py non-finite guard checks; tests/test_halo_validation_paths.py validation-path guard checks | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/stability_mhd.py` | MHD stability claims must declare q-profile interpolation, magnetic shear, interior-point radial grids for profile-resolved criteria, Mercier and Troyon criteria, beta-limit inputs, first-unstable-radius search, bounded bootstrap-current fraction, and the exclusion boundary for full ideal-MHD or resistive-MHD eigenmode claims. | Troyon beta-limit reference; Mercier ideal-MHD stability criterion; tokamak q-profile and magnetic-shear references | q and beta dimensionless, plasma current in MA or A with explicit convention, minor radius and major radius in metres, magnetic field in tesla, rho dimensionless, and local bootstrap-current fraction within [0, 1]. | tests/test_stability_mhd.py; tests/test_ballooning_solver.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/integrated_scenario.py` | Integrated scenario claims must declare coupling order, current diffusion, transport, pedestal, ELM, NTM, sawtooth, burn, and control interactions, timestep policy, state exchange units, and failure isolation boundaries. | integrated tokamak scenario modelling references; repository scenario-coupling contract | All coupled states must preserve declared SI or normalised units for current, density, temperature, pressure, q, beta, power, flux, and timing. | tests/test_integrated_scenario.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/kinetic_efit.py` | Kinetic-EFIT coupling claims must declare Thomson or profile-derived electron density and temperature points, ion-temperature points, fast-ion pressure fraction, anisotropy sigma, MSE pitch-angle constraints when used, radial interpolation geometry, pressure-consistency residual, and exclusion from facility-grade P-EFIT unless matched reference equilibria are supplied. | Lao et al. 1985 EFIT equilibrium reconstruction; MSE-constrained kinetic EFIT workflow; Repository fixed-boundary realtime-EFIT contract | R and Z in metres, temperatures in keV, density in 10^19 m^-3, MSE pitch angle in degrees, pressure in pascals, beta dimensionless, and q-profile dimensionless. | tests/test_kinetic_efit.py requires measured profile channels, radial interpolation, measured Ti use, anisotropy residuals, MSE q-profile response, and fast-ion pressure checks | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gk_eigenvalue.py` | Native linear GK eigenvalue solve must match external TGLF, GENE, GS2, CGYRO, or QuaLiKiz growth rates and real frequencies for the same Miller geometry and species inputs. | Miller et al. 1998 local equilibrium; GENE and GACODE published input-output contracts | Growth rate and frequency normalised consistently to c_s/a or declared external-code convention. | ROADMAP.md local-dispersion overprediction note; docs/competitive_analysis.md linear GK quantitative accuracy status; validation/validate_gk_crosscode.py strict real-binary evidence gate | external_dependency_blocked | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/momentum_transport.py` | Momentum transport claims must declare NBI torque, collisional damping, viscous momentum diffusion, rotation-profile boundary conditions, angular-momentum units, and the coupling boundary to the integrated transport solver. | tokamak toroidal momentum transport references; repository NBI torque and rotation-profile contract | SI newton metres, kg m^2/s^2, rad/s, metres, seconds, density in m^-3, and momentum diffusivity in m^2/s. | tests/test_momentum_transport.py; tests/test_momentum_integration.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/neural_equilibrium.py` | Neural equilibrium predictions must be compared against identical P-EFIT or equivalent reference equilibria for psi, pressure, q, and boundary geometry. | EFIT/P-EFIT equilibrium reconstruction workflow; Repository neural equilibrium model contract | SI magnetic flux, pressure, metre-scale geometry, and dimensionless q-profile arrays on declared grids. | ROADMAP.md neural eq cross-validation future item; docs/joss_paper.md neural equilibrium limitation; validation/validate_neural_equilibrium_reference.py strict P-EFIT/reference artifact gate | external_dependency_blocked | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/neural_transport.py` | Neural transport claims must declare input feature normalisation, QLKNN weight provenance, prediction targets, fallback critical-gradient thresholds, bounded density-channel particle diffusivity, bounded profile closure provenance, uncertainty output, out-of-domain handling, and cross-validation against reference transport cases. | QuaLiKiz neural network transport surrogate references; repository QLKNN weight and metric contract | Inputs and outputs use declared transport feature units, diffusivity in m^2/s or declared normalisation, fluxes in SI or gyro-Bohm units with conversion metadata. | tests/test_neural_transport.py density-gradient and shear-dependent fallback particle diffusivity; tests/test_neural_transport_core.py profile fallback particle-channel checks; tests/test_neural_transport_core.py bounded closure provenance and fallback gating checks; tests/test_qlknn_transport.py; validation/validate_neural_transport_reference.py strict QuaLiKiz/reference artifact gate | external_dependency_blocked | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/neural_turbulence.py` | Neural turbulence claims must declare QLKNNSurrogate inputs, finite strictly ordered physical profile grids, feature scaling, banana-regime electron collisionality, bounded analytic quasilinear target variables, fallback behaviour, uncertainty handling, and cross-validation boundary against gyrokinetic or quasilinear turbulence references. | QLKNN turbulence surrogate references; repository neural turbulence surrogate contract | Feature and target units follow the declared turbulence surrogate schema; diffusivities, growth rates, and fluxes require explicit SI or normalised-unit metadata. | tests/test_neural_turbulence.py collisionality scaling, analytic target, training, save-load, and denormalisation checks; validation/validate_neural_turbulence_reference.py strict GK-campaign/reference artifact gate | validation_gap | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/gk_nonlinear.py` | Five-dimensional delta-f flux-tube Vlasov evolution with E cross B bracket, ballooning connection, kinetic electrons, and Sugama collision terms. | Dimits et al. 2000 Cyclone Base Case; Sugama et al. 2006 collision operator | Gyro-Bohm-normalised heat flux and chi_i with explicit R/L_Ti, rho_s, c_s, and reference gradient normalisation. | ROADMAP.md v0.18.0 open revalidation item; docs/joss_paper.md nonlinear GK validation limitation; validation/gk_nonlinear_cyclone.py saturation-evidence assessor | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/orbit_following.py` | Orbit-following claims must declare guiding-centre equations, magnetic moment handling, finite pitch-angle bounds, banana-orbit width scaling, collision or loss assumptions, ensemble sampling, and valid magnetic-geometry bounds. | Cordey 1981 orbit-width estimate; tokamak guiding-centre orbit references | SI metres, seconds, tesla, electronvolts, kg, coulombs, pitch angle in radians on [0, pi], and ensemble probabilities dimensionless. | tests/test_orbit_following.py; validation/validate_orbit_reference.py strict orbit reference artifact gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/realtime_efit.py` | Real-time equilibrium reconstruction claims must declare the fixed-boundary Grad-Shafranov solve, polynomial p-prime and FF-prime source profiles, diagnostic response interpolation, Rogowski current reconstruction, and macroscopic-shape extraction assumptions. | Lao et al. 1985 EFIT equilibrium reconstruction; Strait et al. 2019 real-time EFIT workflow references; Repository GEQDSK and magnetic-diagnostic validation contracts | SI metres, webers per radian, tesla, amperes, pascals per weber, FF-prime units, and dimensionless q95, beta_pol, li, elongation, and triangularity. | tests/test_realtime_efit.py fixed-boundary solve and diagnostic reconstruction checks; tests/test_kinetic_efit.py kinetic EFIT integration checks; tests/test_validate_real_shots_equilibrium.py GEQDSK source residual checks | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gyrokinetic_transport.py` | Reduced gyrokinetic transport claims must declare the dispersion relation, instability branch classification, quasilinear heat-flux closure, critical-gradient threshold, saturation rule, positive integer mode-count domain, tokamak geometry ordering, and mapping from local GK outputs to transport coefficients. | Cyclone Base Case gyrokinetic benchmark; quasilinear gyrokinetic transport closure references | Growth rates and frequencies in declared normalised units, gradients dimensionless or m^-1 with explicit convention, rho and epsilon dimensionless, geometry in metres, heat flux in gyro-Bohm or SI units with conversion metadata. | tests/test_gyrokinetic_transport.py; tests/test_gk_benchmark_linear.py | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/gym_tokamak_env.py` | 0D/1D control plants must declare reduced-state equations, conservation assumptions, and controller-validity bounds before being used for hardware or facility claims. | Wesson 2011 tokamak transport references; Repository Gymnasium plant and digital-twin contracts | Declared plasma current, temperature, density, confinement, and actuator units per state vector field. | README.md validation limitation; docs/use_cases.md synthetic disruption prediction status; tests/test_gym_tokamak_env.py reduced-order plant bounds and action-contract checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/rwm_feedback.py` | RWM feedback claims must declare the no-wall and ideal-wall beta limits, resistive-wall L/R time, optional wall-plasma gap correction, rotation-stabilisation term, active proportional or derivative feedback coupling, controller latency, coil coupling, and ideal-kink exclusion boundary. | Bondeson and Ward 1994 resistive-wall-mode growth-rate model; Fitzpatrick 2001 rotation-stabilisation contribution; Strait et al. 2003 wall-plasma gap correction; Garofalo et al. 2002 active feedback experiments | Normalised beta dimensionless, wall time in seconds, toroidal rotation in rad/s, wall and plasma radii in metres, feedback gain dimensionless per declared coil coupling, growth rate in s^-1. | tests/test_rwm_feedback.py growth-rate, rotation, wall-geometry, feedback, ideal-kink, and required-gain checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/tearing_mode_coupling.py` | Sawtooth-to-NTM seeding claims must declare sawtooth crash trigger, seed-island generation, rational-surface coupling, phase and amplitude assumptions, bounded bootstrap-current fraction, NTM coupling path, and exclusion of full nonlinear MHD crash dynamics. | sawtooth crash and NTM seeding references; repository tearing-mode coupling contract | Time in seconds, island width in metres, q dimensionless, rho dimensionless, phase in radians, local bootstrap-current fraction within [0, 1], and amplitude units declared per coupling signal. | tests/test_tearing_mode_coupling.py; tests/test_integrated_scenario.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/tokamak_digital_twin.py` | Digital-twin claims must declare the 2D poloidal grid, q-profile topology update, rational-surface island mask, finite-difference diffusion, optional gyrokinetic surrogate correction, radiation-loss scaling, actuator latency, RNG seed, and IDS export assumptions. | Repository digital-twin runtime contract; Wesson 2011 radiation-loss reference; Repository IMAS IDS export helper contract | Declared normalised grid indices, keV, m^-3, dimensionless q-profile values, actuator-lag summary units, RNG seed metadata, and IDS-compatible pulse history fields. | tests/test_tokamak_digital_twin.py deterministic twin checks; tests/test_digital_twin_physics.py physics-path checks; tests/test_digital_twin_ingest_runtime.py ingest runtime checks; validation/validate_digital_twin_reference.py strict reference-artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/integrated_transport_solver.py` | Integrated transport claims must declare the axis-to-edge radial grid, heat and particle diffusion equations, neoclassical closure, bootstrap-current fit, source deposition terms, boundary conditions, timestep limits, and coupling contracts to pedestal and gyrokinetic closures. | Chang-Hinton neoclassical transport model; Sauter et al. 1999 bootstrap-current fits; repository integrated transport solver contract | SI metres, seconds, m^-3, eV, W/m^3, m^2/s, amperes per square metre, and dimensionless profile coordinates with explicit axis-to-edge normalisation metadata. | tests/test_integrated_transport_solver.py; tests/test_transport_energy_conservation.py; tests/test_transport_neoclassical_guards.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/volt_second_manager.py` | Volt-second management claims must declare inductive flux, resistive loop-voltage integration, Ejima startup flux, bootstrap-current correction, flat-top duration estimate, ramp and ramp-down decomposition, flux-budget margin assumptions, finite positive machine constants, nonnegative current and voltage domains, positive timesteps, and strictly ordered bootstrap-profile grids. | Wesson 2011 tokamak loop-voltage and flux-balance equations; Ejima et al. 1982 startup flux coefficient; ITER Physics Basis 1999 flat-top flux-budget references | SI webers, volt-seconds, henries, ohms, amperes, mega-amperes, seconds, metres, normalised profile radius, and dimensionless Ejima coefficient. | tests/test_volt_second_manager.py flux budget, scenario analysis, bootstrap-profile, and domain-boundary checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |

## Components

### DIII-D experimental replay

- Fidelity status: `reference_validated`
- Module path: `validation/reference_data/diiid`
- Full-fidelity public claim: allowed
- External validation tracker: none
- Covered source paths: 0
- Required actions:
  - Keep GEQDSK and disruption-shot reference manifests covered by checksum verification
  - Use the MDSplus acquisition spec path for new facility artefacts instead of synthetic fixtures

### DT burn control and alpha-heating model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/burn_controller.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Validate burn operating points against an integrated transport solver or published burn-control benchmark before reactor-control claims
  - Replace the burn-fraction approximation with a calibrated slowing-down and ash-removal model before facility extrapolation

### ELM crash and RMP suppression approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/elm_model.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate ELM frequency, crash depth, and RMP suppression windows against measured or published H-mode cases
  - Persist pedestal pre-crash and post-crash profiles with every ELM validation artefact

### EPED pedestal and peeling-ballooning approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/eped_pedestal.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate pedestal height and width against published EPED benchmark points or measured pedestal databases
  - Record bootstrap-current, beta-limit, and shaping inputs for every pedestal validation artefact

### Grad-Shafranov fusion-kernel numerical contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/fusion_kernel.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Validate reconstructed equilibria against EFIT, GEQDSK, or published fixed-boundary benchmark cases
  - Preserve residual norms, convergence metadata, and Rust/Python parity evidence for every validation artefact

### JAX gyrokinetic numerical parity guard

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/jax_gk_solver.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Run backend parity over CBC, TEM, electromagnetic, and stable-mode cases with pinned tolerances
  - Persist CPU/GPU backend metadata with each parity artefact before claiming accelerator-equivalent physics

### Kuramoto-Sakaguchi phase synchronisation runtime

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/phase/kuramoto.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Validate synchronisation and stability metrics against published Kuramoto-Sakaguchi benchmark cases before broader phase-control claims
  - Persist Rust/Python parity evidence and timestep convergence checks for deployment-target oscillator counts

### MARFE radiation-condensation density-limit contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/marfe.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate onset temperatures and density limits against measured or published MARFE cases
  - Add impurity-specific radiation tables or documented provenance for each supported impurity species

### Miller local-equilibrium geometry and field-pitch contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_geometry.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Compare metric coefficients and field-pitch factors against an independent Miller-geometry reference implementation
  - Store immutable reference cases covering circular, shaped, and high-shear local equilibria

### NTM island evolution and control approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/ntm_dynamics.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate island growth and suppression against measured or published NTM benchmark cases
  - Persist q-profile, rational-surface, seed-island, and ECCD alignment metadata with each validation case

### RZIP rigid vertical stability model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/rzip_model.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Calibrate vertical_inertia_kg against a documented plasma inertia or facility parameter source before facility claims
  - Validate vertical growth rates against a reference RZIP, CREATE-L/NL, TSC, or measured vertical-displacement benchmark

### SCPN FPGA fixed-point export boundary

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/scpn/fpga_export.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Run generated HDL through real Vivado, Quartus, or Yosys synthesis before hardware readiness claims
  - Persist resource utilisation, timing closure, and bit-accurate simulator evidence for every supported target family

### VMEC-lite stellarator equilibrium approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/vmec_lite.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Validate surface geometry and rotational-transform outputs against VMEC or published stellarator benchmark equilibria
  - Persist Fourier truncation, pressure profile, current profile, and residual metadata for each validation case

### advanced SOC turbulence learning controller

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/advanced_soc_fusion_learning.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Replace sandpile turbulence proxy with gyrokinetic or measured turbulence evidence before transport-control claims
  - Validate Q-learning policy behaviour against replayed plasma-control objectives before control-readiness claims

### auxiliary current-drive deposition and efficiency model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/current_drive.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate ECCD and LHCD deposition against ray-tracing or measured deposition reconstructions before facility claims
  - Validate NBI deposition, fast-ion slowing down, and driven-current profiles against a beam-deposition or Fokker-Planck reference before facility claims
  - Persist source power, deposition kernel, rho grid, density and temperature profiles, launch geometry, and efficiency coefficients for every promoted current-drive claim

### blob transport and scrape-off-layer approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/blob_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate blob velocity and spreading against measured or published SOL filament cases
  - Persist filament size, collisionality, sheath, and magnetic-geometry metadata with each validation artefact

### bounded analytical approximations in physics modules

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 0
- Required actions:
  - Keep this parent entry as a non-source guardrail for future core approximation markers
  - Require every new core approximation marker to land with a source-specific traceability entry and validation evidence

### bounded control plant approximations

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 0
- Required actions:
  - Keep this parent entry as a non-source guardrail for future reduced-order control plant markers
  - Require every new control plant marker to land with a controller-specific traceability entry and replay or higher-fidelity evidence

### bounded phase and spiking runtime approximations

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 0
- Required actions:
  - Keep this parent entry as a non-source guardrail for future phase, spiking, and replay runtime markers
  - Require every new runtime marker to land with a source-specific traceability entry plus hardware-target or replay-fixture evidence

### bounded static structured-singular-value control analysis

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/mu_synthesis.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Wire a validated frequency-dependent H-infinity synthesis backend before making full D-K synthesis claims
  - Persist benchmark plants, frequency grids, D-scale fits, and mu upper/lower bounds for every promoted controller claim

### checkpoint state serialisation boundary contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/checkpoint.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Version checkpoint schemas before long-running production campaigns depend on replay compatibility
  - Add migration fixtures for every future checkpoint schema change

### density control and particle-source model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/density_controller.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Replace circular-geometry volume elements with equilibrium-derived flux-surface volumes before facility-density claims
  - Validate recycling source and full density-profile evolution against measured or published fuelling cases

### differentiable transport facade contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/differentiable_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Wire each remaining legacy transport surface through an equivalent differentiable path before claiming end-to-end transport autodiff
  - Validate differentiable transport profiles and equilibrium-weighted gradients against measured discharges or published integrated-modelling benchmarks
  - Persist backend, dtype, radial-grid, boundary-condition, equilibrium-grid, flux-weighting, gradient-tolerance, and replay-drift metadata for controller-tuning campaigns
  - Replace the bounded JAX GK stiffness closure with externally validated GK transport coefficients before quantitative transport-control claims

### direct free-boundary tracking controller

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/free_boundary_tracking.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Promote deterministic acceptance evidence to measured or validated free-boundary replay cases before facility-control claims
  - Attach actuator, latency, and sensor-calibration evidence for the target device before deployment claims

### disruption mitigation contract layer

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/disruption_contracts.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Replace synthetic disruption signals with measured labelled disruption windows before predictive claims
  - Validate mitigation-cocktail, halo, runaway, and TBR couplings against experimental or benchmark artefacts

### disruption sequence phase-ordering contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/disruption_sequence.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate phase timing, post-quench temperature, and mitigation branches against labelled measured disruption windows
  - Persist shot identifiers, phase labels, thermal-quench duration, radiation-time assumption, timing tolerances, and mitigation metadata for sequence validation

### external gyrokinetic interfaces

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/gk_interface.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Run each interface against a real executable or documented public reference output
  - Promote parser fixtures from mock subprocesses to immutable external-code artefacts

### full-chain uncertainty quantification contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/uncertainty.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Calibrate uncertainty priors against measured or published scenario ensembles
  - Persist random seeds, distribution parameters, sample counts, convergence diagnostics, and sensitivity metrics with every UQ artefact

### geometry-neutral stellarator replay fixture

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/scpn/geometry_neutral_replay.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Replace synthetic W7-X-like replay inputs with measured or benchmark stellarator field-line artefacts before device-control claims
  - Persist replay manifests with magnetic-configuration provenance, actuator calibration, latency evidence, and acceptance thresholds

### gyrokinetic OOD detector distribution-bound contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_ood_detector.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Calibrate thresholds against real or published GK campaign ensembles
  - Add false-positive and false-negative acceptance criteria for deployment gating

### gyrokinetic online learner stability contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_online_learner.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Validate learner updates on immutable gyrokinetic campaign artefacts with held-out distribution-shift cases
  - Persist model weights, rollback decisions, and validation metrics for every online update

### gyrokinetic species and collision bounded-operator contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_species.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Validate bounded test-particle collision coefficients against a published gyrokinetic or Fokker-Planck reference case before facility claims
  - Add field-particle momentum-conservation and full multispecies collision-operator evidence before claiming quantitative collisional damping

### halo current and runaway electron disruption model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/halo_re_physics.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate halo current and runaway-current envelopes against measured disruption databases before mitigation claims
  - Attach wall-contact geometry and impurity-radiation evidence before extrapolating beyond the declared ensemble domain

### ideal-MHD stability metric approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/stability_mhd.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Compare stability metrics against an independent MHD stability code or published benchmark profiles
  - Store profile grids, interpolation choices, and unstable-region metadata with each stability validation artefact

### integrated scenario coupling contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/integrated_scenario.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate coupled scenario trajectories against measured discharges or published integrated-modelling benchmarks
  - Persist module-by-module state exchange, timestep, and convergence metadata with every scenario artefact

### kinetic EFIT pressure and q-profile coupling

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/kinetic_efit.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Validate kinetic pressure, q-profile, and anisotropy residuals against matched EFIT or P-EFIT reference equilibria before facility claims
  - Persist diagnostic source, interpolation geometry, fast-ion model provenance, MSE calibration, and uncertainty for every promoted kinetic-equilibrium claim

### linear gyrokinetic cross-code agreement

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/gk_eigenvalue.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Install or provide real external GK binaries
  - Run native-vs-external benchmark cases and store parser artefact evidence

### momentum transport and torque-balance approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/momentum_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate torque deposition and rotation evolution against measured or published NBI momentum cases
  - Add acceptance artefacts for low-torque, high-torque, and sign-changing rotation profiles

### neural equilibrium cross-validation

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_equilibrium.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Acquire matched P-EFIT reference equilibria or an openly redistributable equivalent
  - Add error bounds for psi, pressure, q-profile, LCFS, and magnetic-axis predictions

### neural transport surrogate validation contract

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Acquire or generate immutable reference QLKNN or transport benchmark cases for cross-validation
  - Record weight checksums, feature scaling, fallback density-channel coefficients, closure source metadata, uncertainty metadata, and OOD decisions with every surrogate evaluation

### neural turbulence surrogate validation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/neural_turbulence.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Validate surrogate outputs against immutable gyrokinetic or QuaLiKiz turbulence benchmark cases
  - Persist feature scaling, collisionality convention, model checksum, fallback mode, and error metrics for each validation case

### nonlinear gyrokinetic heat-flux saturation

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_nonlinear.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Run a long enough nonlinear CBC campaign to saturated chi_i
  - Compare saturated heat flux against published or real cross-code reference data

### orbit-following guiding-centre approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/orbit_following.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate orbit widths and loss fractions against an independent orbit-following code or published benchmark cases
  - Persist particle species, pitch, energy, geometry, and ensemble seed metadata with each validation artefact

### real-time EFIT-lite equilibrium reconstruction

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/realtime_efit.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Validate reconstructed psi, Ip, q95, beta_pol, and li against matched EFIT or P-EFIT equilibria before facility claims
  - Replace synthetic diagnostic-response evidence with measured flux-loop, B-probe, and Rogowski artefacts when facility data are available

### reduced gyrokinetic transport closure contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gyrokinetic_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Compare predicted growth rates and transport coefficients with external GK or published quasilinear benchmark cases
  - Store branch classification and saturation metadata for every transport-closure validation case

### reduced-order plant and control environments

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/gym_tokamak_env.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Separate reduced-order training claims from facility-control claims in generated reports
  - Add plant-model validity bounds and cross-checks against higher-fidelity replay cases

### resistive-wall-mode feedback model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/rwm_feedback.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Validate beta-window, rotation, and feedback-gain predictions against measured RWM shots or external MHD stability references before facility-control claims
  - Persist wall geometry, rotation profile, coil coupling, sensor geometry, controller latency, and beta-limit provenance for every promoted RWM claim

### sawtooth-to-NTM seeding approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/tearing_mode_coupling.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate seed-island amplitudes and NTM triggering windows against measured or published sawtooth-triggered NTM cases
  - Persist pre-crash q-profile, crash amplitude, rational-surface, and coupling metadata with each validation case

### tokamak digital twin topology and diffusion model

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/tokamak_digital_twin.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Replace normalised 2D diffusion dynamics with equilibrium and transport-calibrated state evolution before facility twin claims
  - Validate IDS export and actuator-lag summaries against replayed or measured discharge histories

### transport solver neoclassical and source-term approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/integrated_transport_solver.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate radial transport profiles against a measured discharge or published integrated-modelling benchmark
  - Persist source deposition, boundary condition, and convergence metadata with every transport validation artefact

### volt-second budget and flux-consumption manager

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/volt_second_manager.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Validate flux consumption against real loop-voltage and current traces before scenario-duration claims
  - Replace bootstrap-current proxy with neoclassical or transport-solver evidence before facility extrapolation
