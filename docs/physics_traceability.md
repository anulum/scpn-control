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
- Registry entries: 54
- Open fidelity gaps: 53
- Full-fidelity public claims blocked: 53
- Resolved module paths: 54
- Resolved evidence paths: 321
- External validation trackers: 8
- Source marker coverage: 33/33

## External Validation Collaboration Trackers

- External validation artefacts needed for full-fidelity SCPN-CONTROL claims: [#46](https://github.com/anulum/scpn-control/issues/46) — 0 open claim(s) — Parent tracker for external code, reference data, facility replay, benchmark, and hardware evidence requests.
- External gyrokinetic validation artefacts: [#47](https://github.com/anulum/scpn-control/issues/47) — 9 open claim(s) — TGLF, GENE, GS2, CGYRO, QuaLiKiz, nonlinear CBC, Miller geometry, species, JAX parity, OOD, and online-learning artefacts.
- Equilibrium and reconstruction reference artefacts: [#48](https://github.com/anulum/scpn-control/issues/48) — 5 open claim(s) — DIII-D or equivalent shots, EFIT, P-EFIT, GEQDSK, VMEC, and stellarator replay artefacts.
- Transport, edge, MHD, and scenario benchmark artefacts: [#49](https://github.com/anulum/scpn-control/issues/49) — 15 open claim(s) — Integrated transport, momentum, pedestal, ELM, MARFE, NTM, current drive, stability, tearing, SOL, orbit, UQ, and scenario benchmarks.
- Neural surrogate validation artefacts: [#50](https://github.com/anulum/scpn-control/issues/50) — 3 open claim(s) — QLKNN, QuaLiKiz, gyrokinetic, transport, turbulence, and equilibrium surrogate reference datasets and weight provenance.
- Plasma-control and facility replay artefacts: [#51](https://github.com/anulum/scpn-control/issues/51) — 12 open claim(s) — RZIP, RWM, free-boundary, density, digital twin, SOC learning, burn, volt-second, mu-synthesis, and reduced-plant replay evidence.
- Disruption, halo-current, and mitigation benchmark artefacts: [#52](https://github.com/anulum/scpn-control/issues/52) — 4 open claim(s) — Measured disruption databases, labelled disruption windows, halo/runaway envelopes, wall contact, impurity radiation, and mitigation metadata.
- Hardware, HDL, CODAC/EPICS, and runtime deployment evidence: [#53](https://github.com/anulum/scpn-control/issues/53) — 5 open claim(s) — Vivado, Quartus, Yosys, timing closure, simulator evidence, CODAC/EPICS timing, interlocks, backpressure, HIL replay, and runtime parity.

## Module Traceability Table

| Module | Equation or contract | References | Unit contract | Validation evidence | Status | Tracker |
|--------|----------------------|------------|---------------|---------------------|--------|---------|
| `validation/reference_data/diiid` | Replay claims require measured signal arrays, physical units, shot identifiers, retrieval timestamp, licence policy, immutable checksums, and source URI provenance. | DIII-D MDSplus facility data access contract; Repository real-data manifest schema 1.0 | Signal-specific SI or dimensionless units; arbitrary units are rejected for real-shot evidence. | validation/validate_data_manifests.py; src/scpn_control/core/real_data_manifest.py; tests/test_real_diiid_shots.py; tests/test_validate_data_manifests.py; strict acquisition-readiness gate via --require-real-acquisition | reference_validated |  |
| `src/scpn_control/control/burn_controller.py` | Burn-control claims must declare Bosch-Hale DT reactivity, alpha-energy partition, Lawson triple product, burn-fraction approximation, alpha-heating integration volume, tokamak major/minor-radius ordering, strictly ordered normalised alpha-heating profile grids, reactivity-exponent stability boundary, and PI auxiliary-heating limits. | Bosch and Hale 1992 DT reactivity; Lawson 1957 ignition criterion; Mitarai and Muraoka 1999 delayed alpha-heating feedback; Repository fail-closed burn-control claim-admission contract | SI joules, m^-3, seconds, keV, MW, m^3/s, metres, normalised profile radius, dimensionless Q, and Lawson triple product units m^-3 s keV. | tests/test_burn_controller.py alpha-heating, Lawson, burn-fraction, profile-domain, control, and claim-admission checks; validation/validate_burn_reference.py strict burn-control reference artifact gate; validation/benchmark_burn_control_claims.py bounded claim-admission benchmark | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/elm_model.py` | ELM model claims must declare pedestal crash trigger, crash depth, recovery timescale, peeling-ballooning boundary, ELM cycle state, RMP suppression threshold, and energy-particle-loss accounting. | peeling-ballooning ELM trigger references; resonant magnetic perturbation ELM suppression references | Time in seconds, energy in joules, density in m^-3, temperature in eV or keV, pressure in pascals, magnetic perturbation amplitude dimensionless or tesla with explicit convention. | tests/test_elm_model.py; tests/test_integrated_scenario.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/eped_pedestal.py` | EPED claims must declare the pedestal-width scaling, collisionless width versus collisionality-narrowed effective-width ordering, peeling-ballooning pressure limit, bootstrap-current coupling, H-mode entry assumptions, global beta limits, and compatibility boundary to the transport solver. | Snyder et al. EPED pedestal model; Troyon beta-limit reference; Sauter bootstrap-current fit | Pedestal width in normalised flux or metres with convention metadata; pressure in pascals, temperature in eV or keV, density in m^-3, current in amperes, and beta dimensionless. | tests/test_eped_pedestal.py; tests/test_cross_module_physics.py; tests/test_transport_hmode_edge.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/fusion_kernel.py` | Fusion-kernel claims must declare the Grad-Shafranov residual, finite-difference grid, coil Green functions, source-profile parameterisation, boundary conditions, nonlinear iteration controls, convergence criteria, and Rust/Python parity boundary. | Grad-Shafranov equilibrium equation; Green-function tokamak coil-response references; repository Rust/Python fusion-kernel parity contract | SI metres, webers per radian, tesla, amperes, pascals, source derivatives, grid spacings, and dimensionless convergence tolerances. | tests/test_fusion_kernel.py; tests/test_geqdsk_regression.py; tests/test_rust_python_parity.py | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/jax_gk_solver.py` | JAX linear gyrokinetic claims must preserve the native response-matrix eigenvalue formulation, expose identical physical input contracts, document numerical-precision or backend divergence, and keep the stiffness-to-transport closure bounded as a controller-tuning surrogate. | Repository native linear GK response-matrix contract; JAX numerical backend reproducibility guidance | Same growth-rate, frequency, geometry, and species units as native linear GK; backend dtype, tolerance, and normalisation must be explicit in validation artefacts. | tests/test_jax_gk_solver.py response-matrix parity, stiffness, fail-closed, and bounded chi_i-profile closure checks; validation/validate_jax_gk_parity.py strict persisted backend parity artifact gate; source-level native response-matrix parity contract | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/phase/kuramoto.py` | Phase-runtime claims must declare the Kuramoto-Sakaguchi mean-field coupling, order-parameter calculation, exogenous global-driver injection, phase wrapping convention, Euler step, and optional Rust fast-path parity boundary. | Kuramoto and Sakaguchi phase oscillator model; Repository phase synchronisation runtime contract | Phases in radians, angular frequencies in radians per second, timestep in seconds, coupling gains dimensionless or radians per second by declared convention, order parameter dimensionless. | tests/test_phase_kuramoto.py; tests/test_phase_properties.py; tests/test_phase_properties_extended.py | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/marfe.py` | MARFE claims must declare impurity radiation loss, temperature scan, condensation criterion, Greenwald comparison, power-balance inputs, bounded impurity fraction, edge parallel connection-length scaling through q95 and R0, validated front-temperature state, and detection thresholds for detached high-radiation states. | Stangeby 2000 scrape-off-layer and divertor references; Greenwald 2002 density-limit reference; radiation-condensation MARFE onset references | Temperature in eV, density in m^-3, power in MW or W with explicit conversion, plasma current in MA or A with convention metadata, impurity fraction dimensionless on (0, 1], connection length in metres. | tests/test_marfe.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/gk_geometry.py` | Geometry claims must declare Miller-shape coordinates, derivative-resolved ballooning-angle grids, Jacobian, toroidal-field convention, safety-factor field-pitch relation, poloidal-field construction, b-dot-grad-theta metric, metric coefficients, contravariant metric determinant identity, and local-equilibrium validity bounds. | Miller et al. 1998 local equilibrium; Cyclone Base Case local Miller geometry benchmark | Major radius, minor radius, local radius, and gradient lengths in metres; angles in radians; magnetic-field strength in tesla; q, shear, elongation, triangularity, and shaping derivatives dimensionless. | tests/test_gk_geometry.py field-pitch, metric, contravariant metric-determinant identity, curvature, shaping, and interface checks; validation/validate_gk_geometry_reference.py immutable Miller reference cases with b-dot-grad-theta samples; source-level local Miller geometry contract | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/ntm_dynamics.py` | NTM dynamics claims must declare rational-surface search, rational-surface containment inside the minor radius, tokamak major/minor-radius ordering, modified Rutherford equation terms, bootstrap drive, polarisation and curvature terms, ECCD control coupling, seed-island assumptions, and controller-validity limits. | modified Rutherford equation NTM references; ECCD NTM control references; repository rational-surface and island-dynamics contract | Island width in metres, time in seconds, current in amperes, q dimensionless, rho dimensionless, ECCD power in MW or W with conversion metadata. | tests/test_ntm_dynamics.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/rzip_model.py` | Rigid-plasma vertical stability claims must declare the linearised state vector [Z, dZ/dt, circuit currents], mutual-inductance derivative, vertical field index, tokamak major/minor-radius ordering, declared vertical inertia, wall-normalised feedback-gain threshold, wall or active-coil circuit model, calibration source, growth-rate comparison tolerance, and facility-claim admission status. | Lazarus et al. 1990 rigid plasma vertical stability model; Wesson 2011 tokamak vertical stability and field-index references; Repository vessel circuit model contract | SI metres, seconds, amperes, henries, ohms, tesla, and dimensionless vertical field index; growth time is reported in milliseconds. | tests/test_rzip_model.py vertical growth, field-index, declared-inertia, feedback-gain, physical-parameter, and controller-measurement checks; tests/test_rzip_model.py RZIP fallback and singular-circuit checks; validation/validate_rzip_reference.py strict RZIP reference artifact gate; tests/test_rzip_model.py calibration evidence, facility-claim admission, and persisted evidence checks; validation/benchmark_rzip_calibration.py bounded local RZIP calibration benchmark; validation/reports/rzip_calibration.json bounded local calibration report | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/scpn/fpga_export.py` | FPGA-export claims must declare LIF fixed-point quantisation, leak right-shift approximation, threshold scaling, signed weight saturation, generated HDL boundary, target family assumptions, and synthesis-tool responsibility. | Repository SCPN compiler and fixed-point export contract; Leaky integrate-and-fire digital implementation contract | Fixed-point values use declared bit width and fractional-bit scale; clock in MHz, timestep in seconds, FIFO depth and neuron counts dimensionless, weights and thresholds quantised by explicit integer scale. | tests/test_fpga_export.py; tests/test_scpn_compiler.py | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/scpn/formal_verification.py` | Formal verification claims must declare the compiled Petri-net transition relation, exact bounded reachability depth, rational marking arithmetic, marking-bound safety obligations, algebraic place-invariant weights, transition liveness obligations, bounded temporal response and recurrence specifications, inhibitor-arc semantics, and counterexample path reporting. | Petri-net reachability and P-invariant analysis; bounded temporal logic over finite transition systems; repository SCPN compiler transition-relation contract | Markings are dimensionless token densities; arc weights and invariant weights are rationalised from finite decimal inputs; max_depth is a non-negative integer firing bound; temporal response windows count transition firings. | tests/test_scpn_formal_verification.py exact reachability, marking bounds, transition liveness, algebraic place invariants, all-path bounded response, recurrence, counterexample, and invalid-domain checks; src/scpn_control/scpn/formal_verification.py exact explicit-state finite transition relation over rational markings | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/vmec_lite.py` | VMEC-lite claims must declare flux-surface parameterisation, rotational-transform profile, Fourier mode truncation, field-period count, positive R00 and sampled major-radius boundary state, pressure/current assumptions, residual metric, geometry/profile/current provenance, reference-comparison tolerances for R_mn, Z_mn, and iota, convergence status, and explicit exclusion from full VMEC-grade 3D MHD equilibrium claims unless matched references pass the fail-closed admission gate. | VMEC 3D equilibrium references; stellarator flux-surface Fourier parameterisation references | SI metres, tesla, pascals, amperes, webers, dimensionless rotational transform, and Fourier coefficients with declared length units. | tests/test_vmec_lite.py verifies finite spectral reconstruction, bounded force residuals, physical domains, and fail-closed VMEC-lite claim evidence admission; validation/benchmark_vmec_lite_claims.py publishes deterministic bounded synthetic claim evidence with explicit full-VMEC exclusion; validation/reports/vmec_lite_claims.json records bounded Fourier, field-period, profile, residual, q-domain, and positive-major-radius evidence; validation/validate_vmec_reference.py strict VMEC reference artefact gate | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/control/advanced_soc_fusion_learning.py` | SOC turbulence-learning claims must declare the sandpile lattice, critical-gradient threshold, predator-prey zonal-flow coupling, shear-suppression term, bounded substep relaxation, Q-learning state discretisation, action set, and random policy assumptions. | Diamond and Hahm 1995 SOC turbulence reference; Kim and Diamond 2003 zonal-flow predator-prey coupling; Biglari, Diamond and Terry 1990 shear-suppression reference | Dimensionless lattice gradients, flow amplitudes, shear, toppling counts, Q-table values, reward, and RNG-seeded action choices. | tests/test_advanced_soc.py SOC physics and learning checks; tests/test_advanced_soc_verbose_plot.py verbose and plotting-path checks; tests/test_visualization_paths.py SOC visualisation checks; validation/validate_soc_reference.py strict reference-artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/current_drive.py` | Current-drive claims must declare ECCD, LHCD, and NBI source powers, radial deposition centres and widths, grid-normalised deposition conservation, density and temperature normalisation, Fisch-Boozer or Fisch efficiency coefficients, Prater launch-angle factor, Stix slowing-down time and critical energy for NBI, and the absence of ray-tracing or Fokker-Planck facility validation. | Fisch and Boozer 1980 electron-cyclotron current-drive efficiency; Prater 2004 ECCD launch-angle efficiency scaling; Fisch 1978 lower-hybrid current-drive efficiency; Stix 1972 neutral-beam slowing-down and critical-energy formulae; Ehst and Karney 1991 neutral-beam current-drive model; Repository fail-closed current-drive claim-admission contract | Power in MW or W with explicit conversion, rho dimensionless, density in 10^19 m^-3, temperatures and beam energy in keV, current density in A/m^2, total current in amperes, efficiency coefficients in declared normalised A/W form. | tests/test_current_drive.py ECCD, LHCD, and NBI deposition, scaling, source superposition, total-current integration, and claim-admission checks; validation/validate_current_drive_reference.py strict current-drive reference artifact gate; validation/benchmark_current_drive_claims.py bounded claim-admission benchmark | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/blob_transport.py` | Blob transport claims must declare interchange drive, sheath closure, radial velocity scaling, filament size, density and temperature perturbation assumptions, non-empty strictly ordered separatrix-to-wall scrape-off-layer profile coordinates, positive ordered detector-event domains, and scrape-off-layer validity bounds. | Stangeby 2000 scrape-off-layer transport references; blob-filament interchange transport scaling references | SI metres, seconds, m/s, density in m^-3, temperature in eV, magnetic field in tesla, and dimensionless normalised perturbations. | tests/test_blob_transport.py velocity-regime, size-scaling, ensemble, flux, strict-profile-grid, wall-flux, and detector-domain checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core` | Analytical approximations must state their source equation, retained terms, omitted terms, unit contract, and allowed parameter domain. | Cordey 1981 orbit-width estimate; Sauter et al. 1999 neoclassical fits; Stangeby 2000 SOL two-point model | Per-module SI or declared normalised units with explicit conversion boundaries. | docs/physics_methods.md simplification declarations; source grep for simplification and approximation markers | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control` | Control approximations must declare reduced-state dynamics, actuator assumptions, estimator assumptions, and facility-exclusion boundaries before any controller-readiness claim. | Tokamak control reduced-plant literature; Repository control safety and disruption contracts | Per-controller declared SI plasma, actuator, magnetic, force, and timing units. | src/scpn_control/control/gym_tokamak_env.py; src/scpn_control/control/free_boundary_tracking.py | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control` | Runtime approximations must state phase wrapping, fixed-point shift approximations, replay fixture provenance, and hardware export limits. | Kuramoto phase oscillator reference model; Repository fixed-point spiking export contract | Declared phase radians, tick timing, fixed-point scale factors, and replay geometry units. | src/scpn_control/phase/kuramoto.py; src/scpn_control/scpn/fpga_export.py; src/scpn_control/scpn/geometry_neutral_replay.py | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/control/mu_synthesis.py` | Mu-analysis claims must declare the structured uncertainty blocks, positive block bounds, Riccati state-feedback K-step, bound-scaled static closed-loop DC robust-performance map, D-scaling upper-bound fit, finite controller state/timestep domains, and the exclusion of full frequency-dependent H-infinity D-K synthesis unless an external validated backend is wired. | Doyle 1982 structured singular value definition; Balas et al. 1993 mu-analysis and synthesis toolbox; Skogestad and Postlethwaite 2005 multivariable feedback control; Repository fail-closed static mu-analysis claim-admission contract | State, control, output, and uncertainty units inherit the supplied state-space plant contract; mu, uncertainty bounds, and D-scalings are dimensionless. | tests/test_mu_synthesis.py D-scaled upper-bound behaviour, uncertainty-bound scaling, finite-domain rejection, closed-loop plant dependence, and claim-admission checks; validation/validate_mu_synthesis_reference.py strict mu-analysis reference artifact gate; validation/benchmark_mu_synthesis_claims.py bounded claim-admission benchmark; docs/learning/control_theory_primer.md states the bounded static analysis domain | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/checkpoint.py` | Checkpoint claims must declare serialised solver state, non-negative episode counter, object-valued metrics schema, finite-value policy, schema-version boundary, and fail-closed handling for corrupt or missing checkpoint artefacts. | repository restart and replay provenance contract | Units are inherited from stored state and metrics; checkpoint metadata must preserve schema version, episode index, and provenance boundaries. | tests/test_checkpoint.py round-trip, resume, finite-payload, schema-version, and corrupt-payload checks | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/control/density_controller.py` | Density-control claims must declare the Greenwald limit, ITER operating margin, radial particle transport grid, gas-puff source, pellet source, NBI source, cryopump sink, recycling source, controller gains, CFL-limited explicit transport step, geometry/transport/actuator/diagnostic provenance, source integral, particle-inventory change, and matched Greenwald-fraction and inventory reference tolerances before facility-calibrated density-control claims. | Greenwald 2002 density limit; ITER Physics Basis 1999 density operating margin; Parks and Turnbull 1978 neutral gas shielding pellet-ablation model; Milora 1995 neutral gas shielding pellet-ablation model | SI particles per second, metres, seconds, m^-3, m^2/s, m/s, pellet millimetres, beam keV, megawatts, and dimensionless Greenwald fraction. | tests/test_density_controller.py Greenwald, NGS pellet trajectory deposition, sink, controller, geometry, transport-profile, source-input, timestep guard, and fail-closed density-control claim evidence checks; tests/test_pellet_injection.py Parks-Turnbull NGS ablation and pellet trajectory checks; validation/benchmark_density_control_claims.py publishes deterministic bounded synthetic claim evidence with explicit facility-calibrated claim exclusion; validation/reports/density_control_claims.json records bounded geometry, transport, actuator, diagnostic, CFL, Greenwald, source-integral, and inventory evidence; validation/validate_density_reference.py strict density reference artefact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/differentiable_transport.py` | Differentiable transport claims must declare the four-channel profile order, cylindrical Crank-Nicolson diffusion step, uniform normalised radial grid, source-term convention, core zero-gradient boundary, edge Dirichlet boundary, transport-coefficient gradients, one-step and multi-step source-schedule gradient targets, sampled finite-difference gradient audit, audited one-step and rollout gradient-admission latency metadata, neural and reduced-gyrokinetic closure coefficient mapping, campaign provenance metadata, schema-versioned replay metadata, and optional one-step and rollout Grad-Shafranov flux-map radial weighting contract. | repository JAX transport primitive contract; repository integrated transport solver contract | Profiles inherit channel units for electron temperature, ion temperature, electron density, and impurity density; chi uses m^2/s or the declared normalised diffusivity convention; sources use profile units per second; rho is dimensionless and uniformly spaced. | tests/test_differentiable_transport.py diffusion, boundary, fallback, optional JAX coefficient-gradient, one-step and multi-step source-schedule-gradient, sampled finite-difference gradient audit, neural and reduced-GK closure coefficient mapping, campaign metadata provenance, schema-versioned replay metadata, replay-drift guard, and one-step plus rollout equilibrium-weighted GS-flux checks; tests/test_nmpc_controller.py NMPC neural-closure, one-step source-schedule, and multi-step source-rollout tuning fail closed without JAX and preserve gradient-audit admission results; tests/test_nmpc_controller.py NMPC coefficient, one-step source-schedule, and multi-step source-rollout tuning results carry campaign metadata, closure provenance, finite source bounds, and fail-closed gradient-audit evidence; validation/benchmark_differentiable_transport_latency.py audited one-step and rollout gradient-admission latency benchmarks or fail-closed JAX-unavailable reports; validation/reports/differentiable_transport_latency.json and validation/reports/differentiable_transport_rollout_latency.json bounded latency reports or blocked-backend status | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/digital_twin_online_update.py` | Online model-update claims must declare tunable bounded parameters, external TRANSP/TSC simulator artifact provenance, target summary metrics, tolerances, deterministic random seed, Gaussian-process acquisition settings, and fail-closed behavior when external artifacts are absent or malformed. | TRANSP integrated modelling evidence contract; TSC time-dependent simulation evidence contract; Bayesian optimisation for bounded model calibration; Repository digital-twin runtime contract | Tunable density in m^-3, effective charge dimensionless, actuator lag in steps, actuator rate limit dimensionless per step, target metrics in declared digital-twin summary units, and simulator time base in seconds. | tests/test_digital_twin_online_update.py external artifact, loss, Bayesian-update, and deterministic benchmark checks; tests/test_digital_twin_reference_validation.py strict digital-twin artifact gate including TRANSP and TSC; validation/benchmark_digital_twin_online_update.py deterministic bounded online-update benchmark | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/free_boundary_tracking.py` | Free-boundary tracking claims must declare direct kernel-in-the-loop coil-response identification, bounded least-squares correction, actuator lag, slew limits, supervisor rejection, measurement bias, drift, latency, and observer compensation assumptions. | Grad-Shafranov free-boundary control references; Repository FusionKernel free-boundary objective contract; Repository deterministic free-boundary acceptance campaign; Repository fail-closed free-boundary claim-admission contract | SI coil currents, metres, webers per radian, seconds, amperes per second, objective-space residuals, and dimensionless supervisor gains. | tests/test_free_boundary_tracking.py controller, disturbance-observer, and claim-admission checks; tests/test_free_boundary_tracking_variants.py actuator, observer, latency, and supervisor edge-path checks; tests/test_free_boundary_tracking_acceptance.py deterministic acceptance campaign; validation/validate_free_boundary_reference.py strict free-boundary reference artifact gate; validation/benchmark_free_boundary_tracking_claims.py bounded claim-admission benchmark | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/disruption_contracts.py` | Disruption-contract claims must declare synthetic disruption signal generation, toroidal-mode amplitudes, mitigation-cocktail coupling, impurity transport response, halo/runaway post-disruption response, TBR equivalence scaling, and RL action bias assumptions. | Pautasso et al. 2017 disruption current-quench constraints; Riccardo et al. 2010 halo-current rise-time references; Abdou et al. 2015 blanket neutronics calibration references | SI seconds, milliseconds, mega-amperes, megajoules, moles, megawatts, dimensionless risk, toroidal mode amplitudes, and tritium breeding ratio. | tests/test_disruption_contracts.py contract smoke checks; tests/test_disruption_contracts_pure.py pure physics-path checks; tests/test_disruption_edge_cases.py edge-case disruption checks; validation/validate_disruption_reference.py strict disruption reference artifact gate | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/disruption_sequence.py` | Disruption-sequence claims must declare phase ordering, finite positive configuration domains, Rechester-Rosenbluth thermal-quench transport, post-quench radiative-cooling exposure, current-quench timing, mitigation action coupling, runaway-electron beam phase, stochastic event boundaries, halo-force convention, and replay provenance. | ITER disruption mitigation sequence references; repository disruption phase-state contract | Time in seconds or milliseconds with explicit convention, current in amperes or mega-amperes, energy in joules or megajoules, magnetic field in tesla, geometry in metres, density in 10^20 m^-3, and dimensionless phase labels. | tests/test_disruption_sequence.py post-quench cooling, phase ordering, current quench, runaway, and halo-force checks; tests/test_disruption_safe_api.py | validation_gap | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/gk_interface.py` | Generated input decks and parsed outputs must round-trip through real TGLF, GENE, GS2, CGYRO, or QuaLiKiz executables. | TGLF Staebler et al. 2007; GENE Jenko et al. 2000; GS2 Kotschenreuther et al. 1995; CGYRO Candy et al. 2016; QuaLiKiz Bourdelle et al. 2007 | Code-specific flux, growth-rate, frequency, and geometry units converted into repository normalisation with explicit metadata. | docs/joss_paper.md external GK limitation; validation/validate_gk_interface_artifacts.py strict external interface artifact gate | external_dependency_blocked | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/federated_disruption.py` | FedAvg/FedProx disruption classifiers must federate per-facility client updates without centralising raw arrays, validate the shared eight-feature disruption contract, clip and noise facility model deltas under a declared Gaussian differential-privacy budget, and preserve a serialisable privacy ledger. | McMahan et al. 2017 Communication-Efficient Learning of Deep Networks from Decentralized Data; Li et al. 2020 FedProx; Dwork and Roth 2014 Algorithmic Foundations of Differential Privacy; Abadi et al. 2016 Deep Learning with Differential Privacy | Eight disruption features use declared tokamak control variables: Ip, beta_N, q95, n/n_GW, li, dBp/dt, locked-mode amplitude, and n=1 RMS; labels are binary disruption indicators. | tests/test_federated_disruption.py module-specific federation, array-ingestion, DP-ledger, and benchmark contract tests; validation/benchmark_federated_disruption.py deterministic synthetic multi-facility benchmark; validation/reports/federated_disruption_benchmark.json synthetic report with explicit claim boundary | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/uncertainty.py` | Uncertainty claims must declare sampled variables, distributions, correlations, random seed, sample count, propagation chain, convergence and percentile-ordering criteria, finite-value rejection policy, D-T fuel dilution handling, sensitivity outputs, scenario/prior/sensitivity provenance, and matched central-value and sigma reference tolerances before calibrated predictive-UQ claims. | Monte Carlo uncertainty propagation references; repository fusion-performance uncertainty contract | Units inherit each propagated physical quantity; distribution parameters must preserve SI or declared normalised units and dimensionless uncertainty fractions. | tests/test_uncertainty.py verifies IPB98 monotonicity, Bosch-Hale domains, D-T fuel dilution, full-chain sampling, percentile ordering, finite-output checks, and fail-closed UQ claim evidence admission; tests/test_full_chain_uq.py full-chain UQ behavioural coverage; tests/test_uncertainty_sigma_guard.py sigma guard coverage; validation/benchmark_uq_claims.py publishes deterministic bounded synthetic claim evidence with explicit calibrated-UQ exclusion; validation/reports/uq_claims.json records bounded scenario, prior, seed, sample-count, percentile, finite-output, sensitivity, and claim-admission evidence; validation/validate_uncertainty_reference.py strict UQ reference artefact gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/scpn/geometry_neutral_replay.py` | Geometry-neutral replay claims must declare synthetic W7-X-like fixture provenance, field-line spread metric, actuator current bounds, latency model, stuck-actuator fault schedule, controller feature mapping, and replay acceptance thresholds. | Repository geometry-neutral control contract; W7-X-like reduced-order stellarator replay fixture | Field-line spread in radians, currents in amperes, timestep in seconds, latency in microseconds, effective ripple dimensionless, controller objectives and thresholds declared per replay manifest. | tests/test_geometry_neutral_replay.py; tests/test_geometry_neutral_contracts.py | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gk_ood_detector.py` | OOD detector claims must declare the feature vector, training distribution, distance metric, symmetric positive-definite inverse-covariance calibration, threshold calibration, uncertainty handling, non-negative transport ensemble predictions, and behavior outside the calibrated gyrokinetic operating envelope. | Repository gyrokinetic scheduler OOD contract; statistical process monitoring distribution-shift controls | Feature units inherit declared GK inputs and outputs; detector scores are dimensionless with explicit calibration metadata, Mahalanobis metric provenance, threshold provenance, and transport prediction channels in non-negative diffusivity units. | tests/test_gk_ood_detector.py; tests/test_gk_hybrid_integration.py; validation/validate_gk_ood_calibration.py strict persisted campaign calibration gate | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/gk_online_learner.py` | Online learner claims must declare training-window selection, non-empty train/validation split domains, nonnegative transport-coefficient targets, validation loss, rollback policy, OOD-score sample admission, optimiser settings, persisted retraining decisions, and compatibility boundary with gyrokinetic scheduler inputs. | online learning stability and rollback control references; repository gyrokinetic hybrid learner contract | Inputs inherit GK feature units; transport targets are nonnegative chi_e, chi_i, and D_e; losses and OOD scores are dimensionless with explicit scaling, learning rate, epoch count, generation limit, and threshold metadata. | tests/test_gk_online_learner.py OOD admission, holdout retraining, rollback, decision persistence, and invalid-domain checks; tests/test_gk_hybrid_integration.py scheduler integration checks; validation/benchmark_gk_online_learner.py deterministic bounded online-retraining benchmark | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/gk_species.py` | Species and collision claims must declare charge, mass, density, temperature, thermal speed, Larmor radius, gyroaverage Bessel evaluation, diamagnetic frequency, pitch-angle deflection coefficient, thermal energy-relaxation coefficient, electron-ion mass-ratio scaling, field-temperature dependence, valid species parameter bounds, valid velocity quadrature sizes, and strictly ordered finite lambda grids bounded by the local trapped-passing boundary. | Sugama et al. 2006 collision operator; gyrokinetic normalisation and species parameter contracts | Mass in kilograms, charge in coulombs, density in m^-3, temperature in electronvolts, velocity in m/s, gyrofrequency in rad/s, Larmor radius in metres, collision rates in s^-1, lambda on [0, 1] with lambda times B/B0 no greater than one, and positive magnetic-field ratio. | tests/test_gk_species.py thermal-speed, Larmor-radius, Bessel gyroaverage, diamagnetic-frequency sign, collision-frequency scaling, velocity-grid, and pitch-angle-domain checks; tests/test_gk_electromagnetic.py; validation/validate_gk_species_reference.py immutable species and collision reference cases with distinct deflection and energy-relaxation channels | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/halo_re_physics.py` | Post-disruption claims must declare the halo L/R circuit, contact fraction, toroidal peaking factor, current-quench waveform, Connor-Hastie primary generation, Rosenbluth-Putvinski avalanche generation, ensemble uncertainty assumptions, ITER-limit summaries, strict disruption-reference admission, and claim status. | Fitzpatrick 2002 halo current and error-field interaction; Connor and Hastie 1975 runaway electron generation; Rosenbluth and Putvinski 1997 avalanche generation | SI amperes, mega-amperes, seconds, milliseconds, ohms, henries, metres, volts per metre, rates per second, and meganewtons per metre. | tests/test_halo_re_physics.py halo-current, runaway-electron, ensemble, fail-closed claim evidence, and strict reference-admission regression checks; tests/test_halo_nonfinite_guards.py non-finite guard checks; tests/test_halo_validation_paths.py validation-path guard checks; validation/benchmark_disruption_mitigation_claims.py deterministic bounded halo/runaway claim-admission benchmark; validation/validate_disruption_reference.py strict disruption reference artifact gate | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/stability_mhd.py` | MHD stability claims must declare q-profile interpolation, magnetic shear, interior-point radial grids for profile-resolved criteria, Mercier and Troyon criteria, beta-limit inputs, first-unstable-radius search, bounded bootstrap-current fraction, and the exclusion boundary for full ideal-MHD or resistive-MHD eigenmode claims. | Troyon beta-limit reference; Mercier ideal-MHD stability criterion; tokamak q-profile and magnetic-shear references | q and beta dimensionless, plasma current in MA or A with explicit convention, minor radius and major radius in metres, magnetic field in tesla, rho dimensionless, and local bootstrap-current fraction within [0, 1]. | tests/test_stability_mhd.py; tests/test_ballooning_solver.py; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/integrated_scenario.py` | Integrated scenario claims must declare coupling order, current diffusion, transport, pedestal, ELM, NTM, sawtooth, burn, and control interactions, timestep policy, state exchange units, and failure isolation boundaries. | integrated tokamak scenario modelling references; repository scenario-coupling contract | All coupled states must preserve declared SI or normalised units for current, density, temperature, pressure, q, beta, power, flux, and timing. | tests/test_integrated_scenario.py; tests/test_cross_module_physics.py; validation/benchmark_integrated_scenario_coupling.py; validation/reports/integrated_scenario_coupling.json; validation/reports/integrated_scenario_coupling.md | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/kinetic_efit.py` | Kinetic-EFIT coupling claims must declare Thomson or profile-derived electron density and temperature points, ion-temperature points, fast-ion pressure fraction, anisotropy sigma, MSE pitch-angle constraints when used, radial interpolation geometry, pressure-consistency residual, diagnostic/profile/fast-ion/MSE source provenance, matched pressure, q-profile, and anisotropy reference tolerances, and exclusion from facility-grade P-EFIT unless matched reference equilibria pass the fail-closed admission gate. | Lao et al. 1985 EFIT equilibrium reconstruction; MSE-constrained kinetic EFIT workflow; Repository fixed-boundary realtime-EFIT contract | R and Z in metres, temperatures in keV, density in 10^19 m^-3, MSE pitch angle in degrees, pressure in pascals, beta dimensionless, and q-profile dimensionless. | tests/test_kinetic_efit.py requires measured profile channels, radial interpolation, measured Ti use, anisotropy residuals, MSE q-profile response, fast-ion pressure checks, and fail-closed kinetic-EFIT claim evidence admission; validation/benchmark_kinetic_efit_claims.py publishes deterministic bounded synthetic claim evidence with explicit facility-claim exclusion; validation/reports/kinetic_efit_claims.json records bounded provenance, interpolation geometry, pressure consistency, fast-ion beta, and q-profile endpoints | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gk_eigenvalue.py` | Native linear GK eigenvalue solve must match external TGLF, GENE, GS2, CGYRO, or QuaLiKiz growth rates and real frequencies for the same Miller geometry and species inputs. | Miller et al. 1998 local equilibrium; GENE and GACODE published input-output contracts | Growth rate and frequency normalised consistently to c_s/a or declared external-code convention. | ROADMAP.md local-dispersion overprediction note; docs/competitive_analysis.md linear GK quantitative accuracy status; validation/validate_gk_crosscode.py strict real-binary evidence gate | external_dependency_blocked | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/momentum_transport.py` | Momentum transport claims must declare NBI torque, collisional damping, viscous momentum diffusion, rotation-profile boundary conditions, angular-momentum units, and the coupling boundary to the integrated transport solver. | tokamak toroidal momentum transport references; repository NBI torque and rotation-profile contract | SI newton metres, kg m^2/s^2, rad/s, metres, seconds, density in m^-3, and momentum diffusivity in m^2/s. | tests/test_momentum_transport.py; tests/test_momentum_integration.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/neural_equilibrium.py` | Neural equilibrium pretraining may use bounded synthetic Solovev-like equilibria, but fine-tuning and public predictive claims require fail-closed claim evidence tied to the same weight checksum and comparison against identical P-EFIT or documented public reference equilibria for psi, pressure, q-profile, LCFS boundary, and magnetic-axis position. | EFIT/P-EFIT equilibrium reconstruction workflow; Repository neural equilibrium model contract; Bounded Solovev-like synthetic equilibrium pretraining contract | SI magnetic flux, pressure, metre-scale geometry, and dimensionless q-profile arrays on declared grids. | tests/test_neural_equilibrium.py synthetic pretraining, reproducibility, JAX-compatible weight, fail-closed claim evidence, checksum matching, and real-EFIT admission tests; tests/test_jax_neural_equilibrium.py synthetic-pretrained weight loading through JAX inference; validation/benchmark_neural_equilibrium_pretraining.py deterministic synthetic pretraining and claim-admission benchmark; validation/validate_neural_equilibrium_reference.py strict P-EFIT/reference artifact gate | external_dependency_blocked | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/neural_transport.py` | Neural transport claims must declare input feature normalisation, QLKNN weight provenance, prediction targets, fallback critical-gradient thresholds, bounded density-channel particle diffusivity, bounded profile closure provenance, local benchmark errors, strict reference-artifact admission, weight checksum matching, uncertainty output, out-of-domain handling, and cross-validation against reference transport cases. | QuaLiKiz neural network transport surrogate references; repository QLKNN weight and metric contract | Inputs and outputs use declared transport feature units, diffusivity in m^2/s or declared normalisation, fluxes in SI or gyro-Bohm units with conversion metadata. | tests/test_neural_transport.py density-gradient and shear-dependent fallback particle diffusivity; tests/test_neural_transport_core.py profile fallback particle-channel checks, bounded closure provenance, fallback gating, fail-closed claim evidence, and checksum-matched reference admission checks; tests/test_qlknn_transport.py; validation/benchmark_neural_transport_claims.py deterministic local claim-admission benchmark; validation/validate_neural_transport_reference.py strict QuaLiKiz/reference artifact gate | external_dependency_blocked | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/neural_turbulence.py` | Neural turbulence claims must declare QLKNNSurrogate inputs, finite strictly ordered physical profile grids, feature scaling, banana-regime electron collisionality, bounded analytic quasilinear target variables, fallback behaviour, uncertainty handling, local analytic-target benchmark errors, strict reference-artifact admission, exact weight checksum matching, and cross-validation boundary against gyrokinetic or quasilinear turbulence references. | QLKNN turbulence surrogate references; repository neural turbulence surrogate contract | Feature and target units follow the declared turbulence surrogate schema; diffusivities, growth rates, and fluxes require explicit SI or normalised-unit metadata. | tests/test_neural_turbulence.py collisionality scaling, analytic target, training, save-load, denormalisation, fail-closed claim evidence, and checksum-matched reference admission checks; validation/benchmark_neural_turbulence_claims.py deterministic local claim-admission benchmark; validation/validate_neural_turbulence_reference.py strict GK-campaign/reference artifact gate | validation_gap | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/gk_nonlinear.py` | Five-dimensional delta-f flux-tube Vlasov evolution with E cross B bracket, ballooning connection, kinetic electrons, and Sugama collision terms. | Dimits et al. 2000 Cyclone Base Case; Sugama et al. 2006 collision operator | Gyro-Bohm-normalised heat flux and chi_i with explicit R/L_Ti, rho_s, c_s, and reference gradient normalisation. | ROADMAP.md v0.18.0 open revalidation item; docs/joss_paper.md nonlinear GK validation limitation; validation/gk_nonlinear_cyclone.py saturation-evidence assessor | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/orbit_following.py` | Orbit-following claims must declare guiding-centre equations, magnetic moment handling, finite pitch-angle bounds, banana-orbit width scaling, collision or loss assumptions, ensemble sampling, valid magnetic-geometry bounds, geometry/particle/collision/loss-boundary provenance, and matched banana-width and loss-fraction reference tolerances before external orbit-code or measured fast-ion claims. | Cordey 1981 orbit-width estimate; tokamak guiding-centre orbit references | SI metres, seconds, tesla, electronvolts, kg, coulombs, pitch angle in radians on [0, pi], and ensemble probabilities dimensionless. | tests/test_orbit_following.py verifies guiding-centre domain checks, classifier boundaries, Stix slowing-down, banana-width scaling, first-orbit-loss scaling, ensemble counts, and fail-closed orbit-following claim evidence admission; validation/benchmark_orbit_following_claims.py publishes deterministic bounded synthetic claim evidence with explicit external-code claim exclusion; validation/reports/orbit_following_claims.json records bounded geometry, particle, collision, loss-boundary, banana-width, first-orbit-loss, and ensemble-count evidence; validation/validate_orbit_reference.py strict orbit reference artefact gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/realtime_efit.py` | Real-time equilibrium reconstruction claims must declare the fixed-boundary Grad-Shafranov solve, polynomial p-prime and FF-prime source profiles, diagnostic response interpolation, Rogowski current reconstruction, macroscopic-shape extraction assumptions, diagnostic provenance, matched-reference source, psi/Ip/q95/beta_pol/li tolerances, and facility-claim admission status. | Lao et al. 1985 EFIT equilibrium reconstruction; Strait et al. 2019 real-time EFIT workflow references; Repository GEQDSK and magnetic-diagnostic validation contracts | SI metres, webers per radian, tesla, amperes, pascals per weber, FF-prime units, and dimensionless q95, beta_pol, li, elongation, and triangularity. | tests/test_realtime_efit.py fixed-boundary solve and diagnostic reconstruction checks; tests/test_kinetic_efit.py kinetic EFIT integration checks; tests/test_validate_real_shots_equilibrium.py GEQDSK source residual checks; tests/test_realtime_efit.py claim-evidence provenance, matched-reference admission, and invalid-reference checks; validation/benchmark_efit_lite_claims.py bounded synthetic EFIT-lite claim-admission benchmark; validation/reports/efit_lite_claims.json bounded synthetic EFIT-lite claim report | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gyrokinetic_transport.py` | Reduced gyrokinetic transport claims must declare the dispersion relation, instability branch classification, quasilinear heat-flux closure, critical-gradient threshold, saturation rule, positive integer mode-count domain, tokamak geometry ordering, and mapping from local GK outputs to transport coefficients. | Cyclone Base Case gyrokinetic benchmark; quasilinear gyrokinetic transport closure references | Growth rates and frequencies in declared normalised units, gradients dimensionless or m^-1 with explicit convention, rho and epsilon dimensionless, geometry in metres, heat flux in gyro-Bohm or SI units with conversion metadata. | tests/test_gyrokinetic_transport.py dispersion branches, quasilinear fluxes, saturation monotonicity and boundedness, profile evaluation, and invalid-domain checks; tests/test_gk_benchmark_linear.py | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/gym_tokamak_env.py` | 0D/1D control plants must declare reduced-state equations, conservation assumptions, and controller-validity bounds before being used for hardware or facility claims. | Wesson 2011 tokamak transport references; Repository Gymnasium plant and digital-twin contracts | Declared plasma current, temperature, density, confinement, and actuator units per state vector field. | README.md validation limitation; docs/use_cases.md synthetic disruption prediction status; tests/test_gym_tokamak_env.py reduced-order plant bounds and action-contract checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/rwm_feedback.py` | RWM feedback claims must declare the no-wall and ideal-wall beta limits, resistive-wall L/R time, optional wall-plasma gap correction, rotation-stabilisation term, active proportional or derivative feedback coupling, controller latency, coil coupling, sensor/coil topology, ideal-kink exclusion boundary, reference source, closed-loop growth-rate tolerance, and facility-claim admission status. | Bondeson and Ward 1994 resistive-wall-mode growth-rate model; Fitzpatrick 2001 rotation-stabilisation contribution; Strait et al. 2003 wall-plasma gap correction; Garofalo et al. 2002 active feedback experiments | Normalised beta dimensionless, wall time in seconds, toroidal rotation in rad/s, wall and plasma radii in metres, feedback gain dimensionless per declared coil coupling, growth rate in s^-1. | tests/test_rwm_feedback.py growth-rate, rotation, wall-geometry, feedback, ideal-kink, and required-gain checks; tests/test_rwm_feedback.py claim-evidence provenance, external tolerance admission, and invalid-domain checks; validation/benchmark_rwm_claims.py bounded local RWM claim-admission benchmark; validation/reports/rwm_claims.json bounded local RWM claim report | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/tearing_mode_coupling.py` | Sawtooth-to-NTM seeding claims must declare sawtooth crash trigger, seed-island generation, rational-surface coupling, phase and amplitude assumptions, bounded bootstrap-current fraction, NTM coupling path, and exclusion of full nonlinear MHD crash dynamics. | sawtooth crash and NTM seeding references; repository tearing-mode coupling contract | Time in seconds, island width in metres, q dimensionless, rho dimensionless, phase in radians, local bootstrap-current fraction within [0, 1], and amplitude units declared per coupling signal. | tests/test_tearing_mode_coupling.py; tests/test_integrated_scenario.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/tokamak_digital_twin.py` | Digital-twin claims must declare the 2D poloidal grid, q-profile topology update, rational-surface island mask, finite-difference diffusion, radiation-loss scaling with density and effective charge, actuator latency, RNG seed, IDS export assumptions, external simulator artifact provenance, and bounded online model-update parameters. | Repository digital-twin runtime contract; Wesson 2011 radiation-loss reference; TRANSP integrated modelling evidence contract; TSC time-dependent simulation evidence contract; Bayesian optimisation for bounded model calibration | Declared normalised grid indices, keV, m^-3, dimensionless q-profile values, actuator-lag summary units, RNG seed metadata, and IDS-compatible pulse history fields. | tests/test_tokamak_digital_twin.py deterministic twin checks; tests/test_digital_twin_physics.py physics-path checks; tests/test_digital_twin_online_update.py external artifact, loss, Bayesian-update, and deterministic benchmark checks; tests/test_digital_twin_reference_validation.py strict digital-twin artifact gate including TRANSP and TSC; validation/benchmark_digital_twin_online_update.py deterministic bounded online-update benchmark; validation/validate_digital_twin_reference.py strict reference-artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/integrated_transport_solver.py` | Integrated transport claims must declare the axis-to-edge radial grid, heat and particle diffusion equations, neoclassical closure, bootstrap-current fit, source deposition terms, boundary conditions, timestep limits, and coupling contracts to pedestal and gyrokinetic closures. | Chang-Hinton neoclassical transport model; Sauter et al. 1999 bootstrap-current fits; repository integrated transport solver contract | SI metres, seconds, m^-3, eV, W/m^3, m^2/s, amperes per square metre, and dimensionless profile coordinates with explicit axis-to-edge normalisation metadata. | tests/test_integrated_transport_solver.py; tests/test_transport_energy_conservation.py; tests/test_transport_neoclassical_guards.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/volt_second_manager.py` | Volt-second management claims must declare inductive flux, resistive loop-voltage integration, Ejima startup flux, bootstrap-current correction, flat-top duration estimate, ramp and ramp-down decomposition, flux-budget margin assumptions, finite positive machine constants, nonnegative current and voltage domains, positive timesteps, and strictly ordered bootstrap-profile grids. | Wesson 2011 tokamak loop-voltage and flux-balance equations; Ejima et al. 1982 startup flux coefficient; ITER Physics Basis 1999 flat-top flux-budget references; Repository fail-closed volt-second claim-admission contract | SI webers, volt-seconds, henries, ohms, amperes, mega-amperes, seconds, metres, normalised profile radius, and dimensionless Ejima coefficient. | tests/test_volt_second_manager.py flux budget, scenario analysis, bootstrap-profile, domain-boundary, and claim-admission checks; validation/validate_volt_second_reference.py strict volt-second reference artifact gate; validation/benchmark_volt_second_claims.py bounded claim-admission benchmark | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |

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
  - Attach strict documented public, integrated-transport benchmark, or measured burn replay artifacts before reactor-control claims
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

### SCPN formal Petri-net verification contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/scpn/formal_verification.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Persist formal verification reports with every safety-critical compiled controller artifact
  - Admit safety-critical `.scpnctl` artifacts only through the bounded formal-verification manifest gate
  - Validate certification packages with an external safety-case review before hardware-control claims
  - Publish optional Z3 bounded-model-checking artifacts with every SMT-backed proof obligation
  - Treat missing `z3-solver` as blocked SMT evidence, not as a successful proof

### VMEC-lite stellarator equilibrium approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/vmec_lite.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Validate surface geometry and rotational-transform outputs against VMEC or published stellarator benchmark equilibria before full VMEC claims
  - Persist Fourier truncation, field periods, pressure profile, iota profile, current assumption, sampled major-radius bounds, residual tolerance, reference errors, and admission status for every promoted VMEC-lite claim

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
  - Attach strict ray-tracing, Fokker-Planck, measured deposition, or documented public artifacts before external current-drive claims
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
  - Attach strict documented public, external mu-toolbox, or measured control replay artifacts before validated robust-control claims
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
  - Validate Greenwald fraction and particle inventory changes against measured discharge, external particle-balance, or facility replay references before facility-calibrated claims
  - Persist geometry, transport profiles, actuator limits, diagnostic source, timestep/CFL state, source integrals, particle inventory deltas, reference tolerances, and claim-admission status with every promoted density-control claim

### differentiable transport facade contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/differentiable_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Wire each remaining legacy transport surface through an equivalent differentiable path before claiming end-to-end transport autodiff
  - Validate differentiable transport profiles and equilibrium-weighted gradients against measured discharges or published integrated-modelling benchmarks
  - Persist backend, dtype, radial-grid, boundary-condition, rollout-length, equilibrium-grid, flux-weighting, gradient-tolerance, and replay-drift metadata for controller-tuning campaigns
  - Replace bounded reduced-gyrokinetic and JAX GK stiffness closures with externally validated GK transport coefficients before quantitative transport-control claims

### digital twin online model updating

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/digital_twin_online_update.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Provide validated TRANSP, TSC, measured-discharge, or documented public-reference artifacts before measured replay claims
  - Validate online update trajectories against replayed or measured discharge histories before deployment use
  - Persist simulator artifact hashes, units, case ids, and strictly increasing time bases for every promoted update campaign

### direct free-boundary tracking controller

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/free_boundary_tracking.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Attach strict documented public, measured-replay, or external equilibrium benchmark artifacts before facility-control claims
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

### federated disruption prediction

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/federated_disruption.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Supply measured multi-facility disruption shot databases with immutable provenance manifests before claiming cross-facility predictive validation
  - Run the same federation and differential-privacy contracts against those measured facility datasets before deployment claims

### full-chain uncertainty quantification contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/uncertainty.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Calibrate uncertainty priors against measured or published scenario ensembles before predictive facility-UQ claims
  - Persist random seeds, distribution parameters, sample counts, percentile ordering, finite-output checks, convergence diagnostics, sensitivity metrics, reference tolerances, and claim-admission status with every UQ artefact

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
  - Persist model weights, rollback decisions, validation metrics, OOD thresholds, and source provenance for every online update

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
  - Persist ensemble seed, run count, halo/runaway P95 metrics, toroidal-peaking-factor summary, reference source, tolerances, errors, and admission status before promoted mitigation claims

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
  - Persist diagnostic source, interpolation geometry, fast-ion model provenance, MSE calibration, uncertainty, reference tolerances, and claim-admission status for every promoted kinetic-equilibrium claim

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
  - Run fine-tuning only through the strict neural-equilibrium reference-artifact gate
  - Persist claim evidence with model identity, weight checksum, grid shape, feature contract, reference source, reference-equilibria count, tolerances, errors, and admission status before predictive claims
  - Add error bounds for psi, pressure, q-profile, LCFS, and magnetic-axis predictions before facility claims

### neural transport surrogate validation contract

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Acquire or generate immutable reference QLKNN or transport benchmark cases for cross-validation
  - Record weight checksums, QLKNN-10D feature schema, fallback density-channel coefficients, closure source metadata, uncertainty metadata, OOD decisions, reference metrics, tolerances, and admission status with every surrogate evaluation
  - Gate quantitative QuaLiKiz, QLKNN, or documented-reference transport claims on strict reference artifacts that match the exact neural weight checksum

### neural turbulence surrogate validation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/neural_turbulence.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Validate surrogate outputs against immutable gyrokinetic or QuaLiKiz turbulence benchmark cases
  - Persist feature schema, weight checksum, local Q_i/Q_e/Gamma_e errors, critical-gradient accuracy, reference source, tolerances, and claim-admission status before quantitative turbulence claims
  - Gate quantitative gyrokinetic, QuaLiKiz, or documented-reference turbulence claims on strict reference artifacts that match the exact neural weight checksum

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
  - Validate orbit widths and loss fractions against an independent orbit-following code or published benchmark cases before external orbit-code claims
  - Persist particle species, pitch, energy, geometry, ensemble seed or counts, collision model, wall/loss boundary, reference tolerances, and claim-admission status with each validation artefact

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
  - Provide validated TRANSP, TSC, measured-discharge, or documented public-reference artifacts before measured replay claims
  - Replace normalised 2D diffusion dynamics with equilibrium and transport-calibrated state evolution before facility twin claims
  - Validate online update trajectories against replayed or measured discharge histories before deployment use

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
  - Attach strict documented public, measured loop-voltage replay, or external scenario benchmark artifacts before scenario-duration claims
  - Replace bootstrap-current proxy with neoclassical or transport-solver evidence before facility extrapolation
