# Physics Traceability and Bounded Claims

This report is generated from `validation/physics_traceability.json`.
It blocks full-fidelity public claims for entries whose evidence status is still open or bounded.

## Summary

- Status: pass
- Registry entries: 64
- Open fidelity gaps: 64
- Full-fidelity public claims blocked: 64
- Resolved module paths: 64
- Resolved evidence paths: 595
- External validation trackers: 8
- Source marker coverage: 37/37

## External Validation Collaboration Trackers

- External validation artefacts needed for full-fidelity SCPN-CONTROL claims: [#46](https://github.com/anulum/scpn-control/issues/46) — 0 open claim(s) — Parent tracker for external code, reference data, facility replay, benchmark, and hardware evidence requests.
- External gyrokinetic validation artefacts: [#47](https://github.com/anulum/scpn-control/issues/47) — 9 open claim(s) (external_dependency_blocked=2, validation_gap=7) — TGLF, GENE, GS2, CGYRO, QuaLiKiz, nonlinear CBC, Miller geometry, species, JAX parity, OOD, and online-learning artefacts.
- Equilibrium and reconstruction reference artefacts: [#48](https://github.com/anulum/scpn-control/issues/48) — 7 open claim(s) (validation_gap=3, bounded_model=4) — DIII-D or equivalent shots, EFIT, P-EFIT, GEQDSK, VMEC, and stellarator replay artefacts.
- Transport, edge, MHD, and scenario benchmark artefacts: [#49](https://github.com/anulum/scpn-control/issues/49) — 18 open claim(s) (validation_gap=12, bounded_model=6) — Integrated transport, momentum, pedestal, ELM, MARFE, NTM, current drive, stability, tearing, SOL, orbit, UQ, and scenario benchmarks.
- Neural surrogate validation artefacts: [#50](https://github.com/anulum/scpn-control/issues/50) — 3 open claim(s) (external_dependency_blocked=2, validation_gap=1) — QLKNN, QuaLiKiz, gyrokinetic, transport, turbulence, and equilibrium surrogate reference datasets and weight provenance.
- Plasma-control and facility replay artefacts: [#51](https://github.com/anulum/scpn-control/issues/51) — 15 open claim(s) (validation_gap=1, bounded_model=14) — RZIP, RWM, free-boundary, density, digital twin, SOC learning, burn, volt-second, mu-synthesis, and reduced-plant replay evidence.
- Disruption, halo-current, and mitigation benchmark artefacts: [#52](https://github.com/anulum/scpn-control/issues/52) — 6 open claim(s) (bounded_model=6) — Measured disruption databases, labelled disruption windows, halo/runaway envelopes, wall contact, impurity radiation, and mitigation metadata.
- Hardware, HDL, CODAC/EPICS, and runtime deployment evidence: [#53](https://github.com/anulum/scpn-control/issues/53) — 6 open claim(s) (bounded_model=6) — Vivado, Quartus, Yosys, timing closure, simulator evidence, CODAC/EPICS timing, interlocks, backpressure, HIL replay, and runtime parity.

## Module Traceability Table

| Module | Equation or contract | References | Unit contract | Validation evidence | Status | Tracker |
|--------|----------------------|------------|---------------|---------------------|--------|---------|
| `validation/reference_data/diiid` | Facility replay claims require measured signal arrays, physical units, shot identifiers, retrieval timestamp, licence policy, immutable checksums, and source URI provenance. Current repository DIII-D-like artefacts are synthetic reference fixtures only. | DIII-D MDSplus facility data access contract; Repository real/synthetic data manifest schema 1.0; Repository-curated synthetic DIII-D-like fixture manifests | Synthetic fixture signals retain declared SI or dimensionless units for parser and control-path tests, but units do not promote the fixtures to measured-shot evidence. | validation/validate_data_manifests.py; src/scpn_control/core/real_data_manifest.py; tests/test_synthetic_diiid_reference_shots.py; tests/test_validate_data_manifests.py; strict acquisition-readiness gate via --require-real-acquisition blocks missing measured MDSplus manifests | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/control/burn_controller.py` | Burn-control claims must declare Bosch-Hale DT reactivity, alpha-energy partition, Lawson triple product, burn-fraction approximation, alpha-heating integration volume, tokamak major/minor-radius ordering, strictly ordered normalised alpha-heating profile grids, reactivity-exponent stability boundary, and PI auxiliary-heating limits. | Bosch and Hale 1992 DT reactivity; Lawson 1957 ignition criterion; Mitarai and Muraoka 1999 delayed alpha-heating feedback; Repository fail-closed burn-control claim-admission contract | SI joules, m^-3, seconds, keV, MW, m^3/s, metres, normalised profile radius, dimensionless Q, and Lawson triple product units m^-3 s keV. | tests/test_burn_controller.py alpha-heating, Lawson, burn-fraction, profile-domain, control, and claim-admission checks; validation/validate_burn_reference.py strict burn-control reference artifact gate; validation/benchmark_burn_control_claims.py bounded claim-admission benchmark; validation/validate_burn_control.py validates the alpha-energy partition E_fus/E_alpha = 5, the alpha power density (n_e/2)^2 <sigma v> E_alpha, the alpha-power volume integral, the energy gain Q = 5 P_alpha/P_aux with its ignition limits, the Lawson triple product and ignition margin, the burn fraction, the reactivity-exponent centred finite difference, and the energy-gain/Lawson/burn-fraction scaling laws against exact closed forms (holding the separately validated Bosch-Hale reactivity as the shared input), with tamper-evident sealed evidence; tests/test_burn_control_validation.py energy-partition, power-density, volume-integral, energy-gain, ignition-limit, Lawson, burn-fraction, reactivity-exponent, scaling-law, configuration-guard, and evidence-seal checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/elm_model.py` | ELM model claims must declare pedestal crash trigger, crash depth, recovery timescale, peeling-ballooning boundary, ELM cycle state, RMP suppression threshold, and energy-particle-loss accounting. | Snyder et al. 2002 peeling-ballooning ELM trigger; Sauter et al. 1999 edge shaping factor; Loarte et al. 2003 Type-I ELM energy loss; resonant magnetic perturbation ELM suppression references | Time in seconds, energy in joules, density in m^-3, temperature in eV or keV, pressure in pascals, magnetic perturbation amplitude dimensionless or tesla with explicit convention. | tests/test_elm_model.py; tests/test_integrated_scenario.py; tests/test_elm_reference_validation.py strict reference-artifact admission checks; validation/validate_elm_reference.py strict digest-bound ELM/RMP reference gate; validation/validate_elm_peeling_ballooning.py validates the ballooning and peeling limits and scalings, the elliptical peeling-ballooning boundary margin and flag consistency, and the Type-I ELM crash energy conservation (W_post = (1-f) W_ped) against exact closed forms, with tamper-evident sealed evidence; tests/test_elm_peeling_ballooning_validation.py ballooning, peeling-scaling, elliptical-boundary, crash-energy, and evidence-seal checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/eped_pedestal.py` | EPED claims must declare the pedestal-width scaling, collisionless width versus collisionality-narrowed effective-width ordering, peeling-ballooning pressure limit, bootstrap-current coupling, H-mode entry assumptions, global beta limits, and compatibility boundary to the transport solver. | Snyder et al. 2009 EPED1 pedestal model and KBM width scaling; Snyder et al. 2011 collisionality width narrowing; Connor et al. 1998 ballooning first-stability boundary; Troyon beta-limit reference; Sauter bootstrap-current fit | Pedestal width in normalised flux or metres with convention metadata; pressure in pascals, temperature in eV or keV, density in m^-3, current in amperes, and beta dimensionless. | tests/test_eped_pedestal.py; tests/test_eped_reference_validation.py strict reference-artifact admission checks; validation/validate_eped_reference.py strict digest-bound EPED pedestal reference gate; tests/test_cross_module_physics.py; tests/test_transport_hmode_edge.py; validation/validate_eped_pedestal.py validates the q95 formula, alpha-inversion pedestal pressure, poloidal-beta definition, ideal-gas temperature, collisionality width narrowing, shaping-factor reference, and the KBM width constraint against exact construction relations, with tamper-evident sealed evidence; tests/test_eped_pedestal_validation.py construction-relation, collisionality, shaping, KBM-fixed-point, and evidence-seal checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/gs_phase_sync.py` | Reduced-order Paper-27 / ζ·sin(Ψ−θ) phase-reduction and multi-step Lyapunov helpers extracted from FusionKernel must declare the phase-reduction approximation, Kuramoto coupling assumptions, and that they are not a full plasma-phase control law until facility phase-control validation is complete. | Paper-27 phase reduction / ζ·sin(Ψ−θ) references; Kuramoto-Sakaguchi mean-field synchronisation literature; Repository FusionKernel dual-home phase-sync contract | Dimensionless oscillator phases, coupling strengths, and Lyapunov metrics in SI time for multi-step tracking helpers. | tests/test_gs_phase_sync.py; tests/test_fusion_kernel.py; tests/test_kuramoto_synchronisation_validation.py | validation_gap | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/fusion_kernel.py` | Fusion-kernel claims must declare the Grad-Shafranov residual, finite-difference grid, coil Green functions, source-profile parameterisation, boundary conditions, nonlinear iteration controls, convergence criteria, and Rust/Python parity boundary. | Grad-Shafranov equilibrium equation; Solov'ev 1968 exact toroidal equilibrium solution; Cerfon and Freidberg 2010 analytic Grad-Shafranov solutions; Jardin 2010 Computational Methods in Plasma Physics; Green-function tokamak coil-response references; repository Rust/Python fusion-kernel parity contract | SI metres, webers per radian, tesla, amperes, pascals, source derivatives, grid spacings, and dimensionless convergence tolerances. | tests/test_fusion_kernel.py; tests/test_geqdsk_regression.py; tests/test_rust_python_parity.py; validation/validate_grad_shafranov_solovev.py production discrete-operator, SOR-solver, and Python-multigrid second-order convergence against the exact Solov'ev equilibrium with tamper-evident sealed evidence plus informational Rust multigrid record; tests/test_grad_shafranov_solovev_validation.py analytic-identity, operator-convergence, SOR-reconstruction, Python-multigrid reconstruction, Rust-backend-record, and evidence-seal checks | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/jax_gk_solver.py` | JAX linear gyrokinetic claims must preserve the native local-dispersion eigenvalue formulation, expose identical physical input contracts, document numerical-precision or backend divergence, and keep the stiffness-to-transport closure bounded as a controller-tuning surrogate. | Repository native linear GK local-dispersion contract; JAX numerical backend reproducibility guidance | Same growth-rate, frequency, geometry, and species units as native linear GK; backend dtype, tolerance, and normalisation must be explicit in validation artefacts. | tests/test_jax_gk_solver.py local-dispersion parity, stiffness, fail-closed, and bounded chi_i-profile closure checks; tests/test_jax_gk_parity_validation.py schema-versioned artifact digest, case-parameter digest, mode-spectrum, and required case/backend admission checks; validation/benchmark_jax_gk_parity.py strict persisted multi-case backend parity artifact producer; validation/validate_jax_gk_parity.py strict persisted backend parity artifact gate with named case and backend coverage requirements; source-level native local-dispersion parity contract; validation/reports/jax_gk_parity/cyclone_base_case_cpu_cpu.json CPU parity artifact with canonical payload digest; validation/reports/jax_gk_parity/tem_kinetic_electron_cpu_cpu.json CPU parity artifact with kinetic-electron TEM mode-spectrum contract; validation/reports/jax_gk_parity/stable_mode_cpu_cpu.json CPU parity artifact with low-drive bounded-growth contract; validation/reports/jax_gk_parity/cyclone_base_case_gpu_nvidia_geforce_gtx_1060_6gb.json GPU parity artifact with canonical payload digest; validation/reports/jax_gk_parity/tem_kinetic_electron_gpu_nvidia_geforce_gtx_1060_6gb.json GPU parity artifact with kinetic-electron TEM mode-spectrum contract; validation/reports/jax_gk_parity/stable_mode_gpu_nvidia_geforce_gtx_1060_6gb.json GPU parity artifact with low-drive bounded-growth contract; validation/reports/jax_gk_parity/jax_gk_parity.md CPU/GPU parity evidence summary | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/sawtooth.py` | Kadomtsev sawtooth-crash claims must declare the q=1 surface location, the helical-flux proxy psi*(rho) = integral rho(1/q - 1) drho, the mixing radius where psi* returns to zero, volume-average redistribution of temperature and density inside the mixing radius, post-crash q=1 reset, conservation of volume-integrated energy and particles, profile invariance outside the mixing radius, and exclusion of full nonlinear MHD reconnection dynamics. | Kadomtsev 1975 full-reconnection sawtooth crash model; Porcelli et al. 1996 sawtooth trigger and reconnection contract | Normalised radius dimensionless, temperature in keV, density in 10^19 m^-3, q dimensionless, helical-flux proxy dimensionless, and major/minor radii in metres. | tests/test_sawtooth.py crash, trigger, monitor, and cycle checks; validation/validate_sawtooth_kadomtsev.py validates volume-integral energy and particle conservation, the helical-flux mixing condition, profile flattening and q reset inside the mixing radius, outside invariance, the analytic q=1-radius convergence, and the no-crash guard against exact references, with tamper-evident sealed evidence; tests/test_sawtooth_kadomtsev_validation.py conservation, helical-flux, flattening, outside-invariance, q1-convergence, no-crash, and evidence-seal checks | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/phase/kuramoto.py` | Phase-runtime claims must declare the Kuramoto-Sakaguchi mean-field coupling, order-parameter calculation, exogenous global-driver injection, phase wrapping convention, Euler step, and optional Rust fast-path parity boundary. | Kuramoto and Sakaguchi phase oscillator model; Sakaguchi H., Kuramoto Y. (1986) Prog. Theor. Phys. 76, 576; Strogatz S. H. (2000) From Kuramoto to Crawford, Physica D 143, 1; Acebron J. A. et al. (2005) Rev. Mod. Phys. 77, 137; Repository phase synchronisation runtime contract | Phases in radians, angular frequencies in radians per second, timestep in seconds, coupling gains dimensionless or radians per second by declared convention, order parameter dimensionless. | tests/test_phase_kuramoto.py; tests/test_phase_properties.py; tests/test_phase_properties_extended.py; tests/test_kuramoto_synchronisation_validation.py; validation/benchmark_kuramoto_runtime_evidence.py schema-versioned runtime parity and timestep-refinement evidence producer; validation/validate_kuramoto_synchronisation.py published-benchmark synchronisation validation: critical coupling Kc(alpha)=2*gamma/cos(alpha), exact Lorentzian order-parameter branch R=sqrt(1-Kc/K), and Sakaguchi onset shift | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/marfe.py` | MARFE claims must declare impurity radiation loss, temperature scan, condensation criterion, Greenwald comparison, power-balance inputs, bounded impurity fraction, edge parallel connection-length scaling through q95 and R0, validated front-temperature state, and detection thresholds for detached high-radiation states. | Stangeby 2000 scrape-off-layer and divertor references; Greenwald 2002 density-limit reference; radiation-condensation MARFE onset references | Temperature in eV, density in m^-3, power in MW or W with explicit conversion, plasma current in MA or A with convention metadata, impurity fraction dimensionless on (0, 1], connection length in metres. | tests/test_marfe.py; tests/test_marfe_reference_validation.py strict reference-artifact admission checks; validation/validate_marfe_reference.py strict digest-bound MARFE reference gate; validation/validate_marfe_onset.py validates the Greenwald limit, bounded MARFE density-limit scaling, edge connection length, radiation-condensation critical density, onset cooling-slope membership, stability-diagram boundary, and front-temperature detector thresholds against exact closed forms, with tamper-evident sealed evidence; validation/reports/marfe_onset.json sealed exact-closed-form MARFE evidence; validation/reports/marfe_onset.md human-readable MARFE onset and density-limit summary; validation/benchmark_marfe_onset_claims.py bounded local-regression benchmark with production_claim_allowed=false; validation/reports/marfe_onset_claims.json local-regression benchmark evidence for the sealed MARFE validator; tests/test_marfe_onset_validation.py Greenwald, MARFE-limit, condensation, scan-boundary, front-detector, configuration-guard, CLI, and evidence-seal checks; tests/test_cross_module_physics.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/gk_geometry.py` | Geometry claims must declare Miller-shape coordinates, derivative-resolved ballooning-angle grids, Jacobian, toroidal-field convention, safety-factor field-pitch relation, poloidal-field construction, b-dot-grad-theta metric, metric coefficients, contravariant metric determinant identity, and local-equilibrium validity bounds. | Miller et al. 1998 local equilibrium; Cyclone Base Case local Miller geometry benchmark | Major radius, minor radius, local radius, and gradient lengths in metres; angles in radians; magnetic-field strength in tesla; q, shear, elongation, triangularity, and shaping derivatives dimensionless. | tests/test_gk_geometry.py field-pitch, metric, contravariant metric-determinant identity, curvature, shaping, and interface checks; validation/validate_gk_geometry_reference.py schema-versioned immutable Miller reference cases with b-dot-grad-theta samples, per-case digests, unit contracts, tolerance metadata, and canonical payload SHA-256; validation/validate_gk_geometry_independent.py independent finite-difference cross-check of the production metric against validation/gk_geometry_independent_reference.py across circular, shaped, high-shear, and finite shaping-shear (s_kappa/s_delta) local equilibria; closes the metric radial-derivative fidelity gap; source-level local Miller geometry contract | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/ntm_dynamics.py` | NTM dynamics claims must declare rational-surface search, rational-surface containment inside the minor radius, tokamak major/minor-radius ordering, modified Rutherford equation terms, bootstrap drive, polarisation and curvature terms, ECCD control coupling, seed-island assumptions, and controller-validity limits. | Rutherford 1973 tearing-mode island evolution; Sauter et al. 1997 neoclassical tearing-mode model; La Haye 2006 modified Rutherford equation and ECCD control; ECCD NTM control references; repository rational-surface and island-dynamics contract | Island width in metres, time in seconds, current in amperes, q dimensionless, rho dimensionless, ECCD power in MW or W with conversion metadata. | tests/test_ntm_dynamics.py; tests/test_ntm_reference_validation.py strict reference-artifact admission checks; validation/validate_ntm_reference.py strict digest-bound NTM reference gate; tests/test_cross_module_physics.py; validation/validate_ntm_island_dynamics.py validates the production Modified Rutherford Equation solver against the exact classical-only separable trajectory and the closed-form classical+bootstrap saturated-width fixed point, with tamper-evident sealed evidence; tests/test_ntm_island_dynamics_validation.py classical-trajectory, saturated-width-attractor, and evidence-seal checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/runtime_admission.py` | Runtime-admission evidence must bind the requested native execution backend, spin pacing mode, requested SNN/Z3/network/heartbeat cores, kernel realtime capability, scheduler policy, per-core CPU governor state, CPU affinity, host load, admission status, admission errors, latency statistics, benchmark command, and canonical payload SHA-256 before release evidence can rely on local runtime-admission measurements. | Linux PREEMPT_RT and realtime scheduler admission contract; Repository native hardware campaign runtime-admission contract; Repository release-evidence benchmark admission contract | Core identifiers are non-negative CPU indices, scheduler policies use Linux scheduler names, CPU governors use sysfs governor strings, load averages are host process-queue averages, latencies are microseconds, and report and payload digests are lowercase SHA-256 hex strings. | tests/test_runtime_admission.py Python runtime-admission policy, sysfs parsing, affinity, and fail-closed checks; tests/test_cli_runtime_admission.py CLI runtime-admission policy checks; tests/test_rust_engine_runtime_admission.py optional PyO3 runtime-admission snapshot checks; tests/test_bench_runtime_admission.py benchmark context, load, command, and payload digest checks; tests/test_runtime_admission_evidence_validation.py strict persisted runtime-admission benchmark admission checks; tests/test_release_evidence_validation.py top-level release-evidence mandatory runtime-admission gate checks; validation/reports/runtime_admission_release_20260605T000000Z.json local-regression report with canonical payload SHA-256 and fail-closed production-claim boundary | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/control/rzip_model.py` | Rigid-plasma vertical stability claims must declare the linearised state vector [Z, dZ/dt, circuit currents], mutual-inductance derivative, vertical field index, tokamak major/minor-radius ordering, declared vertical inertia, wall-normalised feedback-gain threshold, wall or active-coil circuit model, calibration source, growth-rate comparison tolerance, tamper-evident calibration payload digest, and facility-claim admission status. | Lazarus et al. 1990 rigid plasma vertical stability model; Wesson 2011 tokamak vertical stability and field-index references; Repository vessel circuit model contract | SI metres, seconds, amperes, henries, ohms, tesla, and dimensionless vertical field index; growth time is reported in milliseconds; calibration evidence digests are SHA-256 hex strings. | tests/test_rzip_model.py vertical growth, field-index, declared-inertia, feedback-gain, physical-parameter, controller-measurement, SciPy Riccati, NumPy fallback Riccati, zero-gain last-resort, and singular-circuit checks; validation/validate_rzip_reference.py strict RZIP reference artifact gate; tests/test_rzip_model.py calibration evidence, facility-claim admission, and persisted evidence checks; validation/benchmark_rzip_calibration.py bounded local RZIP calibration benchmark; validation/reports/rzip_calibration.json bounded local calibration report with evidence payload digest; validation/validate_rzip_vertical_stability.py validates the production state-space growth rate against the exact no-wall rigid-mode eigenvalue sqrt(-K/M_eff), the stable-index oscillation frequency, the growth-time identity, exact Ip/index/inertia scaling laws, and resistive-wall stabilisation, with tamper-evident sealed evidence; tests/test_rzip_vertical_stability_validation.py no-wall growth, oscillation-frequency, scaling-law, wall-stabilisation, and evidence-seal checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/scpn/fpga_export.py` | FPGA-export claims must declare LIF fixed-point quantisation, leak right-shift approximation, threshold scaling, signed weight saturation, generated HDL boundary, target family assumptions, and synthesis-tool responsibility. | Repository SCPN compiler and fixed-point export contract; Leaky integrate-and-fire digital implementation contract | Fixed-point values use declared bit width and fractional-bit scale; clock in MHz, timestep in seconds, FIFO depth and neuron counts dimensionless, weights and thresholds quantised by explicit integer scale. | tests/test_fpga_export.py; tests/test_scpn_compiler.py; tests/test_fpga_public_claim_boundary.py public-surface guard against FPGA bitstream export claims | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/scpn/formal_verification.py` | Formal verification claims must declare the compiled Petri-net transition relation, exact bounded reachability depth, rational marking arithmetic, marking-bound safety obligations, algebraic place-invariant weights, transition liveness obligations, bounded temporal response and recurrence specifications, inhibitor-arc semantics, and counterexample path reporting. | Petri-net reachability and P-invariant analysis; bounded temporal logic over finite transition systems; repository SCPN compiler transition-relation contract | Markings are dimensionless token densities; arc weights and invariant weights are rationalised from finite decimal inputs; max_depth is a non-negative integer firing bound; temporal response windows count transition firings. | tests/test_scpn_formal_verification.py exact reachability, marking bounds, transition liveness, algebraic place invariants, all-path bounded response, recurrence, counterexample, and invalid-domain checks; src/scpn_control/scpn/formal_verification.py exact explicit-state finite transition relation over rational markings; validation/reports/scpn_z3_formal.json schema-versioned bounded SMT evidence with canonical payload SHA-256; tests/test_scpn_formal_verification.py Z3 formal report duplicate-key loader rejection checks; tests/test_scpn_formal_verification.py Z3 formal report unknown top-level and proof-section field rejection checks; tests/test_scpn_formal_verification.py Z3 formal report counterexample record schema rejection checks; tests/test_scpn_formal_verification.py Z3 section solver-status consistency rejection checks; tests/test_scpn_formal_verification.py blocked Z3 report solver-availability boundary checks; tests/test_scpn_formal_verification.py Z3 section checked_specs uniqueness and malformed-entry rejection checks; tests/test_controller_safety_case.py target-hardware readiness artifact digest and E2E latency admission checks; validation/validate_release_evidence.py native formal certificate release admission requires AOT certificate evidence, report digest, evidence class, production-claim boundary, and empty validator-error list; tests/test_release_evidence_validation.py native formal production-overclaim, production-class boundary, and validator-error rejection checks; validation/validate_scpn_lean_formal.py duplicate-key and Lean proof-report admission gate routed through load_lean_formal_report; tests/test_scpn_lean_verification.py Lean solver/version binding, duplicate-key rejection, unsupported-contract rejection, theorem namespace, module-path, safety-case-ID, assumption, and mandatory digest checks; tests/test_scpn_lean_formal_validation.py validator-level Lean report, mandatory digest, unrelated module-path rejection, and artifact admission checks; tests/test_artifact_codec.py safety-critical artifact-manifest Z3/Lean duplicate report key, Z3 unknown report and counterexample fields, Z3 solver-status consistency, Z3 section checked_specs uniqueness, Lean solver/version, unsupported-contract, namespace, module-path, safety-case-ID, and unknown-field rejection checks; tests/test_scpn_formal_verification.py Z3 unknown solver-section counterexample rejection checks; tests/test_artifact_codec.py safety-critical artifact-manifest Z3 unknown solver-section counterexample rejection checks; tests/test_scpn_formal_verification.py Z3 pass-report unavailable-solver rejection checks; tests/test_artifact_codec.py safety-critical artifact-manifest Z3 pass-report unavailable-solver rejection checks; tests/test_scpn_formal_verification.py Z3 pass-report non-Z3 solver-label rejection checks; tests/test_artifact_codec.py safety-critical artifact-manifest Z3 pass-report non-Z3 solver-label rejection checks; tests/test_controller_runtime_safety_admission.py live NeuroSymbolicController runtime-certificate admission, artifact-topology drift, partial-evidence, and failed-replay rejection checks | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/vmec_lite.py` | VMEC-lite claims must declare flux-surface parameterisation, rotational-transform profile, Fourier mode truncation, field-period count, positive R00 and sampled major-radius boundary state, pressure/current assumptions, residual metric, geometry/profile/current provenance, reference-comparison tolerances for R_mn, Z_mn, and iota, convergence status, and explicit exclusion from full VMEC-grade 3D MHD equilibrium claims unless matched references pass the fail-closed admission gate. | VMEC 3D equilibrium references; stellarator flux-surface Fourier parameterisation references | SI metres, tesla, pascals, amperes, webers, dimensionless rotational transform, and Fourier coefficients with declared length units. | tests/test_vmec_lite.py verifies finite spectral reconstruction, bounded force residuals, physical domains, and fail-closed VMEC-lite claim evidence admission; validation/validate_vmec_lite_geometry.py validates spectral mode count, direct Fourier evaluation, axisymmetric boundary coefficients, fixed-boundary radial scaling, q/iota reciprocity, B-coefficient construction, and positive sampled major radius against exact repository forms, with tamper-evident sealed evidence; tests/test_vmec_lite_geometry_validation.py exact-geometry, evidence-seal, CLI, and failure-path checks; validation/reports/vmec_lite_geometry.json sealed exact-geometry VMEC-lite evidence; validation/reports/vmec_lite_geometry.md human-readable VMEC-lite exact-geometry summary; validation/benchmark_vmec_lite_claims.py publishes deterministic bounded synthetic claim evidence with explicit full-VMEC exclusion and sealed geometry-validation payload binding; validation/reports/vmec_lite_claims.json records bounded Fourier, field-period, profile, residual, q-domain, positive-major-radius, and geometry-validation evidence; scpn-control-rs/crates/control-core/src/vmec_interface.rs Rust VMEC interface uses the same non-duplicated m=0 toroidal-mode convention as the Python VMEC-lite spectral basis and carries VMEC interface unit tests; validation/validate_vmec_reference.py strict VMEC reference artefact gate | validation_gap | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/control/advanced_soc_fusion_learning.py` | SOC turbulence-learning claims must declare the sandpile lattice, critical-gradient threshold, predator-prey zonal-flow coupling, shear-suppression term, bounded substep relaxation, Q-learning state discretisation, action set, and random policy assumptions. | Diamond and Hahm 1995 SOC turbulence reference; Kim and Diamond 2003 zonal-flow predator-prey coupling; Biglari, Diamond and Terry 1990 shear-suppression reference | Dimensionless lattice gradients, flow amplitudes, shear, toppling counts, Q-table values, reward, and RNG-seeded action choices. | tests/test_advanced_soc.py SOC physics and learning checks; tests/test_advanced_soc_verbose_plot.py verbose and plotting-path checks; tests/test_visualization_paths.py SOC visualisation checks; validation/validate_soc_reference.py strict reference-artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/current_drive.py` | Current-drive claims must declare ECCD, LHCD, and NBI source powers, radial deposition centres and widths, grid-normalised deposition conservation, density and temperature normalisation, Fisch-Boozer or Fisch efficiency coefficients, Prater launch-angle factor, Stix slowing-down time and critical energy for NBI, and the absence of ray-tracing or Fokker-Planck facility validation. | Fisch and Boozer 1980 electron-cyclotron current-drive efficiency; Prater 2004 ECCD launch-angle efficiency scaling; Fisch 1978 lower-hybrid current-drive efficiency; Stix 1972 neutral-beam slowing-down and critical-energy formulae; Ehst and Karney 1991 neutral-beam current-drive model; Repository fail-closed current-drive claim-admission contract | Power in MW or W with explicit conversion, rho dimensionless, density in 10^19 m^-3, temperatures and beam energy in keV, current density in A/m^2, total current in amperes, efficiency coefficients in declared normalised A/W form. | tests/test_current_drive.py ECCD, LHCD, and NBI deposition, scaling, source superposition, total-current integration, and claim-admission checks; validation/validate_current_drive_reference.py strict current-drive reference artifact gate; validation/benchmark_current_drive_claims.py bounded claim-admission benchmark; validation/validate_current_drive.py validates grid-normalised deposition power conservation, the deposition centroid, the Stix critical energy and slowing-down formulae and scalings, the Prater ECCD efficiency and launch-angle maximisation, the ECCD/LHCD driven-current proportionality, and the neutral-beam current chain against exact closed forms, with tamper-evident sealed evidence; tests/test_current_drive_validation.py deposition-conservation, Stix-formula, Prater-efficiency, j_cd-proportionality, NBI-chain, and evidence-seal checks | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/blob_transport.py` | Blob transport claims must declare interchange drive, sheath closure, radial velocity scaling, filament size, density and temperature perturbation assumptions, non-empty strictly ordered separatrix-to-wall scrape-off-layer profile coordinates, positive ordered detector-event domains, and scrape-off-layer validity bounds. | Stangeby 2000 scrape-off-layer transport references; blob-filament interchange transport scaling references | SI metres, seconds, m/s, density in m^-3, temperature in eV, magnetic field in tesla, and dimensionless normalised perturbations. | tests/test_blob_transport.py velocity-regime, size-scaling, ensemble, flux, strict-profile-grid, wall-flux, and detector-domain checks; tests/test_blob_transport_reference_validation.py strict reference-artifact admission checks; validation/validate_blob_transport_reference.py strict digest-bound SOL blob reference gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core` | Analytical approximations must state their source equation, retained terms, omitted terms, unit contract, and allowed parameter domain. | Cordey 1981 orbit-width estimate; Sauter et al. 1999 neoclassical fits; Stangeby 2000 SOL two-point model | Per-module SI or declared normalised units with explicit conversion boundaries. | docs/physics_methods.md simplification declarations; source grep for simplification and approximation markers | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control` | Control approximations must declare reduced-state dynamics, actuator assumptions, estimator assumptions, and facility-exclusion boundaries before any controller-readiness claim. | Tokamak control reduced-plant literature; Repository control safety and disruption contracts | Per-controller declared SI plasma, actuator, magnetic, force, and timing units. | src/scpn_control/control/gym_tokamak_env.py; src/scpn_control/control/free_boundary_tracking.py | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control` | Runtime approximations must state phase wrapping, fixed-point shift approximations, replay fixture provenance, and hardware export limits. | Kuramoto phase oscillator reference model; Repository fixed-point spiking export contract | Declared phase radians, tick timing, fixed-point scale factors, and replay geometry units. | src/scpn_control/phase/kuramoto.py; src/scpn_control/scpn/fpga_export.py; src/scpn_control/scpn/geometry_neutral_replay.py; validation/validate_tracker53_evidence.py aggregate tracker #53 hardware/runtime evidence gate with fail-closed production-claim mode; tests/test_tracker53_evidence_gate.py tracker #53 manifest, digest, missing-report, and production-overclaim checks; validation/reports/tracker53_evidence_gate.json digest-bound tracker #53 local bounded evidence manifest | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/control/mu_synthesis.py` | Mu-analysis claims must declare the structured uncertainty blocks, positive block bounds, Riccati state-feedback K-step, bound-scaled static closed-loop DC robust-performance map, D-scaling upper-bound fit, finite controller state/timestep domains, and the exclusion of full frequency-dependent H-infinity D-K synthesis unless an external validated backend is wired. | Doyle 1982 structured singular value definition; Packard and Doyle 1993 the complex structured singular value; Balas et al. 1993 mu-analysis and synthesis toolbox; Skogestad and Postlethwaite 2005 multivariable feedback control; Repository fail-closed static mu-analysis claim-admission contract | State, control, output, and uncertainty units inherit the supplied state-space plant contract; mu, uncertainty bounds, and D-scalings are dimensionless. | tests/test_mu_synthesis.py D-scaled upper-bound behaviour, uncertainty-bound scaling, finite-domain rejection, closed-loop plant dependence, claim-admission, persisted digest, tamper, and duplicate-key checks; validation/validate_mu_structured_singular_value.py closed-form exact-identity validation of compute_mu_upper_bound (full-block mu=sigma_max, diagonal mu=max\|M_ii\|, rank-one mu=sum\|u_i v_i\|, and the rho<=mu<=sigma_max sandwich) with tamper-evident sealed evidence; tests/test_mu_structured_singular_value_validation.py exact-identity, sandwich, determinism, diagnostic-not-gated, and evidence-seal checks; validation/validate_mu_synthesis_reference.py strict mu-analysis reference artefact gate; validation/benchmark_mu_synthesis_claims.py bounded claim-admission benchmark; docs/learning/control_theory_primer.md states the bounded static analysis domain | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/checkpoint.py` | Checkpoint claims must declare serialised solver state, non-negative episode counter, object-valued metrics schema, finite-value policy, schema-version boundary, and fail-closed handling for corrupt or missing checkpoint artefacts. | repository restart and replay provenance contract | Units are inherited from stored state and metrics; checkpoint metadata must preserve schema version, episode index, and provenance boundaries. | tests/test_checkpoint.py round-trip, resume, finite-payload, schema-version, and corrupt-payload checks | bounded_model | [#53](https://github.com/anulum/scpn-control/issues/53) |
| `src/scpn_control/core/differentiable_scenario.py` | Coupled differentiable scenario claims must declare the analytic Solov'ev-form equilibrium parameters, R/Z flux grid, flux-derived radial weights, four-channel transport rollout, source-schedule target, gradient tolerance, sampled finite-difference audit of both equilibrium parameters and source schedule, local non-isolated timing context, and fail-closed readiness state before any end-to-end differentiable scenario claim is promoted. | Solov'ev 1968 analytic equilibrium reference; repository differentiable transport facade contract; repository coupled scenario readiness contract | Equilibrium parameters retain the declared analytic flux normalisation; R and Z grids use metres; rho is dimensionless and uniformly spaced; transport profiles, coefficients, and sources inherit the four-channel differentiable-transport units; audit losses, gradient errors, and p95 timing are finite non-negative scalar evidence. | tests/test_differentiable_scenario.py Solov'ev closed-form flux, NumPy/JAX loss parity, source and equilibrium gradient coupling, sampled finite-difference audit, metadata, and fail-closed readiness checks; tests/test_differentiable_scenario_validation.py report duplicate-key, promotion-overclaim, inconsistent-audit, blocked-backend, and repository evidence validation checks; validation/benchmark_differentiable_scenario.py emits bounded coupled-scenario readiness evidence or a blocked JAX-unavailable report; validation/validate_differentiable_scenario.py admits the schema-versioned report while requiring claim_admissible=false until physics traceability is satisfied; validation/reports/differentiable_scenario_readiness.json records campaign metadata, gradient audit, readiness state, and local non-isolated timing context; validation/reports/differentiable_scenario_readiness.md provides the human-readable bounded evidence summary | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/density_controller.py` | Density-control claims must declare the Greenwald limit, ITER operating margin, radial particle transport grid, gas-puff source, pellet source, NBI source, cryopump sink, recycling source, controller gains, CFL-limited explicit transport step, geometry/transport/actuator/diagnostic provenance, source integral, particle-inventory change, and matched Greenwald-fraction and inventory reference tolerances before facility-calibrated density-control claims. | Greenwald 2002 density limit; ITER Physics Basis 1999 density operating margin; Parks and Turnbull 1978 neutral gas shielding pellet-ablation model; Milora 1995 neutral gas shielding pellet-ablation model | SI particles per second, metres, seconds, m^-3, m^2/s, m/s, pellet millimetres, beam keV, megawatts, and dimensionless Greenwald fraction. | tests/test_density_controller.py Greenwald, NGS pellet trajectory deposition, sink, controller, geometry, transport-profile, source-input, timestep guard, and fail-closed density-control claim evidence checks; tests/test_pellet_injection.py Parks-Turnbull NGS ablation and pellet trajectory checks; validation/benchmark_density_control_claims.py publishes deterministic bounded synthetic claim evidence with explicit facility-calibrated claim exclusion; validation/reports/density_control_claims.json records bounded geometry, transport, actuator, diagnostic, CFL, Greenwald, source-integral, and inventory evidence; validation/validate_density_reference.py strict density reference artefact gate; validation/validate_density_control.py validates the Greenwald limit I_p/(pi a^2), the volume-averaged Greenwald fraction, the circular flux-surface volume elements V' and V, the gas-puff, neutral-beam, and recycling source normalisation (particle conservation of the source integral), the cryopump edge sink, the diffusion operator vanishing on a uniform interior, and the Greenwald scaling laws against exact closed forms, with tamper-evident sealed evidence; tests/test_density_control_validation.py Greenwald-limit, Greenwald-fraction, volume-element, source-conservation, cryopump-sink, diffusion-invariance, scaling-law, configuration-guard, and evidence-seal checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/differentiable_transport.py` | Differentiable transport claims must declare the four-channel profile order, cylindrical Crank-Nicolson diffusion step, uniform normalised radial grid, source-term convention, core zero-gradient boundary, edge Dirichlet boundary, transport-coefficient gradients, one-step and multi-step source-schedule gradient targets, sampled finite-difference gradient audit, audited one-step and rollout gradient-admission latency metadata, runtime provenance for CPU/GPU comparison campaigns, bounded Z3 controller-formal evidence binding, neural and reduced-gyrokinetic closure coefficient mapping, campaign provenance metadata, schema-versioned replay metadata, traced JAX rollout-gradient loss construction, JAX x64 precision enablement before `jax.numpy` import, finite non-negative audit losses and errors, tolerance agreement with campaign metadata, unique in-domain sampled audit indices, pass/fail consistency with maximum audit error, ordered latency percentiles, and optional one-step and rollout Grad-Shafranov flux-map radial weighting contract. | repository JAX transport primitive contract; repository integrated transport solver contract | Profiles inherit channel units for electron temperature, ion temperature, electron density, and impurity density; chi uses m^2/s or the declared normalised diffusivity convention; sources use profile units per second; rho is dimensionless and uniformly spaced; audit loss, audit errors, and latency percentiles are finite non-negative scalar evidence. | tests/test_differentiable_transport.py diffusion, boundary, fallback, optional JAX coefficient-gradient, one-step and multi-step source-schedule-gradient, sampled finite-difference gradient audit, neural and reduced-GK closure coefficient mapping, campaign metadata provenance, schema-versioned replay metadata, replay-drift guard, audit-evidence admission semantics, latency-report runtime-provenance persistence guards, controller-formal readiness digest binding, and one-step plus rollout equilibrium-weighted GS-flux checks; tests/test_nmpc_transport_tuning.py NMPC neural-closure, one-step source-schedule, and multi-step source-rollout tuning fail closed without JAX and preserve gradient-audit admission results; tests/test_nmpc_transport_tuning.py NMPC coefficient, one-step source-schedule, and multi-step source-rollout tuning results carry campaign metadata, closure provenance, finite source bounds, and fail-closed gradient-audit evidence; validation/benchmark_differentiable_transport_latency.py audited one-step and rollout gradient-admission latency benchmarks or fail-closed JAX-unavailable reports; validation/reports/differentiable_transport_latency.json and validation/reports/differentiable_transport_rollout_latency.json bounded latency reports or blocked-backend status | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/digital_twin_online_update.py` | Online model-update claims must declare tunable bounded parameters, external TRANSP/TSC simulator artifact provenance, target summary metrics, tolerances, deterministic random seed, Gaussian-process acquisition settings, finite non-negative loss history, source-bound Bayesian results, parameter-domain admission, observation-unit coverage, and fail-closed behavior when external artifacts are absent or malformed. | TRANSP integrated modelling evidence contract; TSC time-dependent simulation evidence contract; Bayesian optimisation for bounded model calibration; Repository digital-twin runtime contract | Tunable density in m^-3, effective charge dimensionless, actuator lag in steps, actuator rate limit dimensionless per step, target metrics in declared digital-twin summary units, non-negative loss values, and simulator time base in seconds. | tests/test_digital_twin_online_update.py external artifact, loss, Bayesian-update, and deterministic benchmark checks; tests/test_digital_twin_reference_validation.py strict digital-twin artifact gate including TRANSP and TSC; validation/benchmark_digital_twin_online_update.py deterministic bounded online-update benchmark | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/free_boundary_tracking.py` | Free-boundary tracking claims must declare direct kernel-in-the-loop coil-response identification, bounded least-squares correction, actuator lag, slew limits, supervisor rejection, measurement bias, drift, latency, and observer compensation assumptions. | Grad-Shafranov free-boundary control references; Repository FusionKernel free-boundary objective contract; Repository deterministic free-boundary acceptance campaign; Repository fail-closed free-boundary claim-admission contract | SI coil currents, metres, webers per radian, seconds, amperes per second, objective-space residuals, and dimensionless supervisor gains. | tests/test_free_boundary_tracking.py controller, disturbance-observer, and controller-to-claim pipeline checks; tests/test_free_boundary_tracking_claims.py fail-closed claim-evidence validation, reference-artifact admission, and persistence checks; tests/test_free_boundary_tracking_variants.py actuator, observer, latency, and supervisor edge-path checks; tests/test_free_boundary_tracking_acceptance.py deterministic acceptance campaign; validation/validate_free_boundary_reference.py strict free-boundary reference artifact gate; validation/benchmark_free_boundary_tracking_claims.py bounded claim-admission benchmark | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/disruption_roc.py` | Disruption ROC-core claims must declare the per-window risk-scoring convention (the dBdt signal window plus n=1/n=2 toroidal observables with the n=3 amplitude as a bounded 0.4*n=2 approximation and a toroidal-amplitude clip), the endpoint-forced FPR-sorted trapezoidal AUC assembly, and the warning-time convention that an alarm on a disruptive shot is a true positive only if it fires strictly before the labelled disruption sample. | Repository per-window disruption risk-scoring convention shared with disruption_contracts.run_real_shot_replay; n=3 toroidal amplitude bounded approximation 0.4 * n=2 (no dedicated n=3 diagnostic channel); DisruptionBench-style warning-time convention: true positive only when the alarm precedes the labelled disruption sample | dBdt in gauss per second, dimensionless toroidal mode amplitudes clipped to [0, 10], dimensionless risk in [0, 1], false-positive and true-positive rates in [0, 1], and warning lead time in seconds on the shot timebase. | tests/test_disruption_roc.py ROC, scoring, and warning-time unit checks | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/control/disruption_contracts.py` | Disruption-contract claims must declare synthetic disruption signal generation, toroidal-mode amplitudes, mitigation-cocktail coupling, impurity transport response, halo/runaway post-disruption response, TBR equivalence scaling, and RL action bias assumptions. | Pautasso et al. 2017 disruption current-quench constraints; Riccardo et al. 2010 halo-current rise-time references; Abdou et al. 2015 blanket neutronics calibration references | SI seconds, milliseconds, mega-amperes, megajoules, moles, megawatts, dimensionless risk, toroidal mode amplitudes, and tritium breeding ratio. | tests/test_disruption_contracts.py contract smoke checks; tests/test_disruption_contracts_pure.py pure physics-path checks; tests/test_disruption_edge_cases.py edge-case disruption checks; validation/validate_disruption_reference.py strict disruption reference artifact gate | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/disruption_sequence.py` | Disruption-sequence claims must declare phase ordering, finite positive configuration domains, Rechester-Rosenbluth thermal-quench transport, post-quench radiative-cooling exposure, current-quench timing, mitigation action coupling, runaway-electron beam phase, stochastic event boundaries, halo-force convention, and replay provenance. | ITER disruption mitigation sequence references; repository disruption phase-state contract | Time in seconds or milliseconds with explicit convention, current in amperes or mega-amperes, energy in joules or megajoules, magnetic field in tesla, geometry in metres, density in 10^20 m^-3, and dimensionless phase labels. | tests/test_disruption_sequence.py post-quench cooling, phase ordering, current quench, runaway, and halo-force checks; tests/test_disruption_safe_api.py; validation/validate_disruption_sequence.py validates the bounded phase-ordering, total-duration, wall-heat-load, current-trace, halo-force, post-TQ temperature, and SPI mitigation-branch contracts against exact repository-owned identities, with tamper-evident sealed evidence; tests/test_disruption_sequence_validation.py exact phase-order, current-trace, mitigation-branch, CLI, and evidence-seal checks | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/control/disruption_predictor.py` | predict_disruption_risk claims must declare that the score is a deterministic fixed-weight logistic combination of thermal-quench and toroidal-asymmetry features (n=1,2,3 mode amplitudes) with a hand-chosen logit bias, not a model trained or fitted on a real disruption database, and must declare the feature vector contract and the synthetic sanity-check provenance. | Hand-chosen fixed-weight logistic heuristic over toroidal-asymmetry observables; Not trained on a real disruption database (e.g. DIII-D/JET disruption warning DBs) | Dimensionless risk in [0, 1]; mode amplitudes and asymmetry features dimensionless; logit bias dimensionless. | tests/test_disruption_predictor.py heuristic feature-vector and bounded-output checks; tests/test_disruption_predictor_claim_boundary.py machine-readable claim-boundary and public-surface checks; validation/reports/disruption_replay_pipeline_benchmark.md synthetic sanity-check provenance | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gk_interface.py` | Generated input decks and parsed outputs must round-trip through real TGLF, GENE, GS2, CGYRO, or QuaLiKiz executables. | TGLF Staebler et al. 2007; GENE Jenko et al. 2000; GS2 Kotschenreuther et al. 1995; CGYRO Candy et al. 2016; QuaLiKiz Bourdelle et al. 2007 | Code-specific flux, growth-rate, frequency, and geometry units converted into repository normalisation with explicit metadata. | docs/joss_paper.md external GK limitation; validation/validate_gk_interface_artifacts.py strict digest-bound external interface artefact gate with report schema, payload digest, artefact file digest, portable paths, duplicate code/run rejection, and blocked public-claim state | external_dependency_blocked | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/federated_disruption.py` | FedAvg/FedProx disruption classifiers must federate per-facility client updates without centralising raw arrays, validate the shared eight-feature disruption contract, clip and noise facility model deltas under a declared Gaussian differential-privacy budget, and preserve a serialisable privacy ledger. | McMahan et al. 2017 Communication-Efficient Learning of Deep Networks from Decentralized Data; Li et al. 2020 FedProx; Dwork and Roth 2014 Algorithmic Foundations of Differential Privacy; Abadi et al. 2016 Deep Learning with Differential Privacy | Eight disruption features use declared tokamak control variables: Ip, beta_N, q95, n/n_GW, li, dBp/dt, locked-mode amplitude, and n=1 RMS; labels are binary disruption indicators. | tests/test_federated_disruption.py module-specific federation, array-ingestion, DP-ledger, and benchmark contract tests; validation/benchmark_federated_disruption.py deterministic synthetic multi-facility benchmark; validation/reports/federated_disruption_benchmark.json synthetic report with explicit claim boundary | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/uncertainty.py` | Uncertainty claims must declare sampled variables, distributions, correlations, random seed, sample count, propagation chain, convergence and percentile-ordering criteria, finite-value rejection policy, D-T fuel dilution handling, sensitivity outputs, scenario/prior/sensitivity provenance, and matched central-value and sigma reference tolerances before calibrated predictive-UQ claims. | Monte Carlo uncertainty propagation references; repository fusion-performance uncertainty contract | Units inherit each propagated physical quantity; distribution parameters must preserve SI or declared normalised units and dimensionless uncertainty fractions. | tests/test_uncertainty.py verifies IPB98 monotonicity, Bosch-Hale domains, D-T fuel dilution, full-chain sampling, percentile ordering, finite-output checks, and fail-closed UQ claim evidence admission; tests/test_full_chain_uq.py full-chain UQ behavioural coverage; tests/test_uncertainty_sigma_guard.py sigma guard coverage; validation/benchmark_uq_claims.py publishes deterministic bounded synthetic claim evidence with explicit calibrated-UQ exclusion; validation/reports/uq_claims.json records bounded scenario, prior, seed, sample-count, percentile, finite-output, sensitivity, and claim-admission evidence; validation/validate_uncertainty_reference.py strict UQ reference artefact gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/scpn/geometry_neutral_replay.py` | Geometry-neutral replay claims must declare synthetic W7-X-like fixture provenance, field-line spread metric, actuator current bounds, latency model, stuck-actuator fault schedule, controller feature mapping, replay acceptance thresholds, tamper-evident manifest digests, digest-bound AER admission evidence when present, and fail-closed scenario admission for initial-frame alignment, objective metric availability, current-envelope constraints, actuator-supported fault modes, and AER monotonic-ingress consistency. | Repository geometry-neutral control contract; W7-X-like reduced-order stellarator replay fixture; Repository AER admission replay contract | Field-line spread in radians, currents in amperes, timestep in seconds, latency in microseconds, effective ripple dimensionless, AER timestamps in nanoseconds, AER feature counts dimensionless, controller objectives and thresholds declared per replay manifest, manifest digests encoded as SHA-256 hex. | tests/test_geometry_neutral_replay.py; tests/test_geometry_neutral_contracts.py; tests/test_geometry_neutral_replay_v1_1.py AER admission digest, strict-monotonic, tamper, retention-domain, and schema checks; tests/test_geometry_neutral_replay.py replay evidence admission, digest, bounded-claim, device-claim, tamper, and duplicate-key checks | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/global_design_scanner.py` | Design-scanner claims must declare the aspect-ratio minor-radius inference, elongated-volume estimate, Greenwald-density fraction, bounded temperature and DT-reactivity proxies, confinement-time proxy, auxiliary-power Q denominator, neutron-wall-loading geometry, beta-p proxy, and relative cost index before disruption-episode use. | Greenwald density limit scaling; DT fusion power density and neutron wall loading contracts; IPB98-like confinement-time scaling used as a bounded repository proxy | Major and minor radii in metres, toroidal magnetic field in tesla, plasma current in mega-amperes, density in m^-3, volume in m^3, power in MW, wall loading in MW/m^2, confinement time in seconds, and Q, beta_p_proxy, Greenwald fraction, and cost index dimensionless. | tests/test_global_design_scanner.py positive-domain, metric-shape, and configuration tests; tests/test_disruption_contracts.py production-surface disruption episode smoke coverage with the real scanner; tests/test_disruption_episode.py real GlobalDesignExplorer episode coverage; docs/api.md public API documentation for the bounded scanner | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/gk_ood_detector.py` | OOD detector claims must declare the feature vector, training distribution, distance metric, symmetric positive-definite inverse-covariance calibration, threshold calibration, uncertainty handling, non-negative transport ensemble predictions, and behavior outside the calibrated gyrokinetic operating envelope. | Repository gyrokinetic scheduler OOD contract; statistical process monitoring distribution-shift controls | Feature units inherit declared GK inputs and outputs; detector scores are dimensionless with explicit calibration metadata, Mahalanobis metric provenance, threshold provenance, and transport prediction channels in non-negative diffusivity units. | tests/test_gk_ood_detector.py; tests/test_gk_hybrid_integration.py; validation/validate_gk_ood_calibration.py strict persisted campaign calibration gate with report and artefact digests, Mahalanobis-metric provenance, duplicate-campaign rejection, and acceptance-bound checks | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/gk_online_learner.py` | Online learner claims must declare training-window selection, non-empty train/validation split domains, nonnegative transport-coefficient targets, validation loss, rollback policy, OOD-score sample admission, optimiser settings, persisted retraining decisions, and compatibility boundary with gyrokinetic scheduler inputs. | online learning stability and rollback control references; repository gyrokinetic hybrid learner contract | Inputs inherit GK feature units; transport targets are nonnegative chi_e, chi_i, and D_e; losses and OOD scores are dimensionless with explicit scaling, learning rate, epoch count, generation limit, and threshold metadata. | tests/test_gk_online_learner.py OOD admission, holdout retraining, rollback, decision persistence, and invalid-domain checks; tests/test_gk_hybrid_integration.py scheduler integration checks; validation/benchmark_gk_online_learner.py deterministic bounded online-retraining benchmark | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/gk_species.py` | Species and collision claims must declare charge, mass, density, temperature, thermal speed, Larmor radius, gyroaverage Bessel evaluation, diamagnetic frequency, pitch-angle deflection coefficient, thermal energy-relaxation coefficient, electron-ion mass-ratio scaling, field-temperature dependence, valid species parameter bounds, valid velocity quadrature sizes, and strictly ordered finite lambda grids bounded by the local trapped-passing boundary. | Sugama et al. 2006 collision operator; gyrokinetic normalisation and species parameter contracts | Mass in kilograms, charge in coulombs, density in m^-3, temperature in electronvolts, velocity in m/s, gyrofrequency in rad/s, Larmor radius in metres, collision rates in s^-1, lambda on [0, 1] with lambda times B/B0 no greater than one, and positive magnetic-field ratio. | tests/test_gk_species.py thermal-speed, Larmor-radius, Bessel gyroaverage, diamagnetic-frequency sign, collision-frequency scaling, velocity-grid, and pitch-angle-domain checks; tests/test_gk_electromagnetic.py; validation/validate_gk_species_reference.py immutable species, gyroaverage, diamagnetic-drive, velocity-grid, pitch-angle operator, and collision reference cases with distinct deflection and energy-relaxation channels; validation/validate_gk_collision_independent.py independent Fokker-Planck cross-check of the production collision coefficients against validation/gk_collision_independent_reference.py (velocity-dependent Chandrasekhar deflection frequency Maxwellian-averaged, plus a Braginskii/NRL closed-form anchor and an elastic energy-transfer identity) across density, temperature, effective-charge, field-temperature, and species-mass cases: the production/reference deflection ratio is constant to machine precision (functional scaling matches), the O(1) prefactor is bounded and recorded (0.271 of the Fokker-Planck thermal deflection rate, 0.500 of the Braginskii rate), and the energy-relaxation channel matches the independent elastic energy-transfer efficiency | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/halo_re_physics.py` | Post-disruption claims must declare the halo L/R circuit, contact fraction, toroidal peaking factor, current-quench waveform, Connor-Hastie primary generation, Rosenbluth-Putvinski avalanche generation, ensemble uncertainty assumptions, ITER-limit summaries, strict disruption-reference admission, and claim status. | Fitzpatrick 2002 halo current and error-field interaction; Connor and Hastie 1975 runaway electron generation; Rosenbluth and Putvinski 1997 avalanche generation | SI amperes, mega-amperes, seconds, milliseconds, ohms, henries, metres, volts per metre, rates per second, and meganewtons per metre. | tests/test_halo_re_physics.py halo-current, runaway-electron, ensemble, fail-closed claim evidence, and strict reference-admission regression checks; tests/test_halo_nonfinite_guards.py non-finite guard checks; tests/test_halo_validation_paths.py validation-path guard checks; validation/benchmark_disruption_mitigation_claims.py deterministic bounded halo/runaway claim-admission benchmark; validation/validate_disruption_reference.py strict disruption reference artifact gate; validation/validate_runaway_electron.py validates the Connor-Hastie critical and Dreicer fields, the collision and avalanche time constants, the impurity-aware critical field, and the Rosenbluth-Putvinski avalanche growth rate (threshold, linearity, and RMP deconfinement) against exact closed forms, with tamper-evident sealed evidence; tests/test_runaway_electron_validation.py critical-field, Dreicer, avalanche-rate, threshold, linearity, scaling, and evidence-seal checks; validation/validate_halo_current.py validates the Fitzpatrick halo L/R circuit (halo resistance, halo inductance, mutual inductance, and L/R time constant), the halo-resistance scaling laws, the electromagnetic wall force and toroidal-peaking product, and the fast-circuit quasi-static tracking limit against exact closed forms, with tamper-evident sealed evidence; tests/test_halo_current_validation.py halo-resistance, inductance, mutual-inductance, time-constant, scaling-law, wall-force, TPF-product, quasi-static-tracking, configuration-guard, and evidence-seal checks | bounded_model | [#52](https://github.com/anulum/scpn-control/issues/52) |
| `src/scpn_control/core/stability_mhd.py` | MHD stability claims must declare q-profile interpolation, magnetic shear, interior-point radial grids for profile-resolved criteria, Mercier and Troyon criteria, beta-limit inputs, first-unstable-radius search, bounded bootstrap-current fraction, and the exclusion boundary for full ideal-MHD or resistive-MHD eigenmode claims. | Troyon et al. 1984 normalised-beta limit; Freidberg 2014 Mercier interchange criterion; Connor-Hastie-Taylor 1978 ballooning first-stability boundary; Kruskal-Schwarzschild 1954 external-kink criterion; tokamak q-profile and magnetic-shear references | q and beta dimensionless, plasma current in MA or A with explicit convention, minor radius and major radius in metres, magnetic field in tesla, rho dimensionless, and local bootstrap-current fraction within [0, 1]. | tests/test_stability_mhd.py; tests/test_ballooning_solver.py; tests/test_cross_module_physics.py; validation/validate_mhd_stability.py validates the Troyon beta_N limit and scalings, the Mercier interchange index, the Connor-Hastie-Taylor ballooning boundary, and the Kruskal-Shafranov criterion against exact closed forms with consistent stability flags, with tamper-evident sealed evidence; tests/test_mhd_stability_validation.py Troyon, Mercier, ballooning, Kruskal-Shafranov, and evidence-seal checks; validation/validate_stability_mhd_ballooning.py cross-checks the analytic s-alpha ballooning boundary against a structurally independent NUMERICAL solve of the ballooning ODE (validation/ballooning_independent_reference.py): the analytic alpha_crit(s) approximation tracks the numerically resolved marginal boundary to within ~5% across the first-stability regime s in [0.5, 2.0] (low-shear second stability excluded and recorded) | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/integrated_scenario.py` | Integrated scenario claims must declare coupling order, current diffusion, transport, pedestal, ELM, NTM, sawtooth, burn, controller-feedback interactions, actuator bounds, timestep policy, state exchange units, and failure isolation boundaries. | integrated tokamak scenario modelling references; repository scenario-coupling contract | All coupled states must preserve declared SI or normalised units for current, density, temperature, pressure, q, beta, power, flux, and timing. | tests/test_integrated_scenario.py; tests/test_cross_module_physics.py; tests/test_closed_loop_scenario.py; tests/test_cli_closed_loop_demo.py; validation/benchmark_integrated_scenario_coupling.py; validation/reports/integrated_scenario_coupling.json; validation/reports/integrated_scenario_coupling.md | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/kinetic_efit.py` | Kinetic-EFIT coupling claims must declare Thomson or profile-derived electron density and temperature points, ion-temperature points, fast-ion pressure fraction, anisotropy sigma, MSE pitch-angle constraints when used, radial interpolation geometry, pressure-consistency residual, diagnostic/profile/fast-ion/MSE source provenance, matched pressure, q-profile, and anisotropy reference tolerances, and exclusion from facility-grade P-EFIT unless matched reference equilibria pass the fail-closed admission gate. | Lao et al. 1985 EFIT equilibrium reconstruction; MSE-constrained kinetic EFIT workflow; Repository fixed-boundary realtime-EFIT contract | R and Z in metres, temperatures in keV, density in 10^19 m^-3, MSE pitch angle in degrees, pressure in pascals, beta dimensionless, and q-profile dimensionless. | tests/test_kinetic_efit.py requires measured profile channels, radial interpolation, measured Ti use, anisotropy residuals, MSE q-profile response, fast-ion pressure checks, and fail-closed kinetic-EFIT claim evidence admission; validation/benchmark_kinetic_efit_claims.py publishes deterministic bounded synthetic claim evidence with explicit facility-claim exclusion; validation/reports/kinetic_efit_claims.json records bounded provenance, interpolation geometry, pressure consistency, fast-ion beta, and q-profile endpoints | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gk_eigenvalue.py` | Native linear GK eigenvalue solve must match external TGLF, GENE, GS2, CGYRO, or QuaLiKiz growth rates and real frequencies for the same Miller geometry and species inputs. | Miller et al. 1998 local equilibrium; GENE and GACODE published input-output contracts | Growth rate and frequency normalised consistently to c_s/a or declared external-code convention. | ROADMAP.md local-dispersion overprediction note; docs/competitive_analysis.md linear GK quantitative accuracy status; validation/validate_gk_crosscode.py strict real-binary schema-versioned and digest-bound evidence gate | external_dependency_blocked | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/momentum_transport.py` | Momentum transport claims must declare NBI torque, collisional damping, viscous momentum diffusion, rotation-profile boundary conditions, angular-momentum units, and the coupling boundary to the integrated transport solver. | Stacey and Sigmar 1985 NBI torque; Hinton and Hazeltine 1976 radial force balance; Burrell 1997 E x B shearing rate; Rice et al. 2007 intrinsic-rotation scaling; repository NBI torque and rotation-profile contract | SI newton metres, kg m^2/s^2, rad/s, metres, seconds, density in m^-3, and momentum diffusivity in m^2/s. | tests/test_momentum_transport.py verifies NBI torque direction, viscous diffusion damping, implicit bounded collisional damping, fail-closed damping-profile validation, force-balance helpers, and diagnostic guards; tests/test_momentum_integration.py; validation/validate_momentum_transport.py validates the NBI torque, the Hinton-Hazeltine radial electric field (constant and linear pressure), the Burrell E x B shearing rate, the Biglari-Diamond-Terry suppression factor, the Rice intrinsic-rotation scaling, and the toroidal Mach number against exact closed forms, with tamper-evident sealed evidence; tests/test_momentum_transport_validation.py NBI-torque, force-balance, E x B-shearing, Rice-scaling, Mach, and evidence-seal checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/multi_shot_campaign.py` | Multi-shot campaign replay evidence that consumes pulsed-MPC decisions must preserve each admitted decision digest as a lowercase SHA-256 value, bind the digest count into the campaign payload hash, and carry the digest through Python, Rust, PyO3, and replay v1.1 extension fields without changing the scheduler or capacitor-bank physics. | Repository pulsed-scenario lifecycle contract; Repository capacitor-bank RLC admission contract; Repository pulsed-MPC decision evidence contract; Repository geometry-neutral replay v1.1 pulsed metadata contract | Shot timestamps are seconds, capacitor energy is joules, trigger timestamps are nanoseconds, pulse identifiers are UUID-formatted deterministic IDs, and pulsed-MPC admission digests are lowercase SHA-256 hex strings over the upstream decision evidence. | tests/test_multi_shot_campaign.py Python campaign report, replay extension, invalid digest, and optional PyO3 table-bridge checks; scpn-control-rs/crates/control-control/tests/multi_shot_campaign.rs native Rust campaign digest and invalid-digest checks; benchmarks/bench_multi_shot_campaign.py local regression benchmark with per-shot pulsed-MPC digest evidence; scpn-control-rs/crates/control-control/examples/bench_multi_shot_campaign.rs native Rust local regression benchmark with per-shot pulsed-MPC digest evidence; validation/reports/multi_shot_campaign_pulsed_mpc_evidence_python_pyo3_20260604T172543Z.json Python and PyO3 local regression report with per-shot pulsed-MPC digest evidence; validation/reports/multi_shot_campaign_pulsed_mpc_evidence_rust_20260604T173322Z.json native Rust local regression report with per-shot pulsed-MPC digest evidence and benchmark context; validation/validate_multi_shot_campaign_evidence.py release-admission gate for Python, PyO3, Rust, digest count, SHA-256, and benchmark-context evidence; tests/test_multi_shot_campaign_evidence_validation.py module-specific release-admission checks; tests/test_release_evidence_validation.py top-level release-evidence mandatory gate checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/neural_equilibrium.py` | Neural equilibrium pretraining may use bounded synthetic Solovev-like equilibria, but fine-tuning and public predictive claims require fail-closed claim evidence tied to the same weight checksum and comparison against identical P-EFIT or documented public reference equilibria for psi, pressure, q-profile, LCFS boundary, and magnetic-axis position. | EFIT/P-EFIT equilibrium reconstruction workflow; Repository neural equilibrium model contract; Bounded Solovev-like synthetic equilibrium pretraining contract | SI magnetic flux, pressure, metre-scale geometry, and dimensionless q-profile arrays on declared grids. | tests/test_neural_equilibrium.py synthetic pretraining, reproducibility, JAX-compatible weight, fail-closed claim evidence, checksum matching, and real-EFIT admission tests; tests/test_jax_neural_equilibrium.py synthetic-pretrained weight loading through JAX inference; validation/benchmark_neural_equilibrium_pretraining.py deterministic synthetic pretraining and claim-admission benchmark; validation/validate_neural_equilibrium_reference.py strict digest-bound P-EFIT/reference artefact gate with report schema, payload digest, artefact file digest, portable paths, duplicate reference-set rejection, and blocked public-claim state; validation/convert_mast_efm_neural_equilibrium_reference.py converts public MAST EFM measured-shot Zarr campaigns into checksum-bound reference-candidate arrays while preserving strict predictive-claim blockage pending exact-model predictions and metric admission; validation/evaluate_mast_efm_neural_equilibrium.py evaluates current neural-equilibrium weights against converted public MAST EFM reference-candidate arrays with exact profile_r/profile_z grids, writes prediction artefacts, reports flux masked RMSE plus derived magnetic-axis and LCFS residuals, and preserves fail-closed predictive-claim blockage | external_dependency_blocked | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/neural_transport.py` | Neural transport claims must declare input feature normalisation, QLKNN weight provenance, prediction targets, fallback critical-gradient thresholds, bounded density-channel particle diffusivity, bounded profile closure provenance, local benchmark errors, strict reference-artifact admission, weight checksum matching, uncertainty output, out-of-domain handling, and cross-validation against reference transport cases. | QuaLiKiz neural network transport surrogate references; repository QLKNN weight and metric contract | Inputs and outputs use declared transport feature units, diffusivity in m^2/s or declared normalisation, fluxes in SI or gyro-Bohm units with conversion metadata. | tests/test_neural_transport.py density-gradient and shear-dependent fallback particle diffusivity; tests/test_neural_transport_core.py profile fallback particle-channel checks, bounded closure provenance, fallback gating, fail-closed claim evidence, and checksum-matched reference admission checks; tests/test_qlknn_transport.py; validation/benchmark_neural_transport_claims.py deterministic local claim-admission benchmark; validation/validate_neural_transport_reference.py strict digest-bound QuaLiKiz/reference artifact gate | external_dependency_blocked | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/neural_turbulence.py` | Neural turbulence claims must declare QLKNNSurrogate inputs, finite strictly ordered physical profile grids, feature scaling, banana-regime electron collisionality, bounded analytic quasilinear target variables, fallback behaviour, uncertainty handling, local analytic-target benchmark errors, strict reference-artifact admission, exact weight checksum matching, and cross-validation boundary against gyrokinetic or quasilinear turbulence references. | QLKNN turbulence surrogate references; repository neural turbulence surrogate contract | Feature and target units follow the declared turbulence surrogate schema; diffusivities, growth rates, and fluxes require explicit SI or normalised-unit metadata. | tests/test_neural_turbulence.py collisionality scaling, analytic target, training, save-load, denormalisation, fail-closed claim evidence, and checksum-matched reference admission checks; validation/benchmark_neural_turbulence_claims.py deterministic local claim-admission benchmark; validation/validate_neural_turbulence_reference.py strict GK-campaign/reference artifact gate | validation_gap | [#50](https://github.com/anulum/scpn-control/issues/50) |
| `src/scpn_control/core/gk_nonlinear.py` | Five-dimensional delta-f flux-tube Vlasov evolution with E cross B bracket, ballooning connection, kinetic electrons, and Sugama collision terms. | Dimits et al. 2000 Cyclone Base Case; Sugama et al. 2006 collision operator | Gyro-Bohm-normalised heat flux and chi_i with explicit R/L_Ti, rho_s, c_s, and reference gradient normalisation. | ROADMAP.md v0.18.0 open revalidation item; docs/joss_paper.md nonlinear GK validation limitation; validation/gk_nonlinear_cyclone.py saturation-evidence assessor; validation/reference_data/cyclone_base/cyclone_nonlinear_chi_i_reference.json obtained published cross-code reference (Dimits et al. 2000, verified at source): offset-linear fit chi_i L_n/(rho_i^2 v_ti) = 15.4(1 - 6.0 L_T/R) giving ~2.0 at R/L_T=6.9, Dimits shift (linear critical R/L_T ~4.0, nonlinear effective ~6.0), and cross-model factors, with an explicit unit-conversion caveat to the module's chi_i_gB normalisation | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/core/orbit_following.py` | Orbit-following claims must declare guiding-centre equations, magnetic moment handling, finite pitch-angle bounds, banana-orbit width scaling, collision or loss assumptions, ensemble sampling, valid magnetic-geometry bounds, geometry/particle/collision/loss-boundary provenance, and matched banana-width and loss-fraction reference tolerances before external orbit-code or measured fast-ion claims. | Cordey 1981 orbit-width estimate; Boozer 2004 guiding-centre equations of motion; White 2014 guiding-centre canonical invariants; tokamak guiding-centre orbit references | SI metres, seconds, tesla, electronvolts, kg, coulombs, pitch angle in radians on [0, pi], and ensemble probabilities dimensionless. | tests/test_orbit_following.py verifies guiding-centre domain checks, classifier boundaries, Stix slowing-down, banana-width scaling, first-orbit-loss scaling, ensemble counts, and fail-closed orbit-following claim evidence admission; validation/validate_guiding_centre_conservation.py integrates the production GuidingCenterOrbit RK4 stepper in a static analytic axisymmetric field and verifies exact conservation of kinetic energy and canonical toroidal momentum over passing and trapped orbits for deuterons and 3.5 MeV alphas, with tamper-evident sealed evidence; tests/test_guiding_centre_conservation_validation.py divergence-free-field, energy/momentum-conservation, passing/trapped-coverage, degenerate-baseline, and evidence-seal checks; validation/benchmark_orbit_following_claims.py publishes deterministic bounded synthetic claim evidence with explicit external-code claim exclusion; validation/reports/orbit_following_claims.json records bounded geometry, particle, collision, loss-boundary, banana-width, first-orbit-loss, and ensemble-count evidence; validation/validate_orbit_reference.py strict orbit reference artefact gate | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/fusion_neural_mpc.py` | Pulsed-shot MPC admission claims must bind the first gradient-MPC action to the pulsed lifecycle state, capacitor-bank feasibility result, constraint slack, objective value, burn-action mask, safe action, peak-current estimate, and schema-versioned admission digest before replay or campaign evidence can rely on the command. | Repository pulsed-scenario lifecycle contract; Repository capacitor-bank RLC admission contract; Repository gradient MPC admission adapter contract | State vectors use the MPC surrogate units declared by the caller; action components and safe-action components share coil/current actuator units; capacitor-bank slack is joules; pulse duration is seconds; peak current is amperes; digests are SHA-256 hex strings. | tests/test_fusion_neural_mpc_pulsed_adapter.py module-specific Python admission and decision-digest checks; tests/test_fusion_neural_mpc_pulsed_adapter_rust_parity.py optional PyO3 parity checks for evidence fields; scpn-control-rs/crates/control-control/tests/pulsed_mpc_adapter.rs native Rust admission and decision-digest checks; benchmarks/bench_pulsed_mpc_adapter.py local regression benchmark with decision evidence; scpn-control-rs/crates/control-control/examples/bench_pulsed_mpc_adapter.rs native Rust local regression benchmark with decision evidence; validation/reports/pulsed_mpc_adapter_pyo3_decision_evidence_python_20260604T171015Z.json PyO3-inclusive local regression report; validation/reports/pulsed_mpc_adapter_pyo3_decision_evidence_rust_20260604T171015Z.json native Rust local regression report | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/realtime_efit.py` | Real-time equilibrium reconstruction claims must declare the Grad-Shafranov forward solve (fixed-boundary five-point Delta-star operator with polynomial p-prime and FF-prime source profiles), the least-squares response-function inverse (Tikhonov-regularised weighted least-squares over the p-prime/FF-prime basis with an outer Picard loop over the normalised-flux geometry, reporting the measured chi-squared, iteration count, and plasma current recovered from the fitted flux rather than forced), the optional free-boundary mode (per-coil toroidal Green's-function flux columns with a von Hagenow free-space boundary condition and a joint coil-current least-squares fit), diagnostic response interpolation, macroscopic-shape extraction from the boundary contour, diagnostic provenance, matched-reference source, psi/Ip/q95/beta_pol/li tolerances, and facility-claim admission status. | Lao et al. 1985 EFIT equilibrium reconstruction; Strait et al. 2019 real-time EFIT workflow references; Repository GEQDSK and magnetic-diagnostic validation contracts | SI metres, webers per radian, tesla, amperes, pascals per weber, FF-prime units, and dimensionless q95, beta_pol, li, elongation, and triangularity. | tests/test_realtime_efit.py fixed-boundary and free-boundary least-squares inverse closure (psi, coil-current, and recovered-Ip recovery) and diagnostic reconstruction checks; tests/test_kinetic_efit.py kinetic EFIT integration checks; tests/test_validate_real_shots_equilibrium.py GEQDSK source residual checks; tests/test_realtime_efit.py claim-evidence provenance, matched-reference admission, and invalid-reference checks; validation/benchmark_efit_lite_claims.py bounded synthetic EFIT-lite claim-admission benchmark; validation/reports/efit_lite_claims.json bounded synthetic EFIT-lite claim report | bounded_model | [#48](https://github.com/anulum/scpn-control/issues/48) |
| `src/scpn_control/core/gyrokinetic_transport.py` | Reduced gyrokinetic transport claims must declare the dispersion relation, instability branch classification, quasilinear heat-flux closure, critical-gradient threshold, saturation rule, positive integer mode-count domain, tokamak geometry ordering, and mapping from local GK outputs to transport coefficients. | Cyclone Base Case gyrokinetic benchmark; quasilinear gyrokinetic transport closure references | Growth rates and frequencies in declared normalised units, gradients dimensionless or m^-1 with explicit convention, rho and epsilon dimensionless, geometry in metres, heat flux in gyro-Bohm or SI units with conversion metadata. | tests/test_gyrokinetic_transport.py dispersion branches, quasilinear fluxes, saturation monotonicity and boundedness, profile evaluation, and invalid-domain checks; tests/test_gk_benchmark_linear.py | validation_gap | [#47](https://github.com/anulum/scpn-control/issues/47) |
| `src/scpn_control/control/gym_tokamak_env.py` | 0D/1D control plants must declare reduced-state equations, conservation assumptions, and controller-validity bounds before being used for hardware or facility claims. | Wesson 2011 tokamak transport references; Repository Gymnasium plant and digital-twin contracts | Declared plasma current, temperature, density, confinement, and actuator units per state vector field. | README.md validation limitation; docs/use_cases.md synthetic disruption prediction status; tests/test_gym_tokamak_env.py reduced-order plant bounds and action-contract checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/control/rwm_feedback.py` | RWM feedback claims must declare the no-wall and ideal-wall beta limits, resistive-wall L/R time, optional wall-plasma gap correction, rotation-stabilisation term, active proportional or derivative feedback coupling, controller latency, coil coupling, sensor/coil topology, ideal-kink exclusion boundary, reference source, closed-loop growth-rate tolerance, and facility-claim admission status. | Bondeson and Ward 1994 resistive-wall-mode growth-rate model; Fitzpatrick 2001 rotation-stabilisation contribution; Strait et al. 2003 wall-plasma gap correction; Garofalo et al. 2002 active feedback experiments | Normalised beta dimensionless, wall time in seconds, toroidal rotation in rad/s, wall and plasma radii in metres, feedback gain dimensionless per declared coil coupling, growth rate in s^-1. | tests/test_rwm_feedback.py growth-rate, rotation, wall-geometry, feedback, ideal-kink, and required-gain checks; tests/test_rwm_feedback.py claim-evidence provenance, external tolerance admission, and invalid-domain checks; validation/benchmark_rwm_claims.py bounded local RWM claim-admission benchmark; validation/reports/rwm_claims.json bounded local RWM claim report; validation/validate_rwm_feedback.py validates the Bondeson-Ward growth rate, wall-gap tau_eff, Fitzpatrick rotation stabilisation, critical-rotation marginality, required-feedback-gain marginalisation, ideal-kink and no-wall boundaries, and 1/tau_wall scaling against exact closed forms, with tamper-evident sealed evidence; tests/test_rwm_feedback_validation.py growth-rate, rotation, critical-rotation, feedback-marginality, boundary, and evidence-seal checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/tearing_mode_coupling.py` | Sawtooth-to-NTM seeding claims must declare sawtooth crash trigger, seed-island generation, rational-surface coupling, phase and amplitude assumptions, bounded bootstrap-current fraction, NTM coupling path, and exclusion of full nonlinear MHD crash dynamics. | sawtooth crash and NTM seeding references; repository tearing-mode coupling contract | Time in seconds, island width in metres, q dimensionless, rho dimensionless, phase in radians, local bootstrap-current fraction within [0, 1], and amplitude units declared per coupling signal. | tests/test_tearing_mode_coupling.py; tests/test_integrated_scenario.py | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/tokamak_digital_twin.py` | Digital-twin claims must declare the 2D poloidal grid, q-profile topology update, rational-surface island mask, finite-difference diffusion, radiation-loss scaling with density and effective charge, actuator latency, RNG seed, IDS export assumptions, external simulator artifact provenance, and bounded online model-update parameters. | Repository digital-twin runtime contract; Wesson 2011 radiation-loss reference; TRANSP integrated modelling evidence contract; TSC time-dependent simulation evidence contract; Bayesian optimisation for bounded model calibration | Declared normalised grid indices, keV, m^-3, dimensionless q-profile values, actuator-lag summary units, RNG seed metadata, and IDS-compatible pulse history fields. | tests/test_tokamak_digital_twin.py deterministic twin checks; tests/test_digital_twin_physics.py physics-path checks; tests/test_digital_twin_online_update.py external artifact, loss, Bayesian-update, and deterministic benchmark checks; tests/test_digital_twin_reference_validation.py strict digital-twin artifact gate including TRANSP and TSC; validation/benchmark_digital_twin_online_update.py deterministic bounded online-update benchmark; validation/validate_digital_twin_reference.py strict reference-artifact gate | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |
| `src/scpn_control/core/integrated_transport_solver.py` | Integrated transport claims must declare the axis-to-edge radial grid, heat and particle diffusion equations, neoclassical closure, bootstrap-current fit, source deposition terms, boundary conditions, timestep limits, and coupling contracts to pedestal and gyrokinetic closures. | Chang-Hinton neoclassical transport model; Sauter et al. 1999 bootstrap-current fits; Wesson 2011 radial transport equations; Crank and Nicolson 1947 implicit diffusion scheme; Abramowitz and Stegun 1965 Bessel functions; repository integrated transport solver contract | SI metres, seconds, m^-3, eV, W/m^3, m^2/s, amperes per square metre, and dimensionless profile coordinates with explicit axis-to-edge normalisation metadata. | tests/test_integrated_transport_solver.py; tests/test_transport_energy_conservation.py; tests/test_transport_neoclassical_guards.py; validation/validate_transport_diffusion.py validates the production cylindrical heat-diffusion operator against the exact Bessel eigenvalue and the Crank-Nicolson tridiagonal solve against a manufactured steady state, both at second order, and checks Python/Rust Thomas-solver parity, with tamper-evident sealed evidence; tests/test_transport_diffusion_validation.py operator-eigenvalue, manufactured-steady-state, Python/Rust Thomas-parity, and evidence-seal checks | validation_gap | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/core/sol_model.py` | Two-point SOL claims must declare the connection length L_par = pi q95 R0, the parallel heat-flux mapping with the poloidal-to-total field ratio, the Spitzer-Härm upstream conduction integral q_par = kappa_0 T_u^{7/2}/(7/2 L_par), the upstream-target pressure balance n_u T_u = 2 n_t T_t, the Eich heat-flux-width regression exponents, the divertor peak-heat-flux geometry, the sheath heat-transmission target temperature, the detachment-onset criterion, and exclusion of full edge-transport facility validation. | Stangeby 2000 two-point model and sheath heat transmission; Eich et al. 2013 scrape-off-layer heat-flux-width regression; Lipschultz et al. 1999 divertor detachment onset | SI metres, megawatts, density in 10^19 m^-3, temperature in eV, heat flux in MW/m^2, connection length in metres, and dimensionless inverse aspect ratio, safety factor, and radiated fraction. | tests/test_sol_model.py two-point, Eich, peak-heat-flux, and detachment checks; validation/validate_sol_two_point.py validates the connection length, parallel-flux mapping, Spitzer-Härm upstream conduction integral, pressure balance, Eich regression exponents, peak target heat flux, and the detachment density boundary against exact closed forms, with tamper-evident sealed evidence; tests/test_sol_two_point_validation.py conduction, pressure-balance, Eich-exponent, peak-flux, detachment-boundary, and evidence-seal checks | bounded_model | [#49](https://github.com/anulum/scpn-control/issues/49) |
| `src/scpn_control/control/volt_second_manager.py` | Volt-second management claims must declare inductive flux, resistive loop-voltage integration, Ejima startup flux, bootstrap-current correction, flat-top duration estimate, ramp and ramp-down decomposition, flux-budget margin assumptions, finite positive machine constants, nonnegative current and voltage domains, positive timesteps, and strictly ordered bootstrap-profile grids. | Wesson 2011 tokamak loop-voltage and flux-balance equations; Ejima et al. 1982 startup flux coefficient; ITER Physics Basis 1999 flat-top flux-budget references; Repository fail-closed volt-second claim-admission contract | SI webers, volt-seconds, henries, ohms, amperes, mega-amperes, seconds, metres, normalised profile radius, and dimensionless Ejima coefficient. | tests/test_volt_second_manager.py flux budget, scenario analysis, bootstrap-profile, domain-boundary, and claim-admission checks; validation/validate_volt_second_reference.py strict volt-second reference artifact gate; validation/benchmark_volt_second_claims.py bounded claim-admission benchmark; validation/validate_volt_second.py validates the inductive flux L_p I_p, the Ejima startup flux C_E mu0 R0 I_p, the resistive ramp integral, the flat-top budget closure (flat-top resistive consumption equals remaining flux at tau_flat), the ramp/flat-top/ramp-down scenario decomposition and margin, the consumption integrator, the linear ramp optimiser, and the inductive and Ejima flux scaling laws against exact closed forms, with tamper-evident sealed evidence; tests/test_volt_second_validation.py inductive-flux, Ejima-flux, resistive-ramp, flat-top-closure, scenario-decomposition, consumption-integrator, ramp-optimiser, scaling-law, configuration-guard, and evidence-seal checks | bounded_model | [#51](https://github.com/anulum/scpn-control/issues/51) |

## Components

### DIII-D experimental replay

- Fidelity status: `validation_gap`
- Module path: `validation/reference_data/diiid`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 0
- Required actions:
  - Keep synthetic GEQDSK and disruption-shot fixture manifests covered by checksum verification
  - Acquire measured DIII-D artefacts through the MDSplus acquisition spec path before restoring any real replay claim

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
  - Validate ELM frequency, crash depth, pedestal profile drops, RMP suppression windows, and peak heat flux against measured H-mode campaign or published ELM cases
  - Persist schema-versioned pre-crash, post-crash, event-catalog, and RMP artifact URIs with SHA-256 digests, pedestal rho grids, event windows, Type-I energy-fraction bounds, metrics, tolerances, and canonical payload digests with every ELM validation artefact
  - Keep full-fidelity ELM/RMP claims blocked until measured or documented public reference artifacts pass the strict admission gate

### EPED pedestal and peeling-ballooning approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/eped_pedestal.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate pedestal height and width against published EPED benchmark points or measured pedestal databases
  - Record bootstrap-current, beta-limit, and shaping inputs for every pedestal validation artefact
  - Keep full-fidelity EPED pedestal claims blocked until measured pedestal-database or documented public reference artifacts pass the strict admission gate

### Fusion-kernel reduced-order phase-sync leaf

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gs_phase_sync.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Keep FusionKernel thin wrappers as the product surface under dual-home C
  - Validate reduced-order phase-sync helpers against mean-field Kuramoto benchmarks before plasma-phase control claims
  - Do not claim a full plasma-phase control law from this leaf alone

### Grad-Shafranov fusion-kernel numerical contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/fusion_kernel.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Validate reconstructed equilibria against EFIT, GEQDSK, or published fixed-boundary benchmark cases
  - Preserve residual norms, convergence metadata, and Rust/Python parity evidence for every validation artefact
  - Replace analytic Solov'ev backend evidence with matched EFIT, GEQDSK, or published fixed-boundary benchmark evidence before any facility reconstruction claim

### JAX gyrokinetic numerical parity guard

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/jax_gk_solver.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Run backend parity over CBC, kinetic-electron TEM, and stable-mode cases with pinned tolerances
  - Keep CPU/GPU backend parity evidence current for every release and continue blocking quantitative GK claims until external-code validation artefacts are supplied

### Kadomtsev sawtooth crash and reconnection contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/sawtooth.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate crash timing, mixing radius, and post-crash profiles against measured or published sawtooth-crash cases before facility claims
  - Persist pre-crash q-profile, q=1 radius, mixing radius, and redistribution metadata with each validation case

### Kuramoto-Sakaguchi phase synchronisation runtime

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/phase/kuramoto.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Synchronisation onset and order-parameter metrics are validated against the exact mean-field Lorentzian Kuramoto results in validation/validate_kuramoto_synchronisation.py; remaining work is plasma-phase control-law validation before broader phase-control claims
  - Generate current deployment-target Rust/Python parity and timestep-convergence evidence with validation/benchmark_kuramoto_runtime_evidence.py before deployment-target phase claims

### MARFE radiation-condensation density-limit contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/marfe.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate onset temperatures and density limits against measured or published MARFE cases
  - Add impurity-specific radiation tables or documented provenance for each supported impurity species
  - Keep full-fidelity MARFE density-limit claims blocked until measured MARFE campaign or documented public reference artifacts pass the strict admission gate

### Miller local-equilibrium geometry and field-pitch contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_geometry.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Compare metric coefficients and field-pitch factors against an independent Miller-geometry reference implementation or external equilibrium-code evidence
  - Keep immutable schema-versioned evidence covering circular, shaped, and high-shear local equilibria

### NTM island evolution and control approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/ntm_dynamics.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate island growth and suppression against measured or published NTM benchmark cases
  - Persist q-profile, rational-surface, seed-island, and ECCD alignment metadata with each validation case
  - Keep full-fidelity NTM forecasting and suppression claims blocked until measured NTM campaign or documented public reference artifacts pass the strict admission gate

### PREEMPT_RT runtime admission evidence

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/runtime_admission.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Capture production timing evidence only on a PREEMPT_RT or otherwise qualified realtime kernel with SCHED_FIFO or SCHED_RR execution, performance governors, hard-isolated cores, IRQ shielding, and documented host load
  - Keep local workstation reports in fail-closed local-regression mode with production_claim_allowed false until qualified runtime evidence is present
  - Keep Python, PyO3, Rust, CLI, release-evidence, and benchmark report admission semantics in parity whenever runtime-admission policy changes

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
  - Validate certification packages with an external safety-case review before hardware-control claims
  - Promote optional SMT-backed proof obligations only after solver-specific proof artifacts are persisted
  - Require schema-versioned Z3 report payload digests to match safety-critical proof manifests before artifact admission
  - Reject duplicate JSON keys through the public Z3 report loader and safety-critical artifact report-root manifests
  - Reject unknown Z3 report top-level and proof-section fields through the public report loader and safety-critical artifact report-root manifests
  - Reject malformed or foreign-field Z3 counterexample records before admitting bounded proof evidence
  - Reject inconsistent Z3 solver-status, holds, and counterexample combinations before admitting bounded proof evidence
  - Reject blocked Z3 reports that carry proof depth, proof specifications, or a live solver label
  - Reject duplicate or malformed Z3 proof-section checked_specs before top-level proof-manifest matching
  - Resolve typed readiness artifact URIs under an explicit artifact root and validate target-hardware timing reports through the E2E latency evidence gate before promotion readiness.
  - Require qualified HIL replay, HDL export, CODAC/EPICS runtime, and WebSocket runtime evidence artifacts before controller safety-case promotion readiness.
  - Admit safety-critical `.scpnctl` artifacts only through the bounded formal-verification manifest gate
  - Bind admitted proof manifests to the exact canonical artifact payload digest and declared report digest
  - Reject Lean proof reports with solver/version drift or unsupported proved-contract overclaims before safety-critical artifact admission
  - Reject Lean proof reports that pad admitted PID/SNN evidence with unrelated theorem namespaces, module paths, or safety-case IDs
  - Apply the same Lean exact-link admission policy to `.scpnctl` artifact manifests before report-root byte comparison
  - Reject unknown Lean report and artifact formal-verification fields instead of ignoring stale or foreign proof evidence
  - Reject digestless Lean proof reports by requiring the canonical payload_sha256 self-digest
  - Reject duplicate JSON keys through the public Lean report loader and release validation executable
  - Reject duplicate JSON keys in Lean reports reached through safety-critical artifact report-root manifests
  - Bind promoted controller safety-case bundles to the same controller artifact digest across formal proof, differentiable-transport, and digital-twin update evidence
  - Persist controller safety-case bundles with schema-versioned integrity manifests before replay admission
  - Keep controller safety-case readiness blocked until external physics validation, target-hardware timing, qualified runtime/hardware evidence, and independent safety-review digests are all present
  - Prefer typed readiness artifacts with safe relative URIs, producers, timestamps, and kind-specific SHA-256 digests over anonymous promotion digests
  - Persist controller safety-case readiness decisions with schema-versioned integrity manifests before promotion replay
  - Publish optional Z3 bounded-model-checking artifacts with every SMT-backed proof obligation
  - Treat missing `z3-solver` as blocked SMT evidence, not as a successful proof
  - Reject Z3 unknown solver sections that carry counterexamples before admitting bounded proof evidence
  - Reject pass/fail Z3 reports that reuse the unavailable solver label before admitting bounded proof evidence
  - Reject pass/fail Z3 reports whose solver metadata does not identify z3-solver before admitting bounded proof evidence

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
  - Validate blob velocity and spreading against measured probe-campaign or published SOL filament cases
  - Persist schema-versioned reference, profile, and detector artifact URIs with SHA-256 digests, SOL unit contracts, separatrix-to-wall coordinates, detector event domains, blob-size domains, magnetic-geometry metadata, metrics, tolerances, and canonical payload digests with each validation artefact
  - Keep full-fidelity SOL blob-transport claims blocked until measured or documented public reference artifacts pass the strict admission gate

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
  - Persist benchmark plants, frequency grids, D-scale fits, mu upper/lower bounds, and canonical evidence payload digests for every promoted controller claim
  - Tighten the static D-scaling descent so the bound becomes D-scaling invariant to machine precision (currently a recorded diagnostic with a per-orientation spread)

### checkpoint state serialisation boundary contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/checkpoint.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#53](https://github.com/anulum/scpn-control/issues/53) — Hardware, HDL, CODAC/EPICS, and runtime deployment evidence
- Covered source paths: 1
- Required actions:
  - Version checkpoint schemas before long-running production campaigns depend on replay compatibility
  - Add migration fixtures for every future checkpoint schema change

### coupled differentiable scenario facade contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/differentiable_scenario.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Replace the bounded analytic Solov'ev-form surface with matched EFIT, P-EFIT, or documented public-reference equilibrium evidence before facility-facing scenario-gradient claims
  - Validate coupled source-schedule gradients against measured discharges or published integrated-modelling benchmarks before predictive controller-tuning claims
  - Promote timing evidence only after isolated hardware runs record affinity, host load, governor context, backend, dtype, and repeated latency distributions
  - Bind any promoted coupled-scenario evidence to the safety-critical controller proof artifact digest when used in a controller safety case

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
  - Persist backend, dtype, runtime-provenance, radial-grid, boundary-condition, rollout-length, equilibrium-grid, flux-weighting, gradient-tolerance, gradient-audit digest, and replay-drift metadata for controller-tuning campaigns
  - Replace bounded reduced-gyrokinetic and JAX GK stiffness closures with externally validated GK transport coefficients before quantitative transport-control claims
  - Bind promoted differentiable-transport evidence to the safety-critical controller proof artifact digest when used in a controller safety case

### digital twin online model updating

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/digital_twin_online_update.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Provide validated TRANSP, TSC, measured-discharge, or documented public-reference artifacts before measured replay claims
  - Validate online update trajectories against replayed or measured discharge histories before deployment use
  - Persist simulator artifact hashes, units, case ids, strictly increasing time bases, observation digests, prior digests, result digests, finite loss-history minima, parameter-domain admission, and baseline-improvement evidence for every promoted update campaign
  - Bind promoted online-update evidence to the safety-critical controller proof artifact digest when used in a controller safety case

### direct free-boundary tracking controller

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/free_boundary_tracking.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Attach strict documented public, measured-replay, or external equilibrium benchmark artifacts before facility-control claims
  - Attach actuator, latency, and sensor-calibration evidence for the target device before deployment claims

### disruption ROC scoring and warning-time core

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/disruption_roc.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Replace the n=3-from-n=2 bounded approximation with a measured n=3 toroidal diagnostic channel when available
  - Validate the scoring convention and ROC/warning-time metrics against measured labelled disruption windows before any predictive claim

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

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/disruption_sequence.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate phase timing, post-quench temperature, and mitigation branches against labelled measured disruption windows
  - Persist shot identifiers, phase labels, thermal-quench duration, radiation-time assumption, timing tolerances, and mitigation metadata for sequence validation

### disruption-risk heuristic baseline

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/disruption_predictor.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#48](https://github.com/anulum/scpn-control/issues/48) — Equilibrium and reconstruction reference artefacts
- Covered source paths: 1
- Required actions:
  - Train or fit on a real disruption database before any disruption-prediction facility claim
  - Until then keep the docstring and docs labelling it a heuristic baseline, not a trained model

### external gyrokinetic interfaces

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/gk_interface.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Run each interface against a real executable or documented public reference output
  - Persist schema-versioned artifacts with safe deck/raw-output/parsed-output URIs and SHA-256 digests
  - Promote parser fixtures from mock subprocesses to immutable external-code artefacts sealed by canonical payload digests

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
  - Generate schema-versioned replay evidence for every promoted replay and keep measured or benchmark device claims blocked until external artefact digests and non-synthetic magnetic provenance are supplied

### global tokamak design scanner for disruption episodes

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/global_design_scanner.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#52](https://github.com/anulum/scpn-control/issues/52) — Disruption, halo-current, and mitigation benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Replace bounded scalar design proxies with a systems-code design scan before device-design optimisation claims
  - Validate Q, confinement, beta, wall loading, and cost metrics against matched reference designs before facility or economics claims
  - Keep disruption episodes labelled as bounded contract exercises until measured disruption and design evidence pass strict admission

### gyrokinetic OOD detector distribution-bound contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/gk_ood_detector.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47) — External gyrokinetic validation artefacts
- Covered source paths: 1
- Required actions:
  - Calibrate thresholds against real or published GK campaign ensembles
  - Keep strict schema-versioned calibration evidence with Mahalanobis-metric provenance, false-positive, false-negative, and OOD-recall acceptance criteria for deployment gating

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
  - Persist SHA-256 digests for external input decks, external outputs, native inputs, and canonical comparison reports before full-fidelity GK claims

### momentum transport and torque-balance approximation contract

- Fidelity status: `validation_gap`
- Module path: `src/scpn_control/core/momentum_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate torque deposition and rotation evolution against measured or published NBI momentum cases
  - Add acceptance artefacts for low-torque, high-torque, and sign-changing rotation profiles

### multi-shot campaign pulsed-MPC replay evidence

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/multi_shot_campaign.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Attach measured scheduler telemetry, facility interlock evidence, and target-hardware actuator path evidence before PCS or facility-control claims
  - Keep Python, Rust, PyO3, and benchmark campaign evidence fields in parity whenever multi-shot replay metadata changes

### neural equilibrium cross-validation

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_equilibrium.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Acquire matched P-EFIT reference equilibria or an openly redistributable equivalent
  - Persist schema-versioned reference and prediction artefact URIs with SHA-256 digests, exact weight checksum, grid shape, target schema, unit contracts, and declared psi/pressure/q-profile/boundary/axis tolerances
  - Keep predictive equilibrium claims blocked until immutable reference artefacts are sealed by canonical payload digests and validate the exact surrogate weight checksum

### neural transport surrogate validation contract

- Fidelity status: `external_dependency_blocked`
- Module path: `src/scpn_control/core/neural_transport.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#50](https://github.com/anulum/scpn-control/issues/50) — Neural surrogate validation artefacts
- Covered source paths: 1
- Required actions:
  - Acquire or generate immutable reference QLKNN or transport benchmark cases for cross-validation
  - Persist schema-versioned reference and prediction artifact URIs with SHA-256 digests, exact weight checksum, QLKNN-10D feature schema, target schema, uncertainty metadata, OOD decisions, reference metrics, tolerances, and admission status with every surrogate evaluation
  - Gate quantitative QuaLiKiz, QLKNN, or documented-reference transport claims on strict reference artifacts sealed by canonical payload digests and matched to the exact neural weight checksum

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

### pulsed-shot MPC admission evidence

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/fusion_neural_mpc.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Attach measured scheduler telemetry, capacitor-bank interlock evidence, and facility pulse-response artefacts before PCS or facility-control claims
  - Keep Python, Rust, PyO3, and benchmark decision-evidence fields in parity whenever pulsed-MPC admission changes

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

### two-point scrape-off-layer divertor model contract

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/core/sol_model.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#49](https://github.com/anulum/scpn-control/issues/49) — Transport, edge, MHD, and scenario benchmark artefacts
- Covered source paths: 1
- Required actions:
  - Validate upstream and target conditions against measured divertor probe or published edge-transport benchmark cases before facility claims
  - Persist geometry, power, density, radiated fraction, and detachment metadata with each validation case

### volt-second budget and flux-consumption manager

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control/volt_second_manager.py`
- Full-fidelity public claim: blocked
- External validation tracker: [#51](https://github.com/anulum/scpn-control/issues/51) — Plasma-control and facility replay artefacts
- Covered source paths: 1
- Required actions:
  - Attach strict documented public, measured loop-voltage replay, or external scenario benchmark artifacts before scenario-duration claims
  - Replace bootstrap-current proxy with neoclassical or transport-solver evidence before facility extrapolation
