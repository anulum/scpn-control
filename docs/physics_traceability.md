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
- Registry entries: 43
- Open fidelity gaps: 43
- Full-fidelity public claims blocked: 43
- Resolved module paths: 43
- Resolved evidence paths: 136
- Source marker coverage: 41/41

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
  - Add per-module traceability entries for each source file that declares an approximation or bounded model
  - Add validation tests for each declared validity domain before allowing broader claims

### bounded control plant approximations

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control/control`
- Full-fidelity public claim: blocked
- Covered source paths: 0
- Required actions:
  - Replace aggregate control coverage with per-controller traceability entries
  - Attach replay or higher-fidelity plant evidence before facility-control claims

### bounded phase and spiking runtime approximations

- Fidelity status: `bounded_model`
- Module path: `src/scpn_control`
- Full-fidelity public claim: blocked
- Covered source paths: 3
- Required actions:
  - Split phase, FPGA export, and geometry replay coverage into dedicated traceability entries
  - Add hardware-target and replay-fixture evidence before deployment claims

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
- Covered source paths: 0
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
