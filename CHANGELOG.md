<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: Release changelog. -->

# Changelog

## Unreleased

### Added
- Added CONTROL-owned `PulsedScenarioScheduler v2` for reusable pulsed-fusion
  shot lifecycles, with Python and Rust scheduler surfaces, audit-log
  semantics, guard validation, public API documentation, and direct tests.
- Exposed the Rust pulsed-scenario scheduler through the optional PyO3
  `scpn_control_rs` extension with direct parity tests.

## [0.20.3] - 2026-06-02

### Changed
- Bumped package, citation, archive, API, README capability, and release-note
  metadata to `0.20.3`.
- Added v0.20.3 release notes so the public release tag includes the
  Lean formal-evidence security remediation.

### Security
- Replaced regex-based Lean theorem, module, and safety-case identifier
  admission with linear-time validators to remove the CodeQL ReDoS finding
  while preserving the formal-evidence identifier contract.

## [0.20.2] - 2026-06-02

### Changed
- Bumped package, citation, archive, API, README capability, and release-note
  metadata to `0.20.2`.
- Reworked the README into clearer product, reading-path, feature, evidence,
  and limitation sections while preserving existing project content.
- Added v0.20.2 release notes and exposed them in the MkDocs navigation.
- Tightened public wording for gyrokinetic, real-time latency, disruption,
  validation-summary, and coverage claims so external-code, measured-shot,
  target-hardware, peer-reviewed, and plant-deployment claims remain blocked
  until strict evidence admission exists.
- Documented the benchmark-regression and native C++ build-hardening changes
  as release-readiness improvements.

## [0.20.1] - 2026-06-02

### Changed
- Bumped package, citation, and archive metadata to `0.20.1` for the documentation and evidence-polish release candidate.
- Expanded the README, documentation landing page, onboarding guide, tutorials, notebook gallery, use-case page, production-readiness boundary, and compute-validation financing page so new users can understand the software purpose, applications, market value, collaboration needs, and strict evidence boundaries.
- Added v0.20.1 release notes and exposed additional public documentation pages in the MkDocs navigation.
- Preserved the release boundary: predictive EFIT/P-EFIT, external-code GK, target-hardware real-time, and plant-deployment claims remain blocked until strict admission artefacts exist.

## [0.20.0] - 2026-06-01

### Changed
- Bumped the package and citation metadata to `0.20.0` for the release-preparation candidate.
- Reworked the README, documentation landing page, onboarding guide, use-case page, tutorial index, notebook gallery, API version snippet, MkDocs navigation, and release notes so new users can understand the controller-facing evidence layer, application value, collaboration needs, and strict production-readiness boundary.
- Regenerated the capability manifest and README capability snapshot from the current tree.
- Added a repository-published MAST EFM neural-equilibrium campaign report that aggregates the six-shot public EFM evaluation, records SAS-relative artefact references and digests, and keeps predictive EFIT/P-EFIT admission blocked pending full-output evidence.
- Added a SAS-hosted MAST EFM neural-equilibrium supervised dataset builder and repository-published dataset evidence report with deterministic shot-held-out train, validation, and test splits, padded LCFS geometry metadata, SHA-256 traceability, and an explicit block on predictive EFIT/P-EFIT admission pending full-output model validation.
- Added a neural-equilibrium training-campaign planner that verifies prepared MAST EFM SAS payloads, tracks deferred QLKNN/QuaLiKiz and external EFIT/P-EFIT dataset lanes, publishes run-order evidence, and records GPU-hour planning budgets without launching long training jobs.
- Added a dry-run-first MAST EFM neural-equilibrium trainer that validates the prepared dataset/report contract by default, emits a launch report, and requires explicit `--execute` before writing deterministic full-output baseline weights or holdout metrics.
- Added a MAST EFM feature-provenance audit and ML350 dry-run launch evidence showing the prepared SAS dataset validates on the storage host while `Ip_MA`, `Bt_T`, and `ffprime_scale` remain blocked because the converted public EFM bundles do not contain direct source keys.
- Added an original public MAST Level 1 EFM Zarr source audit that admits `plasma_current_x` for `Ip_MA` and `bphi_rmag` for `Bt_T`, while keeping dataset rebuild blocked until the `ffprime` profile-to-scalar policy for `ffprime_scale` is declared.
- Declared and implemented the public `ffprime` RMS-to-campaign-median policy, regenerated the MAST EFM converted references and supervised dataset on ML350 SAS, and updated the dataset, provenance, original-source, and dry-run launch reports so the former fallback feature list is empty.
- Clarified neural-equilibrium campaign execution policy: ML350 is storage-only, while any `--execute` training must run on this workstation or external cloud compute with SAS-mounted or copied data.
- Added a fail-closed MAST EFM compute-execution package: `--execute` now requires explicit workstation or external-cloud admission, matching dataset SHA-256, passing feature/source provenance reports, non-SAS weight output, and repository-published result templates for holdout, latency, GPU-cost, and admission-certificate evidence.
- Hardened the MAST EFM neural-equilibrium launch and result-template evidence path with explicit report validators, canonical payload digests, ML350 storage-only output checks, and tamper-detection tests before any future workstation or cloud training run can be cited.
- Hardened JAX gyrokinetic parity evidence with aggregate case/backend coverage digests, portable report paths, a separate local CPU timing benchmark report, and refreshed CPU parity artifacts while preserving the backend-parity-only claim boundary.

## [0.19.3] - 2026-06-01

### Fixed
- Added `validation/convert_mast_efm_neural_equilibrium_reference.py`
  to convert public MAST Level 1 EFM measured-shot Zarr campaigns into
  checksum-bound neural-equilibrium reference-candidate arrays while
  keeping predictive EFIT/P-EFIT claims blocked until exact-model
  predictions, pressure reconstruction, declared metrics, and strict
  admission artefacts exist.
- Added `validation/evaluate_mast_efm_neural_equilibrium.py` to run
  current neural-equilibrium weights against the converted public MAST EFM
  reference-candidate arrays, persist prediction artefacts on ML350 SAS,
  persist exact public EFM `profile_r`/`profile_z` coordinate grids,
  report flux RMSE plus derived magnetic-axis and LCFS residual evidence
  for 527 slices, and keep predictive EFIT/P-EFIT admission blocked until
  full pressure, q-profile, and exact-input evidence exists.
- Hardened neural equilibrium reference admission so
  `validation/validate_neural_equilibrium_reference.py` now emits
  schema-versioned, digest-bound reports with portable paths, explicit
  predictive-claim state, artefact-file SHA-256 digests, duplicate
  model/weight/reference-set rejection, fail-closed strict-mode report
  persistence, and an explicit block on predictive EFIT/P-EFIT claims until
  real P-EFIT or documented public-reference artefacts are supplied.
- Hardened external GK interface artefact admission so
  `validation/validate_gk_interface_artifacts.py` now emits schema-versioned,
  digest-bound reports with portable paths, explicit public-claim state,
  artefact-file SHA-256 digests, duplicate interface-code/run-id rejection,
  fail-closed strict-mode report persistence, and an explicit block on
  external-interface and full GK cross-code claims until real executable or
  documented public-reference artefacts are supplied.
- Hardened GK OOD calibration admission so
  `validation/validate_gk_ood_calibration.py` now emits schema-versioned,
  digest-bound reports with portable paths, explicit feature-schema and public
  claim state, raw and canonical calibration-artefact SHA-256 digests,
  Mahalanobis-metric provenance checks, duplicate-campaign rejection,
  fail-closed strict-mode report persistence, and an explicit block on
  deployment-calibration claims until real published, external-code, or
  facility GK campaign evidence is supplied.
- Hardened GK geometry reference admission so
  `validation/validate_gk_geometry_reference.py` now emits schema-versioned,
  digest-bound evidence with immutable reference-file SHA-256, per-case
  digests, explicit SI units, bounded local Miller-geometry admission status,
  duplicate-case rejection, direct source-checkout execution support, and an
  explicit block on full equilibrium-reconstruction claims pending independent
  Miller-geometry implementation or external equilibrium-code evidence.
- Hardened GK species reference admission so
  `validation/validate_gk_species_reference.py` now emits schema-versioned,
  digest-bound evidence with immutable reference-file SHA-256, per-case
  digests, explicit SI units, bounded-operator admission status, duplicate-case
  rejection, direct source-checkout execution support, and an explicit block on
  full collision-operator claims pending field-particle and external
  Fokker-Planck evidence.
- Hardened nonlinear Cyclone Base Case evidence so
  `validation/gk_nonlinear_cyclone.py` writes schema-versioned JSON/Markdown
  reports with canonical payload SHA-256, separates diagnostic checks from
  saturated `chi_i` admission, emits boolean-safe persisted evidence, supports
  direct source-checkout execution, and keeps the current 200-step V4 run
  blocked for saturated nonlinear CBC claims.
- Hardened the TORAX code-to-code benchmark evidence boundary with a
  schema-versioned JSON/Markdown report, canonical scenario and payload
  SHA-256 digests, direct source-checkout execution support, explicit
  admitted/blocked/not-requested external-reference states, finite-metric
  admission checks, and a `--require-external` gate that fails closed unless a
  real TORAX comparison is present.
- Added a fail-closed differentiable-transport full-fidelity readiness gate
  that binds campaign metadata, one-step and rollout latency reports, gradient
  audit digests, controller formal-proof digests, equilibrium-coupled metadata,
  and admitted external reference evidence before any full-fidelity claim can
  pass, and extended the benchmark/validator evidence path to publish and
  structurally admit the corresponding blocked readiness artefact.
- Added a GitHub Pages compute-validation financing page that explains GPU-hour,
  storage, public-data, and external-code validation needs while preserving the
  repository claim boundary.
- Added public QLKNN and QuaLiKiz Zenodo acquisition metadata, strict
  public-data acquisition manifest validation, module-specific regression
  tests, and documentation that separates acquired normalised metadata from
  deferred multi-GB tensor payloads and neural-transport validation evidence.
- Added strict NTM island-dynamics reference-artifact admission so
  full-fidelity q-profile, rational-surface, island-growth, saturated-width,
  suppression-time, seed-island, and ECCD-alignment claims require
  schema-versioned `scpn-control.ntm-reference.v1` evidence with measured NTM
  campaign or documented public-reference provenance, safe q-profile,
  rational-surface, island-width-trace, and ECCD-alignment artifact URIs,
  SHA-256 digests for every artifact and the canonical payload, NTM unit
  contracts, ordered rho grids, positive q-profile domains, rational-surface
  tokamak-ordering metadata, positive seed-island domains, tolerance-checked
  physical metrics, and module-specific tamper/domain tests.
- Added strict MARFE radiation-condensation reference-artifact admission so
  full-fidelity onset-temperature, density-limit, Greenwald-fraction,
  front-temperature, radiative-growth, impurity-fraction, connection-length,
  and power-balance claims require schema-versioned
  `scpn-control.marfe-reference.v1` evidence with measured MARFE campaign or
  documented public-reference provenance, safe temperature-profile,
  density-limit, radiation-curve, and power-balance artifact URIs, SHA-256
  digests for every artifact and the canonical payload, MARFE unit contracts,
  ordered temperature and density scans, bounded impurity-fraction domains,
  finite tokamak geometry and power-balance metadata, tolerance-checked
  physical metrics, and module-specific tamper/domain tests.
- Added strict EPED pedestal reference-artifact admission so full-fidelity
  pedestal height, pedestal width, peeling-ballooning pressure-limit,
  bootstrap-current, collisionality-width-ordering, and shaping-input claims
  require schema-versioned `scpn-control.eped-reference.v1` evidence with
  measured pedestal-database or documented public-reference provenance, safe
  pedestal-profile, EPED-prediction, bootstrap-current, and
  peeling-ballooning artifact URIs, SHA-256 digests for every artifact and the
  canonical payload, EPED unit contracts, ordered rho grids, positive
  width/beta domains, finite tokamak shaping metadata, tolerance-checked
  physical metrics, and module-specific tamper/domain tests.
- Added strict ELM crash and RMP suppression reference-artifact admission so
  full-fidelity ELM frequency, crash-depth, pedestal-drop, RMP-window, and
  heat-flux claims require schema-versioned `scpn-control.elm-reference.v1`
  evidence with measured H-mode campaign or documented public-reference
  provenance, safe pre-crash/post-crash/event/RMP artifact URIs, SHA-256
  digests for every artifact and the canonical payload, ELM/RMP unit
  contracts, ordered pedestal grids, Type-I energy-fraction bounds,
  tolerance-checked physical metrics, and module-specific tamper/domain tests.
- Added strict SOL blob-transport reference-artifact admission so full-fidelity
  blob velocity, spreading, wall-flux, and detector-event claims require
  schema-versioned `scpn-control.blob-transport-reference.v1` evidence with
  measured probe-campaign or documented public-reference provenance, safe
  reference/profile/detector artifact URIs, SHA-256 digests for every artifact
  and the canonical payload, SOL unit contracts, strictly ordered
  separatrix-to-wall coordinates, positive detector and blob-size domains,
  positive magnetic-geometry metadata, tolerance-checked physical metrics, and
  module-specific tamper/domain tests.
- Hardened neural equilibrium reference-artifact admission so predictive
  EFIT/P-EFIT or documented-reference equilibrium claims must use
  schema-versioned `scpn-control.neural-equilibrium-reference.v1` evidence with
  safe reference/prediction artifact URIs, trained-weight,
  reference-artifact, prediction-artifact, and canonical payload SHA-256
  digests, explicit target schema, grid/unit contracts, finite
  tolerance-checked psi/pressure/q-profile/boundary/axis metrics, admitted real
  P-EFIT executable provenance, and tamper-detection tests.
- Hardened neural transport reference-artifact admission so quantitative
  QuaLiKiz, QLKNN, or documented-reference transport claims must use
  schema-versioned `scpn-control.neural-transport-reference.v1` evidence with
  safe reference/prediction artifact URIs, trained-weight,
  reference-artifact, prediction-artifact, and canonical payload SHA-256
  digests, explicit target schema, QLKNN-10D feature order, unit contracts,
  finite tolerance-checked metrics, admitted real QuaLiKiz executable
  provenance, and tamper-detection tests.
- Hardened persisted external gyrokinetic interface artifact admission so real
  executable and documented public-reference parser evidence must use
  schema-versioned `scpn-control.gk-interface-artifact.v1` reports with safe
  deck/raw-output/parsed-output artifact URIs, SHA-256 digests for each
  artifact, a canonical payload digest, explicit transport/frequency/wavenumber
  unit declarations, finite physical fields, admitted executable provenance,
  and tamper-detection tests.
- Hardened the real external-code linear gyrokinetic cross-code admission gate
  so full-fidelity GK agreement evidence must use schema-versioned
  `scpn-control.gk-crosscode.v1` reports with SHA-256 digests for the external
  input deck, external output, native input, and canonical payload, finite
  growth-rate/frequency/wavenumber fields, admitted executable provenance, and
  bounded native-vs-external tolerances.
- Added strict admission for persisted differentiable transport one-step and
  rollout gradient-latency reports, including duplicate-key rejection, backend,
  dtype, claim-boundary, audit-error, sampled-index, run-count, and latency
  percentile checks before the reports can support release evidence.
- Hardened local preflight release-evidence admission so the non-test
  `release-evidence` gate now generates a temporary `scpn-control validate
  --json-out` report and admits it through `scpn-control
  validate-release-evidence`, matching the CI artifact-admission path.
- Exposed release-evidence report admission as
  `scpn-control validate-release-evidence REPORT`, so operators can validate
  CI or release JSON reports through the public CLI instead of importing the
  Python validation module directly.
- Added strict release-evidence report admission for the CI artifact produced
  by `scpn-control validate --json-out`, including duplicate-key rejection,
  mandatory pass status for data manifests, JAX GK parity, and physics
  traceability, complete CPU/GPU parity case coverage, and an admission JSON
  artifact with the source report SHA-256 digest.
- Added a CI `release-evidence-gate` job that runs top-level
  `scpn-control validate --json-out` and uploads the JSON report, so remote
  runs now publish auditable data-manifest, JAX GK parity, and physics
  traceability admission evidence.
- Added the top-level `scpn-control validate --json-out` path to local
  preflight as a non-test release-evidence gate, with source-tree import
  precedence, so `make preflight` and `make preflight-fast` now catch data
  provenance, JAX GK parity, and physics traceability drift before push.
- Promoted physics traceability validation into the top-level
  `scpn-control validate` command with staged-registry and scoped-skip controls
  so local release validation now fails on bounded-claim registry drift in the
  same path as data provenance and JAX GK parity evidence.
- Promoted strict persisted JAX gyrokinetic CPU/GPU parity evidence admission
  into the top-level `scpn-control validate` command, added scoped staging and
  skip flags, and exposed case/backend requirements on
  `validate-jax-gk-parity` so local release validation matches the CI gate.
- Wired strict persisted JAX gyrokinetic CPU/GPU parity evidence admission into
  the `jax-parity` CI job and added a module-specific repository-evidence
  regression so missing CBC, kinetic-electron TEM, or stable-mode backend
  pairs cannot silently drift out of release evidence.
- Persisted the current JAX gyrokinetic CPU parity campaign alongside the
  existing GPU campaign, regenerated the strict parity summary, and updated the
  traceability and validation docs so backend parity is admitted for CPU/GPU
  reproducibility without promoting quantitative external-code GK claims.
- Replaced WebSocket runtime-configuration raw SHA-256 digests with a
  domain-separated HMAC-SHA256 evidence digest so deterministic admission
  remains stable while code scanning no longer treats the configuration proof
  as weak sensitive-data hashing.
- Hardened mu-synthesis claim evidence persistence so bounded static
  robust-control reports now carry canonical payload SHA-256 digests and load
  admission rejects duplicate keys, schema drift, edited metrics, and bounded
  evidence presented as validated robust-control evidence.
- Added schema-versioned geometry-neutral replay evidence admission so
  stellarator replay claims bind the validated report, scenario, trace,
  metrics, thresholds, magnetic-configuration provenance, actuator
  calibration, latency model, and fault model by SHA-256 digest while keeping
  synthetic W7-X-like replay separate from measured or benchmark device claims.
- Added schema-versioned Kuramoto phase-runtime evidence admission so
  deployment-target phase claims bind deterministic input digests, Python
  reference output digests, optional Rust parity errors, deployment-target
  oscillator coverage, and timestep-refinement convergence before runtime
  claims can cite the optional Rust fast path.
- Hardened controller safety-case readiness so promotion now also requires a
  typed `websocket_runtime_evidence` artifact that resolves under the declared
  evidence root, matches its SHA-256 bytes, and passes qualified WebSocket
  runtime admission before deployment readiness can cite authenticated command
  streams, TLS enforcement, payload caps, token-bucket limiting, broadcast
  delivery, or backpressure absence.
- Added schema-versioned FPGA HDL export evidence admission so generated
  Verilog/VHDL project claims bind controller artifact SHA-256, generated HDL,
  weight memory, timing constraints, Makefile, resource estimates, synthesis
  report digests, safe report URIs, non-negative timing slack, and local-only
  versus qualified synthesis claim status before safety-case readiness can cite
  hardware export evidence.
- Hardened controller safety-case readiness so promotion now also requires a
  typed `codac_runtime_evidence` artifact that resolves under the declared
  evidence root, matches its SHA-256 bytes, and passes qualified CODAC/EPICS
  runtime admission before deployment readiness can cite CODAC timing,
  interlock, export, or backpressure evidence.
- Added schema-versioned CODAC/EPICS runtime evidence admission so control
  boundary claims bind generated EPICS database and OPC-UA nodeset hashes,
  cycle-deadline percentiles, exercised interlock blocking, backpressure
  counts, local-only versus qualified claim status, duplicate-key-safe JSON
  loading, and canonical SHA-256 payload digests before facility runtime
  claims can cite CODAC evidence.
- Hardened controller safety-case readiness so promotion now requires a typed
  `hil_replay_evidence` artefact in addition to external physics validation,
  target-hardware timing, and independent review; the artifact must resolve
  under the declared evidence root, match its SHA-256 bytes, and pass
  qualified-target HIL replay admission before deployment readiness can cite it.
- Added schema-versioned HIL replay evidence admission for CONTROL-owned
  runtime deployment claims, including canonical SHA-256 payload digests,
  replay digests over controller, timing, target-hardware, interlock, and
  backpressure fields, fail-closed target-hardware promotion checks, duplicate
  JSON key rejection on load, and module-specific behavioural tests that keep
  local workstation replay evidence separate from qualified deployment
  evidence.
- Hardened reviewed control-runtime defects by replacing fixed-window
  WebSocket command limiting with token buckets per connection and peer,
  adding structured WebSocket security audit logs, arming native-solver cleanup
  only after a C++ library loads successfully, rejecting degenerate
  Grad-Shafranov flux normalisation, fixing multigrid restriction on
  rectangular odd grids, vectorising free-boundary coil Green's flux over the
  grid, tightening NMPC SPD symmetry admission, and adding optional JAX
  autodiff plant linearisation.
- Added schema-versioned JAX gyrokinetic parity artifact production and
  stricter admission so native/JAX local-dispersion comparisons bind backend,
  device, platform, dtype, X64 state, solver kwargs, tolerances, and canonical
  SHA-256 payload digests while keeping the evidence boundary limited to
  backend parity until external GK validation artifacts are supplied; admission
  now requires named CBC, kinetic-electron TEM, and low-drive stable-mode case
  coverage when requested, binds case-parameter digests, rejects mode-spectrum
  replay, and supports backend coverage requirements.
- Added a quantum-enhanced disruption bridge facade that keeps quantum backend
  ownership in `scpn-quantum-control`, lazily imports optional quantum
  dependencies, maps the CONTROL 8-feature disruption contract to the ITER
  11-feature contract with explicit default provenance, emits bounded
  amplitude-kernel evidence, records admission evidence with feature digests
  and external-evidence requirements, adds schema-versioned advisory
  certificates for bridge and kernel reports, publishes a machine-readable
  dependency contract for the `scpn-quantum-control` backend, embeds that
  dependency contract in advisory reports, pins the Qiskit simulator dependency
  name, records backend-contract attestation when the optional backend exposes
  its own contract, adds certificate-bound advisory decision evidence with
  score-basis provenance, deterministic risk-band thresholds, backend-contract
  validation state, and blocked control action, and fail-closes advisory reports
  behind tamper-evident claim boundaries.
- Aligned the CI and local coverage gate to the current validated 93.74%
  repository result as explicit temporary debt after the quantum-disruption and
  physics-debugging surfaces expanded; the policy remains module-specific
  behavioural tests only, with no synthetic gate-chasing tests.
- Added a local-first physics debugging assistant for validation-gap analysis,
  falsifiable hypothesis generation, and campaign suggestions with loopback
  provider defaults, explicit endpoint allowlisting for facility gateways,
  onsite provider profiles, response normalization for common local gateway
  protocols, secret redaction, cited-evidence enforcement, risk-control checks,
  prompt-injection neutralization for evidence text, advisory safety-policy
  admission, optional hallucination guardrail review with a default
  `director-ai` profile, reviewed-draft digest binding, high-severity
  fail-closed enforcement, provider and policy replay binding, provider quorum
  admission, and tamper-evident advisory report digests.
- Added hash-addressed formal safety certificate bundle artifact admission so
  release gates verify safe relative bundle URIs, bundle SHA-256 bytes,
  tamper-evident artifact metadata digests, UTC creation timestamps, embedded
  certificate digests, artifact binding, backend, and required certificate
  policy before replaying certificate evidence.
- Added formal safety certificate admission policies so certification campaigns
  can require minimum proof depth, controller artifact binding, CTL/LTL
  evidence, and named checked specifications before JSON/Markdown certificate
  artifacts are emitted or admitted.
- Added a one-call formal safety certificate workflow that resolves one
  Petri-net verifier backend, runs base safety/liveness plus CTL/LTL
  obligations, binds optional controller artifact bytes by SHA-256, and
  persists certificate JSON/Markdown evidence.
- Hardened formal safety certificate publication with JSON/Markdown writers and
  semantic section-admission checks that reject internally inconsistent
  certificate evidence even when the payload digest is recomputed.
- Added bounded CTL/LTL formula facades and schema-versioned formal safety
  certificate payloads for SCPN Petri-net controllers, including
  tamper-evident payload digests and shared explicit-state/Z3 formula checking.
- Hardened stellarator geometry configuration with a Pydantic v2 schema/export
  path, construction-time physics-bound validation, and immutable validated
  config objects so ISS04, Boozer-surface, and neoclassical calculations cannot
  observe post-validation mutation.
- Hardened EPED pedestal configuration with a Pydantic v2 schema/export path,
  construction-time physics-bound validation, and immutable validated config
  objects so solver calls cannot observe post-validation mutation.
- Hardened FusionKernel configuration loading so runtime JSON is retained as a
  typed Pydantic v2 model with schema-normalised boundary variants while the
  legacy dict view remains a compatibility export for solver internals.
- Hardened controller safety-case readiness artefacts so target-hardware timing evidence must resolve under a supplied artifact root, match the declared SHA-256 bytes, and pass the schema-versioned E2E latency validator before promotion readiness can cite it.
- Hardened end-to-end control-latency evidence so benchmark reports now use a
  schema-versioned canonical payload digest and admission rejects tampering,
  non-positive run counts, unordered percentiles, mismatched overhead factors,
  unqualified target-hardware metadata, and altered local-evidence boundaries
  before runtime-readiness claims can cite latency reports.
- Hardened Z3-backed SCPN formal-verification evidence so pass, fail, and
  blocked reports use a schema-versioned payload, canonical SHA-256 integrity
  digest, explicit solver metadata, and manifest-matching status, depth, solver,
  and checked-specification admission before safety-critical controller
  artifacts can reference SMT proof reports.
- Hardened differentiable-transport evidence admission so gradient-audit
  evidence revalidates finite non-negative losses, tolerance agreement with
  campaign metadata, unique in-domain sampled audit indices, pass/fail
  consistency with maximum audit error, strict integer latency run counts, and
  ordered latency percentiles before controller-tuning evidence is persisted or
  admitted; the rollout source-gradient loss now remains inside the traced JAX
  graph and the module enables JAX x64 before importing `jax.numpy` so dtype
  evidence is not silently downgraded.
- Hardened digital-twin online-update evidence admission so bounded Bayesian
  update claims revalidate finite non-negative losses, loss-history minima,
  source binding, unique bounded parameter priors, best-parameter domains,
  strict integer campaign settings, and simulator units for every observation
  target before evidence digests are admitted.
- Hardened geometry-neutral stellarator replay admission so scenarios fail
  closed on nonzero initial frames, missing objective metrics, impossible
  current constraints, unsupported stuck-fault modes, and non-integer runtime
  report inputs before replay evidence or manifest digests are produced.
- Added tamper-evident SHA-256 payload digests to persisted RZIP
  calibration evidence and benchmark reports so admission rejects modified
  evidence payloads before facility-claim promotion.
- Hardened RZIP facility-claim admission so calibration evidence is
  revalidated against source class, reference growth-rate presence, finite
  physical fields, and declared growth-rate tolerance at admission time.
- Hardened the JAX traceable runtime public boundary so single-loop rollouts
  reject batched command arrays, reject vector initial states deterministically,
  and validate parity integer seeds before reproducibility campaigns are built.
- Added tamper-evident geometry-neutral stellarator replay manifests that bind
  scenario, trace, metric, and threshold payloads with SHA-256 digests and
  fail closed on manifest, trace, or acceptance tampering.
- Renamed the IMAS ODS adapter contract test file to avoid the retired generic
  test filename pattern while preserving the same module-specific behaviour.
- Added a fail-closed formal proof-manifest gate for safety-critical SCPN
  controller artifacts, including bounded-claim enforcement, hash-addressed
  report metadata, and mandatory counterexample paths for failed proof
  evidence.
- Hardened safety-critical proof-manifest admission by binding evidence to the
  canonical controller-artifact payload SHA-256, rejecting unsafe report URIs,
  constraining formal backends, and optionally verifying report bytes under a
  caller-supplied report root.
- Added tamper-evident differentiable transport admission evidence that binds
  JAX campaign metadata, sampled gradient-audit results, equilibrium coupling,
  and optional safety-critical controller proof artifact digests before
  controller-tuning claims can be promoted.
- Added tamper-evident digital-twin online-update evidence that binds TRANSP
  and TSC simulator metadata, observation and prior digests, Bayesian-update
  result digests, baseline-improvement status, and optional safety-critical
  controller proof artifact digests.
- Added a bounded controller safety-case evidence workflow that links a passing
  formal controller proof manifest, audited differentiable-transport evidence,
  and TRANSP/TSC-backed digital-twin update evidence to the same canonical
  controller artifact digest.
- Added schema-versioned controller safety-case manifest persistence with an
  integrity digest so archived evidence bundles fail closed on malformed schema
  or payload tampering before replay admission.
- Added an explicit controller safety-case readiness gate that remains blocked
  until external physics validation, target-hardware timing evidence, and
  independent safety-review digests are all present and bound to the current
  safety-case bundle.
- Added schema-versioned controller safety-case readiness manifest persistence
  with an integrity digest so promotion decisions fail closed on malformed
  schema or payload tampering before replay.
- Added typed controller safety-case readiness artefacts so promotion evidence
  requires kind-specific external validation, target-hardware timing, and
  independent-review artifacts with safe relative URIs, producers, timestamps,
  and SHA-256 digests.
- Hardened the phase WebSocket stream with explicit payload-size limits,
  server-side frame caps, default client authentication, TLS-required startup
  mode, disabled-by-default query-token authentication, fail-closed plaintext
  non-loopback binding, browser-origin allowlisting, command allowlisting, and
  CLI/documentation coverage for authenticated remote exposure.
- Added a Pydantic v2 schema/export path to `TokamakConfig`, extended E2E
  latency evidence with target-hardware metadata, and exposed NMPC optional
  `casadi`/fail-closed `acados` solver backend contracts.
- Implemented the optional NMPC `acados` OCP interface with injected runtime
  factories, symbolic discrete dynamics support, augmented-state slew-rate path
  constraints, SQP/partial-condensing HPIPM defaults, exact-Hessian mode,
  warm-start staging, and fail-closed solver-status handling.
- Hardened NMPC `acados` result admission with finite state/control trajectory
  checks, state and terminal-set enforcement, slew-rate revalidation, and a
  runtime plant-consistency residual gate for symbolic-dynamics drift.
- Added a strict E2E latency-evidence validator so unqualified local benchmark
  runs cannot be admitted as target-hardware or real-time performance evidence.
- Double-gated controller bit-flip fault injection behind both constructor and
  environment opt-ins, and hardened controller JSONL append handling against
  symlink-target writes where platform support is available.
- Hardened real-data manifest artefact verification so checksum-covered local
  evidence must use relative, non-traversing paths resolved under the manifest
  evidence tree or repository root.
- Added shared external-reference URI admission for density, RZIP, orbit, and
  VMEC validation gates so external artifact claims reject ambiguous relative
  paths, arbitrary local file URIs, hosted file URIs, and traversal paths.
- Hardened external executable provenance admission for GK cross-code, GK
  interface, and neural transport reference validators so real external-code
  claims reject URI, relative, traversal, temporary, and system-control
  `binary_path` values.
- Added a public production-readiness boundary that separates
  production-oriented library engineering from facility deployment, external
  validation, measured-shot validation, and certification claims.
- Hardened fail-closed physics and mathematics boundaries across MHD,
  pedestal, edge, transport, orbit-following, scenario, and uncertainty
  surfaces without promoting unsupported facility or full-fidelity claims.
- Added release-blocking boundary coverage for edge cases, invalid inputs,
  monotonicity contracts, conservation boundaries, and finite-output contracts
  in module-specific test files.
- Persisted differentiable transport campaign metadata and added a replay guard
  that rejects backend, grid, boundary, closure, tolerance, and equilibrium
  drift before controller-tuning reruns.
- Added differentiable transport source-schedule gradients so controller
  tuning can optimise additive heating, fuelling, and impurity-source inputs
  through the same JAX Crank-Nicolson facade as transport coefficients.
- Added bounded multi-step differentiable transport rollout gradients so
  controller tuning can optimise time-distributed source schedules without
  finite-difference plant evaluations.
- Wired NMPC source-rollout tuning to the multi-step differentiable transport
  gradient path with explicit source bounds and sampled finite-difference
  audit admission.
- Added audited multi-step differentiable transport rollout-gradient latency
  reporting for bounded NMPC source-rollout admission evidence.
- Added Grad-Shafranov flux-weighted multi-step transport rollout loss and
  source/equilibrium gradients for bounded controller-tuning studies.
- Added a bounded reduced-gyrokinetic transport closure adapter for mapping
  existing quasilinear GK profile outputs into differentiable transport
  coefficient channels with explicit provenance.
- Added explicit gyrokinetic species diamagnetic-frequency bookkeeping with
  charge-direction, density-gradient, temperature-gradient, and zero-drive
  regression coverage.
- Added the Miller geometry contravariant metric-determinant identity to the
  public geometry result and regression coverage.
- Added an explicit reduced-gyrokinetic saturation-rate utility with monotone,
  non-negative, and field-line-rate-bounded regression coverage.
- Added a differentiable transport gradient audit that compares JAX transport
  coefficient and source-schedule gradients against sampled finite-difference
  perturbations before controller-tuning admission.
- Added differentiable transport gradient-latency reporting for the audited
  controller-tuning admission path with persisted bounded benchmark artefacts.
- Wired the NMPC transport-tuning path to require that gradient audit by
  default and to persist the audit result with each coefficient update.
- Added audited NMPC source-schedule tuning for additive heating, fuelling, and
  impurity-source controls with explicit finite source bounds.
- Added RZIP vertical-stability calibration evidence and fail-closed
  facility-claim admission with bounded local benchmark artefacts.
- Added resistive-wall-mode feedback claim evidence and fail-closed
  facility-claim admission with bounded wall, rotation, coil, and latency
  provenance reports.
- Added EFIT-lite reconstruction claim evidence and fail-closed facility-claim
  admission with bounded diagnostic and shape provenance reports.
- Added kinetic-EFIT pressure and q-profile claim evidence with fail-closed
  facility-claim admission for matched pressure, q-profile, anisotropy,
  diagnostic, profile, fast-ion, MSE-calibration, and interpolation provenance.
- Added VMEC-lite spectral facade claim evidence with fail-closed full-VMEC
  admission for matched Fourier geometry, rotational transform, residual, and
  convergence provenance.
- Added orbit-following claim evidence with fail-closed external-code
  admission for matched banana-width, first-orbit-loss, particle, geometry,
  collision-model, and loss-boundary provenance.
- Added uncertainty-quantification claim evidence with fail-closed calibrated
  predictive-UQ admission for matched central values, sigma statistics, seed,
  prior, scenario, propagation-chain, and sensitivity provenance.
- Added density-control claim evidence with fail-closed facility-calibrated
  admission for matched Greenwald fraction, particle inventory, geometry,
  transport, actuator, diagnostic, and CFL provenance.
- Added neural-equilibrium claim evidence with fail-closed predictive-claim
  admission for matched P-EFIT or documented public reference artefacts,
  weight checksums, flux, pressure, q-profile, boundary, and axis tolerances.
- Added neural-transport claim evidence with fail-closed quantitative
  admission for matched QuaLiKiz or documented public reference artefacts,
  weight checksums, QLKNN-10D feature ordering, diffusivity errors, and branch
  accuracy.
- Added neural-turbulence claim evidence with fail-closed quantitative
  admission for matched gyrokinetic campaign or documented public reference
  artefacts, weight checksums, QLKNN-class feature ordering, gyro-Bohm flux
  errors, and critical-gradient accuracy.
- Added disruption-mitigation claim evidence with fail-closed mitigation
  admission for measured, external-benchmark, or documented public reference
  artefacts covering warning lead time, mitigation outcome, halo-current
  envelope, runaway-beam envelope, and tritium-breeding-ratio metrics.
- Added free-boundary tracking claim evidence with fail-closed facility-control
  admission for matched public, measured-replay, or external equilibrium
  artefacts covering shape, X-point, divertor, coil-current, response-rank,
  latency, and supervisor provenance.
- Added DT burn-control claim evidence with fail-closed reactor-control
  admission for matched public, integrated-transport benchmark, or measured
  burn replay artefacts covering alpha power, Q, Lawson margin, burn fraction,
  reactivity exponent, and controller-limit provenance.
- Added volt-second claim evidence with fail-closed pulse-duration admission
  for matched public, measured loop-voltage replay, or external scenario
  artefacts covering total flux, flat-top duration, Ejima flux, bootstrap
  current, and budget-margin provenance.
- Added current-drive claim evidence with fail-closed external deposition
  admission for matched public, ray-tracing, Fokker-Planck, or measured
  deposition artefacts covering absorbed power, driven current, deposition
  centroid, peak current density, and NBI slowing-down provenance.
- Added mu-synthesis claim evidence with fail-closed validated robust-control
  admission for matched public, external mu-toolbox, or measured replay
  artefacts covering mu upper bounds, robustness margin, controller gain,
  D-scaling, and closed-loop spectral-abscissa provenance.
- Hardened SCPN formal verification with algebraic place-invariant proofs and
  bounded temporal response and recurrence specifications over all bounded
  firing paths.
- Added optional Z3 bounded model checking for compiled SCPN control logic,
  including SMT-backed marking-bound counterexamples, temporal exclusivity and
  response specs, and JSON/Markdown evidence publication with explicit blocked
  status when `z3-solver` is unavailable.
- Hardened federated disruption prediction with per-facility array ingestion,
  facility-update differential privacy accounting, serialisable privacy
  ledgers, and a deterministic synthetic multi-facility benchmark report. This
  remains bounded synthetic evidence, not measured cross-facility validation.
- Added neural equilibrium synthetic pretraining with deterministic
  JAX-compatible weights, benchmark reports, and a fail-closed real EFIT/P-EFIT
  fine-tuning admission gate backed by persisted reference artefacts.
- Added digital-twin online model updating with fail-closed TRANSP/TSC
  simulator artifact metadata validation, deterministic Bayesian optimisation
  over bounded density, effective-charge, and actuator parameters, and a
  synthetic online-update benchmark report.
- Hardened the gyrokinetic online learner with OOD-threshold sample admission,
  auditable retraining decisions, persisted update reports, and a deterministic
  synthetic online-retraining benchmark.
- Added integrated-scenario coupling audits with deterministic replay metadata,
  module-by-module exchange records, timestep consistency checks, and bounded
  current and thermal-energy diagnostics while preserving the external
  validation claim boundary.
- Preserved JAX gyrokinetic stiffness-closure monotonicity under the CI JAX
  backend while keeping the closure explicitly bounded as a controller-tuning
  surrogate.
- Applied the formatter changes required by the remote pre-commit workflow so
  release-candidate CI starts from the same formatting state as local checks.

### Changed
- Package version bumped to `0.19.3`.
- Documented the hardening release-candidate scope, residual validation gaps,
  and tag gate before creating a release tag.
- Retained the physics traceability boundary: public full-fidelity claims remain
  blocked until the required external artefacts are supplied and validated.
- Hardened traceability summary checks so generated report, roadmap, and
  release documentation derive open-gap counts from the live registry instead of
  stale fixed numbers.
- GitHub Actions for the current `main` head are green after the documentation,
  traceability, formatter, and JAX stiffness fixes.

### Validation
- `pre-commit run --all-files`
- `python tools/check_test_quality_policy.py`
- `python validation/validate_physics_traceability.py --registry validation/physics_traceability.json --json-out`
- `python validation/benchmark_federated_disruption.py`
- `python validation/benchmark_neural_equilibrium_pretraining.py`
- `python validation/benchmark_differentiable_transport_latency.py`
- `python validation/benchmark_rzip_calibration.py`
- `python validation/benchmark_rwm_claims.py`
- `python validation/benchmark_efit_lite_claims.py`
- `python validation/benchmark_kinetic_efit_claims.py`
- `python validation/benchmark_vmec_lite_claims.py`
- `python validation/benchmark_orbit_following_claims.py`
- `python validation/benchmark_uq_claims.py`
- `python validation/benchmark_density_control_claims.py`
- `python validation/benchmark_neural_transport_claims.py`
- `python validation/benchmark_neural_turbulence_claims.py`
- `python validation/benchmark_disruption_mitigation_claims.py`
- `python validation/benchmark_digital_twin_online_update.py`
- `python validation/benchmark_gk_online_learner.py`
- `python validation/benchmark_integrated_scenario_coupling.py`
- `python tools/capability_manifest.py --check`
- `python -m tools.check_generated_traceability`

## [0.19.1] — 2026-05-21

### Fixed
- Hardened fail-closed contracts across external GK solver surfaces:
  QuaLiKiz, GENE, GS2, and CGYRO now require explicit dual-gate opt-in
  for degraded fallback behaviour.
- Hardened integrated transport GK acceptance criteria:
  converged results must also be finite and non-negative for
  `chi_i`, `chi_e`, and `D_e`, otherwise fail closed by default.
- Hardened hybrid-GK control-plane contracts:
  scheduler config/runtime input validation, OOD detector result/ensemble
  schema checks, online learner sample/weight validation, corrector
  config/profile validation, and verification-report telemetry guards.

### Changed
- Package version bumped to `0.19.1`.
- Regenerated capability manifest and README capability snapshot to match
  current repository state.

## [0.19.0] — 2026-03-18

### Fixed — Physics Audit (33 equation corrections)
- **Integration wiring**: sawtooth crash→psi writeback, NTM seeding from crash
  energy, ELM/stability/L-H modules connected to scenario loop, beta_N and li
  computed from profiles (were hardcoded placeholders)
- **GK**: Te/Ti ratio in quasilinear (was trivially 1.0), omega_r←gamma_net in
  TGLF native output, electron drive phi→phi_eff for EM flutter, Ampere skin-depth
  term, CFL k→k² for Poisson bracket, Sugama conservation via 3×3 Gram matrix
- **MHD**: NTM MRE prefactor (extra r_s removed), polarization (w_pol/w)³,
  Porcelli Condition 2 units, EPED q95 sqrt→linear, TAE k_∥=1/(2qR₀),
  locked-mode I_eff from geometry (was 5 orders too small)
- **Transport**: Sauter L31 collisionality denominator, banana boundary ε^(3/2),
  GS residual → GS* operator, tau_eq from Braginskii, halo f_halo=0.3,
  orbit drift ÷B³, ISS04 s_ref=(2/3)²
- **Edge**: Connor-Hastie h(Z) E_ratio removed, hot-tail→erfc, SOL sheath BC
  two-point model, GS source full weight (was 50/50 blend), BH Table VII citation,
  noqa dead code removed
- 3,300+ tests (235 files), 100% coverage, 20 CI jobs

## [0.18.0] — 2026-03-17

### Added
- **Nonlinear δf gyrokinetic solver** (`gk_nonlinear.py`):
  5D Vlasov in flux-tube geometry with dealiased E×B bracket (Orszag 2/3 rule),
  4th-order parallel streaming, curvature/grad-B drift, Krook collision operator,
  RK4 with CFL-adaptive dt, zonal flow diagnostics.
  JAX-accelerated variant (`jax_gk_nonlinear.py`) with `jax.checkpoint`.
  CBC validation: linear recovery, energy conservation, zonal flows, saturated state.
- **Native TGLF-like approximation** (`gk_tglf_native.py`):
  SAT0/SAT1/SAT2 spectral saturation (Staebler 2007/2017),
  E×B shear quench (Waltz 1997), trapped-particle damping (Connor 1974),
  multi-scale ITG-ETG cross-scale coupling (Maeyama 2015, α_cs=3.0),
  quasilinear weights with Γ₀ FLR + particle pinch.
  Implements `GKSolverBase` — usable as drop-in for external TGLF.
- `"tglf_native"` transport mode in `integrated_transport_solver.py`
- `validation/gk_nonlinear_cyclone.py` — CBC benchmark (4 tests, all passing)
- **Dimits shift proven** at n_kx=256: zero transport below critical gradient
- **Electromagnetic A_∥** via Ampere's law (KBM/MTM capable)
- **Sugama collision operator**: pitch-angle scattering with ν(v) ∝ v⁻³,
  particle/momentum/energy conservation (<3e-8/<1e-23/<2e-8)
- **Kinetic electron species**: semi-implicit backward-Euler, removes adiabatic
  approximation. chi_i = 1.3 χ_gB (kinetic) vs 2.0 χ_gB (adiabatic)
- **Ballooning connection BC**: kx shift at θ boundaries via FFT phase multiply
- **Rosenbluth-Hinton zonal Krook damping**: dynamic relaxation on bounce time
- 53 new tests (27 TGLF native + 26 nonlinear)
- **Physics deepening sprint** — 18 modules, ~50 paper citations, 118 new tests:
  - Neoclassical: Pfirsch-Schlüter regime, regime auto-detection, full Sauter L31/L32/L34 bootstrap
  - EPED pedestal: collisionality-dependent width (Snyder 2011), shaping factor (Connor 1998)
  - Sawtooth: Porcelli 1996 three-condition trigger, Bussac δW_MHD
  - RWM feedback: rotation stabilization (Fitzpatrick 2001), wall geometry, critical rotation
  - NTM dynamics: diamagnetic shear (Sauter 1997), GGJ Δ' (Glasser 1975), bootstrap from local params
  - Alfvén eigenmodes: electron Landau damping (Rosenbluth 1975), Fu-Van Dam resonance, RSAE+BAE
  - Integrated scenario: transport solver wired, Strang splitting, bootstrap+ohmic, NTM coupling
  - Current drive: Fisch-Boozer ECCD, Stix slowing-down, Prater geometric efficiency
  - L-H transition: Martin 2008 scaling, low-density branch (Ryter 2014), Kim-Diamond predator-prey
  - Momentum transport: Prandtl number (Peeters 2011), Rice rotation, Burrell ExB shear
  - Orbit following: Boozer 2004 equations, Stix slowing-down, Goldston first-orbit loss
  - Locked mode: Fitzpatrick 1993 EM torque, La Haye locking condition
  - Tearing coupling: Chirikov 1979 overlap, La Haye-Buttery 2009 coupling
  - MARFE: Drake 1987 instability, Greenwald 2002, Lipschultz 1987 onset
  - Impurity transport: Hirshman-Sigmar 1981 pinch, Post 1977 radiation
  - Runaway electrons: Wesson Coulomb log, R&P full avalanche, Martin-Solis synchrotron limit
  - Plasma startup: Lieberman-Lichtenberg Townsend, Janev ionization rate
  - Current diffusion: temperature-dependent ln_Λ, Jardin 2010 citations
- Python 3.14 added to CI matrix and classifiers

### Changed
- **Nengo replaced** with pure LIF+NEF engine (numpy 2.x compatible, no external dependency)
- All mypy errors fixed across 10 source files
- 3,300+ tests (235 files), 100% coverage, 20 CI jobs

### Fixed
- CFL fix for hyperdiffusion stability in nonlinear GK solver
- Subprocess PYTHONPATH in test_controller_oracle_serve for Python 3.14
- Martin L-H scaling 10x density unit bug (n_e19 → n_e20 conversion)

## [0.17.0] — 2026-03-14

### Added
- **Gyrokinetic Three-Path Transport System** (16 new modules):
  - **Path A — External GK coupling** (5 codes):
    `gk_interface.py` (universal ABC + GKLocalParams/GKOutput),
    `gk_tglf.py` (TGLF namelist generation + subprocess),
    `gk_gene.py` (GENE parameters + nrg parsing),
    `gk_gs2.py` (GS2 namelist + NetCDF/omega parsing),
    `gk_cgyro.py` (CGYRO input + freq parsing),
    `gk_qualikiz.py` (Python API + subprocess fallback)
  - **Path B — Native linear GK eigenvalue solver**:
    `gk_geometry.py` (Miller flux-tube parameterisation, metric coefficients, curvature),
    `gk_species.py` (species, Gauss-Legendre velocity grid, Sugama collision operator),
    `gk_eigenvalue.py` (response-matrix eigenvalue solver in ballooning space),
    `gk_quasilinear.py` (mixing-length saturation, quasilinear chi_i/chi_e/D_e)
  - **Path C — Hybrid surrogate+GK validation layer**:
    `gk_ood_detector.py` (Mahalanobis + range + ensemble OOD detection),
    `gk_scheduler.py` (periodic/adaptive/critical-region spot-check scheduling),
    `gk_corrector.py` (multiplicative/additive/replace correction with EMA smoothing),
    `gk_online_learner.py` (buffer-based retraining with validation holdout + rollback),
    `gk_verification_report.py` (per-session verification stats, JSON export)
  - **SCPN phase bridge**: `phase/gk_upde_bridge.py` (GK fluxes → adaptive K_nm for
    P0↔P1 microturbulence↔zonal, P1↔P4 zonal↔barrier, P3↔P4 sawtooth↔barrier)
  - `"external_gk"` transport mode wired into `integrated_transport_solver.py`
  - Cyclone Base Case validation (Dimits et al. 2000) with reference data
  - `validation/benchmark_gk_linear.py` — CBC, gradient scan, multi-code, SPARC/ITER
  - `validation/benchmark_hybrid_accuracy.py` — end-to-end hybrid accuracy
  - `examples/tutorial_08_gyrokinetic_solver.py` — 5-section GK demo
  - 163 new tests across 15 test files

### Changed
- License: MIT OR Apache-2.0 → **GNU AGPL v3 | Commercial licensing available**
- Removed Michal Reiprich from all authorship records
- 3,015 tests (220+ files), 100% coverage, 20 CI jobs

## [0.16.0] — 2026-03-13

### Added
- **Phase 3 — Frontier physics** (10 modules in `core/`):
  - `gyrokinetic_transport.py` — quasilinear TGLF-10 instability spectrum (ITG/TEM/ETG
    growth rates and mode identification from local plasma parameters)
  - `ballooning_solver.py` — second-order ODE eigenvalue solver in s-alpha geometry;
    binary-search marginal-stability finder; full stability diagram computation
  - `current_diffusion.py` — parallel current evolution PDE with neoclassical
    resistivity (Sauter-Angioni), ohmic heating, and bootstrap source
  - `current_drive.py` — ECCD, NBI, LHCD auxiliary current-drive models with
    absorption efficiency and radial deposition profiles
  - `ntm_dynamics.py` — modified Rutherford equation for neoclassical tearing modes
    (2/1, 3/2); ECCD stabilization factor; NTM controller with mode-tracking
  - `rwm_feedback.py` — resistive wall mode n=1 feedback with active coils, Galerkin
    gain computation, and passive-wall eigenvalue analysis
  - `sawtooth.py` — Porcelli-like trigger (shear at q=1), Kadomtsev reconnection
    crash model, density/energy conservation, SawtoothCycler with crash history
  - `sol_model.py` — two-point SOL model (upstream-to-target), Eich heat-flux width
    scaling (Goldston heuristic), sheath-limited and conduction-limited regimes
  - `rzip_model.py` — linearised tokamak vertical stability model (RZIp plant);
    eigenvalue-based growth rate; passive structure model
  - `integrated_scenario.py` — full integrated scenario simulator coupling transport,
    current diffusion, current drive, sawteeth, NTM, and SOL models; ships with
    ITER baseline, ITER hybrid, and NSTX-U preset scenarios
- **Phase 4 — Absolute control** (10 modules in `control/`):
  - `nmpc_controller.py` — nonlinear MPC with SQP over 20-step horizon; state/input
    box constraints and slew-rate limits on Ip, beta_N, q95, li, Te, nbar
  - `mu_synthesis.py` — D-K iteration for structured robust control; D-scaling
    optimization minimising structured singular value mu; MuSynthesisController
  - `realtime_efit.py` — streaming equilibrium reconstruction from partial
    measurements; coil-current-to-psi mapping; sub-10ms latency target
  - `gain_scheduled_controller.py` — PID gains scheduled on operating regime
    (Ip, beta_N); automatic interpolation with hysteresis-aware regime detection
  - `shape_controller.py` — plasma shape feedback via divertor/shaping coils;
    boundary-geometry Jacobian; x-point and separatrix tracking
  - `safe_rl_controller.py` — PPO wrapper with MHD constraint checker; vetoes
    actions violating stability limits; Gymnasium-compatible
  - `sliding_mode_vertical.py` — sliding-mode controller for vertical stability;
    continuous control law with dead-band saturation; configurable sliding surface
  - `scenario_scheduler.py` — shot timeline manager for startup→ramp→flattop→
    rampdown; actuator scheduling with power budgets; scipy.optimize trajectory
  - `fault_tolerant_control.py` — sensor/actuator fault detection via innovation
    monitoring; reduced-rank operation under faults; stuck-sensor reconstruction
  - `control_benchmark_suite.py` — standardised benchmark scenarios (step tracking,
    disturbance rejection, noise resilience) with JSON+Markdown report generation

### Fixed
- `np.trapz` → `scipy.integrate.trapezoid` across all files (numpy 2.x compat)
- Ballooning test hardened (alpha 0.9→1.5) for cross-platform robustness
- 46 mypy errors fixed across 17 files (no-any-return, attr-defined, assignment)
- scipy event function pattern refactored to class-based callable

### Changed
- 2,786 tests (178 files), 100% coverage, 26 CI jobs
- Version bump: v0.15.0 → v0.16.0

## [0.15.0] — 2026-03-11

### Fixed
- **GS\* stencil sign bug**: east/west coefficients in Jacobi, SOR, multigrid,
  and JAX solvers had the 1/(2R·dR) sign swapped — implementing the cylindrical
  Laplacian (∂²ψ/∂R² + (1/R)∂ψ/∂R) instead of the correct GS\* operator
  (∂²ψ/∂R² − (1/R)∂ψ/∂R). Python now matches Rust sor.rs. Verified via
  Solov'ev exact solution (< 1% error on 33×33 grid).
- **beta_N formula** (TokamakEnv): replaced dimensionally incorrect sqrt(Ip)
  expression with Troyon scaling β_N = c·T/Ip, calibrated to ITER baseline
- **gain_margin_db misnomer**: renamed to `stability_margin_db` (eigenvalue-based,
  not Bode gain margin); backward-compat alias retained
- **MPC docstring**: states clearly this is gradient-based trajectory optimization,
  not Rawlings-Mayne MPC
- **Physics citations**: Braginskii tau_eq, Martin L-H threshold, Troyon beta_N,
  Wesson q95 — all hardcoded constants now cite source
- **H-infinity Y Riccati tolerance**: tightened from 1.0 to 0.01
- **IPB98(y,2) RMSE gate**: tightened from 200% to 80%

### Added
- Solov'ev analytic equilibrium test (GS solver vs exact ψ)
- Crank-Nicolson CN-vs-Euler convergence test and pure diffusion decay test
- `validate-rmse` CLI command (full RMSE dashboard)
- 16 analytic regression tests in `test_p0_regression.py`

### Changed
- 2,420 tests, 0 failures, 105 skipped
- Version bump: v0.14.1 → v0.15.0

## [0.14.1] — 2026-03-11

### Fixed
- **Jacobi step**: add 1/R toroidal stencil (was Cartesian Laplacian, affecting
  fallback solver path)
- **Vacuum field**: multiply coil current by `turns` (was ignoring multi-turn coils)
- **JAX GS boundary**: use ψ_bdry=0.0 Dirichlet BC (was reading corner value)
- **UPDE Rust fast-path**: return `"Psi_global"` key (was `"Psi"`, breaking
  downstream code when Rust backend active)
- **Green's function**: divide by k² not k in toroidal formula
- **Bootstrap current**: use minor radius `a`, not domain extent `R_max−R_min`
- **lyapunov_v docstring**: range is [0, 2] not [0, 1]
- **Neural transport docstring**: honest about MLP architecture (not full QLKNN-10D)

### Added
- 13 analytic regression tests (`test_p0_regression.py`): Jacobi toroidal stencil,
  vacuum field turns scaling, UPDE key parity, GS boundary conditions, 2-oscillator
  Kuramoto exponential convergence, sub/supercritical phase transition thresholds
- `tutorial-smoke` CI job (tutorials 01, 04, 05)
- `[all]` optional-dependency group in pyproject.toml

### Changed
- MG parity tolerance widened (Rust multigrid uses Cartesian smoother)
- 2,417 tests, 99.99% coverage, 26 CI jobs

## [0.14.0] — 2026-03-10

### Added
- **PPO 500K cloud training** on JarvisLabs RTX5000 (3 seeds x 500K timesteps)
- PPO reward=143.7 beats MPC (58.1) and PID (-912.3), 0% disruption rate
- Reproducible: 3 seeds yield consistent +-0.2 mean reward
- Per-seed weights: `ppo_tokamak_seed{42,123,456}.zip`
- Benchmark report: `benchmarks/rl_vs_classical.json`
- Cloud training script: `tools/train_rl_upcloud.sh` (multi-seed, best-select)
- JarvisLabs automation: `tools/jarvislabs_train.py`

## [0.13.0] — 2026-03-10

### Added
- **JAX-differentiable Grad-Shafranov solver** (`jax_gs_solver.py`): full Picard
  iteration via `jax.lax.fori_loop` with damped Jacobi inner sweeps
- `jax.grad` through the complete equilibrium solve — closes autodiff depth gap
  with TORAX (JAX) and FUSE (Julia AD)
- `jax_gs_solve()` public API with NumPy fallback
- `jax_gs_grad_Ip()` convenience function for d(psi)/d(Ip) gradient
- 20 JAX GS tests: NumPy parity, boundary conditions, symmetry, autodiff
  (finite, nonzero, sign, beta_mix, finite-difference agreement)
- `examples/quickstart.py` — 30-second Python demo (equilibrium + transport +
  SNN compile + autodiff)
- README "Python in 30 Seconds" quickstart block

### Changed
- TokamakEnv reward: added survival bonus, progress shaping (Ng et al. 1999),
  increased disruption penalty — improves PPO learning speed
- JOSS paper updated: 57 modules, ~22,900 LOC, 2,201 tests, JAX GS mention
- Competitive analysis: equilibrium autodiff depth marked RESOLVED (5/6 gaps closed)

## [0.12.0] — 2026-03-10

### Added
- **QLKNN-10D trained neural transport model**: 3-layer MLP (10→128→64→3) trained
  on synthetic critical-gradient data (5000 samples, van de Plassche et al. 2020 paradigm)
- Training script `tools/train_neural_transport_qlknn.py` with `--synthetic` CI mode
  and `--data-dir` for real Zenodo dataset
- Auto-discovery: `NeuralTransportModel()` loads weights from `weights/` if present
- **PPO agent** on `TokamakEnv` via stable-baselines3 (`tools/train_rl_tokamak.py`)
- Gymnasium-compatible `GymTokamakEnv` wrapper with proper `spaces.Box` definitions
- PID and 1-step MPC baseline controllers for comparison
- RL vs classical benchmark (`benchmarks/rl_vs_classical.py`): PPO vs PID vs MPC
- `[rl]` optional dependency group (`stable-baselines3`, `gymnasium`)
- 20 QLKNN tests + 14 RL tests (weight loading, inference, training E2E, benchmark)

### Fixed
- NumPy 2.x deprecation: `int(data["version"])` → `int(data["version"].item())`
- TokamakEnv q95 formula: added elongation factor (q95 ≈ 3.0 at 15 MA, was 1.77)

## [0.11.0] — 2026-03-10

### Added
- **JAX-accelerated neural equilibrium** (`scpn_control.core.jax_neural_equilibrium`):
  JIT-compiled PCA + MLP surrogate for Grad-Shafranov equilibrium with GPU dispatch,
  `jax.grad` for adjoint-based shape optimization, and `jax.vmap` batch inference
- 13 new tests: JAX/NumPy parity, autodiff gradients, batched vmap, weight conversion
- JAX neural equilibrium tests added to `jax-parity` CI job

## [0.10.0] — 2026-03-10

### Added
- **JAX-accelerated transport primitives** (`scpn_control.core.jax_solvers`):
  Thomas tridiagonal solver, Crank-Nicolson diffusion operator, and batched
  transport via `jax.vmap` — all JIT-compiled, GPU-compatible, and
  differentiable via `jax.grad` for sensitivity analysis
- **Real DIII-D shot validation** (`tests/test_real_diiid_shots.py`): 95 tests
  validating data integrity, physical ranges, disruption labels, phase-sync
  pipeline handling, and disruption-precursor feature extraction against 17
  real DIII-D disruption shots (H-mode, VDE, beta-limit, locked-mode,
  density-limit, tearing, snowflake, negative-delta, high-beta)
- CI Job: `real-diiid` — validates against real DIII-D shot data (25 CI jobs total)
- CI: JAX solver parity tests added to `jax-parity` job
- **JOSS paper** (`paper.md`, `paper.bib`): submission-ready for Journal of
  Open Source Software review
- API docs: JAX transport primitives added to `docs/api.md`

## [0.9.0] — 2026-03-10

### Added
- `py.typed` PEP 561 marker for downstream IDE type inference
- `[jax]` optional dependency group (`jax>=0.4.20`, `jaxlib>=0.4.20`)
- `[loihi]` optional dependency group (`nengo>=4.0`, `nengo-loihi>=1.0`)
- CI Job 16: JAX backend parity (`jax-parity`) — validates `jax_traceable_runtime.py`
- CI Job 17: Nengo Loihi test (`nengo-loihi`) — validates `nengo_snn_wrapper.py`
- CI: Windows and macOS runners in python-tests matrix (Python 3.12)
- API docs: 16 previously undocumented modules added to `docs/api.md`
  (neural_equilibrium, neural_transport, stability_mhd, hpc_bridge,
  adaptive_knm, plasma_knm, analytic_solver, bio_holonomic, digital_twin_ingest,
  director_interface, fueling_mode, halo_re, hil_harness, jax_traceable,
  neuro_cybernetic, torax_hybrid_loop)
- VALIDATION.md: RMSE regression threshold table with sources

### Changed
- mypy: `disallow_untyped_defs = true`, `warn_return_any = true` across all 54 modules (134 annotations added)
- mypy: `files` simplified to `["src/scpn_control/"]` (full package)
- Codecov: `fail_ci_if_error: true` (was false with TODO)
- Removed `black` from dev extras (redundant with ruff-format)
- Cleaned 49 unused imports across 30+ test files (ruff auto-fix)
- U-002 (Nengo Loihi) marked RESOLVED
- U-006 (JAX CI) marked RESOLVED
- All 7 UNDERDEVELOPED_REGISTER items now RESOLVED

## [0.8.1] — 2026-03-10

### Added
- Rust H-inf: Padé(6,6) scaling-and-squaring `matrix_exp` replacing Euler discretization
- Rust H-inf: `zoh_discretize` matching Python `_zoh_discretize` (exact ZOH via matrix exponential)
- 6 new Rust tests for matrix_exp + ZOH (diagonal, nilpotent, large-norm, Euler agreement)
- `benchmarks/e2e_control_latency.py` — honest E2E pipeline benchmark (sensor→equilibrium→transport→control→actuator)

### Changed
- README, pitch.md, use_cases.md, VALIDATION.md, competitive_analysis.md: honesty sweep
  - "formal verification" → "contract-based checking"
  - "DIII-D shot replay" → stated as synthetic mock data
  - Comparison table: kernel step ≠ full control cycle caveat
  - VALIDATION.md: Scope & Limitations table, "What does NOT exist" list
  - use_cases.md: added "Real tokamak data" and "Peer-reviewed papers" rows (both No)
- U-001 marked RESOLVED in UNDERDEVELOPED_REGISTER
- Stale doc counts: 2019 tests, 118 files, 54 modules

## [0.8.0] — 2026-03-09

### Added
- Python fallback for `rust_svd_optimal_correction()`: truncated SVD pseudoinverse
  with singular-value cutoff (`_python_svd_optimal_correction`)
- Python fallback for `RustSPIMitigation`: 3-phase disruption sim matching Rust
  spi.rs constants (Assimilation → ThermalQuench → CurrentQuench)
- Python fallback for `rust_multigrid_vcycle()`: delegates to
  `FusionKernel._multigrid_vcycle` with isolated instance
- `require_bounded_float` validator: arbitrary inclusive/exclusive bound checks
- `require_finite_array` validator: ndim/shape constraints + finiteness
- `tests/test_rust_fallbacks.py` — 16 tests (SVD, SPI, multigrid fallbacks)
- Input validation on `RustSPIMitigation.__init__()` for both Rust and Python paths

### Changed
- `h_infinity_controller.py`: inline `np.isfinite` checks replaced with shared validators
- `disruption_predictor.py`: 7 inline checks replaced with shared validators
- `advanced_soc_fusion_learning.py`: 8 inline checks replaced with shared validators
- U-003, U-004, U-005 marked RESOLVED in UNDERDEVELOPED_REGISTER
- U-007 marked RESOLVED (shared validators in place, P1 modules converted)

## [0.7.1] — 2026-03-09

### Added
- Enterprise root files: SUPPORT.md, GOVERNANCE.md, CONTRIBUTORS.md, NOTICE.md,
  ARCHITECTURE.md, VALIDATION.md, REUSE.toml, .gitattributes, .dockerignore,
  Makefile, requirements-dev.txt
- Workflows: pre-commit.yml, codeql.yml, scorecard.yml, release.yml, stale.yml
- .github/ISSUE_TEMPLATE/config.yml (Security Advisories + SUPPORT.md links)
- `_typos.toml` domain allowlist (46 terms: physics, plasma, Rust identifiers)
- 17 GitHub labels (dependencies, ci, security, performance, plasma-control, etc.)

### Changed
- ci.yml, docs-pages.yml, publish-pypi.yml: SHA-pinned all actions, `permissions: {}`,
  concurrency groups, SPDX headers
- pyproject.toml: ruff UP/SIM rules, coverage exclude_lines, dev deps added
- .pre-commit-config.yaml: check-toml + crate-ci/typos hooks
- dependabot.yml: commit-message prefixes, dependency groups
- FUNDING.yml: github sponsor link
- SECURITY.md: Security Advisories as preferred reporting method
- scorecard-action bumped v2.4.0 → v2.4.3
- black bumped 25.1.0 → 25.11.0

### Repository
- Squash-only merge, delete-branch-on-merge, discussions enabled
- 10 topic tags, homepage set to GH Pages docs URL
- Dependabot PR triage: 1 merged (#21 black), 4 closed (incompatible Cargo bumps)

## [0.7.0] — 2026-03-02

### Added
- `tests/test_nengo_snn_wrapper.py` — 14 mocked tests for the only untested module (389 LOC)
- `tests/test_e2e_compile_to_control.py` — 5 E2E integration tests (compile → artifact → controller → step)
- `require_range` validator in `core/_validators.py`
- `///` doc comments on 11 public Rust functions (`mpi_domain.rs`, `vmec_interface.rs`)
- `keywords` and `categories` in all 5 Rust `Cargo.toml` files
- Paper 27 Reviewer Integration page in mkdocs nav

### Fixed
- Public API typo: `TokamakTopoloy` → `TokamakTopology` (deprecated alias retained)
- `print()` → `logging` in 13 control modules (58 call sites total)
- Remaining `Union[str, Path]` → `str | Path` in 3 files (`eqdsk.py`, `realtime_monitor.py`, `artifact.py`)
- CLI hardcoded module/test counts → dynamic `Path.rglob` computation
- Magic number `b0=5.3` → named constant `ITER_B0_VACUUM_T` with citation
- Stale doc counts across README, architecture, pitch, use_cases, CONTRIBUTING (53 modules, 115 files, 1969 tests, 15 CI)
- Dead `grid_index()` function removed from Rust `gmres.rs`

### Changed
- Coverage gate ratcheted: `fail_under = 62` → `85`
- `from __future__ import annotations` added to `core/__init__.py` and `scpn/__init__.py`
- 4 duplicate validators in `halo_re_physics.py` replaced with `core._validators` imports
- ROADMAP.md rewritten: v0.6.0 moved to Shipped, unshipped items to v0.7.0+
- 12 additional tests across 5 thin test files

## [0.6.0] — 2026-03-02

### Added
- `.editorconfig` and `.github/CODEOWNERS`
- Copyright headers on all 3 CI workflow files
- `repository` field in all 5 Rust Cargo.toml files
- `tests/test_validators.py` — 49 parametrized tests for `core/_validators.py`
- `tests/test_phase_properties_extended.py` — 14 Hypothesis property tests (knm, upde, adaptive_knm)
- Paper 27 citations on `OMEGA_N_16` and `build_knm_paper27` constants

### Fixed
- `.zenodo.json` license `"MIT"` → `"AGPL-3.0-or-later"` (matches pyproject.toml)
- `docs/api.md` version stuck at `"0.5.0"` → `"0.6.0"`
- `print()` → `logger.info()` in `spi_mitigation.py` (3 sites)
- Anti-slop: renamed unused param `proposed_action` → `_proposed_action`, deleted 4 narration comments in `cli.py`
- Flaky timing test: absolute 5s threshold → relative warmup baseline
- Dead `DEFAULT_GAIN` constant removed from Rust `optimal.rs`

### Changed
- Typing modernization: `from __future__ import annotations` + `Optional[X]` → `X | None` in 21 files
- Shared test fixtures extracted to `conftest.py` (3 controller test files deduplicated)
- `pyproject.toml` keywords + author email added

## [0.5.2] — 2026-03-02

### Fixed
- Codecov `fail_ci_if_error: false` → `true` (matches v0.5.0 CHANGELOG claim)
- Stale doc counts: architecture.md (17→21 modules, 1243→~1900 tests), pitch.md, use_cases.md
- Bug report template version placeholder 0.3.0 → 0.5.x
- Development.md release example v0.2.1 → vX.Y.Z
- Magic number citations: ITER Physics Basis for SHOT_DURATION, TARGET_R, TARGET_Z, u_max
- Anti-slop: "leveraging" → "using" (nengo), narration → TMR median voter (hil), dead pass block (eqdsk)

### Changed
- `docs/changelog.md` synced with root CHANGELOG (was frozen at v0.3.3)
- ROADMAP.md rewritten for v0.5.x shipped state
- `require_non_negative_float` added to `core/_validators.py`; scaling_laws and spi_mitigation use shared validators
- `control/__init__.__all__` includes `normalize_bounds`
- Legacy typing imports replaced in 7 files (phase/ + scpn/): Optional→|None, List→list, Dict→dict, Tuple→tuple

## [0.5.1] — 2026-03-02

### Fixed
- CITATION.cff DOI description said v0.4.0 (now v0.5.0)
- `docs/api.md` version string stuck at 0.3.3 (now 0.5.0)
- CONTRIBUTING.md stale test count (1243→~1900), coverage (50→62%), CI jobs (17→16)
- `docs/development.md` stale coverage (55→62%) and release process (no longer uses `__init__.py`)
- Two "Approximate" comments cleaned per anti-slop rule #4

### Changed
- `require_int` deduplicated: canonical `core/_validators.py` replaces 3 copies
- `deny.toml` wildcards: "allow" → "deny"
- Pre-commit: added `check-merge-conflict`, `detect-private-key` hooks
- Paper 27 phase dynamics page added to mkdocs nav

### Added
- `SECURITY.md` responsible disclosure policy
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- U-007 in UNDERDEVELOPED_REGISTER (np.isfinite boilerplate)
- GitHub issues gh-13, gh-14, gh-15 for U-003/004/005 Rust fallback gaps

## [0.5.0] — 2026-03-02

### Fixed
- `__version__` now derived from package metadata (PEP 621), was stuck at 0.3.3
- Rust H-inf `update_discretization` TODO tracked as gh-10, param renamed `_dt` → `dt`
- README test/CI job counts updated to actual values

### Added
- 27 new tests: 9 Rust (h_infinity, xpoint, bfield, chebyshev) + 18 Python (rust_compat_wrapper)
- `cargo-deny` supply-chain policy (`deny.toml`) + CI Job 15
- `ruff format --check` CI gate + pre-commit hook
- `UNDERDEVELOPED_REGISTER.md` tracking 6 known gaps
- Python 3.13 in CI matrix

### Changed
- Coverage gate: 55% → 62% (actual: 93%)
- Codecov `fail_ci_if_error: true`
- Pre-commit: Rust hooks no longer gated to `stages: [manual]`
- Removed unused `proptest` dev-dependency from 3 Cargo.toml files

## [0.4.0] — 2026-03-01

### Added
- Real-time adaptive Knm engine driven by tokamak diagnostics
  (`AdaptiveKnmEngine`, `DiagnosticSnapshot`, `AdaptiveKnmConfig`)
- Five adaptation channels: beta scaling, MHD risk amplification,
  coherence PI control, per-element rate limiting, Lyapunov guard veto
- `K_override` parameter on `UPDESystem.step()`, `.run()`, `.run_lyapunov()`
- `RealtimeMonitor.from_plasma()` constructor with adaptive engine support
- Diagnostic kwargs (`beta_n`, `q95`, `disruption_risk`, `mirnov_rms`) on `tick()`
- 46 new tests (1888 total)

### Changed
- `.zenodo.json`: complete Zenodo metadata (related identifiers, communities, notes)
- `CITATION.cff`: version bump, date update

## [0.3.3] — 2026-02-27

### Changed
- License: AGPL-3.0 → AGPL-3.0-or-later dual (137 files, zero AGPL remaining)
- README: `pip install -e "."` → `pip install scpn-control` (PyPI install path)
- CONTRIBUTING: CI job count 14 → 17

## [0.3.2] — 2026-02-27

### Added
- VectorizedSCLayer + Rust backend path in SNN compiler (512× real-time)
- Two-tier import: v3.8.0+ preferred → legacy bit-ops → numpy float fallback
- Test for v3.8 detection and VectorizedSCLayer forward-path benchmark
- sc-neurocore listed first in optional deps table (crown jewel)

### Changed
- README: engine callout and dep table updated for sc-neurocore

## [0.3.0] — 2026-02-27

### Added
- Ruff linter (E/F/W/I/B rules) — CI job + pyproject.toml config
- Property-based tests for phase/ module (Hypothesis, 11 properties)
- Gymnasium-compatible TokamakEnv (control/gym_tokamak_env.py, 10 tests)
- IMAS/OMAS equilibrium adapter (core/imas_adapter.py)
- CLI `scpn-control info` command (version, Rust status, weights, Python/NumPy)
- Weight provenance manifest (reproduction commands, hardware, training config)
- Paper 27 + H-infinity notebooks in CI smoke tests

### Fixed
- API docs: wrong snapshot keys (R→R_global, V→V_global, lambda→lambda_exp)
- API docs: wrong UPDESystem constructor and LyapunovGuard API examples
- README/CHANGELOG: "14 CI jobs" → actual count, test counts updated
- `_rust_compat.py`: calculate_thermodynamics/vacuum_field now delegate to Python
- 163 ruff auto-fixes (whitespace, import sorting, unused imports)
- 26 manual ruff fixes (raise-from, unused variables, one-liners, E402)
- Bandit now fails on medium+ severity (was --exit-zero)
- TeX build artifacts (.aux/.log/.out/.toc) excluded from repo

### Changed
- Coverage threshold raised from 50% → 55% (actual: 61%)
- CI: 12 → 13 jobs (added python-lint)
- Test suite: 680 → 701 tests (50 test files)

## [0.2.0] — 2026-02-26

### Added
- Paper 27 phase dynamics engine (`src/scpn_control/phase/`, 7 modules)
- Kuramoto-Sakaguchi step with global field driver (kuramoto.py)
- 16x16 Knm coupling matrix builder with calibration anchors (knm.py)
- UPDE multi-layer solver with PAC gating (upde.py)
- LyapunovGuard sliding-window stability monitor (lyapunov_guard.py)
- RealtimeMonitor tick-by-tick UPDE + trajectory recorder (realtime_monitor.py)
- PhaseStreamServer async WebSocket live stream (ws_phase_stream.py)
- CLI `scpn-control live` command for real-time WS phase sync server
- Streamlit WS client (`examples/streamlit_ws_client.py`)
- Streamlit Cloud deployment (`streamlit_app.py`, `.streamlit/config.toml`)
- Mock DIII-D shot generator (`tests/mock_diiid.py`)
- E2E phase sync with shot data tests (`tests/test_e2e_phase_diiid.py`)
- Phase sync convergence video (MP4 + GIF) and generator script
- PyPI publish script (`tools/publish.py`)
- Rust `upde_tick()` in control-math + PyRealtimeMonitor PyO3 binding

### Changed
- CI expanded from 6 to 12 jobs
- Test suite expanded from 482 to 680 tests (680 passing, 94 skipped)
- README updated with `<video>` MP4 embed, Streamlit Cloud badge
- `.gitignore` updated to allow docs GIF/PNG and Streamlit config

## [0.1.0] — 2026-02-19

### Added
- Initial extraction from scpn-fusion-core v3.4.0
- 41 Python source files (minimal control transitive closure)
- 5 Rust crates (control-types, control-math, control-core, control-control, control-python)
- Slim PyO3 bindings (~474 LOC, control-only)
- Clean `__init__.py` files (no matplotlib/GPU/ML import bombs)
- Click CLI with 4 commands (demo, benchmark, validate, hil-test)
- Streamlit dashboard (optional, `[dashboard]` extra)
- CI workflow (6 jobs: python-tests, rmse-gate, rust-tests, rust-python-interop, rust-benchmarks, rust-audit)
- 37 test files (482 passing, 122 skipped, 0 failures)
- 16 disruption shot reference data files
- 8 SPARC EFIT equilibria
- 3 pretrained weight files
- Validation scripts and tools
- Stochastic Petri Net logic map header (`docs/scpn_control_header.png`)

### Changed (vs scpn-fusion-core)
- Required deps: numpy, scipy, click ONLY (was: numpy, scipy, matplotlib, streamlit)
- matplotlib, streamlit, torch, nengo moved to optional extras
- All imports renamed: `scpn_fusion` -> `scpn_control`, `scpn_fusion_rs` -> `scpn_control_rs`
- Rust workspace reduced from 11 crates to 5
- CI reduced from 13 jobs to 6
- `hpc_bridge.py` relocated from `hpc/` to `core/` subpackage
- Import guards added for excluded modules (stability_analyzer, global_design_scanner, imas_connector, diagnostics.forward, fusion_ignition_sim)
