# SCPN Control v0.24.0 Release Notes

v0.24.0 is a minor release for the v0.2x control-evidence line. It carries an
item-by-item qualification of every validation and tool module, real-data
binding for the MAST campaign tools, gradient corrections in the surrogate and
ITER trainers, source-bound benchmark provenance, and dependency and supply-chain
security work. Some campaign outputs, arguments and public surfaces were renamed
or removed, so the release contains behaviour-affecting changes and is a minor
(not patch) bump under 0.x SemVer.

## What changed

### Breaking changes

- The flight-simulator controller campaigns (`validation/stress_test_campaign.py`,
  `validation/controller_comparison.py`) rename their metrics: `disrupted`,
  `disruption_rate`, `energy_efficiency` and `mean_energy_efficiency` are now
  `exceeded_error_limit`, `error_limit_rate`, `actuator_tracking` and
  `mean_actuator_tracking`. The limit is the named heuristic
  `MEAN_AXIS_ERROR_LIMIT_M` (0.5 m), not a physical disruption criterion. The
  disruption extension factor and the per-episode disruption time are removed,
  and so are the stress campaign's `noise_level` and `delay_ms` arguments, which
  were never applied. A failed episode now raises `EpisodeFailure` and the
  commands exit 1 instead of reporting zero-filled metrics.
- The duplicate public shape controller, whose response matrix was a fixed
  random array, is removed; `FreeBoundaryTrackingController` is the sole
  documented plasma shape-control owner.
- The mislabelled H-infinity observer/LQR paths are replaced by the normalised
  continuous-time DGKF central output-feedback controller; its API requires
  explicit normalised feedthrough matrices and fails closed on every existence
  condition.
- Seven console scripts that were declared in wheel metadata but could not
  resolve after installation are removed.

### Qualification of the validation and tool surface

- Every tracked validation and tool module was read in full and checked for
  accurate docstrings, strict typing and complete line and branch coverage on
  the CI pins. Ten manual GPU gyrokinetic campaign scripts keep a documented
  runtime-limited status until an owner-scheduled GPU run.
- Validation and tool commands run as documented: as a script and with
  `python -m` from a checkout without an installed package, without changing the
  caller's import path when imported.
- Number guards refuse oversized and non-finite inputs with the documented
  `ValueError`, and output aliases that would overwrite inputs are refused.

### Data, training and control fixes

- The MAST disruption evaluation binds its shots to the real-data manifest, the
  FAIR-MAST manifest builder refuses non-finite plasma current and duplicate
  shots, and the EFM campaign publisher verifies its candidate and evaluation
  inputs.
- The surrogate trainers and the ITER neural-equilibrium trainer take gradients
  from one forward pass and keep the best-epoch weights; the gradient MPC uses
  the exact horizon gradient; the GK online learner trains fully and admits on
  the same slice.
- The magnetic axis is read through one checked accessor; the gyro-Bohm
  coefficient provenance is corrected; benchmark records are bound to their
  source bytes and runtime.

### Security and supply chain

- `url` 2.5.9 in both Rust locks (RUSTSEC-2026-0592); aligned Python and Rust
  dependency locks; advisory exception gates for the Studio and Rust locks that
  expire and name the excepted package.
- Pinned GitHub Actions refreshed to their verified tag commits.

See `CHANGELOG.md` for the complete, itemised list.

## Evidence boundary

This release does not certify facility deployment, PREEMPT_RT operation,
external-code equivalence, P-EFIT predictive validity, TORAX parity, or plant
PCS readiness. The flight-simulator campaigns cannot run on either repository
ITER configuration and refuse instead of reporting numbers. The release evidence
remains a bounded software, CI, dependency, and local validation checkpoint, and
the strict admission gates keep facility, measured-shot, target-hardware, and
external-code claims blocked until matching evidence exists.

## Publication boundary

This page records source-level release history only. Tag, package-index, hosted-CI, security, and deployment status are external mutable state and must be verified at the corresponding service; this document does not expose or prescribe release operations.

## Practical use and scope

Use this note to verify what changed in the 0.24.0 line.

- Upgrade from `0.23.0` for the qualified validation and tool surface, the MAST
  real-data binding and the trainer fixes; update any code that read the renamed
  campaign metrics, passed `noise_level` or `delay_ms`, used the removed shape
  controller or relied on the removed console scripts.
- Keep broader facility claims blocked unless their strict admission gates accept
  matching external, measured-shot, target-hardware, and review evidence.
