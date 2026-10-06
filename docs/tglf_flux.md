# Standalone signed TGLF fluxes

The installed package reads retained GACODE output: `read_tglf_fluxes` returns
signed normalised species fluxes with all reported spectral modes from a run
directory, and needs no provider.

The launcher that produces such a directory is `TGLFFluxSolver` in
`validation/tglf_launcher.py`, a validation command of the repository and not
part of the installed package. It executes an existing GACODE `key=value` input
deck, creates a fresh directory for each execution and retains the input,
output, stdout, stderr and a hashed execution receipt. It requires POSIX, a
repository checkout and a configured GACODE installation. It lives outside the
package because the hosted test environments have no provider, so its code
cannot be exercised there.

The output is electron-first, followed by ions. Particle, energy, momentum
and exchange fluxes retain their signs. They are **not diffusivities**.
Physical conversion requires the deck's reference density, temperature,
length, mass, magnetic field and coordinate convention. `physical_tglf_flux` converts particle and energy moments to SI when given
explicit `TGLFReferenceUnits` matching the GYRO-normalised deck. It uses
Gamma_GB = n0*cs*(rho_s/a0)^2 and Q_GB = T0*Gamma_GB, retains signs, and
returns flux across physical minor-radius surfaces. A different coordinate
requires its matching derivative and volume Jacobian. It does not infer the
reference magnetic field from toroidal field or separate diffusion from pinch.
No automatic `GKLocalParams` mapping or `TransportSolver` coupling is supplied.

The existing coefficient-only `external_gk` path remains unqualified;
this raw-flux interface provides the provider boundary needed for its repair.
Passing file checks establishes structural consistency, not physical
calibration or proof of numerical convergence.

```python
from pathlib import Path

from scpn_control.core.tglf_flux import read_tglf_fluxes
from validation.tglf_launcher import TGLFFluxSolver  # repository checkout only

solver = TGLFFluxSolver(
    work_dir=Path("/path/to/owned/run-evidence"),
    binary="/path/to/configured/tglf",
)
flux = solver.run(Path("/path/to/input.tglf"), timeout_s=30)
print(flux.particle_flux_gb)

# Any later reader needs only the retained directory.
assert read_tglf_fluxes(flux.run_dir) == flux
```

An optional `environment` mapping supplies child-only GACODE/dependency
settings. Environment values are not included in execution receipts.
Timeouts kill the process group; failures retain their run directories.
Callers own evidence retention. The captured test specimen comes from an
actual upstream default run; its provenance and SHA-256 values accompany it.
Real execution tests are in `tests/test_tglf_launcher.py`. They require
`SCPN_TGLF_BINARY` and optionally `SCPN_TGLF_ENV_JSON`; without them those
tests are explicitly skipped. The reader tests in `tests/test_tglf_flux.py`
run everywhere on the captured specimen.

See the [TGLF flux API][scpn_control.core.tglf_flux] for the reader and result
contracts.

See the [TGLF unit API][scpn_control.core.tglf_units] for reference units and SI
conversion contracts.

## Execution and retained evidence

Logs go directly to files. Every completed launch attempts to terminate its
owned POSIX process group, including after success, timeout or interruption.
Leader reaping has a separate two-second cleanup deadline. Original wait
exceptions and cancellation propagate after cleanup; cleanup failures are
recorded in `execution.json` and attached to the exception. A descendant that
creates another session or process group is outside this termination contract.
Its inherited log descriptors do not hold the call open. This is not a process
sandbox or a guarantee that escaped descendants terminate.

The receipt hashes the same immutable byte snapshots that the parser consumes.
Retained outputs, the copied input and launcher identity are checked before
publishing success. The launcher hash is captured before launch and does not
authenticate the dependent executable or libraries. Evidence directories remain
mutable: `read_tglf_fluxes` verifies a present successful execution receipt and
rejects changed input/output files. A receipt-free directory is parsed for
structural consistency only. These are point-in-time custody checks, not a
filesystem lock or a promise against changes after return.

A nonzero normalised particle or energy moment that rounds to zero during SI
conversion is rejected. Exact zero and representable subnormal results remain
valid; no epsilon threshold or clipping is applied.

## Moment precision and supported output contract

Flux moments come from `out.tglf.sum_flux_spectrum`. In GACODE's
`write_tglf_sum_flux_spectrum`, each row already contains its ky quadrature
weight and the sum over modes. The reader sums these increments once across
ky and all configured fields. It returns particle flux, energy flux, toroidal
stress and exchange; the separate parallel-stress column is validated but
is not returned as toroidal momentum.

`out.tglf.gbflux` uses the Fortran `1pe11.4` format: five significant digits.
Rounding species independently can spoil ambipolarity in a multi-ion case.
The reader uses this file only to cross-check each integrated moment against
its printed decimal half-unit rounding interval. It does not relax the
transport charge tolerance, overwrite electron flux, or fall back to coarse
moments. A disagreement is refused; a passing comparison is an artifact
consistency check, not an error bound on the physical turbulence model.

The resolved `input.tglf.gen` is also required and hashed. The supported
contract is GYRO normalisation, kinetic electrons, transport-model flux
output, matching species/spatial/mode dimensions, and complete ordered
species/field blocks. `USE_BPER` selects two fields and `USE_BPAR` selects
three, matching the upstream writer. Each block must contain exactly the
reported number of ky rows and five finite moments. Missing, duplicate,
reordered, nonfinite or overflowing spectra are refused. Archives and older
receipts lacking either new required artifact must be reacquired or migrated
with verifiable source evidence; the reader does not silently upgrade them.

Real-provider regressions cover an explicit local three-species Miller case
through SI conversion and a conservative face step, plus two- and three-field
electromagnetic executions. The face-step case repeats prescribed local flux
across faces, uses a common ion temperature and a charge-two ion with
mass/reference ratio two. It establishes this numerical boundary, not a
calibrated helium model, automatic radial sampling or coupled convergence.

## Physical Miller input construction

`miller_tglf_deck` constructs a nonrotating, isotropic GYRO SAT0 input from
`TGLFSpecies`, `TGLFMillerGeometry` and explicit `TGLFReferenceUnits`. It
accepts physical `dn/dr` and `dT/dr`, includes all supplied species in the
pressure derivative and effective charge, and checks both density and density-
gradient charge neutrality. Absent species must be omitted explicitly.
Electron collision frequency is an explicit input in the TGLF convention;
the builder does not guess it from `nu_star` or choose a collision formula.

For reference length `a0`, local minor radius `r`, and pressure `p` in Pa:

- `RLNS = -a0/n * dn/dr`, `RLTS = -a0/T * dT/dr`;
- `Q_PRIME_LOC = q*a0²/r * dq/dr`;
- `P_PRIME_LOC = μ0/(4π Bunit²) * q*a0²/r * dp/dr`;
- `S_KAPPA_LOC = r/kappa * dkappa/dr`, `S_DELTA_LOC = r * ddelta/dr`.

The last two follow the current geometry kernel and
[GACODE input definitions](https://gacode.io/tglf/tglf_table.html).
The legacy `tglf_TM_driver.f90` example uses different shape-shear formulas;
copying those formulas would be wrong when the shape derivatives are nonzero.
Finite-difference tests of physical Miller surfaces check both radial shape
derivatives, alongside real executions with nonzero shape gradients.

The caller supplies Bunit from matching magnetic geometry; toroidal field is
not silently substituted. All velocities and shears, elevation, squareness and
higher shape harmonics are explicitly zero. Both supported SAT0 mode fits
(two and four modes) are available, including electromagnetic field switches.
Other saturation models and rotation need their own qualified contract.
Unlisted numerical defaults remain provider-owned and are retained with the
execution. The provider may apply presets after generating `input.tglf.gen`;
the builder selects SAT0/GYRO with two/four modes to avoid the known unit/mode
rewrites. A generic supplied deck's generated input is not proof that all
internal provider settings remain unchanged.

```python
from scpn_control.core.tglf_miller import (
    TGLFMillerGeometry, TGLFSpecies, miller_tglf_deck,
)
from scpn_control.core.tglf_units import TGLFReferenceUnits

reference = TGLFReferenceUnits(5e19, 2.0, 2.0, 3.34524384738e-27, 2.0)
electron = TGLFSpecies(-1, 9.1093837139e-31, 5e19, 2.0, -2.5e19, -3.0)
ion = TGLFSpecies(1, reference.mass_kg, 5e19, 3.0, -2.5e19, -3.0)
geometry = TGLFMillerGeometry(1.0, 6.0, 2.0, 2.0, elongation=1.7, triangularity=0.3)
deck_text = miller_tglf_deck(
    reference, (electron, ion), geometry,
    electron_collision_rate_s=34195.32091515598,
)
```

This is a prescribed local example. Constructing a deck does not certify
nested surfaces, force balance or machine calibration. Radial sampling,
profile interpolation, rotating/anisotropic species and coupled convergence
remain separate work; repeating a local flux at every face does not supply them.
The current conservative regression uses a cylindrical volume/area harness;
the explicit metric route additionally supports Miller dV/dr and cell volumes.
A separate real four-face regression uses distinct local TGLF inputs and Miller
metrics for three successive frozen-flux stages. Each later stage reconstructs
positive face profiles and physical radial gradients from the updated nodes,
then executes four fresh provider runs with matching local reference scales.
The twelve retained runs check input/output custody, adjacent inventories and
accumulated boundary exchange. This is a prescribed collisionless electron/ion
case with fixed shape and magnetic reference, not a reusable orchestration API,
coupled temporal-convergence study or self-consistent equilibrium.

See the [Miller geometry API][scpn_control.core.tglf_miller.TGLFMillerGeometry],
[species API][scpn_control.core.tglf_miller.TGLFSpecies] and
[input-deck factory][scpn_control.core.tglf_miller.miller_tglf_deck] for the
corresponding input contracts.

`out.tglf.scalar_saturation_parameters` is required as the seventh captured
artifact. Its effective `SAT_RULE` must match the request and its effective
`UNITS` must remain `GYRO`. Real SAT2/3 executions demonstrate why this matters:
the provider can retain requested `GYRO` in `input.tglf.gen` while reporting
`CGYRO` after startup presets. Those runs are refused by this GYRO-only
boundary; real SAT1 retaining GYRO remains admissible. This check establishes
the declared model convention and does not assert that every numerical default
is unchanged. Missing or duplicate effective settings are refused, and the
whole scalar-output file participates in receipt custody.

Numeric decoding refuses a nonzero decimal token that would round to binary64
zero, including Fortran D exponents. This applies to flux moments and numerical
grid/ky/eigenvalue inputs. Exact zeros and representable signed subnormals
remain valid. Deliberately corrupted-output tests exercise this boundary after
real GACODE execution; they do not imply that its REAL64 writer normally emits
out-of-range decimal tokens. Such runs retain a failed validation receipt.
