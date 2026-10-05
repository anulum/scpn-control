# Validation boundaries

This page records current numerical and evidence limitations. It is a public
description of observed behavior, not an operational backlog or work order.

## Equilibrium source consistency

The bounded equilibrium lane computes the source-balanced Grad-Shafranov
residual

$$
\frac{\lVert\Delta^*\psi + \mu_0 R J_\phi\rVert}
{\max(\lVert\mu_0 R J_\phi\rVert,\Delta\psi)}.
$$

The earlier source-free proxy misclassified shaped and high-beta equilibria by
omitting the reconstructed $R p' + FF'/(\mu_0 R)$ source. The current validator
reconstructs $J_\phi$ from GEQDSK $p'$ and $FF'$ profiles.

The available repository inputs have distinct evidence classes:

- SPARC GEQDSK inputs are `public_reference` design equilibria, not measured
  facility shots.
- DIII-D-like GEQDSK fixtures are `synthetic` and exercise numerical plumbing.
- A q95 value read from the same GEQDSK is a self-consistency observation, not
  an independent reconstruction comparison.

These inputs can pass a computational threshold without admitting physics,
measured-shot, facility, public, or production claims.

## Disruption-prediction replay

The repository replay data are synthetic. Recall and false-positive-rate
results therefore describe the fixed-weight heuristic on those fixtures only.
They are not a facility-database ROC, prospective warning-time study, or
commissioned disruption-mitigation result.

Malformed or unsafe NPZ payloads fail closed per file and are never loaded with
pickle enabled.

## Transport scaling

The transport lane compares the implemented IPB98(y,2) calculation with curated
published-reference rows. Its evidence class is `public_reference`. Agreement
within the declared uncertainty band is a computational comparison against
those rows; it does not establish a new multi-machine experimental validation.

## Campaign-level interpretation

`validation/validate_real_shots.py` now emits the schema
`scpn-control.reference-evidence-validation.v1`. Every lane declares exactly
one of `real`, `public_reference`, `synthetic`, or `local_proxy` and reports
data provenance separately from computational success.

The repository campaign mixes public-reference and synthetic evidence. Its
real-shot, facility, public-claim, and production admissions are therefore
fail-closed even when every numerical lane passes. The JSON and Markdown
reports display those fields independently so a green calculation cannot be
read as a green facility claim.

## Reproduction

```bash
python validation/validate_real_shots.py --help
python validation/validate_real_shots.py \
  --output-json artifacts/reference_evidence_validation.json \
  --output-markdown artifacts/reference_evidence_validation.md
```

The [validation guide](validation.md) explains the wider evidence taxonomy and
the generated [physics traceability report](physics_traceability.md) identifies
component-level claim status.


## Multi-machine validation

The compatibility facade remains `validation.multi_machine_validation`.
Defining owners are `validation.machine_inputs` (inputs and presets),
`validation.synthetic_diagnostics` (legacy noisy examples),
`validation.confinement_reference` (local scaling comparison) and
`validation.equilibrium_execution` (actual fixed-boundary execution).
Existing defined-name imports and numerical statements are preserved; defining
module metadata now identifies these owners. Native examples/renderers and the
ordinary documentation gate inspect the defining owners directly.

`validation.multi_machine_validation.MultiMachineValidator` currently refuses
`run_all`, `save_json` and `save_markdown` with `RuntimeError`. Its former
seven passing metrics were generated from random values without evaluating the
machine configuration or physical models. Those legacy outputs cannot support
claims of equilibrium convergence, transport agreement, conservation, stability
or diagnostic reconstruction.

The current refusal occurs before generating results or opening output files.
Real machine-bound model execution and reference evidence are required before
this API can publish validation results. Machine presets and the separately
available synthetic diagnostic examples do not satisfy that requirement.


### Capturing machine inputs

`MachineConfig.snapshot(rho)` captures the actual scalar parameters and sampled
`ne`, `Te` and `Ti` profiles in versioned JSON with explicit units. Hash its UTF-8
bytes to identify the sampled inputs for a future model run. This identifies
values; it does not identify callback source code or attest measured provenance.
The radius grid must increase strictly from zero to one with at least three
points. Invalid geometry, nonfinite values, negative profiles and mismatched
profile shapes are rejected. Zero edge values are retained; a solver requiring
positive edge values needs an explicitly supplied boundary contract.

Callbacks receive separate copies of the grid, and each output is captured
before the next callback executes. The returned string remains unchanged if
callbacks or machine objects are subsequently modified. Stateful callbacks can
produce different snapshots on later calls; execute the eventual model against
the captured inputs rather than resampling them. No equilibrium, transport,
conservation, stability or reconstruction result is produced by input capture.
The full seven-domain campaign remains unavailable pending its model and
reference integrations. Executable examples live in the native docstrings of
`MachineConfig.snapshot` and `ConfinementReference.evaluate`; the owning tests
run them against the actual public APIs and shipped reference files. Computing
a digest of these local files binds the example inputs but supplies no
independent provenance or trust anchor.


### Confinement reference comparison

`ConfinementReference.evaluate()` executes the existing IPB98(y,2) implementation
on one explicit CSV operating point. The caller supplies the reference and
coefficient paths with expected SHA-256 digests, zero-based data-row index,
machine/shot identity, source classification and positive relative-error
tolerance. Python and NumPy booleans are rejected as tolerances. Each file is
read once, and those exact bytes are hashed and parsed.
Changed bytes, ambiguous CSV headers, mismatched identities and invalid numeric
inputs are rejected. Coefficient JSON also rejects duplicate keys at any depth
and nonfinite numeric tokens, including overflow and unused metadata. Hash
agreement alone does not establish a unique, valid model interpretation.

The returned JSON includes the source row, digests, input units, prediction and
reference in seconds, relative error and the declared tolerance. It takes loss
power and isotope mass explicitly from the row; preset auxiliary power is not a
substitute. `comparison_pass` reports only the declared numerical comparison.
Source classification remains caller-declared, training-domain membership is
not assessed, and action/facility authority remains false. Supported source
classes are `derived_calibration`, `design_reference` and `synthetic`; none
establishes held-out empirical validation.

With the shipped coefficient/reference bytes and a 10% tolerance, the ITER design
row predicts about 3.606 s against 3.70 s and passes the comparison. JET row 92436
predicts about 0.520 s against 0.41 s and fails. These are regression observations
on the supplied illustrative reference data. They do not validate arbitrary
profiles under the same machine name or complete the seven-domain campaign.


### Shipped equilibrium configuration names

A filename or reactor label containing `Validated` does not establish a successful
validation. An unchanged run of `validation/iter_validated_config.json` on the
current Python SOR solver reached its 1,000-iteration limit with
`converged=false` and an interior GS residual RMS of approximately 2.6191.
Independent evaluation of the discrete operator on the returned flux/current
arrays reproduced that residual. The configuration uses permeability 1.0 and
current in configured units; these values cannot silently be interpreted as SI
or matched to a 15 MA preset by name. Its rectangular computational domain also
does not establish a measured plasma boundary or minor radius.

The solver normalises its current distribution toward the declared total.
Integrating that same distribution back to the target checks that normalisation;
it does not independently establish physical current conservation. A campaign
must retain the convergence failure, explicit unit convention, boundary/source
contract and separate comparison reference.


### Capturing an equilibrium execution

`EquilibriumExecution.evaluate(scratch_directory)` runs the existing
fixed-boundary solver on a temporary byte-identical copy of a SHA-256-bound
configuration. The reactor identity must match, and a positive GS RMS acceptance
threshold is supplied explicitly. Source files remain untouched; the temporary
copy is removed after execution.

A configuration digest binds bytes; it does not certify every numerical solver
control. The adapter refuses nonfinite returned fields/grids or independently
computed GS RMS with `RuntimeError`, before returning JSON. Actual runtime
regressions exercise malformed and extreme copied relaxation controls; a loose
stopping policy in those negative copies grants no admission and changes no
canonical solver policy. These tests retain the 1000-iteration budget and verify
source preservation and temporary-copy cleanup. Solver/parser errors propagate.

The returned JSON retains original and effective configuration, solver histories,
flux/current fields and R/Z grids. It independently applies the discrete GS
operator to the returned fields and compares the residual against the declared
bound. Numerical acceptance also requires the solver's own convergence result.
Nonconvergence remains a recorded failure. Units remain configuration-defined;
no SI conversion, plasma geometry or physical identity is inferred from names.
This execution does not yet compare an external reference equilibrium, and its
external-reference, action and facility flags remain false. Full seven-domain
integration remains unfinished.


### Status of the transport formulation

The subsections from here to the end of this page describe the transport
discretisation implemented in this repository: density-weighted
Crank–Nicolson diffusion, balanced internal exchange, conservative thermal
storage and the prescribed face stage. It is a candidate numerical
formulation. It has not been independently reviewed, and nothing on this page
is a claim of physical validity, of validation against reference data, or that
it is the reference transport mathematics.


### Transport energy gate and returned profiles

The transport energy check now runs after internal exchange, pedestal overrides and
final profile sanitisation. Previously, `enforce_conservation=True` could accept
a state whose temperatures were subsequently changed before return, while
`energy_balance_error` still described the earlier state. The corrected gate
checks the final thermal profiles with the existing 1% threshold. The subsequent
balanced-exchange correction removes the legacy per-channel rescaling. This does not
establish an independent physical energy-balance reference or validate all
transport approximations.


The initial energy also retains the pre-species density together with the initial
temperatures. Using the evolved density with old temperatures previously rewrote
the starting energy during a multi-ion step. The corrected accounting preserves
a consistent initial state. Historical before/after measurements predate the
balanced-exchange correction below; multi-ion rejection remains tested. This remains
an internal check under the current `ni = ne` energy convention, not validation of
a complete species-resolved physical energy model.


### Public thermal accounting snapshot

`TransportSolver.last_energy_balance` exposes an immutable `ThermalEnergyBalance`
after thermal assessment, including a step rejected by `enforce_conservation`.
It contains timestep in seconds, auxiliary power in MW, initial/final/source
energies in joules and the relative balance error. Consumers can check the
reported arithmetic without reading private solver attributes. A saved record
retains its values if profiles or the solver are subsequently changed.

Each `evolve_profiles` call first invalidates the previous record. A zero-time
call or an error before assessment leaves `None`, so an earlier record cannot
be mistaken for a measurement of that attempt. Nonfinite arithmetic may remain
in a diagnostic record and must not be admitted as finite evidence. The record
uses species-count ion capacity and electron density for its two channels;
it is not a complete species-resolved or independent plant energy balance.


### Balanced internal ion–electron exchange

Previously, electron equilibration had no opposite ion transfer and was counted
as an external source. A retained 50 kJ auxiliary-input case reported about
101.8 MJ of net sources. This was a model defect, not a physical validation.

The step now transfers equal and opposite energies after the separate
transport solves, using frozen-rate relaxation weighted by the respective ion
and electron heat capacities. Internal exchange contributes zero
to net external sources. The fixed edge temperatures remain boundary conditions;
pedestal overrides and sanitisation still precede the final energy check.
Per-channel zero-heating rescaling was removed because a recipient channel can
legitimately heat through internal transfer with no auxiliary input.

Runtime regressions exercise hot ions, hot electrons, equal initial temperatures
and small/stiff timesteps, checking core heat, bounded relaxation, source signs
and fixed edges. The inherited equilibration rate is not independently validated.
The common-ion-temperature closure, splitting accuracy and external physical
references remain open; numerical boundary power is accounted separately. These changes do not complete
seven-domain multi-machine validation or establish facility readiness.


### Radiation power and thermal heat capacity

Bremsstrahlung and multi-ion tungsten radiation are supplied in W/m³. Their
conversion to keV/s now divides by the channel heat capacity `3/2 n e_keV`,
consistent with the thermal energy used in the balance record. Previously the
conversion omitted `3/2`, so integrating the temperature sink removed 1.5 times
the declared radiative energy. The same correction applies to the evolved
profiles and the integrated source record. The existing half/half allocation
of multi-ion line radiation between the channels is retained.

Public runtime tests independently integrate the prescribed radiative powers
and compare them with the source record, both with and without auxiliary power,
in single-ion and multi-ion modes. They supplement core-temperature tests, so
correcting only the diagnostic cannot satisfy the checks. This fixes a units
conversion; it does not validate the radiation fits or their coefficients,
species-resolved heat capacities or the legacy single-ion impurity cooling
proxy, whose output is already treated as a temperature rate.

The thermal capacity convention is consistent with the heat-equation prefactor
in the [TORAX equation summary](https://torax.readthedocs.io/en/stable/equation_summary.html).
Full boundary, species and independent physical validation remain open.


### Boundary constraints in the implicit transport solve

The axis Neumann condition and fixed edge temperatures are now imposed in the
Crank–Nicolson linear systems before solving. Previously identity boundary rows
solved for source-updated boundary values, and those values were overwritten
afterwards. Neighbouring interior cells therefore used boundary values different
from the returned profiles.

Public-runtime regressions reconstruct face fluxes from initial and returned
profiles and check the interior Crank–Nicolson balance, including cells next to
both boundaries. Equal channel diffusivities make internal exchange cancel in
the sum equation. Four timestep/diffusivity combinations reproduce the original
failure and pass with the boundary rows enforced. This verifies a discrete
boundary contract. The numerical boundary-power record is described below;
physical boundary calibration and general splitting accuracy remain unfinished.
The existing final-state 1% energy gate has not been relaxed or corrected with
an inferred boundary loss.


### Density-weighted thermal diffusion

Transport now uses the conservative operator
`L(T) = (1/a²)/(rho n) d/drho(rho n chi dT/drho)`, with arithmetic face averages
of `n chi` and cell density in the heat-capacity denominator. The explicit and
implicit halves use the same density frozen after species evolution. Previously
both operators diffused temperature with unit capacity while the energy record
weighted temperatures by density; internal face transfers could then create or
remove total heat when density varied.

Public-runtime tests independently compare the interior energy change with the
two boundary face powers and radiative losses for rising and falling density
profiles. Equal channel diffusivities let the internal exchange cancel in that
comparison. A separate closed-boundary public CN solve checks total heat
conservation with variable diffusivity and density. These checks do not replace
independent validation of species evolution or unequal-channel splitting.

The generic radial operators accept an optional positive `density` vector.
Omitting it retains unit-capacity diffusion for callers solving that equation;
the thermal transport caller always supplies its density. Nonfinite, nonpositive,
complex, Boolean and wrongly shaped density arrays are rejected. A uniform
change in density units leaves the temperature operator unchanged. Here `chi`
denotes thermal diffusivity; no new physical calibration of transport closures
is claimed. The boundary-power record below uses this operator; multispecies
heat capacities and physical boundary validation remain open.


### Boundary terms in the public thermal energy record

`ThermalEnergyBalance` now reports two signed boundary contributions in joules:

- `diffusive_boundary_energy_j`: net heat entering the evolved interior cells
  through their inner and outer faces. It integrates density-weighted face
  conductivities and the old/new Crank–Nicolson temperature midpoint, separately
  for each channel, before internal exchange or pedestal overrides.
- `prescribed_edge_energy_j`: the energy needed to force the stored edge node
  to its fixed temperature, minus the volumetric source that the fixed boundary
  row does not evolve. This term follows the current nodal-volume convention;
  it is not an independently measured separatrix power.

Positive values add heat. `source_energy_j` still contains the original all-cell
volumetric source integral. The relative error subtracts that source and both
boundary terms from the final-minus-initial energy, with the same denominator
and 1% admission threshold. Boundary terms are calculated from the actual
operator and its imposed edge condition, not fitted to the observed residual.
Previously, physically transported heat could incorrectly trip the source-only
gate. Unmodelled pedestal energy, species changes and numerical recovery remain
visible in the residual; adding boundary terms does not admit them automatically.

Runtime regressions cover inward/outward heat transport, rising/falling density,
initial edge mismatch and both enforcement modes. Independent dense CN solves
check the flux record with unequal, spatially varying channel diffusivities.
A real pedestal override still fails admission when its energy is unaccounted.
These checks establish discrete accounting under `ni = ne`, not complete plasma,
multispecies, nonlinear splitting or facility validation.


### Physical geometry in species diffusion

D/T/He diffusion now uses the cylindrical operator
`(1/a^2)/rho * d/drho(rho * D * dn/drho)`, with diffusivity in m²/s and
physical minor radius in metres. The explicit stability step scales with
`(a * drho)^2`; zero diffusivity uses a single source step. The species API
requires `rho` and `a_minor`, and the integrated solver passes its actual grid
and radius. Geometry-free callers must migrate to these explicit arguments.
Invalid grids, radii, diffusivities and timesteps raise `ValueError`.

Public runtime tests compare parabolic-density evolution with the analytic
cylindrical operator across three physical radii. Small-radius cases check the
density maximum principle under diffusion and nonnegative burn losses. These
verify numerical geometry, not a complete multispecies plasma model. Particle
diagnostics account for signed boundary exchange as described below; numerical
clipping remains an error and thermal source coupling remains incomplete.


### Species-count thermal capacity

The multi-ion thermal solver now uses `ni = n_D + n_T + n_He + n_impurity`
for ion heat capacity, assuming all ions share `Ti`. Electron heat capacity
still uses `ne`; charge multiplicity does not create additional thermal ions.
The public `ion_density` property returns a separate array. Single-ion mode
retains `ni = ne`.

Stored energy, confinement time, pressure mapping, ion heating/radiation
conversion, density-weighted diffusion and boundary power use these channel
capacities consistently. Frozen-rate exchange preserves `ni*Ti + ne*Te` with
unequal capacities; its inherited relaxation-time coefficient remains
uncalibrated for a species-resolved collision closure. External source energy
still excludes internal exchange, and the 1 percent admission threshold is
unchanged. Changes in species populations are not automatically counted as
external heat; their transport and reaction energy remain an open
coupling contract. The explicit volumetric pumping heat model is described below. This is a thermal-capacity correction, not full multispecies
energy conservation or validated fusion self-heating.


### Integrated fusion consumption and helium pumping

Each species substep now applies diffusion, an analytic D-T burn update at
fixed incoming temperature, then exponential helium pumping. Burn preserves
`n_D - n_T`, consumes equal D/T counts and produces one He nucleus per reaction.
It cannot consume more than the available limiting fuel. Pumping removes only
available helium; positive infinite `tau_He` disables it.

`SpeciesEvolutionResult.fusion_reactions` and `helium_pumped` contain the
integrated local counts in units of 10^19 m^-3 before imposed boundaries.
`S_He` now means integrated production divided by the timestep, rather than the
incoming instantaneous production rate. At zero timestep the three input
species profiles remain unchanged and all integrated source diagnostics are
zero. Callers relying on the previous instantaneous meaning must migrate.
The particle residual uses actual integrated reaction and pumping counts, plus
the independently computed boundary contributions described below.

Positive fuel floors were removed from the species kernel and thermal runtime;
a prescribed D/T recycling density of 0.01 remains at the edge. Thermal evolution
rejects a cell with zero ion heat capacity. Invalid negative/nonfinite species
inputs and invalid pumping times are rejected by the species API.

Separate source operators are exact at fixed temperature, but their combined
splitting is first-order. Repeated public calls demonstrate convergence against
an independently integrated coupled burn/pumping reference. This does not
calibrate the inherited fusion-reactivity fit or couple reaction and
particle transport energy to the thermal solver. Pumping heat is specified below. Fusion self-heating and the
full seven-domain machine validation remain incomplete.


### Species boundary inventory

`SpeciesEvolutionResult.particle_balance` is a frozen scalar
`SpeciesParticleBalance` record. Its initial/final counts measure D, T and He
nuclei, rather than charge or density. One fusion reaction decreases this total
by one; pumping removes the integrated helium count. Impurity inventory is not
part of this D/T/He diagnostic.

`diffusive_boundary_count` integrates signed flux through the two faces bounding
the evolved interior. `prescribed_boundary_count` measures the signed change
from imposing the edge recycling densities after each substep. Positive values
enter the domain. Both are computed from the applied operators, independently
of the final inventory residual. The error is the absolute value of
`final - initial + fusion + pumped - diffusive_boundary - prescribed_boundary`,
divided by the initial count (denominator floor 1e-10).
`numerical_correction_count` reports particles introduced by diffusion clipping;
it is deliberately not subtracted from that error.

The supported quadrature requires `dV = C * rho * drho`, with finite positive
`C`, zero axis weight, and positive interior and edge weights. Inconsistent
weights raise `ValueError`; this diagnostic does not support arbitrary geometry.

`TransportSolver.last_particle_balance` exposes the completed species
stage. Every thermal evolution attempt resets it before validating the timestep.
Single-ion evolution and zero timesteps leave no record. A subsequent thermal
admission failure preserves the completed species record for inspection; later
thermal sanitisation or external profile mutation does not rewrite it. It is
not a certificate for the final thermal-state inventory.

Public tests check signed inward/outward parabolic flux against an analytic
reference, imposed-edge counts, combined diffusion/burn/pumping closure across
physical radii and substeps, rejected volume weights, and record lifetime after
real thermal rejection. Particle closure does not establish thermal coupling,
calibrated fusion rates, or full machine validation.


### Quasineutral electron capacity at low density

The multi-ion species result now retains ne = n_D + n_T + 2*n_He + 10*n_impurity
without adding a minimum electron population. Runtime electron sanitisation no
longer imposes the single-ion density floor or ceiling in multi-ion mode.
The prescribed fuel edge therefore has electron density 0.02 in units of
10^19 m^-3 when no impurity is present, rather than the former artificial 0.1.
A vacuum species result may have zero electrons; thermal evolution requires
positive ion and electron capacities.

Auxiliary deposition rates are converted from the helper's floored reference
capacity to the actual electron capacity before ion-capacity conversion.
Electron radiation rates likewise use the actual positive capacity. Dilute
public runtime tests reconstruct the initial stored energy and external source
power from independent volume and radiation integrals. The inherited exchange
time regularisation remains unchanged and uncalibrated. These changes remove
an artificial charge/energy contribution; they do not supply the still-missing
thermal energy transport or reaction closure; pumping heat is specified below.


### Pressure consistency in the opt-in bootstrap approximation

The legacy bootstrap approximation now differentiates the same species-count
pressure used by thermal storage and equilibrium profile mapping:
P = (ni*Ti + ne*Te) * 10^19 * e_keV, with the exact keV-to-joule conversion.
The previous ne*(Ti+Te) expression overcounted helium ion pressure. Public
bootstrap calls with analytic parabolic profiles check zero, moderate and high
helium fractions against their pressure gradients.

This correction applies to the explicitly enabled legacy approximation. Its
trapped-particle coefficient, current-sign convention and fixed effective-charge
correction remain inherited and uncalibrated. It does not validate the default
Sauter closure, the current-conservation domain, or particle-energy coupling.


### Conservative thermal storage with evolving density

The thermal Crank–Nicolson step now discretises the change of n*T. For each
channel its storage is n_new*T_new - n_old*T_old; the explicit conductive flux
uses n_old and T_old, and the implicit conductive flux uses n_new and T_new.
Volumetric heat sources retain the current channel-capacity normalisation.
The thermal boundary record averages the corresponding old and new face fluxes,
and prescribed-edge energy uses both old and new edge capacities.

Previously, changing density at frozen temperature created or removed thermal
energy even when particle boundary flux, reactions and pumping were all zero.
Six public runtime cases now close that energy balance, with rising, falling
and uniform temperature and two timesteps. Independent dense matrix solutions
use separate old/new storage and conductivity matrices to check local channel
energy and boundary records. Single-ion fixed-density behaviour is retained.

This corrects storage for the existing heat equation with conductive transport
and its declared sources. No separate particle-associated heat convection,
pressure-work closure or fusion-product heating has been inferred from density
changes. Without a specified heat source, changing population alone does not
remove modelled channel energy: temperatures respond to the changed capacity.
The ash-removal source is specified separately below. Discrete energy closure
does not validate a physical source model. The remaining source/flux closures and
coupled timestep convergence remain required for full multispecies validation.
Numerical recovery and unmodeled pedestal changes remain visible to the
unchanged 1 percent admission threshold.


### Thermal energy carried by helium pumping

The distributed ash sink now uses a velocity-independent removal model: removed
He nuclei carry their local mean ion thermal energy, with two accompanying
electrons carrying their mean electron thermal energy. For the integrated local
removed density p (in 10^19 m^-3), the step removes
1.5*10^19*e_keV*p*(Ti_old + 2*Te_old) joules per cubic metre. The ion and electron
parts enter their own channel equations using the completed positive capacities.
Counts come from the actual substepped species kernel, including ash produced
before pumping. Temperatures are frozen at entry to the outer step; combined
thermal/source accuracy remains limited by that splitting.

ThermalEnergyBalance.helium_pumping_energy_j records this signed contribution.
It is already included in source_energy_j and must not be summed again. The
prescribed thermal edge row cancels ignored volumetric sources as before. Zero
timesteps clear the step record; disabled pumping and single-ion mode contribute
zero pumping heat. Independent no-fusion exponential-removal tests reconstruct
the count-weighted ion/electron energy integral. A separate dense thermal oracle
uses public species counts to check local deposition and channel profiles.

This is a volumetric thermal-ash model, not a flowing-plasma enthalpy or sheath
model. It assumes He is thermalised at the common ion temperature and electron
removal maintains quasineutrality; energy-selective pumping, fast-alpha dynamics,
pressure work and boundary exhaust temperatures are not represented. Fusion
product heating, particle-associated heat convection and full coupled convergence
remain open.


### Coupled local pumping, exchange and radiation convergence

A public runtime refinement test compares the uniform core against independently
integrated two-temperature ODEs with exact exponential helium survival. The
reference includes the inherited exchange-time coefficient, the runtime's
arithmetic effective charge (including its prescribed edge node), and
bremsstrahlung. With local mean-energy pumping, the removal terms cancel the
corresponding density derivatives in the temperature equations.

At 64, 128 and 256 steps over 0.03 seconds, the maximum temperature errors are
approximately 1.342e-4, 6.384e-5 and 3.160e-5 keV for initially hotter ions, and
8.657e-4, 4.304e-4 and 2.146e-4 keV for initially hotter electrons. Both sequences
show first-order refinement while particle survival and per-step discrete
energy remain near numerical precision.

This checks temporal convergence of the specified local thermal-ash model.
Particle diffusion is disabled and the sampled uniform core is outside the
thermal boundary layer. It does not establish spatial convergence, reactive
D-T evolution, particle heat convection, or calibration of the inherited
exchange and radiation coefficients. Those broader coupled checks remain open.


### Preserve resolved gyrokinetic transport channels

The transport update now retains separate electron thermal diffusivity and
D_n outputs from the quasilinear gyrokinetic, native TGLF and external GK paths.
The ion channel retains its configured Chang-Hinton contribution. Previously,
the final shared assignment overwrote both thermal channels with the ion-based
estimate and replaced the computed particle coefficient by 0.1*chi_e.

Resolved turbulent channels no longer receive the additional legacy
critical-gradient estimate. Gyro-Bohm/explicit constant-fallback modes retain
that shared-channel prescription; existing H-mode pedestal profile handling
remains separate. Real built-in quasilinear model comparisons at two electron/
ion temperature ratios verify retained channel outputs through the public
transport update. This is integration correctness, not gyrokinetic calibration.

Native/external nested fallback is cellwise and uses the same three-channel
composition: chi_gB = max(raw_gyro_bohm, 0.01), then chi_i = chi_gB +
chi_nc, chi_e = chi_gB, D_n = 0.1*chi_gB on each failing non-core cell.
It does not restore the direct legacy mode's additional critical-gradient
term. The core/vacuum driver floor is 0.01 for each returned channel, before
adding the ion neoclassical contribution. Both explicit fallback permissions
are required. External execution errors are eligible for this fallback;
native exceptions propagate, while invalid/unconverged native output can
fall back. A selected transport mode is not evidence of actual backend success.

Public update tests compare a real native solve at rho=0.5 against independently
specified local gradients at two electron/ion temperature ratios. Additional
public tests use actual invalid native output and an actual absent external
binary to verify refusal and permitted fallback composition. Configured real
solver instances are supplied through the existing cached-solver attributes;
no provider outputs are mocked. Malformed-config and missing-binary cases are
routing robustness evidence. External successful execution and flux conversion
remain unqualified; these tests do not establish physical calibration.

A real GACODE standalone run succeeds independently of this adapter. Its
standard output is out.tglf.gbflux: signed particle, energy, momentum and
exchange fluxes in gyro-Bohm normalisation, ordered with electrons first.
These are not the chi_i, chi_e and d_e coefficients expected by the adapter's
out.tglf.transport parser. A reference run exhibits inward particle transport;
clipping that signed flux or labelling it a positive diffusivity would change
its physical meaning. A flux-aware coupling and normalisation contract is
required before the external path can be admitted.

The [standalone signed-flux API](tglf_flux.md) now executes actual GACODE
decks in fresh directories and preserves normalised signed fluxes and
multi-mode spectra with hashed execution evidence. It does not infer
diffusivities or supply the outstanding conservative runtime coupling.

The current adapter also emits a Fortran namelist where the standalone parser
requires key=value records, supplies unsupported SHAT/ALPHA_MHD keys, and uses
initialisation-only -i with a file path instead of executing a simulation
directory. Real eigenvalue output has separate ky and multi-mode spectra,
not the three-column table currently assumed. Correcting executable discovery
alone therefore cannot qualify this external path.

D_n is still distinct from the configured D_species used by the D/T/He kernel.
Mapping an electron-particle coefficient to species diffusion and deriving a
particle-associated heat flux require an explicit closure; preserving the
computed coefficient does not establish either mapping. No such closure is
silently inferred by this change.


### Radially varying species diffusivity

D_species accepts a finite nonnegative scalar or a real vector matching rho,
within the finite float64 working range. Wider floating-point inputs above
that range are rejected before conversion, face fluxes or CFL arithmetic.
The common D/T/He operator is (1/a²)/rho * d/drho(rho*D(rho)*dn/drho), using
arithmetic averages of D at cell faces. Its signed boundary inventory uses the
same face coefficients. The explicit physical CFL bound is
0.4*(a*drho)²/max(D); an all-zero profile retains the single source substep.
Scalar input remains supported and constant-profile trajectories are identical.

Manufactured quadratic density/diffusivity cases independently check the
cylindrical increment for rising/falling D at two radii, including the finite
spacing term from arithmetic face averaging. A sharp alternating coefficient
checks the maximum principle and zero numerical clipping across many substeps.
Malformed, negative, nonfinite, boolean and complex profiles are rejected.
A public thermal/species call verifies profile routing and input preservation.

All three fuel/ash species still share this configured coefficient. D_n is not
automatically mapped into it: species-dependent transport, impurity motion,
ambipolar closure and particle-associated heat convection remain separate
physical requirements. This change supports spatial coefficients without
claiming those model choices or calibrated machine transport.


### Conservative prescribed face transport

The [signed face-flux stage](transport_flux.md) advances explicit electron,
D/T/He particle number and electron/total-ion thermal energy through the
public TransportSolver.evolve_fluxes method. Local updates and boundary
ledgers use identical physical face areas and fluxes. Nonambipolar or
nonphysical steps reject without profile mutation. This supplies conservative
coupling for prescribed physical fluxes, including values from the explicit
TGLF SI conversion. Automatic spatial provider sampling, normalisation and
species mapping remain unqualified; source/operator composition is explicit.
