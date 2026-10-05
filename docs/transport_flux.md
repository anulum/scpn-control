# Conservative signed face transport

`TransportSolver.evolve_fluxes(dt, flux)` advances multi-ion density and
thermal energy from explicit physical minor-radius fluxes. Particle rows are
electron, D, T and He; energy rows are electron and total ions. Columns are the
`nr-1` faces between radial nodes. Positive flux points outward.

This stage belongs to a candidate numerical formulation; see the
[status of the transport formulation](validation_deficiencies.md#status-of-the-transport-formulation).
It carries no claim of physical validity, validation or adoption.

The same face areas and fluxes determine the local divergence and integrated
boundary exchange. Electron particle flux must equal D + T + 2 He; tungsten
is static with charge ten. The edge node is a fixed reservoir. The first and
last faces exchange with the evolved interior, and the zero-volume axis uses
the adjacent ion profiles. These are explicit reservoir boundaries.

The step advances conserved particle number and thermal energy before
recovering temperatures with the new densities. It rejects nonphysical
candidates without clipping or changing profiles. `last_flux_balance`
records the completed flux stage and resets before each flux attempt.

The caller supplies the physical species mapping and complete thermal energy
moment. This stage does not add particle-associated energy, diffusion, fusion,
pumping, radiation or ion/electron exchange. Compose these separately without
double counting. Prescribed fluxes are frozen, giving a first-order time step;
rejection may require a smaller step or a corrected physical prescription.

The [TGLF SI conversion](tglf_flux.md) supplies signed physical flux values,
but spatial sampling, reference matching and provider-species mapping remain
explicit caller responsibilities. This API does not establish automatic
external-GK admission or calibrated machine transport.

See the [transport flux API][scpn_control.core.transport_flux] for the geometry,
state, face-flux and conservation-result records and their callable contracts.

Finite inputs can still overflow intermediate arithmetic. The public step
converts overflow and invalid arithmetic to `ValueError` before any caller
profile is changed, and rejects geometry whose cell volumes or face areas
collapse to zero. NumPy error settings are restored after the call. These
checks do not establish accuracy or convergence for extreme parameter regimes.

Final electron density must satisfy the same relative quasineutrality tolerance
as the initial state. Near depletion can amplify an initially tolerated charge
difference; such a proposed state is rejected before solver profiles change.
Electrons are not overwritten to hide a failed interior charge check.

## Explicit flux-surface metrics

`TransportFaceGeometry` supplies physical cell volumes and face `dV/dr`.
Both `advance_face_flux` and `TransportSolver.evolve_fluxes` accept it through
`geometry=`. The circular-torus default remains available. Other solver
operators are not automatically converted to a supplied metric.

TGLF returns the radial flux-surface moment whose divergence is
`(1/V') d(V' F_r)/dr`. Its integrated rate uses `V'=dV/dr`, which generally
differs from geometric area for shaped surfaces. `miller_volume_metric`
computes enclosed volume and this derivative from the same physical Miller
shape and radial derivatives used by the input builder. Interior cell volumes
are differences of enclosed volumes at neighbouring faces; axis and fixed-edge
reservoir weights may be zero. Interior weights and every face metric must be
positive and finite, and the copied axis must have zero weight.

The Miller calculation integrates Green's theorem and its radial derivative,
checks sampled positive Jacobians and requires successive Gauss-Legendre
estimates to agree. A positive sampled Jacobian is not a global nesting or
force-balance proof. Independent elliptical and Bessel-volume references,
radial finite differences and a manufactured transport refinement check
qualify the numerical metric. A real four-face TGLF regression advances
analytic prescribed profiles using distinct face inputs and these weights,
then refreshes the provider inputs from the updated positive node profiles for
two further frozen-flux stages. All twelve executions retain distinct run
directories, matching copied-deck digests and validated output receipts.
Adjacent stage inventories and accumulated reservoir exchange are checked.
Shape, magnetic reference and collisionless electron/ion composition remain
prescribed; the case supplies no general production orchestration, coupled
temporal-convergence result or machine calibration.
