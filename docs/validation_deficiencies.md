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
