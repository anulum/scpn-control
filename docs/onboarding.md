<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Control — Onboarding guide
-->

# Onboarding

This guide explains how to approach SCPN Control without needing to read the
entire codebase first.

## Mental model

Think of SCPN Control as three layers:

| Layer | What it answers |
| --- | --- |
| Controller logic | What should the controller do, and what invariants must never be violated? |
| Physics and replay facades | Which equilibrium, transport, disruption, and digital-twin signals feed the controller? |
| Admission evidence | Which artefacts prove that a result is reproducible and allowed to be claimed? |

A new workflow should normally touch all three layers. A controller without an
evidence gate is not ready for promotion. A physics result without a controller
contract is not yet a control feature. A benchmark without hardware and claim
metadata is only a local timing observation.

## Install paths

```bash
pip install scpn-control
pip install "scpn-control[ws,formal,jax]"
```

Use editable install for development:

```bash
git clone https://github.com/anulum/scpn-control.git
cd scpn-control
pip install -e ".[dev,docs,formal,jax,ws]"
```

## First hour checklist

1. Run `scpn-control demo --steps 1000` to confirm the package is importable.
2. Read [Production Readiness](production_readiness.md) before interpreting any
   benchmark or validation result.
3. Run one tutorial from [Tutorials](tutorials.md) and one notebook from
   [Notebook Gallery](notebooks.md).
4. Inspect [API Reference](api.md) for the surface you want to use.
5. If you want to make a claim, find the matching validator in
   [Validation and QA](validation.md) before writing code.

## Choosing a workflow

| Goal | Start here |
| --- | --- |
| Learn the controller API | `examples/tutorial_01_closed_loop_control.py` |
| Explore differentiable physics | `examples/tutorial_02_jax_autodiff.py` |
| Evaluate safety certificates | `scpn_control.scpn.formal_verification` in [API Reference](api.md) |
| Build a replay or validation report | [Validation and QA](validation.md) |
| Understand deployment limits | [Production Readiness](production_readiness.md) |
| Discuss funding or collaboration | [Compute Validation Funding](compute_validation_financing.md) |

## What not to assume

- A fast local benchmark is not a full control-cycle guarantee.
- A bounded proof is not facility certification.
- A public dataset conversion is not predictive EFIT or P-EFIT validation until
  the strict admission gate passes.
- A simulator bridge is not a TRANSP, TSC, CODAC, or EPICS acceptance test.
- A neural surrogate is not a substitute for external-code or measured-shot
  evidence unless the corresponding validator admits that claim.
