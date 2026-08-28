# Competitive Evidence — SCPN Control and Adjacent Fusion Systems

> **Evidence date:** 2026-08-28. **SCPN Control version:** v0.23.0.
> The machine-readable source registry is
> [`docs/_data/competitive_evidence.json`](_data/competitive_evidence.json).

This page compares documented scope and evidence. It is not a product ranking
or a claim that every relevant system has been surveyed. A blank or undocumented
capability is reported as **not assessed**, never converted into “does not
exist.”

## Reading the comparison

Each external row is bound to an exact public release or a primary paper.
Project documentation supports capability statements; it does not by itself
establish accuracy, deployment readiness, or equivalence to another code.

No cross-project numeric result is admitted in this revision. A quantitative
comparison requires the same problem, inputs, precision, tolerances, convergence
criteria, warm-up and compilation treatment, sample definition, isolated
hardware, and failure accounting. The previous page mixed local loopback,
bare-kernel, solver-step, reconstruction-iteration, and full control-cycle
timings; those values answer different questions and are not retained as a
ranking.

## Assessed systems and strongest documented evidence

| System and assessed artifact | Primary role | Strongest evidence relevant to this comparison | SCPN Control boundary |
|---|---|---|---|
| **SCPN Control v0.23.0 snapshot f1f6e36d** | Neuro-symbolic plasma-control research toolkit and evidence-admission layer | Executable classical, robust, predictive and neuro-symbolic control surfaces; formal/admission contracts; Python, Rust, TypeScript, Lean and C++ interfaces | Research software: not a commissioned PCS; external-code parity, target-hardware timing and facility validation are separate fail-closed claims |
| **TORAX v1.4.3** | Differentiable JAX core-transport simulator | Coupled heat, particle and current-diffusion PDEs; differentiable nonlinear solves; QLKNN, direct TGLF and IMAS core-source integration | Reduced transport and TORAX-shaped interfaces are not an identical-input TORAX result |
| **FUSE v1.2.0** | Integrated fusion-device simulation and design | Plasma physics, engineering, control, balance-of-plant and costing actors; constrained multi-objective optimisation; IMAS/OMAS interoperability | SCPN Control supplies control and evidence components, not an equivalent whole-device design environment |
| **FreeGS v0.8.2** | Static free-boundary Grad–Shafranov equilibrium | Coils, plasma profiles, machine circuits, inverse constraints and G-EQDSK in a transparent Python reference | Code-to-code agreement is useful verification, not experimental truth |
| **FreeGSNKE v3.0.1** | Static and evolutive free-boundary equilibrium and circuits | Fourth-order Newton–Krylov solves; active/passive structures and magnetic diagnostics; MAST-U/EFIT++ validation literature | Equivalent evolutive and shot-matched validation is not currently admitted in SCPN Control |
| **DREAM v26.5** | Fluid-kinetic disruption and runaway-electron simulation | Coupled nonlinear fluid/kinetic equations with specialised runaway and shattered-pellet-injection physics | SCPN Control's disruption and mitigation models are control-grade models, not a DREAM-equivalent kinetic solver |
| **PROCESS v3.4.2** | Reactor systems design | Integrated reactor physics, engineering, costing and optimisation with versioned/Zenodo release lineage | SCPN Control is not a reactor design or costing system |
| **OMAS v0.97.3** | ITER IMAS data-model interoperability | Multiple file/database formats mapped to the IMAS model and used as an OMFIT integration substrate | IMAS-facing contracts do not yet demonstrate comparable ecosystem breadth |
| **TCV deep-RL magnetic control, doi:10.1038/s41586-021-04301-9** | Learned magnetic control demonstrated on a tokamak | Peer-reviewed 10 kHz closed-loop TCV experiments, zero-shot simulation-to-hardware transfer and 19-coil actuation across several plasma configurations | SCPN Control has no commissioned tokamak actuation result; simulation and offline replay are different evidence classes |
| **EFIT-AI GPU reconstruction, doi:10.1145/3624062.3624607** | Performance-portable inverse equilibrium reconstruction | Named NVIDIA, AMD and Intel accelerator evaluation of GPU-offloaded EFIT reconstruction kernels | No admitted identical-input, full-output, named-GPU comparison exists in SCPN Control |

## Where external evidence is currently stronger

These are evidence differences, not statements about the worth of either
project.

- **Facility operation:** the TCV deep-RL work reports actual tokamak actuation;
  SCPN Control reports simulation, local runtime and offline evidence only.
- **Evolutive free-boundary validation:** FreeGSNKE documents self-consistent
  circuit/plasma evolution and peer-reviewed MAST-U/EFIT++ validation. SCPN
  Control does not currently admit equivalent evidence.
- **Core-transport depth and integration:** TORAX documents a dedicated,
  differentiable transport stack with direct TGLF and expanding IMAS support.
- **Integrated design breadth:** FUSE and PROCESS cover whole-device engineering,
  costing and optimisation scopes outside SCPN Control's role.
- **Disruption/runaway fidelity:** DREAM is purpose-built around fluid-kinetic
  runaway-electron physics and has a dedicated validation/publication lineage.
- **Data and workflow adoption:** OMAS/OMFIT expose a broader public
  interoperability and module ecosystem than SCPN Control currently
  demonstrates.
- **Accelerator reconstruction evidence:** the EFIT-AI study reports named
  multi-vendor GPU measurements. SCPN Control's equilibrium timing evidence
  does not satisfy that matched boundary.

## What SCPN Control demonstrates today

Within this assessed set, the sources document a different emphasis for SCPN
Control: neuro-symbolic controller construction, executable safety/admission
contracts, formal traceability, polyglot runtime surfaces, and evidence objects
that keep simulation, external-code, hardware and facility claims separate.
That observation is limited to the sources below and is not a market-wide
novelty claim.

Current package counts are generated rather than duplicated here. See the
[capability manifest](_generated/capability_snapshot.md). Scientific and
deployment limits are maintained in [production readiness](production_readiness.md)
and [validation deficiencies](validation_deficiencies.md).

## Quantitative comparison status

The admitted cross-project quantitative comparison set is currently empty.
Repository-local Python/Rust or controller-baseline comparisons remain useful
for implementation decisions when their own provenance is complete, but they do
not rank SCPN Control against TORAX, FUSE, FreeGSNKE, DREAM, TCV control or
EFIT-AI.

When a matched external case becomes available, the public result must preserve
all runs, including failures, and report the exact code versions, inputs,
hardware/load, warm-up, precision, tolerances, samples and admission decision.

## Primary sources

- SCPN Control v0.23.0 snapshot `f1f6e36d05f259f81c7b6c33d0d1d0089c921635`: [production-readiness contract](https://github.com/anulum/scpn-control/blob/f1f6e36d05f259f81c7b6c33d0d1d0089c921635/docs/production_readiness.md) and [generated capability manifest](https://github.com/anulum/scpn-control/blob/f1f6e36d05f259f81c7b6c33d0d1d0089c921635/docs/_generated/capability_manifest.json).
- TORAX v1.4.3 (`4aea2377385ba4dfe37b0ef4396374162af1314b`): [release](https://github.com/google-deepmind/torax/releases/tag/v1.4.3) and [tagged overview](https://github.com/google-deepmind/torax/blob/v1.4.3/README.md).
- FUSE v1.2.0 (`9ef2f99af73497706a097d99a2aaac2f08405370`): [release](https://github.com/ProjectTorreyPines/FUSE.jl/releases/tag/v1.2.0) and [tagged overview](https://github.com/ProjectTorreyPines/FUSE.jl/blob/v1.2.0/README.md).
- FreeGS v0.8.2 (`8b838c1df162ca770a6937ac79d8c73e8f10a53b`): [release](https://github.com/freegs-plasma/freegs/releases/tag/v0.8.2) and [tagged overview](https://github.com/freegs-plasma/freegs/blob/v0.8.2/README.md).
- FreeGSNKE v3.0.1 (`f776e908c8c333411f9824cbcfed674fafff8dfd`): [release](https://github.com/FusionComputingLab/freegsnke/releases/tag/v3.0.1), [tagged overview](https://github.com/FusionComputingLab/freegsnke/blob/v3.0.1/README.md), and [MAST-U/EFIT++ validation paper](https://doi.org/10.1088/1402-4896/ada192).
- DREAM v26.5 (`ecdd5e146537c77602c9d7cc76b36100200e4b9a`): [release](https://github.com/chalmersplasmatheory/DREAM/releases/tag/v26.5), [tagged overview](https://github.com/chalmersplasmatheory/DREAM/blob/v26.5/README.md), and [framework paper](https://doi.org/10.1016/j.cpc.2021.108098).
- PROCESS v3.4.2 (`c0ae5b28649f2b20fb7efc7904628b6defe4151c`): [release](https://github.com/ukaea/PROCESS/releases/tag/v3.4.2) and [tagged overview](https://github.com/ukaea/PROCESS/blob/v3.4.2/README.md).
- OMAS v0.97.3 (`e95c785e8c4c461adb66cc130e16b8950139b103`): [release](https://github.com/gafusion/omas/releases/tag/v0.97.3), [tagged overview](https://github.com/gafusion/omas/blob/v0.97.3/README.md), and [OMFIT public module catalogue](https://www.omfit.io/modules.html).
- TCV deep-RL magnetic control: [Nature article and source data](https://doi.org/10.1038/s41586-021-04301-9).
- EFIT-AI GPU reconstruction: [SC-W 2023 paper](https://doi.org/10.1145/3624062.3624607).
