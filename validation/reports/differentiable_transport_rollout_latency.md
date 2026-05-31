<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Differentiable transport rollout latency benchmark report -->

# Differentiable Transport Rollout Gradient-Latency Benchmark

This report measures the local audited multi-step source-rollout
gradient-admission path for controller-tuning studies. It is not
a real-time control-loop guarantee.

- Backend: `jax`
- dtype: `float64`
- Radial points: `21`
- Rollout steps: `4`
- Timed runs: `5`
- Audit passed: `True`
- P50 latency [ms]: `1439.271255`
- P95 latency [ms]: `1922.004129`
- Max latency [ms]: `1965.289422`
- Claim boundary: `local audited rollout source-gradient latency only; not a real-time control-loop guarantee`
