<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Differentiable transport latency benchmark report -->

# Differentiable Transport Gradient-Latency Benchmark

This report measures the local audited gradient-admission path for
controller-tuning studies. It is not a real-time control-loop guarantee.

- Backend: `jax`
- dtype: `float64`
- Radial points: `21`
- Timed runs: `5`
- Audit passed: `True`
- P50 latency [ms]: `30.768817`
- P95 latency [ms]: `37.788684`
- Max latency [ms]: `38.426561`
- Claim boundary: `local audited gradient-admission latency only; not a real-time control-loop guarantee`
