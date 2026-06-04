<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Geometry-neutral replay AER admission report. -->
# Geometry-Neutral Stellarator Replay

- Schema: `scpn-control.geometry-neutral-replay.v1.1`
- Deterministic replay: `True`
- Replay signature: `f01e70bab912662f`
- Threshold pass: `YES`

## Metrics

- Initial field-line spread: `0.074231`
- Final field-line spread: `0.024924`
- Improvement fraction: `0.664240`
- Max absolute current: `1200.000 A`
- P95 latency: `138.120 us`

## AER Admission

- Decode strategy: `rate`
- Decode window: `100 ns`
- Feature count: `4`
- Monotonic input: `True`
- Out-of-order events: `0`
- Overflowed: `False`
- Strict monotonic required: `True`

## Limitations

- This compact replay is not a production PCS.
- No external company data is used.
- The replay validates geometry-neutral SCPN control plumbing and actuator constraints.
