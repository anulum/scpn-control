<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: Geometry-neutral replay schema and pulsed-shot metadata guide. -->

# Geometry-Neutral Replay

Geometry-neutral replay is the SCPN Control evidence surface for replaying a
bounded controller scenario without tying the report to one tokamak,
stellarator, or pulsed FRC geometry. It records the scenario, trace, metrics,
thresholds, manifest digests, and limitations needed to admit a replay as a
bounded repository artefact.

The report is useful for three jobs:

1. Compare a controller policy across magnetic-configuration contracts without
   rewriting the replay harness.
2. Bind acceptance metrics and trace digests into a deterministic manifest.
3. Preserve claim boundaries so synthetic replay evidence is not mistaken for a
   plant, HIL, or measured-shot admission.

## Schema versions

`scpn-control.geometry-neutral-replay.v1` is the original replay schema. It
contains the deterministic replay trace, scenario declaration, geometry
contract, acceptance metrics, thresholds, limitations, and manifest digests.

`scpn-control.geometry-neutral-replay.v1.1` extends the same base report with
optional pulsed-shot context fields. A v1 report remains loadable while the v1.1
schema bundle is installed, so existing replay artefacts do not need migration.

The v1.1 optional fields are:

| Field | Type | Validation |
| --- | --- | --- |
| `pulse_id` | UUID string | Must parse as a canonical UUID. |
| `capacitor_state_initial_J` | number | Must be finite and non-negative. |
| `trigger_timestamp_ns` | integer | Must be non-negative. |
| `energy_recovered_J` | number | Must be finite and non-negative. |
| `shot_phase_log` | list of objects | Entries must be sorted by `t`; each row needs finite non-negative `t`, non-empty `state`, and non-empty `reason`. |
| `frc_diagnostics.s_parameter_at_burn` | number | Must be finite and non-negative when present. |
| `frc_diagnostics.mrti_peak_amplitude_m` | number | Must be finite and non-negative when present. |
| `frc_diagnostics.tilt_growth_rate_s_inv` | number | Must be finite when present. |

## Python API

```python
from scpn_control.scpn.geometry_neutral_replay import (
    SCHEMA_VERSION_V1_1,
    assert_v1_replay_loadable_under_v1_1_schema_bundle,
    generate_report,
    load_geometry_neutral_replay_report,
    register_v1_1_schema,
    save_geometry_neutral_replay_report,
    validate_report,
)

report = generate_report(steps=12, seed=314159)
bench = report["geometry_neutral_replay"]
bench["schema_version"] = SCHEMA_VERSION_V1_1
bench["pulse_id"] = "123e4567-e89b-12d3-a456-426614174000"
bench["capacitor_state_initial_J"] = 18250.0
bench["trigger_timestamp_ns"] = 500000
bench["energy_recovered_J"] = 7350.25
bench["shot_phase_log"] = [
    {"t": 0.0, "state": "precharge", "reason": "bank voltage admitted"},
    {"t": 0.0005, "state": "trigger", "reason": "ignitron trigger issued"},
    {"t": 0.0010, "state": "burn", "reason": "FRC burn window entered"},
]
bench["frc_diagnostics"] = {
    "s_parameter_at_burn": 1.42,
    "mrti_peak_amplitude_m": 0.0032,
    "tilt_growth_rate_s_inv": -8.5,
}

register_v1_1_schema()
validate_report(report)
path = save_geometry_neutral_replay_report(report, "validation/reports/replay.json")
loaded = load_geometry_neutral_replay_report(path)
assert loaded == report
```

For existing v1 reports:

```python
report = generate_report()
assert_v1_replay_loadable_under_v1_1_schema_bundle(report)
```

## Runtime and benchmark impact

The v1.1 change is metadata admission and schema validation. It does not change
the control loop, Rust data plane, solver hot path, transport layer, or replay
numerics. No new performance benchmark is required for this schema lane. If a
future change wires these fields into runtime scheduling, capacitor-bank control,
FRC state estimation, or transport egress, the matching Python/Rust/PyO3
benchmarks must be rerun and documented with the hardware context.

## Claim boundary

A geometry-neutral replay report is repository evidence for deterministic
scenario replay. It is not a measured-shot cross-validation, public P-EFIT
admission, HIL campaign, real-time PCS certification, or facility safety case.
Those claims remain blocked until their own artefacts are produced, digested,
reviewed, and admitted.
