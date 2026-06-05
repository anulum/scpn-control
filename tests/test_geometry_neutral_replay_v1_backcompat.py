# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometry-neutral replay v1 back-compatibility tests.
from __future__ import annotations

import pytest

from scpn_control.scpn.geometry_neutral_replay import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V1_1,
    assert_v1_replay_loadable_under_v1_1_schema_bundle,
    generate_report,
    load_replay_schema,
    register_v1_1_schema,
)


@pytest.mark.parametrize(
    ("steps", "seed"),
    [
        (8, 20260604),
        (9, 20260605),
        (10, 20260606),
        (11, 20260607),
        (12, 314159),
        (13, 271828),
        (14, 161803),
        (15, 141421),
    ],
)
def test_v1_report_fixtures_load_under_v1_1_schema_bundle(steps: int, seed: int) -> None:
    report = generate_report(steps=steps, seed=seed)

    assert load_replay_schema(SCHEMA_VERSION)["$id"] == SCHEMA_VERSION
    assert register_v1_1_schema()["$id"] == SCHEMA_VERSION_V1_1
    assert assert_v1_replay_loadable_under_v1_1_schema_bundle(report) is report


def test_v1_1_backcompat_admission_rejects_non_v1_reports() -> None:
    report = generate_report(steps=8, seed=20260604)
    report["geometry_neutral_replay"]["schema_version"] = SCHEMA_VERSION_V1_1

    with pytest.raises(ValueError, match="expected a v1 replay report"):
        assert_v1_replay_loadable_under_v1_1_schema_bundle(report)
