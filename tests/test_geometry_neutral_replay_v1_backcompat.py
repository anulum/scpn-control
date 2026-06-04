# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Tests for geometry-neutral replay v1 compatibility under v1.1 bundle.
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


def test_v1_report_loads_under_v1_1_schema_bundle() -> None:
    report = generate_report(steps=8, seed=20260604)

    assert load_replay_schema(SCHEMA_VERSION)["$id"] == SCHEMA_VERSION
    assert register_v1_1_schema()["$id"] == SCHEMA_VERSION_V1_1
    assert assert_v1_replay_loadable_under_v1_1_schema_bundle(report) is report


def test_v1_1_backcompat_admission_rejects_non_v1_reports() -> None:
    report = generate_report(steps=8, seed=20260604)
    report["geometry_neutral_replay"]["schema_version"] = SCHEMA_VERSION_V1_1

    with pytest.raises(ValueError, match="expected a v1 replay report"):
        assert_v1_replay_loadable_under_v1_1_schema_bundle(report)
