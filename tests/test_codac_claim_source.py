# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CODAC claim-source regression tests.

"""Exercise public CODAC evidence refusal without hardware provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from scpn_control.control.codac_evidence import (
    CODAC_RUNTIME_EVIDENCE_QUALIFIED,
    CODACRuntimeEvidence,
    _payload_sha256,
)
from scpn_control.control.codac_interface import (
    CODACConfig,
    CODACInterface,
    assert_codac_runtime_claim_admissible,
    codac_runtime_evidence,
    load_codac_runtime_evidence,
    save_codac_runtime_evidence,
)


def _local_evidence() -> CODACRuntimeEvidence:
    """Build a bounded synthetic record with plausible timing counters."""
    interface = CODACInterface(CODACConfig(), controller=object())
    return codac_runtime_evidence(
        interface,
        controller_id="synthetic-controller",
        observed_cycle_us=[410.0, 420.0, 430.0],
        interlock_checks=2,
        interlock_blocks=1,
        generated_utc="2026-09-23T00:00:00Z",
    )


def test_caller_counters_cannot_admit_facility_evidence() -> None:
    """Plausible caller timings and interlock counts have no runtime origin."""
    interface = CODACInterface(CODACConfig(), controller=object())
    with pytest.raises(ValueError, match="independent"):
        codac_runtime_evidence(
            interface,
            controller_id="synthetic-controller",
            observed_cycle_us=[410.0, 420.0, 430.0],
            interlock_checks=2,
            interlock_blocks=1,
            facility_claim_allowed=True,
        )


def test_resealed_facility_flag_cannot_be_loaded_or_saved(tmp_path: Path) -> None:
    """A digest over caller bytes does not establish CODAC runtime provenance."""
    evidence = _local_evidence()
    payload: dict[str, Any] = asdict(evidence)
    payload["facility_claim_allowed"] = True
    payload["claim_status"] = CODAC_RUNTIME_EVIDENCE_QUALIFIED
    payload["payload_sha256"] = _payload_sha256(payload)
    destination = tmp_path / "evidence.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independent"):
        load_codac_runtime_evidence(destination)
    with pytest.raises(ValueError, match="independent"):
        assert_codac_runtime_claim_admissible(replace(evidence, **payload))

    new_destination = tmp_path / "new" / "evidence.json"
    with pytest.raises(ValueError, match="independent"):
        save_codac_runtime_evidence(replace(evidence, **payload), new_destination)
    assert not new_destination.parent.exists()


def test_v2_qualification_contract_is_rejected(tmp_path: Path) -> None:
    """The old self-qualified schema cannot be reinterpreted as current evidence."""
    payload: dict[str, Any] = asdict(_local_evidence())
    payload["schema_version"] = "scpn-control.codac-runtime-evidence.v2"
    payload["payload_sha256"] = _payload_sha256(payload)
    destination = tmp_path / "legacy.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_codac_runtime_evidence(destination)


def test_resealed_nonfinite_extension_is_rejected(tmp_path: Path) -> None:
    """An unrecognized field cannot smuggle nonstandard JSON into evidence."""
    payload: dict[str, Any] = asdict(_local_evidence())
    payload["extra"] = float("nan")
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    destination = tmp_path / "nonfinite.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_codac_runtime_evidence(destination)
