# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sealed device diagnostic review admission tests.

"""Real public-boundary tests for exact SPO diagnostic review admission."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from scpn_phase_orchestrator.reactor_semantics import (
    device_diagnostic_plan_review_from_bytes,
)

from scpn_control.reactor_semantic_admission import (
    SPO_CONTRACT_MODULE_SHA256,
    SPO_DISTRIBUTION_VERSION,
    SPO_RELEASE_WHEEL_SHA256,
    SPO_REVIEW_SCHEMA,
    SPO_REVIEW_SCHEMA_VERSION,
    TOKAMAK_CLOCK_CUSTODY_SHA256,
    TOKAMAK_CONFIGURATIONS,
    TOKAMAK_REVIEW_ID,
    TOKAMAK_REVIEW_SHA256,
    TOKAMAK_SOURCE_ARTIFACT_SHA256,
    TOKAMAK_SOURCE_ENVELOPE_SHA256,
    TOKAMAK_SOURCE_MANIFEST_SHA256,
    TOKAMAK_SOURCE_PLAN_SHA256,
    TOKAMAK_SOURCE_PROJECT,
    TOKAMAK_SOURCE_REVISION,
    DeviceDiagnosticReviewAdmissionPolicy,
    DeviceDiagnosticReviewAdmissionStatus,
    admit_device_diagnostic_plan_review,
    device_diagnostic_review_clock_custody_digest,
    tokamak_device_diagnostic_review_policy,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/tokamak_device_diagnostic_review_v1.json"


def _review_bytes() -> bytes:
    """Return the upstream canonical bytes carried in a newline-terminated file."""
    carrier = FIXTURE.read_bytes()
    assert carrier.endswith(b"\n")
    payload = carrier[:-1]
    assert len(payload) == 16_476
    assert hashlib.sha256(payload).hexdigest() == TOKAMAK_REVIEW_SHA256
    return payload


def _reseal(record: dict[str, object]) -> bytes:
    payload = record["payload"]
    canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    record["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def test_exact_published_tokamak_review_is_accepted_without_authority() -> None:
    """Accept the upstream-sealed review and preserve every false authority bit."""
    payload = _review_bytes()
    decision = admit_device_diagnostic_plan_review(
        payload,
        policy=tokamak_device_diagnostic_review_policy(),
    )

    assert decision.decision is DeviceDiagnosticReviewAdmissionStatus.ACCEPTED_FOR_REVIEW
    assert decision.accepted_for_review is True
    assert decision.review_sha256 == TOKAMAK_REVIEW_SHA256
    assert decision.review_id == TOKAMAK_REVIEW_ID
    assert decision.review_schema == SPO_REVIEW_SCHEMA
    assert decision.review_schema_version == SPO_REVIEW_SCHEMA_VERSION
    assert decision.source_project == TOKAMAK_SOURCE_PROJECT
    assert decision.source_revision == TOKAMAK_SOURCE_REVISION
    assert decision.source_artifact_sha256 == TOKAMAK_SOURCE_ARTIFACT_SHA256
    assert decision.source_manifest_sha256 == TOKAMAK_SOURCE_MANIFEST_SHA256
    assert decision.source_envelope_sha256 == TOKAMAK_SOURCE_ENVELOPE_SHA256
    assert decision.source_plan_sha256 == TOKAMAK_SOURCE_PLAN_SHA256
    assert decision.configurations == TOKAMAK_CONFIGURATIONS
    assert decision.clock_custody_sha256 == TOKAMAK_CLOCK_CUSTODY_SHA256
    assert decision.spo_distribution_version == SPO_DISTRIBUTION_VERSION
    assert decision.spo_release_wheel_sha256 == SPO_RELEASE_WHEEL_SHA256
    assert decision.spo_contract_module_sha256 == SPO_CONTRACT_MODULE_SHA256
    assert decision.refusal_codes == ()
    assert decision.review_only is True
    assert decision.evidence_claimed is False
    assert decision.observation_claimed is False
    assert decision.measurement_claimed is False
    assert decision.facility_binding_claimed is False
    assert decision.classification_performed is False
    assert decision.semantic_ingress_declared is False
    assert decision.control_intent_created is False
    assert decision.actionable is False
    assert decision.execution_authorised is False
    assert decision.actuation_authorised is False


def test_clock_custody_binds_all_three_nonisomorphic_clocks() -> None:
    """Bind facility, shot-relative and simulation clocks without mapping claims."""
    review = device_diagnostic_plan_review_from_bytes(_review_bytes())
    clocks = {
        item.plan_clock_identifier: (
            item.plan_clock_kind,
            item.compatibility.value,
            item.mapping_evidence_claimed,
        )
        for item in review.clock_reviews
    }

    assert clocks == {
        "clk_facility": ("facility_monotonic", "unmapped", False),
        "clk_shot": ("shot_event_epoch", "event_relative_compatible", False),
        "clk_sim": ("simulation", "synthetic_compatible", False),
    }
    assert device_diagnostic_review_clock_custody_digest(review) == (TOKAMAK_CLOCK_CUSTODY_SHA256)


@pytest.mark.parametrize(
    ("change", "expected_code"),
    [
        ({"expected_review_sha256": "0" * 64}, "review_digest_mismatch"),
        ({"expected_review_id": "0" * 64}, "review_identity_mismatch"),
        ({"expected_source_project": "SCPN-Z-PINCH-CORE"}, "source_project_mismatch"),
        ({"expected_source_revision": "0" * 40}, "source_revision_mismatch"),
        ({"expected_source_artifact_sha256": "0" * 64}, "source_artifact_mismatch"),
        ({"expected_source_manifest_sha256": "0" * 64}, "source_manifest_digest_mismatch"),
        ({"expected_source_envelope_sha256": "0" * 64}, "source_envelope_digest_mismatch"),
        ({"expected_source_plan_sha256": "0" * 64}, "source_plan_digest_mismatch"),
        ({"expected_configurations": ("spherical_tokamak",)}, "configuration_mismatch"),
        ({"expected_clock_custody_sha256": "0" * 64}, "clock_custody_mismatch"),
    ],
)
def test_each_consumer_binding_drift_fails_independently(
    change: dict[str, object],
    expected_code: str,
) -> None:
    """Reject stale, wrong-device, digest, configuration and clock bindings."""
    policy = replace(tokamak_device_diagnostic_review_policy(), **cast(Any, change))
    decision = admit_device_diagnostic_plan_review(_review_bytes(), policy=policy)

    assert decision.accepted_for_review is False
    assert expected_code in decision.refusal_codes
    assert decision.actionable is False
    assert decision.execution_authorised is False
    assert decision.actuation_authorised is False


@pytest.mark.parametrize(
    "mutation",
    [
        "payload_tamper",
        "wrong_version",
        "wrong_digest",
        "unknown_clock",
        "contradictory_authority",
        "noncanonical",
    ],
)
def test_public_spo_decoder_failures_become_identity_empty_rejections(
    mutation: str,
) -> None:
    """Use the installed public SPO decoder for every malformed-review case."""
    valid = _review_bytes()
    record = cast("dict[str, Any]", json.loads(valid))
    if mutation == "payload_tamper":
        record["payload"]["review_id"] = "0" * 64
        changed = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    elif mutation == "wrong_version":
        record["schema_version"] = "2.0.0"
        changed = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    elif mutation == "wrong_digest":
        record["payload_sha256"] = "0" * 64
        changed = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    elif mutation == "unknown_clock":
        record["payload"]["clock_reviews"][0]["plan_clock_kind"] = "unknown_clock"
        changed = _reseal(record)
    elif mutation == "contradictory_authority":
        record["payload"]["authority"] = "execution"
        changed = _reseal(record)
    else:
        changed = valid + b"\n"

    decision = admit_device_diagnostic_plan_review(
        changed,
        policy=tokamak_device_diagnostic_review_policy(),
    )

    assert decision.accepted_for_review is False
    assert decision.refusal_codes == ("review_decode_failed",)
    assert decision.review_sha256 == hashlib.sha256(changed).hexdigest()
    assert decision.review_id is None
    assert decision.source_project is None
    assert decision.source_revision is None
    assert decision.source_artifact_sha256 is None
    assert decision.source_manifest_sha256 is None
    assert decision.source_envelope_sha256 is None
    assert decision.source_plan_sha256 is None
    assert decision.configurations == ()
    assert decision.clock_custody_sha256 is None
    assert decision.evidence_claimed is False
    assert decision.observation_claimed is False
    assert decision.measurement_claimed is False
    assert decision.facility_binding_claimed is False
    assert decision.classification_performed is False
    assert decision.semantic_ingress_declared is False
    assert decision.control_intent_created is False
    assert decision.actionable is False
    assert decision.execution_authorised is False
    assert decision.actuation_authorised is False


def test_non_bytes_ingress_fails_without_inventing_a_digest() -> None:
    """Reject non-byte inputs without guessing review or source identity."""
    decision = admit_device_diagnostic_plan_review(
        cast(bytes, "not bytes"),
        policy=tokamak_device_diagnostic_review_policy(),
    )

    assert decision.refusal_codes == ("review_decode_failed",)
    assert decision.review_sha256 is None
    assert decision.review_id is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"expected_review_sha256": "bad"}, "expected_review_sha256"),
        ({"expected_review_id": "bad"}, "expected_review_id"),
        ({"expected_source_revision": "main"}, "expected_source_revision"),
        ({"expected_source_project": ""}, "expected_source_project"),
        ({"expected_configurations": ["conventional_tokamak"]}, "expected_configurations"),
        ({"expected_configurations": ("",)}, "expected_configurations"),
        (
            {"expected_configurations": ("spherical_tokamak", "conventional_tokamak")},
            "sorted and unique",
        ),
    ],
)
def test_policy_rejects_ambiguous_or_unsealed_expectations(
    change: dict[str, object],
    message: str,
) -> None:
    """Reject malformed consumer policy before any review is admitted."""
    with pytest.raises(ValueError, match=message):
        replace(tokamak_device_diagnostic_review_policy(), **cast(Any, change))


def test_policy_type_is_public_and_immutable() -> None:
    """Expose one explicit immutable policy type rather than hidden constants."""
    policy = tokamak_device_diagnostic_review_policy()
    assert isinstance(policy, DeviceDiagnosticReviewAdmissionPolicy)
    with pytest.raises(AttributeError):
        policy.expected_source_project = "changed"  # type: ignore[misc] # frozen dataclass


_PACKAGE_LAYOUT_PROBE = """
import json
import sys
from pathlib import Path
source_root, fixture, review_path = sys.argv[1:]
sys.path[:] = [source_root, fixture] + [
    path for path in sys.path
    if "site-packages" not in path and "dist-packages" not in path
]
from scpn_control.reactor_semantic_admission import (
    admit_device_diagnostic_plan_review, tokamak_device_diagnostic_review_policy,
)
carrier = Path(review_path).read_bytes()
decision = admit_device_diagnostic_plan_review(
    carrier[:-1], policy=tokamak_device_diagnostic_review_policy()
)
print(json.dumps({"codes": list(decision.refusal_codes), "source": decision.source_project}))
"""


def _write_distribution_metadata(root: Path, version: str) -> None:
    dist_info = root / f"scpn_phase_orchestrator-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: scpn-phase-orchestrator\nVersion: {version}\n",
        encoding="utf-8",
    )


def _run_package_layout_probe(fixture: Path) -> dict[str, object]:
    source = (
        Path(__file__).parents[1] / "src/scpn_control/reactor_semantic_admission/device_diagnostic_review_admission.py"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PACKAGE_LAYOUT_PROBE,
            str(source.parents[2]),
            str(fixture),
            str(FIXTURE),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20.0,
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing-distribution", "spo_contract_unavailable"),
        ("wrong-version", "spo_distribution_version_mismatch"),
        ("missing-contract", "spo_contract_unavailable"),
        ("missing-source-path", "spo_contract_unavailable"),
        ("unreadable-source", "spo_contract_unavailable"),
        ("mismatched-source", "spo_contract_module_digest_mismatch"),
    ],
)
def test_public_admission_rejects_incompatible_package_layouts(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    """Exercise isolated package layouts through the public admission function."""
    package = tmp_path / "scpn_phase_orchestrator"
    facade = package / "reactor_semantics"
    facade.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (facade / "__init__.py").write_text("", encoding="utf-8")
    contract = facade / "diagnostic_plan_review.py"

    if case != "missing-distribution":
        _write_distribution_metadata(tmp_path, "9.9.0" if case == "wrong-version" else "1.4.3")
    if case == "missing-contract":
        pass
    elif case == "missing-source-path":
        contract.write_text("del __file__\n", encoding="utf-8")
    elif case == "unreadable-source":
        contract.write_text(f"__file__ = {str(facade)!r}\n", encoding="utf-8")
    else:
        contract.write_text("CONTRACT_PRESENT = True\n", encoding="utf-8")

    assert _run_package_layout_probe(tmp_path) == {
        "codes": [expected_code],
        "source": None,
    }


_PUBLIC_CONTRACT_DRIFT_PROBE = """
import json
import sys
from pathlib import Path
import scpn_phase_orchestrator.reactor_semantics as facade

case, review_path = sys.argv[1:]
if case == "absent-api":
    del facade.device_diagnostic_plan_review_from_bytes
elif case == "replaced-decoder":
    facade.device_diagnostic_plan_review_from_bytes = lambda payload: payload
else:
    facade.DEVICE_DIAGNOSTIC_PLAN_REVIEW_VERSION = "9.9.9"
from scpn_control.reactor_semantic_admission import (
    admit_device_diagnostic_plan_review,
    tokamak_device_diagnostic_review_policy,
)
carrier = Path(review_path).read_bytes()
decision = admit_device_diagnostic_plan_review(
    carrier[:-1], policy=tokamak_device_diagnostic_review_policy()
)
print(json.dumps({"codes": list(decision.refusal_codes), "source": decision.source_project}))
"""


@pytest.mark.parametrize("case", ["absent-api", "replaced-decoder", "schema-drift"])
def test_public_admission_rejects_hostile_installed_facade_drift(case: str) -> None:
    """Reject missing public API and schema drift in an isolated real install."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PUBLIC_CONTRACT_DRIFT_PROBE), case, str(FIXTURE)],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert json.loads(completed.stdout) == {
        "codes": ["spo_contract_unavailable"],
        "source": None,
    }
