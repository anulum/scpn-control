# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Published SPO review to CONTROL decision integration.

"""End-to-end exchange over installed public SPO and CONTROL byte contracts."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from scpn_phase_orchestrator.reactor_semantics import (
    device_diagnostic_plan_review_from_bytes,
)

from scpn_control.reactor_semantic_admission import (
    SPO_CONTRACT_MODULE_SHA256,
    SPO_RELEASE_WHEEL_SHA256,
    TOKAMAK_REVIEW_ID,
    TOKAMAK_REVIEW_SHA256,
    admit_device_diagnostic_plan_review,
    device_diagnostic_review_decision_from_bytes,
    device_diagnostic_review_decision_to_bytes,
    tokamak_device_diagnostic_review_policy,
)

CONTROL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/tokamak_device_diagnostic_review_v1.json"


def _review_bytes() -> bytes:
    carrier = FIXTURE.read_bytes()
    assert carrier.endswith(b"\n")
    return carrier[:-1]


def test_published_spo_review_crosses_the_public_control_byte_boundary() -> None:
    """Decode the exact upstream review and round-trip CONTROL's sealed result."""
    review_bytes = _review_bytes()
    assert hashlib.sha256(review_bytes).hexdigest() == TOKAMAK_REVIEW_SHA256
    review = device_diagnostic_plan_review_from_bytes(review_bytes)
    assert review.review_id == TOKAMAK_REVIEW_ID

    decision = admit_device_diagnostic_plan_review(
        review_bytes,
        policy=tokamak_device_diagnostic_review_policy(),
    )
    encoded = device_diagnostic_review_decision_to_bytes(decision)

    assert device_diagnostic_review_decision_from_bytes(encoded) == decision
    assert decision.accepted_for_review is True
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


def test_e2e_uses_the_exact_installed_public_spo_release() -> None:
    """Bind the integration path to public SPO 1.4.3 and its decoder source."""
    distribution = importlib.metadata.distribution("scpn-phase-orchestrator")
    package_root = Path(str(distribution.locate_file("scpn_phase_orchestrator"))).resolve()
    module = package_root / "reactor_semantics/diagnostic_plan_review.py"

    assert distribution.version == "1.4.3"
    assert "site-packages" in package_root.parts
    assert hashlib.sha256(module.read_bytes()).hexdigest() == SPO_CONTRACT_MODULE_SHA256
    assert SPO_RELEASE_WHEEL_SHA256 == ("5da94500760f9394a637f7edec044a844c12230d8d16c25a573b6e67a1ddb409")


def test_review_admission_import_stays_isolated_from_action_surfaces() -> None:
    """Import the public gate without loading controllers or actuator modules."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CONTROL_ROOT / "src")
    command = (
        "import sys; "
        "import scpn_control.reactor_semantic_admission as admission; "
        "assert hasattr(admission, 'admit_device_diagnostic_plan_review'); "
        "forbidden=('scpn_control.control','scpn_control.scpn','scpn_control.codac',"
        "'scpn_control.hardware'); "
        "assert not any(any(name.startswith(prefix) for prefix in forbidden) "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=CONTROL_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.stdout == ""


def test_review_admission_source_excludes_raw_producer_and_sibling_dependencies() -> None:
    """Keep the CONTROL owner on typed SPO review fields only."""
    source = CONTROL_ROOT / "src/scpn_control/reactor_semantic_admission/device_diagnostic_review_admission.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported_modules: set[str] = set()
    dynamic_imports: set[str] = set()
    json_calls: set[str] = set()
    accessed_attributes: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Attribute):
            accessed_attributes.add(node.attr)
            if isinstance(node.value, ast.Name) and node.value.id == "json":
                json_calls.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            dynamic_imports.add(node.args[0].value)

    forbidden_module_prefixes = (
        "scpn_fusion",
        "scpn_mif",
        "scpn_tokamak",
    )
    assert not any(module.startswith(forbidden_module_prefixes) for module in imported_modules)
    assert dynamic_imports == {
        "scpn_phase_orchestrator.reactor_semantics",
        "scpn_phase_orchestrator.reactor_semantics.diagnostic_plan_review",
    }
    assert json_calls.isdisjoint({"JSONDecoder", "load", "loads"})
    assert accessed_attributes.isdisjoint(
        {
            "source_manifest_json",
            "source_envelope_json",
            "source_plan_json",
        }
    )
