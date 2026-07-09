# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Report Tests

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

import pytest
import validation.generate_physics_traceability_report as traceability_report
from validation.generate_physics_traceability_report import generate_physics_traceability_markdown, main
from validation.validate_physics_traceability import validate_physics_traceability


ROOT = Path(__file__).resolve().parents[1]


def test_generate_physics_traceability_markdown_bounds_public_claims() -> None:
    registry_path = ROOT / "validation" / "physics_traceability.json"
    report = validate_physics_traceability(registry_path)
    tracker_counts: dict[int, int] = {}
    tracker_status_counts: dict[int, dict[str, int]] = {}
    for entry in report["entries"]:
        issue = entry.get("external_validation_tracker_issue")
        if isinstance(issue, int):
            tracker_counts[issue] = tracker_counts.get(issue, 0) + 1
            status = entry.get("fidelity_status")
            assert isinstance(status, str)
            tracker_status_counts.setdefault(issue, {})
            tracker_status_counts[issue][status] = tracker_status_counts[issue].get(status, 0) + 1
    markdown = generate_physics_traceability_markdown(registry_path)

    assert markdown.startswith("# Physics Traceability and Bounded Claims\n")
    assert "SPDX-License-Identifier" not in "\n".join(markdown.splitlines()[:8])
    assert "# Physics Traceability and Bounded Claims" in markdown
    assert f"Open fidelity gaps: {report['open_fidelity_gaps']}" in markdown
    assert f"Full-fidelity public claims blocked: {report['public_claim_blocked']}" in markdown
    assert f"External validation trackers: {report['external_validation_tracker_count']}" in markdown
    coverage_match = re.search(r"Source marker coverage: (\d+)/(\d+)", markdown)
    assert coverage_match is not None
    covered = int(coverage_match.group(1))
    total = int(coverage_match.group(2))
    assert total > 0
    assert covered == total
    assert "## Module Traceability Table" in markdown
    assert "## External Validation Collaboration Trackers" in markdown
    assert "External validation artefacts needed for full-fidelity SCPN-CONTROL claims" in markdown
    assert "Parent tracker for external code, reference data, facility replay" in markdown
    assert "https://github.com/anulum/scpn-control/issues/46" in markdown
    assert "https://github.com/anulum/scpn-control/issues/53" in markdown
    assert f"#47](https://github.com/anulum/scpn-control/issues/47) — {tracker_counts[47]} open claim(s)" in markdown
    assert f"#49](https://github.com/anulum/scpn-control/issues/49) — {tracker_counts[49]} open claim(s)" in markdown
    assert f"#53](https://github.com/anulum/scpn-control/issues/53) — {tracker_counts[53]} open claim(s)" in markdown
    assert tracker_status_counts[49]["validation_gap"] == 12
    assert tracker_status_counts[49]["bounded_model"] == 6
    assert (
        f"#49](https://github.com/anulum/scpn-control/issues/49) — {tracker_counts[49]} open claim(s) "
        "(validation_gap=12, bounded_model=6)"
    ) in markdown
    assert (
        f"#50](https://github.com/anulum/scpn-control/issues/50) — {tracker_counts[50]} open claim(s) "
        "(external_dependency_blocked=2, validation_gap=1)"
    ) in markdown
    assert (
        "| Module | Equation or contract | References | Unit contract | Validation evidence | Status | Tracker |"
        in markdown
    )
    assert "External validation tracker: [#47](https://github.com/anulum/scpn-control/issues/47)" in markdown
    assert "`src/scpn_control/core/gk_nonlinear.py`" in markdown
    assert "Five-dimensional delta-f flux-tube Vlasov evolution" in markdown
    assert "Dimits et al. 2000 Cyclone Base Case" in markdown
    assert "Gyro-Bohm-normalised heat flux" in markdown
    assert "docs/joss_paper.md nonlinear GK validation limitation" in markdown
    assert "DIII-D experimental replay" in markdown
    assert "RZIP rigid vertical stability model" in markdown
    assert "resistive-wall-mode feedback model" in markdown
    assert "real-time EFIT-lite equilibrium reconstruction" in markdown
    assert "kinetic EFIT pressure and q-profile coupling" in markdown
    assert "halo current and runaway electron disruption model" in markdown
    assert "direct free-boundary tracking controller" in markdown
    assert "density control and particle-source model" in markdown
    assert "disruption mitigation contract layer" in markdown
    assert "tokamak digital twin topology and diffusion model" in markdown
    assert "advanced SOC turbulence learning controller" in markdown
    assert "DT burn control and alpha-heating model" in markdown
    assert "volt-second budget and flux-consumption manager" in markdown
    assert "Kuramoto-Sakaguchi phase synchronisation runtime" in markdown
    assert "SCPN FPGA fixed-point export boundary" in markdown
    assert "geometry-neutral stellarator replay fixture" in markdown
    assert "bounded control plant approximations" in markdown
    assert "bounded phase and spiking runtime approximations" in markdown
    assert "nonlinear gyrokinetic heat-flux saturation" in markdown
    assert "Miller local-equilibrium geometry and field-pitch contract" in markdown
    assert "gyrokinetic species and collision bounded-operator contract" in markdown
    assert "JAX gyrokinetic numerical parity guard" in markdown
    assert "gyrokinetic OOD detector distribution-bound contract" in markdown
    assert "strict digest-bound external interface artefact gate" in markdown
    assert "transport solver neoclassical and source-term approximation contract" in markdown
    assert "reduced gyrokinetic transport closure contract" in markdown
    assert "momentum transport and torque-balance approximation contract" in markdown
    assert "EPED pedestal and peeling-ballooning approximation contract" in markdown
    assert "ELM crash and RMP suppression approximation contract" in markdown
    assert "MARFE radiation-condensation density-limit contract" in markdown
    assert "NTM island evolution and control approximation contract" in markdown
    assert "auxiliary current-drive deposition and efficiency model" in markdown
    assert "ideal-MHD stability metric approximation contract" in markdown
    assert "sawtooth-to-NTM seeding approximation contract" in markdown
    assert "blob transport and scrape-off-layer approximation contract" in markdown
    assert "checkpoint state serialisation boundary contract" in markdown
    assert "disruption sequence phase-ordering contract" in markdown
    assert "Grad-Shafranov fusion-kernel numerical contract" in markdown
    assert "gyrokinetic online learner stability contract" in markdown
    assert "integrated scenario coupling contract" in markdown
    assert "neural transport surrogate validation contract" in markdown
    assert "neural turbulence surrogate validation contract" in markdown
    assert "orbit-following guiding-centre approximation contract" in markdown
    assert "full-chain uncertainty quantification contract" in markdown
    assert "VMEC-lite stellarator equilibrium approximation contract" in markdown
    assert "- Covered source paths: 1" in markdown
    assert "Full-fidelity public claim: blocked" in markdown
    for banned_identity in ("Co" + "dex", "Open" + "AI"):
        assert banned_identity not in markdown


def test_generate_markdown_handles_missing_optional_report_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    report: dict[str, Any] = {
        "status": "fail",
        "total": 1,
        "open_fidelity_gaps": 1,
        "public_claim_blocked": 1,
        "resolved_module_paths": 0,
        "resolved_evidence_paths": 0,
        "external_validation_tracker_count": 0,
        "source_marker_coverage": "unknown",
        "external_validation_trackers": "not-a-list",
        "entries": [
            {
                "component": "pipe component",
                "module_path": "src/example|module.py",
                "equation_contract": "contract",
                "model_references": "not-a-list",
                "unit_contract": "unit",
                "validation_evidence": ["evidence"],
                "fidelity_status": "validation_gap",
                "public_claim_allowed": False,
                "required_actions": ["add evidence"],
            }
        ],
        "errors": [{"field": "entries[0]", "error": "synthetic failure"}],
    }
    monkeypatch.setattr(traceability_report, "validate_physics_traceability", lambda _registry: report)

    markdown = traceability_report.generate_physics_traceability_markdown("registry.json")

    assert "Source marker coverage: 0/0" in markdown
    assert "`entries[0]`: synthetic failure" in markdown
    assert "`src/example\\|module.py`" in markdown
    assert "External validation tracker: none" in markdown


def test_generate_markdown_handles_invalid_entries_and_marker_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    report: dict[str, Any] = {
        "status": "pass",
        "total": 0,
        "open_fidelity_gaps": 0,
        "public_claim_blocked": 0,
        "resolved_module_paths": 0,
        "resolved_evidence_paths": 0,
        "external_validation_tracker_count": 0,
        "source_marker_coverage": {"covered": "0", "total": 0},
        "external_validation_trackers": [],
        "entries": "not-a-list",
        "errors": [],
    }
    monkeypatch.setattr(traceability_report, "validate_physics_traceability", lambda _registry: report)

    markdown = traceability_report.generate_physics_traceability_markdown("registry.json")

    assert "Source marker coverage: 0/0" in markdown
    assert "## Components\n" in markdown


def test_generate_markdown_skips_invalid_tracker_status(monkeypatch: pytest.MonkeyPatch) -> None:
    report: dict[str, Any] = {
        "status": "pass",
        "total": 1,
        "open_fidelity_gaps": 0,
        "public_claim_blocked": 0,
        "resolved_module_paths": 1,
        "resolved_evidence_paths": 1,
        "external_validation_tracker_count": 1,
        "source_marker_coverage": {"covered": 1, "total": 1},
        "external_validation_trackers": [
            {"issue": 1, "title": "Tracker", "url": "https://example.invalid/1", "scope": "scope"}
        ],
        "entries": [
            {
                "component": "tracked",
                "module_path": "src/tracked.py",
                "equation_contract": "contract",
                "model_references": [],
                "unit_contract": "unit",
                "validation_evidence": [],
                "fidelity_status": 7,
                "external_validation_tracker_issue": 1,
                "public_claim_allowed": True,
                "required_actions": [],
            }
        ],
        "errors": [],
    }
    monkeypatch.setattr(traceability_report, "validate_physics_traceability", lambda _registry: report)

    markdown = traceability_report.generate_physics_traceability_markdown("registry.json")

    assert "Tracker: [#1](https://example.invalid/1) — 1 open claim(s) — scope" in markdown
    assert "External validation tracker: [#1](https://example.invalid/1) — Tracker" in markdown


def test_module_bootstrap_adds_repo_root_to_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    root = str(ROOT)
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != root])
    spec = importlib.util.spec_from_file_location(
        "physics_traceability_bootstrap_test",
        ROOT / "validation" / "generate_physics_traceability_report.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert isinstance(module, ModuleType)
    assert root in sys.path


def test_generate_physics_traceability_report_writes_file(tmp_path: Path) -> None:
    output = tmp_path / "physics_traceability.md"

    exit_code = main(
        [
            "--registry",
            str(ROOT / "validation" / "physics_traceability.json"),
            "--output-md",
            str(output),
        ]
    )

    assert exit_code == 0
    content = output.read_text(encoding="utf-8")
    assert "Resolved evidence paths:" in content
    assert "validation/physics_traceability.json" in content


def test_repository_physics_traceability_report_is_current() -> None:
    expected = generate_physics_traceability_markdown(ROOT / "validation" / "physics_traceability.json")
    actual = (ROOT / "docs" / "physics_traceability.md").read_text(encoding="utf-8")

    assert actual == expected
