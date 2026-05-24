# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Report Tests

from __future__ import annotations

from pathlib import Path
import re

from validation.generate_physics_traceability_report import generate_physics_traceability_markdown, main


ROOT = Path(__file__).resolve().parents[1]


def test_generate_physics_traceability_markdown_bounds_public_claims() -> None:
    markdown = generate_physics_traceability_markdown(ROOT / "validation" / "physics_traceability.json")

    assert markdown.startswith("<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->")
    assert "# Physics Traceability and Bounded Claims" in markdown
    assert "Open fidelity gaps: 49" in markdown
    assert "Full-fidelity public claims blocked: 49" in markdown
    coverage_match = re.search(r"Source marker coverage: (\d+)/(\d+)", markdown)
    assert coverage_match is not None
    covered = int(coverage_match.group(1))
    total = int(coverage_match.group(2))
    assert total > 0
    assert covered == total
    assert "## Module Traceability Table" in markdown
    assert "## External Validation Collaboration Trackers" in markdown
    assert "https://github.com/anulum/scpn-control/issues/46" in markdown
    assert "https://github.com/anulum/scpn-control/issues/53" in markdown
    assert "| Module | Equation or contract | References | Unit contract | Validation evidence | Status |" in markdown
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
    assert "strict external interface artifact gate" in markdown
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
