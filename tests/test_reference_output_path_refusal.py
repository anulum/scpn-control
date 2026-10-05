# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference report path refusal regressions

"""Exercise cyclic output refusal across actual declaration APIs, scripts and registered root commands."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main as root_cli


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("neural_equilibrium", "Neural equilibrium"),
        ("orbit", "Orbit"),
        ("uncertainty", "Uncertainty"),
        ("vmec", "VMEC"),
        ("eped", "EPED"),
        ("marfe", "MARFE"),
        ("ntm", "NTM"),
    ],
)
def test_real_output_cycle_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, label: str) -> None:
    """Each public writer propagates a real path loop; both CLI implementations refuse it without raw path errors.

    Optional empty-root declaration inspection is a valid pass, exercised again
    after refusal. A real unrelated file and the cyclic link remain unchanged;
    no persisted physics carrier, model execution or CLI replacement is used.
    """
    module = importlib.import_module(f"validation.validate_{name}_reference")
    root = tmp_path / "absent"
    marker = tmp_path / "preserved.data"
    marker.write_bytes(b"preserve actual bytes")
    output = tmp_path / "loop.json"
    output.symlink_to(output.name)
    reader = getattr(module, f"validate_{name}_reference")
    writer = getattr(module, f"write_{name}_reference_report")
    report = reader(root)
    assert report["status"] == "pass" and report["reference_artifacts"] == 0
    with pytest.raises((RuntimeError, OSError)):
        writer(report, output, artifact_root=root)
    refusal = f"{label} reference FAILED: could not inspect artifacts or write report"
    assert module.main(["--artifact-root", str(root), "--output-json", str(output)]) == 1
    assert capsys.readouterr().err.strip() == refusal
    result = CliRunner().invoke(
        root_cli,
        [f"validate-{name.replace('_', '-')}-reference", "--artifact-root", str(root), "--output-json", str(output)],
    )
    assert result.exit_code == 1 and refusal in result.output
    assert not isinstance(result.exception, RuntimeError)
    assert "loop.json" not in result.output and "Traceback" not in result.output
    module_path = module.__file__
    assert isinstance(module_path, str)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cold = subprocess.run(
        [
            sys.executable,
            str(Path(module_path).resolve()),
            "--artifact-root",
            str(root),
            "--output-json",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cold.returncode == 1 and cold.stderr.strip() == refusal and not cold.stdout
    saved = tmp_path / "reports" / "valid-output.json"
    writer(report, saved, artifact_root=root)
    assert json.loads(saved.read_text()) == report
    assert reader(root)["status"] == "pass"
    assert marker.read_bytes() == b"preserve actual bytes" and output.readlink() == Path(output.name)
