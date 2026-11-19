# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Script entries, import-path effects and the blocked Z3 path of the evidence validators.
"""The evidence and gate validators answer ``--help`` as scripts and edit the import path only then.

The help run reaches each module's ``__main__`` line without validating or
writing anything. Thirteen validators put the checkout or its sources on the
import path for a script run; imported as ``validation.<module>`` they must
leave it alone. The import-path checks strip the checkout and its sources
first, because with them present an unguarded "insert if absent" would also
leave the path unchanged. The Z3 publisher's unavailable-solver path is driven
by making the defining checker raise its documented RuntimeError.
"""

from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from validation import validate_scpn_z3_formal as z3_publisher

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

EVIDENCE_VALIDATORS = (
    "validate_benchmark_regression_gates",
    "validate_data_manifests",
    "validate_differentiable_transport_latency",
    "validate_e2e_latency_evidence",
    "validate_gk_collision_independent",
    "validate_gk_geometry_independent",
    "validate_gk_interface_artifacts",
    "validate_gk_ood_calibration",
    "validate_jax_gk_parity",
    "validate_multi_shot_campaign_evidence",
    "validate_native_formal_certificate_evidence",
    "validate_physics_traceability",
    "validate_public_data_acquisition",
    "validate_real_shots",
    "validate_release_evidence",
    "validate_runtime_admission_evidence",
    "validate_scpn_lean_formal",
    "validate_scpn_z3_formal",
    "validate_tracker53_evidence",
)

# Module name -> import-path entries a script run puts first, in order.
SCRIPT_PREFIX: dict[str, tuple[Path, ...]] = {
    **{
        name: (ROOT,)
        for name in (
            "validate_data_manifests",
            "validate_differentiable_transport_latency",
            "validate_e2e_latency_evidence",
            "validate_gk_interface_artifacts",
            "validate_gk_ood_calibration",
            "validate_jax_gk_parity",
            "validate_physics_traceability",
            "validate_tracker53_evidence",
        )
    },
    "validate_gk_collision_independent": (ROOT, SRC),
    "validate_gk_geometry_independent": (ROOT, SRC),
    "validate_real_shots": (SRC,),
    "validate_scpn_lean_formal": (SRC,),
    "validate_scpn_z3_formal": (SRC,),
}


def _stripped() -> list[str]:
    """Return the current import path without the checkout and its sources."""
    return [entry for entry in sys.path if Path(entry or ".").resolve() not in {ROOT, SRC}]


def _run_help(name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Run the validator as a script with ``--help`` and require argparse's help and exit zero."""
    script = ROOT / "validation" / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("name", EVIDENCE_VALIDATORS)
def test_script_entry_answers_help_without_validating(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The script entry hands the command line to ``main``, whose parser prints help and exits zero."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    _run_help(name, monkeypatch, capsys)


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_package_import_leaves_the_import_path_unchanged(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reloading the validator as a package module inserts nothing into a path without the checkout."""
    module = importlib.import_module(f"validation.{name}")
    stripped = _stripped()
    monkeypatch.setattr(sys, "path", list(stripped))
    importlib.reload(module)
    assert sys.path == stripped


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_script_run_puts_its_entries_first(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script from a path without the checkout, the validator inserts its entries in order."""
    monkeypatch.setattr(sys, "path", _stripped())
    _run_help(name, monkeypatch, capsys)
    expected = [str(entry) for entry in SCRIPT_PREFIX[name]]
    assert sys.path[: len(expected)] == expected


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_script_run_with_the_checkout_present_adds_no_duplicate(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script with the checkout and sources already importable, the validator adds no second entry."""
    present = [str(ROOT), str(SRC), *_stripped()]
    monkeypatch.setattr(sys, "path", list(present))
    _run_help(name, monkeypatch, capsys)
    assert sys.path == present


def _unavailable_z3(*args: object, **kwargs: object) -> object:
    """Fail as the defining checker does when z3-solver cannot be imported."""
    raise RuntimeError("z3-solver is not installed")


def test_z3_publisher_writes_a_blocked_declaration_when_z3_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without Z3 the publisher writes and returns a blocked declaration that certifies nothing."""
    monkeypatch.setattr(z3_publisher, "verify_z3_formal_contracts", _unavailable_z3)
    json_path = tmp_path / "out" / "z3.json"
    markdown_path = tmp_path / "md" / "z3.md"
    result = z3_publisher.publish_report(json_path=json_path, markdown_path=markdown_path, require_z3=False)
    assert result["status"] == "blocked"
    assert json.loads(json_path.read_text(encoding="utf-8")) == result
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# SCPN Z3 Formal Verification Report")
    assert "- Status: `blocked`" in markdown


def test_z3_publisher_reraises_when_z3_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With Z3 required the blocked declaration is still written, then the failure propagates."""
    monkeypatch.setattr(z3_publisher, "verify_z3_formal_contracts", _unavailable_z3)
    json_path = tmp_path / "z3.json"
    with pytest.raises(RuntimeError, match="z3-solver is not installed"):
        z3_publisher.publish_report(json_path=json_path, markdown_path=tmp_path / "z3.md", require_z3=True)
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "blocked"


@pytest.mark.parametrize(("required", "code"), [(False, 0), (True, 1)])
def test_z3_command_accepts_a_blocked_run_unless_z3_is_required(
    required: bool,
    code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A blocked run exits zero with its summary, or one with a blocked message when Z3 is required."""
    monkeypatch.setattr(z3_publisher, "verify_z3_formal_contracts", _unavailable_z3)
    argv = ["--json-out", str(tmp_path / "z3.json"), "--markdown-out", str(tmp_path / "z3.md")]
    assert z3_publisher.main([*argv, "--require-z3"] if required else argv) == code
    captured = capsys.readouterr()
    if required:
        assert "SCPN Z3 formal verification: blocked: z3-solver is not installed" in captured.err
    else:
        assert json.loads(captured.out)["status"] == "blocked"
