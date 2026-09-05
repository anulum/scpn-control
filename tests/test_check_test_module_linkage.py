# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Published SPO review to CONTROL decision integration.

"""Real repository API linkage and parser false-positive regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools/check_test_module_linkage.py"
OWNER = "src/scpn_control/reactor_semantic_admission/regime_assessment_admission.py"


def test_current_repository_resolves_called_admission_facade_owners() -> None:
    """Resolve facade calls across the actual complete repository."""
    result = subprocess.run([sys.executable, str(GUARD)], capture_output=True, text=True, check=True)
    assert "Unexpected modules: 0" in result.stdout


@pytest.mark.parametrize(
    "body",
    [
        "# scpn_control.reactor_semantic_admission.regime_assessment_admission\ndef test_regime_assessment_admission(): pass\n",
        "from scpn_control.reactor_semantic_admission import admit_reactor_regime_assessment\ndef test_unused_import(): pass\n",
        '"scpn_control.reactor_semantic_admission.regime_assessment_admission"\ndef test_string_decoy(): pass\n',
    ],
)
def test_comments_strings_and_unused_imports_cannot_link_owner(tmp_path: Path, body: str) -> None:
    """Reject text-only decoys at the public guard CLI."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_regime_assessment_admission.py").write_text(body)
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps({"allowlisted_modules": []}))
    result = subprocess.run(
        [sys.executable, str(GUARD), "--test-root", str(tests), "--allowlist", str(allowlist)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert OWNER in result.stdout


def test_called_facade_alias_links_real_owner_but_not_untested_sibling(tmp_path: Path) -> None:
    """Resolve an aliased call without admitting a different owner."""
    (tmp_path / "test_alias.py").write_text(
        "from scpn_control.reactor_semantic_admission import admit_reactor_regime_assessment as admit\n"
        "def test_admission(): admit(b'{}', policy=None)\n"
    )
    result = subprocess.run(
        [sys.executable, str(GUARD), "--test-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert OWNER not in result.stdout
    assert "src/scpn_control/reactor_semantic_admission/mif_admission.py" in result.stdout
