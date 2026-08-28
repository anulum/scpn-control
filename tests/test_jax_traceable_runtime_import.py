# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX traceable runtime import hygiene tests.

"""Import hygiene for optional compiled traceable-runtime backends."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_traceable_runtime_imports_with_deprecation_warnings_as_errors() -> None:
    code = "import scpn_control.control.jax_traceable_runtime"

    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
