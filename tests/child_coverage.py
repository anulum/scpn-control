# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Measurement of child interpreters in tests.

"""Let a child interpreter add its measurement to the parent's coverage data.

Some contracts can only be exercised in a fresh interpreter, for example under
a runtime audit hook, which cannot be removed once installed. The parent's
measurement does not reach such a child. When the parent measures, the child
measures the package itself, in the parent's mode, into a parallel data file
beside the parent's; the parent's coverage plugin combines those files when the
session ends. When the parent does not measure, the child does not either.
"""

from __future__ import annotations

import os
from pathlib import Path

import coverage

_FILE = "SCPN_CHILD_COVERAGE_FILE"
_BRANCH = "SCPN_CHILD_COVERAGE_BRANCH"

CHILD_COVERAGE_PRELUDE = f"""
import atexit as _atexit
import os as _os
if "{_FILE}" in _os.environ:
    import coverage as _coverage
    _measurement = _coverage.Coverage(
        data_file=_os.environ["{_FILE}"],
        data_suffix=True,
        branch=_os.environ["{_BRANCH}"] == "1",
        source=["scpn_control"],
        config_file=False,
    )
    _measurement.start()

    def _save_measurement():
        "Write the child's parallel data file when the interpreter exits."
        _measurement.stop()
        _measurement.save()

    _atexit.register(_save_measurement)
"""


def child_environment(*, source_root: Path) -> dict[str, str]:
    """Return the environment of a child interpreter that imports the checkout.

    Parameters
    ----------
    source_root : pathlib.Path
        Directory that holds the package, placed on the child's import path.

    Returns
    -------
    dict of str to str
        The current environment with the import path set. When the calling
        process measures coverage, it also names the parent's data file and
        mode, which ``CHILD_COVERAGE_PRELUDE`` reads in the child.
    """
    environment = dict(os.environ, PYTHONPATH=str(source_root))
    active = coverage.Coverage.current()
    if active is not None:
        data_file = active.get_option("run:data_file")
        assert isinstance(data_file, str)
        environment[_FILE] = str(Path(data_file).resolve())
        # Coverage refuses to combine branch data with statement data.
        environment[_BRANCH] = "1" if active.get_option("run:branch") else "0"
    return environment
