# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual campaign command observation.

"""Execute public campaign commands and retain their process observations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scpn_control.benchmark_records import CAMPAIGN_ENV

ROOT = Path(__file__).resolve().parents[1]


def campaign_command(module: str, entry: str, *, native: bool = False) -> list[str]:
    """Build a real script/imported-main command, optionally loading the existing Rust image.

    Parameters
    ----------
    module : str
        Dotted public validation module.
    entry : str
        script uses its file; imported-main invokes its public main.
    native : bool
        Load the existing debug PyO3 image by its real import specification,
        before importing the campaign. No installation or substitute backend.

    Returns
    -------
    list[str]
        Python argv prefix; caller appends campaign flags.

    Raises
    ------
    AssertionError
        The selected existing native image is absent.
    """
    path = ROOT / (module.replace(".", "/") + ".py")
    if not native and entry == "script":
        return [sys.executable, str(path)]
    code = ""
    if native:
        binary = ROOT / "scpn-control-rs/target/debug/libscpn_control_rs.so"
        assert binary.is_file()
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('scpn_control_rs'," + repr(str(binary)) + ");"
            "m=importlib.util.module_from_spec(s);sys.modules['scpn_control_rs']=m;s.loader.exec_module(m);"
        )
    if entry == "script":
        code += "import runpy,sys;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
        return [sys.executable, "-c", code, str(path)]
    code += "from " + module + " import main;raise SystemExit(main())"
    return [sys.executable, "-c", code]


def observe_command(
    tmp_path: Path, argv: list[str], *, include_repository: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run an actual public entry with temporary cwd and preserve argv/result.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned working directory and command receipt destination.
    argv : list[str]
        Complete command. A recorded runner may add custody inside its child.
    include_repository : bool
        Add the real repository to PYTHONPATH for imported validation modules.
        Scripts otherwise receive only the real src directory.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual exit status, stdout and stderr, also retained as command-N.json.

    Raises
    ------
    subprocess.TimeoutExpired, OSError
        A real command exceeds 180 seconds or cannot be started/recorded.

    Notes
    -----
    Inherited campaign IDs are removed to exercise an explicit unrecorded
    base environment. Existing source/artifact bytes and backend flags are
    never replaced. BLAS/OpenMP use one thread for these local checks.
    """
    env = dict(
        os.environ,
        PYTHONPATH=(str(ROOT) + os.pathsep if include_repository else "") + str(ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    env.pop(CAMPAIGN_ENV, None)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=180)
    name = f"command-{len(list(tmp_path.glob('command-*.json')))}.json"
    (tmp_path / name).write_text(
        json.dumps(dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return result
