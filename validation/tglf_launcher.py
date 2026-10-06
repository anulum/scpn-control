# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Standalone TGLF provider launcher.

"""Execute GACODE decks and retain a hashed receipt beside the provider output.

The launcher needs an installed GACODE provider, which the hosted test
environments do not have. It therefore lives with the validation commands and
not in the measured package. The package keeps the reader of the retained
output, ``scpn_control.core.tglf_flux``, which needs no provider.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from scpn_control.core.tglf_flux import (
    TGLFFluxError,
    TGLFFluxResult,
    capture_tglf_outputs,
    parse_captured_tglf_fluxes,
)


class TGLFFluxSolver:
    """Run an existing GACODE input deck in a fresh directory on POSIX.

    Parameters
    ----------
    work_dir : Path
        Caller-owned parent for persistent per-execution evidence directories.
    binary : str
        GACODE standalone launcher name or absolute path.
    environment : mapping or None
        Child-only overrides for the existing process environment, including
        GACODE_ROOT and dependency paths if needed. Values are not logged.

    Notes
    -----
    Input is the provider's key=value deck, not GKLocalParams or a namelist.
    The provider owns defaults and input validation. This class deliberately
    returns normalised signed fluxes rather than the coefficient-only GKOutput.
    Runtime coupling must specify reference units and a conservation contract.
    """

    def __init__(
        self,
        work_dir: Path,
        binary: str = "tglf",
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Store evidence paths and copy child environment overrides without launching a provider."""
        self.work_dir = work_dir
        self.binary = binary
        self.environment = dict(environment or {})

    def run(self, input_deck: Path, *, timeout_s: float = 30.0) -> TGLFFluxResult:
        """Execute a copied input deck and retain logs and a hashed execution receipt.

        Parameters
        ----------
        input_deck : Path
            Existing standalone input.tglf-format file. Its bytes are copied;
            neighbouring outputs are never imported.
        timeout_s : float
            Positive finite child-process timeout in seconds; booleans are
            rejected. On timeout the
            owned POSIX process group is killed and its leader reaped. Cleanup
            is also attempted after success, exceptions and cancellation, with
            a separate two-second leader-reaping limit. Descendants that create
            another session/group are outside that termination boundary.

        Returns
        -------
        TGLFFluxResult
            Validated raw signed output from this execution directory.

        Raises
        ------
        ValueError
            Timeout is boolean or is not positive and finite.
        TGLFFluxError
            Unsupported OS, unavailable executable, execution failure, timeout
            or incomplete/changed output. The failed run directory and logs remain.
            Parsing and output hashes use the same captured bytes; retained
            files are checked before publishing success. Later edits are
            detected by read_tglf_fluxes, not prevented by this API.
            Launcher identity is checked before/after execution; it does not
            authenticate the launcher's dependent executables.
        BaseException
            Original wait/cancellation failure, after bounded cleanup. Cleanup
            failures are recorded and attached as exception notes.
        OSError
            Input or evidence directory cannot be read or written.
        """
        if isinstance(timeout_s, (bool, np.bool_)) or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if os.name != "posix":
            raise TGLFFluxError("Standalone TGLF process-group execution requires POSIX")
        env = dict(os.environ)
        env.update(self.environment)
        binary = shutil.which(self.binary, path=env.get("PATH"))
        if binary is None:
            raise TGLFFluxError(f"TGLF launcher is unavailable: {self.binary}")
        binary = str(Path(binary).resolve())
        deck = input_deck.read_bytes()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="tglf-", dir=self.work_dir)).resolve()
        (run_dir / "input.tglf").write_bytes(deck)
        argv = [binary, "-e", "."]
        launcher_bytes = Path(binary).read_bytes()
        failure: BaseException | None = None
        cleanup_errors: list[str] = []
        process: subprocess.Popen[bytes] | None = None
        # Regular files cannot leave communicate() waiting for pipes retained by
        # descendants that have escaped the owned POSIX process group.
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    argv, cwd=run_dir, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                )
                process.wait(timeout=timeout_s)
            except BaseException as exc:
                failure = exc
            finally:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except BaseException as exc:
                        cleanup_errors.append(f"process-group cleanup: {type(exc).__name__}: {exc}")
                        if failure is None:
                            failure = exc
                    try:
                        process.wait(timeout=2.0)
                    except BaseException as exc:
                        cleanup_errors.append(f"leader reaping: {type(exc).__name__}: {exc}")
                        if failure is None:
                            failure = exc
        timed_out = isinstance(failure, subprocess.TimeoutExpired)
        receipt: dict[str, object] = {
            "argv": argv,
            "input_sha256": hashlib.sha256(deck).hexdigest(),
            "launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
            "launcher_unchanged": False,
            "output_validated": False,
            "pid": None if process is None else process.pid,
            "returncode": None if process is None else process.returncode,
            "timed_out": timed_out,
            "failure_type": None if failure is None else type(failure).__name__,
            "cleanup_errors": cleanup_errors,
            "output_sha256": {},
        }
        try:
            (run_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        except OSError as exc:
            if failure is None:
                raise
            failure.add_note(f"Cannot write TGLF failure receipt: {exc}")
        if failure is not None:
            for error in cleanup_errors:
                failure.add_note(error)
            if timed_out:
                raise TGLFFluxError(f"TGLF execution failed (timeout=True) in {run_dir}") from failure
            raise failure
        if process is None or process.returncode != 0:
            raise TGLFFluxError(f"TGLF execution failed (exit={receipt['returncode']}) in {run_dir}")
        captured = capture_tglf_outputs(run_dir)
        receipt["output_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
        (run_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result = parse_captured_tglf_fluxes(run_dir, captured)
        if (run_dir / "input.tglf").read_bytes() != deck or Path(binary).read_bytes() != launcher_bytes:
            raise TGLFFluxError("TGLF input or launcher changed during execution")
        receipt["launcher_unchanged"] = True
        receipt["output_validated"] = True
        (run_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return result
