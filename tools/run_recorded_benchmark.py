#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Recorded benchmark command runner
"""Run one benchmark command inside an immutable evidence campaign.

The command is executed as an argument vector without a shell, with the resolved
repository as its working directory and its campaign identifier in the inherited
environment. Declared artifacts are reserved before execution and sealed through
``BenchmarkRun`` afterward. Outputs may be files or directories; every declared
destination must be recreated for a zero-exit producer to be successful.

Sample/warm-up/repeat metadata are observations of command spelling, not measured
execution or numerical validation. This module supplies a Python source-checkout
runner for producers in any language; it is not a scientific benchmark itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from scpn_control.benchmark_records import CAMPAIGN_ENV, BenchmarkOutput, BenchmarkRun, load_verified_latest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_artifact(value: str, repository_root: Path) -> BenchmarkOutput:
    """Decode one role/path declaration against the selected repository.

    Parameters
    ----------
    value : str
        ROLE=PATH spelling; the first equals sign separates the role and path.
    repository_root : Path
        Resolved repository used to prefix relative destinations.

    Returns
    -------
    BenchmarkOutput
        Valid identifier role and native path, without opening the destination.

    Raises
    ------
    argparse.ArgumentTypeError
        Missing separator/path or an invalid output-role identifier.
    """
    role, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("artifact must use ROLE=PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repository_root / path
    try:
        return BenchmarkOutput(role=role, path=path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _command_measurement(command: Sequence[str]) -> dict[str, Any]:
    """Extract the last spelling of recognised producer measurement options.

    Parameters
    ----------
    command : sequence of str
        Executable argument vector, including the executable and other options.

    Returns
    -------
    dict of str to int or str
        Steps, iterations, warmup, repeats and samples; --n-bench is a samples
        alias. Inline and following-argument values are parsed as integers when
        possible. Other values, including a missing final value, remain strings.

    Notes
    -----
    These labels do not enforce positivity, sample counts or producer semantics.
    A following option can be retained as the preceding option's string value.
    Caller-supplied measurement JSON overrides these inferred labels.
    """
    measurements: dict[str, Any] = {}
    names = {
        "--steps": "steps",
        "--iterations": "iterations",
        "--warmup": "warmup",
        "--repeats": "repeats",
        "--samples": "samples",
        "--n-bench": "samples",
    }
    index = 0
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        key = names.get(option)
        if key is None:
            index += 1
            continue
        value = inline_value if separator else command[index + 1] if index + 1 < len(command) else ""
        try:
            measurements[key] = int(value)
        except ValueError:
            measurements[key] = value
        index += 1 if separator else 2
    return measurements


def _records_root(path: Path, repository_root: Path) -> Path:
    """Resolve the immutable-records root inside the selected repository.

    Parameters
    ----------
    path : Path
        Native absolute path or repository-relative custody root.
    repository_root : Path
        Repository boundary used after native path and symlink resolution.

    Returns
    -------
    Path
        Resolved custody root equal to or beneath the repository.

    Raises
    ------
    ValueError
        Resolved records root escapes the repository.
    """
    resolved = path if path.is_absolute() else repository_root / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise ValueError("records root must remain inside the repository")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Run a command and fail if its declared outputs are not freshly complete.

    Parameters
    ----------
    argv : list of str or None, optional
        Wrapper arguments followed by a command, conventionally after --.
        None selects sys.argv. Repository and records roots default to this
        source checkout and artifacts/benchmarks/records, respectively.

    Returns
    -------
    int
        Producer return code, 127 for native launch failure, or 130 for an
        interrupt caught while waiting. A zero-exit producer with missing
        declared output, or a zero-byte/empty directory result, returns one.
        Success returns zero after sealing and verifying the current latest carrier.

    Raises
    ------
    SystemExit
        Native argparse help/usage exits zero/two before campaign reservation.
    ValueError, RuntimeError, OSError
        Native record reservation, archival, finalisation or filesystem failure;
        errors outside the producer wait are not mapped to return codes.

    Notes
    -----
    --artifact is repeatable ROLE=PATH, with relative paths rooted at the selected
    repository. --measurement-json must be a JSON object and overrides inferred
    command metadata. --campaign-id and --evidence-class are passed to the record
    owner unchanged. The child inherits stdio/environment with CAMPAIGN_ENV set
    to the reserved campaign. No shell, timeout, scheduling or descendant-process
    supervisor is provided. Native subprocess.run owns direct-child waiting and
    interrupted cleanup; interruption does not guarantee immediate OS reaping.
    Failed/incomplete manifests cannot
    advance latest. Failed fresh destinations move to the invocation's
    failed-output directory before all original destinations are restored;
    restoration errors retain their recovery reservation. Record-finalisation
    errors can retain a recovery reservation.
    Direct negative POSIX child return codes are retained by this API; Python
    SystemExit maps such integers to the platform's command-line exit status.
    Command/sample metadata and successful custody do not grant production or
    scientific admission. Source HEAD metadata does not hash dirty source bytes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="Descriptive benchmark family identifier.")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository whose commit and dependency locks the run records.",
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path("artifacts/benchmarks/records"),
        help="Repository-relative immutable custody root.",
    )
    parser.add_argument("--artifact", action="append", required=True, help="ROLE=PATH output.")
    parser.add_argument("--campaign-id", help="Explicit shared campaign identifier; generated when omitted.")
    parser.add_argument("--evidence-class", default="local_regression")
    parser.add_argument(
        "--measurement-json",
        default="{}",
        help="Additional JSON object describing sample, warm-up, and repeat semantics.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a benchmark command is required after --")
    try:
        additional_measurement = json.loads(args.measurement_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--measurement-json is not valid JSON: {exc}")
    if not isinstance(additional_measurement, dict):
        parser.error("--measurement-json must decode to an object")
    measurement = {**_command_measurement(command), **additional_measurement}
    repository_root = args.repository_root.resolve()
    try:
        outputs = [_parse_artifact(value, repository_root) for value in args.artifact]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    try:
        records_root = _records_root(args.records_root, repository_root)
    except ValueError as exc:
        parser.error(str(exc))

    run = BenchmarkRun.begin(
        repository_root=repository_root,
        records_root=records_root,
        family=args.family,
        outputs=outputs,
        command=command,
        campaign_id=args.campaign_id,
        evidence_class=args.evidence_class,
        measurement=measurement,
    )
    environment = os.environ.copy()
    environment[CAMPAIGN_ENV] = run.campaign_id
    exit_code = 127
    try:
        completed = subprocess.run(command, cwd=repository_root, env=environment, check=False)
        exit_code = completed.returncode
    except OSError as exc:
        print(f"recorded benchmark could not start: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        exit_code = 130
    manifest = run.finish(exit_code=exit_code)
    print(f"benchmark record: {manifest.relative_to(repository_root).as_posix()}", file=sys.stderr)
    if exit_code == 0 and json.loads(manifest.read_text())["status"] != "succeeded":
        return 1
    if exit_code == 0:
        load_verified_latest(run.records_root, run.family)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
