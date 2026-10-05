# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native coverage executable-step inspection.

"""Inspect direct coverage commands and artifact actions in parsed CI jobs."""

from __future__ import annotations

import re
import shlex
from typing import cast

_PYTHON_UPLOAD_IF = "matrix.python-version == '3.12' && matrix.os == 'ubuntu-latest'"


def _mapping(value: object) -> dict[str, object]:
    """Return a string-keyed mapping, or an empty mapping for another shape."""
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return cast(dict[str, object], value)
    return {}


def _active(value: dict[str, object], allowed_if: tuple[str, ...] = ()) -> bool:
    """Require a non-optional job/step with a recognised execution condition."""
    condition = value.get("if")
    if isinstance(condition, str):
        condition = condition.strip().removeprefix("${{").removesuffix("}}").strip()
    return value.get("continue-on-error", False) is False and (
        condition is None or condition is True or condition == "true" or condition in allowed_if
    )


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    """Return mapped step declarations for one enabled executable job."""
    raw = job.get("steps")
    if not _active(job) or not isinstance(raw, list):
        return []
    return [_mapping(step) for step in raw]


def _shell_commands(script: str) -> list[list[str]]:
    """Read direct commands, ignoring heredoc bodies and masked/control-flow calls.

    No shell is executed. Conditional shell blocks are not admitted as direct
    coverage commands. Malformed tokenisation raises ValueError for the caller.
    """
    commands: list[list[str]] = []
    pending = ""
    heredoc: str | None = None
    for raw in script.splitlines():
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        line = pending + raw.strip()
        if line.endswith("\\"):
            pending = line[:-1] + " "
            continue
        pending = ""
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
        if not tokens:
            continue
        if tokens[0] in {
            "if",
            "for",
            "while",
            "until",
            "case",
            "function",
            "fi",
            "done",
            "esac",
            "exit",
            "return",
            "exec",
        }:
            return []
        if tokens[0] == "set" and any("+e" in token or token == "+o" for token in tokens[1:]):
            return []
        if tokens[0] in {"export", "unset", "cd", "source"} or (
            tokens[0] == "." and tokens[1:] != [".venv/bin/activate"]
        ):
            return []
        if all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token) for token in tokens):
            return []
        if "<<" in tokens:
            index = tokens.index("<<")
            if index + 1 >= len(tokens):
                raise ValueError("coverage workflow heredoc requires a delimiter")
            heredoc = tokens[index + 1].removeprefix("-")
            continue
        if any(set(token) <= set(";&|<>") for token in tokens):
            continue
        commands.append(tokens)
    if pending or heredoc is not None:
        raise ValueError("coverage workflow shell declaration is incomplete")
    return commands


def _command_rows(
    job: dict[str, object], allowed_if: tuple[str, ...] = ()
) -> list[tuple[tuple[int, int], dict[str, object], list[str]]]:
    """Collect direct commands with literal job/step/inline environment precedence."""
    rows: list[tuple[tuple[int, int], dict[str, object], list[str]]] = []
    for index, step in enumerate(_steps(job)):
        script = step.get("run")
        if not _active(step, allowed_if) or not isinstance(script, str):
            continue
        defaults = _mapping(_mapping(job.get("defaults")).get("run"))
        if step.get("working-directory", defaults.get("working-directory", ".")) != ".":
            continue
        if step.get("shell", defaults.get("shell", "bash")) not in {"bash", "sh"}:
            continue
        for ordinal, tokens in enumerate(_shell_commands(script), start=1):
            env = {**_mapping(job.get("env")), **_mapping(step.get("env"))}
            args = tokens.copy()
            while args and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", args[0]):
                key, value = args.pop(0).split("=", 1)
                env[key] = value
            rows.append(((index, ordinal), env, args))
    return rows


def _python_args(args: list[str]) -> list[str]:
    """Strip a literal Python executable prefix from a direct command."""
    return args[1:] if re.fullmatch(r"(?:.*/)?python(?:3(?:\.\d+)?)?", args[0]) else []


def _coverage_args(args: list[str]) -> list[str]:
    """Return coverage's arguments for a direct Python-module or coverage call."""
    python = _python_args(args)
    if python[:2] == ["-m", "coverage"]:
        return python[2:]
    return args[1:] if args[0] == "coverage" else []


def _python_variant(job: dict[str, object]) -> bool:
    """Require the selected Ubuntu/Python lane to exist in literal matrix axes."""
    matrix = _mapping(_mapping(job.get("strategy")).get("matrix"))
    versions, systems = matrix.get("python-version"), matrix.get("os")
    if not isinstance(versions, list) or not isinstance(systems, list):
        return False
    selected = {"python-version": "3.12", "os": "ubuntu-latest"}
    excluded = matrix.get("exclude", [])
    return (
        selected["python-version"] in versions
        and selected["os"] in systems
        and isinstance(excluded, list)
        and not any(
            bool(_mapping(row)) and all(selected.get(key) == value for key, value in _mapping(row).items())
            for row in excluded
        )
    )


def _artifact_index(
    job: dict[str, object], action: str, name: str, path: str, *, hidden: bool = False
) -> tuple[int, int] | None:
    """Find a pinned enabled upload/download with exact name/path/hidden-file semantics."""
    for index, step in enumerate(_steps(job)):
        uses = step.get("uses")
        allowed = (
            (_PYTHON_UPLOAD_IF,)
            if name == "coverage-data-python" and action == "upload" and _python_variant(job)
            else ()
        )
        if not _active(step, allowed) or not isinstance(uses, str):
            continue
        if not re.fullmatch(rf"actions/{action}-artifact@[0-9a-f]{{40}}", uses):
            continue
        options = _mapping(step.get("with"))
        if options.get("name") == name and options.get("path") == path:
            if not hidden or options.get("include-hidden-files") is True:
                return (index, 0)
    return None


def _producer(job: dict[str, object], native_tests: tuple[str, ...] = ()) -> bool:
    """Bind coverage collection, saved data and upload to one enabled producer job."""
    native = bool(native_tests)
    label = "rust" if native else "python"
    output = f"artifacts/coverage/{label}/.coverage.{label}"
    upload = _artifact_index(job, "upload", f"coverage-data-{label}", output, hidden=True)
    if upload is None:
        return False
    rows = _command_rows(job, (_PYTHON_UPLOAD_IF,) if not native and _python_variant(job) else ())
    collected: list[tuple[int, int]] = []
    moved: list[tuple[int, int]] = []
    for index, env, args in rows:
        py = _python_args(args)
        pytest = py[2:] if py[:2] == ["-m", "pytest"] else args[1:] if args[0] == "pytest" else []
        coverage = _coverage_args(args)
        if {arg.split("=", 1)[0] for arg in args} & {"--collect-only", "--co", "--no-cov"}:
            continue
        if native:
            if (
                coverage[:1] == ["run"]
                and "--source=scpn_control" in coverage
                and env.get("COVERAGE_FILE") == ".coverage.rust"
            ):
                if any(coverage[i : i + 2] == ["-m", "pytest"] for i in range(1, len(coverage))):
                    if set(native_tests) <= set(coverage):
                        collected.append(index)
        elif (
            "--cov=scpn_control" in pytest
            and [arg for arg in pytest if arg.startswith("--cov-fail-under")] == ["--cov-fail-under=100"]
            and env.get("COVERAGE_FILE", ".coverage") == ".coverage"
        ):
            collected.append(index)
        source = ".coverage.rust" if native else ".coverage"
        if args[0] in {"cp", "mv"} and args[1:] == [source, output]:
            moved.append(index)
    return any(start <= save < upload for start in collected for save in moved)


def _combined(job: dict[str, object], needs_ok: bool) -> bool:
    """Require both downloads and direct merge/XML/100-percent gate before upload."""
    if not needs_ok:
        return False
    python = _artifact_index(job, "download", "coverage-data-python", "artifacts/coverage/python")
    rust = _artifact_index(job, "download", "coverage-data-rust", "artifacts/coverage/rust")
    upload = _artifact_index(job, "upload", "coverage-report-combined", "coverage.xml")
    if python is None or rust is None or upload is None:
        return False
    merge: list[tuple[int, int]] = []
    xml: list[tuple[int, int]] = []
    gate: list[tuple[int, int]] = []
    guard: list[tuple[int, int]] = []
    for index, env, args in _command_rows(job):
        if env.get("COVERAGE_FILE", ".coverage") != ".coverage":
            continue
        coverage = _coverage_args(args)
        if coverage == ["combine", "--keep", "artifacts/coverage/python", "artifacts/coverage/rust"]:
            merge.append(index)
        if coverage == ["xml", "-o", "coverage.xml"]:
            xml.append(index)
        if coverage in (["report", "--fail-under=100"], ["report", "--fail-under", "100"]):
            gate.append(index)
        if _python_args(args) in (["tools/native_coverage_matrix.py"], ["-m", "tools.native_coverage_matrix"]):
            guard.append(index)
    return any(
        max(python, rust) < check <= combine <= report < upload and combine <= write < upload
        for check in guard
        for combine in merge
        for report in gate
        for write in xml
    )
