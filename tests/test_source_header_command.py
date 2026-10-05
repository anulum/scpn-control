# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native source-header policy and command contracts.

"""Exercise typed TOML admission and real Git/CLI source-header behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tools import check_source_headers as guard

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'schema = "scpn-control.source-header-policy.v1"\n'
REASON = 'category = "probe"\nreason = "A specific reviewed native-format exemption."\n'


def _policy_file(tmp_path: Path, text: str) -> Path:
    """Write a literal policy for public native TOML loading.

    Parameters
    ----------
    tmp_path : Path
        Physical isolated allocation outside the checkout.
    text : str
        Literal TOML, without schema or value coercion.

    Returns
    -------
    Path
        Existing policy file with exactly the supplied UTF-8 text.
    """
    path = tmp_path / "policy.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "field", ["enforced.suffixes", "enforced.names", "exemption.suffixes", "exemption.names", "exemption.paths"]
)
@pytest.mark.parametrize(
    "value",
    [
        '".py"',
        "17",
        "true",
        "[true]",
        "[3]",
        "[1.2]",
        '["valid", false]',
        "[[1]]",
        '{ item = "value" }',
        "[2026-10-04]",
    ],
)
def test_public_loader_refuses_non_string_arrays(tmp_path: Path, field: str, value: str) -> None:
    """Scalar collections and non-string members cannot alter match scope.

    Parameters
    ----------
    tmp_path : Path
        Isolated native policy allocation.
    field : str
        Enforced or exemption collection under test.
    value : str
        Actual TOML scalar, array or inline-table value with invalid type.
    """
    table, key = field.split(".")
    text = SCHEMA + "[enforced]\n"
    if table == "exemption":
        text += "[[exemptions]]\n" + REASON
    text += f"{key} = {value}\n"
    with pytest.raises(ValueError, match="TOML array of strings"):
        guard.load_policy(_policy_file(tmp_path, text))


@pytest.mark.parametrize("value", ['""', "17", "true", '{ category = "probe" }'])
def test_public_loader_refuses_non_array_exemptions(tmp_path: Path, value: str) -> None:
    """Top-level exemptions must be an actual TOML array of tables.

    Parameters
    ----------
    tmp_path : Path
        Isolated policy allocation.
    value : str
        Wrong native collection shape at the top level.
    """
    with pytest.raises(ValueError, match="array of tables"):
        guard.load_policy(_policy_file(tmp_path, SCHEMA + f"exemptions = {value}\n[enforced]\n"))


def test_public_loader_preserves_exact_valid_strings_and_empty_defaults(tmp_path: Path) -> None:
    """Valid arrays deduplicate, suffixes case-fold and omitted fields stay empty.

    Parameters
    ----------
    tmp_path : Path
        Isolated policy allocation.
    """
    empty = guard.load_policy(_policy_file(tmp_path, SCHEMA + "[enforced]\n"))
    assert empty.enforced_suffixes == empty.enforced_names == frozenset()
    assert empty.exemptions == ()
    text = (
        SCHEMA
        + '[enforced]\nsuffixes = [".PY", ".py"]\nnames = ["Makefile", "Makefile"]\n'
        + "[[exemptions]]\n"
        + REASON
        + 'suffixes = [".MD", ".md"]\nnames = ["LICENSE"]\npaths = ["captured/item.data"]\n'
    )
    policy = guard.load_policy(_policy_file(tmp_path, text))
    assert policy.enforced_suffixes == frozenset({".py"})
    assert policy.enforced_names == frozenset({"Makefile"})
    assert policy.exemptions[0].suffixes == frozenset({".md"})
    assert policy.exemptions[0].names == frozenset({"LICENSE"})
    assert policy.exemptions[0].paths == frozenset({"captured/item.data"})
    assert guard.classify(Path("PROBE.PY"), policy) == ("enforced", "")
    assert guard.classify(Path("nested/Makefile"), policy) == ("enforced", "")
    assert guard.classify(Path("nested/LICENSE"), policy) == ("exempt", "probe")
    assert guard.classify(Path("nested/license"), policy) == ("unclassified", "")


@pytest.mark.parametrize(
    "declarations",
    [
        "[[exemptions]]\n"
        + REASON
        + 'paths = ["capture/item.data"]\n[[exemptions]]\n'
        + REASON
        + 'paths = ["capture/item.data"]\n',
        "[[exemptions]]\n" + REASON + 'paths = ["capture/probe.PY"]\n',
        "[[exemptions]]\n" + REASON + 'paths = ["capture/Makefile"]\n',
        "[[exemptions]]\n" + REASON + 'paths = ["capture\\\\item.data"]\n',
    ],
)
def test_public_loader_rejects_remaining_exact_path_conflicts(tmp_path: Path, declarations: str) -> None:
    """Repeated, suffix/name-overlapping and backslash paths are refused.

    Parameters
    ----------
    tmp_path : Path
        Isolated policy allocation.
    declarations : str
        Actual conflicting or non-POSIX exemption tables.
    """
    text = SCHEMA + '[enforced]\nsuffixes = [".py"]\nnames = ["Makefile"]\n' + declarations
    with pytest.raises(ValueError, match="overlapping|exact relative POSIX"):
        guard.load_policy(_policy_file(tmp_path, text))


def test_native_frozen_values_do_not_revalidate_construction() -> None:
    """Dataclasses freeze fields without adding hidden coercion or policy admission."""
    finding = guard.Finding("probe.py", "header_mismatch", "expected exact seven-line semantics")
    pytest.raises(FrozenInstanceError, setattr, finding, "path", "changed.py")
    exemption = guard.Exemption("probe", "short", frozenset(), frozenset())
    policy = guard.Policy("caller-supplied", frozenset(), frozenset(), (exemption,))
    assert exemption.paths == frozenset()
    assert guard.classify(Path("probe.py"), policy) == ("unclassified", "")
    assert guard.FindingPayload(path="p", category="c", detail="d") == {"path": "p", "category": "c", "detail": "d"}
    result = guard.AuditResult(
        schema="caller",
        policy_schema="caller",
        source_head="",
        passed=True,
        classifications={},
        exemptions={},
        findings=[],
    )
    assert isinstance(result, dict) and result["schema"] == "caller"


@pytest.fixture()
def native_repository(tmp_path: Path) -> Path:
    """Create a committed Git repository and byte-identical maintained CLI.

    Parameters
    ----------
    tmp_path : Path
        Physical allocation for the CLI and real tracked source files.

    Returns
    -------
    Path
        Root whose valid Python/Lean/HTML/Rust headers and exempt README are
        tracked; tool and policy copies remain untracked.
    """
    root = tmp_path / "source-header-policy"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "check_source_headers.py").write_bytes((ROOT / "tools/check_source_headers.py").read_bytes())
    (tools / "source_header_policy.toml").write_bytes((ROOT / "tools/source_header_policy.toml").read_bytes())
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    for name in ["probe.py", "probe.lean", "probe.html", "probe.rs"]:
        relative = Path(name)
        (root / relative).write_text(
            "\n".join(guard.expected_header(relative, "Native command fixture.")) + "\n", encoding="utf-8"
        )
    (root / "README.md").write_text("Native command fixture.\n", encoding="utf-8")
    for argv in [
        ["git", "init", "-q"],
        ["git", "add", "probe.py", "probe.lean", "probe.html", "probe.rs", "README.md"],
        [
            "git",
            "-c",
            "user.name=Source Header Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "native source fixture",
        ],
    ]:
        subprocess.run(argv, cwd=root, env=env, check=True, capture_output=True)
    return root


def _command(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Launch the actual byte-identical script from outside its repository.

    Parameters
    ----------
    root : Path
        Physical fixture repository with the maintained CLI copy.
    args : list[str]
        Native command options, retaining actual argparse behavior.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual exit status and captured UTF-8 stdout/stderr. An optional
        coverage environment variable instruments the same subprocess only.
    """
    script = root / "tools/check_source_headers.py"
    argv = [sys.executable]
    rc = os.environ.get("SCPN_SOURCE_HEADER_COVERAGE_RC")
    if rc:
        argv += ["-m", "coverage", "run", "--rcfile=" + rc, "--parallel-mode"]
    argv += [str(script), *args]
    return subprocess.run(argv, cwd=root.parent, check=False, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("mode", ["text", "json", "relative"])
def test_actual_source_header_command_success(native_repository: Path, mode: str) -> None:
    """Defaults and relative root/policy options preserve real Git counts and HEAD.

    Parameters
    ----------
    native_repository : Path
        Byte-identical CLI and valid tracked sources.
    mode : str
        Text defaults, JSON defaults or explicit caller-relative options.
    """
    args = [] if mode == "text" else ["--json"]
    if mode == "relative":
        args += [
            "--root",
            native_repository.name,
            "--policy",
            native_repository.name + "/tools/source_header_policy.toml",
        ]
    result = _command(native_repository, args)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    if mode == "text":
        assert result.stdout == "Source-header policy passed\n"
    else:
        payload = json.loads(result.stdout)
        assert payload["passed"] is True and payload["findings"] == []
        assert payload["classifications"] == {"enforced": 4, "exempt": 1}
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=native_repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        assert payload["source_head"] == head


@pytest.mark.parametrize("defect", ["identity", "non_utf8", "unclassified"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_actual_source_header_command_finding(native_repository: Path, defect: str, json_mode: bool) -> None:
    """Real tracked content or format defects produce exit one and honest findings.

    Parameters
    ----------
    native_repository : Path
        Committed physical fixture repository.
    defect : str
        Native source mutation or newly tracked unclassified format.
    json_mode : bool
        Select complete JSON versus authored text diagnostics.
    """
    if defect == "identity":
        target = native_repository / "probe.py"
        target.write_text(target.read_text().replace("Šotek", "Sotek", 1))
        category = "header_mismatch"
    elif defect == "non_utf8":
        (native_repository / "probe.py").write_bytes(b"\xff\xfe")
        category = "non_utf8_enforced"
    else:
        (native_repository / "probe.unknown").write_text("unclassified format\n")
        subprocess.run(["git", "add", "probe.unknown"], cwd=native_repository, check=True, capture_output=True)
        category = "unclassified"
    result = _command(native_repository, ["--json"] if json_mode else [])
    assert result.returncode == 1
    if json_mode:
        payload = json.loads(result.stdout)
        assert payload["passed"] is False and any(row["category"] == category for row in payload["findings"])
        assert result.stderr == ""
    else:
        assert category in result.stderr and result.stdout == ""


@pytest.mark.parametrize(
    "defect",
    [
        "scalar_collection",
        "wrong_schema",
        "malformed_toml",
        "missing_policy",
        "missing_source",
        "nonrepository",
        "unborn_head",
    ],
)
def test_actual_source_header_command_errors(native_repository: Path, defect: str) -> None:
    """Policy, native filesystem and real Git errors return two without success JSON.

    Parameters
    ----------
    native_repository : Path
        Physical CLI and Git fixture, without mocked processes.
    defect : str
        Real policy, source, repository or HEAD failure.
    """
    policy = native_repository / "tools/source_header_policy.toml"
    args = ["--json"]
    if defect == "scalar_collection":
        policy.write_text(SCHEMA + '[enforced]\nsuffixes = ".py"\n')
    elif defect == "wrong_schema":
        policy.write_text('schema = "wrong"\n')
    elif defect == "malformed_toml":
        policy.write_text("schema = [\n")
    elif defect == "missing_policy":
        policy.unlink()
    elif defect == "missing_source":
        (native_repository / "probe.py").unlink()
    elif defect == "nonrepository":
        empty = native_repository.parent / "without-git"
        empty.mkdir()
        args += ["--root", str(empty)]
    else:
        empty = native_repository.parent / "unborn"
        empty.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=empty, check=True, capture_output=True)
        args += ["--root", str(empty)]
    result = _command(native_repository, args)
    assert result.returncode == 2
    assert result.stdout == "" and result.stderr.startswith("source-header policy error:")
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("args,exit_code", [(["--help"], 0), (["--unknown"], 2)])
def test_actual_source_header_command_argparse(native_repository: Path, args: list[str], exit_code: int) -> None:
    """Native help and invalid arguments exit before policy or repository reads.

    Parameters
    ----------
    native_repository : Path
        Actual maintained CLI in an isolated repository.
    args : list[str]
        Real help or invalid command arguments.
    exit_code : int
        Expected native argparse status.
    """
    policy = native_repository / "tools/source_header_policy.toml"
    policy.unlink()
    result = _command(native_repository, args)
    assert result.returncode == exit_code
    assert "usage:" in result.stdout + result.stderr
    assert "source-header policy error:" not in result.stderr
