# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust advisory exception tests.

"""Tests for the expiring, build-only RustSec exceptions and their configuration binding.

The graph fixture follows ``cargo metadata --format-version 1`` for this
workspace: ``lax`` depends on ``intel-mkl-src`` as a normal dependency, and
``intel-mkl-src`` reaches ``ocipkg`` only through a build edge.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tools import rust_advisory_exceptions as rae

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "rust_advisory_exceptions.py"
GRANTED = date(2026, 11, 17)
EXPIRES = date(2026, 11, 24)
IDS = ["RUSTSEC-2026-0449", "RUSTSEC-2026-0450"]


def _exception(**overrides: object) -> dict[str, object]:
    """Return a valid exception entry, with selected fields replaced."""
    item: dict[str, object] = {
        "advisory": "RUSTSEC-2026-0450",
        "package": "ocipkg",
        "scope": "build",
        "granted": GRANTED.isoformat(),
        "expires": EXPIRES.isoformat(),
        "reason": "build-dependency of intel-mkl-src only",
    }
    item.update(overrides)
    return item


def _exceptions() -> tuple[rae.AdvisoryException, ...]:
    """Return both committed-style exceptions."""
    return tuple(rae.parse_exception(_exception(advisory=advisory)) for advisory in IDS)


def _edge(pkg: str, *kinds: str | None) -> dict[str, Any]:
    """Return one resolved dependency edge with the given kinds."""
    return {"pkg": pkg, "dep_kinds": [{"kind": kind, "target": None} for kind in kinds]}


GRAPH: dict[str, Any] = {
    "packages": [
        {"id": f"{name} 1.0", "name": name} for name in ("app", "lax", "intel-mkl-src", "ocipkg", "tar", "proptest")
    ],
    "workspace_members": ["app 1.0"],
    "resolve": {
        "nodes": [
            {"id": "app 1.0", "deps": [_edge("lax 1.0", None), _edge("proptest 1.0", "dev")]},
            {"id": "lax 1.0", "deps": [_edge("intel-mkl-src 1.0", None)]},
            {"id": "intel-mkl-src 1.0", "deps": [_edge("ocipkg 1.0", "build")]},
            {"id": "ocipkg 1.0", "deps": [_edge("tar 1.0", None), _edge("intel-mkl-src 1.0", None)]},
            {"id": "tar 1.0", "deps": []},
            {"id": "proptest 1.0", "deps": [_edge("app 1.0", None)]},
        ]
    },
}


class _Runner:
    """Stand-in for ``subprocess.run`` that returns one fixed result."""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        """Store the fixed result and an empty call log."""
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record the call and return the fixed result."""
        self.calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def _files(tmp_path: Path, items: list[dict[str, object]], deny: str, audit: str) -> list[str]:
    """Write the exceptions, deny.toml and audit.toml files; return the CLI arguments."""
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(json.dumps({"schema": 1, "exceptions": items}), encoding="utf-8")
    (tmp_path / "deny.toml").write_text(deny, encoding="utf-8")
    (tmp_path / "audit.toml").write_text(audit, encoding="utf-8")
    return [
        "--exceptions",
        str(exceptions),
        "--deny",
        str(tmp_path / "deny.toml"),
        "--audit-config",
        str(tmp_path / "audit.toml"),
        "--workspace",
        str(tmp_path),
    ]


DENY = "\n".join(
    ["[advisories]", "ignore = ["]
    + [f'  {{ id = "{advisory}", reason = "build only" }},' for advisory in IDS]
    + ["]", ""]
)
AUDIT = '[advisories]\nignore = ["RUSTSEC-2026-0449", "RUSTSEC-2026-0450"]\n'


# --- exceptions ------------------------------------------------------------


def test_parse_exception_accepts_a_seven_day_build_grant() -> None:
    """A seven-day, build-scoped exception parses into its fields."""
    item = rae.parse_exception(_exception())
    assert (item.advisory, item.package, item.granted, item.expires) == (
        "RUSTSEC-2026-0450",
        "ocipkg",
        GRANTED,
        EXPIRES,
    )


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("not a mapping", "exactly the fields"),
        ({**_exception(), "extra": 1}, "exactly the fields"),
        (_exception(advisory="GHSA-aaaa-bbbb-cccc"), "not a RustSec advisory id"),
        (_exception(package=""), "names its crate and its reason"),
        (_exception(reason=" "), "names its crate and its reason"),
        (_exception(scope="dev"), "build dependencies only"),
        (_exception(expires="2026-11-17"), "not 0"),
        (_exception(expires="2026-11-25"), "not 8"),
    ],
)
def test_parse_exception_refuses_invalid_entries(item: object, message: str) -> None:
    """Shape, id, crate, reason, scope and span are all enforced."""
    with pytest.raises(rae.ExceptionsError, match=message):
        rae.parse_exception(item)


def test_load_exceptions_refuses_another_shape(tmp_path: Path) -> None:
    """Only schema 1 with an exceptions list is accepted."""
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps({"schema": 2, "exceptions": []}), encoding="utf-8")
    with pytest.raises(rae.ExceptionsError, match="schema: 1"):
        rae.load_exceptions(path)


def test_committed_exceptions_and_configuration_agree() -> None:
    """The committed exceptions are bounded and both ignore lists name exactly them."""
    items = rae.load_exceptions(ROOT / ".github" / "rust-advisory-exceptions.json")
    for item in items:
        assert 1 <= (item.expires - item.granted).days <= rae.MAX_EXCEPTION_DAYS
    ids = sorted(item.advisory for item in items)
    assert sorted(rae.deny_ignores(ROOT / "scpn-control-rs" / "deny.toml")) == ids
    assert sorted(rae.audit_ignores(ROOT / "scpn-control-rs" / ".cargo" / "audit.toml")) == ids


# --- configuration files ---------------------------------------------------


def test_deny_ignores_reads_ids_and_requires_reasons(tmp_path: Path) -> None:
    """deny.toml entries are tables with an id and a reason."""
    path = tmp_path / "deny.toml"
    path.write_text(DENY, encoding="utf-8")
    assert rae.deny_ignores(path) == IDS
    path.write_text('[licenses]\nallow = ["MIT"]\n', encoding="utf-8")
    assert rae.deny_ignores(path) == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('[advisories]\nignore = ["RUSTSEC-2026-0449"]\n', "table with an id"),
        ('[advisories]\nignore = [{ reason = "x" }]\n', "table with an id"),
        ('[advisories]\nignore = [{ id = "RUSTSEC-2026-0449" }]\n', "without a reason"),
    ],
)
def test_deny_ignores_refuses_incomplete_entries(tmp_path: Path, content: str, message: str) -> None:
    """A bare id or a missing reason is refused."""
    path = tmp_path / "deny.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(rae.ExceptionsError, match=message):
        rae.deny_ignores(path)


def test_audit_ignores_reads_a_list_of_ids(tmp_path: Path) -> None:
    """audit.toml holds a plain list of ids; a missing section means none."""
    path = tmp_path / "audit.toml"
    path.write_text(AUDIT, encoding="utf-8")
    assert rae.audit_ignores(path) == IDS
    path.write_text("[output]\nquiet = true\n", encoding="utf-8")
    assert rae.audit_ignores(path) == []


@pytest.mark.parametrize("content", ['[advisories]\nignore = "RUSTSEC-2026-0449"\n', "[advisories]\nignore = [1]\n"])
def test_audit_ignores_refuses_other_shapes(tmp_path: Path, content: str) -> None:
    """A string or a list of non-strings is refused."""
    path = tmp_path / "audit.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(rae.ExceptionsError, match="list of ids"):
        rae.audit_ignores(path)


# --- cargo metadata and reachability -----------------------------------------


def test_cargo_metadata_runs_locked_in_the_workspace(tmp_path: Path) -> None:
    """The graph comes from a locked cargo metadata run in the workspace."""
    runner = _Runner(json.dumps(GRAPH))
    assert rae.cargo_metadata(tmp_path, runner) == GRAPH
    assert runner.calls[0]["command"] == ["cargo", "metadata", "--format-version", "1", "--locked"]
    assert runner.calls[0]["cwd"] == tmp_path


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (_Runner("", 101, "error: the lock file needs to be updated"), "cargo metadata failed"),
        (_Runner("warning only"), "printed no JSON"),
        (_Runner(json.dumps({"packages": []})), "no resolved graph"),
        (_Runner(json.dumps([])), "no resolved graph"),
    ],
)
def test_cargo_metadata_refuses_unusable_results(tmp_path: Path, runner: _Runner, message: str) -> None:
    """A failed, empty or unresolved metadata run is refused."""
    with pytest.raises(rae.ExceptionsError, match=message):
        rae.cargo_metadata(tmp_path, runner)


def test_non_build_reach_stops_at_build_edges() -> None:
    """Normal and dev edges are followed; a build edge is not; cycles end."""
    assert rae.non_build_reach(GRAPH) == {"app", "lax", "intel-mkl-src", "proptest"}


def test_non_build_reach_treats_an_edge_without_kinds_as_normal() -> None:
    """An edge that reports no kind is followed like a normal dependency."""
    graph = copy.deepcopy(GRAPH)
    graph["resolve"]["nodes"][2]["deps"] = [{"pkg": "ocipkg 1.0", "dep_kinds": []}]
    assert "ocipkg" in rae.non_build_reach(graph)


def test_non_build_reach_follows_an_edge_with_build_and_normal_kinds() -> None:
    """An edge used both as a build and as a normal dependency counts as normal."""
    graph = copy.deepcopy(GRAPH)
    graph["resolve"]["nodes"][2]["deps"] = [_edge("ocipkg 1.0", "build", None)]
    assert "ocipkg" in rae.non_build_reach(graph)


# --- judging -----------------------------------------------------------------


def test_check_accepts_active_build_only_exceptions() -> None:
    """Matching ignore lists and a build-only crate pass while the grant is active."""
    assert rae.check(_exceptions(), IDS, IDS, [GRAPH], date(2026, 11, 23)) == []


def test_check_fails_from_the_day_of_expiry() -> None:
    """On the expiry date each exception fails and the lists no longer match."""
    failures = rae.check(_exceptions(), IDS, IDS, [GRAPH], EXPIRES)
    assert "RUSTSEC-2026-0449 (ocipkg): the exception expired on 2026-11-24" in failures
    assert "RUSTSEC-2026-0450 (ocipkg): the exception expired on 2026-11-24" in failures
    assert "deny.toml ignores ['RUSTSEC-2026-0449', 'RUSTSEC-2026-0450'], the active exceptions are []" in failures


def test_check_names_the_crate_of_each_expired_exception() -> None:
    """One advisory granted for two crates reports the crate whose grant expired."""
    exceptions = (
        rae.parse_exception(_exception()),
        rae.parse_exception(_exception(package="tar", expires="2026-11-20")),
    )
    failures = rae.check(exceptions, IDS[1:], IDS[1:], [], date(2026, 11, 21))
    assert failures == ["RUSTSEC-2026-0450 (tar): the exception expired on 2026-11-20"]


@pytest.mark.parametrize(
    ("deny_ids", "audit_ids", "label"),
    [
        (IDS[:1], IDS, "deny.toml"),
        (IDS, IDS + ["RUSTSEC-2026-0001"], ".cargo/audit.toml"),
        (IDS + IDS[:1], IDS, "deny.toml"),
    ],
)
def test_check_fails_an_ignore_list_that_differs(deny_ids: list[str], audit_ids: list[str], label: str) -> None:
    """A missing, extra or duplicated ignore id fails."""
    failures = rae.check(_exceptions(), deny_ids, audit_ids, [GRAPH], GRANTED)
    assert len(failures) == 1 and failures[0].startswith(f"{label} ignores ")


def test_check_fails_a_crate_reachable_outside_build_dependencies() -> None:
    """An excepted crate reached through a normal edge fails in every graph that reaches it."""
    graph = copy.deepcopy(GRAPH)
    graph["resolve"]["nodes"][0]["deps"].append(_edge("ocipkg 1.0", None))
    failures = rae.check(_exceptions(), IDS, IDS, [GRAPH, graph], GRANTED)
    assert failures == [
        "RUSTSEC-2026-0449: ocipkg is reachable outside build dependencies",
        "RUSTSEC-2026-0450: ocipkg is reachable outside build dependencies",
    ]


# --- command line ------------------------------------------------------------


def test_main_accepts_and_reports_the_exceptions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A consistent configuration exits 0 and names each accepted advisory."""
    args = _files(tmp_path, [_exception(advisory=advisory) for advisory in IDS], DENY, AUDIT)
    status = rae.main(args, runner=_Runner(json.dumps(GRAPH)), today=GRANTED)
    out = capsys.readouterr().out
    assert status == 0
    assert "RUSTSEC-2026-0449 (ocipkg) accepted for build dependencies only; it expires on 2026-11-24" in out
    assert "RUSTSEC-2026-0450 (ocipkg) accepted for build dependencies only; it expires on 2026-11-24" in out


def test_main_reports_no_exceptions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without exceptions and ignores the check passes and says so."""
    args = _files(tmp_path, [], "[advisories]\n", "[advisories]\nignore = []\n")
    assert rae.main(args, runner=_Runner(json.dumps(GRAPH))) == 0
    assert capsys.readouterr().out == "no Rust advisory exceptions\n"


def test_main_fails_on_a_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An ignore without an exception exits 1 and names the difference."""
    args = _files(tmp_path, [], DENY, "[advisories]\nignore = []\n")
    assert rae.main(args, runner=_Runner(json.dumps(GRAPH)), today=GRANTED) == 1
    assert "deny.toml ignores" in capsys.readouterr().err


def test_main_stops_on_unusable_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An invalid exceptions file or a failed metadata run exits 2."""
    args = _files(tmp_path, [{}], DENY, AUDIT)
    assert rae.main(args, runner=_Runner(json.dumps(GRAPH))) == 2
    args = _files(tmp_path, [], "[advisories]\n", "[advisories]\nignore = []\n")
    assert rae.main(args, runner=_Runner("", 101, "boom")) == 2
    assert capsys.readouterr().err.count("rust advisory exceptions: ") == 2


def test_script_entry_point_exits_two_on_a_missing_file(tmp_path: Path) -> None:
    """Run as a script, a missing exceptions file exits 2."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--exceptions",
            str(tmp_path / "absent.json"),
            "--deny",
            str(tmp_path / "deny.toml"),
            "--audit-config",
            str(tmp_path / "audit.toml"),
            "--workspace",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=ROOT,
    )
    assert completed.returncode == 2
    assert "rust advisory exceptions:" in completed.stderr
