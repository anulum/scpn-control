# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native static test-linkage contracts.

"""Exercise source ownership through public APIs and byte-identical real commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_test_module_linkage import collect_source_modules, collect_unlinked_modules, load_allowlist, main

GUARD = Path(__file__).resolve().parents[2] / "tools/check_test_module_linkage.py"
PREFIX = "from scpn_control.owner import api\n"


def _tree(tmp_path: Path, body: str = "", files: dict[str, str] | None = None) -> Path:
    """Create real source/test/allowlist files and the unchanged native command."""
    root = tmp_path / "module-linkage-root"
    (root / "tools").mkdir(parents=True)
    (root / "src/scpn_control").mkdir(parents=True)
    (root / "tests").mkdir()
    shutil.copyfile(GUARD, root / "tools/check_test_module_linkage.py")
    (root / "src/scpn_control/__init__.py").write_text("", encoding="utf-8")
    (root / "src/scpn_control/owner.py").write_text("def api(): return 'real'\n", encoding="utf-8")
    (root / "tests/test_calls.py").write_text(body, encoding="utf-8")
    (root / "tools/untested_module_allowlist.json").write_text('{"allowlisted_modules": []}\n', encoding="utf-8")
    for name, text in (files or {}).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def _cli(root: Path, *flags: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the actual fixture command, instrumenting it only when explicitly requested."""
    argv = [sys.executable]
    config = os.environ.get("SCPN_LINKAGE_COVERAGE_RC")
    if config:
        argv += ["-m", "coverage", "run", "--parallel-mode", "--rcfile=" + config]
    argv += [str(root / "tools/check_test_module_linkage.py"), *flags]
    result = subprocess.run(argv, cwd=cwd or root, text=True, capture_output=True, check=False, timeout=30)
    if config:
        observation = root / ("command-" + str(len(list(root.glob("command-*.json")))) + ".json")
        observation.write_text(
            json.dumps(
                {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            ),
            encoding="utf-8",
        )
    return result


@pytest.mark.parametrize(
    "body,linked",
    [
        (PREFIX + "def test_call(): api()\n", True),
        ("import scpn_control.owner\ndef test_call(): scpn_control.owner.api()\n", True),
        ("import scpn_control.owner as owner\ndef test_call(): owner.api()\n", True),
        ("from scpn_control import owner\ndef test_call(): owner.api()\n", True),
        ("from scpn_control.owner import api as call\ndef test_call(): assert call\n", True),
        (PREFIX + "def test_call(): assert api() == 'real'\n", True),
        (PREFIX + "def unused(): api()\ndef test_idle(): pass\n", False),
        (PREFIX + "def test_idle():\n    def unused(): api()\n    assert True\n", False),
        (PREFIX + "api = lambda: 'local'\ndef test_call(): assert api() == 'local'\n", False),
        (PREFIX + "def api(): return 'local'\ndef test_call(): api()\n", False),
        (PREFIX + "del api\ndef test_call(): api()\n", False),
        (PREFIX + "class api: pass\ndef test_call(): api()\n", False),
        (PREFIX + "def test_call(api): api()\n", False),
        (PREFIX + "def test_call(*api): api()\n", False),
        (PREFIX + "def test_call(**api): api()\n", False),
        (PREFIX + "def test_call():\n    api = lambda: 'local'\n    def helper(): api()\n    helper()\n", False),
        (PREFIX + "def helper(api):\n    def nested(): api()\n    nested()\ndef test_call(): helper(None)\n", False),
        ("def test_call():\n    from scpn_control.owner import api\n    def helper(): api()\n    helper()\n", True),
        (PREFIX + "def helper(): api()\ndef test_call(): helper(); helper()\n", True),
        (PREFIX + "async def helper(): api()\nasync def test_call(): await helper()\n", True),
        (PREFIX + "def helper(): api()\ndef test_call():\n    helper = lambda: None\n    helper()\n", False),
        (PREFIX + "def helper(): api()\nhelper = lambda: None\ndef test_call(): helper()\n", False),
        (PREFIX + "def test_call():\n    def helper(): api()\n    helper = lambda: None\n    helper()\n", False),
        (PREFIX + "def test_call():\n    for api in []: api()\n", False),
        (PREFIX + "def test_call(): assert [api() for api in []]\n", False),
        (PREFIX + "def test_call():\n    try: api()\n    except Exception as api: pass\n", False),
        (PREFIX + "def test_call():\n    try: api()\n    except Exception: pass\n", True),
        (PREFIX + "def test_call():\n    match None:\n        case api: api()\n", False),
        (PREFIX + "def test_call():\n    match []:\n        case [*api]: api()\n", False),
        (PREFIX + "def test_call():\n    match {}:\n        case {**api}: api()\n", False),
        (PREFIX + "def test_call():\n    match 1:\n        case 1: api()\n", True),
        (PREFIX + "def test_call():\n    global api\n    api()\n", False),
        (
            PREFIX
            + "def test_call():\n    def helper():\n        nonlocal api\n        api()\n    api = lambda: None\n    helper()\n",
            False,
        ),
        (PREFIX + "def test_call(): assert (lambda api: api())(None)\n", False),
        (PREFIX + "def test_call(): api().run()\n", True),
        ("def test_call(): factory().api()\n", False),
        (
            "def test_call():\n    import scpn_control.owner as api\n    from pathlib import Path as api\n    api()\n",
            False,
        ),
        (
            "def test_call():\n    from scpn_control.owner import api\n    from scpn_control.owner import api\n    api()\n",
            True,
        ),
        ("def test_call():\n    from . import api\n    api()\n", False),
        (PREFIX + "class TestCalls:\n    def helper(self): api()\n    def test_call(self): self.helper()\n", True),
        (
            PREFIX
            + "class TestCalls:\n    async def helper(self): api()\n    async def test_call(self): await self.helper()\n",
            True,
        ),
        (PREFIX + "class TestCalls:\n    def helper(cls): api()\n    def test_call(cls): cls.helper()\n", True),
        (
            PREFIX
            + "class TestCalls:\n    def helper(self): api()\n    def test_call(self): self = None; self.helper()\n",
            False,
        ),
        (PREFIX + "class TestCalls:\n    def helper(self): api()\n    def test_call(self): self.missing()\n", False),
        (PREFIX + "class TestCalls:\n    def helper(self): api()\n    def test_call(other): self.helper()\n", False),
        (PREFIX + "class Other:\n    def test_call(self): api()\n", False),
        (PREFIX + "class TestIdle:\n    def helper(self): api()\n", False),
        (PREFIX + "class TestEmpty: pass\n", False),
        (
            PREFIX + "def helper(): api()\ndef test_call():\n    try: pass\n    except Exception as helper: helper()\n",
            False,
        ),
        ("# scpn_control.owner\n'decoy api()'\ndef test_idle(): pass\n", False),
    ],
)
def test_real_lexical_scope_and_reachability(tmp_path: Path, body: str, linked: bool) -> None:
    """The real AST reader follows lexical imports and refuses shadowed or unused calls."""
    root = _tree(tmp_path, body)
    source, tests = root / "src/scpn_control", root / "tests"
    expected = [] if linked else [(source / "owner.py").as_posix()]
    assert collect_unlinked_modules(source_root=source, test_root=tests) == expected
    result = _cli(root)
    assert result.returncode == (0 if linked else 1), result.stdout + result.stderr
    assert f"Unexpected modules: {0 if linked else 1}" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize(
    "facade,body,linked",
    [
        ("from ..owner import api\n", "from scpn_control.facade import api\ndef test_call(): api()\n", True),
        (
            "from ..owner import api as export\n",
            "from scpn_control.facade import export\ndef test_call(): export()\n",
            True,
        ),
        (
            "def unused():\n    from ..owner import api\n",
            "from scpn_control.facade import api\ndef test_call(): api()\n",
            False,
        ),
        (
            "from ..owner import api\napi = lambda: None\n",
            "from scpn_control.facade import api\ndef test_call(): api()\n",
            False,
        ),
        (
            "from ..owner import api\ndef api(): pass\n",
            "from scpn_control.facade import api\ndef test_call(): api()\n",
            False,
        ),
        ("from . import api\n", "from scpn_control.facade import api\ndef test_call(): api()\n", False),
        ("", "from scpn_control.facade import absent\ndef test_call(): absent()\n", False),
    ],
)
def test_real_facade_owner_resolution(tmp_path: Path, facade: str, body: str, linked: bool) -> None:
    """Package re-exports use visible bindings and terminate self-referential facade cycles."""
    root = _tree(tmp_path, body, {"src/scpn_control/facade/__init__.py": facade})
    result = _cli(root)
    assert result.returncode == (0 if linked else 1), result.stdout + result.stderr


def test_root_facade_and_nested_api_attribute(tmp_path: Path) -> None:
    """A visible package-root re-export resolves to its real implementation owner."""
    root = _tree(
        tmp_path,
        "from scpn_control import exported\ndef test_call(): exported.method()\n",
        {"src/scpn_control/__init__.py": "from .owner import api as exported\n"},
    )
    assert _cli(root).returncode == 0


@pytest.mark.parametrize(
    "files,body,missing",
    [
        (
            {"facade.py": "from .owner import api\nfrom .sibling import unused\n"},
            "from scpn_control.facade import api\ndef test_export(): assert api() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"facade.py": "from scpn_control.owner import api as exported\n"},
            "from scpn_control.facade import exported as call\ndef test_export(): assert call() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"facade.py": "import scpn_control.owner as implementation\n"},
            "import scpn_control.facade as facade\ndef test_export(): assert facade.implementation.api() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"facade.py": "from . import owner as implementation\n"},
            "from scpn_control.facade import implementation\ndef test_export(): assert implementation.api() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"facade.py": "from . import owner as implementation\n"},
            "from scpn_control.facade import implementation\ndef test_export(): assert implementation\n",
            ["sibling.py"],
        ),
        (
            {"facade.py": "from .owner import api\nfrom .owner import api as again\n"},
            "from scpn_control.facade import api, again\ndef test_export(): assert api() == again() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"middle.py": "from .owner import api\n", "facade.py": "from .middle import api as exported\n"},
            "from scpn_control.facade import exported\ndef test_export(): assert exported() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"__init__.py": "from .facade import exported\n", "facade.py": "from .owner import api as exported\n"},
            "from scpn_control import exported\ndef test_export(): assert exported() == 'real'\n",
            ["sibling.py"],
        ),
        (
            {"facade.py": "from .owner import api\napi = lambda: 'local'\n"},
            "from scpn_control.facade import api\ndef test_export(): assert api() == 'local'\n",
            ["owner.py", "sibling.py"],
        ),
        (
            {"facade.py": "from .owner import api\ndef api(): return 'local'\n"},
            "from scpn_control.facade import api\ndef test_export(): assert api() == 'local'\n",
            ["owner.py", "sibling.py"],
        ),
        (
            {"facade.py": "def unused_helper():\n    from .owner import api\ndef api(): return 'local'\n"},
            "from scpn_control.facade import api\ndef test_export(): assert api() == 'local'\n",
            ["owner.py", "sibling.py"],
        ),
        (
            {"facade.py": "from .owner import api\nfrom pathlib import Path as api\n"},
            "from scpn_control.facade import api\ndef test_export(): assert str(api('local')) == 'local'\n",
            ["owner.py", "sibling.py"],
        ),
        (
            {"facade.py": "from pathlib import Path as api\n"},
            "from scpn_control.facade import api\ndef test_export(): assert str(api('local')) == 'local'\n",
            ["owner.py", "sibling.py"],
        ),
        (
            {"facade.py": "from .owner import api\nfrom .sibling import unused\n"},
            "from scpn_control.facade import api\ndef test_idle(): assert 2 + 2 == 4\n",
            ["facade.py", "owner.py", "sibling.py"],
        ),
    ],
)
def test_executed_module_exports_link_only_selected_owners(
    tmp_path: Path, files: dict[str, str], body: str, missing: list[str]
) -> None:
    """Execute real exports/decoys, then inspect the selected alias and untouched sibling."""
    source_files = {"src/scpn_control/" + name: value for name, value in files.items()}
    source_files["src/scpn_control/sibling.py"] = "def unused(): return 'sibling'\n"
    root = _tree(tmp_path, body, source_files)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.py")}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", str(root / "tests")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    expected = [(root / "src/scpn_control" / name).as_posix() for name in missing]
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == expected
    result = _cli(root)
    assert result.returncode == 1 and not result.stderr
    observed = [line.removeprefix("  - ") for line in result.stdout.splitlines() if line.startswith("  - ")]
    assert observed == ["src/scpn_control/" + name for name in missing]
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.py")} == before


@pytest.mark.parametrize(
    "facade,missing",
    [
        ("from .middle import api\n", ["owner.py"]),
        ("from .facade.api import suffix as api\n", ["middle.py", "owner.py"]),
    ],
)
def test_reexport_cycle_terminates_without_crediting_unrelated_owner(
    tmp_path: Path, facade: str, missing: list[str]
) -> None:
    """Inspect a cyclic file export graph without importing it or inventing a leaf link."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import api\ndef test_call(): api()\n",
        {
            "src/scpn_control/facade.py": facade,
            "src/scpn_control/middle.py": "from .facade import api\n",
        },
    )
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == [
        (root / "src/scpn_control" / name).as_posix() for name in missing
    ]
    result = _cli(root)
    assert result.returncode == 1 and f"Unexpected modules: {len(missing)}" in result.stdout and not result.stderr


@pytest.mark.parametrize("payload,error", [(b"def api(:\n", SyntaxError), (b"\xff", UnicodeDecodeError)])
def test_traversed_file_export_parse_errors_propagate(tmp_path: Path, payload: bytes, error: type[Exception]) -> None:
    """Refuse native syntax/decode errors when a referenced file's exports are inspected."""
    root = _tree(tmp_path, PREFIX + "def test_call(): api()\n")
    owner = root / "src/scpn_control/owner.py"
    owner.write_bytes(payload)
    assert collect_source_modules(root / "src/scpn_control") == [owner]
    with pytest.raises(error):
        collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests")
    result = _cli(root)
    assert result.returncode == 1 and error.__name__ in result.stderr and not result.stdout
    assert owner.read_bytes() == payload


def test_unreferenced_file_body_is_not_parsed(tmp_path: Path) -> None:
    """An unreferenced malformed implementation remains an inventoried missing owner."""
    root = _tree(tmp_path, PREFIX + "def test_call(): assert api() == 'real'\n")
    invalid = root / "src/scpn_control/unreferenced.py"
    invalid.write_bytes(b"\xff")
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == [
        invalid.as_posix()
    ]
    result = _cli(root)
    assert result.returncode == 1 and "Unexpected modules: 1" in result.stdout and not result.stderr


@pytest.mark.parametrize(
    "facade,missing_base",
    [
        ("from .owner import Base\nclass API(Base): pass\n", False),
        ("import scpn_control.owner as leaf\nclass API(leaf.Base): pass\n", False),
        ("from .owner import Base as Imported\nclass Local(Imported): pass\nclass API(Local): pass\n", False),
        (
            "from .owner import Base\nfrom .sibling import Other\nclass API(Base): pass\nclass Uncalled(Other): pass\n",
            False,
        ),
        ("from .owner import Base\nclass Uncalled(Base): pass\nclass API:\n    value=7\n", True),
        (
            "from .owner import Base\nfrom types import SimpleNamespace\nclass API(Base): pass\nAPI=lambda: SimpleNamespace(value=7)\n",
            True,
        ),
        (
            "from .owner import Base\nfrom types import SimpleNamespace\nclass API(Base): pass\ndef API(): return SimpleNamespace(value=7)\n",
            True,
        ),
        ("from .owner import Base\nclass API(Base): pass\nclass API:\n    value=7\n", True),
        (
            "from .owner import Base\nfrom types import SimpleNamespace as API\nclass API(Base): pass\n",
            True,
        ),
        ("from .owner import Base\nfrom math import *\nclass API(Base): pass\n", True),
        (
            "from .owner import Base\ndef replace(cls):\n    class Local: value=7\n    return Local\n@replace\nclass API(Base): pass\n",
            True,
        ),
        (
            "from .owner import Base\nclass Meta(type):\n    def __new__(cls,name,bases,ns): return type(name,(),{'value':7})\nclass API(Base,metaclass=Meta): pass\n",
            True,
        ),
        ("from .owner import Base\ndef base_factory(): return Base\nclass API(base_factory()): pass\n", True),
        ("from .owner import Base\nclass API(object):\n    value=7\n", True),
    ],
)
def test_executed_class_bases_refuse_unused_or_ambiguous_declarations(
    tmp_path: Path, facade: str, missing_base: bool
) -> None:
    """Execute class/decoy constructors and follow only supported named base identities."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import API\ndef test_constructor(): assert API().value == 7\n",
        {
            "src/scpn_control/owner.py": "class Base:\n    def __init__(self): self.value=7\n",
            "src/scpn_control/sibling.py": "class Other: pass\n",
            "src/scpn_control/facade.py": facade,
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", str(root / "tests")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    missing = (["owner.py"] if missing_base else []) + ["sibling.py"]
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == [
        (root / "src/scpn_control" / name).as_posix() for name in missing
    ]
    result = _cli(root)
    assert result.returncode == 1 and f"Unexpected modules: {len(missing)}" in result.stdout and not result.stderr
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")} == before


def test_named_class_inheritance_cycle_terminates_without_runtime_import(tmp_path: Path) -> None:
    """Bound an inspected circular base graph while preserving the unreferenced leaf."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import API\ndef test_call(): API()\n",
        {
            "src/scpn_control/facade.py": "from .middle import Parent\nclass API(Parent): pass\n",
            "src/scpn_control/middle.py": "from .facade import API\nclass Parent(API): pass\n",
        },
    )
    result = _cli(root)
    assert result.returncode == 1 and "Unexpected modules: 1" in result.stdout and not result.stderr
    assert "src/scpn_control/owner.py" in result.stdout


def test_source_inventory_and_sorted_missing_owners(tmp_path: Path) -> None:
    """Inventory ignores initializers and returns stable sorted native file paths."""
    root = _tree(tmp_path, files={"src/scpn_control/nested/z.py": "pass\n"})
    source = root / "src/scpn_control"
    assert collect_source_modules(source) == [source / "nested/z.py", source / "owner.py"]
    assert collect_unlinked_modules(source_root=source, test_root=root / "tests") == [
        (source / "nested/z.py").as_posix(),
        (source / "owner.py").as_posix(),
    ]


@pytest.mark.parametrize("which", ["source", "tests"])
@pytest.mark.parametrize("exists_as_file", [False, True])
def test_root_must_be_a_real_directory(tmp_path: Path, which: str, exists_as_file: bool) -> None:
    """Missing roots and ordinary files cannot produce an empty successful inspection."""
    root = _tree(tmp_path)
    invalid = root / "invalid"
    if exists_as_file:
        invalid.write_text("not a directory", encoding="utf-8")
    source = invalid if which == "source" else root / "src/scpn_control"
    tests = invalid if which == "tests" else root / "tests"
    with pytest.raises(NotADirectoryError, match="root must be a directory"):
        collect_unlinked_modules(source_root=source, test_root=tests)
    if which == "source":
        with pytest.raises(NotADirectoryError):
            collect_source_modules(source)
    result = _cli(root, "--source-root", str(source), "--test-root", str(tests))
    assert result.returncode == 1 and "NotADirectoryError" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"allowlisted_modules": {}},
        {"allowlisted_modules": [0]},
        {"allowlisted_modules": [{}]},
        {"allowlisted_modules": [{"path": ""}]},
        {"allowlisted_modules": [{"path": 3}]},
    ],
)
def test_allowlist_exact_value_shapes(tmp_path: Path, payload: object) -> None:
    """The public decoder refuses malformed object/list/string shapes without coercion."""
    root = _tree(tmp_path)
    target = root / "tools/untested_module_allowlist.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(ValueError):
        load_allowlist(target)
    result = _cli(root)
    assert result.returncode == 1 and "ValueError" in result.stderr
    assert target.read_bytes() == before


@pytest.mark.parametrize("fault", ["missing", "directory", "json", "utf8", "test_syntax", "facade_syntax"])
def test_native_io_decode_and_syntax_errors(tmp_path: Path, fault: str) -> None:
    """Real filesystem/decode/AST failures propagate before success output or writes."""
    root = _tree(tmp_path, "from scpn_control.facade import api\ndef test_call(): api()\n")
    allowlist = root / "tools/untested_module_allowlist.json"
    flags: list[str] = []
    if fault == "missing":
        flags = ["--allowlist", str(root / "missing.json")]
    elif fault == "directory":
        flags = ["--allowlist", str(root / "tests")]
    elif fault == "json":
        allowlist.write_text("{", encoding="utf-8")
    elif fault == "utf8":
        allowlist.write_bytes(b"\xff")
    elif fault == "test_syntax":
        (root / "tests/test_calls.py").write_text("def test_call(:\n", encoding="utf-8")
    else:
        (root / "src/scpn_control/facade").mkdir()
        (root / "src/scpn_control/facade/__init__.py").write_text("def export(:\n", encoding="utf-8")
    result = _cli(root, *flags)
    assert result.returncode == 1 and not result.stdout
    assert {
        "missing": "FileNotFoundError",
        "directory": "PermissionError" if os.name == "nt" else "IsADirectoryError",
        "json": "JSONDecodeError",
        "utf8": "UnicodeDecodeError",
        "test_syntax": "SyntaxError",
        "facade_syntax": "SyntaxError",
    }[fault] in result.stderr


@pytest.mark.parametrize("linked,allow_stale,expected", [(False, False, 0), (True, False, 1), (True, True, 0)])
def test_explicit_allowlist_and_stale_refusal(tmp_path: Path, linked: bool, allow_stale: bool, expected: int) -> None:
    """An exemption admits only its exact path, and stale exemptions require the explicit flag."""
    root = _tree(tmp_path, PREFIX + "def test_call(): api()\n" if linked else "")
    policy = root / "tools/untested_module_allowlist.json"
    policy.write_text(
        json.dumps(
            {"allowlisted_modules": [{"path": "src/scpn_control/owner.py"}, {"path": "src/scpn_control/owner.py"}]}
        ),
        encoding="utf-8",
    )
    assert load_allowlist(policy) == {"src/scpn_control/owner.py"}
    result = _cli(root, *(["--allow-stale-allowlist"] if allow_stale else []))
    assert result.returncode == expected
    assert f"Stale allowlist entries: {int(linked)}" in result.stdout


def test_unexpected_owner_precedes_stale_allowlist(tmp_path: Path) -> None:
    """The stale-allowlist option never admits an unexpected unlinked implementation."""
    root = _tree(tmp_path)
    (root / "tools/untested_module_allowlist.json").write_text(
        '{"allowlisted_modules": [{"path": "unrelated.py"}]}', encoding="utf-8"
    )
    result = _cli(root, "--allow-stale-allowlist")
    assert result.returncode == 1 and "new modules without direct test linkage" in result.stdout


def test_relative_roots_follow_script_repository_and_public_main(tmp_path: Path) -> None:
    """CLI-relative roots resolve at the script checkout, independently of the working directory."""
    root = _tree(tmp_path, PREFIX + "def test_call(): api()\n")
    result = _cli(
        root,
        "--source-root",
        "src/scpn_control",
        "--test-root",
        "tests",
        "--allowlist",
        "tools/untested_module_allowlist.json",
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert (
        main(
            [
                "--source-root",
                str(root / "src/scpn_control"),
                "--test-root",
                str(root / "tests"),
                "--allowlist",
                str(root / "tools/untested_module_allowlist.json"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("flag,expected", [("--help", 0), ("--unsupported", 2)])
def test_argparse_precedes_invalid_root_inspection(tmp_path: Path, flag: str, expected: int) -> None:
    """Help and parser refusals occur before nonexistent roots are inspected."""
    root = _tree(tmp_path)
    result = _cli(root, "--source-root", str(root / "absent"), flag)
    assert result.returncode == expected and "NotADirectoryError" not in result.stderr


@pytest.mark.parametrize(
    "facade,body,missing_leaf",
    [
        ("from .owner import api\ndef run(): return api()\n", "assert run() == 'real'", False),
        ("import scpn_control.owner as leaf\ndef run(): return leaf.api()\n", "assert run() == 'real'", False),
        ("def run():\n    from .owner import api\n    return api()\n", "assert run() == 'real'", False),
        (
            "from .owner import api\ndef helper(): return api()\ndef run(): return helper()\n",
            "assert run() == 'real'",
            False,
        ),
        (
            "from .owner import api\ndef helper(): return api()\ndef run(): return 'local'\n",
            "assert run() == 'local'",
            True,
        ),
        (
            "def run():\n    def unused():\n        from .owner import api\n        return api()\n    return 'local'\n",
            "assert run() == 'local'",
            True,
        ),
        ("from .owner import api\ndef run(api): return api()\n", "assert run(lambda: 'local') == 'local'", True),
        (
            "from .owner import api\ndef run():\n    api=lambda: 'local'\n    return api()\n",
            "assert run() == 'local'",
            True,
        ),
        (
            "from .owner import api\ndef run():\n    def api(): return 'local'\n    return api()\n",
            "assert run() == 'local'",
            True,
        ),
        ("from .owner import api\ndef run(): return api()\nrun=lambda: 'local'\n", "assert run() == 'local'", True),
        (
            "from .owner import api\ndef run(): return api()\ndef run(): return 'local'\n",
            "assert run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nfrom types import SimpleNamespace as run\ndef run(): return api()\n",
            "assert run() == 'real'",
            True,
        ),
        (
            "from .owner import api\ndef replace(fn): return lambda: 'local'\n@replace\ndef run(): return api()\n",
            "assert run() == 'local'",
            True,
        ),
        ("from .owner import api\nfrom math import *\ndef run(): return api()\n", "assert run() == 'real'", True),
        ("from .owner import api\ndef run(result=api()): return result\n", "assert run() == 'real'", True),
        ("from .owner import api\ndef run(result: api() = 'local'): return result\n", "assert run() == 'local'", True),
        (
            "from .owner import api\ndef run(): return 'local'\nrun.extra=lambda: 'attribute'\n",
            "assert run.extra() == 'attribute'",
            True,
        ),
        (
            "from .owner import api\nasync def run(): return api()\n",
            "import asyncio; assert asyncio.run(run()) == 'real'",
            False,
        ),
        (
            "def run():\n    from .owner import api\n    from pathlib import Path as api\n    return api('local').as_posix()\n",
            "assert run() == 'local'",
            True,
        ),
        (
            "from .owner import api\ndef run():\n    try: return api()\n    except ValueError as api: return 'error'\n",
            "import pytest\n    with pytest.raises(UnboundLocalError): run()",
            True,
        ),
        ("from .owner import api\ndef run():\n    global api\n    return api()\n", "assert run() == 'real'", True),
        ("from .owner import api\nclass run: pass\ndef run(): return api()\n", "assert run() == 'real'", True),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def deliver(self): return api()\ndef run(worker=None):\n    worker=Worker()\n    return worker.deliver()\n",
            "assert run() == 'real'",
            True,
            id="source_parameter_receiver_function",
        ),
        pytest.param(
            "from .owner import api\nclass Delegate:\n    def deliver(self): return api()\nclass Worker:\n    def run(self, worker=None):\n        worker=Delegate()\n        return worker.deliver()\ndef run(): return Worker.run(Worker())\n",
            "assert run() == 'real'",
            True,
            id="source_parameter_receiver_method",
        ),
    ],
)
def test_executed_function_delegation_uses_only_selected_lexical_body(
    tmp_path: Path, facade: str, body: str, missing_leaf: bool
) -> None:
    """Execute public functions and inspect genuine calls without crediting their decoys."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import run\ndef test_run():\n    " + body + "\n",
        {
            "src/scpn_control/facade.py": facade,
            "src/scpn_control/sibling.py": "def unused(): return 'sibling'\n",
        },
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", str(root / "tests")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    missing = (["owner.py"] if missing_leaf else []) + ["sibling.py"]
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == [
        (root / "src/scpn_control" / name).as_posix() for name in missing
    ]
    result = _cli(root)
    assert result.returncode == 1 and not result.stderr
    observed = [line.removeprefix("  - ") for line in result.stdout.splitlines() if line.startswith("  - ")]
    assert observed == ["src/scpn_control/" + name for name in missing]
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")} == before


def test_recursive_source_function_calls_terminate_without_runtime_execution(tmp_path: Path) -> None:
    """Bound a recursive function graph while keeping its unused sibling unlinked."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import run\ndef test_run(): run()\n",
        {
            "src/scpn_control/facade.py": "from .middle import recurse\ndef run(): return recurse()\n",
            "src/scpn_control/middle.py": "from .facade import run\ndef recurse(): return run()\n",
        },
    )
    result = _cli(root)
    assert result.returncode == 1 and "Unexpected modules: 1" in result.stdout and not result.stderr
    assert "src/scpn_control/owner.py" in result.stdout


@pytest.mark.parametrize(
    "facade,body,missing_leaf",
    [
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'real'",
            False,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "assert Worker.run(Worker()) == 'real'",
            False,
        ),
        (
            "class Worker:\n    def run(self):\n        from .owner import api\n        return api()\n",
            "worker=Worker(); assert worker.run() == 'real'",
            False,
        ),
        (
            "from .owner import api\ndef helper(): return api()\nclass Worker:\n    def run(self): return helper()\n",
            "worker=Worker(); assert worker.run() == 'real'",
            False,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return 'local'\n    def unused(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self, api): return api()\n",
            "worker=Worker(); assert worker.run(lambda: 'local') == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self):\n        api=lambda: 'local'\n        return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n    def run(self): return 'local'\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n    run=lambda self: 'local'\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\nWorker.run=lambda self: 'local'\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\ndef replace(fn): return lambda self: 'local'\nclass Worker:\n    @replace\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    @staticmethod\n    def run(): return api()\n",
            "worker=Worker(); assert worker.run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    @property\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    async def run(self): return api()\n",
            "import asyncio; worker=Worker(); assert asyncio.run(worker.run()) == 'real'",
            False,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "from types import SimpleNamespace\n    worker=Worker(); worker=SimpleNamespace(run=lambda: 'local'); assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.run=lambda: 'local'; assert worker.run() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'real'; del worker",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "if True: worker=Worker()\n    assert worker.run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker: Worker=Worker(); assert worker.run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=other=Worker(); assert worker.run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker,other=(Worker(), None); assert worker.run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker; assert worker is Worker; assert worker().run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nfrom types import SimpleNamespace\ndef Worker(): return SimpleNamespace(run=api)\n",
            "worker=Worker(); assert worker.run() == 'real'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "import pytest\n    with pytest.raises(UnboundLocalError): worker.run()\n    worker=Worker()",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run == worker.run",
            False,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker()\n    def worker(): return 'local'\n    assert worker() == 'local'",
            True,
        ),
        (
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker()\n    from types import SimpleNamespace as worker\n    assert worker(value=7).value == 7",
            True,
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.state=3; assert worker.run() == 'real'",
            False,
            id="ordinary_receiver_data_write",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.state=[[1,2]]; worker.state[0][0]=3; del worker.state[0][1]; worker.state[0][0]+=1; assert worker.state == [[4]]; assert worker.run() == 'real'",
            False,
            id="nested_data_store_delete_and_increment",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __init__(self): self.state=3\n    def run(receiver, /): return api()\n",
            "worker=Worker(); worker.state+=1; assert worker.state == 4; assert worker.run() == 'real'",
            False,
            id="constructor_data_and_positional_receiver",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "Worker().state=3; worker=Worker(); assert worker.run() == 'real'",
            False,
            id="unqualified_temporary_object_write",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.run=[lambda: 'before']; worker.run[0]=lambda: 'local'; assert worker.run[0]() == 'local'",
            True,
            id="selected_callable_container_replacement",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.run=lambda: 'local'; del worker.run; assert worker.run() == 'real'",
            True,
            id="selected_method_delete_whole_scope_refusal",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "class Other:\n        def run(self): return 'local'\n    worker=Worker(); worker.__class__=Other; assert worker.run() == 'local'",
            True,
            id="receiver_type_replacement",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.__dict__={'run': lambda: 'local'}; assert worker.run() == 'local'",
            True,
            id="receiver_dictionary_replacement",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.__dict__['run']=lambda: 'local'; assert worker.run() == 'local'",
            True,
            id="receiver_dictionary_member_replacement",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "worker=Worker(); worker.__dict__['state']={'x': 1}; worker.__dict__['state']['x']=3; assert worker.run() == 'real'",
            True,
            id="nested_receiver_dictionary_is_opaque",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __init__(self): self.run=lambda: 'local'\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
            id="source_constructor_overrides_method",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    @property\n    def state(self): return 3\n    @state.setter\n    def state(self, value): self.run=lambda: 'local'\n    def run(self): return api()\n",
            "worker=Worker(); worker.state=3; assert worker.run() == 'local'",
            True,
            id="source_property_setter_overrides_method",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def disable(self): self.run=lambda: 'local'\n    def run(self): return api()\n",
            "worker=Worker(); worker.disable(); assert worker.run() == 'local'",
            True,
            id="source_mutator_overrides_method",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __getattribute__(self, name):\n        return (lambda: 'local') if name == 'run' else object.__getattribute__(self, name)\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
            id="custom_lookup_redirects_method",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __getattr__(self, name): return 'fallback'\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.other == 'fallback'; assert worker.run() == 'real'",
            True,
            id="custom_fallback_lookup_is_opaque",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __setattr__(self, name, value): object.__setattr__(self, name, value)\n    def run(self): return api()\n",
            "worker=Worker(); worker.state=3; assert worker.state == 3; assert worker.run() == 'real'",
            True,
            id="custom_setter_is_opaque",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __delattr__(self, name): object.__delattr__(self, name)\n    def run(self): return api()\n",
            "worker=Worker(); worker.state=3; del worker.state; assert worker.run() == 'real'",
            True,
            id="custom_deletion_is_opaque",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    state=3\n    def status(): return 'ok'\n    def run(self): return api()\n",
            "worker=Worker(); assert Worker.status() == 'ok'; assert worker.run() == 'real'",
            False,
            id="class_data_and_no_argument_method",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self):\n        data=[0]; data[0]=3\n        return api()\n",
            "worker=Worker(); assert worker.run() == 'real'",
            False,
            id="source_local_container_write",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    async def run(receiver, /):\n        receiver.state=3\n        return api()\n",
            "import asyncio; worker=Worker(); assert asyncio.run(worker.run()) == 'real'; assert worker.state == 3",
            False,
            id="async_positional_receiver_state_write",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "Worker.run=lambda self: 'local'; worker=Worker(); assert worker.run() == 'local'",
            True,
            id="test_replaces_source_class_method",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "Worker.__init__=lambda self: setattr(self, 'run', lambda: 'local'); worker=Worker(); assert worker.run() == 'local'",
            True,
            id="test_replaces_source_constructor",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def __init__(self): self.__dict__['run']=lambda: 'local'\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
            id="source_constructor_dictionary_override",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "from types import SimpleNamespace\n    Worker.__new__=lambda cls: SimpleNamespace(run=lambda: 'local'); worker=Worker(); assert worker.run() == 'local'",
            True,
            id="test_replaces_instance_allocator",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "import scpn_control.facade as namespace\n    namespace.Worker.run=lambda self: 'local'; worker=Worker(); assert worker.run() == 'local'",
            True,
            id="imported_namespace_mutates_same_class",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "Worker.__new__notes='metadata'; worker=Worker(); assert Worker.__new__notes == 'metadata'; assert worker.run() == 'real'",
            False,
            id="constructor_metadata_is_not_a_hook_write",
        ),
        pytest.param(
            "from .owner import api\nclass Worker:\n    def run(self): return api()\n",
            "Worker.__init__=lambda self: setattr(self, 'run', lambda: 'local'); worker=Worker(); assert Worker.run(worker) == 'real'",
            False,
            id="explicit_unbound_method_after_constructor_change",
        ),
        pytest.param(
            "from .owner import api\nclass Other:\n    def run(self): return 'local'\nclass Worker:\n    def __init__(self): self.__class__=Other\n    def run(self): return api()\n",
            "worker=Worker(); assert worker.run() == 'local'",
            True,
            id="source_constructor_changes_receiver_type",
        ),
    ],
)
def test_executed_plain_methods_bind_only_unambiguous_receivers(
    tmp_path: Path, facade: str, body: str, missing_leaf: bool
) -> None:
    """Execute public methods and refuse unused, overridden or ambiguous receiver edges."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import Worker\ndef test_run():\n    " + body + "\n",
        {"src/scpn_control/facade.py": facade, "src/scpn_control/sibling.py": "def unused(): return 'sibling'\n"},
    )
    env = dict(
        os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", str(root / "tests")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    missing = (["owner.py"] if missing_leaf else []) + ["sibling.py"]
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == [
        (root / "src/scpn_control" / name).as_posix() for name in missing
    ]
    result = _cli(root)
    assert result.returncode == 1 and not result.stderr
    observed = [line.removeprefix("  - ") for line in result.stdout.splitlines() if line.startswith("  - ")]
    assert observed == ["src/scpn_control/" + name for name in missing]
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")} == before


@pytest.mark.parametrize("override", [False, True])
def test_inherited_method_bodies_are_not_inferred(tmp_path: Path, override: bool) -> None:
    """Execute inherited and overridden methods without attributing runtime MRO to the leaf."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import Worker\ndef test_run():\n    worker=Worker(); assert worker.run() == "
        + repr("local" if override else "real")
        + "\n",
        {
            "src/scpn_control/middle.py": "from .owner import api\nclass Base:\n    def run(self): return api()\n",
            "src/scpn_control/facade.py": "from .middle import Base\nclass Worker(Base):\n"
            + ("    def run(self): return 'local'\n" if override else "    pass\n"),
        },
    )
    env = dict(
        os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", str(root / "tests")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert collect_unlinked_modules(source_root=root / "src/scpn_control", test_root=root / "tests") == [
        (root / "src/scpn_control/owner.py").as_posix()
    ]
    result = _cli(root)
    assert result.returncode == 1 and "Unexpected modules: 1" in result.stdout and not result.stderr


def test_recursive_plain_method_references_terminate_without_runtime_execution(tmp_path: Path) -> None:
    """Bound selected unbound-method recursion without executing an infinite call graph."""
    root = _tree(
        tmp_path,
        "from scpn_control.facade import Worker\ndef test_run():\n    worker=Worker(); worker.run()\n",
        {
            "src/scpn_control/facade.py": "class Worker:\n    def run(self): return Worker.again(self)\n    def again(self): return Worker.run(self)\n"
        },
    )
    result = _cli(root)
    assert result.returncode == 1 and "Unexpected modules: 1" in result.stdout and not result.stderr
