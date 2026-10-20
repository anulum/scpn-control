# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Owner debt ledger filesystem and profile contracts.
"""Exercise real JSON files and exact profile binding through the public loader."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.docstring_debt_ledger import (
    SOURCE_FINGERPRINT_PROTOCOL,
    measure_owner_sources,
    measurement_profile,
    owner_path,
    read_debt_ceiling,
    read_owner_ledger,
)


@pytest.fixture
def ledger_payload(tmp_path: Path) -> dict[str, object]:
    """Declare manufactured historical bounds using an actual configuration digest."""
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    return {
        "total": 3,
        "scope": ["tests", "tools", "validation"],
        "by_code": {"D103": 3},
        "per_file": {"tests/owner.py": {"D103": 2}},
        "profile": measurement_profile(tmp_path),
        "source_fingerprints": {"tests/owner.py": "a" * 64},
        "source_fingerprint_protocol": SOURCE_FINGERPRINT_PROTOCOL,
        "origin_revision": "a" * 40,
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Persist one real JSON carrier without replacing production state."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_native_owner_ledger_preserves_stricter_bounds_below_aggregate(
    tmp_path: Path, ledger_payload: dict[str, object]
) -> None:
    """A preserved aggregate ceiling cannot enlarge the exact owner allowance."""
    path = _write(tmp_path, ledger_payload)
    before = path.read_bytes()
    ledger = read_owner_ledger(path)
    assert ledger.total == 3 and ledger.per_file == {"tests/owner.py": {"D103": 2}}
    assert ledger.profile == measurement_profile(tmp_path) and ledger.origin_revision == "a" * 40
    assert read_debt_ceiling(path) == 3 and path.read_bytes() == before


def test_legacy_aggregate_is_readable_but_cannot_clear_owner_enforcement(tmp_path: Path) -> None:
    """Existing aggregate API compatibility supplies no fallback for required owner bounds."""
    path = _write(tmp_path, {"total": 0, "scope": ["tests", "tools", "validation"]})
    assert read_debt_ceiling(path) == 0
    with pytest.raises(RuntimeError, match="per_file"):
        read_owner_ledger(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("per_file", []),
        ("per_file", {"tests/owner.py": []}),
        ("per_file", {"tests/owner.py": {"F401": 1}}),
        ("per_file", {"tests/owner.py": {"D103": True}}),
        ("per_file", {"tests/owner.py": {"D103": -1}}),
        ("per_file", {"tests/owner.py": {"D103": 4}}),
        ("profile", None),
        ("profile", {"ruff_version": "0.16.10"}),
        ("profile", {"ruff_version": 3, "configuration_sha256": "a" * 64}),
        ("profile", {"ruff_version": "", "configuration_sha256": "a" * 64}),
        ("profile", {"ruff_version": " 0.16.10", "configuration_sha256": "a" * 64}),
        ("profile", {"ruff_version": "0.16.10", "configuration_sha256": "bad"}),
        ("profile", {"ruff_version": "0.16.10", "configuration_sha256": 3}),
        ("origin_revision", "not-a-commit"),
        ("origin_revision", None),
        ("source_fingerprints", None),
        ("source_fingerprints", {}),
        ("source_fingerprints", {"tests/owner.py": "bad"}),
        ("source_fingerprints", {"tests/owner.py": 3}),
        ("source_fingerprint_protocol", "unknown"),
    ],
)
def test_actual_ledger_rejects_inconsistent_owner_fields(
    tmp_path: Path, ledger_payload: dict[str, object], field: str, value: object
) -> None:
    """Malformed owner/profile metadata cannot pass even the legacy read API."""
    ledger_payload[field] = value
    path = _write(tmp_path, ledger_payload)
    for loader in [read_owner_ledger, read_debt_ceiling]:
        with pytest.raises(RuntimeError, match="malformed docstring debt ledger"):
            loader(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "/tests/a.py",
        "tests/../tools/a.py",
        "tests//a.py",
        "tests/./a.py",
        "src/a.py",
        "tools/a.txt",
        "tests\\a.py",
        "tests/a\n.py",
    ],
)
def test_owner_keys_cannot_alias_or_escape_the_declared_scope(
    tmp_path: Path, ledger_payload: dict[str, object], path: str
) -> None:
    """Only canonical slash-separated owner identities can hold debt allowances."""
    ledger_payload["per_file"] = {path: {"D103": 1}}
    with pytest.raises(RuntimeError, match="owner paths"):
        read_owner_ledger(_write(tmp_path, ledger_payload))


def test_owner_path_retains_unicode_identity() -> None:
    """Native path keys retain Unicode rather than stripping or case-folding names."""
    assert owner_path("tests/život.py") == "tests/život.py"


def test_missing_configuration_refuses_an_unbound_profile(tmp_path: Path) -> None:
    """A missing actual configuration cannot invent a comparable measurement profile."""
    with pytest.raises(RuntimeError, match="could not bind"):
        measurement_profile(tmp_path)


def test_omitted_origin_is_not_a_fabricated_source_attestation(
    tmp_path: Path, ledger_payload: dict[str, object]
) -> None:
    """A profile without an optional origin remains explicit about absent provenance."""
    ledger_payload.pop("origin_revision")
    assert read_owner_ledger(_write(tmp_path, ledger_payload)).origin_revision == ""


def test_actual_interpreter_without_site_packages_cannot_bind_ruff(tmp_path: Path) -> None:
    """An isolated native interpreter actually lacks Ruff metadata and must refuse."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    source = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tools.docstring_debt_ledger import measurement_profile; "
        "from pathlib import Path; measurement_profile(Path(sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-S", "-c", source, str(root), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1 and "could not bind" in result.stderr and result.stdout == ""


def test_source_fingerprints_normalise_native_line_endings_only(tmp_path: Path) -> None:
    """CRLF/LF encodings of one real source agree; another code value must differ."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "lf.py").write_bytes(b"value = 1\n")
    (tools / "crlf.py").write_bytes(b"value = 1\r\n")
    (tools / "changed.py").write_bytes(b"value = 2\n")
    result = measure_owner_sources(tmp_path, ["tools/lf.py", "tools/crlf.py", "tools/changed.py"])
    assert result["tools/lf.py"] == result["tools/crlf.py"] != result["tools/changed.py"]


@pytest.mark.parametrize("defect", ["missing", "directory", "encoding", "alias"])
def test_source_inspection_refuses_unavailable_or_invalid_owner(tmp_path: Path, defect: str) -> None:
    """No uninspected source can acquire an invented historical fingerprint."""
    tools = tmp_path / "tools"
    tools.mkdir()
    path = tools / "owner.py"
    if defect == "directory":
        path.mkdir()
    elif defect == "encoding":
        path.write_bytes(b"\xff")
    owner = "tools/../owner.py" if defect == "alias" else "tools/owner.py"
    with pytest.raises(RuntimeError, match="fingerprint"):
        measure_owner_sources(tmp_path, [owner])
