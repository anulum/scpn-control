# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Cross-language public API ownership and contract gate.

"""Compare static API declarations and renderer fragments with local TOML.

Inventory equality is a declaration check, not semantic review or execution.
The registry remains an operator-supplied local file; it is not authenticated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tools/api_contract_registry.toml"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools.api_contract_inventory import (
    CLASSIFICATIONS,
    ExportInventory,
    Inventory,
)
from tools.api_contract_inventory import (
    ApiContractInspectionError as ApiContractInspectionError,
)
from tools.api_contract_inventory import (
    Candidate as Candidate,
)
from tools.api_contract_inventory import (
    build_inventory as build_inventory,
)


def _table(value: object) -> dict[str, object]:
    """Require a string-keyed TOML table before inspecting declared fields."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ApiContractInspectionError("API registry requires a table for each declared family")
    return cast(dict[str, object], value)


def _count(value: object) -> int:
    """Require a nonnegative integer, explicitly refusing TOML booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ApiContractInspectionError("API registry counts require nonnegative integers")
    return value


def _digest(value: object) -> str:
    """Require a lowercase 64-digit hexadecimal declared name digest."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ApiContractInspectionError("API registry digests require 64 lowercase hexadecimal digits")
    return value


def _symbols(value: object) -> list[str]:
    """Validate unique native identifier lists without compiling declarations."""
    if not isinstance(value, list):
        raise ApiContractInspectionError("API registry symbols require identifier lists")
    symbols: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.isidentifier() or item in symbols:
            raise ApiContractInspectionError("API registry symbols require unique identifier strings")
        symbols.append(item)
    return symbols


def _export_inventory(table: dict[str, object], prefix: str) -> ExportInventory:
    """Read one declared native export count and name digest."""
    return {
        "export_count": _count(table.get(prefix + "_export_count")),
        "export_sha256": _digest(table.get(prefix + "_export_sha256")),
    }


def _expected_inventory(registry: dict[str, object]) -> Inventory:
    """Validate the v1 inventory shape without interpreting policy prose."""
    if registry.get("schema") != "scpn-control.api-contract-registry.v1":
        raise ApiContractInspectionError("API registry schema is unsupported or missing")
    table = _table(registry.get("inventory"))
    raw_counts = _table(table.get("python_classifications"))
    raw_hashes = _table(table.get("python_classification_sha256"))
    if set(raw_counts) != set(CLASSIFICATIONS) or set(raw_hashes) != set(CLASSIFICATIONS):
        raise ApiContractInspectionError("API registry must declare exactly three Python ownership classes")
    counts = {name: _count(raw_counts[name]) for name in CLASSIFICATIONS}
    candidate_count = _count(table.get("python_candidate_count"))
    if sum(counts.values()) != candidate_count:
        raise ApiContractInspectionError("API registry classification counts must partition its candidates")
    return {
        "python": {
            "candidate_count": candidate_count,
            "candidate_sha256": _digest(table.get("python_candidate_sha256")),
            "stable_export_count": _count(table.get("python_stable_export_count")),
            "stable_export_sha256": _digest(table.get("python_stable_export_sha256")),
            "classifications": counts,
            "classification_sha256": {name: _digest(raw_hashes[name]) for name in CLASSIFICATIONS},
        },
        "c": {"symbols": _symbols(table.get("c_symbols"))},
        "lean": {"symbols": _symbols(table.get("lean_symbols"))},
        "rust": _export_inventory(table, "rust"),
        "typescript": _export_inventory(table, "typescript"),
    }


def _renderer_contracts(value: object) -> dict[str, list[str]]:
    """Validate nonempty relative fragment declarations before reading files."""
    table = _table(value)
    if not table:
        raise ApiContractInspectionError("API registry must declare renderer fragments")
    contracts: dict[str, list[str]] = {}
    for relative, raw in table.items():
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            raise ApiContractInspectionError("renderer contract paths must be relative without parent traversal")
        if not isinstance(raw, list) or not raw:
            raise ApiContractInspectionError("renderer contracts require nonempty fragment lists")
        fragments: list[str] = []
        for fragment in raw:
            if not isinstance(fragment, str) or not fragment.strip():
                raise ApiContractInspectionError("renderer fragments must be nonblank strings")
            fragments.append(fragment)
        contracts[relative] = fragments
    return contracts


def check_contracts(repo: Path = ROOT, registry_path: Path | None = None) -> list[str]:
    """Return declaration drift and missing literal renderer fragments.

    Parameters
    ----------
    repo : pathlib.Path
        Root for static inventory and renderer input paths, relative to cwd.
    registry_path : pathlib.Path or None
        Explicit TOML path relative to cwd, or tools/api_contract_registry.toml
        under repo. Only the supported v1 required fields are interpreted;
        extra metadata and policy prose are not execution or review evidence.

    Returns
    -------
    list[str]
        Fresh errors in renderer-table/fragment order followed by inventory
        drift. Equality uses counts, name digests and ordered C/Lean symbols.
        Diagnostics retain declared paths/fragments; there is no redaction.

    Raises
    ------
    ApiContractInspectionError
        Inspection or required TOML field validation fails. No partial success
        is returned. Files are read as UTF-8 with universal newlines, following
        symlinks without containment or coherent snapshot guarantees.

    Notes
    -----
    This read-only operation does not run renderers, import optional backends,
    authenticate the registry, verify source-byte hashes or grant stability,
    scientific admission or independent review. Comments can satisfy fragments.
    """
    current = build_inventory(repo)
    path = registry_path if registry_path is not None else repo / "tools/api_contract_registry.toml"
    try:
        registry = _table(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ApiContractInspectionError("could not read a valid UTF-8 API registry") from None
    expected = _expected_inventory(registry)
    contracts = _renderer_contracts(registry.get("renderer_contracts"))
    errors: list[str] = []
    for relative, fragments in contracts.items():
        try:
            content = (repo / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise ApiContractInspectionError("could not read a required renderer input as UTF-8") from None
        for fragment in fragments:
            if fragment not in content:
                errors.append(f"{relative} lacks required renderer contract: {fragment}")
    if current != expected:
        errors.append(
            "API declaration inventory differs from the registry\n"
            f"expected={json.dumps(expected, sort_keys=True)}\n"
            f"current={json.dumps(current, sort_keys=True)}"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Inspect local declarations through the operator CLI.

    Parameters
    ----------
    argv : list[str] or None
        Explicit argument tokens or process arguments. --repo defaults to the
        script repository and is resolved by the CLI. --registry is cwd-relative.
        --print-inventory prints JSON without reading a registry or renderers.

    Returns
    -------
    int
        0 for matching declarations/fragments or inventory JSON, 1 for policy
        drift, 2 for authored inspection refusal. Diagnostics use stdout.

    Raises
    ------
    SystemExit
        Argparse help returns 0; malformed arguments return 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--print-inventory", action="store_true")
    args = parser.parse_args(argv)
    try:
        try:
            repo = args.repo.resolve()
        except (OSError, RuntimeError):
            raise ApiContractInspectionError("could not resolve the API repository root") from None
        if args.print_inventory:
            print(json.dumps(build_inventory(repo), indent=2, sort_keys=True))
            return 0
        errors = check_contracts(repo, args.registry)
    except ApiContractInspectionError as refusal:
        print(f"API contract inspection refused: {refusal}")
        return 2
    if errors:
        print("API contract gate failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("API declarations match the registry and required renderer fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
