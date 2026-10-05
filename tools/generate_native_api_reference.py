# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Generate the native C ABI and Lean declaration reference.

"""Generate a checked reference from C header and Lean declaration comments."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/scpn_control/core/solver.h"
LEAN = ROOT / "lean/SCPNControl/PulsedFSM.lean"
OUTPUT = ROOT / "docs/_generated/native_api_reference.md"


def _clean_c_comment(raw: str) -> str:
    """Remove Doxygen leading stars and outer blank space while retaining contract line order."""
    lines = [re.sub(r"^\s*\* ?", "", line).rstrip() for line in raw.splitlines()]
    return "\n".join(lines).strip()


def _normalise_signature(raw: str) -> str:
    """Trim declaration boundaries and trailing line whitespace, retaining one final semicolon."""
    return "\n".join(line.rstrip() for line in raw.strip().splitlines()) + ";"


def _c_declarations(source: str) -> list[tuple[str | None, str]]:
    """Pair exported lexical C declarations with immediately adjacent Doxygen blocks.

    A preceding type comment cannot cross its closing delimiter to become a
    function contract. This recognises the maintained header corpus, not general
    C syntax, preprocessing, compilation or ABI compatibility.
    """
    declaration = re.compile(r"^SCPN_SOLVER_API\s+(.+?);", re.DOTALL | re.MULTILINE)
    documented = re.compile(r"/\*\*((?:(?!\*/).)*)\*/\s*^SCPN_SOLVER_API\s+(.+?);", re.DOTALL | re.MULTILINE)
    comments = {match.group(2).strip(): _clean_c_comment(match.group(1)) for match in documented.finditer(source)}
    return [
        (comments.get(match.group(1).strip()), _normalise_signature(match.group(1)))
        for match in declaration.finditer(source)
    ]


def _symbol_name(signature: str) -> str:
    """Require a lexical C function identifier before the first argument-list parenthesis."""
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    if match is None:
        raise ValueError(f"unable to identify C declaration: {signature}")
    return match.group(1)


def _lean_declarations(source: str) -> list[tuple[str, str, str]]:
    """Pair adjacent non-nested Lean doc blocks with maintained inductive/def/theorem declarations.

    Comment whitespace is flattened for Markdown. This does not parse or check
    Lean proofs, nested comments or arbitrary syntax; the owning Lean toolchain
    remains responsible for proof checking.
    """
    pattern = re.compile(
        r"/--\s*((?:(?!-/).)*?)\s*-/\s*(inductive|def|theorem)\s+([A-Za-z_][A-Za-z0-9_]*)",
        re.DOTALL,
    )
    return [(match.group(2), match.group(3), " ".join(match.group(1).split())) for match in pattern.finditer(source)]


def render(*, header_path: Path = HEADER, lean_path: Path = LEAN) -> str:
    """Read selected UTF-8 C/Lean sources and return deterministic Markdown without writing.

    The maintained corpus requires five versioned and five legacy C functions,
    ABI version 1, nonempty adjacent versioned Doxygen comments, and nine
    nonempty adjacent documented Lean declarations. Source IO/decode errors propagate; malformed declarations
    or incomplete comment/cardinality contracts raise ValueError. No compiler,
    proof checker, dynamic library or plant model is executed. The normative
    comments retain units, array/handle ownership and model limitations verbatim.
    Custom source paths support qualifying actual source changes independently
    of the default tracked document; no source authentication is inferred.
    """
    header = header_path.read_text(encoding="utf-8")
    if re.findall(r"^#define\s+SCPN_SOLVER_ABI_VERSION\s+(\S+)\s*$", header, re.MULTILINE) != ["1"]:
        raise ValueError("selected C header must declare ABI version 1 exactly once")
    c_declarations = _c_declarations(header)
    lean_declarations = _lean_declarations(lean_path.read_text(encoding="utf-8"))
    versioned = [(comment, sig) for comment, sig in c_declarations if _symbol_name(sig).startswith("scpn_solver_")]
    legacy = [(comment, sig) for comment, sig in c_declarations if not _symbol_name(sig).startswith("scpn_solver_")]

    if len(versioned) != 5 or len(legacy) != 5:
        raise ValueError(
            f"expected five versioned and five legacy C functions, found {len(versioned)} and {len(legacy)}"
        )
    if any(not comment for comment, _ in versioned):
        raise ValueError("every versioned C function requires a nonempty adjacent Doxygen contract")
    if len(lean_declarations) != 9:
        raise ValueError(f"expected nine documented Lean declarations, found {len(lean_declarations)}")
    if any(not comment for _, _, comment in lean_declarations):
        raise ValueError("every selected Lean declaration requires a nonempty adjacent contract")

    lines = [
        "# Native API and checked proof reference",
        "",
        "<!-- Generated by tools/generate_native_api_reference.py; do not edit. -->",
        "",
        "This reference is generated from the selected C header and Lean source.",
        "The default corpus is `src/scpn_control/core/solver.h` and",
        "`lean/SCPNControl/PulsedFSM.lean`. The selected C header is the normative ABI",
        "contract. The Lean section describes only the checked finite-state model;",
        "it is not evidence of continuous plant or plasma safety.",
        "",
        "## C ABI version 1",
        "",
        "ABI version: `SCPN_SOLVER_ABI_VERSION == 1`. Status `OK` means the call",
        "was valid; convergence is reported separately by the convergence function.",
        "",
    ]
    for comment, signature in versioned:
        lines.extend([f"### `{_symbol_name(signature)}`", "", str(comment), "", "```c", signature, "```", ""])

    lines.extend(
        [
            "## Legacy compatibility ABI",
            "",
            "These unversioned symbols remain loadable for existing clients. They",
            "preserve historical null, zero, or silent error signalling. New clients",
            "should use the typed version 1 ABI above.",
            "",
        ]
    )
    for _, signature in legacy:
        lines.extend([f"### `{_symbol_name(signature)}`", "", "```c", signature, "```", ""])

    lines.extend(["## Lean checked declarations", ""])
    for kind, name, comment in lean_declarations:
        lines.extend([f"### `{name}`", "", f"Kind: `{kind}`. {comment}", ""])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Return zero for exact checked text or a successful local write, one for stale/refused IO or source contracts.

    Optional header/Lean/output paths select local files. A write refuses resolved
    path and existing hard-link aliases of either selected source. Non-alias
    output can be replaced. Check mode only reads and compares UTF-8 bytes;
    parser help/usage retain exits zero/two. No compilation or proof checking is
    implied by generation or comparison success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of updating stale output")
    parser.add_argument("--header", type=Path, default=HEADER, help="selected C header")
    parser.add_argument("--lean", type=Path, default=LEAN, help="selected Lean declaration source")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="selected Markdown reference")
    args = parser.parse_args(argv)
    try:
        if not args.check:
            for source in (args.header, args.lean):
                if args.output.resolve() == source.resolve() or (
                    args.output.exists() and source.exists() and args.output.samefile(source)
                ):
                    raise ValueError("native reference output must not overwrite selected source")
        rendered = render(header_path=args.header, lean_path=args.lean)
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
                print(f"stale generated native API reference: {args.output}")
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"native API reference FAILED: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print("native API reference is current")
        return 0
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
