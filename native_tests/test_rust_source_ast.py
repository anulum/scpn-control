# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual Rust parser documentation and attribute contracts.
"""Exercise the pinned native AST on hidden, test, nested and macro syntax."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.rust_source_ast import _string, rust_ast_errors
from tools.rust_source_profile import _PARSER, _native

_VALID = "//! Define a native source contract.\n\n/// Return a fixed value.\npub fn value() -> i32 { 1 }\n"


def _dump(tmp_path: Path, source: str) -> tuple[str, Path]:
    """Require actual successful pre-expansion parsing of a physical source file."""
    path = tmp_path / "source.rs"
    path.write_text(source, encoding="utf-8")
    result = _native(
        tmp_path,
        [
            "rustc",
            f"+{_PARSER}",
            "--edition=2021",
            "--crate-type=lib",
            "--crate-name=source",
            "-Zunpretty=ast-tree",
            str(path),
        ],
        30.0,
        target_dir=tmp_path.parent / "rust-ast-target",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout, path


def test_native_ast_accepts_documented_hidden_test_and_nested_items(tmp_path: Path) -> None:
    """Real parsed owners retain comments even when ordinary Clippy excludes them."""
    source = (
        _VALID
        + """/// Keep a hidden implementation.
#[doc(hidden)]
fn hidden() {}
/// Exercise the implementation.
#[cfg(test)]
mod tests {
    /// Check the fixed value.
    #[test]
    fn actual() { assert_eq!(super::value(), 1); }
}
/// Return a nested implementation value.
fn outer() -> i32 {
    /// Return the local value.
    fn inner() -> i32 { 1 }
    inner()
}
"""
    )
    dump, path = _dump(tmp_path, source)
    assert rust_ast_errors(dump, path) == []


@pytest.mark.parametrize(
    "source",
    [
        _VALID.replace("/// Return a fixed value.\n", ""),
        _VALID.replace("//! Define a native source contract.\n", ""),
        _VALID + "#[doc(hidden)]\nfn hidden() {}\n",
        _VALID + "#[cfg(test)]\nmod tests { #[test] fn missing() {} }\n",
        _VALID + "/// Hold a value.\nstruct Value { missing: i32 }\n",
        _VALID + "/// Choose a mode.\nenum Mode { Missing }\n",
        _VALID.replace("Return a fixed value.", "SCPN Control — Example."),
        _VALID.replace("Return a fixed value.", ""),
    ],
)
def test_actual_parser_detects_missing_contracts_outside_default_doc_lints(tmp_path: Path, source: str) -> None:
    """Complete physical declarations cannot disappear through visibility, cfg or empty prose."""
    dump, path = _dump(tmp_path, source)
    assert any("requires an attached contract" in error for error in rust_ast_errors(dump, path))


@pytest.mark.parametrize(
    "attribute",
    ["#[allow(dead_code)]", "#[expect(dead_code)]", "#[rustfmt::skip]", "#[cfg_attr(test, allow(dead_code))]"],
)
def test_actual_native_configuration_attributes_cannot_waive_the_profile(tmp_path: Path, attribute: str) -> None:
    """Native attribute records expose actual waivers without scanning raw source strings."""
    dump, path = _dump(tmp_path, _VALID.replace("pub fn", attribute + "\npub fn"))
    assert any("profile configuration" in error for error in rust_ast_errors(dump, path))


def test_native_strings_and_comments_do_not_become_attributes(tmp_path: Path) -> None:
    """Actual raw strings and doc examples keep their data meaning in the native AST."""
    source = (
        _VALID
        + """/// Preserve attribute examples: #[allow(dead_code)] and \\u{3b1}.
fn example() -> &'static str { r#"#[rustfmt::skip] fn imaginary() {}"# }
"""
    )
    dump, path = _dump(tmp_path, source)
    assert rust_ast_errors(dump, path) == []


def test_native_conditional_attributes_preserve_ordinary_configuration(tmp_path: Path) -> None:
    """Real cfg_attr preserves harmless settings and quoted feature data while actual waivers refuse."""
    source = _VALID.replace("pub fn", '#[cfg_attr(feature = "allow", inline)]\npub fn')
    dump, path = _dump(tmp_path, source)
    assert rust_ast_errors(dump, path) == []


def test_actual_unicode_block_docs_and_escaped_controls_decode(tmp_path: Path) -> None:
    """Real parser string escaping preserves Greek, NUL notation and multiline contract prose."""
    source = (
        _VALID
        + "/** Return α and describe \\0 without executing code.\n * Keep the source contract. */\nfn alpha() {}\n"
    )
    dump, path = _dump(tmp_path, source)
    assert rust_ast_errors(dump, path) == []
    assert _string('"\\u{3b1}\\n\\0\\t\\r\\\\"') == "α\n\0\t\r\\"


@pytest.mark.parametrize("source", ["//! Only prose.\n", ""])
def test_actual_empty_implementation_cannot_certify_a_profile(tmp_path: Path, source: str) -> None:
    """A real parsed empty crate is not a completed implementation."""
    dump, path = _dump(tmp_path, source)
    with pytest.raises(RuntimeError, match="nonempty"):
        rust_ast_errors(dump, path)


def test_macro_declarations_refuse_while_quoted_keyword_data_pass(tmp_path: Path) -> None:
    """Native macro token kinds distinguish a declaration keyword from literal payload data."""
    dump, path = _dump(tmp_path, _VALID + '/// Print keyword data.\nfn print_data() { println!("fn"); }\n')
    assert rust_ast_errors(dump, path) == []
    dump, path = _dump(tmp_path, _VALID + "make_item!(fn hidden() {});\n")
    assert any("generated-source" in error for error in rust_ast_errors(dump, path))


@pytest.mark.parametrize(
    "fault",
    [
        "truncated",
        "closure",
        "indent",
        "kind",
        "span",
        "attribute",
        "crate",
        "kind-missing",
        "span-missing",
        "comment-string",
        "attribute-path",
        "string-value",
    ],
)
def test_damage_to_actual_native_protocol_refuses(tmp_path: Path, fault: str) -> None:
    """Negative framing changes preserve the actual native parser as the source of original data."""
    dump, path = _dump(tmp_path, _VALID)
    if fault == "truncated":
        dump = dump[:-3]
    elif fault == "closure":
        dump = dump.replace("            kind: Fn(", "            kind: Unknown(", 1)
        dump = dump.replace("        },\n", "    },\n", 1)
    elif fault == "indent":
        dump = dump.replace("        Item {", "       Item {", 1)
    elif fault == "kind":
        dump = dump.replace("kind: Fn(", "kind: Unknown(")
    elif fault == "span":
        dump = dump.replace(str(path), str(path.with_name("other.rs")))
    elif fault == "attribute":
        dump = dump.replace("kind: DocComment(", "kind: Unknown(", 1)
    elif fault == "crate":
        dump = dump.replace("    items: [", "    Crate {\n    },\n    items: [", 1)
    elif fault == "kind-missing":
        dump = dump.replace("kind: Fn(", "lost: Fn(")
    elif fault == "span-missing":
        dump = "\n".join(line for line in dump.splitlines() if not line.lstrip().startswith("span:"))
    elif fault == "comment-string":
        dump = dump.replace('" Define a native source contract.",', "missing,", 1)
    elif fault == "attribute-path":
        dump, path = _dump(tmp_path, _VALID.replace("pub fn", "#[inline]\npub fn"))
        dump = dump.replace("ident: inline#0,", "lost: inline#0,")
    elif fault == "string-value":
        dump = dump.replace('" Define a native source contract.",', '"\\u{110000}",', 1)
    with pytest.raises(RuntimeError):
        rust_ast_errors(dump, path)


@pytest.mark.parametrize("spelling", ["null", '"\\q"', '"\\u{d800}"', '"\\u{110000}"', '"\\u{12"', '"a"b"', '"\n"'])
def test_incomplete_native_string_values_are_negative_controls(spelling: str) -> None:
    """Malformed native debug values cannot turn a corrupted protocol into contract prose."""
    with pytest.raises((RuntimeError, ValueError)):
        _string(spelling)


def test_actual_macro_without_declarations_preserves_token_data(tmp_path: Path) -> None:
    """An ordinary native macro token stream contains no generated declaration contract."""
    dump, path = _dump(tmp_path, _VALID + "/// Check the value.\nfn check() { assert_eq!(value(), 1); }\n")
    assert rust_ast_errors(dump, path) == []


def test_native_item_macro_without_declarations_preserves_data(tmp_path: Path) -> None:
    """A real item macro's literal payload is not a generated declaration token."""
    dump, path = _dump(tmp_path, _VALID + "make_item!(42);\n")
    assert rust_ast_errors(dump, path) == []


@pytest.mark.parametrize(
    "source",
    [
        "/// Group generated code.\nfn outer() { make_item!(fn hidden() {}); }\n",
        "/// Generate an item.\nmacro_rules! create { () => { fn hidden() {} }; }\n",
    ],
)
def test_actual_nested_macro_declaration_requires_qualification(tmp_path: Path, source: str) -> None:
    """The native parser retains declaration tokens inside a function's macro invocation."""
    dump, path = _dump(tmp_path, _VALID + source)
    assert any("generated-source" in finding for finding in rust_ast_errors(dump, path))
