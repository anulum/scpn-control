# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native API reference generation contracts.

"""Tests for the generated native C and Lean API reference."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import generate_native_api_reference as generator


def test_render_includes_all_versioned_legacy_and_lean_declarations() -> None:
    """The generated reference covers both ABI generations and Lean scope."""
    rendered = generator.render()

    assert rendered.count("### `scpn_solver_") == 5
    assert "### `create_solver`" in rendered
    assert "### `destroy_solver`" in rendered
    assert "### `pulsed_fsm_eventually_returns_to_idle`" in rendered
    assert "not evidence of continuous plant or plasma safety" in rendered


def test_checked_reference_is_current() -> None:
    """The checked-in Markdown is byte-current with normative sources."""
    assert generator.main(["--check"]) == 0


@pytest.fixture
def selected_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Copy both complete maintained source corpora for public path-selection tests."""
    header = tmp_path / "solver.h"
    lean = tmp_path / "PulsedFSM.lean"
    header.write_bytes(generator.HEADER.read_bytes())
    lean.write_bytes(generator.LEAN.read_bytes())
    return header, lean


def test_comments_are_adjacent_and_preserve_function_contracts(selected_sources: tuple[Path, Path]) -> None:
    """Unattached type/namespace comments cannot become part of function/declaration contracts."""
    header, lean = selected_sources
    lean.write_text(
        lean.read_text().replace(
            "namespace SCPNControl.PulsedFSM",
            "/-- UNATTACHED_NAMESPACE_COMMENT -/\nnamespace SCPNControl.PulsedFSM",
        )
    )
    rendered = generator.render(header_path=header, lean_path=lean)
    assert "Opaque solver allocation" not in rendered
    assert "Machine-readable outcome" not in rendered
    assert "typedef struct" not in rendered
    assert "UNATTACHED_NAMESPACE_COMMENT" not in rendered
    assert "Allocate a fixed-boundary Grad-Shafranov SOR state." in rendered
    assert "Concurrent calls on the same handle are unsupported;" in rendered
    assert "solution array (Wb/rad in this kernel)." in rendered
    assert "Kind: `inductive`. The eight ordered phases" in rendered
    assert generator.render(header_path=header, lean_path=lean) == rendered


@pytest.mark.parametrize("fault", ["absent", "two", "duplicate"])
def test_selected_header_requires_supported_abi(selected_sources: tuple[Path, Path], fault: str) -> None:
    """A selected incompatible or ambiguous ABI cannot receive a version-one reference."""
    header, lean = selected_sources
    declaration = "#define SCPN_SOLVER_ABI_VERSION 1"
    replacements = {"absent": "", "two": declaration[:-1] + "2", "duplicate": declaration + "\n" + declaration}
    header.write_text(header.read_text().replace(declaration, replacements[fault]))
    with pytest.raises(ValueError, match="ABI version 1 exactly once"):
        generator.render(header_path=header, lean_path=lean)


@pytest.mark.parametrize("fault", ["missing_function", "missing_comment", "empty_comment", "invalid_signature"])
def test_selected_c_contract_refusals(selected_sources: tuple[Path, Path], fault: str) -> None:
    """Public rendering refuses incomplete exports, undocumented creation and malformed function syntax."""
    header, lean = selected_sources
    source = header.read_text()
    if fault == "missing_function":
        source = source.replace("SCPN_SOLVER_API void destroy_solver(void* solver);", "")
        error = "five versioned and five legacy"
    elif fault in {"missing_comment", "empty_comment"}:
        source = re.sub(
            r"/\*\*\n \* Allocate a fixed-boundary.*?\*/",
            "" if fault == "missing_comment" else "/** */",
            source,
            count=1,
            flags=re.DOTALL,
        )
        error = "nonempty adjacent Doxygen"
    else:
        source = re.sub(
            r"SCPN_SOLVER_API void\* create_solver\(.*?\);",
            "SCPN_SOLVER_API int missing_function_name;",
            source,
            count=1,
            flags=re.DOTALL,
        )
        error = "unable to identify C declaration"
    header.write_text(source)
    with pytest.raises(ValueError, match=error):
        generator.render(header_path=header, lean_path=lean)


@pytest.mark.parametrize("empty", [False, True])
def test_selected_lean_requires_adjacent_nonempty_contracts(selected_sources: tuple[Path, Path], empty: bool) -> None:
    """Removing or emptying a real declaration comment cannot inherit an earlier closed block."""
    header, lean = selected_sources
    comment = "/-- Return the sole admitted successor of a scheduler state. -/"
    lean.write_text(lean.read_text().replace(comment, "/-- -/" if empty else ""))
    error = "nonempty adjacent contract" if empty else "nine documented Lean"
    with pytest.raises(ValueError, match=error):
        generator.render(header_path=header, lean_path=lean)


def test_public_write_check_and_stale_outputs(selected_sources: tuple[Path, Path], tmp_path: Path) -> None:
    """Write nested output, compare exact text, and refuse both missing and stale checked files."""
    header, lean = selected_sources
    output = tmp_path / "nested" / "reference.md"
    args = ["--header", str(header), "--lean", str(lean), "--output", str(output)]
    original = (header.read_bytes(), lean.read_bytes())
    assert generator.main([*args, "--check"]) == 1
    assert not output.exists()
    assert generator.main(args) == 0
    assert output.read_text() == generator.render(header_path=header, lean_path=lean)
    assert generator.main([*args, "--check"]) == 0
    output.write_text("stale reference\n")
    assert generator.main([*args, "--check"]) == 1
    assert output.read_text() == "stale reference\n"
    assert (header.read_bytes(), lean.read_bytes()) == original


@pytest.mark.parametrize("source_index", [0, 1])
@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_public_write_preserves_source_aliases(
    selected_sources: tuple[Path, Path], tmp_path: Path, source_index: int, alias_kind: str
) -> None:
    """Direct, symbolic and hard-link output aliases cannot overwrite either selected input."""
    header, lean = selected_sources
    source = selected_sources[source_index]
    original = source.read_bytes()
    output = source if alias_kind == "same" else tmp_path / "aliased.md"
    if alias_kind == "symlink":
        output.symlink_to(source)
    elif alias_kind == "hardlink":
        output.hardlink_to(source)
    assert generator.main(["--header", str(header), "--lean", str(lean), "--output", str(output)]) == 1
    assert source.read_bytes() == original


@pytest.mark.parametrize("failure", ["missing_header", "invalid_header_utf8", "blocked_parent", "invalid_output_utf8"])
def test_public_cli_authored_io_errors(
    selected_sources: tuple[Path, Path], tmp_path: Path, failure: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Expected source/decode/write failures return one with authored diagnostics and preserve inputs."""
    header, lean = selected_sources
    output = tmp_path / "reference.md"
    args = ["--header", str(header), "--lean", str(lean), "--output", str(output)]
    if failure == "missing_header":
        header.unlink()
    elif failure == "invalid_header_utf8":
        header.write_bytes(b"\xff")
    elif failure == "blocked_parent":
        tmp_path.joinpath("blocker").write_text("regular file")
        args[-1] = str(tmp_path / "blocker" / "reference.md")
    else:
        output.write_bytes(b"\xff")
        args.append("--check")
    assert generator.main(args) == 1
    captured = capsys.readouterr()
    assert "native API reference FAILED:" in captured.err
    assert "Traceback" not in captured.err
    assert lean.read_bytes() == generator.LEAN.read_bytes()


def test_cli_help_usage_and_contract_failure(
    selected_sources: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Public parser retains help/usage exits and catches an unsupported selected ABI without a traceback."""
    for args, exit_code in [(["--help"], 0), (["--unknown"], 2)]:
        with pytest.raises(SystemExit) as failure:
            generator.main(args)
        assert failure.value.code == exit_code
    header, lean = selected_sources
    header.write_text(header.read_text().replace("#define SCPN_SOLVER_ABI_VERSION 1", ""))
    assert generator.main(["--header", str(header), "--lean", str(lean), "--check"]) == 1
    captured = capsys.readouterr()
    assert "native API reference FAILED:" in captured.err
    assert "ABI version 1 exactly once" in captured.err


def test_real_cold_cli_writes_and_checks_selected_corpora(selected_sources: tuple[Path, Path], tmp_path: Path) -> None:
    """Launch the actual script from an unrelated cwd for write and exact comparison without import scaffolding."""
    header, lean = selected_sources
    output = tmp_path / "cold" / "reference.md"
    argv = [
        sys.executable,
        str(Path(generator.__file__).resolve()),
        "--header",
        str(header),
        "--lean",
        str(lean),
        "--output",
        str(output),
    ]
    written = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert written.returncode == 0, written.stderr
    assert output.read_text() == generator.render(header_path=header, lean_path=lean)
    checked = subprocess.run([*argv, "--check"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stderr
    assert "native API reference is current" in checked.stdout
