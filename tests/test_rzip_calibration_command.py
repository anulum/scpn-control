# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real bounded RZIP benchmark command and filesystem contracts.
"""Exercise the actual calibration benchmark API and report filesystem."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_control.control import rzip_model as model
from validation import benchmark_rzip_calibration as producer


def test_real_benchmark_stdout_and_pair(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The module API runs the real model without writes and with explicit temporary outputs."""
    assert producer.main(["--no-write"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["facility_claim_allowed"] is False
    out = tmp_path / "nested" / "rzip.json"
    md = tmp_path / "nested-md" / "rzip.md"
    assert producer.main(["--json-out", str(out), "--markdown-out", str(md)]) == 0
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out) == payload
    assert "bounded local regression evidence" in md.read_text()


@pytest.mark.parametrize("kind", ["same", "symlink", "hardlink"])
def test_benchmark_pair_aliases_refuse(tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Distinct spellings of one actual output file refuse before replacing its bytes."""
    first = tmp_path / "first.json"
    first.write_text("preserved")
    second = tmp_path / "second.md"
    if kind == "same":
        second = first
    elif kind == "symlink":
        second.symlink_to(first)
    else:
        second.hardlink_to(first)
    assert producer.main(["--json-out", str(first), "--markdown-out", str(second)]) == 1
    assert first.read_text() == "preserved"
    assert "aliases" in capsys.readouterr().err


@pytest.mark.parametrize("json_is_source", [True, False])
def test_benchmark_protects_actual_sources(
    tmp_path: Path, json_is_source: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Either output naming the real plant source refuses without source mutation."""
    source = Path(model.__file__)
    original = source.read_bytes()
    other = tmp_path / "report"
    json_out, md_out = (source, other) if json_is_source else (other, source)
    assert producer.main(["--json-out", str(json_out), "--markdown-out", str(md_out)]) == 1
    assert source.read_bytes() == original and not other.exists()
    assert "aliases" in capsys.readouterr().err


def test_benchmark_partial_pair_io_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A real later Markdown directory failure leaves the already written checked JSON."""
    out = tmp_path / "rzip.json"
    directory = tmp_path / "is-directory"
    directory.mkdir()
    assert producer.main(["--json-out", str(out), "--markdown-out", str(directory)]) == 1
    assert json.loads(out.read_text())["facility_claim_allowed"] is False
    assert directory.is_dir() and "refused" in capsys.readouterr().err
