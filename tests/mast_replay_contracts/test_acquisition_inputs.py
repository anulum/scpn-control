# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST acquisition input and native command contracts
"""Exercise acquisition refusals before real filesystem and network boundaries."""

from __future__ import annotations

import errno
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from validation import acquire_mast_disruption_shots as acquisition


@pytest.mark.parametrize(
    "selection", ["", " ", "30421-30419", "30421 30421", "1-2,2", "0", "-1", "1.5", "1e3", "1-100001", str(2**63)]
)
def test_acquisition_selection_refuses_invalid_domain(selection: str) -> None:
    """Reject ambiguous or oversized selections through the public parser."""
    with pytest.raises(ValueError):
        acquisition.parse_shots(selection)


def test_acquisition_selection_preserves_order_and_inclusive_ranges() -> None:
    """Keep the real parser's valid caller order and int64 endpoint."""
    assert acquisition.parse_shots("30424,30419-30421") == [30424, 30419, 30420, 30421]
    assert acquisition.parse_shots(str(2**63 - 1)) == [2**63 - 1]


@pytest.mark.parametrize("invalid", [[], [0], [-1], [True], [1.5], [1, 1], [2**63]])
def test_acquire_refuses_invalid_batch_before_any_output(tmp_path: Path, invalid: object) -> None:
    """Use the production defaults while refusing invalid identities before I/O."""
    material, cache = tmp_path / "material", tmp_path / "cache"
    with pytest.raises(ValueError):
        acquisition.acquire(
            cast(list[int], invalid), out_dir=material, cache_dir=cache, generated_at="fixed", retrieved_at="fixed"
        )
    assert not material.exists() and not cache.exists()


@pytest.mark.parametrize("invalid", ["", " ", None, 3])
def test_acquire_refuses_invalid_label_before_any_output(tmp_path: Path, invalid: object) -> None:
    """Translate invalid label types/domains before cache or output creation."""
    with pytest.raises(ValueError, match="non-empty reproducibility"):
        acquisition.acquire(
            [30421],
            out_dir=tmp_path / "material",
            cache_dir=tmp_path / "cache",
            generated_at=cast(str, invalid),
            retrieved_at="fixed",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_acquire_protects_actual_runtime_source_alias(tmp_path: Path, link: str) -> None:
    """Refuse a real selected NPZ alias of the runtime owner before network I/O."""
    source = Path(acquisition.__file__).resolve()
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    material = tmp_path / "material"
    material.mkdir()
    target = material / "shot_30421.npz"
    if link == "symlink":
        target.symlink_to(source)
    else:
        try:
            os.link(source, target)
        except OSError as error:
            if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
                raise
            pytest.skip("the filesystem cannot hard-link the checkout into the temporary directory: other volume")
    with pytest.raises(ValueError, match="aliases a selected input"):
        acquisition.acquire(
            [30421], out_dir=material, cache_dir=tmp_path / "cache", generated_at="fixed", retrieved_at="fixed"
        )
    assert not (tmp_path / "cache").exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


@pytest.mark.parametrize("selection", ["", "30421-30419", "30421 30421"])
def test_actual_native_acquisition_command_refuses_before_outputs(tmp_path: Path, selection: str) -> None:
    """Run the real source-module CLI and preserve its fixed refusal/exit2."""
    material, cache, report = tmp_path / "material", tmp_path / "cache", tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.acquire_mast_disruption_shots",
            "--shots",
            selection,
            "--out-dir",
            str(material),
            "--cache-dir",
            str(cache),
            "--manifest-out",
            str(report),
            "--generated-at",
            "fixed",
            "--retrieved-at",
            "fixed",
        ],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "Could not acquire MAST disruption shots."
    assert not material.exists() and not cache.exists() and not report.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_generation_pin_refuses_invalid_byte_count(value: object) -> None:
    """Refuse invalid declared byte counts before any cache-generation record."""
    with pytest.raises(ValueError, match="byte_count"):
        acquisition.SourceGenerationPin("s3://mast/level2/shots/30421.zarr", "a" * 64, cast(int, value), None, None)


def test_generation_pin_keeps_declared_advisory_values_and_fresh_mapping() -> None:
    """Serialize valid immutable declarations without claiming source authentication."""
    pin = acquisition.SourceGenerationPin("s3://mast/level2/shots/30421.zarr", "a" * 64, 64, "etag", "label")
    first, second = pin.to_dict(), pin.to_dict()
    first["sha256"] = "changed"
    assert second["sha256"] == "a" * 64 and second["etag"] == "etag"


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"null",
        b'{"zarr_format":2}',
        b'{"zarr_format":3.0}',
        b'{"zarr_format":3}',
        b'{"zarr_format":3,"zarr_format":3}',
        b'{"zarr_format":3,"x":NaN}',
        b"\xff",
    ],
)
def test_captured_root_decoder_refuses_invalid_actual_bytes(raw: bytes) -> None:
    """Refuse concrete malformed/nonfinite root bytes through the reader's public decoder."""
    with pytest.raises(acquisition.SourceGenerationError):
        acquisition.decode_source_generation(30421, raw)


def test_captured_root_decoder_binds_exact_bytes_without_reencoding() -> None:
    """Keep raw formatting in root hashes and retain advisory headers separately."""
    raw = b'{ "zarr_format":3, "consolidated_metadata":{"kind":"inline","metadata":{}} }'
    pin = acquisition.decode_source_generation(30421, raw, etag="advisory", last_modified="label")
    assert pin.sha256 == hashlib.sha256(raw).hexdigest() and pin.byte_count == len(raw)
    assert pin.etag == "advisory"
    assert acquisition.decode_source_generation(30421, raw + b" ").sha256 != pin.sha256


@pytest.mark.parametrize("identity", [True, 0, -1, 1.5, None, 2**63])
def test_root_reader_and_decoder_refuse_identity_before_network(identity: object) -> None:
    """Refuse invalid identities at both public root entry points."""
    raw = b'{"zarr_format":3,"consolidated_metadata":{"kind":"inline"}}'
    with pytest.raises(acquisition.SourceGenerationError, match="shot_id"):
        acquisition.read_source_generation(cast(int, identity))
    with pytest.raises(acquisition.SourceGenerationError, match="shot_id"):
        acquisition.decode_source_generation(cast(int, identity), raw)


# Explicit ids: the default id of the oversized case is its 16 MiB value, which
# exceeds the length Windows allows for the variable that names the current test.
@pytest.mark.parametrize(
    "raw",
    [None, "{}", bytearray(b"{}"), b" " * ((16 << 20) + 1)],
    ids=["none", "text", "bytearray", "oversized-bytes"],
)
def test_root_decoder_refuses_type_and_physical_size(raw: object) -> None:
    """Bound the actual supplied byte buffer without decoding an oversized input."""
    with pytest.raises(acquisition.SourceGenerationError):
        acquisition.decode_source_generation(30421, cast(bytes, raw))


@pytest.mark.parametrize("uri", [None, "https://example.invalid/shot", "s3://mast/level2/shots/0.zarr"])
def test_generation_pin_refuses_uri_spelling(uri: object) -> None:
    """Keep an invalid source identity out of immutable pin declarations."""
    with pytest.raises(acquisition.SourceGenerationError, match="source_uri"):
        acquisition.SourceGenerationPin(cast(str, uri), "a" * 64, 64, None, None)


@pytest.mark.parametrize("sha", [None, "a" * 63, "A" * 64])
def test_generation_pin_refuses_digest_spelling(sha: object) -> None:
    """Require an exact lowercase digest before a declaration can be serialized."""
    with pytest.raises(acquisition.SourceGenerationError, match="sha256"):
        acquisition.SourceGenerationPin("s3://mast/level2/shots/30421.zarr", cast(str, sha), 64, None, None)


@pytest.mark.parametrize("header", [True, 7])
def test_generation_pin_refuses_nonstring_advisory_header(header: object) -> None:
    """Reject invalid optional header types independently of content identity."""
    with pytest.raises(acquisition.SourceGenerationError, match="advisory"):
        acquisition.SourceGenerationPin("s3://mast/level2/shots/30421.zarr", "a" * 64, 64, cast(str, header), None)


def test_selection_and_batch_limits_refuse_before_output(tmp_path: Path) -> None:
    """Exercise the last single-token and batch-size limits with real identities."""
    with pytest.raises(ValueError, match="100000"):
        acquisition.parse_shots("1-100000 100001")
    with pytest.raises(ValueError, match="100000"):
        acquisition.acquire(
            list(range(1, 100002)),
            out_dir=tmp_path / "material",
            cache_dir=tmp_path / "cache",
            generated_at="fixed",
            retrieved_at="fixed",
        )
    assert not list(tmp_path.iterdir())
