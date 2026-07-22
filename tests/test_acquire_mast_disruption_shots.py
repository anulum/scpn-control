# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST level2 high-fidelity acquisition
"""Offline tests for :mod:`validation.acquire_mast_disruption_shots`."""

from __future__ import annotations

import hashlib
import json
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from validation import acquire_mast_disruption_shots as acq
from validation import mast_source_object_manifest as source_manifest

_ANGLES = np.tile((np.arange(12, dtype=np.float64) * 30.0).reshape(-1, 1), (1, 28))


def _generation(shot_id: int, *, sha256: str | None = None) -> acq.SourceGenerationPin:
    return acq.SourceGenerationPin(
        source_uri=f"s3://mast/level2/shots/{shot_id}.zarr",
        sha256=sha256 or f"{shot_id:064x}",
        byte_count=742024,
        etag=f'"etag-{shot_id}"',
        last_modified="Thu, 18 Jun 2026 15:57:36 GMT",
    )


def _stable_generation_reader(shot_id: int) -> acq.SourceGenerationPin:
    return _generation(shot_id)


class _FakeVar:
    def __init__(self, values: NDArray[np.float64], key: str) -> None:
        self.values = values
        self.dims: tuple[str, ...]
        if values.ndim == 2:
            self.dims = ("channel", "time_saddle" if "field" in key else "sample")
        else:
            self.dims = ("time",)
        self.attrs = {"units": "arb", "long_name": key}
        self.chunks: None = None


class _FakeDataset:
    def __init__(self, variables: dict[str, NDArray[np.float64]]) -> None:
        self.variables = variables

    def __getitem__(self, key: str) -> _FakeVar:
        return _FakeVar(self.variables[key], key)


def _group_vars(group: str, *, with_saddle: bool = True) -> dict[str, NDArray[np.float64]]:
    if group == "summary":
        return {"time": np.linspace(0, 0.06, 60), "ip": np.full(60, 5.0e5), "line_average_n_e": np.full(60, 3.0e19)}
    if group == "equilibrium":
        return {
            "time": np.linspace(0, 0.06, 20),
            "q95": np.full(20, 3.8),
            "beta_tor_normal": np.full(20, 1.5),
            "magnetic_axis_z": np.linspace(-0.02, 0.02, 20),
        }
    if group == "interferometer":
        return {"time": np.linspace(0, 0.06, 100), "n_e_line": np.full(100, 3.0e19)}
    if group == "magnetics":
        base: dict[str, NDArray[np.float64]] = {"time_saddle": np.linspace(0, 0.06, 500, dtype=np.float64)}
        if with_saddle:
            base["b_field_tor_probe_saddle_field"] = np.ones((12, 500))
            base["b_field_tor_probe_saddle_m_phi"] = _ANGLES
        return base
    return {}


def _open_group(with_saddle: bool = True) -> Any:
    def opener(_fs: Any, _shot_id: int, group: str) -> _FakeDataset:
        return _FakeDataset(_group_vars(group, with_saddle=with_saddle))

    return opener


# --------------------------------------------------------------------------- #
# mirror_shot
# --------------------------------------------------------------------------- #
def test_mirror_shot_collects_present_variables() -> None:
    """The acquisition surface keeps exact present arrays and source metadata."""
    metadata: dict[str, dict[str, Any]] = {}
    payload = acq.mirror_shot(object(), 30421, open_group=_open_group(), metadata_out=metadata)
    assert "magnetics.b_field_tor_probe_saddle_field" in payload
    assert "summary.ip" in payload
    assert payload["equilibrium.q95"].shape == (20,)
    assert payload["equilibrium.magnetic_axis_z"].shape == (20,)
    assert metadata["summary.ip"]["dimensions"] == ["time"]
    assert metadata["summary.ip"]["timebase"] == {"kind": "source_dimension", "dimensions": ["time"]}


def test_mirror_shot_requires_saddle_array() -> None:
    """A shot without the required toroidal saddle array fails closed."""
    with pytest.raises(ValueError, match="no toroidal saddle array"):
        acq.mirror_shot(object(), 30421, open_group=_open_group(with_saddle=False))


# --------------------------------------------------------------------------- #
# acquire (directory + manifest)
# --------------------------------------------------------------------------- #
def test_acquire_writes_mirrors_and_manifest(tmp_path: Path) -> None:
    """Acquisition writes verified NPZ caches and a v2 lineage manifest."""
    manifest = acq.acquire(
        [30421, 30424],
        out_dir=tmp_path / "material",
        cache_dir=tmp_path / "cache",
        generated_at="2026-07-10T00:00:00Z",
        retrieved_at="2026-07-10T00:00:00Z",
        make_fs=lambda _c: object(),
        open_group=_open_group(),
        read_generation=_stable_generation_reader,
    )
    assert manifest["schema_version"] == "scpn-control.source-object-manifest.v2.0.0"
    assert manifest["manifest_kind"] == "source_object_inventory"
    assert manifest["machine"] == "MAST"
    assert manifest["n_acquired"] == 2
    assert manifest["synthetic"] is False
    assert manifest["consumers"] == ["SCPN-CONTROL", "SCPN-FUSION-CORE", "MIF-CORE"]
    assert manifest["licence"] == "CC-BY-SA-4.0"
    assert manifest["licence_spdx"] == "CC-BY-SA-4.0"
    assert manifest["licence_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert len(manifest["citations"]) == 2
    assert "10.1109/TPS.2025.3583419" in manifest["citation"]
    assert manifest["source_policy_url"] == "https://mastapp.site/"
    assert manifest["payload_sha256"] == source_manifest.canonical_json_sha256({**manifest, "payload_sha256": None})
    record = next(r for r in manifest["shots"] if r["shot_id"] == 30421)
    assert record["summary"]["saddle_channels"] == 12 and record["summary"]["saddle_samples"] == 500
    artifact = record["artifacts"][0]
    assert artifact["artifact_kind"] == "derived_npz_cache"
    assert artifact["local_path"] == "shot_30421.npz"
    assert len(artifact["sha256"]) == 64
    assert artifact["parent_digest"] == artifact["parent"]["sha256"]
    assert artifact["transform_digest"] == artifact["transform"]["sha256"]
    assert artifact["fidelity"]["native_zarr_bytes_preserved"] is False
    assert artifact["fidelity"]["source_generation_pinned"] is True
    assert artifact["source_generation"] == artifact["parent"]["descriptor"]["source_generation"]
    assert artifact["source_generation"]["sha256"] == f"{30421:064x}"
    assert record["cache_generation"]["existing_cache_reused"] is False
    assert record["cache_generation"]["pre_and_post_source_generation_match"] is True
    assert manifest["cache_policy"]["persistent_cross_run_reuse"] is False
    saddle = next(array for array in artifact["arrays"] if array["archive_key"].endswith("saddle_field"))
    assert saddle["group"] == "magnetics"
    assert saddle["dimensions"] == ["channel", "time_saddle"]
    assert saddle["units"] == "arb"
    assert saddle["metadata_status"] == "source_xarray"
    with np.load(tmp_path / "material" / "shot_30421.npz", allow_pickle=False) as loaded:
        assert "magnetics.b_field_tor_probe_saddle_field" in loaded.files
    source_manifest.validate_source_object_manifest(manifest, artifact_root=tmp_path / "material")


def test_acquire_records_failed_shot(tmp_path: Path) -> None:
    """A failed source read remains an explicit failed manifest record."""
    manifest = acq.acquire(
        [30421],
        out_dir=tmp_path / "material",
        cache_dir=tmp_path / "cache",
        generated_at="2026-07-10T00:00:00Z",
        retrieved_at="2026-07-10T00:00:00Z",
        make_fs=lambda _c: object(),
        open_group=_open_group(with_saddle=False),
        read_generation=_stable_generation_reader,
    )
    assert manifest["n_acquired"] == 0
    assert manifest["status"] == "empty"
    assert manifest["shots"][0]["status"] == "failed"
    assert manifest["shots"][0]["programme_class"] == "unknown"


# --------------------------------------------------------------------------- #
# make_filesystem / _open_group (S3 seam via stubbed modules)
# --------------------------------------------------------------------------- #
def test_make_filesystem_uses_fsspec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache filesystem is anonymous and rooted at the caller cache path."""
    calls: dict[str, Any] = {}
    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.filesystem = lambda protocol, **kw: calls.update({"protocol": protocol, **kw}) or "FS"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)
    fs = acq.make_filesystem(tmp_path / "cache")
    assert fs == "FS"
    assert calls["protocol"] == "simplecache"
    assert calls["target_options"]["anon"] is True


def test_read_source_generation_hashes_exact_uncached_root_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation identity comes from bounded exact Zarr-v3 root metadata bytes."""
    raw = json.dumps({"zarr_format": 3, "consolidated_metadata": {"kind": "inline", "metadata": {}}}).encode()

    class _Response:
        headers = {"ETag": '"etag"', "Last-Modified": "Thu, 18 Jun 2026 15:57:36 GMT"}

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return raw

    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        seen.update(url=request.full_url, timeout=timeout)
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    pin = acq.read_source_generation(30421)
    assert pin.source_uri == "s3://mast/level2/shots/30421.zarr"
    assert pin.sha256 == hashlib.sha256(raw).hexdigest()
    assert pin.byte_count == len(raw)
    assert pin.etag == '"etag"'
    assert seen["url"].endswith("/mast/level2/shots/30421.zarr/zarr.json")
    assert seen["timeout"] == 30.0


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"{", "invalid upstream root metadata"),
        (b'{"zarr_format":2}', "not Zarr format 3"),
        (b'{"zarr_format":3}', "not inline consolidated"),
        (b'{"zarr_format":3,"zarr_format":3}', "duplicate JSON key"),
    ],
)
def test_read_source_generation_rejects_inadmissible_metadata(
    raw: bytes,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed, ambiguous, or non-consolidated roots never reach simplecache."""

    class _Response:
        headers: dict[str, str] = {}

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return raw

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())
    with pytest.raises(acq.SourceGenerationError, match=match):
        acq.read_source_generation(30421)


def test_read_source_generation_rejects_oversized_or_unreadable_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct generation probe is bounded and translates transport failures."""

    class _OversizedResponse:
        headers: dict[str, str] = {}

        def __enter__(self) -> _OversizedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            return b"x" * limit

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: _OversizedResponse())
    with pytest.raises(acq.SourceGenerationError, match="exceeds"):
        acq.read_source_generation(30421)

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(acq.SourceGenerationError, match="cannot read"):
        acq.read_source_generation(30421)


@pytest.mark.parametrize("shot_id", [0, -1, True, "30421"])
def test_read_source_generation_rejects_invalid_shot_id(shot_id: Any) -> None:
    """Only positive integer shot identifiers may reach the endpoint."""
    with pytest.raises(acq.SourceGenerationError, match="positive integer"):
        acq.read_source_generation(shot_id)


def test_acquire_rejects_source_generation_drift_before_npz_write(tmp_path: Path) -> None:
    """A changed root metadata identity makes the shot fail closed."""
    calls = 0

    def drifting_reader(shot_id: int) -> acq.SourceGenerationPin:
        nonlocal calls
        calls += 1
        return _generation(shot_id, sha256=("a" if calls == 1 else "b") * 64)

    manifest = acq.acquire(
        [30421],
        out_dir=tmp_path / "material",
        cache_dir=tmp_path / "cache",
        generated_at="2026-07-10T00:00:00Z",
        retrieved_at="2026-07-10T00:00:00Z",
        make_fs=lambda _c: object(),
        open_group=_open_group(),
        read_generation=drifting_reader,
    )

    assert manifest["n_acquired"] == 0
    assert "changed while shot 30421" in manifest["shots"][0]["error"]
    assert not (tmp_path / "material" / "shot_30421.npz").exists()


def test_acquire_ignores_advisory_header_drift_when_exact_bytes_match(tmp_path: Path) -> None:
    """ETag/header churn cannot masquerade as a source-byte generation change."""
    calls = 0

    def header_drift_reader(shot_id: int) -> acq.SourceGenerationPin:
        nonlocal calls
        calls += 1
        pin = _generation(shot_id)
        return acq.SourceGenerationPin(
            source_uri=pin.source_uri,
            sha256=pin.sha256,
            byte_count=pin.byte_count,
            etag=f'"etag-{calls}"',
            last_modified=f"revision-{calls}",
        )

    manifest = acq.acquire(
        [30421],
        out_dir=tmp_path / "material",
        cache_dir=tmp_path / "cache",
        generated_at="2026-07-10T00:00:00Z",
        retrieved_at="2026-07-10T00:00:00Z",
        make_fs=lambda _c: object(),
        open_group=_open_group(),
        read_generation=header_drift_reader,
    )

    assert manifest["n_acquired"] == 1
    assert manifest["shots"][0]["artifacts"][0]["source_generation"]["etag"] == '"etag-1"'


def test_acquire_refuses_cross_run_cache_namespace_reuse(tmp_path: Path) -> None:
    """Identical run labels cannot reopen a persistent cache namespace."""
    kwargs = {
        "cache_dir": tmp_path / "cache",
        "generated_at": "2026-07-10T00:00:00Z",
        "retrieved_at": "2026-07-10T00:00:00Z",
        "make_fs": lambda _c: object(),
        "open_group": _open_group(),
        "read_generation": _stable_generation_reader,
    }
    first = acq.acquire([30421], out_dir=tmp_path / "first", **kwargs)
    second = acq.acquire([30421], out_dir=tmp_path / "second", **kwargs)

    assert first["n_acquired"] == 1
    assert second["n_acquired"] == 0
    assert "refusing cross-run cache reuse" in second["shots"][0]["error"]
    assert not (tmp_path / "second" / "shot_30421.npz").exists()


@pytest.mark.parametrize(("generated_at", "retrieved_at"), [("", "x"), ("x", " ")])
def test_acquire_requires_nonempty_reproducibility_labels(
    tmp_path: Path,
    generated_at: str,
    retrieved_at: str,
) -> None:
    """Blank acquisition labels are rejected before filesystem mutation."""
    with pytest.raises(ValueError, match="must be non-empty"):
        acq.acquire(
            [30421],
            out_dir=tmp_path / "material",
            cache_dir=tmp_path / "cache",
            generated_at=generated_at,
            retrieved_at=retrieved_at,
            read_generation=_stable_generation_reader,
        )
    assert not tmp_path.joinpath("material").exists()


def test_open_group_uses_xarray(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source opener selects the exact shot group through consolidated Zarr."""
    seen: dict[str, Any] = {}

    def _open_zarr(store: Any, group: str, consolidated: bool) -> str:
        seen.update({"store": store, "group": group, "consolidated": consolidated})
        return "DS"

    fake_xr = types.ModuleType("xarray")
    fake_xr.open_zarr = _open_zarr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarray", fake_xr)

    class _FS:
        def get_mapper(self, url: str) -> str:
            seen["url"] = url
            return "MAP"

    assert acq._open_group(_FS(), 30421, "magnetics") == "DS"
    assert seen["group"] == "magnetics" and seen["consolidated"] is True
    assert seen["url"].endswith("level2/shots/30421.zarr")


# --------------------------------------------------------------------------- #
# main (default seams resolved dynamically for monkeypatching)
# --------------------------------------------------------------------------- #
def test_main_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI parses shots and writes its production manifest output."""
    monkeypatch.setattr(acq, "make_filesystem", lambda _c: object())
    monkeypatch.setattr(acq, "_open_group", lambda _fs, _s, g: _FakeDataset(_group_vars(g)))
    monkeypatch.setattr(acq, "read_source_generation", _stable_generation_reader)
    manifest_out = tmp_path / "manifest.json"
    code = acq.main(
        [
            "--shots",
            "30419-30420 30424",
            "--out-dir",
            str(tmp_path / "material"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--manifest-out",
            str(manifest_out),
            "--generated-at",
            "2026-07-10T00:00:00Z",
            "--retrieved-at",
            "2026-07-10T00:00:00Z",
        ]
    )
    assert code == 0
    assert json.loads(manifest_out.read_text(encoding="utf-8"))["n_acquired"] == 3


def test_parse_shots_ranges_and_ids() -> None:
    """Shot-range parsing preserves the requested deterministic order."""
    assert acq._parse_shots("11766 30419-30421, 29876") == [11766, 30419, 30420, 30421, 29876]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (3, 3),
        (1.25, 1.25),
        (float("inf"), {"non_finite_float": "inf"}),
        (np.int64(4), 4),
        (np.asarray([1, 2]), [1, 2]),
        (b"ab", {"bytes_hex": "6162"}),
        ({2: (np.float32(1.5),)}, {"2": [1.5]}),
    ],
)
def test_json_metadata_value_preserves_supported_types(value: Any, expected: Any) -> None:
    """Supported metadata types retain their JSON information content."""
    assert acq._json_metadata_value(value) == expected


def test_json_metadata_value_rejects_unsupported_types() -> None:
    """Unsupported source attributes fail instead of stringifying silently."""
    with pytest.raises(TypeError, match="unsupported source metadata type"):
        acq._json_metadata_value(object())


def test_source_array_metadata_records_chunks_without_inventing_units_or_timebase() -> None:
    """Metadata capture records chunks and preserves absent physical semantics."""
    source = types.SimpleNamespace(dims=("channel",), attrs={"calibration": 1}, chunks=((2,),))
    metadata = acq._source_array_metadata(source)
    assert metadata["units"] is None
    assert metadata["timebase"] is None
    assert metadata["source_chunks"] == [[2]]
