# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NPZ pickle-load safety guard tests.
"""Guard that untrusted ``.npz`` archives are never unpickled on load.

A ``.npz`` produced by an attacker can embed a pickled object array whose
``__reduce__`` runs arbitrary code the moment numpy materialises it. Every
first-party reader of a caller-supplied archive therefore opens it with
``allow_pickle=False``; numpy then refuses object arrays instead of executing
them. These tests pin that behaviour at the reachable call sites and forbid the
unsafe flag from reappearing in tracked source.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.neural_turbulence import QLKNNSurrogate
from validation.validate_real_shots import validate_disruption

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_PREFIXES = ("src/", "validation/", "tools/")


def _tracked_python_sources() -> list[Path]:
    """Return tracked first-party ``.py`` files under the scanned prefixes."""
    listing = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in listing.stdout.splitlines() if line.startswith(_SCANNED_PREFIXES)]


def _write_object_array_npz(path: Path, key: str) -> None:
    """Write a one-member ``.npz`` whose array is pickled object data."""
    payload = {key: np.array([{"payload": "arbitrary"}], dtype=object)}
    np.savez(str(path), **payload)  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool


def test_tracked_sources_never_enable_pickle_load() -> None:
    """No tracked first-party source opts into numpy pickle loading."""
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _tracked_python_sources()
        if "allow_pickle=True" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"unsafe pickle load on untrusted .npz: {offenders}"


def test_scan_covers_the_known_reader_modules() -> None:
    """The tracked-source scan actually reaches the hardened reader modules."""
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _tracked_python_sources()}

    assert "src/scpn_control/cli.py" in scanned
    assert "src/scpn_control/core/neural_turbulence.py" in scanned
    assert "validation/validate_real_shots.py" in scanned


def test_load_weights_refuses_pickled_object_array(tmp_path: Path) -> None:
    """QLKNN weight loading rejects a hostile object array rather than unpickling it."""
    hostile = tmp_path / "hostile-weights.npz"
    _write_object_array_npz(hostile, "w0")
    model = QLKNNSurrogate(hidden_layers=[16], pretrained=False)

    with pytest.raises(ValueError, match="allow_pickle=False"):
        model.load_weights(str(hostile))


def test_load_weights_still_round_trips_numeric_weights(tmp_path: Path) -> None:
    """Disabling pickle does not break the legitimate numeric round trip."""
    path = str(tmp_path / "weights.npz")
    model = QLKNNSurrogate(hidden_layers=[16], pretrained=False)
    reference = [w.copy() for w in model.weights]
    model.save_weights(path)

    reloaded = QLKNNSurrogate(hidden_layers=[16], pretrained=False)
    reloaded.load_weights(path)

    for original, restored in zip(reference, reloaded.weights, strict=True):
        np.testing.assert_array_equal(restored, original)


def test_validate_disruption_refuses_pickled_object_array(tmp_path: Path) -> None:
    """The real-shot disruption harness rejects a hostile object array on load."""
    _write_object_array_npz(tmp_path / "hostile-shot.npz", "is_disruption")

    with pytest.raises(ValueError, match="allow_pickle=False"):
        validate_disruption(tmp_path)
