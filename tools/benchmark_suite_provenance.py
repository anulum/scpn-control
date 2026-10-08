# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark suite host context observations.
"""Capture host declarations and represent unavailable observations explicitly."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

from tools.benchmark_suite_metrics import checked_mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_CARGO = REPO_ROOT / "scpn-control-rs/Cargo.toml"
resource_module: ModuleType | None
try:
    resource_module = importlib.import_module("resource")
except ModuleNotFoundError:
    resource_module = None


def _cpu_model(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str:
    """Read the CPU model declaration, falling back to the platform description.

    Parameters
    ----------
    cpuinfo_path : pathlib.Path, optional
        Selected UTF-8 declaration file; defaults to /proc/cpuinfo. Missing,
        unreadable or undecodable input uses the platform description.

    Returns
    -------
    str
        Procfs or platform label, or unknown. This is not host authentication.
    """
    try:
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except (OSError, UnicodeError):
        pass
    return platform.processor() or "unknown"


def _affinity() -> list[int] | None:
    """Observe process CPU affinity when the operating system exposes it.

    Returns
    -------
    list of int or None
        Sorted allowed CPU identifiers, or None for an unavailable observation.

    Notes
    -----
    Reading a mask does not set affinity or establish execution isolation.
    """
    try:
        return sorted(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        return None


def _loadavg() -> list[float] | None:
    """Observe native load averages without inventing a portable replacement.

    Returns
    -------
    list of float or None
        Native one/five/fifteen-minute values, or None if unavailable.
    """
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):  # getloadavg is absent on Windows.
        return None


def _git_commit() -> str:
    """Ask Git for the current source-root HEAD declaration.

    Returns
    -------
    str
        Git output or unknown when unavailable. Uncommitted source and binary
        identity are not bound by this field.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _rust_release_profile(manifest: Path = RUST_CARGO) -> dict[str, object]:
    """Read the workspace release table as a declaration of build settings.

    Parameters
    ----------
    manifest : pathlib.Path, optional
        Selected binary-read TOML workspace manifest. Defaults to the owning
        workspace Cargo.toml; unavailable or unusable release tables yield {}.

    Returns
    -------
    dict of str to object
        TOML release table, or an empty mapping for missing/unusable metadata.

    Notes
    -----
    This does not inspect a loaded extension or attest its compiler flags.
    """
    try:
        with manifest.open("rb") as handle:
            data = tomllib.load(handle)
        profile = data.get("profile")
        if not isinstance(profile, dict):
            return {}
        release = profile.get("release")
        return checked_mapping(release, "release profile") if isinstance(release, dict) else {}
    except (OSError, ValueError):
        return {}


def _peak_rss_mb() -> float | None:
    """Observe process peak resident memory in MiB when resource is available.

    Returns
    -------
    float or None
        Rounded peak RSS, using bytes on Darwin and KiB on other resource hosts.
        None indicates an unavailable observation; it is not a zero measurement.

    Notes
    -----
    The historical key peak_rss_mb is retained although its unit is MiB. No Windows
    resource measurement or actual Darwin receipt is inferred from this function.
    """
    if resource_module is None:
        return None
    raw: object = resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        return None
    scale = 1048576.0 if sys.platform == "darwin" else 1024.0
    return round(float(raw) / scale, 1)
