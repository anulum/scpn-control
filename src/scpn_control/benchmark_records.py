# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Immutable benchmark run records
"""Create immutable benchmark records and digest-bound latest indexes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

RUN_SCHEMA = "scpn-control.benchmark-run.v1"
LATEST_SCHEMA = "scpn-control.benchmark-latest.v1"
CAMPAIGN_ENV = "SCPN_BENCHMARK_CAMPAIGN_ID"

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
_SECRET_OPTION = re.compile(r"(?i)(token|secret|password|credential|api[-_]?key)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_files(path: Path) -> list[Path]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if any(candidate.is_symlink() for candidate in path.rglob("*")):
        raise ValueError(f"benchmark artifact directories cannot contain symlinks: {path}")
    return files


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for file_path in _directory_files(path):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(file_path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in _directory_files(path))


def _validate_identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_IDENTIFIER.pattern}")
    return value


def new_campaign_id() -> str:
    """Return a collision-resistant UTC benchmark campaign identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:12]}"


def redact_command(command: Sequence[str]) -> list[str]:
    """Return command arguments with credential-like values redacted."""
    redacted: list[str] = []
    hide_next = False
    for argument in command:
        if hide_next:
            redacted.append("[REDACTED_SECRET]")
            hide_next = False
            continue
        if argument.startswith("--") and "=" in argument:
            option, value = argument.split("=", 1)
            redacted.append(f"{option}=[REDACTED_SECRET]" if _SECRET_OPTION.search(option) else argument)
            continue
        redacted.append(argument)
        if argument.startswith("-") and _SECRET_OPTION.search(argument):
            hide_next = True
    return redacted


def require_recorded_campaign(*outputs: Path, repository_root: Path) -> str | None:
    """Require wrapper custody before a producer writes persistent evidence.

    Paths outside the repository benchmark evidence roots are treated as
    temporary scratch and do not require a campaign. Persistent destinations
    under ``validation/reports`` or ``artifacts`` fail closed unless
    :data:`CAMPAIGN_ENV` is present.

    Returns
    -------
    str | None
        The shared campaign identifier, or ``None`` for temporary outputs.

    Raises
    ------
    RuntimeError
        If a persistent destination is used outside the recorded runner.
    ValueError
        If a supplied campaign identifier is malformed.
    """
    root = repository_root.resolve()
    persistent_roots = (
        (root / "validation" / "reports").resolve(),
        (root / "artifacts").resolve(),
        (root / "benchmarks").resolve(),
        (root / "gpu_results").resolve(),
    )
    persistent = any(
        output.resolve().is_relative_to(persistent_root) for output in outputs for persistent_root in persistent_roots
    )
    if not persistent:
        return None
    campaign_id = os.environ.get(CAMPAIGN_ENV)
    if campaign_id is None:
        raise RuntimeError(
            f"persistent benchmark output requires tools/run_recorded_benchmark.py; {CAMPAIGN_ENV} is not set"
        )
    return _validate_identifier(campaign_id, "campaign id")


def _git_commit(repository_root: Path) -> str:
    """Resolve HEAD without spawning a command or trusting shell state."""
    try:
        marker = repository_root / ".git"
        if marker.is_file():
            marker_text = marker.read_text(encoding="utf-8").strip()
            prefix = "gitdir: "
            if not marker_text.startswith(prefix):
                return "unknown"
            git_directory = (repository_root / marker_text.removeprefix(prefix)).resolve()
        else:
            git_directory = marker
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head.lower() if re.fullmatch(r"[0-9a-fA-F]{40,64}", head) else "unknown"
        reference = head.removeprefix("ref: ")
        common_directory = git_directory
        common_marker = git_directory / "commondir"
        if common_marker.is_file():
            common_directory = (git_directory / common_marker.read_text(encoding="utf-8").strip()).resolve()
        loose_reference = common_directory / reference
        if loose_reference.is_file():
            candidate = loose_reference.read_text(encoding="utf-8").strip()
            return candidate.lower() if re.fullmatch(r"[0-9a-fA-F]{40,64}", candidate) else "unknown"
        packed_references = common_directory / "packed-refs"
        if packed_references.is_file():
            suffix = f" {reference}"
            for line in packed_references.read_text(encoding="utf-8").splitlines():
                if line.endswith(suffix):
                    candidate = line.split(" ", 1)[0]
                    return candidate.lower() if re.fullmatch(r"[0-9a-fA-F]{40,64}", candidate) else "unknown"
    except OSError:
        return "unknown"
    return "unknown"


def _dependency_lock_digest(repository_root: Path) -> tuple[str, list[str]]:
    candidates = (
        Path("uv.lock"),
        Path("requirements/ci-test.txt"),
        Path("scpn-control-rs/Cargo.lock"),
    )
    digest = hashlib.sha256()
    included: list[str] = []
    for relative_path in candidates:
        path = repository_root / relative_path
        if not path.is_file():
            continue
        included.append(relative_path.as_posix())
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), included


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _host_context() -> dict[str, Any]:
    try:
        affinity: list[int] | None = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    try:
        load_average: list[float] | None = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "load_average": load_average,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "isolation": os.environ.get("SCPN_BENCHMARK_ISOLATION", "unspecified"),
        "concurrent_heavy_jobs": os.environ.get("SCPN_BENCHMARK_CONCURRENT_HEAVY_JOBS", "unspecified"),
    }


def _display_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    with destination.open("xb") as target, source.open("rb") as source_handle:
        shutil.copyfileobj(source_handle, target)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _write_exclusive(temporary, data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class BenchmarkOutput:
    """Describe one output produced by a recorded benchmark command.

    Parameters
    ----------
    role:
        Stable descriptive role such as ``report`` or ``markdown``.
    path:
        Producer destination. Existing bytes are retained before execution.
    """

    role: str
    path: Path

    def __post_init__(self) -> None:
        _validate_identifier(self.role, "output role")


@dataclass
class BenchmarkRun:
    """Own one immutable benchmark campaign from reservation to finalisation.

    Instances are created with :meth:`begin`. A run directory is reserved
    before the producer executes, and :meth:`finish` records every resulting
    artifact plus success or failure. Only successful runs advance ``latest``.
    """

    repository_root: Path
    records_root: Path
    family: str
    campaign_id: str
    run_directory: Path
    outputs: tuple[BenchmarkOutput, ...]
    command: tuple[str, ...]
    started_utc: str
    source_commit: str
    dependency_lock_sha256: str
    dependency_locks: tuple[str, ...]
    host_start: Mapping[str, Any]
    evidence_class: str
    measurement: Mapping[str, Any]
    legacy: tuple[Mapping[str, Any], ...]

    @classmethod
    def begin(
        cls,
        *,
        repository_root: Path,
        records_root: Path,
        family: str,
        outputs: Sequence[BenchmarkOutput],
        command: Sequence[str],
        campaign_id: str | None = None,
        evidence_class: str = "local_regression",
        measurement: Mapping[str, Any] | None = None,
    ) -> BenchmarkRun:
        """Reserve an immutable run and preserve existing output bytes.

        Raises
        ------
        FileExistsError
            If the requested campaign identifier already exists.
        ValueError
            If identifiers or output roles are invalid or duplicated.
        """
        root = repository_root.resolve()
        custody = records_root.resolve()
        run_family = _validate_identifier(family, "benchmark family")
        run_id = _validate_identifier(campaign_id or os.environ.get(CAMPAIGN_ENV) or new_campaign_id(), "campaign id")
        normalised_outputs = tuple(outputs)
        roles = [output.role for output in normalised_outputs]
        if len(set(roles)) != len(roles):
            raise ValueError("benchmark output roles must be unique")

        run_directory = custody / "runs" / run_family / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        legacy_entries: list[Mapping[str, Any]] = []
        for output in normalised_outputs:
            if not output.path.exists():
                continue
            digest = _sha256_path(output.path)
            suffix = (output.path.suffix or ".bin") if output.path.is_file() else ""
            legacy_path = custody / "legacy" / digest / f"{output.role}{suffix}"
            if not legacy_path.exists():
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                if output.path.is_dir():
                    try:
                        shutil.copytree(output.path, legacy_path)
                    except FileExistsError:
                        pass
                else:
                    try:
                        _copy_file_exclusive(output.path, legacy_path)
                    except FileExistsError:
                        pass
            if _sha256_path(legacy_path) != digest:
                raise RuntimeError(f"legacy benchmark digest mismatch at {legacy_path}")
            legacy_entries.append(
                {
                    "role": output.role,
                    "source_path": _display_path(output.path, root),
                    "archived_path": _display_path(legacy_path, root),
                    "sha256": digest,
                }
            )

        lock_digest, lock_paths = _dependency_lock_digest(root)
        return cls(
            repository_root=root,
            records_root=custody,
            family=run_family,
            campaign_id=run_id,
            run_directory=run_directory,
            outputs=normalised_outputs,
            command=tuple(redact_command(command)),
            started_utc=_utc_now(),
            source_commit=_git_commit(root),
            dependency_lock_sha256=lock_digest,
            dependency_locks=tuple(lock_paths),
            host_start=_host_context(),
            evidence_class=evidence_class,
            measurement=dict(measurement or {}),
            legacy=tuple(legacy_entries),
        )

    def finish(self, *, exit_code: int) -> Path:
        """Seal output artifacts and return the immutable manifest path.

        A zero exit code with every declared output present is successful.
        Failed or incomplete runs remain preserved but cannot update the
        digest-bound latest index.
        """
        manifest_path = self.run_directory / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"benchmark run is already finalised: {manifest_path}")

        artifact_entries: list[dict[str, Any]] = []
        missing_roles: list[str] = []
        for output in self.outputs:
            if not output.path.exists():
                missing_roles.append(output.role)
                continue
            suffix = (output.path.suffix or ".bin") if output.path.is_file() else ""
            immutable_path = self.run_directory / "artifacts" / f"{output.role}{suffix}"
            if output.path.is_dir():
                shutil.copytree(output.path, immutable_path)
            else:
                _write_exclusive(immutable_path, output.path.read_bytes())
            artifact_entries.append(
                {
                    "role": output.role,
                    "kind": "directory" if output.path.is_dir() else "file",
                    "source_path": _display_path(output.path, self.repository_root),
                    "immutable_path": _display_path(immutable_path, self.repository_root),
                    "size_bytes": _path_size(output.path),
                    "sha256": _sha256_path(output.path),
                }
            )

        succeeded = exit_code == 0 and not missing_roles
        manifest: dict[str, Any] = {
            "schema_version": RUN_SCHEMA,
            "campaign_id": self.campaign_id,
            "benchmark_family": self.family,
            "status": "succeeded" if succeeded else "failed",
            "exit_code": exit_code,
            "started_utc": self.started_utc,
            "finished_utc": _utc_now(),
            "evidence_class": self.evidence_class,
            "production_claim_allowed": False,
            "source_commit": self.source_commit,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "dependency_locks": list(self.dependency_locks),
            "command": list(self.command),
            "measurement": dict(self.measurement),
            "host_start": dict(self.host_start),
            "host_end": _host_context(),
            "artifacts": artifact_entries,
            "missing_output_roles": missing_roles,
            "legacy_inputs": list(self.legacy),
        }
        unsigned = _json_bytes(manifest)
        manifest["payload_sha256"] = _sha256_bytes(unsigned)
        manifest_bytes = _json_bytes(manifest)
        _write_exclusive(manifest_path, manifest_bytes)

        if succeeded:
            manifest_sha256 = _sha256_bytes(manifest_bytes)
            latest = {
                "schema_version": LATEST_SCHEMA,
                "benchmark_family": self.family,
                "campaign_id": self.campaign_id,
                "manifest_path": manifest_path.relative_to(self.records_root).as_posix(),
                "manifest_sha256": manifest_sha256,
                "artifact_sha256": {entry["role"]: entry["sha256"] for entry in artifact_entries},
                "updated_utc": _utc_now(),
            }
            _atomic_replace(self.records_root / "latest" / f"{self.family}.json", _json_bytes(latest))
        return manifest_path


def load_verified_latest(records_root: Path, family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a latest index and its manifest after verifying the bound digest.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The latest index and the referenced immutable run manifest.

    Raises
    ------
    ValueError
        If the index schema, family, path, or digest is invalid.
    """
    run_family = _validate_identifier(family, "benchmark family")
    root = records_root.resolve()
    latest_path = root / "latest" / f"{run_family}.json"
    latest: dict[str, Any] = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("schema_version") != LATEST_SCHEMA or latest.get("benchmark_family") != run_family:
        raise ValueError("benchmark latest index schema or family mismatch")
    manifest_path = root / Path(str(latest.get("manifest_path", "")))
    resolved_manifest = manifest_path.resolve()
    runs_root = (root / "runs").resolve()
    if not resolved_manifest.is_relative_to(runs_root):
        raise ValueError("benchmark latest manifest escapes the immutable runs root")
    manifest_bytes = resolved_manifest.read_bytes()
    if _sha256_bytes(manifest_bytes) != latest.get("manifest_sha256"):
        raise ValueError("benchmark latest manifest digest mismatch")
    manifest: dict[str, Any] = json.loads(manifest_bytes)
    if manifest.get("status") != "succeeded" or manifest.get("campaign_id") != latest.get("campaign_id"):
        raise ValueError("benchmark latest references an inadmissible run")
    return latest, manifest


__all__ = [
    "BenchmarkOutput",
    "BenchmarkRun",
    "CAMPAIGN_ENV",
    "LATEST_SCHEMA",
    "RUN_SCHEMA",
    "load_verified_latest",
    "new_campaign_id",
    "redact_command",
    "require_recorded_campaign",
]
