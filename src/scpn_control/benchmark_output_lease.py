# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark output reservation
"""Reserve disjoint output namespaces across cooperating benchmark processes."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@contextmanager
def _registry_lock(root: Path) -> Iterator[None]:
    """Serialise reservation updates with a process-owned operating-system lock."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "registry.lock").open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class BenchmarkOutputLease:
    """Reserve output paths until finalisation, retaining interrupted ownership.

    Attributes
    ----------
    marker : Path
        Reservation metadata under the repository's benchmark lease registry.

    Notes
    -----
    All campaigns for a repository use the same registry, independently of their
    chosen records root. Interrupted reservations deliberately fail closed until
    their outputs and unsealed records are recovered; PID reuse is not recovery
    authority. This coordinates recorded producers, not hostile local writers.
    """

    marker: Path

    @classmethod
    def acquire(cls, repository_root: Path, outputs: Sequence[Path], run_directory: Path) -> BenchmarkOutputLease:
        """Reserve nonoverlapping paths before any output can be displaced.

        Parameters
        ----------
        repository_root : Path
            Canonical repository shared by participating producers.
        outputs : Sequence[Path]
            Exact file or directory destinations for this invocation.
        run_directory : Path
            Reserved run directory identifying recovery evidence.

        Returns
        -------
        BenchmarkOutputLease
            Exclusive ownership of the declared output namespaces.

        Raises
        ------
        ValueError
            If no outputs, duplicate/overlapping paths or custody overlap exist.
        RuntimeError
            If an active or interrupted campaign already owns an output.
        """
        root = repository_root.resolve() / "artifacts/benchmarks/output-leases"
        paths = tuple(path.resolve() for path in outputs)
        if not paths:
            raise ValueError("a benchmark run requires at least one output")
        for index, path in enumerate(paths):
            if root.is_relative_to(path) or path.is_relative_to(root):
                raise ValueError("benchmark output overlaps the lease registry")
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[:index]):
                raise ValueError("benchmark output paths must not overlap")
        with _registry_lock(root):
            for marker in root.glob("*.json"):
                record = json.loads(marker.read_text())
                held = record.get("outputs") if isinstance(record, dict) else None
                if (
                    not isinstance(held, list)
                    or not held
                    or not all(isinstance(path, str) and Path(path).is_absolute() for path in held)
                ):
                    raise RuntimeError(f"invalid benchmark reservation: {marker}")
                if any(
                    path.is_relative_to(Path(other)) or Path(other).is_relative_to(path)
                    for path in paths
                    for other in held
                ):
                    raise RuntimeError(f"benchmark outputs already reserved: {marker}")
            marker = root / f"{uuid4().hex}.json"
            with marker.open("x") as handle:
                json.dump(
                    {"outputs": [str(path) for path in paths], "run_directory": str(run_directory), "pid": os.getpid()},
                    handle,
                )
        return cls(marker)

    def release(self) -> None:
        """Release only this invocation's reservation after output finalisation."""
        with _registry_lock(self.marker.parent):
            self.marker.unlink(missing_ok=True)
