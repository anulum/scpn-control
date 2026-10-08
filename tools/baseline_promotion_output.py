# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Baseline history publication and input custody
"""Publish a mutable baseline and exclusive history names with handled recovery."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from tools.baseline_promotion_payloads import (
    BASELINE_SCHEMA,
    PROMOTION_SCHEMA,
    PromotionInputError,
    _string,
    canonical_digest,
    json_bytes,
)
from tools.inventory_file_output import publish_guarded_outputs


class PromotionCollisionError(FileExistsError):
    """Refuse a reused immutable promotion identifier with authored text.

    Parameters
    ----------
    message : str
        Deliberate refusal, catchable by previous FileExistsError callers.
    """


def publish_promotion(
    baseline: Mapping[str, object],
    *,
    repository_root: Path,
    manifest_path: Path,
    artifact_path: Path,
    baseline_path: Path,
    campaign_id: str,
    artifact_role: str,
    promotion_id: str,
) -> Path:
    """Publish checked baseline bytes and immutable predecessor/receipt files.

    Parameters
    ----------
    baseline : mapping of str to object
        Checked version-one baseline from the public baseline builder.
    repository_root : pathlib.Path
        Selected repository boundary for every output.
    manifest_path, artifact_path : pathlib.Path
        Consumed immutable run inputs, protected from replacement and aliases.
    baseline_path : pathlib.Path
        Mutable repository-relative or absolute output outside run/history trees.
    campaign_id, artifact_role, promotion_id : str
        Declared source campaign, selected artifact role and immutable receipt ID.

    Returns
    -------
    pathlib.Path
        Completed immutable receipt path under the selected history tree.

    Raises
    ------
    PromotionInputError
        Malformed consumed metadata, unsafe namespace or corrupt prior archive.
    PromotionCollisionError
        Receipt ID already has a file, directory or symlink occupant.
    tools.report_inventory_output.InventoryOutputError
        Unsafe/overlapping output or incomplete recovery requiring inspection.
    OSError, ValueError, TypeError
        Native filesystem inspection/publication fails after handled recovery.

    Notes
    -----
    New archives and receipt names are created exclusively from complete staged
    sibling bytes. Baseline replacement precedes receipt publication. Recovery
    is per handled failure, not a power-loss transaction or hostile sandbox;
    cooperating callers must coordinate source and baseline namespaces.
    """
    root = repository_root.resolve()
    identifier = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
    if identifier.fullmatch(promotion_id) is None:
        raise PromotionInputError("promotion identifier must be filesystem-safe")
    if not campaign_id.strip() or not artifact_role.strip():
        raise PromotionInputError("source campaign and artifact role must be non-empty")
    if not manifest_path.resolve().is_relative_to(root) or not artifact_path.resolve().is_relative_to(root):
        raise PromotionInputError("promotion inputs must remain inside the repository")
    destination = baseline_path if baseline_path.is_absolute() else root / baseline_path
    destination = destination.absolute()
    if not destination.resolve().is_relative_to(root):
        raise PromotionInputError("baseline path must remain inside the repository")
    history_namespace = root / "benchmarks/baseline_history"
    source_namespace = next(
        (parent for parent in manifest_path.parents if parent.name == "runs" and parent.is_relative_to(root)), None
    )
    if source_namespace is None:
        raise PromotionInputError("source manifest must be in repository-local immutable runs")
    if destination.resolve().is_relative_to(history_namespace.resolve()):
        raise PromotionInputError("baseline path must remain outside immutable baseline history")
    if destination.resolve().is_relative_to(source_namespace.resolve()):
        raise PromotionInputError("baseline path must remain outside immutable source runs")
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        raise PromotionInputError("baseline payload must use the maintained version-one schema")
    suite = _string(baseline, "suite")
    if identifier.fullmatch(suite) is None:
        raise PromotionInputError("suite must be filesystem-safe")
    if baseline.get("production_claim_allowed") is not False:
        raise PromotionInputError("baseline must not declare production permission")
    promotion = baseline.get("promotion")
    if not isinstance(promotion, dict):
        raise PromotionInputError("baseline promotion metadata must be an object")
    metadata = cast(dict[str, object], promotion)
    source_reference = _string(metadata, "source_manifest")
    source_digest = _string(metadata, "source_artifact_sha256")
    authority = _string(metadata, "authority_ref")
    compatibility = _string(metadata, "hardware_compatibility")
    promoted_utc = _string(metadata, "promoted_utc")
    if source_reference != manifest_path.relative_to(root).as_posix():
        raise PromotionInputError("source manifest reference must match the consumed path")
    if re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise PromotionInputError("source artifact digest must be a lowercase SHA-256")
    if compatibility not in {"matched", "initial-baseline", "reviewed-mismatch"}:
        raise PromotionInputError("hardware compatibility decision is invalid")
    metrics = baseline.get("benchmarks")
    if not isinstance(metrics, dict) or not metrics or baseline.get("baseline_sha256") != canonical_digest(metrics):
        raise PromotionInputError("baseline metric digest must match its declared metrics")
    history = history_namespace / suite
    receipt_path = history / "promotions" / (promotion_id + ".json")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise PromotionCollisionError("promotion identifier already exists")
    baseline_bytes = json_bytes(baseline)
    outputs: list[tuple[Path, bytes]] = [(destination, baseline_bytes)]
    exclusive: list[Path] = []
    protected = [manifest_path, artifact_path]
    previous: dict[str, object] | None = None
    if destination.is_file() and not destination.is_symlink():
        previous_bytes = destination.read_bytes()
        previous_digest = hashlib.sha256(previous_bytes).hexdigest()
        archive_path = history / "baselines" / (previous_digest + ".json")
        if archive_path.exists() or archive_path.is_symlink():
            if archive_path.is_symlink() or not archive_path.is_file() or archive_path.read_bytes() != previous_bytes:
                raise PromotionInputError("previous baseline archive bytes do not match their digest")
            protected.append(archive_path)
        else:
            outputs.append((archive_path, previous_bytes))
            exclusive.append(archive_path)
        previous = {"path": archive_path.relative_to(root).as_posix(), "sha256": previous_digest}
    receipt: dict[str, object] = {
        "schema_version": PROMOTION_SCHEMA,
        "promotion_id": promotion_id,
        "suite": suite,
        "campaign_id": campaign_id,
        "source_manifest": source_reference,
        "source_artifact_role": artifact_role,
        "source_artifact_sha256": source_digest,
        "baseline_path": destination.relative_to(root).as_posix(),
        "baseline_file_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "baseline_metrics_sha256": _string(baseline, "baseline_sha256"),
        "authority_ref": authority,
        "hardware_compatibility": compatibility,
        "promoted_utc": promoted_utc,
        "previous_baseline": previous,
    }
    receipt["payload_sha256"] = canonical_digest(receipt)
    outputs.append((receipt_path, json_bytes(receipt)))
    exclusive.append(receipt_path)
    publish_guarded_outputs(
        tuple(outputs),
        protected_files=tuple(protected),
        protected_roots=(source_namespace,),
        exclusive_files=tuple(exclusive),
    )
    return receipt_path
