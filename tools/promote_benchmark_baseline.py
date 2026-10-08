# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Explicit benchmark baseline promotion
"""Promote digest-bound benchmark declarations while preserving input/history custody."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scpn_control.benchmark_records import RUN_SCHEMA as RUN_SCHEMA
from tools.baseline_promotion_output import PromotionCollisionError, publish_promotion
from tools.baseline_promotion_payloads import BASELINE_SCHEMA as BASELINE_SCHEMA
from tools.baseline_promotion_payloads import PROMOTION_SCHEMA as PROMOTION_SCHEMA
from tools.baseline_promotion_payloads import REPORT_SCHEMA as REPORT_SCHEMA
from tools.baseline_promotion_payloads import PromotionInputError, _string
from tools.baseline_promotion_payloads import build_baseline as build_baseline
from tools.baseline_promotion_records import _repository_path as _repository_path
from tools.baseline_promotion_records import load_promotion_source
from tools.inventory_file_output import InventoryOutputError

REPO_ROOT = Path(__file__).resolve().parents[1]
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def promote(
    *,
    source_manifest: Path,
    artifact_role: str,
    expected_source_sha256: str,
    baseline_path: Path,
    suite: str,
    authority_ref: str,
    hardware_compatibility: str,
    promotion_id: str,
    repository_root: Path = REPO_ROOT,
) -> Path:
    """Verify declarations and publish a recoverable baseline/history update.

    Parameters
    ----------
    source_manifest : pathlib.Path
        Repository-local successful immutable run envelope.
    artifact_role : str
        Exactly one selected immutable file role.
    expected_source_sha256 : str
        Caller-supplied lowercase artifact-byte SHA-256.
    baseline_path : pathlib.Path
        Mutable repository-local baseline outside immutable source/history trees.
    suite, promotion_id : str
        Filesystem-safe identifiers; the receipt ID cannot be reused.
    authority_ref : str
        Nonblank caller reference, without owner authentication.
    hardware_compatibility : str
        Declared matched, initial-baseline or reviewed-mismatch decision.
    repository_root : pathlib.Path, optional
        Repository boundary; explicit paths support isolated owned custody copies.

    Returns
    -------
    pathlib.Path
        Completed immutable promotion receipt after all publication succeeds.

    Raises
    ------
    PromotionInputError, PromotionCollisionError, InventoryOutputError
        Invalid metadata, immutable collision or unsafe/incomplete publication.
    OSError, ValueError, TypeError
        Native read/decode/publication failure, after handled recovery.

    Notes
    -----
    Source digest and envelope consistency do not prove fresh measurements,
    physical validity, hardware equivalence or authority. Source/baseline writers
    must coordinate; handled recovery does not give global crash atomicity.
    """
    root = repository_root.resolve()
    if _IDENTIFIER.fullmatch(suite) is None or _IDENTIFIER.fullmatch(promotion_id) is None:
        raise PromotionInputError("suite and promotion identifier must be filesystem-safe identifiers")
    if not authority_ref.strip():
        raise PromotionInputError("authority reference must not be empty")
    if hardware_compatibility not in {"matched", "initial-baseline", "reviewed-mismatch"}:
        raise PromotionInputError("hardware compatibility decision is invalid")
    if _SHA256.fullmatch(expected_source_sha256) is None:
        raise PromotionInputError("expected source digest must be a lowercase SHA-256 digest")
    manifest_path = _repository_path(source_manifest, "source manifest", root)
    manifest, artifact_path, digest, report = load_promotion_source(
        manifest_path,
        artifact_role,
        expected_source_sha256,
        root,
    )
    campaign_id = _string(manifest, "campaign_id")
    promoted_utc = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    baseline = build_baseline(
        report,
        suite=suite,
        source_manifest=manifest_path.relative_to(root).as_posix(),
        source_sha256=digest,
        authority_ref=authority_ref,
        hardware_compatibility=hardware_compatibility,
        promoted_utc=promoted_utc,
    )
    return publish_promotion(
        baseline,
        repository_root=root,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        baseline_path=baseline_path,
        campaign_id=campaign_id,
        artifact_role=artifact_role,
        promotion_id=promotion_id,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse promotion metadata and expose only authored or fixed caller errors.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments, or process arguments when omitted.

    Returns
    -------
    int
        Zero after a complete receipt, one after a supported refusal/failure.

    Raises
    ------
    SystemExit
        Invalid parser arguments or requested help.

    Notes
    -----
    No operator approval is derived from a reference string. Caller output does
    not interpolate caught filesystem, decoder or interpreter error details.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--artifact-role", default="report")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--authority-ref", required=True)
    parser.add_argument(
        "--hardware-compatibility", required=True, choices=("matched", "initial-baseline", "reviewed-mismatch")
    )
    parser.add_argument("--promotion-id", default="")
    args = parser.parse_args(argv)
    identifier = args.promotion_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    try:
        receipt = promote(
            source_manifest=args.source_manifest,
            artifact_role=args.artifact_role,
            expected_source_sha256=args.expected_source_sha256,
            baseline_path=args.baseline,
            suite=args.suite,
            authority_ref=args.authority_ref,
            hardware_compatibility=args.hardware_compatibility,
            promotion_id=identifier,
            repository_root=args.repository_root,
        )
    except (PromotionInputError, PromotionCollisionError, InventoryOutputError) as error:
        print(f"baseline promotion FAILED: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError):
        print("baseline promotion FAILED: inputs or outputs could not be inspected", file=sys.stderr)
        return 1
    print(f"baseline promotion receipt: {receipt.relative_to(args.repository_root.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
