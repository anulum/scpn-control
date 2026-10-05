# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign evidence admission.
"""Inspect persisted Python/PyO3 and Rust campaign report declarations.

This standard-library reader admits two report schemas, their canonical
self-digests and declared surface/count/context fields. It does not execute
a campaign, reopen per-shot decision digests, authenticate a producer or
qualify hardware, a controller or production timing. A resealed report is
still a declaration. The Python and native Rust context serializations differ.

Use the standalone script with --python-report, --rust-report and optional
--json-out, or the registered scpn-control validate consumer. Paths follow
caller cwd and symlinks. No containment or input size/depth budget is imposed.
Extra fields are unchecked except duplicate/nonfinite floating JSON refusal
and inclusion in the canonical self-digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[1]
MULTI_SHOT_CAMPAIGN_EVIDENCE_SCHEMA_VERSION = "scpn-control.multi-shot-campaign-evidence-admission.v1"
PYTHON_BENCHMARK_SCHEMA_VERSION = "scpn-control.multi-shot-campaign-benchmark.v1.1"
RUST_BENCHMARK_SCHEMA_VERSION = "scpn-control.rust-multi-shot-campaign-benchmark.v1.1"
DEFAULT_PYTHON_REPORT = (
    ROOT / "validation" / "reports" / "multi_shot_campaign_pulsed_mpc_evidence_python_pyo3_20260604T172543Z.json"
)
DEFAULT_RUST_REPORT = (
    ROOT / "validation" / "reports" / "multi_shot_campaign_pulsed_mpc_evidence_rust_20260604T173322Z.json"
)
BENCHMARK_EVIDENCE_CLASSES = frozenset({"local_regression", "production_benchmark"})


@dataclass(frozen=True)
class MultiShotCampaignEvidenceAdmission:
    """Frozen reader findings and declared report metadata.

    Attributes
    ----------
    status : str
        pass when all reader checks succeed; otherwise fail.
    errors : tuple[str, ...]
        Ordered argument, Python and Rust read/schema/context/count findings.
    python_report_sha256, rust_report_sha256 : str or None
        SHA-256 of exact successfully decoded JSON-object bytes, including an
        empty object on FAIL. Read/decode/root-type errors yield None.
    python_payload_sha256, rust_payload_sha256 : str or None
        Declared string self-digests, including malformed strings on FAIL.
        Non-string values yield None; these fields do not authenticate reports.
    admitted_surfaces : tuple[str, ...]
        Exactly python, pyo3, rust when the reader passes, otherwise empty.
    pyo3_status : str or None
        Declared string, including unknown strings on FAIL; other types None.
    production_claim_allowed : bool
        True if either inspected report declares the literal boolean true.
        This preserves declarations even on FAIL and grants no qualification.
    minimum_digest_count : int
        Positive requested count, or fallback one after an invalid argument
        finding. Counts are declarations; per-shot digest bytes are not read.
    """

    status: str
    errors: tuple[str, ...]
    python_report_sha256: str | None
    rust_report_sha256: str | None
    python_payload_sha256: str | None
    rust_payload_sha256: str | None
    admitted_surfaces: tuple[str, ...]
    pyo3_status: str | None
    production_claim_allowed: bool
    minimum_digest_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return all result fields plus schema, with fresh mutable errors/surface lists.

        The frozen result is unchanged by edits to the returned mapping.
        Top-level release admission consumes declarations, not reopened reports.
        """
        return {
            "schema_version": MULTI_SHOT_CAMPAIGN_EVIDENCE_SCHEMA_VERSION,
            "status": self.status,
            "errors": list(self.errors),
            "python_report_sha256": self.python_report_sha256,
            "rust_report_sha256": self.rust_report_sha256,
            "python_payload_sha256": self.python_payload_sha256,
            "rust_payload_sha256": self.rust_payload_sha256,
            "admitted_surfaces": list(self.admitted_surfaces),
            "pyo3_status": self.pyo3_status,
            "production_claim_allowed": self.production_claim_allowed,
            "minimum_digest_count": self.minimum_digest_count,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated keys at every JSON depth, including extra metadata."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_float(token: str) -> float:
    """Reject nonfinite constants and overflowing exponents at any JSON depth."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {token}")
    return value


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    """Read once, decode a UTF-8 object and hash those same exact bytes."""
    blob = path.read_bytes()
    payload = json.loads(
        blob.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_float=_parse_json_float,
        parse_constant=_parse_json_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return payload, hashlib.sha256(blob).hexdigest()


def _sha256_hex(value: object) -> bool:
    """Recognise lowercase SHA-256 spelling without coercing other types."""
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _positive_int(value: object) -> TypeGuard[int]:
    """Recognise positive integers, excluding booleans and floating values."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_empty_list(value: object) -> bool:
    """Recognise a nonempty list; its context elements are not interpreted."""
    return isinstance(value, list) and bool(value)


def _validate_payload_hash(payload: dict[str, Any], errors: list[str], prefix: str) -> None:
    """Check self-consistency with digest blank and all extra fields included."""
    supplied = payload.get("payload_sha256")
    if not _sha256_hex(supplied):
        errors.append(f"{prefix}.payload_sha256 must be a SHA-256 hex digest")
        return
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied != digest:
        errors.append(f"{prefix}.payload_sha256 does not match canonical payload")


def _validate_result_section(
    section: object,
    *,
    errors: list[str],
    prefix: str,
    minimum_digest_count: int,
) -> None:
    """Require declared positive passed/sample counts and the minimum digest count.

    No sum/steps/count reconciliation, latency check or per-shot digest replay
    occurs. The three actual Python, PyO3 and Rust summaries share this shape.
    """
    if not isinstance(section, dict):
        errors.append(f"{prefix} must be an object")
        return
    if not _positive_int(section.get("last_passed_count")):
        errors.append(f"{prefix}.last_passed_count must be a positive integer")
    digest_count = section.get("last_pulsed_mpc_admission_digest_count")
    if not (_positive_int(digest_count) and int(digest_count) >= minimum_digest_count):
        errors.append(f"{prefix}.last_pulsed_mpc_admission_digest_count must be at least {minimum_digest_count}")
    stats = section.get("stats")
    if not isinstance(stats, dict):
        errors.append(f"{prefix}.stats must be an object")
    elif not _positive_int(stats.get("samples")):
        errors.append(f"{prefix}.stats.samples must be a positive integer")


def _validate_common_benchmark(
    payload: dict[str, Any],
    *,
    expected_schema: str,
    errors: list[str],
    prefix: str,
) -> bool:
    """Check schema/class/flag/command declarations without production qualification.

    local_regression cannot declare a true production flag. A production-class
    report may retain false; either literal true is reflected even on FAIL.
    """
    if payload.get("schema_version") != expected_schema:
        errors.append(f"{prefix}.schema_version must be {expected_schema!r}")
    evidence_class = payload.get("evidence_class")
    if not isinstance(evidence_class, str) or evidence_class not in BENCHMARK_EVIDENCE_CLASSES:
        errors.append(f"{prefix}.evidence_class must be recognised")
    production_claim_allowed = payload.get("production_claim_allowed")
    if not isinstance(production_claim_allowed, bool):
        errors.append(f"{prefix}.production_claim_allowed must be a boolean")
    elif evidence_class == "local_regression" and production_claim_allowed:
        errors.append(f"{prefix}.local_regression must not allow production benchmark claims")
    if not isinstance(payload.get("command"), str) or "bench_multi_shot_campaign" not in str(payload.get("command")):
        errors.append(f"{prefix}.command must identify the multi-shot benchmark")
    if not _sha256_hex(payload.get("payload_sha256")):
        errors.append(f"{prefix}.payload_sha256 must be a SHA-256 hex digest")
    return production_claim_allowed is True


def _validate_python_report(
    payload: dict[str, Any],
    *,
    errors: list[str],
    minimum_digest_count: int,
) -> bool:
    """Check Python and PyO3 summaries with Python-shaped context declarations.

    Affinity is a nonempty list; its elements and recorded non-None load fields
    are not validated. No current host or per-shot campaign is inspected.
    """
    production = _validate_common_benchmark(
        payload,
        expected_schema=PYTHON_BENCHMARK_SCHEMA_VERSION,
        errors=errors,
        prefix="multi_shot_campaign.python",
    )
    _validate_payload_hash(payload, errors, "multi_shot_campaign.python")
    context = payload.get("context")
    if not isinstance(context, dict):
        errors.append("multi_shot_campaign.python.context must be an object")
    else:
        if not _non_empty_list(context.get("cpu_affinity")):
            errors.append("multi_shot_campaign.python.context.cpu_affinity must be a non-empty list")
        if context.get("loadavg_start") is None or context.get("loadavg_end") is None:
            errors.append("multi_shot_campaign.python.context must record loadavg_start and loadavg_end")
    _validate_result_section(
        payload.get("result"),
        errors=errors,
        prefix="multi_shot_campaign.python.result",
        minimum_digest_count=minimum_digest_count,
    )
    if payload.get("pyo3_status") != "ok":
        errors.append("multi_shot_campaign.pyo3_status must be 'ok'")
    _validate_result_section(
        payload.get("pyo3_result"),
        errors=errors,
        prefix="multi_shot_campaign.pyo3_result",
        minimum_digest_count=minimum_digest_count,
    )
    return production


def _validate_rust_report(
    payload: dict[str, Any],
    *,
    errors: list[str],
    minimum_digest_count: int,
) -> bool:
    """Check the Rust summary while retaining native textual context serialization.

    Affinity is nonblank text; recorded non-None loads are not parsed. Linux
    list syntax, live host availability and isolation qualification are not
    proved. This is a report reader, not the native campaign kernel.
    """
    production = _validate_common_benchmark(
        payload,
        expected_schema=RUST_BENCHMARK_SCHEMA_VERSION,
        errors=errors,
        prefix="multi_shot_campaign.rust",
    )
    _validate_payload_hash(payload, errors, "multi_shot_campaign.rust")
    context = payload.get("context")
    if not isinstance(context, dict):
        errors.append("multi_shot_campaign.rust.context must be an object")
    else:
        if not isinstance(context.get("cpu_affinity"), str) or not str(context.get("cpu_affinity")).strip():
            errors.append("multi_shot_campaign.rust.context.cpu_affinity must be recorded")
        if context.get("loadavg_start") is None or context.get("loadavg_end") is None:
            errors.append("multi_shot_campaign.rust.context must record loadavg_start and loadavg_end")
    _validate_result_section(
        payload.get("result"),
        errors=errors,
        prefix="multi_shot_campaign.rust.result",
        minimum_digest_count=minimum_digest_count,
    )
    return production


def validate_multi_shot_campaign_evidence(
    python_report: str | Path = DEFAULT_PYTHON_REPORT,
    rust_report: str | Path = DEFAULT_RUST_REPORT,
    *,
    minimum_digest_count: int = 2,
) -> MultiShotCampaignEvidenceAdmission:
    """Inspect two persisted campaign reports through the public metadata API.

    Parameters
    ----------
    python_report, rust_report : str or Path
        Caller-relative paths, or repository historical defaults. Symlinks are
        followed; files are read, never written. Python includes its PyO3 result.
    minimum_digest_count : int
        Positive non-boolean integer required in each of the three result
        declarations. An invalid argument records FAIL and uses fallback one.

    Returns
    -------
    MultiShotCampaignEvidenceAdmission
        Structured findings for supported read, UTF-8, duplicate/nonfinite,
        JSON/root-type, schema/type/count/context and self-digest refusals.
        Read/decode failures have no exact-byte report digest. Empty objects
        are hashed but fail required declarations. No physical campaign or
        benchmark is run, per-shot digests are not reopened, and latency or
        context contents beyond presence/shape are not qualified.

    Examples
    --------
    Empty persisted objects cannot establish any campaign surface:

    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     report = Path(directory) / 'empty.json'
    ...     _ = report.write_text('{}', encoding='utf-8')
    ...     result = validate_multi_shot_campaign_evidence(report, report)
    ...     (result.status, result.admitted_surfaces)
    ('fail', ())
    """
    errors: list[str] = []
    if not _positive_int(minimum_digest_count):
        errors.append("minimum_digest_count must be a positive integer")
        minimum_digest_count = 1

    python_payload: dict[str, Any] = {}
    rust_payload: dict[str, Any] = {}
    python_report_sha256: str | None = None
    rust_report_sha256: str | None = None
    try:
        python_payload, python_report_sha256 = _load_json(Path(python_report))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        errors.append(f"multi_shot_campaign.python_report: {exc}")
    try:
        rust_payload, rust_report_sha256 = _load_json(Path(rust_report))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        errors.append(f"multi_shot_campaign.rust_report: {exc}")

    production_claim_allowed = False
    if python_report_sha256 is not None:
        production_claim_allowed = (
            _validate_python_report(
                python_payload,
                errors=errors,
                minimum_digest_count=minimum_digest_count,
            )
            or production_claim_allowed
        )
    if rust_report_sha256 is not None:
        production_claim_allowed = (
            _validate_rust_report(
                rust_payload,
                errors=errors,
                minimum_digest_count=minimum_digest_count,
            )
            or production_claim_allowed
        )

    admitted_surfaces = ("python", "pyo3", "rust") if not errors else ()
    return MultiShotCampaignEvidenceAdmission(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        python_report_sha256=python_report_sha256,
        rust_report_sha256=rust_report_sha256,
        python_payload_sha256=python_payload.get("payload_sha256")
        if isinstance(python_payload.get("payload_sha256"), str)
        else None,
        rust_payload_sha256=rust_payload.get("payload_sha256")
        if isinstance(rust_payload.get("payload_sha256"), str)
        else None,
        admitted_surfaces=admitted_surfaces,
        pyo3_status=python_payload.get("pyo3_status") if isinstance(python_payload.get("pyo3_status"), str) else None,
        production_claim_allowed=production_claim_allowed,
        minimum_digest_count=minimum_digest_count,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the standard-library reader without generating or altering evidence.

    argv excludes the executable; None reads process arguments. Defaults are
    absolute repository report paths. --json-out prints the complete result;
    text mode prints status and ordered ERROR findings. Return zero on reader
    PASS, one on refusal. Argparse errors exit two. No current-host test or
    production-control permission is granted.
    """
    parser = argparse.ArgumentParser(description="Validate multi-shot campaign pulsed-MPC evidence")
    parser.add_argument("--python-report", default=str(DEFAULT_PYTHON_REPORT), help="Python/PyO3 benchmark JSON report")
    parser.add_argument("--rust-report", default=str(DEFAULT_RUST_REPORT), help="Rust benchmark JSON report")
    parser.add_argument("--minimum-digest-count", type=int, default=2)
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args(argv)

    result = validate_multi_shot_campaign_evidence(
        args.python_report,
        args.rust_report,
        minimum_digest_count=args.minimum_digest_count,
    )
    payload = result.as_dict()
    if args.json_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Multi-shot campaign evidence: {result.status}")
        for error in result.errors:
            print(f"ERROR {error}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
