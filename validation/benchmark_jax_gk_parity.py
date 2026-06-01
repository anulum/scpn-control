#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity artifact benchmark

"""Generate persisted JAX/native gyrokinetic backend parity evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scpn_control.core.jax_gk_solver import JAX_GK_PARITY_CASES, has_jax, write_jax_gk_parity_artifact
from validation.validate_jax_gk_parity import validate_jax_gk_parity


def _display_path(path: Path) -> str:
    """Render repository paths relative to the checkout for stable reports."""

    try:
        return str(path.resolve(strict=False).relative_to(ROOT))
    except ValueError:
        return str(path)


def build_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for parity-evidence review."""
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — JAX GK Parity Evidence -->",
        "",
        "# JAX GK Parity Evidence",
        "",
        f"- Status: `{report['status']}`",
        f"- Artifacts: `{report['parity_artifacts']}`",
        f"- Max gamma relative error: `{report.get('max_gamma_relative_error')}`",
        f"- Max omega absolute error: `{report.get('max_omega_absolute_error')}`",
        f"- Entries payload SHA-256: `{report.get('entries_payload_sha256')}`",
        "- Claim boundary: backend parity only; external GK validation remains required.",
        "",
        "| Case | Backend | Dtype | X64 | Native dominant | JAX dominant | Gamma relative error | Omega absolute error |",
        "|---|---|---|---:|---|---|---:|---:|",
    ]
    for entry in report["entries"]:
        lines.append(
            "| {case} | {backend} | {dtype} | {x64_enabled} | {native_dominant_mode_type} | {jax_dominant_mode_type} | {gamma_relative_error:.6g} | {omega_absolute_error:.6g} |".format(
                **entry
            )
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in report["errors"]:
            lines.append(f"- `{error['path']}` `{error.get('field', 'unknown')}`: {error['error']}")
    return "\n".join(lines) + "\n"


def _sha256_json(payload: dict[str, Any]) -> str:
    import hashlib

    digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_benchmark_report(
    *,
    artifact_root: Path,
    generated_artifacts: list[dict[str, Any]],
    validation_report: dict[str, Any],
    total_elapsed_s: float,
    cases: tuple[str, ...],
) -> dict[str, Any]:
    """Build a benchmark report outside the parity-artifact directory."""

    report: dict[str, Any] = {
        "schema_version": "scpn-control.jax-gk-parity-benchmark.v1",
        "status": validation_report["status"],
        "claim_boundary": "local benchmark timing and backend parity evidence only; external GK validation remains required",
        "artifact_root": _display_path(artifact_root),
        "cases_requested": list(cases),
        "generated_artifacts": generated_artifacts,
        "generated_artifact_count": len(generated_artifacts),
        "total_elapsed_s": float(total_elapsed_s),
        "validation_report_payload_sha256": validation_report["report_payload_sha256"],
        "entries_payload_sha256": validation_report["entries_payload_sha256"],
        "max_gamma_relative_error": validation_report["max_gamma_relative_error"],
        "max_omega_absolute_error": validation_report["max_omega_absolute_error"],
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def write_benchmark_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Persist benchmark JSON and Markdown reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — JAX GK Parity Benchmark Report -->",
        "",
        "# JAX GK Parity Benchmark Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Generated artifacts: `{report['generated_artifact_count']}`",
        f"- Total elapsed seconds: `{report['total_elapsed_s']:.6f}`",
        f"- Entries payload SHA-256: `{report['entries_payload_sha256']}`",
        f"- Report payload SHA-256: `{report['payload_sha256']}`",
        "- Claim boundary: local benchmark timing and backend parity evidence only; external GK validation remains required.",
        "",
        "| Case | Backend | Device | Path | Elapsed s |",
        "|---|---|---|---|---:|",
    ]
    for artifact in report["generated_artifacts"]:
        lines.append(
            f"| `{artifact['case']}` | `{artifact['backend']}` | `{artifact['device_kind']}` | "
            f"`{artifact['path']}` | {artifact['elapsed_s']:.6f} |"
        )
    lines.append("")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "jax_gk_parity"),
        help="Directory for generated JAX GK parity artifacts",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=JAX_GK_PARITY_CASES,
        help="Parity case to run; repeat to select multiple cases. Default: all built-in cases.",
    )
    parser.add_argument("--n-ky-ion", type=int, default=4, help="Number of ion-scale ky samples")
    parser.add_argument("--n-theta", type=int, default=16, help="Number of ballooning theta samples")
    parser.add_argument("--gamma-relative-tolerance", type=float, default=0.25)
    parser.add_argument("--omega-absolute-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--benchmark-json-out",
        default=str(ROOT / "validation" / "reports" / "jax_gk_parity_benchmark.json"),
        help="Write aggregate benchmark timing JSON outside the artifact directory",
    )
    parser.add_argument(
        "--benchmark-report-out",
        default=str(ROOT / "validation" / "reports" / "jax_gk_parity_benchmark.md"),
        help="Write aggregate benchmark timing Markdown outside the artifact directory",
    )
    parser.add_argument("--json-out", action="store_true", help="Print validation report JSON")
    args = parser.parse_args(argv)

    artifact_root = Path(args.artifact_root)
    if not has_jax():
        report = {
            "status": "blocked",
            "root": str(artifact_root),
            "parity_artifacts": 0,
            "require_parity_artifacts": True,
            "entries": [],
            "errors": [{"path": str(artifact_root), "field": "jax", "error": "JAX runtime unavailable"}],
        }
        if args.json_out:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("JAX GK parity: blocked parity_artifacts=0", file=sys.stderr)
        return 2

    cases = tuple(args.case) if args.case else JAX_GK_PARITY_CASES
    artifact_paths: list[Path] = []
    generated_artifacts: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for case in cases:
        case_start = time.perf_counter()
        _, artifact_path = write_jax_gk_parity_artifact(
            artifact_root,
            case=case,
            solver_kwargs={"n_ky_ion": args.n_ky_ion, "n_theta": args.n_theta},
            gamma_relative_tolerance=args.gamma_relative_tolerance,
            omega_absolute_tolerance=args.omega_absolute_tolerance,
        )
        elapsed_s = time.perf_counter() - case_start
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        generated_artifacts.append(
            {
                "case": payload["case"],
                "backend": payload["backend"],
                "device_kind": payload["device_kind"],
                "path": _display_path(artifact_path),
                "payload_sha256": payload["payload_sha256"],
                "elapsed_s": elapsed_s,
            }
        )
        artifact_paths.append(artifact_path)
    report = validate_jax_gk_parity(artifact_root, require_parity_artifacts=True, require_cases=cases)
    markdown_path = artifact_root / "jax_gk_parity.md"
    markdown_path.write_text(build_report_markdown(report), encoding="utf-8")
    benchmark_report = build_benchmark_report(
        artifact_root=artifact_root,
        generated_artifacts=generated_artifacts,
        validation_report=report,
        total_elapsed_s=time.perf_counter() - total_start,
        cases=cases,
    )
    write_benchmark_report(benchmark_report, Path(args.benchmark_json_out), Path(args.benchmark_report_out))
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"JAX GK parity: {report['status']} parity_artifacts={report['parity_artifacts']}")
        for artifact_path in artifact_paths:
            print(f"artifact={artifact_path}")
        print(f"summary={markdown_path}")
        print(f"benchmark={args.benchmark_json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
