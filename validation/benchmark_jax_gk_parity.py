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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scpn_control.core.jax_gk_solver import JAX_GK_PARITY_CASES, has_jax, write_jax_gk_parity_artifact
from validation.validate_jax_gk_parity import validate_jax_gk_parity


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
    for case in cases:
        _, artifact_path = write_jax_gk_parity_artifact(
            artifact_root,
            case=case,
            solver_kwargs={"n_ky_ion": args.n_ky_ion, "n_theta": args.n_theta},
            gamma_relative_tolerance=args.gamma_relative_tolerance,
            omega_absolute_tolerance=args.omega_absolute_tolerance,
        )
        artifact_paths.append(artifact_path)
    report = validate_jax_gk_parity(artifact_root, require_parity_artifacts=True, require_cases=cases)
    markdown_path = artifact_root / "jax_gk_parity.md"
    markdown_path.write_text(build_report_markdown(report), encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"JAX GK parity: {report['status']} parity_artifacts={report['parity_artifacts']}")
        for artifact_path in artifact_paths:
            print(f"artifact={artifact_path}")
        print(f"summary={markdown_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
