# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL diagnostic rendering and checked CLI outputs.

"""Run the real SOL algebraic diagnostic with checked report paths and custody."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scpn_control.benchmark_records import require_recorded_campaign
from validation.report_output_paths import checked_report_destination
from validation.sol_two_point_contracts.evidence import build_evidence, validate_evidence_payload
from validation.sol_two_point_contracts.models import validate_sol_two_point

ROOT = Path(__file__).resolve().parents[2]


def render_markdown(evidence: Mapping[str, Any]) -> str:
    """Return the readable table for complete self-sealed v1 observations.

    Parameters
    ----------
    evidence
        Full v1 report accepted by validate_evidence_payload. False outcomes
        render normally. Geometry is metres/tesla; density is 1e19 m^-3 and
        relative errors/ratios are dimensionless.

    Returns
    -------
    str
        Markdown ending in a newline, with the explicit algebraic-only boundary.
        The target string uses JSON escaping so line breaks remain literal.

    Raises
    ------
    ValueError
        Evidence is invalid or inconsistent. No file or model is modified.
    """
    validate_evidence_payload(evidence)
    config = evidence["config"]
    detach = evidence["detachment"]
    lines = [
        "",
        "# Two-Point Scrape-Off-Layer Model Validation",
        "",
        "Algebraic consistency diagnostic; independent scientific/facility/safety/training admission is not supplied.",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: {json.dumps(evidence['target_id'], ensure_ascii=True)}",
        f"- Geometry: R0={config['r0']} m, a={config['a']} m, q95={config['q95']}, B_pol={config['b_pol']} T",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact closed-form references (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| reference | value |",
        "| --- | --- |",
        f"| connection length L_par = pi q95 R0 | {evidence['connection_length_rel_error']:.3e} |",
        f"| parallel-flux mapping | {evidence['max_flux_mapping_rel_error']:.3e} |",
        f"| Spitzer-Härm upstream conduction integral | {evidence['max_conduction_rel_error']:.3e} |",
        f"| pressure balance n_u T_u = 2 n_t T_t | {evidence['max_pressure_balance_rel_error']:.3e} |",
        f"| Eich regression exponents | {evidence['max_scaling_rel_error']:.3e} |",
        f"| peak target heat flux | {evidence['peak_heat_flux_rel_error']:.3e} |",
        "",
        "## Eich regression scaling exponents",
        "",
        "| exponent | measured ratio | expected | rel error |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {check['name']} | {check['measured_ratio']:.6f} | {check['expected_ratio']:.6f} | {check['rel_error']:.3e} |"
        for check in evidence["scaling"]
    ]
    lines += [
        "",
        "## Detachment onset boundary",
        "",
        f"- Analytic critical density: {detach['critical_density_19']:.4f} x 10^19 m^-3",
        f"- Attached below critical (detached={detach['detached_below_critical']}), "
        f"detached above critical (detached={detach['detached_above_critical']})",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run real checks and emit stdout or sequential caller-selected report files.

    Parameters
    ----------
    argv
        Arguments, or None for process arguments. --exact-tol is a positive
        finite dimensionless strict error threshold; --target-id is nonempty
        descriptive text. --json-out selects JSON stdout. --report selects JSON
        and sibling .md paths relative to the process working directory.

    Returns
    -------
    int
        Zero for passing diagnostics, one for consistent failed diagnostics,
        two for supported input/custody/arithmetic/IO refusals. Argparse help
        and usage preserve exits zero and two.

    Notes
    -----
    Both report paths are checked against each other and selected facade,
    calculation, evidence, command and production SOL sources before solving.
    Symlinks/hard links are refused. Persistent repository evidence roots require
    the recorded runner campaign. Parent directories are created only after
    validation. Unrelated outputs are replaced JSON then Markdown, without
    atomic-pair/lock/snapshot guarantees. A later write failure can leave JSON.
    A digest or passing status supplies no independent physical admission.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-id", default="local-sol-two-point")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--exact-tol", type=float, default=1e-9, help="positive finite strict relative-error threshold")
    parser.add_argument("--report", type=Path, default=None, help="write sealed JSON and a sibling Markdown summary")
    args = parser.parse_args(argv)
    try:
        json_path: Path | None = args.report
        if json_path is not None:
            md_path = json_path.with_suffix(".md")
            inputs = [
                ROOT / "validation/validate_sol_two_point.py",
                *sorted(Path(__file__).parent.glob("*.py")),
                ROOT / "src/scpn_control/core/sol_model.py",
            ]
            checked_report_destination(json_path, inputs=[*inputs, md_path])
            checked_report_destination(md_path, inputs=[*inputs, json_path])
            require_recorded_campaign(json_path, md_path, repository_root=ROOT)
        result = validate_sol_two_point(exact_tol=args.exact_tol)
        evidence = build_evidence(result, target_id=args.target_id)
        json_text = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
        markdown = render_markdown(evidence)
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json_text, encoding="utf-8")
            json_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    except (ValueError, RuntimeError, OSError, OverflowError, ZeroDivisionError):
        print("SOL diagnostic input, custody, arithmetic or IO refused.", file=sys.stderr)
        return 2
    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("Two-point scrape-off-layer model validation")
        print(
            f"  conduction + pressure: cond={result.max_conduction_rel_error:.3e} "
            f"press={result.max_pressure_balance_rel_error:.3e} "
            f"{'ok' if result.conduction_passed and result.pressure_passed else 'FAIL'}"
        )
        print(
            f"  flux mapping + L_par:  flux={result.max_flux_mapping_rel_error:.3e} "
            f"L_par={result.connection_length_rel_error:.3e} "
            f"{'ok' if result.flux_mapping_passed and result.connection_passed else 'FAIL'}"
        )
        print(
            f"  Eich + peak flux:      eich={result.max_scaling_rel_error:.3e} "
            f"peak={result.peak_heat_flux_rel_error:.3e} "
            f"{'ok' if result.scaling_passed and result.peak_flux_passed else 'FAIL'}"
        )
        print(
            f"  detachment boundary:   n_crit={result.detachment.critical_density_19:.3f}e19 "
            f"{'ok' if result.detachment_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1
