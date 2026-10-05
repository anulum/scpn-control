# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RMSE dashboard command.
"""Compute report lanes and publish selected local JSON/Markdown/PNG outputs.

The registered validation command and historical script facade share this
owner. Destination paths resolve against cwd; source references follow the
script repository root. Output success does not imply CI regression admission,
physical validity or externally published evidence. File writes are sequential.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from validation.rmse_dashboard_metrics import (
    beta_rmse_iter_sparc,
    confinement_rmse_iter_sparc,
    confinement_rmse_itpa,
    forward_diagnostics_rmse,
    sparc_axis_rmse,
)
from validation.rmse_dashboard_rendering import render_markdown, render_plots

ROOT = Path(__file__).resolve().parents[1]
try:
    from validation.psi_pointwise_rmse import sparc_psi_rmse as _sparc_psi_rmse

    _HAS_PSI_RMSE = True
except ImportError:
    _HAS_PSI_RMSE = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse report destinations; relative paths resolve against the working directory.

    ``--plot-dir`` defaults to the JSON report's parent directory. Unknown
    arguments exit with status two; parsing does not write files.

    Examples
    --------
    >>> args = parse_args(["--output-json", "report.json", "--plot-dir", "figures"])
    >>> args.output_json, str(args.plot_dir)
    ('report.json', 'figures')
    """
    parser = argparse.ArgumentParser(description="Generate SCPN RMSE validation dashboard.")
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "rmse_dashboard.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "rmse_dashboard.md"),
        help="Path to write Markdown report.",
    )
    parser.add_argument("--plot-dir", type=Path, help="Plot directory; defaults to the JSON report parent.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compute bounded reference lanes and write JSON, Markdown and optional plots.

    Parameters
    ----------
    argv
        Arguments excluding the executable; None reads process arguments.
        Report destinations default to validation/reports in the owning repo.
        Plot destinations default to the selected JSON parent.

    Returns
    -------
    int
        Zero after output generation, including explicit unavailable lanes.
        This status does not imply that the regression guard passes. Invalid
        inputs or write failures propagate; argparse exits two on invalid flags.
    """
    args = parse_args(argv)
    t0 = time.perf_counter()

    validation_dir = ROOT / "validation"
    reference_dir = validation_dir / "reference_data"
    itpa_csv = reference_dir / "itpa" / "hmode_confinement.csv"
    sparc_eq_dir = reference_dir / "sparc"

    report: dict[str, Any] = {
        "schema": "scpn-control.rmse-dashboard.v1",
        "claim_boundary": "Bounded regression comparisons; no held-out physics or facility validation admitted.",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "confinement_itpa": confinement_rmse_itpa(itpa_csv),
        "confinement_iter_sparc": confinement_rmse_iter_sparc(reference_dir),
        "beta_iter_sparc": beta_rmse_iter_sparc(reference_dir, validation_dir),
        "sparc_axis": sparc_axis_rmse(sparc_eq_dir),
        "sparc_psi_rmse": _sparc_psi_rmse(sparc_eq_dir) if _HAS_PSI_RMSE else {"skipped": True},
        "forward_diagnostics": forward_diagnostics_rmse(),
    }
    report["runtime_seconds"] = time.perf_counter() - t0

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    plot_dir = args.plot_dir if args.plot_dir is not None else json_path.parent
    saved_plots = render_plots(report, plot_dir)
    # Use a robust relative path from the markdown directory to the plots directory.
    plot_rel = (
        Path(
            os.path.relpath(
                plot_dir.resolve(),
                start=md_path.parent.resolve(),
            )
        ).as_posix()
        if saved_plots
        else None
    )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(report, plot_dir=plot_rel))

    print("SCPN RMSE dashboard generated.")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    if saved_plots:
        print(f"Plots: {len(saved_plots)} figures saved to {plot_dir}")
    beta_summary = (
        "unavailable" if report["beta_iter_sparc"].get("skipped") else f"{report['beta_iter_sparc']['beta_n_rmse']:.4f}"
    )
    print(
        "Summary -> "
        f"ITPA tau_E RMSE={report['confinement_itpa']['tau_rmse_s']:.4f}s, "
        f"beta_N RMSE={beta_summary}, "
        f"SPARC axis RMSE={report['sparc_axis']['axis_rmse_m']:.6f}m"
    )
    return 0
