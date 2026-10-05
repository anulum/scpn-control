# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RMSE dashboard public facade.
"""Expose the bounded reference-comparison API and historical script entrypoint.

Metrics, presentation and local publication have separate owning modules.
Design/reference comparisons do not admit held-out facility accuracy.
Unavailable beta remains null/skipped pending a version-bound approved model.
CLI success records output generation; the separate CI guard admits regressions.

Examples
--------
Use the public paired-error definition without importing a burn model:

>>> rmse([1.0, 3.0], [2.0, 4.0])
1.0
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.rmse_dashboard_command import ROOT, main, parse_args
from validation.rmse_dashboard_metrics import (
    beta_rmse_iter_sparc,
    compare_eq_axis,
    confinement_rmse_iter_sparc,
    confinement_rmse_itpa,
    estimate_beta_n_from_burn,
    forward_diagnostics_rmse,
    ipb98_tau_e,
    load_json,
    rmse,
    sparc_axis_rmse,
)
from validation.rmse_dashboard_rendering import THRESHOLDS, render_markdown, render_plots

__all__ = [
    "ipb98_tau_e",
    "rmse",
    "load_json",
    "compare_eq_axis",
    "confinement_rmse_itpa",
    "confinement_rmse_iter_sparc",
    "estimate_beta_n_from_burn",
    "beta_rmse_iter_sparc",
    "sparc_axis_rmse",
    "forward_diagnostics_rmse",
    "render_markdown",
    "render_plots",
    "parse_args",
    "main",
    "ROOT",
    "THRESHOLDS",
]

if __name__ == "__main__":
    raise SystemExit(main())
