# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RMSE report presentation.
"""Render supplied dashboard mappings as Markdown and optional PNG files.

This owner presents metrics without importing the compute owner or admitting
scientific evidence. Heuristic flag thresholds differ from the CI regression
guard. Missing optional Matplotlib yields no figures. Caller-provided report
values are formatted directly; malformed fields and write errors propagate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

THRESHOLDS: dict[str, dict[str, Any]] = {
    "tau_rmse_pct": {
        "warn": 25.0,
        "fail": 30.0,
        "direction": "lower_better",
        "label": "tau_E MAE %",
    },
    "beta_n_rmse": {
        "warn": 0.08,
        "fail": 0.10,
        "direction": "lower_better",
        "label": "beta_N RMSE",
    },
    "sparc_axis_rmse_m": {
        "warn": 0.08,
        "fail": 0.10,
        "direction": "lower_better",
        "label": "SPARC axis RMSE (m)",
    },
    "fpr": {
        "warn": 0.10,
        "fail": 0.15,
        "direction": "lower_better",
        "label": "Disruption FPR",
    },
    "tbr": {
        "lo": 1.00,
        "hi": 1.40,
        "direction": "range",
        "label": "TBR (corrected)",
    },
    "q_max": {
        "warn": 12.0,
        "fail": 15.0,
        "direction": "lower_better",
        "label": "Q peak",
    },
}


def _flag(value: float, key: str) -> str:
    """Classify one supplied value using the shared display-threshold mapping.

    Unknown keys yield an empty flag. A range key passes inclusive lo/hi;
    other keys pass through inclusive warn, warn through inclusive fail and
    otherwise fail. NaN comparisons fail both forms; no finiteness/type repair
    occurs. THRESHOLDS is a mutable module mapping, read on every call. These
    labels describe display policy and differ from CI regression admission.
    """
    spec = THRESHOLDS.get(key)
    if spec is None:
        return ""
    if spec["direction"] == "range":
        if spec["lo"] <= value <= spec["hi"]:
            return "PASS"
        return "FAIL"
    # lower_better
    if value <= spec["warn"]:
        return "PASS"
    if value <= spec["fail"]:
        return "WARN"
    return "FAIL"


_FLAG_EMOJI = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "": ""}


def render_markdown(report: dict[str, Any], plot_dir: str | None = None) -> str:
    """Render report metrics and explicit unavailable beta status as Markdown.

    Parameters
    ----------
    report
        Complete dashboard mapping. Confinement error uses seconds/percent,
        axis error metres and beta_N error is dimensionless. Optional forward
        diagnostics retain channel-specific units. Missing keys raise KeyError.
    plot_dir
        Relative or absolute link prefix for optional PNG figures; None emits
        no figure links. This function does not write or validate image files.

    Returns
    -------
    str
        Markdown text with dashboard heuristic flags, separate from the CI
        regression limits. Unavailable beta has no zero metric or PASS flag.
        Neither rendered flags nor synthetic forward comparisons grant physical
        validity; input text is rendered without Markdown escaping.
    """
    lines: list[str] = []
    lines.append("# SCPN RMSE Dashboard")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at_utc']}`")
    lines.append(f"- Runtime: `{report['runtime_seconds']:.2f}s`")
    lines.append("")

    # ── Auto-flag summary ─────────────────────────────────────────────
    itpa = report["confinement_itpa"]
    beta = report["beta_iter_sparc"]
    sparc_axis = report["sparc_axis"]

    flag_rows: list[tuple[str, float, str, str]] = [
        ("tau_E MAE %", itpa["tau_mae_rel_pct"], "tau_rmse_pct", f"{itpa['tau_mae_rel_pct']:.2f}%"),
        ("SPARC axis RMSE (m)", sparc_axis["axis_rmse_m"], "sparc_axis_rmse_m", f"{sparc_axis['axis_rmse_m']:.6f}"),
    ]

    if not beta.get("skipped", False):
        flag_rows.insert(1, ("beta_N RMSE", beta["beta_n_rmse"], "beta_n_rmse", f"{beta['beta_n_rmse']:.4f}"))
    lines.append("## Auto-Flag Summary")
    lines.append("")
    lines.append("| Metric | Value | Flag |")
    lines.append("|--------|-------|------|")
    for label, value, key, fmt_val in flag_rows:
        f = _flag(value, key)
        lines.append(f"| {label} | `{fmt_val}` | **{_FLAG_EMOJI[f]}** |")
    lines.append("")

    # ── Confinement ITPA ──────────────────────────────────────────────
    tau_flag = _flag(itpa["tau_mae_rel_pct"], "tau_rmse_pct")
    lines.append("## Confinement RMSE (ITPA H-mode)")
    lines.append("")
    lines.append(f"- Samples: `{itpa['count']}`")
    lines.append(f"- `tau_E` RMSE: `{itpa['tau_rmse_s']:.4f} s`")
    lines.append(
        f"- `tau_E` mean absolute relative error: `{itpa['tau_mae_rel_pct']:.2f}%` **{_FLAG_EMOJI[tau_flag]}**"
    )
    lines.append(f"- `H98(y,2)` RMSE: `{itpa['h98_rmse']:.4f}`")
    lines.append("")

    if plot_dir:
        lines.append(f"![tau_E scatter]({plot_dir}/tau_e_scatter.png)")
        lines.append("")

    # ── Confinement ITER+SPARC ────────────────────────────────────────
    lines.append("## Confinement RMSE (ITER + SPARC references)")
    lines.append("")
    itsp = report["confinement_iter_sparc"]
    lines.append(f"- Samples: `{itsp['count']}`")
    lines.append(f"- `tau_E` RMSE: `{itsp['tau_rmse_s']:.4f} s`")
    lines.append("")

    # ── Beta_N ────────────────────────────────────────────────────────
    beta_flag = "" if beta.get("skipped", False) else _flag(beta["beta_n_rmse"], "beta_n_rmse")
    lines.append("## Beta_N RMSE (ITER + SPARC references)")
    lines.append("")
    lines.append(f"- Samples: `{beta['count']}`")
    if beta.get("skipped", False):
        lines.append(f"- Unavailable: {beta['reason']}")
    else:
        lines.append(f"- `beta_N` RMSE: `{beta['beta_n_rmse']:.4f}` **{_FLAG_EMOJI[beta_flag]}**")
    lines.append("")

    if plot_dir and beta.get("rows"):
        lines.append(f"![beta_N scatter]({plot_dir}/beta_n_scatter.png)")
        lines.append("")

    # ── SPARC axis ────────────────────────────────────────────────────
    axis_flag = _flag(sparc_axis["axis_rmse_m"], "sparc_axis_rmse_m")
    lines.append("## SPARC GEQDSK Axis Error")
    lines.append("")
    lines.append(f"- Files: `{sparc_axis['count']}`")
    lines.append(f"- Axis RMSE: `{sparc_axis['axis_rmse_m']:.6f} m` **{_FLAG_EMOJI[axis_flag]}**")
    lines.append("")

    # ── Psi RMSE ──────────────────────────────────────────────────────
    psi_rmse = report.get("sparc_psi_rmse")
    if isinstance(psi_rmse, dict) and "mean_psi_rmse_norm" in psi_rmse:
        lines.append("## SPARC Point-wise psi(R,Z) RMSE")
        lines.append("")
        lines.append(f"- Files: `{psi_rmse['count']}`")
        lines.append(f"- Mean normalised psi RMSE: `{psi_rmse['mean_psi_rmse_norm']:.6f}`")
        lines.append(f"- Mean relative L2: `{psi_rmse['mean_psi_relative_l2']:.6f}`")
        lines.append(f"- Mean GS residual (rel L2): `{psi_rmse['mean_gs_residual_l2']:.4f}`")
        lines.append(f"- Worst file: `{psi_rmse['worst_file']}` (psi_N RMSE = `{psi_rmse['worst_psi_rmse_norm']:.6f}`)")
        lines.append("")

    # ── Forward diagnostics ───────────────────────────────────────────
    fwd = report.get("forward_diagnostics")
    if isinstance(fwd, dict):
        lines.append("## Forward Diagnostics RMSE")
        lines.append("")
        lines.append(f"- Interferometer channels: `{fwd['count_interferometer_channels']}`")
        lines.append(f"- Interferometer phase RMSE: `{fwd['phase_rmse_rad']:.6e} rad`")
        lines.append(f"- Neutron-count relative error: `{fwd['neutron_rate_rel_error_pct']:.3f}%`")
        if "thomson_voltage_rmse_v" in fwd:
            lines.append(f"- Thomson channels: `{fwd.get('count_thomson_channels', 0)}`")
            lines.append(f"- Thomson voltage RMSE: `{fwd['thomson_voltage_rmse_v']:.6e} V`")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    if beta.get("skipped", False):
        lines.append("- No beta_N model prediction was available; the lane cannot pass its regression guard.")
    else:
        lines.append("- Supplied beta_N comparison rows do not establish a model run or physical validity.")
        lines.append(
            "- The historical dynamic-model path uses target-calibrated correction factor 1.446; modern FUSION conversion requires separate version-bound approval."
        )
    lines.append("- Use this report for trend tracking; not as a replacement for full transport/MHD validation.")
    lines.append("")

    return "\n".join(lines)


def render_plots(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """Generate matplotlib validation plots and return list of saved paths.

    Uses the supplied confinement/axis/beta rows to write up to three PNGs
    into output_dir, creating it as needed. Existing same-named figures are
    overwritten. Missing Matplotlib returns an empty list; unavailable lanes
    without rows produce no figure. These plots render the supplied comparison
    records and do not attest empirical validity. Input and write errors
    propagate; each created Matplotlib figure is closed even when preparing or
    writing its plot fails.
    """
    if not _HAS_MPL:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # ── 1. tau_E predicted vs measured scatter (ITPA) ─────────────────
    itpa = report["confinement_itpa"]
    if itpa.get("rows"):
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            tau_m = [r["tau_measured_s"] for r in itpa["rows"]]
            tau_p = [r["tau_pred_s"] for r in itpa["rows"]]
            ax.scatter(tau_m, tau_p, s=40, edgecolors="black", linewidths=0.5, zorder=3)
            lo = min(min(tau_m), min(tau_p)) * 0.8
            hi = max(max(tau_m), max(tau_p)) * 1.2
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
            ax.set_xlabel("tau_E measured (s)")
            ax.set_ylabel("tau_E predicted (s)")
            ax.set_title(f"ITPA H-mode: tau_E (N={itpa['count']}, RMSE={itpa['tau_rmse_s']:.4f} s)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = output_dir / "tau_e_scatter.png"
            fig.savefig(p, dpi=150)
            saved.append(p)
        finally:
            plt.close(fig)

    # ── 2. beta_N predicted vs measured scatter (ITER+SPARC) ──────────
    beta = report["beta_iter_sparc"]
    if beta.get("rows"):
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            bm = [r["beta_n_measured"] for r in beta["rows"]]
            bp = [r["beta_n_estimated"] for r in beta["rows"]]
            labels = [r["scenario"] for r in beta["rows"]]
            ax.scatter(bm, bp, s=80, edgecolors="black", linewidths=0.5, zorder=3)
            for x, y, lbl in zip(bm, bp, labels):
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
            lo = min(min(bm), min(bp)) * 0.7
            hi = max(max(bm), max(bp)) * 1.3
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
            ax.set_xlabel("beta_N reference")
            ax.set_ylabel("beta_N estimated")
            ax.set_title(f"beta_N (RMSE={beta['beta_n_rmse']:.4f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = output_dir / "beta_n_scatter.png"
            fig.savefig(p, dpi=150)
            saved.append(p)
        finally:
            plt.close(fig)

    # ── 3. SPARC axis error bar chart ─────────────────────────────────
    sparc_axis = report["sparc_axis"]
    if sparc_axis.get("rows"):
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            names = [r["file"] for r in sparc_axis["rows"]]
            errs = [r["axis_error_m"] for r in sparc_axis["rows"]]
            x = range(len(names))
            ax.bar(x, errs, color="steelblue", edgecolor="black", linewidth=0.5)
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Axis error (m)")
            ax.set_title(f"SPARC GEQDSK Axis Error (RMSE={sparc_axis['axis_rmse_m']:.6f} m)")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            p = output_dir / "sparc_axis_error.png"
            fig.savefig(p, dpi=150)
            saved.append(p)
        finally:
            plt.close(fig)

    return saved
