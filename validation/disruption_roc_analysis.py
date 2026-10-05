# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Roc Analysis.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Predictor ROC Analysis
# © 1996–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""Evaluate synthetic disruption risk using absolute observed event indices.

The fixed cohort and thresholds provide a local diagnostic, not held-out
facility discrimination or authenticated warning-time evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.control.disruption_predictor import (
    predict_disruption_risk,
    simulate_tearing_mode,
)
from scpn_control.control.disruption_roc import roc_auc_from_curve


def generate_scenario_batch(n_total: int = 100) -> list[dict[str, Any]]:
    """Draw a reproducible balanced cohort from the actual synthetic simulator.

    Parameters
    ----------
    n_total : int
        Desired count. Non-positive values return an empty list; otherwise
        floor(n_total/2) shots are disruptive and the remainder safe.

    Returns
    -------
    list[dict[str, Any]]
        signal is a variable-length one-dimensional array, at most 500
        samples. label is 0/1; mode is ntm/density_limit/vde for a disruptive
        shot and safe otherwise. t_disrupt is the zero-based absolute index
        of the observed threshold-crossing final sample, or -1 for safe shots.

    Notes
    -----
    A fresh NumPy seed 42 generator feeds mode selection and public simulation.
    Rejected quota/class samples are redrawn without an attempt cap. Global
    RNG state is unchanged. The simulator's third return is elapsed steps
    since its internal trigger; it is not the absolute event index used here.
    Signals are synthetic mechanism proxies, not held-out facility shots.
    """
    rng = np.random.default_rng(42)
    shots: list[dict[str, Any]] = []
    modes = ["ntm", "density_limit", "vde"]
    n_disrupt_target = n_total // 2
    n_disrupt, n_safe = 0, 0

    while len(shots) < n_total:
        mode = rng.choice(modes)
        signal, label, _ = simulate_tearing_mode(steps=500, mode=mode, rng=rng)
        if label == 1 and n_disrupt < n_disrupt_target:
            shots.append({"signal": signal, "label": 1, "t_disrupt": len(signal) - 1, "mode": mode})
            n_disrupt += 1
        elif label == 0 and n_safe < (n_total - n_disrupt_target):
            shots.append({"signal": signal, "label": 0, "t_disrupt": -1, "mode": "safe"})
            n_safe += 1
    return shots


def evaluate_batch(shots: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Count alarms strictly before declared events over sampled risk windows.

    Parameters
    ----------
    shots : list[dict[str, Any]]
        Cohort mappings with one-dimensional signal, integer label, absolute
        sample index t_disrupt and mode. Generated disruptive traces terminate
        at their observed event. Caller-supplied metadata is not authenticated.
    threshold : float
        Literal strict risk > threshold comparator; no range/finite check.
        NaN produces no detections. Normal risk thresholds lie in [0, 1].

    Returns
    -------
    dict[str, Any]
        Dimensionless tpr/fpr. Windows contain 128 past samples and endpoints
        advance by 20 from index 128, excluding len(signal).
        A disruptive alarm at/after t_disrupt counts as FN; safe alarms count
        as FP. Missing-class denominators are clamped to one, so an empty
        cohort yields zero rates rather than a usable discrimination result.

    Raises
    ------
    KeyError, TypeError, ValueError
        Malformed mappings, arrays or core risk inputs propagate.

    Notes
    -----
    The last signal sample in each window scales n1 by 0.2 for ntm, n2 by
    0.1 for density_limit and radial spread by 1.0 for vde; other modes use
    n1=0.05. These ad hoc observable scales do not harmonize physical units
    across mechanisms. Rates do not establish a validated warning time.
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for shot in shots:
        signal, label, t_dis_true, mode = shot["signal"], shot["label"], shot["t_disrupt"], shot["mode"]
        win_size = 128
        detected = False
        t_detect = -1
        for t in range(win_size, len(signal), 20):
            window = signal[t - win_size : t]
            val = window[-1]
            obs = {}
            # More realistic scaling to match predictor weights
            if mode == "ntm":
                obs["toroidal_n1_amp"] = val * 0.2
            elif mode == "density_limit":
                obs["toroidal_n2_amp"] = val * 0.1
            elif mode == "vde":
                obs["toroidal_radial_spread"] = val * 1.0
            else:
                obs["toroidal_n1_amp"] = 0.05

            risk = predict_disruption_risk(window, obs)
            if risk > threshold:
                detected = True
                t_detect = t
                break

        if label == 1:
            # TP only if detected BEFORE actual disruption
            if detected and t_detect < t_dis_true:
                tp += 1
            else:
                fn += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
    return {"tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}


def main() -> None:
    """Write a fixed 100-shot, 51-threshold synthetic ROC diagnostic.

    No CLI parser. The cohort seed is 42; thresholds span [0, 1].

    Returns
    -------
    None
        Caller-relative validation/reports/disruption_roc.json and .md are
        created sequentially with the platform text codec. JSON contains AUC
        and raw swept tpr/fpr lists. Markdown prints PASS only for AUC > 0.85;
        FAIL still exits normally. Existing reports may be overwritten.

    Raises
    ------
    OSError
        Directory creation/writing fails; partial output can remain.
    ValueError, TypeError
        Simulation, evaluation or shared curve assembly errors propagate.

    Notes
    -----
    Shared AUC assembly adds canonical endpoint pairs, sorts FPR/TPR and
    uses trapezoidal integration; those added points are absent from the
    reported raw arrays. No recording guard, output transaction or hardware/
    facility/scientific admission is provided. Run from a temporary cwd for
    software verification, preserving canonical scientific reports.
    """
    print("Generating batch...")
    shots = generate_scenario_batch(100)
    thresholds = np.linspace(0.0, 1.0, 51)
    tpr_list, fpr_list = [], []
    print("Sweeping...")
    for th in thresholds:
        res = evaluate_batch(shots, th)
        tpr_list.append(res["tpr"])
        fpr_list.append(res["fpr"])

    # AUC assembly (endpoint forcing, FPR sort, trapezoidal integration) is the
    # shared disruption_roc.roc_auc_from_curve helper; the reported curve keeps
    # the raw swept points.
    auc = roc_auc_from_curve(fpr_list, tpr_list)

    results = {"auc": float(auc), "tpr": [float(x) for x in tpr_list], "fpr": [float(x) for x in fpr_list]}
    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "disruption_roc.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(report_dir / "disruption_roc.md", "w") as f:
        f.write(f"# Disruption Predictor ROC Analysis\n\n- **AUC**: {auc:.4f}\n")
        f.write(f"Result: {'PASS' if auc > 0.85 else 'FAIL'}\n")
    print(f"AUC={auc:.4f}")


if __name__ == "__main__":
    main()
