# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Control Resilience Campaign.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Control Resilience Campaign
# © 1996–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic fault/noise campaign for disruption-control resilience."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scpn_control.control.disruption_predictor import run_fault_noise_campaign
from validation.resilience_campaign_inputs import _normalize_campaign_inputs as _normalize_campaign_inputs


def generate_campaign_report(
    *,
    seed: int = 42,
    episodes: int = 64,
    window: int = 128,
    noise_std: float = 0.03,
    bit_flip_interval: int = 11,
    recovery_window: int = 6,
    recovery_epsilon: float = 0.03,
) -> dict[str, Any]:
    """Measure a deterministic synthetic disruption-risk fault campaign.

    Parameters
    ----------
    seed : int
        Seed for the core's local NumPy generator; normalised with int().
    episodes, window : int
        Number of synthetic traces and samples per trace, >= 1 and >= 16.
    noise_std : float
        Non-negative finite additive noise in synthetic signal units.
    bit_flip_interval : int
        Positive sample interval for binary64 mantissa-bit flips.
    recovery_window : int
        Positive maximum recovery search offset, in samples rather than seconds.
    recovery_epsilon : float
        Positive finite tolerance on dimensionless absolute risk error.

    Returns
    -------
    dict[str, Any]
        UTC generation time, measured runtime_seconds around the core call,
        and campaign metrics. Metrics include risk-error mean/p95, recovery
        offset p95/success fraction, fault count and four threshold checks.
        For fixed normalised inputs campaign metrics repeat; time does not.

    Raises
    ------
    ValueError, TypeError, OverflowError
        Local normalisation or core input validation fails before running.

    Notes
    -----
    The core uses synthetic traces and perturbed observables, no plant actuator
    or hardware faults. Recovery offsets use a tail-capped finite search;
    threshold pass is a synthetic diagnostic, not proven controller recovery.
    Thresholds bound mean error at 0.08, p95 error at 0.22, recovery p95 at
    recovery_window and success fraction at 0.80. No report is written here.
    """
    (
        seed_i,
        episodes_i,
        window_i,
        noise,
        bit_flip_i,
        recovery_window_i,
        recovery_eps,
    ) = _normalize_campaign_inputs(
        seed=seed,
        episodes=episodes,
        window=window,
        noise_std=noise_std,
        bit_flip_interval=bit_flip_interval,
        recovery_window=recovery_window,
        recovery_epsilon=recovery_epsilon,
    )
    start = time.perf_counter()
    metrics = run_fault_noise_campaign(
        seed=seed_i,
        episodes=episodes_i,
        window=window_i,
        noise_std=noise,
        bit_flip_interval=bit_flip_i,
        recovery_window=recovery_window_i,
        recovery_epsilon=recovery_eps,
    )
    elapsed = time.perf_counter() - start
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": elapsed,
        "campaign": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Format one campaign report without running or writing a campaign.

    Parameters
    ----------
    report : dict[str, Any]
        Mapping returned by generate_campaign_report, including all metric
        and threshold fields; no independent schema validation is performed.

    Returns
    -------
    str
        Markdown ending in a newline. Runtime is rounded to three decimals,
        risk errors to six, and recovery statistics to three.
        Threshold pass is printed as YES or NO.

    Raises
    ------
    KeyError, TypeError, ValueError
        Missing fields or incompatible format values propagate to the caller.
    """
    c = report["campaign"]
    lines = [
        "# Control Resilience Campaign",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Seed: `{c['seed']}`",
        f"- Episodes: `{c['episodes']}`",
        f"- Window: `{c['window']}`",
        f"- Noise std: `{c['noise_std']}`",
        f"- Bit-flip interval: `{c['bit_flip_interval']}`",
        f"- Fault count: `{c['fault_count']}`",
        "",
        "## Metrics",
        "",
        f"- Mean abs risk error: `{c['mean_abs_risk_error']:.6f}`",
        f"- P95 abs risk error: `{c['p95_abs_risk_error']:.6f}`",
        f"- P95 recovery steps: `{c['recovery_steps_p95']:.3f}`",
        f"- Recovery success rate: `{c['recovery_success_rate']:.3f}`",
        f"- Threshold pass: `{'YES' if c['passes_thresholds'] else 'NO'}`",
        "",
        "## Thresholds",
        "",
        f"- Max mean abs risk error: `{c['thresholds']['max_mean_abs_risk_error']}`",
        f"- Max P95 abs risk error: `{c['thresholds']['max_p95_abs_risk_error']}`",
        f"- Max P95 recovery steps: `{c['thresholds']['max_recovery_steps_p95']}`",
        f"- Min recovery success rate: `{c['thresholds']['min_recovery_success_rate']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the public campaign and write JSON/Markdown, then apply strict exit.

    Parameters
    ----------
    argv : list[str] or None
        Argparse tokens; None reads sys.argv. Options mirror the campaign
        producer, plus --output-json, --output-md and --strict.
        Default paths are under this checkout's validation/reports.

    Returns
    -------
    int
        Zero normally, including a failed non-strict campaign. Two when
        --strict is requested and the computed thresholds fail.

    Raises
    ------
    SystemExit
        Argparse help exits zero; syntax errors exit two.
    ValueError, TypeError, OverflowError
        Campaign input errors propagate before outputs are written.
    OSError
        UTF-8 parent creation/write fails. Writes are sequential and can
        overwrite prior files; no transaction, alias guard or campaign guard.

    Notes
    -----
    Both outputs and metric stdout are produced before a strict-failure
    return. A successful process exit is not a physical admission verdict.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--bit-flip-interval", type=int, default=11)
    parser.add_argument("--recovery-window", type=int, default=6)
    parser.add_argument("--recovery-epsilon", type=float, default=0.03)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "control_resilience_campaign.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "control_resilience_campaign.md"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when thresholds are not met.",
    )
    args = parser.parse_args(argv)

    report = generate_campaign_report(
        seed=args.seed,
        episodes=args.episodes,
        window=args.window,
        noise_std=args.noise_std,
        bit_flip_interval=args.bit_flip_interval,
        recovery_window=args.recovery_window,
        recovery_epsilon=args.recovery_epsilon,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    c = report["campaign"]
    print("Control resilience campaign complete.")
    print(
        "mean_abs_risk_error="
        f"{c['mean_abs_risk_error']:.6f}, "
        f"p95_abs_risk_error={c['p95_abs_risk_error']:.6f}, "
        f"recovery_steps_p95={c['recovery_steps_p95']:.3f}, "
        f"recovery_success_rate={c['recovery_success_rate']:.3f}, "
        f"passes_thresholds={c['passes_thresholds']}"
    )

    if args.strict and not c["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
