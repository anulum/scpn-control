#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Phase Sync Convergence Video Generator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Generate an animated video of RealtimeMonitor phase sync convergence.

Produces docs/phase_sync_live.gif (always) and docs/phase_sync_live.mp4
(when ffmpeg is available).

Usage::

    python tools/generate_phase_video.py
    python tools/generate_phase_video.py --ticks 500 --fps 30 --layers 8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scpn_control.phase.realtime_monitor import RealtimeMonitor

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def generate(n_ticks: int, L: int, N_per: int, zeta: float, fps: int):
    mon = RealtimeMonitor.from_paper27(
        L=L, N_per=N_per, zeta_uniform=zeta, psi_driver=0.0, seed=42,
    )

    # Pre-compute all ticks
    snaps = []
    for _ in range(n_ticks):
        snaps.append(mon.tick())

    r_global = [s["R_global"] for s in snaps]
    v_global = [s["V_global"] for s in snaps]
    lam_exp = [s["lambda_exp"] for s in snaps]
    r_layer_all = np.array([s["R_layer"] for s in snaps])

    # Subsample frames (target ~10s video)
    target_frames = fps * 10
    step = max(1, n_ticks // target_frames)
    frame_indices = list(range(0, n_ticks, step))
    if frame_indices[-1] != n_ticks - 1:
        frame_indices.append(n_ticks - 1)
    n_frames = len(frame_indices)

    # Dark theme
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    fig.suptitle(
        f"SCPN Phase Sync — {L} layers × {N_per} osc, ζ={zeta}",
        color="#e2e8f0", fontsize=13, fontweight="bold",
    )

    # R_global
    ax_r = axes[0, 0]
    line_r, = ax_r.plot([], [], color="#3b82f6", linewidth=1.5)
    ax_r.set_xlim(0, n_ticks)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("R_global", color="#94a3b8", fontsize=10)
    ax_r.set_title("Global Coherence", color="#e2e8f0", fontsize=11)
    ax_r.axhline(0.9, color="#22c55e", linewidth=0.5, linestyle="--", alpha=0.5)

    # V_global
    ax_v = axes[0, 1]
    line_v, = ax_v.plot([], [], color="#8b5cf6", linewidth=1.5)
    ax_v.set_xlim(0, n_ticks)
    ax_v.set_ylim(0, max(v_global) * 1.1 + 0.01)
    ax_v.set_ylabel("V_global", color="#94a3b8", fontsize=10)
    ax_v.set_title("Lyapunov V(t)", color="#e2e8f0", fontsize=11)

    # Lambda
    ax_l = axes[1, 0]
    line_l, = ax_l.plot([], [], color="#f59e0b", linewidth=1.5)
    ax_l.set_xlim(0, n_ticks)
    lam_min = min(lam_exp) * 1.2 if min(lam_exp) < 0 else -1
    lam_max = max(max(lam_exp) * 1.2, 0.5)
    ax_l.set_ylim(lam_min, lam_max)
    ax_l.set_ylabel("λ exponent", color="#94a3b8", fontsize=10)
    ax_l.set_xlabel("tick", color="#94a3b8", fontsize=10)
    ax_l.set_title("Lyapunov Exponent", color="#e2e8f0", fontsize=11)
    ax_l.axhline(0, color="#ef4444", linewidth=0.5, linestyle="--", alpha=0.5)

    # Per-layer R bars
    ax_b = axes[1, 1]
    bar_x = np.arange(L)
    bars = ax_b.bar(bar_x, np.zeros(L), color="#3b82f6", width=0.7, alpha=0.85)
    ax_b.set_xlim(-0.5, L - 0.5)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("R_layer", color="#94a3b8", fontsize=10)
    ax_b.set_xlabel("Layer", color="#94a3b8", fontsize=10)
    ax_b.set_title("Per-Layer Coherence", color="#e2e8f0", fontsize=11)
    if L <= 16:
        ax_b.set_xticks(range(0, L, max(1, L // 8)))

    # Metrics text
    metrics_text = fig.text(
        0.5, 0.01,
        "", ha="center", va="bottom",
        color="#94a3b8", fontsize=10, fontfamily="monospace",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    # Color map for bars
    cmap = plt.cm.Blues

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        t = idx + 1

        # Update traces
        line_r.set_data(range(t), r_global[:t])
        line_v.set_data(range(t), v_global[:t])
        line_l.set_data(range(t), lam_exp[:t])

        # Update bars
        r_vals = r_layer_all[idx]
        for bar, val in zip(bars, r_vals):
            bar.set_height(val)
            bar.set_color(cmap(0.3 + 0.7 * val))

        # Update metrics
        s = snaps[idx]
        guard = "PASS" if s["guard_approved"] else "HALT"
        metrics_text.set_text(
            f"tick {t}/{n_ticks}    "
            f"R={s['R_global']:.3f}    "
            f"V={s['V_global']:.3f}    "
            f"λ={s['lambda_exp']:.3f}    "
            f"guard={guard}    "
            f"lat={s['latency_us']:.0f}µs"
        )

        return [line_r, line_v, line_l, metrics_text] + list(bars)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)

    # Save GIF
    gif_path = DOCS / "phase_sync_live.gif"
    print(f"Writing {gif_path} ({n_frames} frames, {fps} fps)...")
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))
    gif_size = gif_path.stat().st_size / 1024 / 1024
    print(f"  {gif_path.name}: {gif_size:.1f} MB")

    # Try MP4
    if shutil.which("ffmpeg"):
        mp4_path = DOCS / "phase_sync_live.mp4"
        print(f"Writing {mp4_path}...")
        from matplotlib.animation import FFMpegWriter
        anim.save(str(mp4_path), writer=FFMpegWriter(fps=fps, bitrate=2000))
        mp4_size = mp4_path.stat().st_size / 1024 / 1024
        print(f"  {mp4_path.name}: {mp4_size:.1f} MB")
    else:
        print("  ffmpeg not found — skipping MP4. Install: choco install ffmpeg")

    plt.close(fig)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Generate phase sync convergence video")
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n-per", type=int, default=50)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    generate(args.ticks, args.layers, args.n_per, args.zeta, args.fps)


if __name__ == "__main__":
    main()
