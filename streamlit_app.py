#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Streamlit Cloud Entry Point
# © 1998–2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Streamlit Cloud entry point for live phase sync dashboard.

Runs RealtimeMonitor directly in-process (no WebSocket needed).
Deploy: share.streamlit.io > New app > anulum/scpn-control > streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

# Ensure scpn_control is importable (src-layout, no pip install needed)
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import streamlit as st

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

st.set_page_config(
    page_title="SCPN Phase Sync",
    layout="wide",
)

HISTORY_LEN = 500
TICK_INTERVAL = 0.01


def _tick_loop(layers: int, n_per: int, zeta: float, psi: float,
               buffer: deque, stop: threading.Event):
    """Background thread: run RealtimeMonitor, push snapshots to buffer."""
    from scpn_control.phase.realtime_monitor import RealtimeMonitor

    mon = RealtimeMonitor.from_paper27(
        L=layers, N_per=n_per, zeta_uniform=zeta, psi_driver=psi,
    )
    while not stop.is_set():
        snap = mon.tick()
        buffer.append(snap)
        time.sleep(TICK_INTERVAL)


# ── Session state ──
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=HISTORY_LEN)
    st.session_state.stop = threading.Event()
    st.session_state.thread = None
    st.session_state.started = False

# ── Header ──
st.title("SCPN Phase Sync")
st.caption(
    "Kuramoto-Sakaguchi mean-field | Paper 27 Knm matrix | "
    "[GitHub](https://github.com/anulum/scpn-control)"
)

# ── Sidebar ──
with st.sidebar:
    st.header("Parameters")
    layers = st.slider("Layers (L)", 2, 32, 16)
    n_per = st.slider("Oscillators / layer", 10, 200, 50)
    zeta = st.slider("zeta (global coupling)", 0.0, 2.0, 0.5, 0.05)
    psi = st.slider("Psi driver", -3.14, 3.14, 0.0, 0.1)

    st.divider()
    col1, col2 = st.columns(2)
    start = col1.button("Start", type="primary")
    stop_btn = col2.button("Stop")

    if start:
        st.session_state.stop.clear()
        st.session_state.buffer.clear()
        if st.session_state.thread is None or not st.session_state.thread.is_alive():
            st.session_state.thread = threading.Thread(
                target=_tick_loop,
                args=(layers, n_per, zeta, psi,
                      st.session_state.buffer, st.session_state.stop),
                daemon=True,
            )
            st.session_state.thread.start()
            st.session_state.started = True

    if stop_btn:
        st.session_state.stop.set()
        st.session_state.started = False

# ── Auto-start on first load ──
if not st.session_state.started:
    st.session_state.stop.clear()
    st.session_state.buffer.clear()
    st.session_state.thread = threading.Thread(
        target=_tick_loop,
        args=(16, 50, 0.5, 0.0,
              st.session_state.buffer, st.session_state.stop),
        daemon=True,
    )
    st.session_state.thread.start()
    st.session_state.started = True
    time.sleep(0.3)

# ── Main display ──
frames = list(st.session_state.buffer)
running = (
    st.session_state.thread is not None
    and st.session_state.thread.is_alive()
    and not st.session_state.stop.is_set()
)

c1, c2 = st.columns(2)
c1.metric("Status", "RUNNING" if running else "STOPPED")
c2.metric("Ticks buffered", len(frames))

if not frames:
    st.info("Starting phase sync engine... data arriving shortly.")
    time.sleep(0.5)
    st.rerun()

ticks = [f["tick"] for f in frames]
r_global = [f["R_global"] for f in frames]
v_global = [f["V_global"] for f in frames]
lam_exp = [f["lambda_exp"] for f in frames]
last = frames[-1]

# ── Metrics ──
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("R_global", f"{last['R_global']:.4f}")
m2.metric("V_global", f"{last['V_global']:.4f}")
m3.metric("lambda", f"{last['lambda_exp']:.4f}")
m4.metric("Guard", "PASS" if last["guard_approved"] else "HALT")
m5.metric("Latency", f"{last['latency_us']:.0f} us")

# ── Plots ──
if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    axes[0, 0].plot(ticks, r_global, lw=0.8, color="#3b82f6")
    axes[0, 0].set_ylabel("R_global", color="#94a3b8")
    axes[0, 0].set_title("Global Coherence", color="#e2e8f0")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(ticks, v_global, lw=0.8, color="#8b5cf6")
    axes[0, 1].set_ylabel("V_global", color="#94a3b8")
    axes[0, 1].set_title("Lyapunov V(t)", color="#e2e8f0")

    axes[1, 0].plot(ticks, lam_exp, lw=0.8, color="#f59e0b")
    axes[1, 0].axhline(0, color="#666", ls="--", lw=0.5)
    axes[1, 0].set_ylabel("lambda", color="#94a3b8")
    axes[1, 0].set_xlabel("tick", color="#94a3b8")
    axes[1, 0].set_title("Lyapunov Exponent", color="#e2e8f0")

    if "R_layer" in last:
        r_layer = np.array(last["R_layer"])
        bar_colors = plt.cm.Blues(0.3 + 0.7 * r_layer)
        axes[1, 1].bar(range(len(r_layer)), r_layer, color=bar_colors, width=0.8)
        axes[1, 1].set_ylabel("R_layer", color="#94a3b8")
        axes[1, 1].set_xlabel("Layer", color="#94a3b8")
        axes[1, 1].set_title("Per-Layer Coherence", color="#e2e8f0")
        axes[1, 1].set_ylim(0, 1.05)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.line_chart({"R_global": r_global, "V_global": v_global, "lambda": lam_exp})

# ── Guard log ──
approved = [f["guard_approved"] for f in frames]
halt_ticks = [t for t, a in zip(ticks, approved) if not a]
if halt_ticks:
    st.warning(f"Guard HALT at ticks: {halt_ticks[-10:]}")

# ── Raw data ──
with st.expander("Last tick (raw)"):
    st.json(last)

# ── Auto-refresh ──
if running:
    time.sleep(0.3)
    st.rerun()
