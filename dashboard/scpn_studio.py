#!/usr/bin/env python3
"""SCPN Phase Sync -- Streamlit Cloud Dashboard.

Runs Kuramoto-Sakaguchi phase dynamics via scpn-control (no inlined engine).
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import streamlit as st

from scpn_control import RealtimeMonitor

st.set_page_config(page_title="SCPN Phase Sync", layout="wide")

HISTORY = 500
DT = 0.01


def _tick_loop(monitor: RealtimeMonitor, buf: deque, stop_ev: threading.Event):
    """Background thread: call monitor.tick(), push snapshots to shared buffer."""
    while not stop_ev.is_set():
        snap = monitor.tick(record=False)
        buf.append(snap)
        time.sleep(DT)


# -- Session state --
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=HISTORY)
    st.session_state.stop = threading.Event()
    st.session_state.thread = None
    st.session_state.running = False

# -- Header --
st.title("SCPN Phase Sync")
st.caption(
    "Kuramoto-Sakaguchi mean-field | Paper 27 Knm coupling | "
    "[GitHub](https://github.com/anulum/scpn-control)"
)

# -- Sidebar --
with st.sidebar:
    st.header("Parameters")
    layers = st.slider("Layers (L)", 2, 32, 16)
    n_per = st.slider("Oscillators / layer", 10, 200, 50)
    zeta = st.slider("zeta (coupling)", 0.0, 2.0, 0.5, 0.05)
    psi = st.slider("Psi driver", -3.14, 3.14, 0.0, 0.1)

    st.divider()
    col1, col2 = st.columns(2)
    do_start = col1.button("Start", type="primary")
    do_stop = col2.button("Stop")

    if do_start:
        st.session_state.stop.set()
        if st.session_state.thread is not None:
            st.session_state.thread.join(timeout=1.0)
        st.session_state.stop.clear()
        st.session_state.buffer.clear()
        monitor = RealtimeMonitor.from_paper27(
            L=layers, N_per=n_per,
            zeta_uniform=zeta, psi_driver=psi,
        )
        st.session_state.thread = threading.Thread(
            target=_tick_loop,
            args=(monitor, st.session_state.buffer, st.session_state.stop),
            daemon=True,
        )
        st.session_state.thread.start()
        st.session_state.running = True

    if do_stop:
        st.session_state.stop.set()
        st.session_state.running = False

# -- Auto-start on first visit with defaults --
if not st.session_state.running and (
    st.session_state.thread is None or not st.session_state.thread.is_alive()
):
    st.session_state.stop.clear()
    st.session_state.buffer.clear()
    monitor = RealtimeMonitor.from_paper27(
        L=16, N_per=50, zeta_uniform=0.5, psi_driver=0.0,
    )
    st.session_state.thread = threading.Thread(
        target=_tick_loop,
        args=(monitor, st.session_state.buffer, st.session_state.stop),
        daemon=True,
    )
    st.session_state.thread.start()
    st.session_state.running = True
    time.sleep(0.3)

# -- Main --
frames = list(st.session_state.buffer)
is_live = (
    st.session_state.thread is not None
    and st.session_state.thread.is_alive()
    and not st.session_state.stop.is_set()
)

c1, c2 = st.columns(2)
c1.metric("Status", "RUNNING" if is_live else "STOPPED")
c2.metric("Ticks", len(frames))

if not frames:
    st.info("Starting phase sync engine...")
    time.sleep(0.5)
    st.rerun()

ticks = [f["tick"] for f in frames]
r_global = [f["R_global"] for f in frames]
v_global = [f["V_global"] for f in frames]
lam_exp = [f["lambda_exp"] for f in frames]
last = frames[-1]

# -- Metrics --
m1, m2, m3, m4 = st.columns(4)
m1.metric("R global", f"{last['R_global']:.4f}")
m2.metric("V global", f"{last['V_global']:.4f}")
m3.metric("lambda", f"{last['lambda_exp']:.3f}")
m4.metric("Guard", "PASS" if last["guard_approved"] else "HALT")

# -- Charts --
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    axes[0, 0].plot(ticks, r_global, lw=0.8, color="#3b82f6")
    axes[0, 0].set_ylabel("R", color="#94a3b8")
    axes[0, 0].set_title("Global Coherence", color="#e2e8f0")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(ticks, v_global, lw=0.8, color="#8b5cf6")
    axes[0, 1].set_ylabel("V", color="#94a3b8")
    axes[0, 1].set_title("Lyapunov V(t)", color="#e2e8f0")

    axes[1, 0].plot(ticks, lam_exp, lw=0.8, color="#f59e0b")
    axes[1, 0].axhline(0, color="#666", ls="--", lw=0.5)
    axes[1, 0].set_ylabel("lambda", color="#94a3b8")
    axes[1, 0].set_xlabel("tick", color="#94a3b8")
    axes[1, 0].set_title("Lyapunov Exponent", color="#e2e8f0")

    if "R_layer" in last:
        rl = np.array(last["R_layer"])
        bc = plt.cm.Blues(0.3 + 0.7 * rl)
        axes[1, 1].bar(range(len(rl)), rl, color=bc, width=0.8)
        axes[1, 1].set_ylabel("R layer", color="#94a3b8")
        axes[1, 1].set_xlabel("Layer", color="#94a3b8")
        axes[1, 1].set_title("Per-Layer Coherence", color="#e2e8f0")
        axes[1, 1].set_ylim(0, 1.05)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.line_chart({"R": r_global, "V": v_global, "lambda": lam_exp})

# -- Guard --
approved = [f["guard_approved"] for f in frames]
halts = [t for t, a in zip(ticks, approved) if not a]
if halts:
    st.warning(f"Guard HALT at ticks: {halts[-10:]}")

with st.expander("Raw data (last tick)"):
    st.json(last)

# -- Refresh --
if is_live:
    time.sleep(0.3)
    st.rerun()
