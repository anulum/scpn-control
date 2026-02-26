#!/usr/bin/env python3
"""
SCPN Phase Sync — Streamlit Cloud Dashboard.

Runs Kuramoto-Sakaguchi phase dynamics in-process (no external server).
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

# src-layout: add src/ to path so scpn_control is importable
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import streamlit as st

st.set_page_config(page_title="SCPN Phase Sync", layout="wide")

HISTORY = 500
DT = 0.01


# ── Inline Kuramoto engine (no external import needed) ──
def _kuramoto_tick(theta, omega, K, zeta, psi, dt=1e-3):
    """Single Kuramoto-Sakaguchi step with global driver."""
    N = len(theta)
    R_vec = np.exp(1j * theta)
    Z = np.mean(R_vec)
    R_global = float(np.abs(Z))
    Psi_mean = float(np.angle(Z))

    dtheta = omega.copy()
    for i in range(N):
        coupling = np.sum(K[i] * np.sin(theta - theta[i]))
        dtheta[i] += coupling + zeta * np.sin(psi - theta[i])

    theta_new = theta + dtheta * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    V = float(np.mean(1.0 - np.cos(theta_new - psi)))

    return theta_new, R_global, Psi_mean, V


def _build_knm(L):
    """Simple exponential-decay coupling matrix."""
    K = np.zeros((L, L))
    K_base = 0.45
    alpha = 0.3
    for i in range(L):
        for j in range(L):
            if i != j:
                K[i, j] = K_base * np.exp(-alpha * abs(i - j))
    return K


def _tick_loop(L, N_per, zeta, psi, buf, stop_ev):
    """Background: run multi-layer Kuramoto, push snapshots."""
    rng = np.random.default_rng(42)
    omega_base = np.linspace(1.3, 1.0, L)

    # Per-layer oscillators
    theta_layers = [rng.uniform(-np.pi, np.pi, N_per) for _ in range(L)]
    omega_layers = [
        omega_base[m] + 0.05 * rng.standard_normal(N_per) for m in range(L)
    ]
    K = _build_knm(L)

    tick_num = 0
    lam_history = deque(maxlen=50)

    while not stop_ev.is_set():
        t0 = time.perf_counter_ns()
        tick_num += 1

        R_layer = []
        V_total = 0.0
        for m in range(L):
            # Intra-layer coupling (mean-field)
            Z_m = np.mean(np.exp(1j * theta_layers[m]))
            R_m = float(np.abs(Z_m))
            Psi_m = float(np.angle(Z_m))
            R_layer.append(R_m)

            dtheta = omega_layers[m].copy()
            # Intra-layer
            dtheta += R_m * np.sin(Psi_m - theta_layers[m])
            # Inter-layer
            for n in range(L):
                if n != m:
                    Z_n = np.mean(np.exp(1j * theta_layers[n]))
                    R_n = float(np.abs(Z_n))
                    Psi_n = float(np.angle(Z_n))
                    dtheta += K[n, m] * R_n * np.sin(Psi_n - theta_layers[m])
            # Global driver
            dtheta += zeta * np.sin(psi - theta_layers[m])

            theta_layers[m] = theta_layers[m] + dtheta * 1e-3
            theta_layers[m] = (theta_layers[m] + np.pi) % (2 * np.pi) - np.pi

            V_total += float(np.mean(1.0 - np.cos(theta_layers[m] - psi)))

        # Global order parameter
        all_theta = np.concatenate(theta_layers)
        Z_global = np.mean(np.exp(1j * all_theta))
        R_global = float(np.abs(Z_global))
        V_global = V_total / L

        # Lyapunov exponent estimate
        lam_history.append(V_global)
        if len(lam_history) >= 10:
            v_arr = np.array(lam_history)
            dv = np.diff(v_arr)
            lam_exp = float(np.mean(np.sign(dv)))
        else:
            lam_exp = 0.0

        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        snap = {
            "tick": tick_num,
            "R_global": R_global,
            "V_global": V_global,
            "lambda_exp": lam_exp,
            "R_layer": R_layer,
            "guard_approved": lam_exp <= 0.0 or tick_num < 20,
            "latency_us": latency_us,
        }
        buf.append(snap)
        time.sleep(DT)


# ── Session state ──
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=HISTORY)
    st.session_state.stop = threading.Event()
    st.session_state.thread = None
    st.session_state.running = False

# ── Header ──
st.title("SCPN Phase Sync")
st.caption(
    "Kuramoto-Sakaguchi mean-field | Paper 27 Knm coupling | "
    "[GitHub](https://github.com/anulum/scpn-control)"
)

# ── Sidebar ──
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
            st.session_state.running = True

    if do_stop:
        st.session_state.stop.set()
        st.session_state.running = False

# ── Auto-start on first visit ──
if not st.session_state.running and (
    st.session_state.thread is None or not st.session_state.thread.is_alive()
):
    st.session_state.stop.clear()
    st.session_state.buffer.clear()
    st.session_state.thread = threading.Thread(
        target=_tick_loop,
        args=(16, 50, 0.5, 0.0,
              st.session_state.buffer, st.session_state.stop),
        daemon=True,
    )
    st.session_state.thread.start()
    st.session_state.running = True
    time.sleep(0.3)

# ── Main ──
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

# ── Metrics ──
m1, m2, m3, m4 = st.columns(4)
m1.metric("R global", f"{last['R_global']:.4f}")
m2.metric("V global", f"{last['V_global']:.4f}")
m3.metric("lambda", f"{last['lambda_exp']:.3f}")
m4.metric("Guard", "PASS" if last["guard_approved"] else "HALT")

# ── Charts ──
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

# ── Guard ──
approved = [f["guard_approved"] for f in frames]
halts = [t for t, a in zip(ticks, approved) if not a]
if halts:
    st.warning(f"Guard HALT at ticks: {halts[-10:]}")

with st.expander("Raw data (last tick)"):
    st.json(last)

# ── Refresh ──
if is_live:
    time.sleep(0.3)
    st.rerun()
