#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Streamlit Cloud Entry Point
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Root entry point for Streamlit Cloud deployment.

Runs the phase sync WS client in embedded mode (server + client in one
process) so no external WebSocket server is needed.

Deploy to Streamlit Cloud:
    1. Push to GitHub
    2. https://share.streamlit.io → New app → anulum/scpn-control → streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import streamlit as st
except ImportError:
    raise SystemExit("pip install streamlit")

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Page config ──
st.set_page_config(
    page_title="SCPN Phase Sync — Live Dashboard",
    page_icon="🌀",
    layout="wide",
)

HISTORY_LEN = 500
DEFAULT_PORT = 8765


def _try_ws_connect(url: str, buffer: deque, stop_event: threading.Event):
    try:
        import websockets.sync.client as ws_sync
    except ImportError:
        raise ImportError("pip install websockets")

    while not stop_event.is_set():
        try:
            with ws_sync.connect(url) as ws:
                while not stop_event.is_set():
                    raw = ws.recv(timeout=2.0)
                    frame = json.loads(raw)
                    buffer.append(frame)
        except Exception:
            if not stop_event.is_set():
                time.sleep(0.5)


def _run_embedded_server(layers, n_per, zeta, port, stop_event: threading.Event):
    import asyncio
    from scpn_control.phase.realtime_monitor import RealtimeMonitor
    from scpn_control.phase.ws_phase_stream import PhaseStreamServer

    mon = RealtimeMonitor.from_paper27(L=layers, N_per=n_per, zeta_uniform=zeta)
    server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)

    loop = asyncio.new_event_loop()

    async def _serve_until_stop():
        task = asyncio.ensure_future(server.serve(port=port))
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        task.cancel()

    loop.run_until_complete(_serve_until_stop())


# ── Session state init ──
if "ws_buffer" not in st.session_state:
    st.session_state.ws_buffer = deque(maxlen=HISTORY_LEN)
    st.session_state.ws_stop = threading.Event()
    st.session_state.ws_thread = None
    st.session_state.server_thread = None
    st.session_state.auto_started = False

# ── Header ──
st.title("SCPN Phase Sync — Live Dashboard")
st.caption(
    "Kuramoto–Sakaguchi mean-field with ζ sin(Ψ−θ) global driver | "
    "Paper 27 Knm matrix | "
    "[GitHub](https://github.com/anulum/scpn-control)"
)

# ── Sidebar controls ──
with st.sidebar:
    st.header("Configuration")
    layers = st.slider("Layers (L)", 2, 32, 16)
    n_per = st.slider("Oscillators per layer", 10, 200, 50)
    zeta = st.slider("ζ (global coupling)", 0.0, 2.0, 0.5, 0.05)

    st.divider()
    st.header("Connection")
    ws_url = st.text_input("WebSocket URL", f"ws://localhost:{DEFAULT_PORT}")

    col_start, col_stop = st.columns(2)
    if col_start.button("Start", type="primary"):
        st.session_state.ws_stop.clear()
        st.session_state.ws_buffer.clear()

        if st.session_state.server_thread is None or not st.session_state.server_thread.is_alive():
            st.session_state.server_thread = threading.Thread(
                target=_run_embedded_server,
                args=(layers, n_per, zeta, DEFAULT_PORT, st.session_state.ws_stop),
                daemon=True,
            )
            st.session_state.server_thread.start()
            time.sleep(0.5)

        if st.session_state.ws_thread is None or not st.session_state.ws_thread.is_alive():
            st.session_state.ws_thread = threading.Thread(
                target=_try_ws_connect,
                args=(ws_url, st.session_state.ws_buffer, st.session_state.ws_stop),
                daemon=True,
            )
            st.session_state.ws_thread.start()

    if col_stop.button("Stop"):
        st.session_state.ws_stop.set()

    st.divider()
    st.header("Server Commands")
    new_psi = st.slider("Ψ driver", -3.14, 3.14, 0.0, 0.1)
    if st.button("Send set_psi"):
        try:
            import websockets.sync.client as ws_sync
            with ws_sync.connect(ws_url) as ws:
                ws.send(json.dumps({"action": "set_psi", "value": new_psi}))
        except Exception as e:
            st.error(str(e))

    if st.button("Reset"):
        try:
            import websockets.sync.client as ws_sync
            with ws_sync.connect(ws_url) as ws:
                ws.send(json.dumps({"action": "reset"}))
        except Exception as e:
            st.error(str(e))

# ── Auto-start on first load (Streamlit Cloud) ──
if not st.session_state.auto_started:
    st.session_state.auto_started = True
    st.session_state.ws_stop.clear()
    st.session_state.ws_buffer.clear()

    st.session_state.server_thread = threading.Thread(
        target=_run_embedded_server,
        args=(16, 50, 0.5, DEFAULT_PORT, st.session_state.ws_stop),
        daemon=True,
    )
    st.session_state.server_thread.start()
    time.sleep(0.5)

    st.session_state.ws_thread = threading.Thread(
        target=_try_ws_connect,
        args=(f"ws://localhost:{DEFAULT_PORT}", st.session_state.ws_buffer, st.session_state.ws_stop),
        daemon=True,
    )
    st.session_state.ws_thread.start()
    time.sleep(0.3)

# ── Main display ──
buf = st.session_state.ws_buffer
frames = list(buf)

connected = (
    st.session_state.ws_thread is not None
    and st.session_state.ws_thread.is_alive()
    and not st.session_state.ws_stop.is_set()
)

status_col1, status_col2 = st.columns(2)
status_col1.metric("Status", "CONNECTED" if connected else "DISCONNECTED")
status_col2.metric("Frames buffered", len(frames))

if not frames:
    st.info("Starting embedded phase sync server... data arriving shortly.")
    time.sleep(1.0)
    st.rerun()

# Extract time series
ticks = [f["tick"] for f in frames]
r_global = [f["R_global"] for f in frames]
v_global = [f["V_global"] for f in frames]
lam_exp = [f["lambda_exp"] for f in frames]
last = frames[-1]

# ── Metrics row ──
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("R_global", f"{last['R_global']:.4f}")
m2.metric("V_global", f"{last['V_global']:.4f}")
m3.metric("λ exponent", f"{last['lambda_exp']:.4f}")
m4.metric("Guard", "PASS" if last["guard_approved"] else "HALT")
m5.metric("Latency", f"{last['latency_us']:.0f} µs")

# ── Plots ──
if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    axes[0, 0].plot(ticks, r_global, linewidth=0.8, color="#3b82f6")
    axes[0, 0].set_ylabel("R_global", color="#94a3b8")
    axes[0, 0].set_title("Global Coherence", color="#e2e8f0")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(ticks, v_global, linewidth=0.8, color="#8b5cf6")
    axes[0, 1].set_ylabel("V_global", color="#94a3b8")
    axes[0, 1].set_title("Lyapunov V(t)", color="#e2e8f0")

    axes[1, 0].plot(ticks, lam_exp, linewidth=0.8, color="#f59e0b")
    axes[1, 0].axhline(0, color="#666", linestyle="--", linewidth=0.5)
    axes[1, 0].set_ylabel("λ", color="#94a3b8")
    axes[1, 0].set_xlabel("tick", color="#94a3b8")
    axes[1, 0].set_title("Lyapunov Exponent", color="#e2e8f0")

    if "R_layer" in last:
        r_layer = np.array(last["R_layer"])
        colors = plt.cm.Blues(0.3 + 0.7 * r_layer)
        axes[1, 1].bar(range(len(r_layer)), r_layer, color=colors, width=0.8)
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

# ── Raw frame expander ──
with st.expander("Last WebSocket frame (raw JSON)"):
    st.json(last)

# ── Auto-refresh ──
if connected:
    time.sleep(0.3)
    st.rerun()
