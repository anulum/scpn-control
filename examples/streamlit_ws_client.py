# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Streamlit WebSocket Phase Sync Client
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Live Streamlit client consuming PhaseStreamServer WebSocket ticks.

Start the server first::

    python -m scpn_control.phase.ws_phase_stream --port 8765 --zeta 0.5

Then run this client::

    streamlit run examples/streamlit_ws_client.py

Or run server and client from a single process (embedded mode)::

    streamlit run examples/streamlit_ws_client.py -- --embedded
"""
from __future__ import annotations

import argparse
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
    import matplotlib.animation as animation

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Parse CLI args before Streamlit eats them ──
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--embedded", action="store_true", help="Run server in-process")
_parser.add_argument("--ws-url", default="ws://localhost:8765")
_parser.add_argument("--layers", type=int, default=16)
_parser.add_argument("--n-per", type=int, default=50)
_parser.add_argument("--zeta", type=float, default=0.5)
_args, _ = _parser.parse_known_args(sys.argv[1:])

# ── Page config ──
st.set_page_config(page_title="Phase Sync — Live WS Client", layout="wide")
st.title("Phase Sync — Live WebSocket Client")

HISTORY_LEN = 500


def _try_ws_connect(url: str, buffer: deque, stop_event: threading.Event):
    """Background thread: connect to WS server, push frames into buffer."""
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


def _run_embedded_server(layers, n_per, zeta, stop_event: threading.Event):
    """Background thread: run PhaseStreamServer in-process."""
    import asyncio
    from scpn_control.phase.realtime_monitor import RealtimeMonitor
    from scpn_control.phase.ws_phase_stream import PhaseStreamServer

    mon = RealtimeMonitor.from_paper27(L=layers, N_per=n_per, zeta_uniform=zeta)
    server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)

    loop = asyncio.new_event_loop()

    async def _serve_until_stop():
        task = asyncio.ensure_future(server.serve(port=8765))
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

# ── Sidebar controls ──
with st.sidebar:
    st.header("Connection")
    ws_url = st.text_input("WebSocket URL", _args.ws_url)
    embedded = st.checkbox("Embedded server", value=_args.embedded)

    if embedded:
        st.caption("Server runs in-process on ws://localhost:8765")

    col_start, col_stop = st.columns(2)
    if col_start.button("Connect"):
        st.session_state.ws_stop.clear()
        st.session_state.ws_buffer.clear()

        if embedded and st.session_state.server_thread is None:
            st.session_state.server_thread = threading.Thread(
                target=_run_embedded_server,
                args=(_args.layers, _args.n_per, _args.zeta, st.session_state.ws_stop),
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

    if col_stop.button("Disconnect"):
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

    if st.button("Send reset"):
        try:
            import websockets.sync.client as ws_sync
            with ws_sync.connect(ws_url) as ws:
                ws.send(json.dumps({"action": "reset"}))
        except Exception as e:
            st.error(str(e))

# ── Main display ──
buf = st.session_state.ws_buffer
frames = list(buf)

status_col1, status_col2 = st.columns(2)
connected = (
    st.session_state.ws_thread is not None
    and st.session_state.ws_thread.is_alive()
    and not st.session_state.ws_stop.is_set()
)
status_col1.metric("Status", "CONNECTED" if connected else "DISCONNECTED")
status_col2.metric("Frames buffered", len(frames))

if not frames:
    st.info("No data yet. Click **Connect** in the sidebar to start streaming.")
    st.stop()

# Extract time series
ticks = [f["tick"] for f in frames]
r_global = [f["R_global"] for f in frames]
v_global = [f["V_global"] for f in frames]
lam_exp = [f["lambda_exp"] for f in frames]
latency = [f["latency_us"] for f in frames]
approved = [f["guard_approved"] for f in frames]

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    axes[0, 0].plot(ticks, r_global, linewidth=0.8, color="#3b82f6")
    axes[0, 0].set_ylabel("R_global")
    axes[0, 0].set_title("Global Coherence")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(ticks, v_global, linewidth=0.8, color="#8b5cf6")
    axes[0, 1].set_ylabel("V_global")
    axes[0, 1].set_title("Lyapunov V(t)")

    axes[1, 0].plot(ticks, lam_exp, linewidth=0.8, color="#f59e0b")
    axes[1, 0].axhline(0, color="#666", linestyle="--", linewidth=0.5)
    axes[1, 0].set_ylabel("λ")
    axes[1, 0].set_xlabel("tick")
    axes[1, 0].set_title("Lyapunov Exponent")

    # Per-layer R heatmap from last frame
    if "R_layer" in last:
        r_layer = np.array(last["R_layer"])
        colors = plt.cm.viridis(r_layer)
        axes[1, 1].bar(range(len(r_layer)), r_layer, color=colors, width=0.8)
        axes[1, 1].set_ylabel("R_layer")
        axes[1, 1].set_xlabel("Layer")
        axes[1, 1].set_title("Per-Layer Coherence (latest)")
        axes[1, 1].set_ylim(0, 1.05)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.line_chart({"R_global": r_global, "V_global": v_global, "lambda": lam_exp})

# ── Guard log ──
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
