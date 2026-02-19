# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Streamlit Control Dashboard
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Requires: pip install scpn-control[dashboard]
# Run: streamlit run dashboard/control_dashboard.py
# ──────────────────────────────────────────────────────────────────────
"""Interactive control dashboard for SCPN Control pipeline."""
from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Streamlit not installed. Run: pip install scpn-control[dashboard]"
    )

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    st.error("matplotlib not installed — plots disabled")

st.set_page_config(page_title="SCPN Control Dashboard", layout="wide")
st.title("SCPN Control Dashboard")

tab_traj, tab_rmse, tab_bench, tab_replay = st.tabs([
    "Trajectory Viewer", "RMSE Dashboard", "Timing Benchmark", "Shot Replay"
])

# ─── Trajectory Viewer ───
with tab_traj:
    st.header("Closed-Loop Trajectory")
    steps = st.slider("Simulation steps", 100, 5000, 1000, step=100)
    scenario = st.selectbox("Scenario", ["PID", "SNN", "Combined"])

    if st.button("Run Simulation", key="run_traj"):
        rng = np.random.default_rng(42)
        target = 1.0
        state = 0.5
        states, errors = [], []
        for _ in range(steps):
            error = target - state
            action = 0.1 * error + 0.01 * rng.standard_normal()
            state = np.clip(state + action, 0.0, 2.0)
            states.append(state)
            errors.append(error)

        if HAS_MPL:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(states, linewidth=0.8)
            ax1.axhline(target, color="r", linestyle="--", alpha=0.7, label="Target")
            ax1.set_ylabel("State")
            ax1.legend()
            ax2.plot(errors, linewidth=0.8, color="orange")
            ax2.set_ylabel("Error")
            ax2.set_xlabel("Step")
            st.pyplot(fig)
        else:
            st.line_chart({"state": states, "error": errors})

# ─── RMSE Dashboard ───
with tab_rmse:
    st.header("RMSE Validation")
    st.info("Run `scpn-control validate` from CLI for full results.")

# ─── Timing Benchmark ───
with tab_bench:
    st.header("Controller Latency Benchmark")
    n_bench = st.number_input("Iterations", 100, 50000, 5000, step=1000)
    if st.button("Run Benchmark", key="run_bench"):
        import time
        # PID
        t0 = time.perf_counter()
        kp, ki, kd = 1.0, 0.1, 0.01
        integral, prev_error = 0.0, 0.0
        for _ in range(n_bench):
            error = np.random.randn()
            integral += error
            derivative = error - prev_error
            _ = kp * error + ki * integral + kd * derivative
            prev_error = error
        pid_us = (time.perf_counter() - t0) / n_bench * 1e6

        # SNN
        t0 = time.perf_counter()
        v = np.zeros(50)
        for _ in range(n_bench):
            error = np.random.randn()
            v = v * 0.9 + error * np.random.randn(50) * 0.1
            spikes = v > 1.0
            v[spikes] = 0.0
        snn_us = (time.perf_counter() - t0) / n_bench * 1e6

        col1, col2, col3 = st.columns(3)
        col1.metric("PID", f"{pid_us:.1f} us/step")
        col2.metric("SNN", f"{snn_us:.1f} us/step")
        col3.metric("Ratio", f"{pid_us / max(snn_us, 1e-9):.1f}x")

# ─── Shot Replay ───
with tab_replay:
    st.header("Disruption Shot Replay")
    from pathlib import Path
    shots_dir = Path("validation/reference_data/diiid/disruption_shots")
    if shots_dir.exists():
        shot_files = sorted(shots_dir.glob("*.npz"))
        selected = st.selectbox("Shot", [f.stem for f in shot_files])
        if selected and st.button("Load Shot", key="load_shot"):
            data = np.load(shots_dir / f"{selected}.npz", allow_pickle=True)
            st.write(f"Arrays: {list(data.keys())}")
            for key in data.keys():
                arr = data[key]
                if arr.ndim == 1 and len(arr) < 10000:
                    st.line_chart(arr)
    else:
        st.warning(f"Shots directory not found: {shots_dir}")
