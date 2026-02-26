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

import json
import time
from pathlib import Path

st.set_page_config(page_title="SCPN Control Dashboard", layout="wide")
st.title("SCPN Control Dashboard")

tab_traj, tab_phase, tab_vega, tab_rmse, tab_bench, tab_replay = st.tabs([
    "Trajectory Viewer", "Phase Sync Monitor", "Benchmark Plots",
    "RMSE Dashboard", "Timing Benchmark", "Shot Replay",
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

# ─── Phase Sync Monitor (RealtimeMonitor live hook) ───
with tab_phase:
    st.header("Phase Sync Monitor — Paper 27 UPDE + LyapunovGuard")

    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    L = col_cfg1.number_input("Layers", 2, 16, 8, key="phase_L")
    N_per = col_cfg2.number_input("Oscillators/layer", 10, 200, 50, key="phase_N")
    zeta_val = col_cfg3.slider("ζ (driver)", 0.0, 5.0, 0.5, 0.1, key="phase_zeta")
    psi_val = col_cfg4.slider("Ψ (target)", -3.14, 3.14, 0.0, 0.1, key="phase_psi")

    col_cfg5, col_cfg6 = st.columns(2)
    pac_val = col_cfg5.slider("PAC γ", 0.0, 2.0, 0.0, 0.1, key="phase_pac")
    n_ticks = col_cfg6.number_input("Ticks", 50, 2000, 500, step=50, key="phase_ticks")

    if st.button("Run Phase Sync", key="run_phase"):
        from scpn_control.phase.realtime_monitor import RealtimeMonitor

        mon = RealtimeMonitor.from_paper27(
            L=int(L), N_per=int(N_per),
            zeta_uniform=zeta_val, psi_driver=psi_val,
            pac_gamma=pac_val,
        )

        r_hist, v_hist, lam_hist, lat_hist = [], [], [], []
        r_layer_hist = []
        progress = st.progress(0)
        status = st.empty()

        for i in range(int(n_ticks)):
            snap = mon.tick()
            r_hist.append(snap["R_global"])
            v_hist.append(snap["V_global"])
            lam_hist.append(snap["lambda_exp"])
            lat_hist.append(snap["latency_us"])
            r_layer_hist.append(snap["R_layer"])
            if (i + 1) % 50 == 0:
                progress.progress((i + 1) / int(n_ticks))
                status.text(
                    f"tick {i+1}/{int(n_ticks)}  R={snap['R_global']:.3f}  "
                    f"λ={snap['lambda_exp']:.4f}  guard={'OK' if snap['guard_approved'] else 'HALT'}  "
                    f"lat={snap['latency_us']:.0f} µs"
                )

        progress.progress(1.0)
        final = snap

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("R_global (final)", f"{final['R_global']:.4f}")
        col_m2.metric("λ (Lyapunov)", f"{final['lambda_exp']:.4f}")
        col_m3.metric("Guard", "APPROVED" if final["guard_approved"] else "HALT")
        col_m4.metric("Latency P50", f"{sorted(lat_hist)[len(lat_hist)//2]:.0f} µs")

        if HAS_MPL:
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            axes[0, 0].plot(r_hist, linewidth=0.8, color="#4c78a8")
            axes[0, 0].set_ylabel("R_global")
            axes[0, 0].set_title("Global Coherence")

            axes[0, 1].plot(v_hist, linewidth=0.8, color="#e45756")
            axes[0, 1].set_ylabel("V_global")
            axes[0, 1].set_title("Lyapunov V(t)")

            axes[1, 0].plot(lam_hist, linewidth=0.8, color="#72b7b2")
            axes[1, 0].axhline(0, color="#999", linestyle="--", linewidth=0.5)
            axes[1, 0].set_ylabel("λ")
            axes[1, 0].set_xlabel("tick")
            axes[1, 0].set_title("Lyapunov Exponent")

            r_arr = np.array(r_layer_hist)
            for m in range(r_arr.shape[1]):
                axes[1, 1].plot(r_arr[:, m], linewidth=0.5, alpha=0.7, label=f"L{m}")
            axes[1, 1].set_ylabel("R_layer")
            axes[1, 1].set_xlabel("tick")
            axes[1, 1].set_title("Per-Layer Coherence")
            if r_arr.shape[1] <= 8:
                axes[1, 1].legend(fontsize=6, ncol=2)

            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.line_chart({"R_global": r_hist, "V_global": v_hist})

        with st.expander("DIRECTOR_AI Export (last tick)"):
            st.json(final["director_ai"])

# ─── Benchmark Plots (Vega-Lite interactive) ───
with tab_vega:
    st.header("Interactive Benchmark Plots — Paper 27")

    vega_path = Path(__file__).resolve().parent.parent / "docs" / "bench_interactive.vl.json"
    if vega_path.exists():
        spec = json.loads(vega_path.read_text(encoding="utf-8"))
        st.vega_lite_chart(spec, use_container_width=True)
        st.caption("Click legend entries to filter series. Data from Criterion + Python benchmarks.")
    else:
        st.warning(f"Vega-Lite spec not found: {vega_path}")

    st.subheader("Individual Benchmark Charts")
    for vl_name in ["bench_lyapunov_vs_zeta.vl.json", "bench_pac_vs_nopac.vl.json"]:
        vl_path = Path(__file__).resolve().parent.parent / "docs" / vl_name
        if vl_path.exists():
            vl_spec = json.loads(vl_path.read_text(encoding="utf-8"))
            st.vega_lite_chart(vl_spec, use_container_width=True)

# ─── RMSE Dashboard ───
with tab_rmse:
    st.header("RMSE Validation")
    st.info("Run `scpn-control validate` from CLI for full results.")

# ─── Timing Benchmark ───
with tab_bench:
    st.header("Controller Latency Benchmark")
    n_bench = st.number_input("Iterations", 100, 50000, 5000, step=1000)
    if st.button("Run Benchmark", key="run_bench"):
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
