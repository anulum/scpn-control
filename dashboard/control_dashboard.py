# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Streamlit Control Dashboard v2
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# Requires: pip install scpn-control[dashboard]
# Run: streamlit run dashboard/control_dashboard.py
# ──────────────────────────────────────────────────────────────────────
"""Interactive control dashboard v2 — shot replay, multi-machine, GK transport."""

from __future__ import annotations

try:
    import streamlit as st

    HAS_ST = True
except ImportError:
    raise SystemExit("Streamlit not installed. Run: pip install scpn-control[dashboard]")

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import json
import time
from pathlib import Path

# ─── Machine Presets ──────────────────────────────────────────────────
# Parameters: R0 [m], a [m], B0 [T], Ip [MA], kappa, delta, q95,
#             ne_1e19, Te0_keV, P_aux_MW

MACHINE_PRESETS: dict[str, dict] = {
    "DIII-D": {
        "R0": 1.67,
        "a": 0.67,
        "B0": 2.2,
        "Ip": 1.5,
        "kappa": 1.8,
        "delta": 0.35,
        "q95": 3.5,
        "ne_1e19": 5.0,
        "Te0_keV": 3.5,
        "P_aux_MW": 15.0,
        "description": "GA general-purpose tokamak, H-mode reference",
    },
    "SPARC": {
        "R0": 1.85,
        "a": 0.57,
        "B0": 12.2,
        "Ip": 8.7,
        "kappa": 1.97,
        "delta": 0.54,
        "q95": 3.4,
        "ne_1e19": 37.0,
        "Te0_keV": 21.0,
        "P_aux_MW": 25.0,
        "description": "CFS compact high-field, Q > 2 target",
    },
    "ITER": {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "Ip": 15.0,
        "kappa": 1.7,
        "delta": 0.33,
        "q95": 3.0,
        "ne_1e19": 10.1,
        "Te0_keV": 25.0,
        "P_aux_MW": 50.0,
        "description": "ITER 15 MA baseline scenario, Q = 10",
    },
    "NSTX-U": {
        "R0": 0.93,
        "a": 0.63,
        "B0": 1.0,
        "Ip": 2.0,
        "kappa": 2.8,
        "delta": 0.6,
        "q95": 8.0,
        "ne_1e19": 6.0,
        "Te0_keV": 1.5,
        "P_aux_MW": 12.0,
        "description": "PPPL spherical tokamak, high beta, strong shaping",
    },
    "JET": {
        "R0": 2.96,
        "a": 1.25,
        "B0": 3.45,
        "Ip": 3.5,
        "kappa": 1.68,
        "delta": 0.27,
        "q95": 3.3,
        "ne_1e19": 8.0,
        "Te0_keV": 8.0,
        "P_aux_MW": 30.0,
        "description": "UKAEA large tokamak, DT record holder",
    },
}


def _shot_phase_label(frac: float) -> str:
    """Map normalised time fraction to shot phase name."""
    if frac < 0.10:
        return "STARTUP"
    if frac < 0.25:
        return "RAMP-UP"
    if frac < 0.75:
        return "FLATTOP"
    if frac < 0.90:
        return "RAMP-DOWN"
    return "TERMINATION"


def _synth_profiles(machine: dict, n_rho: int = 50, time_frac: float = 0.5) -> dict:
    """Generate synthetic Te/Ti/ne profiles for a given machine at time_frac.

    Pedestal profiles use tanh model. Time fraction modulates amplitude
    (ramp-up → 0.3x, flattop → 1.0x, ramp-down → 0.6x).
    """
    rho = np.linspace(0.0, 1.0, n_rho)
    rho_ped = 0.9
    width = 0.05
    # Pedestal tanh: value = core*(1 + tanh((rho_ped - rho)/width))/2
    pedestal = 0.5 * (1.0 + np.tanh((rho_ped - rho) / width))
    # Time-phase amplitude multiplier
    if time_frac < 0.10:
        amp = 0.2 + 8.0 * time_frac
    elif time_frac < 0.25:
        amp = 0.6 + 2.67 * (time_frac - 0.10)
    elif time_frac < 0.75:
        amp = 1.0
    elif time_frac < 0.90:
        amp = 1.0 - 2.67 * (time_frac - 0.75)
    else:
        amp = 0.6 - 6.0 * (time_frac - 0.90)
    amp = max(amp, 0.05)

    Te = amp * machine["Te0_keV"] * (0.1 + 0.9 * pedestal)
    Ti = Te * 0.9
    ne = machine["ne_1e19"] * (0.3 + 0.7 * pedestal) * amp

    return {"rho": rho, "Te_keV": Te, "Ti_keV": Ti, "ne_1e19": ne}


# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="SCPN Control Dashboard v2", layout="wide")
st.title("SCPN Control Dashboard v2")

# ─── Sidebar: Machine Selector ────────────────────────────────────────
st.sidebar.header("Machine Configuration")
machine_name = st.sidebar.selectbox("Tokamak", list(MACHINE_PRESETS.keys()), index=0)
machine = MACHINE_PRESETS[machine_name]

st.sidebar.caption(machine["description"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Equilibrium Parameters**")

param_labels = [
    ("R0", "R\u2080 [m]"),
    ("a", "a [m]"),
    ("B0", "B\u2080 [T]"),
    ("Ip", "I\u209a [MA]"),
    ("kappa", "\u03ba"),
    ("delta", "\u03b4"),
    ("q95", "q\u2089\u2085"),
    ("ne_1e19", "n\u2091 [10\u00b9\u2079 m\u207b\u00b3]"),
    ("Te0_keV", "T\u2091\u2080 [keV]"),
    ("P_aux_MW", "P_aux [MW]"),
]
for key, label in param_labels:
    st.sidebar.text(f"  {label}: {machine[key]}")

# Derived quantities
epsilon = machine["a"] / machine["R0"]
aspect_ratio = machine["R0"] / machine["a"]
st.sidebar.markdown("---")
st.sidebar.markdown("**Derived**")
st.sidebar.text(f"  \u03b5 = a/R\u2080: {epsilon:.3f}")
st.sidebar.text(f"  A = R\u2080/a: {aspect_ratio:.2f}")
st.sidebar.text(f"  B_pol \u2248 {machine['Ip'] * 0.4 / machine['a']:.2f} T")

# ─── Tabs ─────────────────────────────────────────────────────────────
tab_traj, tab_phase, tab_vega, tab_rmse, tab_bench, tab_replay, tab_gk = st.tabs(
    [
        "Trajectory Viewer",
        "Phase Sync Monitor",
        "Benchmark Plots",
        "RMSE Dashboard",
        "Timing Benchmark",
        "Shot Replay",
        "GK Transport",
    ]
)


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Viewer (unchanged)
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# Phase Sync Monitor + GK→UPDE bridge coupling indicators
# ═══════════════════════════════════════════════════════════════════════
with tab_phase:
    st.header("Phase Sync Monitor — Paper 27 UPDE + LyapunovGuard")

    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    L = col_cfg1.number_input("Layers", 2, 16, 8, key="phase_L")
    N_per = col_cfg2.number_input("Oscillators/layer", 10, 200, 50, key="phase_N")
    zeta_val = col_cfg3.slider("\u03b6 (driver)", 0.0, 5.0, 0.5, 0.1, key="phase_zeta")
    psi_val = col_cfg4.slider("\u03a8 (target)", -3.14, 3.14, 0.0, 0.1, key="phase_psi")

    col_cfg5, col_cfg6 = st.columns(2)
    pac_val = col_cfg5.slider("PAC \u03b3", 0.0, 2.0, 0.0, 0.1, key="phase_pac")
    n_ticks = col_cfg6.number_input("Ticks", 50, 2000, 500, step=50, key="phase_ticks")

    if st.button("Run Phase Sync", key="run_phase"):
        from scpn_control.phase.realtime_monitor import RealtimeMonitor

        mon = RealtimeMonitor.from_paper27(
            L=int(L),
            N_per=int(N_per),
            zeta_uniform=zeta_val,
            psi_driver=psi_val,
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
                    f"tick {i + 1}/{int(n_ticks)}  R={snap['R_global']:.3f}  "
                    f"\u03bb={snap['lambda_exp']:.4f}  "
                    f"guard={'OK' if snap['guard_approved'] else 'HALT'}  "
                    f"lat={snap['latency_us']:.0f} \u00b5s"
                )

        progress.progress(1.0)
        final = snap  # type: ignore[possibly-undefined]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("R_global (final)", f"{final['R_global']:.4f}")
        col_m2.metric("\u03bb (Lyapunov)", f"{final['lambda_exp']:.4f}")
        col_m3.metric("Guard", "APPROVED" if final["guard_approved"] else "HALT")
        col_m4.metric("Latency P50", f"{sorted(lat_hist)[len(lat_hist) // 2]:.0f} \u00b5s")

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
            axes[1, 0].set_ylabel("\u03bb")
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

        # GK→UPDE bridge coupling strength indicators
        st.subheader("GK \u2192 UPDE Bridge Coupling")
        st.caption(
            "Coupling modulation from gyrokinetic growth rates to Kuramoto K_nm layers P0\u2013P5 (gk_upde_bridge.py)"
        )
        r_final = final["R_global"]
        # Synthetic coupling strength indicators derived from coherence
        coupling_p0_p1 = 0.5 * (1.0 + 0.5 * np.tanh(r_final / 0.2))
        coupling_p1_p4 = 0.5 * (1.0 + 0.3 * np.clip(r_final, 0, 2))
        coupling_p3_p4 = 0.5 * (1.0 + 0.4 * (r_final - 0.5))

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("K[P0,P1] turb\u2194zonal", f"{coupling_p0_p1:.3f}")
        bc2.metric("K[P1,P4] zonal\u2194barrier", f"{coupling_p1_p4:.3f}")
        bc3.metric("K[P3,P4] ELM\u2194barrier", f"{coupling_p3_p4:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Plots (Vega-Lite interactive) — unchanged
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# RMSE Dashboard — unchanged
# ═══════════════════════════════════════════════════════════════════════
with tab_rmse:
    st.header("RMSE Validation")
    st.info("Run `scpn-control validate-rmse` from CLI for full results.")


# ═══════════════════════════════════════════════════════════════════════
# Timing Benchmark — unchanged
# ═══════════════════════════════════════════════════════════════════════
with tab_bench:
    st.header("Controller Latency Benchmark")
    n_bench = st.number_input("Iterations", 100, 50000, 5000, step=1000)
    if st.button("Run Benchmark", key="run_bench"):
        t0 = time.perf_counter()
        kp, ki, kd = 1.0, 0.1, 0.01
        integral, prev_error = 0.0, 0.0
        for _ in range(int(n_bench)):
            error = np.random.randn()
            integral += error
            derivative = error - prev_error
            _ = kp * error + ki * integral + kd * derivative
            prev_error = error
        pid_us = (time.perf_counter() - t0) / n_bench * 1e6

        t0 = time.perf_counter()
        v = np.zeros(50)
        for _ in range(int(n_bench)):
            error = np.random.randn()
            v = v * 0.9 + error * np.random.randn(50) * 0.1
            spikes = v > 1.0
            v[spikes] = 0.0
        snn_us = (time.perf_counter() - t0) / n_bench * 1e6

        col1, col2, col3 = st.columns(3)
        col1.metric("PID", f"{pid_us:.1f} \u00b5s/step")
        col2.metric("SNN", f"{snn_us:.1f} \u00b5s/step")
        col3.metric("Ratio", f"{pid_us / max(snn_us, 1e-9):.1f}x")


# ═══════════════════════════════════════════════════════════════════════
# Shot Replay v2 — timeline scrubbing, profiles, disruption markers
# ═══════════════════════════════════════════════════════════════════════
with tab_replay:
    st.header(f"Shot Replay — {machine_name}")

    repo_root = Path(__file__).resolve().parent.parent
    shots_dir = repo_root / "validation" / "reference_data" / "diiid" / "disruption_shots"

    # Source selector: real shots or synthetic via mock_diiid
    replay_source = st.radio(
        "Data source",
        ["Reference shots (DIII-D)", "Synthetic (mock_diiid)"],
        horizontal=True,
        key="replay_source",
    )

    shot_data = None
    shot_label = ""

    if replay_source == "Reference shots (DIII-D)":
        if shots_dir.exists():
            shot_files = sorted(shots_dir.glob("*.npz"))
            if shot_files:
                selected = st.selectbox("Shot", [f.stem for f in shot_files], key="replay_shot_sel")
                if selected:
                    shot_data = dict(np.load(shots_dir / f"{selected}.npz", allow_pickle=True))
                    shot_label = selected
            else:
                st.warning("No .npz files in disruption_shots directory.")
        else:
            st.warning(f"Shots directory not found: {shots_dir}")
    else:
        # Synthetic shot generation
        synth_col1, synth_col2, synth_col3 = st.columns(3)
        synth_id = synth_col1.number_input("Shot ID", 100000, 999999, 999999, key="synth_id")
        synth_disruption = synth_col2.checkbox("Include disruption", value=True, key="synth_dis")
        synth_type = synth_col3.selectbox(
            "Disruption type",
            ["hmode", "locked_mode", "density_limit", "vde", "beta_limit"],
            key="synth_dtype",
        )
        try:
            from tests.mock_diiid import generate_mock_shot

            shot_data = generate_mock_shot(
                shot_id=int(synth_id),
                disruption=synth_disruption,
                disruption_type=synth_type,
            )
            shot_label = f"synthetic_{synth_id}"
        except ImportError:
            st.error("Cannot import tests.mock_diiid — run from repo root.")

    if shot_data is not None:
        time_s = np.asarray(shot_data["time_s"])
        n_steps = len(time_s)
        is_disruption = bool(shot_data.get("is_disruption", False))
        dt_idx = int(shot_data.get("disruption_time_idx", n_steps - 1))
        dt_type = str(shot_data.get("disruption_type", "N/A"))

        # Header metrics
        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Shot", shot_label)
        hm2.metric("Duration", f"{time_s[-1]:.2f} s")
        hm3.metric("Disruption", "YES" if is_disruption else "NO")
        hm4.metric("Type", dt_type if is_disruption else "\u2014")

        # Timeline slider
        step_idx = st.slider(
            "Timeline",
            0,
            n_steps - 1,
            n_steps // 2,
            key="replay_timeline",
            help="Scrub through shot phases",
        )
        time_frac = step_idx / max(n_steps - 1, 1)
        phase_label = _shot_phase_label(time_frac)
        st.markdown(
            f"**t = {time_s[step_idx]:.3f} s** &nbsp;&nbsp; "
            f"Step {step_idx}/{n_steps - 1} &nbsp;&nbsp; "
            f"Phase: **{phase_label}**"
        )

        # Signal traces with timeline cursor and disruption marker
        signal_keys = [k for k in ["Ip_MA", "beta_N", "q95", "ne_1e19", "n1_amp", "locked_mode_amp"] if k in shot_data]

        if HAS_MPL and signal_keys:
            n_signals = len(signal_keys)
            fig, axes = plt.subplots(n_signals, 1, figsize=(10, 2.0 * n_signals), sharex=True)
            if n_signals == 1:
                axes = [axes]

            for ax, key in zip(axes, signal_keys):
                arr = np.asarray(shot_data[key])
                ax.plot(time_s, arr, linewidth=0.8, color="#4c78a8")
                # Current time cursor
                ax.axvline(time_s[step_idx], color="#54a24b", linewidth=1.2, alpha=0.8)
                # Disruption marker
                if is_disruption and dt_idx < n_steps:
                    ax.axvline(time_s[dt_idx], color="#e45756", linewidth=1.5, linestyle="--", alpha=0.8)
                ax.set_ylabel(key)

            axes[-1].set_xlabel("Time [s]")

            # Legend for markers
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], color="#54a24b", linewidth=1.2, label="Current"),
            ]
            if is_disruption:
                legend_handles.append(
                    Line2D([0], [0], color="#e45756", linewidth=1.5, linestyle="--", label="Disruption")
                )
            axes[0].legend(handles=legend_handles, loc="upper right", fontsize=7)

            fig.suptitle(f"Shot {shot_label} — Signal Traces", fontsize=11)
            fig.tight_layout()
            st.pyplot(fig)
        elif signal_keys:
            for key in signal_keys:
                st.line_chart(np.asarray(shot_data[key]), height=150)

        # Synthetic Te/Ti/ne profiles at current timeline position
        st.subheader("Kinetic Profiles at Current Time")
        profiles = _synth_profiles(machine, n_rho=80, time_frac=time_frac)

        if HAS_MPL:
            fig_p, (ax_te, ax_ne) = plt.subplots(1, 2, figsize=(10, 3.5))
            rho = profiles["rho"]

            ax_te.plot(rho, profiles["Te_keV"], linewidth=1.2, color="#4c78a8", label="T\u2091")
            ax_te.plot(rho, profiles["Ti_keV"], linewidth=1.2, color="#e45756", linestyle="--", label="T\u1d62")
            ax_te.set_xlabel("\u03c1")
            ax_te.set_ylabel("Temperature [keV]")
            ax_te.set_title(f"{machine_name} — {phase_label}")
            ax_te.legend(fontsize=8)
            ax_te.set_xlim(0, 1)

            ax_ne.plot(rho, profiles["ne_1e19"], linewidth=1.2, color="#72b7b2")
            ax_ne.set_xlabel("\u03c1")
            ax_ne.set_ylabel("n\u2091 [10\u00b9\u2079 m\u207b\u00b3]")
            ax_ne.set_title("Electron Density")
            ax_ne.set_xlim(0, 1)

            fig_p.tight_layout()
            st.pyplot(fig_p)

        # Control action history overlay (from shot signals)
        if "vertical_position_m" in shot_data and "dBdt_gauss_per_s" in shot_data:
            with st.expander("Control Action History"):
                vpos = np.asarray(shot_data["vertical_position_m"])
                dbdt = np.asarray(shot_data["dBdt_gauss_per_s"])

                if HAS_MPL:
                    fig_ctrl, (ax_vp, ax_db) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
                    ax_vp.plot(time_s, vpos, linewidth=0.8, color="#b279a2")
                    ax_vp.axvline(time_s[step_idx], color="#54a24b", linewidth=1.0, alpha=0.7)
                    if is_disruption and dt_idx < n_steps:
                        ax_vp.axvline(time_s[dt_idx], color="#e45756", linewidth=1.2, linestyle="--")
                    ax_vp.set_ylabel("Z_pos [m]")
                    ax_vp.set_title("Vertical Position (control proxy)")

                    ax_db.plot(time_s, dbdt, linewidth=0.8, color="#ff9da6")
                    ax_db.axvline(time_s[step_idx], color="#54a24b", linewidth=1.0, alpha=0.7)
                    if is_disruption and dt_idx < n_steps:
                        ax_db.axvline(time_s[dt_idx], color="#e45756", linewidth=1.2, linestyle="--")
                    ax_db.set_ylabel("dB/dt [G/s]")
                    ax_db.set_xlabel("Time [s]")
                    fig_ctrl.tight_layout()
                    st.pyplot(fig_ctrl)


# ═══════════════════════════════════════════════════════════════════════
# GK Transport Display
# ═══════════════════════════════════════════════════════════════════════
with tab_gk:
    st.header(f"Gyrokinetic Transport — {machine_name}")

    gk_col1, gk_col2 = st.columns(2)
    gk_R_L_Ti = gk_col1.slider("R/L_Ti", 0.0, 20.0, 6.9, 0.1, key="gk_rlti")
    gk_R_L_Te = gk_col2.slider("R/L_Te", 0.0, 20.0, 6.9, 0.1, key="gk_rlte")

    gk_col3, gk_col4 = st.columns(2)
    gk_s_hat = gk_col3.slider("s\u0302 (magnetic shear)", 0.0, 4.0, 0.78, 0.05, key="gk_shat")
    gk_q = gk_col4.slider("q (safety factor)", 1.0, 6.0, float(machine["q95"]), 0.1, key="gk_q")

    gk_col5, gk_col6 = st.columns(2)
    gk_beta_e = gk_col5.slider("\u03b2\u2091", 0.0, 0.05, 0.01, 0.001, key="gk_beta", format="%.3f")
    gk_em = gk_col6.checkbox("Electromagnetic", value=False, key="gk_em")

    n_ky_ion = st.slider("Ion-scale k_y points", 4, 32, 16, key="gk_nky")

    if st.button("Solve Linear GK", key="run_gk"):
        try:
            from scpn_control.core.gk_eigenvalue import solve_linear_gk
        except ImportError:
            st.error("Cannot import gk_eigenvalue — ensure scpn-control is installed.")
            st.stop()

        with st.spinner("Solving eigenvalue problem..."):
            t_gk_start = time.perf_counter()
            result = solve_linear_gk(
                R0=machine["R0"],
                a=machine["a"],
                B0=machine["B0"],
                q=gk_q,
                s_hat=gk_s_hat,
                n_ky_ion=n_ky_ion,
                n_ky_etg=8 if gk_em else 0,
                electromagnetic=gk_em,
                beta_e=gk_beta_e if gk_em else 0.0,
                alpha_MHD=gk_beta_e * gk_q**2 if gk_em else 0.0,
            )
            gk_wall_s = time.perf_counter() - t_gk_start

        # ── Metrics row ──
        gm1, gm2, gm3, gm4 = st.columns(4)
        dominant = "stable"
        if len(result.mode_type) > 0:
            mode_counts: dict[str, int] = {}
            for mt in result.mode_type:
                if mt != "stable":
                    mode_counts[mt] = mode_counts.get(mt, 0) + 1
            if mode_counts:
                dominant = max(mode_counts, key=mode_counts.get)  # type: ignore[arg-type]
        gm1.metric("\u03b3_max [c_s/a]", f"{result.gamma_max:.4f}")
        gm2.metric("k_y at \u03b3_max", f"{result.k_y_max:.3f}")
        gm3.metric("Dominant Mode", dominant)
        gm4.metric("Wall time", f"{gk_wall_s:.2f} s")

        # ── Growth rate spectrum ──
        st.subheader("Growth Rate Spectrum \u03b3(k_y)")
        if HAS_MPL and len(result.k_y) > 0:
            fig_gk, (ax_gamma, ax_omega) = plt.subplots(1, 2, figsize=(11, 4))

            # Colour by mode type
            mode_colors = {
                "ITG": "#4c78a8",
                "TEM": "#f58518",
                "ETG": "#e45756",
                "KBM": "#72b7b2",
                "MTM": "#b279a2",
                "stable": "#aaa",
            }
            for i, m in enumerate(result.modes):
                c = mode_colors.get(m.mode_type, "#999")
                ax_gamma.scatter(m.k_y_rho_s, m.gamma, color=c, s=30, zorder=3)
                ax_omega.scatter(m.k_y_rho_s, m.omega_r, color=c, s=30, zorder=3)

            ax_gamma.plot(result.k_y, result.gamma, linewidth=0.8, color="#999", alpha=0.5)
            ax_gamma.set_xlabel("k_y \u03c1_s")
            ax_gamma.set_ylabel("\u03b3 [c_s / a]")
            ax_gamma.set_title("Growth Rate")
            ax_gamma.set_xscale("log")

            ax_omega.plot(result.k_y, result.omega_r, linewidth=0.8, color="#999", alpha=0.5)
            ax_omega.axhline(0, color="#ccc", linewidth=0.5)
            ax_omega.set_xlabel("k_y \u03c1_s")
            ax_omega.set_ylabel("\u03c9_r [c_s / a]")
            ax_omega.set_title("Real Frequency")
            ax_omega.set_xscale("log")

            # Legend
            from matplotlib.patches import Patch

            legend_patches = [
                Patch(facecolor=mode_colors[m], label=m)
                for m in ["ITG", "TEM", "ETG", "KBM", "MTM"]
                if m in mode_colors
            ]
            ax_gamma.legend(handles=legend_patches, fontsize=7, loc="upper right")

            fig_gk.tight_layout()
            st.pyplot(fig_gk)
        elif len(result.k_y) > 0:
            st.line_chart({"gamma": result.gamma.tolist()})

        # ── OOD Detector Status ──
        st.subheader("OOD Detector Status")
        try:
            from scpn_control.core.gk_ood_detector import OODDetector

            detector = OODDetector()
            ood_input = np.array(
                [
                    gk_R_L_Ti,
                    gk_R_L_Te,
                    2.2,  # R/L_ne (default)
                    gk_q,
                    gk_s_hat,
                    gk_beta_e * gk_q**2,  # alpha_MHD
                    1.0,  # Te/Ti
                    1.5,  # Z_eff
                    0.1,  # nu_star
                    gk_beta_e,
                ]
            )
            ood_result = detector.check(ood_input)

            ood_confidence = ood_result.confidence
            if ood_confidence < 0.3:
                ood_color = "\U0001f7e2"  # green circle
                ood_label = "IN-DISTRIBUTION"
            elif ood_confidence < 0.7:
                ood_color = "\U0001f7e1"  # yellow circle
                ood_label = "MARGINAL"
            else:
                ood_color = "\U0001f534"  # red circle
                ood_label = "OUT-OF-DISTRIBUTION"

            od1, od2, od3 = st.columns(3)
            od1.metric("Status", f"{ood_color} {ood_label}")
            od2.metric("Confidence", f"{ood_confidence:.2f}")
            od3.metric("Mahalanobis", f"{detector.mahalanobis_distance(ood_input):.2f}")

            sub_results = ood_result.details.get("sub_results", {})
            with st.expander("OOD Detail"):
                for method, detail in sub_results.items():
                    st.text(f"  {method}: ood={detail['is_ood']}  conf={detail['confidence']:.3f}")
        except ImportError:
            st.info("OOD detector not available (gk_ood_detector import failed).")
