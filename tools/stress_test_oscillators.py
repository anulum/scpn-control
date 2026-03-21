import time
import numpy as np
import scpn_control_rs as rs
from scpn_control.phase.plasma_knm import plasma_omega


def stress_test():
    print("SCPN-CONTROL: Oscillator Capacity Stress Test (16 Layers)")
    print("-" * 60)
    print(f"{'Osc/Layer':>10} | {'Total Osc':>10} | {'Latency (ms)':>15} | {'Max Freq (Hz)':>15}")
    print("-" * 60)

    L = 16
    # Sweep N_per from 10 to 65536
    # Using powers of 2 for clean scaling
    n_per_sweep = [10, 50, 100, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    omega_base = plasma_omega(L)
    dt = 0.001
    K_flat = np.random.uniform(0, 0.5, L * L).astype(np.float64)
    alpha_flat = np.zeros(L * L, dtype=np.float64)
    zeta = np.ones(L, dtype=np.float64) * 0.5
    Psi_global = 0.3
    pac_gamma = 1.0

    results = []

    for n_per in n_per_sweep:
        total_n = L * n_per

        # Initialize phases and frequencies
        theta_flat = np.random.uniform(0, 2 * np.pi, total_n).astype(np.float64)
        omega_flat = np.concatenate([np.full(n_per, omega_base[m]) for m in range(L)]).astype(np.float64)

        # Warmup
        for _ in range(5):
            _ = rs.upde_tick(theta_flat, omega_flat, K_flat, alpha_flat, zeta, L, n_per, dt, Psi_global, pac_gamma)

        # Benchmark
        n_iters = 50 if n_per < 8192 else 10
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = rs.upde_tick(theta_flat, omega_flat, K_flat, alpha_flat, zeta, L, n_per, dt, Psi_global, pac_gamma)
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / n_iters * 1000
        max_hz = 1000 / avg_ms

        print(f"{n_per:10d} | {total_n:10d} | {avg_ms:15.4f} | {max_hz:15.1f}")
        results.append((total_n, avg_ms))

        # Stop if we drop below 10 Hz (too slow for "real-time" demo)
        if max_hz < 10:
            print("-" * 60)
            print(f"Capacity limit reached at ~{total_n} oscillators.")
            break
    else:
        print("-" * 60)
        print("Completed full sweep.")


if __name__ == "__main__":
    stress_test()
