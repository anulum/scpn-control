import time
import numpy as np
import scpn_control_rs as rs


def main():
    print("SCPN-CONTROL: Activated Feature Verification")
    print("-" * 40)

    # 1. Thomas Solver
    a = np.array([0.0] + [-1.0] * 9)
    b = np.array([2.0] * 10)
    c = np.array([-1.0] * 9 + [0.0])
    d = np.zeros(10)
    d[5] = 1.0

    t0 = time.perf_counter()
    x = rs.py_thomas_solve(a, b, c, d)
    t1 = time.perf_counter()
    print(f"Thomas Solver: {(t1 - t0) * 1e6:.2f} us (size 10)")

    # 2. X-Point search
    grid_r = np.linspace(1.0, 9.0, 129)
    grid_z = np.linspace(-5.0, 5.0, 129)
    psi = np.zeros((129, 129))
    # Gaussian at (5.0, -3.0)
    RR, ZZ = np.meshgrid(grid_r, grid_z)
    psi = (RR - 5.0) ** 2 + (ZZ + 3.0) ** 2

    t0 = time.perf_counter()
    fk_tmp = rs.PyFusionKernel("iter_config.json")
    (rx, zx), psix = fk_tmp.find_x_point(-1.0)
    t1 = time.perf_counter()
    # Wait, the find_x_point is a method of PyFusionKernel or standalone?
    # I added it to PyFusionKernel.

    # 3. B-Field
    fk = rs.PyFusionKernel("iter_config.json")
    t0 = time.perf_counter()
    br, bz = fk.compute_b_field()
    t1 = time.perf_counter()
    print(f"B-Field Compute (129x129): {(t1 - t0) * 1e6:.2f} us")

    # 4. AMR Solve (New!)
    amr = rs.PyAmrSolver(max_levels=2, coarse_iters=100)
    source = np.zeros((129, 129))
    source[64, 64] = -1.0

    t0 = time.perf_counter()
    psi_amr = amr.solve_with_hierarchy(psi, grid_r, grid_z)
    t1 = time.perf_counter()
    print(f"AMR Equilibrium Solve: {(t1 - t0):.4f} s")


if __name__ == "__main__":
    main()
