// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Solver.

#ifndef SCPN_CONTROL_SOLVER_H
#define SCPN_CONTROL_SOLVER_H

#if defined(_WIN32)
#if defined(SCPN_SOLVER_BUILD)
#define SCPN_SOLVER_API __declspec(dllexport)
#else
#define SCPN_SOLVER_API __declspec(dllimport)
#endif
#else
#define SCPN_SOLVER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SCPN_SOLVER_ABI_VERSION 1

/** Opaque solver allocation owned by the library. */
typedef struct scpn_solver_v1 scpn_solver_v1;

/** Machine-readable outcome of a version 1 ABI call. */
typedef enum scpn_solver_status_v1 {
    SCPN_SOLVER_STATUS_OK = 0,
    SCPN_SOLVER_STATUS_INVALID_ARGUMENT = 1,
    SCPN_SOLVER_STATUS_INVALID_HANDLE = 2,
    SCPN_SOLVER_STATUS_INVALID_DIMENSIONS = 3,
    SCPN_SOLVER_STATUS_NONFINITE_INPUT = 4,
    SCPN_SOLVER_STATUS_SIZE_MISMATCH = 5,
    SCPN_SOLVER_STATUS_ALLOCATION_FAILED = 6,
    SCPN_SOLVER_STATUS_INTERNAL_ERROR = 7
} scpn_solver_status_v1;

/**
 * Allocate a fixed-boundary Grad-Shafranov SOR state.
 *
 * `nr` and `nz` are point counts and must each be at least 3. Radial and
 * vertical bounds are finite metres with `r_max > r_min` and
 * `z_max > z_min`. On success, `*solver_out` becomes a library-owned opaque
 * handle whose initial poloidal-flux state and boundary are zero. The caller
 * owns that handle and must pass it exactly once to `scpn_solver_destroy_v1`.
 * On failure, a non-null `solver_out` is set to null. This operation is
 * deterministic apart from allocation success and does not retain pointers
 * supplied by the caller.
 */
SCPN_SOLVER_API scpn_solver_status_v1 scpn_solver_create_v1(
    int nr,
    int nz,
    double r_min,
    double r_max,
    double z_min,
    double z_max,
    scpn_solver_v1** solver_out
);

/**
 * Set the Dirichlet value on every edge point of `solver`.
 *
 * `boundary_value` is finite poloidal flux in the same convention and units
 * as the solution array (Wb/rad in this kernel). The call mutates only the
 * referenced handle. Concurrent calls on the same handle are unsupported;
 * distinct handles share no mutable state.
 */
SCPN_SOLVER_API scpn_solver_status_v1 scpn_solver_set_boundary_dirichlet_v1(
    scpn_solver_v1* solver,
    double boundary_value
);

/**
 * Execute a fixed number of red-black SOR sweeps with omega = 1.5.
 *
 * `source` and `psi_out` each address exactly `size == nz * nr` doubles in
 * C row-major `[nz][nr]` order. `source` is read-only, finite, and already
 * scaled as the elliptic right-hand side in solution-units per square metre;
 * it is not interpreted as raw current density. `psi_out` is caller-owned and
 * is overwritten with the complete current solution. The arrays must not
 * overlap. `iterations` must be non-negative. Results are deterministic for
 * the same ABI, compiler arithmetic, inputs, and initial handle state.
 */
SCPN_SOLVER_API scpn_solver_status_v1 scpn_solver_run_steps_v1(
    scpn_solver_v1* solver,
    const double* source,
    double* psi_out,
    int size,
    int iterations
);

/**
 * Execute red-black SOR sweeps until tolerance or the iteration cap.
 *
 * Array layout, units, ownership, non-aliasing, determinism, and thread-safety
 * match `scpn_solver_run_steps_v1`. `max_iterations` must be positive,
 * `omega` finite and in `(0, 2)`, and `tolerance` finite and non-negative in
 * solution units. On `OK`, all three outputs are written: `iterations_used`
 * is in `[1, max_iterations]`, `final_delta` is the maximum absolute update
 * from the final sweep, and `converged` is 1 exactly when
 * `final_delta <= tolerance` (otherwise 0). Scientific non-convergence is an
 * `OK` call outcome, not an ABI error.
 */
SCPN_SOLVER_API scpn_solver_status_v1 scpn_solver_run_until_converged_v1(
    scpn_solver_v1* solver,
    const double* source,
    double* psi_out,
    int size,
    int max_iterations,
    double omega,
    double tolerance,
    int* iterations_used,
    double* final_delta,
    int* converged
);

/**
 * Release a handle returned by `scpn_solver_create_v1`.
 *
 * Passing null is a successful no-op. Any non-null handle becomes invalid as
 * soon as this function is called; subsequent use or a second destroy is
 * undefined caller behaviour. The function does not throw across the C ABI.
 */
SCPN_SOLVER_API scpn_solver_status_v1 scpn_solver_destroy_v1(scpn_solver_v1* solver);

/*
 * Legacy ABI compatibility surface (unversioned, retained for existing
 * clients). Invalid input is reported only through null/zero/silent return;
 * new integrations must use the typed version 1 functions above.
 */
SCPN_SOLVER_API void* create_solver(
    int nr, int nz, double r_min, double r_max, double z_min, double z_max
);
SCPN_SOLVER_API void set_boundary_dirichlet(void* solver, double boundary_value);
SCPN_SOLVER_API void run_step(
    void* solver, double* source, double* psi_out, int size, int iterations
);
SCPN_SOLVER_API int run_step_converged(
    void* solver,
    double* source,
    double* psi_out,
    int size,
    int max_iterations,
    double omega,
    double tolerance,
    double* final_delta
);
SCPN_SOLVER_API void destroy_solver(void* solver);

#ifdef __cplusplus
}
#endif

#endif
