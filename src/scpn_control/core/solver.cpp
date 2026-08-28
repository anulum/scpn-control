// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Solver.

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

struct scpn_solver_v1 {
    int nr;
    int nz;
    double r_min;
    double r_max;
    double z_min;
    double z_max;
    double dr;
    double dz;
    double boundary;
    std::vector<double> psi;
};

namespace {

bool valid_dimensions(int nr, int nz, double r_min, double r_max, double z_min, double z_max) {
    return nr >= 3 && nz >= 3 && std::isfinite(r_min) && std::isfinite(r_max) &&
           std::isfinite(z_min) && std::isfinite(z_max) && r_max > r_min && z_max > z_min &&
           nr <= std::numeric_limits<int>::max() / nz;
}

std::size_t offset(const scpn_solver_v1& solver, int iz, int ir) {
    return static_cast<std::size_t>(iz) * static_cast<std::size_t>(solver.nr) +
           static_cast<std::size_t>(ir);
}

double radius_at(const scpn_solver_v1& solver, int ir) {
    return solver.r_min + solver.dr * static_cast<double>(ir);
}

void apply_boundary(scpn_solver_v1& solver) {
    for (int ir = 0; ir < solver.nr; ++ir) {
        solver.psi[offset(solver, 0, ir)] = solver.boundary;
        solver.psi[offset(solver, solver.nz - 1, ir)] = solver.boundary;
    }
    for (int iz = 0; iz < solver.nz; ++iz) {
        solver.psi[offset(solver, iz, 0)] = solver.boundary;
        solver.psi[offset(solver, iz, solver.nr - 1)] = solver.boundary;
    }
}

bool finite_input(const double* values, int size) {
    if (values == nullptr || size <= 0) {
        return false;
    }
    for (int index = 0; index < size; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

double update_point(scpn_solver_v1& solver, const double* source, int iz, int ir, double omega) {
    const double r = std::max(radius_at(solver, ir), std::numeric_limits<double>::epsilon());
    const double dr_sq = solver.dr * solver.dr;
    const double dz_sq = solver.dz * solver.dz;
    const double c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * solver.dr);
    const double c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * solver.dr);
    const double c_z = 1.0 / dz_sq;
    const double center = 2.0 / dr_sq + 2.0 / dz_sq;
    const std::size_t idx = offset(solver, iz, ir);
    const double old_value = solver.psi[idx];
    const double prediction =
        (source[idx] +
         c_z * (solver.psi[offset(solver, iz + 1, ir)] + solver.psi[offset(solver, iz - 1, ir)]) +
         c_r_plus * solver.psi[offset(solver, iz, ir + 1)] +
         c_r_minus * solver.psi[offset(solver, iz, ir - 1)]) /
        center;
    solver.psi[idx] = (1.0 - omega) * old_value + omega * prediction;
    return std::abs(solver.psi[idx] - old_value);
}

double sor_sweep(scpn_solver_v1& solver, const double* source, double omega) {
    double max_delta = 0.0;
    for (int parity = 0; parity < 2; ++parity) {
        for (int iz = 1; iz < solver.nz - 1; ++iz) {
            for (int ir = 1; ir < solver.nr - 1; ++ir) {
                if (((iz + ir) & 1) == parity) {
                    max_delta = std::max(max_delta, update_point(solver, source, iz, ir, omega));
                }
            }
        }
    }
    apply_boundary(solver);
    return max_delta;
}

scpn_solver_status_v1 validate_arrays(
    scpn_solver_v1* solver,
    const double* source,
    double* psi_out,
    int size
) {
    if (solver == nullptr) {
        return SCPN_SOLVER_STATUS_INVALID_HANDLE;
    }
    if (source == nullptr || psi_out == nullptr || source == psi_out) {
        return SCPN_SOLVER_STATUS_INVALID_ARGUMENT;
    }
    if (size != solver->nr * solver->nz) {
        return SCPN_SOLVER_STATUS_SIZE_MISMATCH;
    }
    if (!finite_input(source, size)) {
        return SCPN_SOLVER_STATUS_NONFINITE_INPUT;
    }
    return SCPN_SOLVER_STATUS_OK;
}

void copy_solution(const scpn_solver_v1& solver, double* psi_out) {
    std::copy(solver.psi.begin(), solver.psi.end(), psi_out);
}

}  // namespace

extern "C" scpn_solver_status_v1 scpn_solver_create_v1(
    int nr,
    int nz,
    double r_min,
    double r_max,
    double z_min,
    double z_max,
    scpn_solver_v1** solver_out
) {
    if (solver_out == nullptr) {
        return SCPN_SOLVER_STATUS_INVALID_ARGUMENT;
    }
    *solver_out = nullptr;
    if (!valid_dimensions(nr, nz, r_min, r_max, z_min, z_max)) {
        return SCPN_SOLVER_STATUS_INVALID_DIMENSIONS;
    }
    try {
        *solver_out = new scpn_solver_v1{
            nr,
            nz,
            r_min,
            r_max,
            z_min,
            z_max,
            (r_max - r_min) / static_cast<double>(nr - 1),
            (z_max - z_min) / static_cast<double>(nz - 1),
            0.0,
            std::vector<double>(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz), 0.0),
        };
        apply_boundary(**solver_out);
        return SCPN_SOLVER_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return SCPN_SOLVER_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return SCPN_SOLVER_STATUS_INTERNAL_ERROR;
    }
}

extern "C" scpn_solver_status_v1 scpn_solver_set_boundary_dirichlet_v1(
    scpn_solver_v1* solver,
    double boundary_value
) {
    if (solver == nullptr) {
        return SCPN_SOLVER_STATUS_INVALID_HANDLE;
    }
    if (!std::isfinite(boundary_value)) {
        return SCPN_SOLVER_STATUS_NONFINITE_INPUT;
    }
    try {
        solver->boundary = boundary_value;
        apply_boundary(*solver);
        return SCPN_SOLVER_STATUS_OK;
    } catch (...) {
        return SCPN_SOLVER_STATUS_INTERNAL_ERROR;
    }
}

extern "C" scpn_solver_status_v1 scpn_solver_run_steps_v1(
    scpn_solver_v1* solver,
    const double* source,
    double* psi_out,
    int size,
    int iterations
) {
    const scpn_solver_status_v1 arrays_status = validate_arrays(solver, source, psi_out, size);
    if (arrays_status != SCPN_SOLVER_STATUS_OK) {
        return arrays_status;
    }
    if (iterations < 0) {
        return SCPN_SOLVER_STATUS_INVALID_ARGUMENT;
    }
    try {
        for (int step = 0; step < iterations; ++step) {
            sor_sweep(*solver, source, 1.5);
        }
        copy_solution(*solver, psi_out);
        return SCPN_SOLVER_STATUS_OK;
    } catch (...) {
        return SCPN_SOLVER_STATUS_INTERNAL_ERROR;
    }
}

extern "C" scpn_solver_status_v1 scpn_solver_run_until_converged_v1(
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
) {
    const scpn_solver_status_v1 arrays_status = validate_arrays(solver, source, psi_out, size);
    if (arrays_status != SCPN_SOLVER_STATUS_OK) {
        return arrays_status;
    }
    if (iterations_used == nullptr || final_delta == nullptr || converged == nullptr || max_iterations < 1 ||
        !std::isfinite(omega) || omega <= 0.0 || omega >= 2.0 || !std::isfinite(tolerance) ||
        tolerance < 0.0) {
        return SCPN_SOLVER_STATUS_INVALID_ARGUMENT;
    }
    *iterations_used = 0;
    *final_delta = std::numeric_limits<double>::quiet_NaN();
    *converged = 0;
    try {
        double delta = std::numeric_limits<double>::infinity();
        for (int used = 1; used <= max_iterations; ++used) {
            delta = sor_sweep(*solver, source, omega);
            *iterations_used = used;
            if (delta <= tolerance) {
                *converged = 1;
                break;
            }
        }
        *final_delta = delta;
        copy_solution(*solver, psi_out);
        return SCPN_SOLVER_STATUS_OK;
    } catch (...) {
        return SCPN_SOLVER_STATUS_INTERNAL_ERROR;
    }
}

extern "C" scpn_solver_status_v1 scpn_solver_destroy_v1(scpn_solver_v1* solver) {
    try {
        delete solver;
        return SCPN_SOLVER_STATUS_OK;
    } catch (...) {
        return SCPN_SOLVER_STATUS_INTERNAL_ERROR;
    }
}

extern "C" void* create_solver(
    int nr,
    int nz,
    double r_min,
    double r_max,
    double z_min,
    double z_max
) {
    scpn_solver_v1* solver = nullptr;
    const scpn_solver_status_v1 status =
        scpn_solver_create_v1(nr, nz, r_min, r_max, z_min, z_max, &solver);
    return status == SCPN_SOLVER_STATUS_OK ? solver : nullptr;
}

extern "C" void set_boundary_dirichlet(void* solver_ptr, double boundary_value) {
    static_cast<void>(
        scpn_solver_set_boundary_dirichlet_v1(static_cast<scpn_solver_v1*>(solver_ptr), boundary_value)
    );
}

extern "C" void run_step(void* solver_ptr, double* source, double* psi_out, int size, int iterations) {
    static_cast<void>(scpn_solver_run_steps_v1(
        static_cast<scpn_solver_v1*>(solver_ptr), source, psi_out, size, std::max(0, iterations)
    ));
}

extern "C" int run_step_converged(
    void* solver_ptr,
    double* source,
    double* psi_out,
    int size,
    int max_iterations,
    double omega,
    double tolerance,
    double* final_delta
) {
    if (max_iterations < 1) {
        if (final_delta != nullptr) {
            *final_delta = std::numeric_limits<double>::quiet_NaN();
        }
        return 0;
    }
    int iterations_used = 0;
    int converged = 0;
    double local_delta = std::numeric_limits<double>::quiet_NaN();
    const scpn_solver_status_v1 status = scpn_solver_run_until_converged_v1(
        static_cast<scpn_solver_v1*>(solver_ptr),
        source,
        psi_out,
        size,
        max_iterations,
        omega,
        tolerance,
        &iterations_used,
        &local_delta,
        &converged
    );
    static_cast<void>(converged);
    if (final_delta != nullptr) {
        *final_delta = local_delta;
    }
    return status == SCPN_SOLVER_STATUS_OK ? iterations_used : 0;
}

extern "C" void destroy_solver(void* solver_ptr) {
    static_cast<void>(scpn_solver_destroy_v1(static_cast<scpn_solver_v1*>(solver_ptr)));
}
